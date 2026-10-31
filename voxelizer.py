"""Fast exact-coverage voxelizer for closed triangle meshes.

Every layer is a horizontal plane cut. Instead of building polygons and
drawing them into an oversampled bitmap (one Python round trip per layer),
the whole volume is processed with vectorised numpy:

1. triangle x plane  -> section segments, oriented by the face normal;
2. segment x scanline -> signed crossings; several sub-scanlines per pixel row;
3. non-zero winding rule per scanline -> inside intervals. Overlapping shells
   form a union, internal cavities stay empty, inverted shells still work;
4. every interval adds its exact length to the pixels it covers, so the
   coverage is exact along X and sampled along Y (and optionally Z).

A scanline whose winding does not return to zero shows that the mesh is open
at that height. Those layers are reported back so the caller can use a
repair path for them.

Output convention matches slicing_engine: volume[x, y, z] in [0, 1]. Pixel
(i, j) covers [-span/2 + i*h, -span/2 + (i+1)*h] x [... j ...], with h = span/N.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class VoxelizeResult:
    volume: np.ndarray            # [x, y, z] coverage: float32 0..1 or uint8 0..255
    open_layers: np.ndarray       # indices of layers with unbalanced winding


def _section_segments(V: np.ndarray, F: np.ndarray, normals: np.ndarray, zs: np.ndarray):
    """Oriented segments of all triangles cut by the planes zs (sorted)."""
    tri = V[F]                                            # (m, 3, 3)
    tz = tri[:, :, 2]
    zmin, zmax = tz.min(axis=1), tz.max(axis=1)
    # Vertex "above" a plane means v_z > z: a triangle is cut when zmin <= z < zmax.
    k0 = np.searchsorted(zs, zmin, side="left")
    k1 = np.searchsorted(zs, zmax, side="left")
    count = k1 - k0
    sel = np.flatnonzero(count > 0)
    if not sel.size:
        return (np.empty(0, np.int64),) + (np.empty(0),) * 4
    count = count[sel]
    t_idx = np.repeat(sel, count)
    plane = np.repeat(k0[sel], count) + (np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count))
    z = zs[plane]
    P = tri[t_idx]                                        # (p, 3, 3)
    above = P[:, :, 2] > z[:, None]
    pts = []
    crosses = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        za, zb = P[:, a, 2], P[:, b, 2]
        c = above[:, a] != above[:, b]
        t = np.divide(z - za, zb - za, out=np.zeros_like(z), where=c)
        pts.append(P[:, a, :2] + t[:, None] * (P[:, b, :2] - P[:, a, :2]))
        crosses.append(c)
    C = np.stack(crosses, axis=1)                         # exactly two True per row
    first = np.argmax(C, axis=1)
    second = 2 - np.argmax(C[:, ::-1], axis=1)
    pts_arr = np.stack(pts, axis=1)                       # (p, 3, 2)
    rows = np.arange(len(plane))
    p1, p2 = pts_arr[rows, first], pts_arr[rows, second]
    # Orient along n x z_hat = (n_y, -n_x): +dy when entering a solid from -x.
    n = normals[t_idx]
    d = p2 - p1
    flip = (d[:, 0] * n[:, 1] - d[:, 1] * n[:, 0]) < 0
    p1, p2 = np.where(flip[:, None], p2, p1), np.where(flip[:, None], p1, p2)
    return plane, p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1]


def voxelize(mesh, z_levels: np.ndarray, grid_res: int, span_mm: float,
             sub_y: int = 4, sub_z: int = 1, z_pitch: Optional[float] = None,
             planes_per_batch: int = 48,
             cancelled: Optional[Callable[[], bool]] = None,
             progress: Optional[Callable[[float], None]] = None,
             out_dtype=np.float32) -> VoxelizeResult:
    """Coverage of `mesh` on an N x N grid centred at the origin, one layer per z level.

    sub_z > 1 averages sub_z planes evenly spaced inside each layer of
    thickness z_pitch: the slab average that one projector row exposes.
    """
    N = int(grid_res)
    h = span_mm / N
    z_levels = np.asarray(z_levels, dtype=np.float64)
    nz = len(z_levels)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    keep = np.linalg.norm(normals, axis=1) > 0
    F, normals = F[keep], normals[keep]
    if sub_z > 1:
        if not z_pitch:
            z_pitch = float(np.median(np.diff(z_levels))) if nz > 1 else h
        offsets = ((np.arange(sub_z) + 0.5) / sub_z - 0.5) * z_pitch
    else:
        offsets = np.zeros(1)
    planes = (z_levels[:, None] + offsets[None, :]).ravel()          # layer-major
    n_lines = N * sub_y
    y0 = -span_mm / 2 + h * 0.5 / sub_y
    dy = h / sub_y
    # float32 coverage in [0, 1], or uint8 coverage 0..255 (a quarter of the memory).
    volume = np.zeros((N, N, nz), out_dtype)
    open_layer = np.zeros(nz, bool)
    tri_z = V[F][:, :, 2]
    tz_min, tz_max = tri_z.min(axis=1), tri_z.max(axis=1)
    layers_per_batch = max(1, planes_per_batch // len(offsets))

    def layer_batch(L0: int) -> None:
        if cancelled is not None and cancelled():
            from slicing_engine import SliceCancelled
            raise SliceCancelled()
        L1 = min(nz, L0 + layers_per_batch)
        zs = planes[L0 * len(offsets):L1 * len(offsets)]
        near = (tz_max > zs[0]) & (tz_min <= zs[-1])
        plane, x1, y1, x2, y2 = _section_segments(V, F[near], normals[near], zs)
        B = len(zs)
        cover = np.zeros(B * N * (N + 2))
        if plane.size:
            ylo, yhi = np.minimum(y1, y2), np.maximum(y1, y2)
            # Sub-scanline l at y0 + l*dy is crossed when ylo <= y_l < yhi.
            l0 = np.clip(np.ceil((ylo - y0) / dy - 1e-12), 0, n_lines).astype(np.int64)
            l1 = np.clip(np.ceil((yhi - y0) / dy - 1e-12), 0, n_lines).astype(np.int64)
            cnt = l1 - l0
            s = np.flatnonzero(cnt > 0)
            cnt = cnt[s]
            seg = np.repeat(s, cnt)
            line = np.repeat(l0[s], cnt) + (np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt))
            yl = y0 + line * dy
            t = (yl - y1[seg]) / (y2[seg] - y1[seg])
            x = x1[seg] + t * (x2[seg] - x1[seg])
            sign = np.where(y2[seg] > y1[seg], 1, -1).astype(np.int64)
            key = plane[seg] * n_lines + line
            order = np.lexsort((x, key))
            key, x, sign = key[order], x[order], sign[order]
            # Winding after each crossing, restarted for every scanline.
            w = np.cumsum(sign)
            starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
            base = np.repeat(np.r_[0, w[starts[1:] - 1]], np.diff(np.r_[starts, len(key)]))
            w = w - base
            w_before = w - sign
            ends = np.r_[starts[1:], len(key)] - 1
            bad_keys = key[ends][w[ends] != 0]
            if bad_keys.size:
                bad_planes = np.unique(bad_keys // n_lines)
                open_layer[L0 + bad_planes // len(offsets)] = True
            enter = (w_before == 0) & (w != 0)
            leave = (w_before != 0) & (w == 0)
            ev = enter | leave
            u = np.clip((x[ev] + span_mm / 2) / h, 0, N)
            sg = np.where(enter[ev], 1.0, -1.0)
            kk = np.minimum(np.floor(u).astype(np.int64), N)
            pl = key[ev] // n_lines
            row = (key[ev] % n_lines) // sub_y
            idx = (pl * N + row) * (N + 2) + kk
            frac = u - kk
            cover += np.bincount(idx, weights=sg * (1 - frac), minlength=cover.size)
            cover += np.bincount(idx + 1, weights=sg * frac, minlength=cover.size)
        cov = np.cumsum(cover.reshape(B, N, N + 2), axis=2)[:, :, :N] / sub_y
        cov = cov.reshape(L1 - L0, len(offsets), N, N).mean(axis=1)            # [layer, y, x]
        cov = np.clip(cov, 0, 1).transpose(2, 1, 0)
        volume[:, :, L0:L1] = np.rint(cov * 255) if volume.dtype == np.uint8 else cov

    # Layer batches are independent and numpy sorts/scans release the GIL: run them
    # on all cores, report progress in order.
    starts = list(range(0, nz, layers_per_batch))
    from concurrent.futures import ThreadPoolExecutor

    from optics.fastcal import engine_threads
    threads = max(1, min(engine_threads(), len(starts)))
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for L0, _ in zip(starts, pool.map(layer_batch, starts)):
            if progress is not None:
                progress(min(nz, L0 + layers_per_batch) / nz)
    return VoxelizeResult(volume, np.flatnonzero(open_layer))
