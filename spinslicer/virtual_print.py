"""Virtual print: what the saved frames would cure, without a printer.

The frames are taken exactly as the projector shows them (gain, cut-off, 8-bit PNG
values, gamma and finite contrast, see slicing_engine._build_print_plan). Their dose is
accumulated in every layer with the FastCAL optical model (refraction, attenuation,
Fresnel, pixel profile, rotation smear), and the resin cures where the dose reaches the
gel threshold of the print plan.

Results:
  * metrics: the exposure that reproduces the model best (as a factor on the dose the
    process window suggests), the exposure range that keeps the part's size within half
    a voxel, and the match at that exposure and at +-5 % (lamp drift, resin batch);
  * virtual_print.stl: surface of the cured resin in mm (any STL viewer opens it);
    virtual_print.ply: the same surface coloured by its deviation from the model (red =
    extra resin, blue = undersize) and virtual_print_missing.stl: parts of the model
    that did not cure at all. The slicer shows these in 3D. Built here, in the worker
    thread, so the UI only loads them;
  * virtual_print_slices.png: a few layers, green = cured as wanted, red = extra resin,
    blue = missing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

LEVEL = 128            # uint8 value of the gel threshold in the stored volume
SCALE = 128.0          # stored value = dose / threshold * SCALE (2x threshold saturates)
SWEEP = np.round(np.arange(0.80, 1.2501, 0.01), 2)   # exposure factors tried


@dataclass
class VirtualPrint:
    cured: np.ndarray          # [x, y, z] uint8, dose / threshold * 128
    origin_mm: tuple[float, float, float]
    spacing_mm: tuple[float, float, float]
    metrics: dict[str, Any]


def simulate(engine: Any, target: np.ndarray, displayed: Callable[[list[int]], np.ndarray],
             threshold: float, z_levels: np.ndarray,
             cancelled: Optional[Callable[[], bool]] = None) -> VirtualPrint:
    """Cured-dose volume and match metrics of a planned job.

    engine: the FastVialEngine that planned the frames; target: [x, y, z] coverage
    (uint8 0..255 or float 0..1); displayed(layers) -> relative projector light
    [layers, detector, angle]; threshold: gel dose in engine dose units.
    """
    from spinslicer.optics.fastcal import polar_to_cartesian_matrix

    N, NZ = engine.N, target.shape[2]
    H = polar_to_cartesian_matrix(engine.op.grid, N, engine.span)
    cured = np.zeros((N, N, NZ), np.uint8)
    gain = np.float32(SCALE / max(float(threshold), 1e-30))
    step = max(1, min(engine.batch, 16))      # small batches: the frames path makes copies

    def layer_batch(layers: list[int]) -> None:
        if cancelled is not None and cancelled():
            from spinslicer.slicing_engine import SliceCancelled
            raise SliceCancelled()
        light = np.asarray(displayed(layers), np.float32)                # [l, s, angle]
        D = engine.planner.dose(np.ascontiguousarray(light.transpose(2, 1, 0)))
        del light
        flat = np.asarray(H @ D.reshape(-1, len(layers)), np.float32)    # [x*N+y, l]
        cured[:, :, layers[0]:layers[-1] + 1] = np.clip(flat * gain, 0, 255).reshape(N, N, -1)

    # Batches are independent; the light conversion and the sparse polar → grid map
    # are single-threaded numpy, so a couple of batches at a time use the other cores.
    from spinslicer.optics.fastcal import engine_threads, run_batches
    batches = [list(range(z0, min(NZ, z0 + step))) for z0 in range(0, NZ, step)]
    run_batches(layer_batch, batches, min(4, engine_threads()))

    z = np.asarray(z_levels, np.float64)
    pitch = engine.span / N
    z_pitch = float(np.median(np.diff(z))) if len(z) > 1 else pitch
    metrics = match_metrics(cured, target, (pitch, pitch, z_pitch))
    origin = (-engine.span / 2 + pitch / 2, -engine.span / 2 + pitch / 2, float(z[0]))
    return VirtualPrint(cured, origin, (pitch, pitch, z_pitch), metrics)


def _histograms(cured: np.ndarray, solid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """256-bin histograms of the stored dose inside and outside the model."""
    try:
        import cv2
        img = np.ascontiguousarray(cured.reshape(cured.shape[0], -1))
        mask = np.ascontiguousarray(solid.reshape(img.shape)).view(np.uint8)
        h_in = cv2.calcHist([img], [0], mask, [256], [0, 256]).ravel().astype(np.float64)
        h_all = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel().astype(np.float64)
        return h_in, h_all - h_in
    except ImportError:                      # pragma: no cover - cv2 is a dependency
        h_in = np.bincount(cured[solid], minlength=256).astype(np.float64)
        return h_in, np.bincount(cured.ravel(), minlength=256).astype(np.float64) - h_in


def match_metrics(cured: np.ndarray, target: np.ndarray,
                  spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> dict[str, Any]:
    """How the cured part matches the model over a sweep of exposures.

    Exposure factor e cures voxels with dose * e >= threshold, i.e. stored >= 128 / e.
    Besides IoU, every exposure gets the mean surface offset: (cured volume - model
    volume) / model surface, in mm (+ = part grows). The usable exposure range is where
    that offset stays within half a voxel: the part keeps its size to sub-voxel accuracy.
    """
    solid = target >= (128 if target.dtype == np.uint8 else 0.5)
    h_in, h_out = _histograms(cured, solid)
    part = max(float(h_in.sum()), 1.0)
    in_at = np.cumsum(h_in[::-1])[::-1]       # cured voxels at stored value >= v
    out_at = np.cumsum(h_out[::-1])[::-1]
    hx, hy, hz = (float(v) for v in spacing_mm)
    faces = (np.count_nonzero(solid[1:] != solid[:-1]) * hy * hz
             + np.count_nonzero(solid[:, 1:] != solid[:, :-1]) * hx * hz
             + np.count_nonzero(solid[:, :, 1:] != solid[:, :, :-1]) * hx * hy
             + np.count_nonzero(solid[0]) * hy * hz + np.count_nonzero(solid[-1]) * hy * hz
             + np.count_nonzero(solid[:, :, 0]) * hx * hy + np.count_nonzero(solid[:, :, -1]) * hx * hy)
    # A voxel staircase has on average 1.5x the area of the smooth surface it samples.
    area = max(faces / 1.5, 1e-9)
    voxel = hx * hy * hz
    tol_mm = 0.5 * min(hx, hy)

    def at(e: float) -> dict[str, float]:
        v = min(255, int(np.ceil(LEVEL / e)))
        inter, extra = float(in_at[v]), float(out_at[v])
        return {"iou": inter / (part + extra), "extra_pct": 100.0 * extra / part,
                "missing_pct": 100.0 * (part - inter) / part,
                "offset_mm": (inter + extra - part) * voxel / area}

    sweep = [at(float(e)) for e in SWEEP]
    ious = np.array([r["iou"] for r in sweep])
    offsets = np.array([r["offset_mm"] for r in sweep])
    k = int(np.argmax(ious))
    best = float(SWEEP[k])
    ok = np.abs(offsets) <= tol_mm
    lo_i = hi_i = k
    while lo_i - 1 >= 0 and ok[lo_i - 1]:
        lo_i -= 1
    while hi_i + 1 < len(SWEEP) and ok[hi_i + 1]:
        hi_i += 1
    low, high = float(SWEEP[lo_i]), float(SWEEP[hi_i])
    tolerance = 100.0 * min(best - low, high - best) / best if ok[k] else 0.0
    metrics: dict[str, Any] = {
        "layers": int(target.shape[2]),
        "best_exposure": best, "exposure_low": low, "exposure_high": high,
        "tolerance_pct": tolerance, "size_tolerance_mm": tol_mm,
        "best": sweep[k], "minus5": at(best * 0.95), "plus5": at(best * 1.05),
    }
    metrics["iou"] = sweep[k]["iou"]
    metrics["worst_iou_5pct"] = min(metrics["minus5"]["iou"], metrics["plus5"]["iou"], metrics["iou"])
    if metrics["iou"] >= 0.97 and tolerance >= 4.0:
        metrics["verdict"] = "ok"
    elif metrics["iou"] >= 0.95:
        metrics["verdict"] = "tight"
    else:
        metrics["verdict"] = "bad"
    return metrics


def cure_level(result: VirtualPrint) -> int:
    """Stored value at which the resin gels at the recommended exposure."""
    return min(255, int(np.ceil(LEVEL / float(result.metrics.get("best_exposure", 1.0)))))


MESH_FILE = "virtual_print.stl"


def cured_mesh(result: VirtualPrint) -> Any:
    """Surface of the cured resin (trimesh, mm), a few hundred thousand triangles at most.

    Rows finer than the voxel pitch are averaged to near-cubic cells first: same
    lateral detail, a third of the triangles.
    """
    import trimesh
    from skimage.measure import marching_cubes

    hx, hy, hz = result.spacing_mm
    cured = result.cured
    zf = max(1, int(round(min(hx, hy) / hz)))
    nz = cured.shape[2] // zf * zf
    if zf > 1 and nz:
        vol = cured[:, :, :nz].reshape(cured.shape[0], cured.shape[1], -1, zf).mean(axis=3, dtype=np.float32)
        if nz < cured.shape[2]:
            tail = cured[:, :, nz:].mean(axis=2, keepdims=True, dtype=np.float32)
            vol = np.concatenate([vol, tail], axis=2)
    else:
        vol = cured.astype(np.float32)
    level = cure_level(result) - 0.5
    if vol.max() < level:
        return trimesh.Trimesh()
    step = 1 if vol.size <= 20_000_000 else 2
    import warnings
    with warnings.catch_warnings():          # skimage's own NumPy deprecation noise
        warnings.simplefilter("ignore", DeprecationWarning)
        verts, faces, _, _ = marching_cubes(np.pad(vol, 1), level=level, step_size=step,
                                            allow_degenerate=False)
    spacing = np.array([hx, hy, hz * zf])
    origin = np.array(result.origin_mm) + np.array([0.0, 0.0, hz * (zf - 1) / 2]) - spacing
    return trimesh.Trimesh(verts * spacing + origin, faces[:, ::-1], process=False)


MESH_COLORED = "virtual_print.ply"
MESH_MISSING = "virtual_print_missing.stl"
RESIN_RGB = (232, 217, 184)      # cured as wanted
EXTRA_RGB = (255, 64, 72)        # resin where the model has none
SHORT_RGB = (62, 128, 255)       # model that did not cure


def _crop(solid: np.ndarray, cured: np.ndarray, pad: int = 6) -> tuple[slice, slice, slice]:
    """Bounding box of both volumes plus ``pad`` empty voxels; may reach outside the array
    (the caller pads), so that distances at the very top/bottom layer still see empty space."""
    both = solid | cured
    box = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        idx = np.flatnonzero(both.any(axis=other))
        lo, hi = (int(idx[0]), int(idx[-1]) + 1) if idx.size else (0, 1)
        box.append(slice(lo - pad, hi + pad))
    return box[0], box[1], box[2]


def _cut(volume: np.ndarray, box: tuple[slice, slice, slice]) -> np.ndarray:
    """volume[box] where voxels outside the array count as empty."""
    out = np.zeros(tuple(b.stop - b.start for b in box), volume.dtype)
    src, dst = [], []
    for b, n in zip(box, volume.shape):
        lo, hi = max(b.start, 0), min(b.stop, n)
        src.append(slice(lo, hi))
        dst.append(slice(lo - b.start, hi - b.start))
    out[tuple(dst)] = volume[tuple(src)]
    return out


def defect_meshes(result: VirtualPrint, target: np.ndarray) -> tuple[Any, Any]:
    """(cured surface coloured by deviation from the model, mesh of the uncured model parts).

    Deviation of a surface point = signed distance to the model surface in mm (+ outside).
    It is neutral within half a voxel, and ramps to red (extra resin) or blue (undersize)
    over the next voxel and a half. Model voxels farther than half a voxel from any cured
    voxel are "missing": thin features and fine detail the light did not build.
    """
    import trimesh
    from scipy import ndimage
    from skimage.measure import marching_cubes

    hx, hy, hz = result.spacing_mm
    cured_mask = result.cured >= cure_level(result)
    solid = target >= (128 if target.dtype == np.uint8 else 0.5)
    box = _crop(solid, cured_mask)
    sol, cur = _cut(solid, box), _cut(cured_mask, box)
    spacing = np.array([hx, hy, hz])
    tol = 0.5 * min(hx, hy)
    mesh = cured_mesh(result)
    colored = mesh
    if len(mesh.faces):
        sdf = (ndimage.distance_transform_edt(~sol, sampling=spacing)
               - ndimage.distance_transform_edt(sol, sampling=spacing)).astype(np.float32)
        start = np.array([b.start for b in box], np.float64)
        idx = (np.asarray(mesh.vertices) - np.asarray(result.origin_mm)) / spacing - start
        dev = ndimage.map_coordinates(sdf, idx.T, order=1, mode="nearest")
        t = np.clip((np.abs(dev) - tol) / (1.5 * min(hx, hy)), 0.0, 1.0)[:, None]
        base = np.array(RESIN_RGB, np.float64)
        end = np.where((dev > 0)[:, None], np.array(EXTRA_RGB, np.float64), np.array(SHORT_RGB, np.float64))
        rgb = np.round(base * (1 - t) + end * t).astype(np.uint8)
        colored = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False,
                                  vertex_colors=np.column_stack([rgb, np.full(len(rgb), 255, np.uint8)]))
    far = ndimage.distance_transform_edt(~cur, sampling=spacing) > tol
    gone = sol & far
    missing = trimesh.Trimesh()
    if gone.any():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            verts, faces, _, _ = marching_cubes(np.pad(gone.astype(np.float32), 1), level=0.5,
                                                allow_degenerate=False)
        origin = np.array(result.origin_mm) + np.array([b.start for b in box]) * spacing - spacing
        missing = trimesh.Trimesh(verts * spacing + origin, faces[:, ::-1], process=False)
    return colored, missing


def save(result: VirtualPrint, target: np.ndarray, out_dir: str) -> None:
    mesh = cured_mesh(result)
    if len(mesh.faces):
        mesh.export(os.path.join(out_dir, MESH_FILE))
        try:
            colored, missing = defect_meshes(result, target)
            colored.export(os.path.join(out_dir, MESH_COLORED))
            if len(missing.faces):
                missing.export(os.path.join(out_dir, MESH_MISSING))
        except Exception:         # the plain surface above is enough to look at
            pass
    save_slices_png(result, target, os.path.join(out_dir, "virtual_print_slices.png"))


def load_view(out_dir: str) -> Optional[dict[str, Any]]:
    """Everything the 3D view needs: coloured surface, missing parts and the match numbers."""
    import json

    import trimesh

    plain = mesh_path(out_dir)
    if plain is None:
        return None
    colored = os.path.join(out_dir, MESH_COLORED)
    mesh = trimesh.load(colored if os.path.isfile(colored) else plain, force="mesh", process=False)
    gone = os.path.join(out_dir, MESH_MISSING)
    missing = trimesh.load(gone, force="mesh") if os.path.isfile(gone) else None
    metrics: dict[str, Any] = {}
    try:
        with open(os.path.join(out_dir, "print_plan.json"), encoding="utf-8") as handle:
            metrics = (json.load(handle).get("virtual_print") or {}).get("best") or {}
    except (OSError, ValueError):
        pass
    return {"mesh": mesh, "missing": missing, "metrics": metrics,
            "colored": os.path.isfile(colored)}


def save_slices_png(result: VirtualPrint, target: np.ndarray, path: str, count: int = 4) -> None:
    from PIL import Image

    cured = result.cured >= cure_level(result)
    solid = target >= (128 if target.dtype == np.uint8 else 0.5)
    filled = np.flatnonzero(solid.any(axis=(0, 1)))
    if not filled.size:
        return
    picks = np.linspace(filled[0], filled[-1], count + 2)[1:-1].round().astype(int)
    xs, ys = np.nonzero(solid.any(axis=2) | cured.any(axis=2))
    x0, x1 = max(xs.min() - 4, 0), min(xs.max() + 5, solid.shape[0])
    y0, y1 = max(ys.min() - 4, 0), min(ys.max() + 5, solid.shape[1])
    tiles = []
    for z in picks:
        s, c = solid[x0:x1, y0:y1, z], cured[x0:x1, y0:y1, z]
        rgb = np.full(s.shape + (3,), 24, np.uint8)
        rgb[s & c] = (120, 200, 120)      # cured as wanted
        rgb[c & ~s] = (235, 80, 70)       # extra resin
        rgb[s & ~c] = (70, 130, 235)      # missing
        tiles.append(np.flipud(rgb.transpose(1, 0, 2)))
        tiles.append(np.full((s.shape[1], 3, 3), 60, np.uint8))
    img = np.concatenate(tiles[:-1], axis=1)
    scale = max(1, int(round(720 / img.shape[1])))
    Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.Resampling.NEAREST).save(path)


def mesh_path(out_dir: str) -> Optional[str]:
    path = os.path.join(out_dir, MESH_FILE)
    return path if os.path.isfile(path) else None
