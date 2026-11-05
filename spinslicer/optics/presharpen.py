"""Projector blur between frame rows: model it and pre-compensate it.

A real projector does not show a frame row as a sharp line: its optics (and defocus
across a deep vial) spread every row over its neighbours. Rows are layers in CAL, so the
blur mixes the frames of neighbouring layers and smears horizontal features (the top of a
shelf, a thin plate, a gear that starts at some height). Blur along a row matters far less:
the tomographic dose already integrates across it (tested 2026-10-30).

With the blur known (Gaussian sigma in mm), the frames are pre-sharpened along the rows:
the nonnegative stack X that the projector turns back into the planned frames,
min ||K X - P||^2 over X >= 0, solved with accelerated projected gradient. Where the
plan changes too abruptly for a nonnegative X, the residual is as small as light allows.
On the stress model with sigma = 0.7 row, IoU after blur went 0.960 -> 0.981. An
overestimated sigma costs accuracy, so the setting must come from a measurement.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d


def row_sigma(blur_mm: float, row_pitch_mm: float) -> float:
    """Blur sigma in rows (layers)."""
    return float(blur_mm) / float(row_pitch_mm) if blur_mm > 0 and row_pitch_mm > 0 else 0.0


def blur_rows(frames: np.ndarray, sigma_rows: float) -> np.ndarray:
    """What the projector shows for frames [row, detector, angle] (float32)."""
    x = np.asarray(frames, np.float32)
    if sigma_rows <= 0:
        return x
    return gaussian_filter1d(x, sigma_rows, axis=0, mode="nearest", truncate=3.0)


def _blur_last(x: np.ndarray, sigma: float) -> np.ndarray:
    """K: the projector's row blur along the last axis (clamped edges, as blur_rows)."""
    return gaussian_filter1d(x, sigma, axis=-1, mode="nearest", truncate=3.0)


def _blur_last_adjoint(y: np.ndarray, sigma: float) -> np.ndarray:
    """K^T. With clamped edges K is not symmetric: the light K takes from beyond an edge
    row comes from that row, so K^T returns it there. Exact adjoint of _blur_last."""
    radius = int(3.0 * float(sigma) + 0.5)          # scipy's kernel radius for truncate=3
    if radius == 0:
        return np.asarray(y, np.float32)
    n = y.shape[-1]
    pad = [(0, 0)] * (y.ndim - 1) + [(radius, radius)]
    full = gaussian_filter1d(np.pad(y, pad), sigma, axis=-1, mode="constant", truncate=3.0)
    out = np.ascontiguousarray(full[..., radius:radius + n])
    out[..., 0] += full[..., :radius].sum(axis=-1)
    out[..., -1] += full[..., radius + n:].sum(axis=-1)
    return out


def _lipschitz(n: int, sigma: float) -> float:
    """L = ||K^T K|| for n rows (power iteration, 1 % margin). Just above 1: the clamped
    edges add a little (1.004 at sigma 1.5 rows, 1.065 at 3), so step 1 could overshoot."""
    v = np.linspace(1.0, 2.0, n)
    lam = 1.0
    for _ in range(30):
        w = _blur_last_adjoint(_blur_last(v, sigma), sigma)
        lam = float(np.linalg.norm(w) / max(float(np.linalg.norm(v)), 1e-30))
        v = w / max(float(np.linalg.norm(w)), 1e-30)
    return max(1.0, 1.01 * lam)


def _presharpen_block(P: np.ndarray, sigma: float, iterations: int) -> np.ndarray:
    """P: [..., row] with rows on the last (contiguous) axis."""
    step = np.float32(1.0 / _lipschitz(P.shape[-1], sigma))

    X = P.copy()
    Y = X.copy()
    t = 1.0
    for _ in range(iterations):                 # FISTA on 1/2 ||K X - P||^2, X >= 0
        X_new = Y - step * _blur_last_adjoint(_blur_last(Y, sigma) - P, sigma)
        np.maximum(X_new, 0, out=X_new)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Y = X_new + np.float32((t - 1.0) / t_new) * (X_new - X)
        X, t = X_new, t_new
    return X


def presharpen_rows(frames: np.ndarray, sigma_rows: float, iterations: int = 8,
                    angles_per_block: int = 30, out: Optional[np.ndarray] = None,
                    cancelled: Optional[Callable[[], None]] = None) -> np.ndarray:
    """Frames [row, detector, angle] -> pre-sharpened frames (same scale).

    Processed in blocks of angles (rows are coupled, angles are not), two at a time.
    `out` may be `frames` itself (in place, e.g. the float16 frame store).
    """
    P = np.asarray(frames)
    if out is None:
        out = np.empty(P.shape, np.float32)
    if sigma_rows <= 0 or P.shape[0] < 2:
        if out is not P:
            out[...] = P
        return out
    blocks = [slice(a, min(a + angles_per_block, P.shape[2])) for a in range(0, P.shape[2], angles_per_block)]

    def work(block: slice) -> None:
        if cancelled is not None:
            cancelled()
        rows_last = np.ascontiguousarray(np.asarray(P[:, :, block], np.float32).transpose(2, 1, 0))
        out[:, :, block] = _presharpen_block(rows_last, sigma_rows, iterations).transpose(2, 1, 0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(work, blocks))
    return out


def blurred_rows_getter(light: Callable[[list[int]], np.ndarray], n_rows: int,
                        sigma_rows: float) -> Callable[[list[int]], np.ndarray]:
    """Wrap light(rows) -> [rows, detector, angle] so it returns what a projector with
    row blur sigma shows on those rows (Gaussian, truncated at 3 sigma, clamped edges)."""
    if sigma_rows <= 0:
        return light
    radius = max(1, int(np.ceil(3.0 * sigma_rows)))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma_rows) ** 2)
    weights = (weights / weights.sum()).astype(np.float32)

    def shown(rows: list[int]) -> np.ndarray:
        need = sorted({int(np.clip(r + d, 0, n_rows - 1)) for r in rows for d in offsets})
        index = {r: i for i, r in enumerate(need)}
        src = np.asarray(light(need), np.float32)
        out = np.zeros((len(rows),) + src.shape[1:], np.float32)
        for i, r in enumerate(rows):
            for d, w in zip(offsets, weights):
                out[i] += w * src[index[int(np.clip(r + d, 0, n_rows - 1))]]
        return out

    return shown
