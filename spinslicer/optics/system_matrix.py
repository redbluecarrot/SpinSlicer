"""Exact sparse ray-integral matrix: the independent reference for FastCAL.

A maps a voxel field [x, y] to ray integrals in mm; A.T is its exact
discrete adjoint (the dose operator). Entries are exact pixel-crossing lengths
(Siddon, 1985), with analytically integrated Beer-Lambert attenuation. Each
angle is traced independently with Snell refraction at the container walls.

The app plans with optics.fastcal; tests and tools/ check FastCAL's
rotation-symmetric operator against this matrix, and SliceParams validation
reuses its geometry checks.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sp

from spinslicer.constants import MAX_ESTIMATED_MEMORY_BYTES

from .geometry import detector_span_mm
from .ray_tracing import intersect_circle, trace_immersion_vial

_matrix_cache: dict[tuple[Any, ...], tuple[sp.csr_matrix, sp.csr_matrix]] = {}


def _check_cancelled(callback):
    if callback and callback():
        from spinslicer.slicing_engine import SliceCancelled
        raise SliceCancelled()


def _validate_geometry(angles, num_s, grid_res, span, outer, inner, indices, alpha, mode):
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim != 1 or not angles.size or not np.isfinite(angles).all():
        raise ValueError("Angles must be a non-empty finite 1-D array.")
    if isinstance(num_s, bool) or int(num_s) != num_s or num_s < 1:
        raise ValueError("num_s must be a positive integer.")
    if isinstance(grid_res, bool) or int(grid_res) != grid_res or grid_res < 2:
        raise ValueError("grid_res must be an integer >= 2.")
    if not np.isfinite([span, outer, inner, alpha, *indices]).all():
        raise ValueError("Optical parameters must be finite.")
    if not 0 < inner <= outer or span < inner:
        raise ValueError("Require 0 < L_in <= L_out and grid_span_mm >= L_in.")
    if alpha < 0 or any(index <= 0 for index in indices):
        raise ValueError("Absorption must be nonnegative and refractive indices positive.")
    if mode not in (0, 2):
        raise ValueError("container_mode must be 0 (vial in air) or 2 (vial in an immersion bath).")
    return angles


def _angle_matrix(origins, directions, valid, num_s, grid_res, span, inner, alpha):
    """Split complete resin chords at every grid crossing, without sampling."""
    entry, exit_, intersects = intersect_circle(origins, directions, inner / 2)
    entry = np.maximum(entry, 0)
    selected = np.flatnonzero(valid & intersects & (exit_ > entry))
    shape = (num_s, grid_res**2)
    if not selected.size:
        return sp.csr_matrix(shape, dtype=np.float32)
    p, v = origins[selected], directions[selected]
    entry, exit_ = entry[selected], exit_[selected]
    edges = np.linspace(-span / 2, span / 2, grid_res + 1)
    crossings = []
    for axis in (0, 1):
        velocity = v[:, axis, None]
        crossings.append(np.divide(
            edges[None, :] - p[:, axis, None], velocity,
            out=np.full((len(selected), len(edges)), np.inf),
            where=np.abs(velocity) > 1e-14,
        ))
    distances = np.concatenate([entry[:, None], *crossings, exit_[:, None]], axis=1)
    np.clip(distances, entry[:, None], exit_[:, None], out=distances)
    distances.sort(axis=1)
    lengths = np.diff(distances, axis=1)
    keep = lengths > span * np.finfo(float).eps * 32
    rows = np.broadcast_to(selected[:, None], lengths.shape)[keep]
    starts = distances[:, :-1][keep]
    lengths = lengths[keep]
    midpoints = starts + lengths / 2
    # Volume storage is [x, y, z], so flat pixel index is x*N + y.
    local_rays = np.broadcast_to(np.arange(len(selected))[:, None], keep.shape)[keep]
    points = p[local_rays] + v[local_rays] * midpoints[:, None]
    cells = np.floor((points + span / 2) * grid_res / span).astype(np.int32)
    np.clip(cells, 0, grid_res - 1, out=cells)
    weights = lengths
    if alpha > 0:
        depth = np.maximum(starts - entry[local_rays], 0)
        weights = np.exp(-alpha * depth) * (-np.expm1(-alpha * lengths)) / alpha
    chunk = sp.csr_matrix(
        (weights.astype(np.float32), (rows, cells[:, 0] * grid_res + cells[:, 1])),
        shape=shape, dtype=np.float32,
    )
    chunk.sum_duplicates()
    chunk.eliminate_zeros()
    return chunk


def build_system_matrix(
    angles_rad: np.ndarray, num_s: int, grid_res: int, grid_span_mm: float,
    L_out: float, L_in: float, n_air: float = 1.0, n_glass: float = 1.51,
    n_liquid: float = 1.45, n_resin: float = 1.50, alpha: float = 0.0,
    container_mode: int = 0, report_cb: Optional[Callable[[float, str], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    memory_budget_bytes: int = MAX_ESTIMATED_MEMORY_BYTES,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return ray integrals A and exact adjoint A.T (both float32 CSR)."""
    angles = _validate_geometry(
        angles_rad, num_s, grid_res, grid_span_mm, L_out, L_in,
        (n_air, n_glass, n_liquid, n_resin), alpha, container_mode,
    )
    _check_cancelled(cancelled_cb)
    key = (angles.tobytes(), int(num_s), int(grid_res), float(grid_span_mm),
           float(L_out), float(L_in), float(n_air), float(n_glass),
           float(n_liquid), float(n_resin), float(alpha), container_mode)
    if key in _matrix_cache:
        return _matrix_cache[key]
    # Two grids of crossings per ray; 3 CSR payloads covers assembly/transpose.
    upper_bytes = len(angles) * num_s * (2 * grid_res + 1) * 24
    if upper_bytes > memory_budget_bytes:
        raise ValueError("Optical matrix exceeds the memory budget; reduce grid or angle count.")
    _matrix_cache.clear()
    # Detector centres and physical pitch must agree with frame export.
    detector_span = detector_span_mm(L_out, container_mode)
    positions = (np.arange(num_s) + 0.5) * detector_span / num_s - detector_span / 2
    batches = []
    for index, theta in enumerate(angles):
        _check_cancelled(cancelled_cb)
        surrounding_index = n_liquid if container_mode == 2 else n_air
        p, v, valid = trace_immersion_vial(
            positions, theta, L_out, L_in, n_air, n_glass, surrounding_index, n_resin,
        )
        batches.append(_angle_matrix(p, v, valid, num_s, grid_res, grid_span_mm, L_in, alpha))
        if report_cb and (index % max(1, len(angles) // 10) == 0 or index == len(angles) - 1):
            report_cb(0.30 + 0.04 * (index + 1) / len(angles),
                      f"Построение матрицы лучей {index + 1}/{len(angles)}...")
    matrix = sp.vstack(batches, format="csr")
    del batches
    transpose = matrix.T.tocsr()
    result = matrix, transpose
    _matrix_cache[key] = result
    return result
