"""Sparse relative-dose planning with Snell rays and nonnegative SIRT.

A maps a voxel field [x, y] to ray integrals in mm; A.T is its exact
discrete adjoint. Entries are exact pixel-crossing lengths (Siddon, 1985),
with analytically integrated Beer-Lambert attenuation. Each angle is built
independently; dense volumes are processed in Z batches. No material gelation
curve is assumed: the target is a relative optical dose, not cured geometry.
"""
from __future__ import annotations

import ctypes
import os
import sys
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.fft import fft, ifft

from constants import MAX_ESTIMATED_MEMORY_BYTES

from .ray_tracing import intersect_circle, trace_cuvette, trace_immersion_vial

_matrix_cache: dict[tuple[Any, ...], tuple[sp.csr_matrix, sp.csr_matrix]] = {}
_sirt_lib = None
_lib_loaded = False
_dll_directory_handle = None
# Bounds transient dense working buffers, independently of total layer count.
DENSE_WORKSPACE_BYTES = 32 * 1024**2


def _check_cancelled(callback):
    if callback and callback():
        from slicing_engine import SliceCancelled
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
    if mode not in (0, 1, 2):
        raise ValueError("container_mode must be 0 (air vial), 1 (square), or 2 (immersion vial).")
    return angles


def _box_interval(origins, directions, half):
    nonparallel = np.abs(directions) > 1e-14
    first = np.divide(-half - origins, directions, out=np.full_like(origins, -np.inf), where=nonparallel)
    second = np.divide(half - origins, directions, out=np.full_like(origins, np.inf), where=nonparallel)
    entry = np.minimum(first, second).max(axis=1)
    exit_ = np.maximum(first, second).min(axis=1)
    valid = np.all(nonparallel | (np.abs(origins) <= half + 1e-10), axis=1)
    return entry, exit_, valid


def _angle_matrix(origins, directions, valid, num_s, grid_res, span, inner, alpha, mode):
    """Split complete resin chords at every grid crossing, without sampling."""
    if mode == 1:
        entry, exit_, intersects = _box_interval(origins, directions, inner / 2)
    else:
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
    # Detector pixel centres, not bin edges.
    positions = (np.arange(num_s) + 0.5) * L_out / num_s - L_out / 2
    batches = []
    for index, theta in enumerate(angles):
        _check_cancelled(cancelled_cb)
        if container_mode == 1:
            p, v, valid = trace_cuvette(positions, theta, L_out, L_in, n_air, n_glass, n_resin)
        else:
            surrounding_index = n_liquid if container_mode == 2 else n_air
            p, v, valid = trace_immersion_vial(
                positions, theta, L_out, L_in, n_air, n_glass, surrounding_index, n_resin,
            )
        batches.append(_angle_matrix(p, v, valid, num_s, grid_res, grid_span_mm, L_in, alpha, container_mode))
        if report_cb and (index % max(1, len(angles) // 10) == 0 or index == len(angles) - 1):
            report_cb(0.30 + 0.04 * (index + 1) / len(angles),
                      f"Построение матрицы лучей {index + 1}/{len(angles)}...")
    matrix = sp.vstack(batches, format="csr")
    del batches
    transpose = matrix.T.tocsr()
    result = matrix, transpose
    _matrix_cache[key] = result
    return result


def filter_sinograms_3d(b_3d: np.ndarray, filter_name: str = "ram-lak") -> np.ndarray:
    """Ram-Lak warm start only; curved-ray inversion is done by SIRT."""
    if filter_name != "ram-lak":
        raise ValueError("Only the ram-lak filter is supported.")
    detector_count = b_3d.shape[0]
    padded = max(64, int(2 ** np.ceil(np.log2(2 * detector_count))))
    n = np.concatenate((np.arange(1, padded // 2 + 1, 2), np.arange(padded // 2 - 1, 0, -2)))
    kernel = np.zeros(padded)
    kernel[0] = 0.25
    kernel[1::2] = -1.0 / (np.pi * n)**2
    ramp = (2 * np.real(fft(kernel))).astype(np.float32)[:, None, None]
    image = np.pad(b_3d, ((0, padded - detector_count), (0, 0), (0, 0)))
    return np.real(ifft(fft(image, axis=0) * ramp, axis=0)[:detector_count]).copy()


def _load_cpp_backend():
    global _sirt_lib, _lib_loaded, _dll_directory_handle
    if _lib_loaded:
        return _sirt_lib
    _lib_loaded = True
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bundle = getattr(sys, "_MEIPASS", None)
    directories = [os.path.join(base, "cpp_backend"), base]
    if bundle:
        directories[:0] = [os.path.join(str(bundle), "cpp_backend"), str(bundle)]
    float_pointer = ctypes.POINTER(ctypes.c_float)
    integer_pointer = ctypes.POINTER(ctypes.c_int32)
    for directory in directories:
        for name in ("sirtsolver.dll", "sirtsolver.so", "libsirt.so", "sirtsolver.dylib"):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            try:
                if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
                    _dll_directory_handle = os.add_dll_directory(directory)
                library = ctypes.CDLL(path)
                # New symbol prevents silently using the old nonlinear solver.
                library.sirt_linear_step.argtypes = [
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, float_pointer, float_pointer,
                    integer_pointer, integer_pointer, float_pointer,
                    integer_pointer, integer_pointer, float_pointer,
                    float_pointer, float_pointer, ctypes.POINTER(ctypes.c_bool), float_pointer,
                ]
                library.sirt_linear_step.restype = None
                library.forward_project.argtypes = [
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, float_pointer,
                    integer_pointer, integer_pointer, float_pointer, float_pointer,
                ]
                library.forward_project.restype = None
                _sirt_lib = library
                return library
            except (OSError, AttributeError):
                continue
    return None


def _csr_pointers(matrix):
    return (
        matrix.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        matrix.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        matrix.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )


def _pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _project(matrix, values, library):
    if library is None:
        return matrix @ values
    result = np.empty((matrix.shape[0], values.shape[1]), dtype=np.float32)
    library.forward_project(matrix.shape[0], matrix.shape[1], values.shape[1],
                            _pointer(values), *_csr_pointers(matrix), _pointer(result))
    return result


def _python_sirt_step(a, at, values, target, ray_norm, voxel_norm, hits, workspace):
    np.subtract(target, at @ values, out=workspace)
    workspace *= voxel_norm[:, None]
    update = a @ workspace
    update *= ray_norm[:, None]
    values += update
    np.maximum(values, 0, out=values)
    values[~hits] = 0


def optimize_projections_sirt_volume(
    target_vol: np.ndarray, angles_rad: np.ndarray, L_out: float, L_in: float,
    n_air: float = 1.0, n_glass: float = 1.51, n_liquid: float = 1.45,
    n_resin: float = 1.50, alpha: float = 0.0, grid_res: int = 128,
    grid_span_mm: float = 50.0, iterations: int = 8, container_mode: int = 0,
    report_cb: Optional[Callable[[float, str], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    batch_layers: Optional[int] = None, num_s: Optional[int] = None,
) -> np.ndarray:
    """Fit nonnegative relative-dose projections; output is [z, detector, angle].

    With B=A.T, each iteration is x <- max(0, x + C B.T R (target-B x)),
    where R/C are reciprocal row/column sums of B. No per-iteration max
    normalization or uncalibrated sigmoid changes the objective.
    """
    target = np.asarray(target_vol)
    if target.ndim != 3 or target.shape[:2] != (grid_res, grid_res) or not target.shape[2]:
        raise ValueError("Target must have shape (grid_res, grid_res, nonzero layers).")
    if not np.isfinite(target).all() or np.any(target < 0) or np.any(target > 1):
        raise ValueError("Relative target dose must be finite and in [0, 1].")
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    detector_count = grid_res if num_s is None else num_s
    a, at = build_system_matrix(
        angles_rad, detector_count, grid_res, grid_span_mm, L_out, L_in,
        n_air, n_glass, n_liquid, n_resin, alpha, container_mode, report_cb, cancelled_cb,
    )
    library = _load_cpp_backend()
    if a.indices.dtype != np.int32 or a.indptr.dtype != np.int32:
        library = None
    if report_cb:
        report_cb(0.35, "SIRT: C++ OpenMP" if library is not None else "SIRT: SciPy (Python)")
    row_sums = np.asarray(a.sum(axis=1)).ravel()
    col_sums = np.asarray(at.sum(axis=1)).ravel()
    ray_norm = np.divide(1, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
    voxel_norm = np.divide(1, col_sums, out=np.zeros_like(col_sums), where=col_sums > 0)
    nz = target.shape[2]
    # FFT warm start and both sparse products fit inside this working budget.
    bytes_per_layer = 4 * (10 * a.shape[0] + 4 * a.shape[1])
    block_size = max(1, DENSE_WORKSPACE_BYTES // bytes_per_layer)
    if batch_layers is not None:
        if isinstance(batch_layers, bool) or int(batch_layers) != batch_layers or batch_layers < 1:
            raise ValueError("batch_layers must be a positive integer.")
        block_size = min(block_size, int(batch_layers))
    sinograms = np.empty((nz, detector_count, len(angles_rad)), dtype=np.float32)
    for start in range(0, nz, block_size):
        _check_cancelled(cancelled_cb)
        stop = min(nz, start + block_size)
        target_block = np.ascontiguousarray(target[:, :, start:stop].reshape(grid_res**2, stop-start), dtype=np.float32)
        projected = _project(a, target_block, library)
        hits = projected > 0
        filtered = filter_sinograms_3d(projected.reshape(len(angles_rad), detector_count, -1).transpose(1, 0, 2))
        values = np.ascontiguousarray(filtered.transpose(1, 0, 2).reshape(a.shape[0], -1), dtype=np.float32)
        del projected, filtered
        np.maximum(values, 0, out=values)
        values[~hits] = 0
        workspace = _project(at, values, library)
        # Least-squares scale of the initializer, independently for each layer.
        numerator = np.einsum("ij,ij,i->j", workspace, target_block, voxel_norm, dtype=np.float64)
        denominator = np.einsum("ij,ij,i->j", workspace, workspace, voxel_norm, dtype=np.float64)
        scale = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        values *= scale.astype(np.float32)[None, :]
        for iteration in range(iterations):
            _check_cancelled(cancelled_cb)
            if library is None:
                _python_sirt_step(a, at, values, target_block, ray_norm, voxel_norm, hits, workspace)
            else:
                library.sirt_linear_step(
                    a.shape[0], a.shape[1], stop-start, _pointer(values), _pointer(target_block),
                    *_csr_pointers(a), *_csr_pointers(at), _pointer(ray_norm), _pointer(voxel_norm),
                    hits.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)), _pointer(workspace),
                )
            if report_cb:
                progress = (start + (stop-start) * (iteration+1) / iterations) / nz
                report_cb(0.36 + 0.42 * progress, f"SIRT {iteration+1}/{iterations}; Z {start+1}–{stop}/{nz}")
        sinograms[start:stop] = values.reshape(len(angles_rad), detector_count, -1).transpose(2, 1, 0)
    maximum = float(np.max(sinograms))
    if maximum > 0:
        sinograms /= maximum
    return sinograms


def optimize_projections_sirt(
    V_target: np.ndarray, angles_rad: np.ndarray, L_out: float, L_in: float,
    n_air: float = 1.0, n_glass: float = 1.51, n_liquid: float = 1.45,
    n_resin: float = 1.50, alpha: float = 0.0, num_s: int = 128,
    grid_span_mm: float = 50.0, iterations: int = 8, container_mode: int = 0,
) -> np.ndarray:
    grid_res = V_target.shape[0]
    result = optimize_projections_sirt_volume(
        V_target[:, :, None], angles_rad, L_out, L_in, n_air, n_glass, n_liquid,
        n_resin, alpha, grid_res, grid_span_mm, iterations, container_mode, num_s=num_s,
    )
    return result[0].T
