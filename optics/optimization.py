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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.fft import fft, ifft

from compute import ComputeProfile, free_memory_bytes
from constants import MAX_ESTIMATED_MEMORY_BYTES
from i18n import tr

from .geometry import detector_span_mm
from .ray_tracing import intersect_circle, trace_cuvette, trace_immersion_vial

_matrix_cache: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_sirt_lib = None
_lib_loaded = False
_dll_directory_handle = None
# Floor for transient dense working buffers, independently of total layer count.
DENSE_WORKSPACE_BYTES = 32 * 1024**2
# Every SIRT product walks the whole sparse matrix once per Z block, so tiny
# blocks make large jobs crawl. The block budget therefore grows with free RAM.
MAX_WORKSPACE_BYTES = 1024**3
WORKSPACE_FREE_FRACTION = 0.25


def _workspace_bytes(profile: Optional[ComputeProfile] = None) -> int:
    cap = profile.workspace_cap if profile is not None else MAX_WORKSPACE_BYTES
    free = free_memory_bytes()
    if free is None:
        return min(cap, 4 * DENSE_WORKSPACE_BYTES)
    return int(min(cap, max(DENSE_WORKSPACE_BYTES, free * WORKSPACE_FREE_FRACTION)))


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


def _stack_csr(batches: list) -> sp.csr_matrix:
    """Stack CSR blocks row-wise, releasing each block once it is copied.

    ``sp.vstack`` keeps every block alive next to the result, which doubles the
    peak memory of building the ray matrix; here the peak is one matrix.
    """
    rows = sum(block.shape[0] for block in batches)
    columns = batches[0].shape[1]
    total = sum(block.nnz for block in batches)
    index_dtype = np.int32 if total < 2**31 - 1 else np.int64
    data = np.empty(total, dtype=np.float32)
    indices = np.empty(total, dtype=index_dtype)
    indptr = np.empty(rows + 1, dtype=index_dtype)
    indptr[0] = 0
    position = row = 0
    for number in range(len(batches)):
        block = batches[number]
        count = block.nnz
        data[position:position + count] = block.data
        indices[position:position + count] = block.indices
        indptr[row + 1:row + block.shape[0] + 1] = block.indptr[1:].astype(index_dtype) + position
        position += count
        row += block.shape[0]
        batches[number] = None  # free the block now
    return sp.csr_matrix((data, indices, indptr), shape=(rows, columns))


def build_system_matrix(
    angles_rad: np.ndarray, num_s: int, grid_res: int, grid_span_mm: float,
    L_out: float, L_in: float, n_air: float = 1.0, n_glass: float = 1.51,
    n_liquid: float = 1.45, n_resin: float = 1.50, alpha: float = 0.0,
    container_mode: int = 0, report_cb: Optional[Callable[[float, str], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    memory_budget_bytes: int = MAX_ESTIMATED_MEMORY_BYTES,
    lean_transpose: bool = False, workers: int = 1, use_cache: bool = True,
) -> tuple[Any, Any]:
    """Return ray integrals A and exact adjoint A.T (float32).

    ``lean_transpose`` returns the transpose as a CSC view of A instead of a second
    CSR copy: half the matrix memory, same values. ``workers`` builds the angles in
    parallel; every angle is independent, so the matrix is identical. The cache holds
    one matrix; ``use_cache=False`` builds one that neither reads nor evicts it.
    """
    angles = _validate_geometry(
        angles_rad, num_s, grid_res, grid_span_mm, L_out, L_in,
        (n_air, n_glass, n_liquid, n_resin), alpha, container_mode,
    )
    _check_cancelled(cancelled_cb)
    key = (angles.tobytes(), int(num_s), int(grid_res), float(grid_span_mm),
           float(L_out), float(L_in), float(n_air), float(n_glass),
           float(n_liquid), float(n_resin), float(alpha), container_mode, bool(lean_transpose))
    if use_cache and key in _matrix_cache:
        return _matrix_cache[key]
    # Two grids of crossings per ray; 3 CSR payloads covers assembly/transpose.
    upper_bytes = len(angles) * num_s * (2 * grid_res + 1) * 24
    if upper_bytes > memory_budget_bytes:
        raise ValueError("Optical matrix exceeds the memory budget; reduce grid or angle count.")
    if use_cache:
        _matrix_cache.clear()
    # A rotating square needs its circumscribed diagonal as a constant field.
    # Detector centres and physical pitch must agree with frame export.
    detector_span = detector_span_mm(L_out, container_mode)
    positions = (np.arange(num_s) + 0.5) * detector_span / num_s - detector_span / 2
    def build_angle(theta: float):
        _check_cancelled(cancelled_cb)
        if container_mode == 1:
            p, v, valid = trace_cuvette(positions, theta, L_out, L_in, n_air, n_glass, n_resin)
        else:
            surrounding_index = n_liquid if container_mode == 2 else n_air
            p, v, valid = trace_immersion_vial(
                positions, theta, L_out, L_in, n_air, n_glass, surrounding_index, n_resin,
            )
        return _angle_matrix(p, v, valid, num_s, grid_res, grid_span_mm, L_in, alpha, container_mode)

    batches: list = []
    pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
    try:
        for index, block in enumerate(pool.map(build_angle, angles)):  # results keep angle order
            batches.append(block)
            if report_cb and (index % max(1, len(angles) // 10) == 0 or index == len(angles) - 1):
                report_cb(0.30 + 0.04 * (index + 1) / len(angles),
                          f"Построение матрицы лучей {index + 1}/{len(angles)}...")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    matrix = _stack_csr(batches)
    del batches
    transpose = matrix.T if lean_transpose else matrix.T.tocsr()
    result = matrix, transpose
    if use_cache:
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


class CpuMatrix:
    """A SciPy matrix whose products use a fixed number of threads."""

    def __init__(self, matrix, workers: int) -> None:
        self.matrix = matrix
        self.workers = max(1, int(workers))
        self.shape = matrix.shape

    def matmul(self, values):
        return _threaded_matmul(self.matrix, values, self.workers)


def _mm(matrix, values):
    """``matrix @ values`` for SciPy matrices and the CPU/GPU wrappers alike."""
    if hasattr(matrix, "matmul"):
        return matrix.matmul(values)
    return _threaded_matmul(matrix, values)


def _gpu_problem(*operators) -> Optional[str]:
    """Why a GPU object gave the device up, if it did."""
    for operator in operators:
        error = getattr(operator, "error", None)
        if error:
            return str(error)
    return None


def _release(*operators) -> None:
    """Free GPU matrices in this thread (a CUDA context belongs to one thread).

    Taichi arrays sit in reference cycles, so they are only finalised by a garbage
    collection. Run it here: a collection triggered later inside some other thread
    (a worker pool, say) would destroy them there and abort the process.
    """
    released = False
    for operator in operators:
        if hasattr(operator, "release"):
            operator.release()
            released = True
    if released:
        import gc

        gc.collect()


def _operators(a, at, profile: Optional[ComputeProfile], library):
    """Choose how the sparse products run: returns (a_op, at_op, library, label)."""
    if profile is None:
        return a, at, library, "C++ OpenMP" if library is not None else "SciPy (Python)"
    if profile.device == "gpu":
        uploaded = None
        try:
            from .gpu_operator import GpuMatrix

            uploaded = GpuMatrix(a)
            second = GpuMatrix(at) if at is not None else None
            return uploaded, second, None, f"GPU {profile.gpu_name}"
        except Exception:  # no device memory, driver trouble: continue on the CPU
            # Free what was already uploaded here, in this thread; leaving it to the
            # garbage collector could destroy it in another thread and abort.
            _release(uploaded)
    if profile.lean_memory:
        library = None
    if library is not None:
        return a, at, library, "C++ OpenMP"
    return (
        CpuMatrix(a, profile.threads), CpuMatrix(at, profile.threads) if at is not None else None,
        None, f"CPU, {profile.threads} потоков",
    )


def _target_block(target: np.ndarray, start: int, stop: int, grid_res: int) -> np.ndarray:
    """Layers ``start:stop`` as float32 (columns, layers); uint8 targets mean value/255."""
    block = target[:, :, start:stop].reshape(grid_res**2, stop - start)
    dense = np.ascontiguousarray(block, dtype=np.float32)
    if target.dtype == np.uint8:
        dense = dense / np.float32(255.0)
    return dense


def _threaded_matmul(matrix, values, workers=None):
    """``matrix @ values`` with the dense columns split across threads.

    scipy releases the GIL inside sparse products, so this scales with cores
    when the C++ OpenMP kernel is not available.
    """
    workers = workers or min(os.cpu_count() or 1, 8)
    columns = values.shape[1]
    if workers < 2 or columns < 2 * workers:
        return matrix @ values
    from concurrent.futures import ThreadPoolExecutor

    edges = np.linspace(0, columns, workers + 1).astype(int)
    out = np.empty((matrix.shape[0], columns), dtype=np.result_type(matrix.dtype, values.dtype))

    def run(index):
        low, high = edges[index], edges[index + 1]
        out[:, low:high] = matrix @ np.ascontiguousarray(values[:, low:high])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, range(workers)))
    return out


def _project(matrix, values, library):
    if library is None:
        return _mm(matrix, values)
    result = np.empty((matrix.shape[0], values.shape[1]), dtype=np.float32)
    library.forward_project(matrix.shape[0], matrix.shape[1], values.shape[1],
                            _pointer(values), *_csr_pointers(matrix), _pointer(result))
    return result


def _python_sirt_step(a, at, values, target, ray_norm, voxel_norm, hits, workspace):
    np.subtract(target, _mm(at, values), out=workspace)
    workspace *= voxel_norm[:, None]
    update = _mm(a, workspace)
    update *= ray_norm[:, None]
    values += update
    np.maximum(values, 0, out=values)
    values[~hits] = 0


# Defaults of the window stage, in units where 1 is the target dose.
WINDOW_LEVEL = 0.8
WINDOW_STEEPNESS = 10.0


def _window_step(
    a, at, values, target, ray_norm, voxel_norm, hits, library,
    level: float = WINDOW_LEVEL, steepness: float = WINDOW_STEEPNESS,
):
    """One projected SIRT step on a sigmoid dose threshold (a heuristic, see below).

    Resin gels above a critical dose, so the least-squares fit, which leaves the
    dose of the part and of its surroundings close together, is followed by steps
    that push the part above ``level`` and the rest below it. The error is
    ``target - sigmoid(steepness * (dose - level))``. The derivative of the sigmoid
    is deliberately left out of the step (as in the native SIRT of v0.2), so this is
    not the gradient of a loss function: it is a bounded, monotone correction with no
    convergence guarantee, and ``level`` is not tied to any measured resin threshold.
    """
    dose = _project(at, values, library)
    arg = np.clip(-steepness * (dose - level), -60.0, 60.0)
    error = target - 1.0 / (1.0 + np.exp(arg))
    error *= voxel_norm[:, None]
    update = _project(a, np.ascontiguousarray(error, dtype=np.float32), library)
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
    window_iterations: int = 0, profile: Optional[ComputeProfile] = None,
    sink: Optional[Callable[[int, int, np.ndarray], None]] = None,
    window_level: float = WINDOW_LEVEL, window_steepness: float = WINDOW_STEEPNESS,
    status: Optional[dict] = None,
) -> np.ndarray:
    """Fit nonnegative relative-dose projections; output is [z, detector, angle].

    The linear stage is nonnegative SIRT on the relative dose and is the part that
    was validated for v0.3.1. ``window_iterations`` extra steps follow it and
    change the objective: they use a sigmoid dose threshold (``_window_step``, a
    heuristic without the sigmoid derivative, so not a gradient of any loss);
    0 keeps the pure least-squares result.

    With ``sink`` every finished Z block is handed over as ``sink(start, stop,
    values)`` (values: angles*detector rows by layers, not normalised) instead of
    being collected, and an empty array is returned.

    With B=A.T, each linear iteration is x <- max(0, x + C B.T R (target-B x)),
    where R/C are reciprocal row/column sums of B; there is no per-iteration max
    normalization. On a GPU profile the whole block (both stages) runs on the
    device; on a CPU profile the native C++ kernel covers only the linear stage.
    If the device fails the job continues on the CPU with the same arithmetic; that is
    reported through ``report_cb`` and, when ``status`` is given, recorded in it as
    ``device_used`` ("gpu" or "cpu") and ``gpu_error``.
    """
    target = np.asarray(target_vol)
    if target.ndim != 3 or target.shape[:2] != (grid_res, grid_res) or not target.shape[2]:
        raise ValueError("Target must have shape (grid_res, grid_res, nonzero layers).")
    if target.dtype != np.uint8 and (
        not np.isfinite(target).all() or np.any(target < 0) or np.any(target > 1)
    ):
        raise ValueError("Relative target dose must be finite and in [0, 1].")
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    detector_count = grid_res if num_s is None else num_s
    a, at = build_system_matrix(
        angles_rad, detector_count, grid_res, grid_span_mm, L_out, L_in,
        n_air, n_glass, n_liquid, n_resin, alpha, container_mode, report_cb, cancelled_cb,
        lean_transpose=bool(profile and profile.lean_memory),
        workers=profile.threads if profile else 1,
    )
    library = _load_cpp_backend()
    if a.indices.dtype != np.int32 or a.indptr.dtype != np.int32:
        library = None
    a_op, at_op, library, label = _operators(a, at, profile, library)
    solver = None
    warned = False
    wanted_gpu = bool(profile and profile.device == "gpu")
    try:
        row_sums = np.asarray(a.sum(axis=1)).ravel()
        col_sums = np.asarray(at.sum(axis=1)).ravel()
        ray_norm = np.divide(1, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
        voxel_norm = np.divide(1, col_sums, out=np.zeros_like(col_sums), where=col_sums > 0)
        nz = target.shape[2]
        # FFT warm start and both sparse products fit inside this working budget.
        bytes_per_layer = 4 * (10 * a.shape[0] + 4 * a.shape[1])
        block_size = max(1, _workspace_bytes(profile) // bytes_per_layer)
        if batch_layers is not None:
            if isinstance(batch_layers, bool) or int(batch_layers) != batch_layers or batch_layers < 1:
                raise ValueError("batch_layers must be a positive integer.")
            block_size = min(block_size, int(batch_layers))
        if isinstance(window_iterations, bool) or int(window_iterations) != window_iterations or window_iterations < 0:
            raise ValueError("window_iterations must be a non-negative integer.")
        total_steps = iterations + window_iterations
        if profile is not None and profile.device == "gpu" and hasattr(a_op, "product_into") \
                and hasattr(at_op, "product_into"):
            from .gpu_operator import GpuSolver

            solver = GpuSolver(a_op, at_op, ray_norm, voxel_norm)
        sinograms = None if sink is not None else np.empty(
            (nz, detector_count, len(angles_rad)), dtype=np.float32
        )
        for start in range(0, nz, block_size):
            _check_cancelled(cancelled_cb)
            stop = min(nz, start + block_size)
            target_block = _target_block(target, start, stop, grid_res)
            projected = _project(a_op, target_block, library)
            hits = projected > 0
            filtered = filter_sinograms_3d(projected.reshape(len(angles_rad), detector_count, -1).transpose(1, 0, 2))
            values = np.ascontiguousarray(filtered.transpose(1, 0, 2).reshape(a.shape[0], -1), dtype=np.float32)
            del projected, filtered
            np.maximum(values, 0, out=values)
            values[~hits] = 0
            workspace = _project(at_op, values, library)
            # Least-squares scale of the initializer, independently for each layer.
            numerator = np.einsum("ij,ij,i->j", workspace, target_block, voxel_norm, dtype=np.float64)
            denominator = np.einsum("ij,ij,i->j", workspace, workspace, voxel_norm, dtype=np.float64)
            scale = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
            values *= scale.astype(np.float32)[None, :]
            on_device = False
            if solver is not None and solver.usable:
                from .gpu_operator import GpuFailure

                def on_step(kind: str, index: int, start=start, stop=stop) -> None:
                    _check_cancelled(cancelled_cb)
                    if not report_cb:
                        return
                    if kind == "ls":
                        progress = (start + (stop-start) * (index+1) / total_steps) / nz
                        report_cb(0.36 + 0.42 * progress, f"SIRT {index+1}/{iterations}; Z {start+1}–{stop}/{nz}")
                    else:
                        progress = (start + (stop-start) * (iterations+index+1) / total_steps) / nz
                        report_cb(
                            0.36 + 0.42 * progress,
                            f"Dose window {index+1}/{window_iterations}; Z {start+1}–{stop}/{nz}",
                        )

                try:
                    values = solver.run(
                        values, target_block, hits, iterations, window_iterations,
                        window_level, window_steepness, on_step,
                    )
                    on_device = True
                except GpuFailure as failure:
                    solver.disabled = True  # redo this block, and the rest, on the CPU
                    solver.error = f"{failure}: {failure.__cause__}" if failure.__cause__ else str(failure)
            if not on_device:
                for iteration in range(iterations):
                    _check_cancelled(cancelled_cb)
                    if library is None:
                        _python_sirt_step(a_op, at_op, values, target_block, ray_norm, voxel_norm, hits, workspace)
                    else:
                        library.sirt_linear_step(
                            a.shape[0], a.shape[1], stop-start, _pointer(values), _pointer(target_block),
                            *_csr_pointers(a), *_csr_pointers(at), _pointer(ray_norm), _pointer(voxel_norm),
                            hits.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)), _pointer(workspace),
                        )
                    if report_cb:
                        progress = (start + (stop-start) * (iteration+1) / total_steps) / nz
                        report_cb(0.36 + 0.42 * progress, f"SIRT {iteration+1}/{iterations}; Z {start+1}–{stop}/{nz}")
                for step in range(window_iterations):
                    _check_cancelled(cancelled_cb)
                    _window_step(
                        a_op, at_op, values, target_block, ray_norm, voxel_norm, hits, library,
                        window_level, window_steepness,
                    )
                    if report_cb:
                        progress = (start + (stop-start) * (iterations+step+1) / total_steps) / nz
                        report_cb(
                            0.36 + 0.42 * progress,
                            f"Dose window {step+1}/{window_iterations}; Z {start+1}–{stop}/{nz}",
                        )
            # Tell the user as soon as the device was given up (after this block, so that a
            # short job, or the last block of a long one, reports it too).
            problem = _gpu_problem(a_op, at_op, solver)
            if problem and not warned:
                warned = True
                if report_cb:
                    report_cb(0.78, tr("GPU недоступна, расчёт продолжен на процессоре: {error}").format(error=problem))
            if sink is not None:
                sink(start, stop, values)
            else:
                assert sinograms is not None
                sinograms[start:stop] = values.reshape(len(angles_rad), detector_count, -1).transpose(2, 1, 0)
        if sinograms is None:
            return np.empty((0, 0, 0), dtype=np.float32)
        maximum = float(np.max(sinograms))
        if maximum > 0:
            sinograms /= maximum
        return sinograms
    finally:
        if status is not None:
            problem = _gpu_problem(a_op, at_op, solver)
            status["device_used"] = "gpu" if wanted_gpu and not problem else "cpu"
            status["gpu_error"] = problem or ""
        _release(solver, a_op, at_op)


def project_volume(
    target_vol: np.ndarray, angles_rad: np.ndarray, L_out: float, L_in: float,
    grid_res: int, grid_span_mm: float, container_mode: int = 0,
    report_cb: Optional[Callable[[float, str], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    batch_layers: Optional[int] = None, profile: Optional[ComputeProfile] = None,
) -> np.ndarray:
    """Bare projection: straight rays, no refraction and no absorption.

    Same container geometry and detector as the optimised frames, but every
    refractive index is 1, so this is what a plain projection of the part looks
    like without any compensation for the vial. Output is [z, detector, angle],
    in ray-length units and not normalised.
    """
    target = np.asarray(target_vol)
    if target.ndim != 3 or target.shape[:2] != (grid_res, grid_res):
        raise ValueError("Target must have shape (grid_res, grid_res, layers).")
    a, _ = build_system_matrix(
        angles_rad, grid_res, grid_res, grid_span_mm, L_out, L_in,
        1.0, 1.0, 1.0, 1.0, 0.0, container_mode, report_cb, cancelled_cb, lean_transpose=True,
        workers=profile.threads if profile else 1,
        use_cache=False,  # used once; it must not push the main matrix out of the cache
    )
    library = _load_cpp_backend()
    if a.indices.dtype != np.int32 or a.indptr.dtype != np.int32:
        library = None
    a_op, _, library, _label = _operators(a, None, profile, library)
    try:
        nz = target.shape[2]
        bytes_per_layer = 4 * (2 * a.shape[0] + a.shape[1])
        block_size = max(1, _workspace_bytes(profile) // bytes_per_layer)
        if batch_layers is not None:
            block_size = min(block_size, int(batch_layers))
        result = np.empty((nz, grid_res, len(angles_rad)), dtype=np.float32)
        for start in range(0, nz, block_size):
            _check_cancelled(cancelled_cb)
            stop = min(nz, start + block_size)
            block = _target_block(target, start, stop, grid_res)
            projected = _project(a_op, block, library)
            result[start:stop] = projected.reshape(len(angles_rad), grid_res, -1).transpose(2, 1, 0)
        return result
    finally:
        _release(a_op)


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
