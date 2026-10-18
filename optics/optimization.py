"""
OpenCAL / Tomo Physical-Optics Projection Optimiser for SpinSlicer.

Implements the canonical OpenCAL (Computed Axial Lithography) algorithm:
  - Vectorised Snell ray tracing with refractive index matching (air, glass, liquid, resin)
  - Continuous ray-beam voxel mapping for artifact-free projector matrices
  - Ram-Lak (ramp) Fourier frequency filtering to invert the 1/r backprojection blur
  - Sigmoid-thresholded photopolymer gelation response gradient descent
  - High-performance batch SpMM (Sparse Matrix-Matrix Multiplication) across all 3D slices simultaneously

References:
  - Kelly et al., Science 363(6431), 1075-1079 (2019)
  - OpenCAL / Tomo software: https://opencal-org.readthedocs.io/
  - VAMToolbox: https://github.com/computed-axial-lithography/VAMToolbox
"""

from typing import Callable, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.fft import fft, ifft

from .ray_tracing import trace_cuvette, trace_immersion_vial

# ---------------------------------------------------------------------------
#  Ray tracing
# ---------------------------------------------------------------------------

def _precompute_rays(
    angles_rad: np.ndarray,
    s_positions: np.ndarray,
    L_out: float,
    L_in: float,
    n_air: float,
    n_glass: float,
    n_liquid: float,
    n_resin: float,
    container_mode: int,
) -> list:
    """Trace ray paths for every angle in the sequence."""
    cache = []
    for theta in angles_rad:
        if container_mode == 1:
            p_resin, v_resin, valid = trace_cuvette(
                s_positions, theta, L_out, L_in, n_air, n_glass, n_resin
            )
        elif container_mode == 2:
            p_resin, v_resin, valid = trace_immersion_vial(
                s_positions, theta, L_out, L_in, n_air, n_glass, n_liquid, n_resin
            )
        else:
            # Mode 0: ideal parallel rays (no container refraction)
            N = len(s_positions)
            p_resin = np.column_stack([s_positions, np.full(N, L_out / 2.0)])
            v_resin = np.tile(np.array([0.0, -1.0]), (N, 1))
            valid = np.ones(N, dtype=bool)
            c, s = np.cos(-theta), np.sin(-theta)
            rot = np.array([[c, -s], [s, c]])
            p_resin = p_resin @ rot.T
            v_resin = v_resin @ rot.T

        valid_idx = np.where(valid)[0]
        cache.append((valid_idx, p_resin, v_resin))
    return cache


# ---------------------------------------------------------------------------
#  Sparse system matrix construction
# ---------------------------------------------------------------------------

_matrix_cache = {}

def build_system_matrix(
    angles_rad: np.ndarray,
    num_s: int,
    grid_res: int,
    grid_span_mm: float,
    L_out: float,
    L_in: float,
    n_air: float = 1.0,
    n_glass: float = 1.51,
    n_liquid: float = 1.45,
    n_resin: float = 1.50,
    alpha: float = 0.0,
    container_mode: int = 0,
    report_cb: Optional[Callable[[float, str], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build the physical optics projector matrix A and adjoint backprojector A^T."""
    cache_key = (
        hash(angles_rad.tobytes()), num_s, grid_res, round(grid_span_mm, 4),
        round(L_out, 4), round(L_in, 4), round(n_air, 4), round(n_glass, 4),
        round(n_liquid, 4), round(n_resin, 4), round(alpha, 4), container_mode
    )
    if cache_key in _matrix_cache:
        return _matrix_cache[cache_key]

    s_positions = np.linspace(-L_out / 2.0, L_out / 2.0, num_s)
    dx = grid_span_mm / grid_res
    R = grid_span_mm / 2.0
    t_samples = np.arange(-R, R, dx)
    N_t = len(t_samples)

    batches = []
    num_angles = len(angles_rad)
    report_interval = max(1, num_angles // 10)

    for i, theta in enumerate(angles_rad):
        if cancelled_cb and cancelled_cb():
            from slicing_engine import SliceCancelled
            raise SliceCancelled()

        if container_mode == 1:
            p_resin, v_resin, valid = trace_cuvette(s_positions, theta, L_out, L_in, n_air, n_glass, n_resin)
        elif container_mode == 2:
            p_resin, v_resin, valid = trace_immersion_vial(s_positions, theta, L_out, L_in, n_air, n_glass, n_liquid, n_resin)
        else:
            N = len(s_positions)
            p_resin = np.column_stack([s_positions, np.full(N, L_out / 2.0)])
            v_resin = np.tile(np.array([0.0, -1.0]), (N, 1))
            valid = np.ones(N, dtype=bool)
            c, s = np.cos(-theta), np.sin(-theta)
            rot = np.array([[c, -s], [s, c]])
            p_resin = p_resin @ rot.T
            v_resin = v_resin @ rot.T

        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            batches.append(sp.csr_matrix((num_s, grid_res**2), dtype=np.float32))
            continue

        p = p_resin[valid_idx]
        v = v_resin[valid_idx]

        P_x = (p[:, 0:1] + v[:, 0:1] * t_samples).ravel()
        P_y = (p[:, 1:2] + v[:, 1:2] * t_samples).ravel()

        inside = (P_x**2 + P_y**2) < (R**2)
        P_x = P_x[inside]
        P_y = P_y[inside]

        gx = (P_x - (-grid_span_mm / 2.0 + dx / 2.0)) / dx
        gy = (P_y - (-grid_span_mm / 2.0 + dx / 2.0)) / dx

        valid_pts = (gx >= 0) & (gx < grid_res - 1) & (gy >= 0) & (gy < grid_res - 1)
        gx = gx[valid_pts]
        gy = gy[valid_pts]

        if len(gx) == 0:
            batches.append(sp.csr_matrix((num_s, grid_res**2), dtype=np.float32))
            continue

        ray_idx = np.repeat(valid_idx, N_t)[inside][valid_pts]

        x0 = np.floor(gx).astype(np.int32)
        y0 = np.floor(gy).astype(np.int32)
        wx = gx - x0
        wy = gy - y0

        w00 = ((1.0 - wx) * (1.0 - wy)).astype(np.float32) * dx
        w10 = (wx * (1.0 - wy)).astype(np.float32) * dx
        w01 = ((1.0 - wx) * wy).astype(np.float32) * dx
        w11 = (wx * wy).astype(np.float32) * dx

        if alpha > 0.0:
            t_flat = np.tile(t_samples, len(valid_idx))[inside][valid_pts]
            p_flat_x = np.repeat(p[:, 0], N_t)[inside][valid_pts]
            p_flat_y = np.repeat(p[:, 1], N_t)[inside][valid_pts]
            v_flat_x = np.repeat(v[:, 0], N_t)[inside][valid_pts]
            v_flat_y = np.repeat(v[:, 1], N_t)[inside][valid_pts]
            p_dot_v = p_flat_x * v_flat_x + p_flat_y * v_flat_y
            p_sq = p_flat_x**2 + p_flat_y**2
            D = p_dot_v**2 - (p_sq - R**2)
            valid_D = D > 0
            dist = np.zeros(len(gx), dtype=np.float32)
            t_entry = -p_dot_v[valid_D] - np.sqrt(D[valid_D])
            dist[valid_D] = np.maximum(t_flat[valid_D] - t_entry, 0.0)
            atten = np.exp(-alpha * dist).astype(np.float32)
            w00 *= atten
            w10 *= atten
            w01 *= atten
            w11 *= atten

        c00 = y0 * grid_res + x0
        c10 = y0 * grid_res + (x0 + 1)
        c01 = (y0 + 1) * grid_res + x0
        c11 = (y0 + 1) * grid_res + (x0 + 1)

        r_cat = np.concatenate([ray_idx, ray_idx, ray_idx, ray_idx])
        c_cat = np.concatenate([c00, c10, c01, c11])
        v_cat = np.concatenate([w00, w10, w01, w11])

        chunk = sp.csr_matrix((v_cat, (r_cat, c_cat)), shape=(num_s, grid_res**2), dtype=np.float32)
        chunk.sum_duplicates()
        batches.append(chunk)

        if report_cb and (i % report_interval == 0 or i == num_angles - 1):
            report_cb(0.30 + 0.04 * (i / num_angles), f"Построение матрицы лучей {i + 1}/{num_angles}...")

    A = sp.vstack(batches, format="csr")
    del batches
    AT = A.T.tocsr()
    res = (A, AT)

    _matrix_cache.clear()
    _matrix_cache[cache_key] = res
    return res


def filter_sinograms_3d(b_3d: np.ndarray, filter_name: str = "ram-lak") -> np.ndarray:
    """Apply Ram-Lak ramp filter along the detector dimension to invert 1/r backprojection blur."""
    pX, pY, pZ = b_3d.shape
    padded = max(64, int(2 ** np.ceil(np.log2(2 * pX))))
    pad_w = ((0, padded - pX), (0, 0), (0, 0))

    # Kak & Slaney spatial-domain Ram-Lak kernel
    n = np.concatenate((
        np.arange(1, padded / 2 + 1, 2, dtype=int),
        np.arange(padded / 2 - 1, 0, -2, dtype=int),
    ))
    f = np.zeros(padded)
    f[0] = 0.25
    f[1::2] = -1.0 / (np.pi * n) ** 2
    fourier_filter = (2.0 * np.real(fft(f)))[:, None, None]

    img = np.pad(b_3d, pad_w, mode="constant", constant_values=0)
    filt_img = fft(img, axis=0) * fourier_filter
    return np.real(ifft(filt_img, axis=0)[:pX, :, :])


import ctypes
import os
import sys

_sirt_lib = None
_lib_loaded = False
_dll_directory_handle = None

def _load_cpp_backend():
    global _sirt_lib, _lib_loaded, _dll_directory_handle
    if _lib_loaded:
        return _sirt_lib

    _lib_loaded = True
    try:
        candidate_dirs = []
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root and os.path.isdir(str(bundle_root)):
            candidate_dirs.append(os.path.join(str(bundle_root), "cpp_backend"))
            candidate_dirs.append(str(bundle_root))

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate_dirs.append(os.path.join(base_dir, "cpp_backend"))
        candidate_dirs.append(base_dir)

        lib_names = ["sirtsolver.dll", "sirtsolver.so", "libsirt.so", "sirtsolver.dylib"]
        lib_path = None
        for cdir in candidate_dirs:
            for name in lib_names:
                full_path = os.path.join(cdir, name)
                if os.path.isfile(full_path):
                    lib_path = full_path
                    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
                        try:
                            _dll_directory_handle = os.add_dll_directory(cdir)
                        except Exception:
                            pass
                    break
            if lib_path:
                break

        if not lib_path:
            return None

        lib = ctypes.CDLL(lib_path)
        lib.forward_project.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        lib.sirt_loop.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_bool),
            ctypes.POINTER(ctypes.c_float)
        ]
        _sirt_lib = lib
        return _sirt_lib
    except Exception as e:
        print(f"Warning: Failed to load C++ backend: {e}")
        return None

# ---------------------------------------------------------------------------
#  High-speed Volume OpenCAL / Tomo Optimiser
# ---------------------------------------------------------------------------

def optimize_projections_sirt_volume(
    target_vol: np.ndarray,
    angles_rad: np.ndarray,
    L_out: float,
    L_in: float,
    n_air: float = 1.0,
    n_glass: float = 1.51,
    n_liquid: float = 1.45,
    n_resin: float = 1.50,
    alpha: float = 0.0,
    grid_res: int = 128,
    grid_span_mm: float = 50.0,
    iterations: int = 8,
    container_mode: int = 0,
    report_cb: Optional[Callable[[float, str], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """Optimise projection sequence using the canonical OpenCAL / Tomo algorithm."""
    nz = target_vol.shape[2]
    num_frames = len(angles_rad)
    num_s = grid_res

    if report_cb:
        report_cb(0.31, "OpenCAL: построение оптической лучевой матрицы (Ray-Driven)...")

    if cancelled_cb and cancelled_cb():
        from slicing_engine import SliceCancelled
        raise SliceCancelled()

    A, AT = build_system_matrix(
        angles_rad,
        num_s=num_s,
        grid_res=grid_res,
        grid_span_mm=grid_span_mm,
        L_out=L_out,
        L_in=L_in,
        n_air=n_air,
        n_glass=n_glass,
        n_liquid=n_liquid,
        n_resin=n_resin,
        alpha=alpha,
        container_mode=container_mode,
        report_cb=report_cb,
        cancelled_cb=cancelled_cb,
    )

    if cancelled_cb and cancelled_cb():
        from slicing_engine import SliceCancelled
        raise SliceCancelled()

    V_target = target_vol.reshape(grid_res * grid_res, nz).astype(np.float32)

    if report_cb:
        report_cb(0.34, "OpenCAL: C++ прямое проецирование и Ram-Lak фильтрация...")

    lib = _load_cpp_backend()

    if lib is not None:
        num_rays = A.shape[0]
        num_pixels = A.shape[1]
        P0 = np.zeros((num_rays, nz), dtype=np.float32)
        A_csr = A.tocsr()
        AT_csr = AT.tocsr()

        lib.forward_project(
            num_rays, num_pixels, nz,
            V_target.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            A_csr.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            A_csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            A_csr.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            P0.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
    else:
        # 1. Forward project target volume: (num_frames * num_s, nz)
        P0 = A.dot(V_target)

    # 2. Reshape to (num_s, num_frames, nz) and apply Ram-Lak frequency filter
    P0_3d = P0.reshape(num_frames, num_s, nz).transpose(1, 0, 2)
    b = filter_sinograms_3d(P0_3d, filter_name="ram-lak")
    b = np.maximum(b, 0.0)

    # Calculate SIRT normalization factors (matrix norm based learning rate)
    ones_pixels = np.ones(A.shape[1], dtype=np.float32)
    ones_rays = np.ones(A.shape[0], dtype=np.float32)

    row_sums = A.dot(ones_pixels)
    col_sums = AT.dot(ones_rays)

    R_norm = np.zeros_like(row_sums)
    R_norm[row_sums > 0] = 1.0 / row_sums[row_sums > 0]

    C_norm = np.zeros_like(col_sums)
    C_norm[col_sums > 0] = 1.0 / col_sums[col_sums > 0]

    V_any = (V_target.sum(axis=1) > 0).astype(np.float32)   # (grid_res^2,)
    ray_hits_any = A.dot(V_any) > 1e-5                        # (num_rays,)

    # 3. OpenCAL iterative gradient descent with photopolymer sigmoid response
    d_h = 0.65  # Gelation threshold dose constraint
    if lib is not None:
        if report_cb:
            report_cb(0.36, "OpenCAL: C++ SIRT оптимизация (AVX/OpenMP)...")

        b_flat = b.transpose(1, 0, 2).reshape(num_frames * num_s, nz).astype(np.float32)
        b_opt = np.zeros_like(b_flat)
        ray_hits_bool = ray_hits_any.astype(np.bool_)

        lib.sirt_loop(
            num_rays, num_pixels, nz, max(1, iterations), d_h,
            b_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            V_target.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            A_csr.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            A_csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            A_csr.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            AT_csr.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            AT_csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            AT_csr.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            R_norm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            C_norm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ray_hits_bool.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            b_opt.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
    else:
        if report_cb:
            report_cb(0.36, "OpenCAL: Python SIRT оптимизация (global normalization)...")

        b_opt = b.transpose(1, 0, 2).reshape(num_frames * num_s, nz)

        for it in range(max(1, iterations)):
            if cancelled_cb and cancelled_cb():
                from slicing_engine import SliceCancelled
                raise SliceCancelled()

            # Reconstruct delivered dose in resin vat: (pixels, nz)
            dose = AT.dot(b_opt)

            # Proper GLOBAL normalization (preserves relative contrast between layers!)
            dose_max = float(np.max(dose))
            if dose_max <= 0.0:
                dose_max = 1.0
            dose_norm = dose / dose_max

            # Sigmoid threshold response modeling resin photopolymerization
            dose_arg = np.clip(-15.0 * (dose_norm - d_h), -80.0, 80.0)
            dose_thresh = 1.0 / (1.0 + np.exp(dose_arg))

            # Error computation with sigmoid derivative (Non-linear Gradient Descent)
            sigmoid_deriv = 15.0 * dose_thresh * (1.0 - dose_thresh)
            sigmoid_deriv = np.clip(sigmoid_deriv, 0.05, None)  # Prevent dead gradients
            error = (V_target - dose_thresh) * sigmoid_deriv

            # Proper SIRT update rule based on matrix norms
            delta_b = A.dot(C_norm[:, None] * error)
            b_opt = b_opt + R_norm[:, None] * delta_b

            # Apply non-negativity constraint
            np.maximum(b_opt, 0.0, out=b_opt)

            # Zero-dose suppression: ray_hits_any is 1-D → broadcasts over nz
            b_opt[~ray_hits_any, :] = 0.0

            if report_cb:
                frac = 0.36 + 0.42 * ((it + 1) / iterations)
                report_cb(frac, f"OpenCAL итерация {it + 1}/{iterations} (резкость + подавление пересветов)...")

    # Reshape b_opt back to (num_frames, num_s, nz)
    b_opt_3d = b_opt.reshape(num_frames, num_s, nz).transpose(1, 0, 2)

    # Shape: (nz, num_s, num_frames) = (nz, grid_res, num_frames)
    sinograms = b_opt_3d.transpose(2, 0, 1).astype(np.float32)

    # Clean normalisation using robust max (99.9th percentile) to avoid hot-pixel darkening
    sino_max = float(np.percentile(sinograms, 99.9))
    if sino_max > 0:
        sinograms /= sino_max
        np.clip(sinograms, 0.0, 1.0, out=sinograms)

    return sinograms


# ---------------------------------------------------------------------------
#  Single-slice backward-compatible wrapper
# ---------------------------------------------------------------------------

def optimize_projections_sirt(
    V_target: np.ndarray,
    angles_rad: np.ndarray,
    L_out: float,
    L_in: float,
    n_air: float = 1.0,
    n_glass: float = 1.51,
    n_liquid: float = 1.45,
    n_resin: float = 1.50,
    alpha: float = 0.0,
    num_s: int = 128,
    grid_span_mm: float = 50.0,
    iterations: int = 8,
    container_mode: int = 0,
) -> np.ndarray:
    """Single-slice compatibility wrapper."""
    grid_res = V_target.shape[0]
    V_vol = V_target.reshape(grid_res, grid_res, 1)
    sinograms = optimize_projections_sirt_volume(
        V_vol,
        angles_rad,
        L_out=L_out,
        L_in=L_in,
        n_air=n_air,
        n_glass=n_glass,
        n_liquid=n_liquid,
        n_resin=n_resin,
        alpha=alpha,
        grid_res=grid_res,
        grid_span_mm=grid_span_mm,
        iterations=iterations,
        container_mode=container_mode,
    )
    return sinograms[0].T
