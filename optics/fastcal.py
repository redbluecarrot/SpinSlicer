"""FastCAL: rotation-symmetric projection planner for cylindrical VAM vials.

Why it is fast
--------------
In a cylindrical vial every projection angle sees the same physical ray
bundle, only rotated. Snell refraction at the round glass walls and
Beer-Lambert attenuation in the resin are both rotation invariant. On a polar
grid whose angular step divides the projection step, a rotation by one frame
is a cyclic shift of the angular index, so the projector is block-circulant.
A DFT along that index splits it into ~N_theta/2 small dense matrices M_k
(detector bins x radial cells).

* Rays are traced for ONE angle only. No full sparse system matrix is stored
  (the 256 x 360 SIRT matrix pair needs ~315 MiB; M_k needs ~66 MiB).
* Forward and back projection are FFTs plus batched dense products (BLAS).
* The unconstrained weighted least-squares problem is solved exactly per
  frequency. This is a refraction- and attenuation-exact analogue of FBP
  and is used as the starting point.

What it optimises
-----------------
A printed part needs dose above the gel threshold inside and below it
outside. It does not need an exact dose value. The objective is therefore a
band constraint, similar to BCLP or OSMO:

    in-part      D >= d_high          (soft cap D <= over)
    out-of-part  D <= d_low
    edge cells   D ~= d_low + (d_high - d_low) * t
                 (partial-volume cells; this pins the threshold crossing to
                 the mesh boundary at t = 0.5)

It is solved with a projected L-BFGS (P >= 0), entirely in float32 numpy.

Neighbouring Z layers are nearly identical. Layers are solved coarse-to-fine:
every 16th layer from scratch, then the layers between them start from
interpolated neighbour solutions. Layers whose geometry differs from that
interpolation are detected and solved from scratch.

Units and conventions are the same as optics.optimization: millimetres;
target volume [x, y, z] with flat pixel index x*N + y; output projections
[z, detector, angle] normalised to a global maximum of 1. The model is a
relative linear dose. It is not a calibrated cure prediction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp

from .geometry import detector_span_mm
from .ray_tracing import intersect_circle

DEFAULT_SCHEDULE = ((16, 30), (8, 10), (4, 6), (2, 4), (1, 3))


BATCH_LAYERS = 24
# Thin-feature awareness (FastVialEngine._thin_features). Chosen with tools/engine_judge.py at
# grid 128, judge 256 (docs/validation/engine-judge-2026-10-31.md): on the resolution target
# surface error 0.172 -> 0.081 mm, worst robust IoU 0.888 -> 0.911, 0.8 mm pins and holes print;
# Benchy and the nut unchanged. Boost 6 lost 0.003 IoU on Benchy; snapping thin solids cost
# worst-case robustness.
THIN_BOOST = 3.0
THIN_RADIUS = 2.5          # voxels
SNAP_THIN_VOID = True
SNAP_THIN_SOLID = False


def detector_bins(grid_res: int, output_res: int) -> int:
    """Detector bins FastCAL plans with: finer than the voxel grid, within the projector's pixels.

    The projector shows ``output_res`` pixels across the vial, far finer than the voxels.
    With one bin per voxel the linear pixel profile spreads every bin's light over two
    voxels; with more bins the dose can turn sharper between voxel centres. Measured with
    tools/engine_judge.py at grid 128 with 2 bins per voxel (judge 256, 2026-10-31): higher
    IoU, lower surface error and better mean robustness on all four test jobs. The cost
    grows with the bins (2 bins: 1.5-2.2x the time), so per quality setting:
      grid <= 128 ("Fast", a quick preview): 1 bin per voxel;
      finer grids ("Normal", "Fine"): 1.5 bins per voxel, at most 576.
    """
    grid = max(1, int(grid_res))
    if grid <= 128:
        return grid
    bins = min(int(round(1.5 * grid)), max(int(output_res), grid), 576)
    return int(max(grid, bins - bins % 2))


# Thin-feature awareness (FastVialEngine._thin_features). Chosen with tools/engine_judge.py at
# grid 128, judge 256 (docs/validation/engine-judge-2026-10-31.md): on the resolution target
# surface error 0.172 -> 0.081 mm, worst robust IoU 0.888 -> 0.911, 0.8 mm pins and holes print;
# Benchy and the nut unchanged. Boost 6 lost 0.003 IoU on Benchy; snapping thin solids cost
# worst-case robustness.
THIN_BOOST = 3.0
THIN_RADIUS = 2.5          # voxels
SNAP_THIN_VOID = True
SNAP_THIN_SOLID = False


# ------------------------------------------------------------------ threads
# The operator is a stack of independent per-frequency matrix products. numpy's
# stacked matmul runs them one after another on a single core; split across all
# cores they scale almost linearly (BLAS itself is kept single-threaded in the
# engine process, see engine_process.py, so the two never oversubscribe).
_POOL: Any = None
_THREADS = 0


def engine_threads() -> int:
    """Worker threads for the engine: SPINSLICER_THREADS, else every core we may use."""
    import os
    value = os.environ.get("SPINSLICER_THREADS", "").strip()
    if value.isdigit() and int(value) > 0:
        return int(value)
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _pool() -> Any:
    global _POOL, _THREADS
    n = engine_threads()
    if _POOL is None or _THREADS != n:
        from concurrent.futures import ThreadPoolExecutor
        if _POOL is not None:
            _POOL.shutdown(wait=False)
        _POOL = ThreadPoolExecutor(max_workers=n, thread_name_prefix="fastcal")
        _THREADS = n
    return _POOL


def _chunks(n: int, parts: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n, parts + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def parallel_map(fn: Callable[[int, int], None], n: int, min_items: int = 2) -> None:
    """Run fn(start, stop) over [0, n) split into one chunk per thread."""
    threads = engine_threads()
    if threads <= 1 or n < min_items:
        fn(0, n)
        return
    list(_pool().map(lambda ab: fn(*ab), _chunks(n, min(threads * 2, n))))


def available_memory_mb() -> Optional[float]:
    """Free RAM the OS could give us now (Linux, Windows, macOS); None if unknown."""
    import os
    import sys
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024.0
        elif sys.platform == "win32":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return status.ullAvailPhys / 2**20
        elif sys.platform == "darwin":
            # No cheap "available" figure: take a third of physical memory.
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**20 / 3
    except (OSError, ValueError, AttributeError):
        pass
    return None


def default_memory_budget_mb() -> int:
    """Working memory for one batch of layers: SPINSLICER_ENGINE_MB, else a fifth of free RAM.

    Wider batches make the per-frequency products far more efficient (measured at
    384 cells / 720 angles: 5.7 ms per layer in batches of 7, 1.5 ms in batches of
    32), so the engine takes what the machine can spare, between 160 and 1280 MB.
    """
    import os
    value = os.environ.get("SPINSLICER_ENGINE_MB", "").strip()
    if value.isdigit() and int(value) > 0:
        return int(value)
    free = available_memory_mb()
    if free is None:
        return 160
    return int(min(1280.0, max(160.0, 0.2 * free)))


def run_batches(fn: Callable[[list[int]], None], batches: list[list[int]], parallel: int) -> None:
    """fn(batch) for every batch, up to `parallel` at a time (exceptions propagate)."""
    if parallel <= 1 or len(batches) <= 1:
        for batch in batches:
            fn(batch)
        return
    from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
    with ThreadPoolExecutor(max_workers=min(parallel, len(batches)), thread_name_prefix="fastcal-batch") as ex:
        futures = [ex.submit(fn, batch) for batch in batches]
        finished, pending = wait(futures, return_when=FIRST_EXCEPTION)
        for future in pending:
            future.cancel()
        for future in futures:
            if future.done() and not future.cancelled():
                future.result()          # re-raise the first error (cancel, memory, ...)


def batched_matmul(A: np.ndarray, B: np.ndarray, conj_out: bool = False,
                   add: Optional[tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """A[k] @ B[k] for every k, on all cores. A may be a strided view (no copy is made).

    add=(c, X) adds c[k] * X[k] to every product (a diagonal term in frequency space).
    """
    K, m = A.shape[0], A.shape[1]
    out = np.empty((K, m, B.shape[2]), np.result_type(A, B))
    work = K * m * A.shape[2] * B.shape[2]

    def run(a: int, b: int) -> None:
        np.matmul(A[a:b], B[a:b], out=out[a:b])
        if conj_out:
            np.conjugate(out[a:b], out=out[a:b])
        if add is not None:
            coef, X = add
            out[a:b] += coef[a:b, None, None] * X[a:b]

    if work < 4_000_000:          # tiny products: thread hand-off costs more than it saves
        run(0, K)
    else:
        parallel_map(run, K)
    return out


# ------------------------------------------------------------------ polar grid
@dataclass
class PolarGrid:
    """Rings of equal radial pitch. Ring r holds n_theta * L[r] equal cells.

    A cell is addressed by (j, q). j in [0, n_theta) is the coarse angular
    index; one projection step shifts j by one. q = q_offset[r] + l, where
    l < L[r] is the sub-cell inside one step.
    """

    radius: float
    n_rings: int
    n_theta: int
    L: np.ndarray
    q_offset: np.ndarray
    orientation: int

    @property
    def dr(self) -> float:
        return self.radius / self.n_rings

    @property
    def Q(self) -> int:
        return int(self.L.sum())

    @property
    def n_cells(self) -> int:
        return self.Q * self.n_theta

    def cell_area(self) -> np.ndarray:
        r = np.arange(self.n_rings)
        ring_area = math.pi * ((r + 1) ** 2 - r ** 2) * self.dr ** 2
        return np.repeat(ring_area / (self.L * self.n_theta), self.L).astype(np.float32)

    def locate(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cell (j, q) of each point. q = -1 outside the grid."""
        rho = np.hypot(x, y)
        phi = np.mod(self.orientation * np.arctan2(y, x), 2 * np.pi)
        ring = np.floor(rho / self.dr).astype(np.int64)
        inside = ring < self.n_rings
        ring = np.minimum(ring, self.n_rings - 1)
        Lr = self.L[ring]
        c = np.minimum(np.floor(phi / (2 * np.pi) * self.n_theta * Lr).astype(np.int64),
                       self.n_theta * Lr - 1)
        j = c // Lr
        q = self.q_offset[ring] + c % Lr
        q[~inside] = -1
        return j, q


def make_polar_grid(radius: float, pitch: float, n_theta: int, orientation: int = 1,
                    angular_oversample: float = 1.0) -> PolarGrid:
    n_rings = max(1, int(math.ceil(radius / pitch)))
    mid = (np.arange(n_rings) + 0.5) * radius / n_rings
    L = np.maximum(1, np.ceil(2 * np.pi * mid * angular_oversample / (n_theta * pitch))).astype(np.int64)
    q_offset = np.concatenate([[0], np.cumsum(L)[:-1]]).astype(np.int64)
    return PolarGrid(float(radius), n_rings, int(n_theta), L, q_offset, int(orientation))


# ----------------------------------------------------------------- ray optics
@dataclass(frozen=True)
class OpticsOptions:
    """Physical effects included in the projector model.

    fresnel         Fresnel (unpolarised) transmission at every refracting wall.
    pixel_profile   Light profile of one detector bin across the beam:
                    "linear" is the hat function produced by the linear
                    up-sampling used for PNG export (model == what is shown);
                    "box" is a flat pixel; "ray" is the legacy single centre ray.
    sub_rays        Rays per bin used to integrate that profile.
    rotation_smear  The vial keeps turning while a frame is shown, so each
                    frame is integrated over its angular step, centred on
                    its scheduled angle.
    smear_steps     Sub-angles used for that integral.
    throw_mm        Projector pupil to vial axis distance; 0 = telecentric
                    (parallel rays). For an immersion tank give the apparent
                    distance in the liquid.
    """

    fresnel: bool = True
    pixel_profile: str = "linear"
    sub_rays: int = 4
    rotation_smear: bool = True
    smear_steps: int = 4
    throw_mm: float = 0.0

    def validate(self) -> None:
        if self.pixel_profile not in ("linear", "box", "ray"):
            raise ValueError("pixel_profile must be 'linear', 'box' or 'ray'.")
        if int(self.sub_rays) < 1 or int(self.smear_steps) < 1:
            raise ValueError("sub_rays and smear_steps must be positive.")
        if not math.isfinite(self.throw_mm) or self.throw_mm < 0:
            raise ValueError("throw_mm must be finite and nonnegative (0 = telecentric).")


LEGACY_OPTICS = OpticsOptions(fresnel=False, pixel_profile="ray", sub_rays=1,
                              rotation_smear=False, smear_steps=1)


def fresnel_transmission(cos_i: np.ndarray, n1: float, n2: float) -> np.ndarray:
    """Unpolarised power transmission 1 - (Rs + Rp) / 2; zero beyond total reflection."""
    cos_i = np.clip(np.abs(cos_i), 0.0, 1.0)
    sin_t2 = (n1 / n2) ** 2 * (1.0 - cos_i ** 2)
    ok = sin_t2 <= 1.0
    cos_t = np.sqrt(np.maximum(1.0 - sin_t2, 0.0))
    rs = (n1 * cos_i - n2 * cos_t) / np.maximum(n1 * cos_i + n2 * cos_t, 1e-300)
    rp = (n2 * cos_i - n1 * cos_t) / np.maximum(n2 * cos_i + n1 * cos_t, 1e-300)
    return np.where(ok, 1.0 - 0.5 * (rs * rs + rp * rp), 0.0)


def _refract(v: np.ndarray, normal: np.ndarray, n1: float, n2: float):
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-300)
    cos_i = -np.sum(v * normal, axis=1)
    normal = np.where(cos_i[:, None] < 0, -normal, normal)
    cos_i = np.abs(cos_i)
    eta = n1 / n2
    disc = 1.0 - eta ** 2 * (1.0 - cos_i ** 2)
    ok = disc >= 0
    out = eta * v + (eta * cos_i - np.sqrt(np.maximum(disc, 0)))[:, None] * normal
    return out, ok, cos_i


def trace_lab_rays(origins: np.ndarray, directions: np.ndarray, L_out: float, L_in: float,
                   n_out: float, n_glass: float, n_resin: float, fresnel: bool):
    """Medium -> round glass wall -> resin, in the lab frame.

    Returns the resin entry points, unit directions, a validity mask and the
    transmitted power fraction.
    """
    t, _, ok = intersect_circle(origins, directions, L_out / 2)
    ok &= np.isfinite(t) & (t >= 0)
    p1 = origins + np.where(ok, t, 0)[:, None] * directions
    v1, ok1, c1 = _refract(directions, p1, n_out, n_glass)
    ok &= ok1
    t, _, ok2 = intersect_circle(p1, v1, L_in / 2)
    ok &= ok2 & np.isfinite(t) & (t >= -1e-9)
    p2 = p1 + np.where(ok, np.maximum(t, 0), 0)[:, None] * v1
    v2, ok3, c2 = _refract(v1, p2, n_glass, n_resin)
    ok &= ok3
    T = np.ones(len(origins))
    if fresnel:
        T = fresnel_transmission(c1, n_out, n_glass) * fresnel_transmission(c2, n_glass, n_resin)
    return p2, v2, ok, T


# -------------------------------------------------------------------- operator
class CirculantVialProjector:
    """A: polar field f[j, q, z] -> projections P[m, s, z]  (and exact A^T).

    angle_m = angle0 + direction * m * 2*pi / n_theta. With LEGACY_OPTICS
    the rays are exactly those of the sparse SIRT matrix builder.
    """

    def __init__(self, n_theta: int, num_s: int, L_out: float, L_in: float,
                 n_air: float = 1.0, n_glass: float = 1.51, n_liquid: float = 1.45,
                 n_resin: float = 1.50, alpha: float = 0.0, container_mode: int = 0,
                 pitch: Optional[float] = None, angle0: float = 0.0, direction: int = 1,
                 angular_oversample: float = 1.0, optics: OpticsOptions = LEGACY_OPTICS,
                 step_fraction: float = 1 / 16):
        if container_mode not in (0, 2):
            raise ValueError("The circulant projector needs a cylindrical vial (mode 0 or 2).")
        optics.validate()
        self.optics = optics
        self.n_theta, self.num_s = int(n_theta), int(num_s)
        detector_span = detector_span_mm(L_out, container_mode)
        self.detector_span = detector_span
        pitch = float(pitch or detector_span / num_s)
        R = L_in / 2
        # Rays rotate by -theta in the object frame (see ray_tracing._co_rotating).
        self.grid = make_polar_grid(R, pitch, n_theta, orientation=-int(direction),
                                    angular_oversample=angular_oversample)
        g = self.grid
        n_out = n_liquid if container_mode == 2 else n_air
        wall = 1.0
        if container_mode == 2 and optics.fresnel:
            # Flat tank window at normal incidence: air -> glass -> liquid.
            wall = float(fresnel_transmission(np.ones(1), n_air, n_glass)[0]
                         * fresnel_transmission(np.ones(1), n_glass, n_liquid)[0])
        # Sub-ray positions at the axis plane and their weights per bin.
        bin_w = detector_span / num_s
        S = 1 if optics.pixel_profile == "ray" else int(optics.sub_rays)
        s_sub = (np.arange(num_s * S) + 0.5) * bin_w / S - detector_span / 2
        if optics.pixel_profile == "linear":
            u = (s_sub + detector_span / 2) / bin_w - 0.5
            lo = np.floor(u).astype(np.int64)
            fr = u - lo
            b0, w0 = np.clip(lo, 0, num_s - 1), 1 - fr
            b1, w1 = np.clip(lo + 1, 0, num_s - 1), fr
        else:
            b0 = np.minimum(np.floor((s_sub + detector_span / 2) / bin_w).astype(np.int64), num_s - 1)
            w0, b1, w1 = np.ones(len(s_sub)), b0, np.zeros(len(s_sub))
        if optics.throw_mm > 0:
            d = optics.throw_mm
            dirs = np.column_stack([s_sub, np.full_like(s_sub, -d)])
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            start = 2.0 * L_out
            origins = np.column_stack([s_sub * (d - start) / d, np.full_like(s_sub, start)])
        else:
            dirs = np.tile([0.0, -1.0], (len(s_sub), 1))
            origins = np.column_stack([s_sub, np.full_like(s_sub, L_out)])
        p_lab, v_lab, ok, T = trace_lab_rays(origins, dirs, L_out, L_in, n_out, n_glass, n_resin,
                                             optics.fresnel)
        T = T * wall / S
        self.transmission = T * S
        impact = np.abs(p_lab[:, 0] * v_lab[:, 1] - p_lab[:, 1] * v_lab[:, 0])
        # Points farther from the axis than this miss some viewing directions.
        self.coverage_radius = float(min(R, impact[ok].max())) if ok.any() else 0.0
        n_smear = int(optics.smear_steps) if optics.rotation_smear else 1
        step = 2 * np.pi / n_theta
        dl = pitch * max(step_fraction, S * n_smear / 64)
        a = np.zeros(self.n_theta * num_s * g.Q, np.float32)
        for k in range(n_smear):
            theta = angle0 + (direction * ((k + 0.5) / n_smear - 0.5) * step if n_smear > 1 else 0.0)
            c, sn = math.cos(theta), math.sin(theta)
            rot = np.array([[c, -sn], [sn, c]])
            p, v = p_lab @ rot, v_lab @ rot
            entry, exit_, hit = intersect_circle(p, v, R)
            entry = np.maximum(np.where(hit, entry, 0), 0)
            good = ok & hit & (exit_ > entry)
            rays = np.flatnonzero(good)
            if not rays.size:
                continue
            e0 = entry[rays]
            nseg = np.maximum(1, np.ceil((exit_[rays] - e0) / dl)).astype(np.int64)
            h = (exit_[rays] - e0) / nseg
            ray_of = np.repeat(np.arange(len(rays)), nseg)
            k_in = np.arange(int(nseg.sum())) - np.repeat(np.cumsum(nseg) - nseg, nseg)
            hh = h[ray_of]
            tt = e0[ray_of] + (k_in + 0.5) * hh
            pts = p[rays][ray_of] + tt[:, None] * v[rays][ray_of]
            j, q = g.locate(pts[:, 0], pts[:, 1])
            depth = k_in * hh
            w = hh if alpha == 0 else np.exp(-alpha * depth) * (-np.expm1(-alpha * hh)) / alpha
            w = w * (T[rays] / n_smear)[ray_of]
            m = q >= 0
            r_m = rays[ray_of[m]]
            base = j[m] * num_s
            for bins, wb in ((b0, w0), (b1, w1)):
                wt = w[m] * wb[r_m]
                nz = wt != 0
                np.add.at(a, ((base + bins[r_m]) * g.Q + q[m])[nz], wt[nz].astype(np.float32))
            del pts, tt, hh, ray_of, k_in, w, j, q
        self.ray_ok = np.zeros(num_s, bool)
        self.ray_ok[np.unique(b0[ok])] = True
        a32 = a.reshape(self.n_theta, num_s, g.Q)                        # a[j, s, q]
        # Forward is a cyclic correlation over j:  P^_k = conj(A^_k) F_k
        A_hat = sfft.rfft(a32, axis=0, workers=-1)
        del a32
        np.conjugate(A_hat, out=A_hat)
        self.M = A_hat.astype(np.complex64, copy=False)                  # (K, Ns, Q)

    def _restore(self, args: tuple, kwargs: dict, data) -> None:
        """Rebuild a projector from cached arrays (geometry is recomputed, it is cheap)."""
        n_theta, num_s, L_out, L_in = args[0], args[1], args[2], args[3]
        container_mode = int(args[9])
        self.optics = kwargs["optics"]
        self.n_theta, self.num_s = int(n_theta), int(num_s)
        self.detector_span = detector_span_mm(L_out, container_mode)
        pitch = float(kwargs["pitch"] or self.detector_span / num_s)
        self.grid = make_polar_grid(L_in / 2, pitch, n_theta, orientation=-int(kwargs["direction"]),
                                    angular_oversample=kwargs["angular_oversample"])
        self.M = np.ascontiguousarray(data["M"])
        self.coverage_radius = float(data["coverage_radius"])
        self.ray_ok = data["ray_ok"]
        self.transmission = data["transmission"]

    @property
    def K(self) -> int:
        return self.M.shape[0]

    @property
    def nbytes(self) -> int:
        return self.M.nbytes

    def forward(self, f: np.ndarray, add_hat: Optional[tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
        """f[j, q, z] -> P[m, s, z].

        add_hat=(c, X): also adds irfft(c[k] X[k]) — a circulant term along the angle
        (the smoothness penalty), for free inside the same inverse transform.
        """
        F = sfft.rfft(f, axis=0, workers=-1)
        return sfft.irfft(batched_matmul(self.M, F, add=add_hat), n=self.n_theta, axis=0,
                          workers=-1).astype(np.float32, copy=False)

    def adjoint(self, P: np.ndarray, keep_hat: bool = False) -> Any:
        """P[m, s, z] -> f[j, q, z]; the exact adjoint of forward.

        keep_hat=True also returns rfft(P) along the angle (reused by the penalty).
        """
        P_hat = sfft.rfft(P, axis=0, workers=-1)
        conj = np.conjugate(P_hat, out=None if keep_hat else P_hat)
        F = batched_matmul(self.M.transpose(0, 2, 1), conj, conj_out=True)
        del conj
        D = sfft.irfft(F, n=self.n_theta, axis=0, workers=-1).astype(np.float32, copy=False)
        return (D, P_hat) if keep_hat else D


# ------------------------------------------------------- Cartesian <-> polar
def cartesian_to_polar_matrix(grid: PolarGrid, N: int, span: float, sub: int = 3) -> sp.csr_matrix:
    """Sparse map from a Cartesian field [x*N+y] to polar cells [j*Q+q].

    Each polar cell is the area-weighted average of sub x sub bilinear samples.
    """
    h = span / N
    rows, cols, vals = [], [], []
    Q = grid.Q
    for r in range(grid.n_rings):
        Lr = int(grid.L[r])
        K_r = grid.n_theta * Lr
        c = np.arange(K_r)
        cell = (c // Lr) * Q + grid.q_offset[r] + c % Lr
        for a in range(sub):
            rho = (r + (a + 0.5) / sub) * grid.dr
            for b in range(sub):
                phi = grid.orientation * (c + (b + 0.5) / sub) * 2 * np.pi / K_r
                u = (rho * np.cos(phi) + span / 2) / h - 0.5
                w_ = (rho * np.sin(phi) + span / 2) / h - 0.5
                i0, k0 = np.floor(u).astype(np.int64), np.floor(w_).astype(np.int64)
                fu, fw = u - i0, w_ - k0
                for di, dk, ww in ((0, 0, (1 - fu) * (1 - fw)), (1, 0, fu * (1 - fw)),
                                   (0, 1, (1 - fu) * fw), (1, 1, fu * fw)):
                    ii, kk = np.clip(i0 + di, 0, N - 1), np.clip(k0 + dk, 0, N - 1)
                    rows.append(cell)
                    cols.append(ii * N + kk)
                    vals.append(ww * rho)
    G = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(grid.n_cells, N * N))
    row_sum = np.asarray(G.sum(axis=1)).ravel()
    G = sp.diags(1 / np.maximum(row_sum, 1e-30)) @ G
    return G.astype(np.float32).tocsr()


def polar_to_cartesian_matrix(grid: PolarGrid, N: int, span: float) -> sp.csr_matrix:
    """Sparse bilinear (radius x angle) interpolation from polar cells to Cartesian
    pixel centres. Pixel centres outside the resin get zero."""
    c = (np.arange(N) + 0.5) * span / N - span / 2
    X, Y = np.meshgrid(c, c, indexing="ij")
    x, y = X.ravel(), Y.ravel()
    rho = np.hypot(x, y)
    phi = np.mod(grid.orientation * np.arctan2(y, x), 2 * np.pi)
    pix = np.flatnonzero(rho < grid.radius)
    u = np.clip(rho[pix] / grid.dr - 0.5, 0, grid.n_rings - 1)
    r0 = np.floor(u).astype(np.int64)
    r1 = np.minimum(r0 + 1, grid.n_rings - 1)
    fr = u - r0
    rows, cols, vals = [], [], []
    for ring, wr in ((r0, 1 - fr), (r1, fr)):
        Lr = grid.L[ring]
        K_r = grid.n_theta * Lr
        v = phi[pix] / (2 * np.pi) * K_r - 0.5
        c0 = np.floor(v).astype(np.int64)
        fa = v - c0
        for cc, wa in ((np.mod(c0, K_r), 1 - fa), (np.mod(c0 + 1, K_r), fa)):
            rows.append(pix)
            cols.append((cc // Lr) * grid.Q + grid.q_offset[ring] + cc % Lr)
            vals.append(wr * wa)
    return sp.csr_matrix((np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(N * N, grid.n_cells))


# --------------------------------------------------------------------- planner
class VialDosePlanner:
    """Dose density in Cartesian-pixel units: D = S A^T P, where S = h^2 / cell area."""

    def __init__(self, op: CirculantVialProjector, pitch: float, lipschitz: Optional[float] = None):
        self.op = op
        area = op.grid.cell_area()
        self.S = (pitch * pitch / area).astype(np.float32)
        self.w = (area / (pitch * pitch)).astype(np.float32)   # objective weight = area
        if lipschitz is not None:
            self.lipschitz = float(lipschitz)
            return
        rng = np.random.default_rng(0)
        x: np.ndarray = rng.random((op.n_theta, op.num_s, 1)).astype(np.float32)
        sw2 = (self.S * self.S * self.w)[None, :, None]
        lam = 1.0
        for _ in range(30):                                     # power iteration
            y = op.forward(op.adjoint(x) * sw2)
            lam = float(np.linalg.norm(y) / max(float(np.linalg.norm(x)), 1e-30))
            x = (y / max(float(np.linalg.norm(y)), 1e-30)).astype(np.float32)
        self.lipschitz = 1.01 * lam

    def dose(self, P: np.ndarray) -> np.ndarray:
        D = self.op.adjoint(P)
        D *= self.S[None, :, None]
        return D

    def dose_and_spectrum(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """dose(P) and rfft(P) along the angle, from one forward transform."""
        D, P_hat = self.op.adjoint(P, keep_hat=True)
        D *= self.S[None, :, None]
        return D, P_hat

    def least_squares(self, t: np.ndarray, lam_rel: float = 1e-3) -> np.ndarray:
        """Per-frequency exact min_P sum_q w_q (D_q - t_q)^2 + lam ||P||^2 (no positivity).

        Solved frequency by frequency on the fly: no second operator-sized array
        is kept in memory.
        """
        op = self.op
        d1 = (self.S * self.S * self.w).astype(np.float32)
        d2 = (self.S * self.w).astype(np.float32)
        lam = lam_rel * self.lipschitz
        T = sfft.rfft(t, axis=0, workers=-1)
        out = np.empty((op.K, op.num_s, t.shape[2]), np.complex64)

        def solve(a: int, b: int) -> None:
            for k in range(a, b):
                Mk = op.M[k]
                Nk = (Mk * d1[None, :]) @ Mk.conj().T
                Nk[np.diag_indices_from(Nk)] += lam
                out[k] = np.linalg.solve(Nk, Mk @ (d2[:, None] * T[k]))

        parallel_map(solve, op.K)
        return sfft.irfft(out, n=op.n_theta, axis=0, workers=-1).astype(np.float32, copy=False)


def band_pqn(planner: VialDosePlanner, t: np.ndarray, iterations: int = 30,
             d_low: float = 0.7, d_high: float = 1.0, over: float = 1.1,
             over_weight: float = 0.1, edge: tuple[float, float] = (0.05, 0.95),
             edge_weight: float = 1.0, mask_rays: bool = True,
             P0: Optional[np.ndarray] = None, history: int = 4, rtol: float = 0.0,
             smooth_theta: float = 0.0, smooth_s: float = 0.0,
             cell_weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Minimise the band objective over P >= 0 with a projected two-metric L-BFGS.

    t: polar target [j, q, z] in [0, 1]. Returns P[m, s, z].
    cell_weight: optional per-cell weight (same shape as t) multiplying both band terms;
    thin features get more weight (see FastVialEngine._thin_features).

    smooth_theta / smooth_s add (mu L / 8) ||dP||^2 along angle / detector, mu in units
    of the dose Lipschitz constant L (at mu = 1 the highest frequency is as stiff as
    the stiffest dose mode). Dose is blind to most of that roughness (null space), so
    the penalty picks the smooth solution among equally good ones: no frame flicker.
    """
    op = planner.op
    f32 = np.float32
    inside = t >= edge[1]
    outside = t <= edge[0]
    edge_cells = ~(inside | outside)
    ramp = (d_low + (d_high - d_low) * t).astype(f32)
    big = f32(1e30)
    lb = np.where(inside, f32(d_high), np.where(edge_cells, ramp, -big)).astype(f32)
    ub = np.where(inside, f32(over), np.where(edge_cells, ramp, f32(d_low))).astype(f32)
    # Weights pre-multiplied by the cell area w: f = ½ Σ tmp · (tmp · cw).
    w3 = planner.w[None, :, None]
    clw = (np.where(inside, f32(1), np.where(edge_cells, f32(edge_weight), f32(0))) * w3).astype(f32)
    cuw = (np.where(inside, f32(over_weight), np.where(edge_cells, f32(edge_weight), f32(1))) * w3).astype(f32)
    if cell_weight is not None:
        clw *= cell_weight.astype(f32, copy=False)
        cuw *= cell_weight.astype(f32, copy=False)
    del ramp, edge_cells
    S3 = planner.S[None, :, None]
    shape = (op.n_theta, op.num_s, t.shape[2])
    # Rays that never touch the part only add unwanted dose: keep them dark.
    hits = (op.forward((~outside).astype(f32)) > 1e-6) if mask_rays else np.ones(shape, bool)
    del outside
    if P0 is None:
        P0 = np.maximum(planner.least_squares(inside.astype(f32)), 0)
        D0 = planner.dose(P0)
        level = float(np.median(D0[inside])) if inside.any() else 1.0
        P0 *= f32(1.0 / max(level, 1e-12))
        del D0
    x: np.ndarray = (P0 * hits).astype(f32)
    tmp = np.empty((op.n_theta, op.grid.Q, t.shape[2]), f32)
    r = np.empty_like(tmp)
    lam_t = f32(smooth_theta * planner.lipschitz / 8)
    lam_s = f32(smooth_s * planner.lipschitz / 8)
    # Angular smoothness lam·Σ(x[t+1]−x[t])² is circulant, so diagonal in frequency:
    # eigenvalue c_k = 2 − 2cos(2πk/n). Its value comes from the spectrum the dose
    # already computes, its gradient 2·lam·c_k·X_k is added inside forward's inverse
    # transform. Same numbers as the direct form, without four passes over memory.
    n_t = op.n_theta
    k = np.arange(op.K)
    c_k = (2.0 - 2.0 * np.cos(2.0 * np.pi * k / n_t)).astype(f32)
    mult = np.full(op.K, 2.0)
    mult[0] = 1.0
    if n_t % 2 == 0:
        mult[-1] = 1.0
    pen_value_w = (float(lam_t) / n_t * mult * c_k).astype(np.float64)
    pen_grad_c = (f32(2) * lam_t * c_k).astype(np.complex64)

    def smooth_s_penalty(xv: np.ndarray, grad: np.ndarray) -> float:
        ds_ = xv[:, 1:] - xv[:, :-1]
        gs = f32(2) * lam_s * ds_
        grad[:, 1:] += gs
        grad[:, :-1] -= gs
        return float(lam_s * np.vdot(ds_, ds_))

    def fg(xv: np.ndarray) -> tuple[float, np.ndarray]:
        if lam_t:
            D, X = planner.dose_and_spectrum(xv)
        else:
            D, X = planner.dose(xv), None
        np.subtract(D, lb, out=tmp)
        np.minimum(tmp, 0, out=tmp)
        np.multiply(tmp, clw, out=r)
        f = 0.5 * float(np.dot(tmp.ravel(), r.ravel()))
        np.subtract(D, ub, out=tmp)
        del D
        np.maximum(tmp, 0, out=tmp)
        over_part = tmp * cuw
        f += 0.5 * float(np.dot(tmp.ravel(), over_part.ravel()))
        np.add(r, over_part, out=r)
        del over_part
        np.multiply(r, S3, out=r)
        if X is not None:
            power = np.empty(op.K)

            def spectrum(a: int, b: int) -> None:
                for kk in range(a, b):
                    v = X[kk].ravel()
                    power[kk] = float(np.vdot(v, v).real)

            parallel_map(spectrum, op.K)
            f += float(np.dot(pen_value_w, power))
            grad = op.forward(r, add_hat=(pen_grad_c, X))
            del X
        else:
            grad = op.forward(r)
        if lam_s:
            f += smooth_s_penalty(xv, grad)
        grad *= hits
        return f, grad

    def dot(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a.ravel(), b.ravel()))

    f, g = fg(x)
    S_hist: list[np.ndarray] = []
    Y_hist: list[np.ndarray] = []
    rho: list[float] = []
    stalls = 0
    free = np.empty(shape, bool)
    d = np.empty(shape, f32)
    for _ in range(iterations):
        np.greater(x, 0, out=free)
        free |= g < 0
        free &= hits
        np.negative(g, out=d)
        d *= free
        alphas = []
        for s_, y_, r_ in zip(reversed(S_hist), reversed(Y_hist), reversed(rho)):
            a_ = r_ * dot(s_, d)
            alphas.append(a_)
            d -= f32(a_) * y_
        gamma = dot(S_hist[-1], Y_hist[-1]) / dot(Y_hist[-1], Y_hist[-1]) if S_hist else 1.0 / planner.lipschitz
        d *= f32(gamma)
        for (s_, y_, r_), a_ in zip(zip(S_hist, Y_hist, rho), reversed(alphas)):
            d += f32(a_ - r_ * dot(y_, d)) * s_
        d *= free
        step = 1.0
        for _ in range(8):                       # projected Armijo backtracking
            x_new = d * f32(step)
            x_new += x
            np.maximum(x_new, 0, out=x_new)
            s_ = np.subtract(x_new, x)
            f_new, g_new = fg(x_new)
            if f_new <= f + 1e-4 * dot(g, s_):
                break
            step *= 0.3
        y_ = np.subtract(g_new, g)
        sy = dot(s_, y_)
        if sy > 1e-12 * dot(y_, y_):
            S_hist.append(s_)
            Y_hist.append(y_)
            rho.append(1.0 / sy)
            if len(S_hist) > history:
                S_hist.pop(0)
                Y_hist.pop(0)
                rho.pop(0)
        stalls = stalls + 1 if f - f_new <= rtol * f else 0
        x, f, g = x_new, f_new, g_new
        if rtol > 0 and stalls >= 2:       # converged: two iterations without real progress
            break
    return x


# ---------------------------------------------------------------- volume driver
def _uniform_angles(angles_rad: np.ndarray) -> tuple[float, int]:
    angles = np.asarray(angles_rad, dtype=np.float64)
    n = len(angles)
    if n < 2:
        raise ValueError("FastCAL needs at least two projection angles.")
    step = 2 * np.pi / n
    direction = 1 if np.angle(np.exp(1j * (angles[1] - angles[0]))) > 0 else -1
    expected = angles[0] + direction * step * np.arange(n)
    if np.max(np.abs(np.angle(np.exp(1j * (angles - expected))))) > 1e-6:
        raise ValueError("FastCAL needs equally spaced angles covering a full turn.")
    return float(angles[0]), direction


def supports(container_mode: int, angles_rad: np.ndarray) -> bool:
    """True when FastCAL can plan this job (round vial, uniform full-turn angles)."""
    if container_mode not in (0, 2):
        return False
    try:
        _uniform_angles(angles_rad)
    except ValueError:
        return False
    return True


def _cache_key(args: tuple, kwargs: dict) -> str:
    import hashlib
    import json
    from dataclasses import asdict
    payload = json.dumps({"v": 1, "args": [float(a) for a in args],
                          "kw": {k: (asdict(v) if k == "optics" else float(v)) for k, v in kwargs.items()}},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _cached_projector(args: tuple, kwargs: dict):
    """Build the projector, or load it from the operator cache (same printer = same operator)."""
    import os
    folder = _cache_dir()
    path = os.path.join(folder, _cache_key(args, kwargs) + ".npz") if folder else None
    if path and os.path.isfile(path):
        try:
            with np.load(path) as data:
                op = CirculantVialProjector.__new__(CirculantVialProjector)
                op._restore(args, kwargs, data)
                lip = float(data["lipschitz"]) if "lipschitz" in data.files else None
            os.utime(path)
            return op, lip, True
        except Exception:
            pass
    op = CirculantVialProjector(*args, **kwargs)
    if path and folder:
        try:
            os.makedirs(folder, exist_ok=True)
            tmp = path + ".tmp.npz"
            np.savez(tmp, M=op.M, coverage_radius=op.coverage_radius, ray_ok=op.ray_ok,
                     transmission=op.transmission)
            os.replace(tmp, path)
            files = sorted((os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npz")),
                           key=os.path.getmtime)
            for old in files[:-4]:            # keep the four most recent printers
                os.remove(old)
        except OSError:
            pass
    return op, None, False


def _store_lipschitz(args: tuple, kwargs: dict, value: float) -> None:
    import os
    folder = _cache_dir()
    if not folder:
        return
    path = os.path.join(folder, _cache_key(args, kwargs) + ".npz")
    try:
        with np.load(path) as data:
            payload = {name: data[name] for name in data.files}
        payload["lipschitz"] = np.float64(value)
        tmp = path + ".tmp.npz"
        np.savez(tmp, **payload)
        os.replace(tmp, path)
    except (OSError, ValueError, KeyError):
        pass


class _ScaledColumns:
    """Column access to a uint8 coverage volume as float32 in [0, 1]."""

    def __init__(self, data: np.ndarray):
        self.data = data

    def __getitem__(self, index):
        return self.data[index].astype(np.float32) * np.float32(1 / 255)


def _cache_dir() -> Optional[str]:
    """Where operators are kept between runs; SPINSLICER_CACHE="" disables it."""
    import os
    value = os.environ.get("SPINSLICER_CACHE")
    if value is not None:
        return value or None
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "spinslicer", "fastcal")


def _distance_tools(size: int) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Euclidean distance transform and a size x size local maximum (OpenCV when available:
    about 10x faster than scipy on a 384 grid, which matters for hundreds of layers)."""
    try:
        import cv2
        kernel = np.ones((size, size), np.uint8)

        def edt(mask: np.ndarray) -> np.ndarray:
            return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        def local_max(a: np.ndarray) -> np.ndarray:
            return cv2.dilate(np.ascontiguousarray(a, dtype=np.float32), kernel)
        return edt, local_max
    except ImportError:  # pragma: no cover - cv2 is a dependency
        from scipy.ndimage import distance_transform_edt, maximum_filter
        return (lambda m: distance_transform_edt(m).astype(np.float32),
                lambda a: maximum_filter(a, size=size))


class FastVialEngine:
    """Plans projections for a whole voxel volume (see the module docstring)."""

    def __init__(self, angles_rad: np.ndarray, num_s: int, grid_res: int, grid_span_mm: float,
                 L_out: float, L_in: float, n_air: float = 1.0, n_glass: float = 1.51,
                 n_liquid: float = 1.45, n_resin: float = 1.50, alpha: float = 0.0,
                 container_mode: int = 0, memory_budget_mb: Optional[int] = None,
                 angular_oversample: float = 0.7, optics: Optional[OpticsOptions] = None,
                 multigrid: bool = True):
        angle0, direction = _uniform_angles(angles_rad)
        n = len(angles_rad)
        self.N, self.span = int(grid_res), float(grid_span_mm)
        self.pitch = self.span / self.N
        optics = OpticsOptions() if optics is None else optics
        args = (n, num_s, L_out, L_in, n_air, n_glass, n_liquid, n_resin, alpha, container_mode)
        kwargs = dict(pitch=self.pitch, angle0=angle0, direction=direction,
                      angular_oversample=angular_oversample, optics=optics)
        self.op, lipschitz, self.cache_hit = _cached_projector(args, kwargs)
        self.alpha = float(alpha)
        self.L_in, self.L_out = float(L_in), float(L_out)
        self.n_out = float(n_liquid if container_mode == 2 else n_air)
        self.n_resin = float(n_resin)
        self.planner = VialDosePlanner(self.op, self.pitch, lipschitz)
        if lipschitz is None:
            _store_lipschitz(args, kwargs, self.planner.lipschitz)
        self.G = cartesian_to_polar_matrix(self.op.grid, self.N, self.span)
        # A layer is re-planned from scratch when its target differs by more than this
        # from the best neighbour guess. 0.15 keeps sloped, organic parts (hulls, figures)
        # on cheap warm starts: same quality, a quarter less time than 0.03 (2026-10-30).
        self.change_tol = 0.15
        self.stats: list[tuple[int, ...]] = []
        if memory_budget_mb is None:
            memory_budget_mb = default_memory_budget_mb()
        per_layer = 4 * (10 * n * self.op.grid.Q + 16 * n * num_s)
        # ~24 layers per batch: the per-frequency products run at full speed from
        # ~20 layers on, while wider batches share one L-BFGS step between too many
        # layers and converge worse (Benchy 256: 20 → 25.8 s, 64 → 27.7 s, lower IoU).
        self.batch = int(max(4, min(BATCH_LAYERS, memory_budget_mb * 2**20 // per_layer)))
        # Batches of one level are independent: solving several at once spreads the
        # element-wise work (bounds, L-BFGS updates) over the cores too, not only the
        # matrix products. Limited by cores and by the memory budget.
        fit = int(memory_budget_mb * 2**20 // max(per_layer * self.batch, 1))
        self.parallel_batches = int(max(1, min(4, engine_threads(), fit)))
        self.refine_iters = 8
        self.coarse: Optional[FastVialEngine] = None
        if multigrid and self.N >= 192 and self.N % 2 == 0 and num_s % 2 == 0:
            self.coarse = FastVialEngine(angles_rad, num_s // 2, self.N // 2, grid_span_mm, L_out, L_in,
                                         n_air, n_glass, n_liquid, n_resin, alpha, container_mode,
                                         memory_budget_mb, angular_oversample, optics, multigrid=False)

    def _coarse_start(self, target: np.ndarray, chunk: list[int], its: int, tp: np.ndarray,
                      band: dict) -> np.ndarray:
        coarse = self.coarse
        assert coarse is not None
        block = np.asarray(target[:, :, chunk], dtype=np.float32)
        if target.dtype == np.uint8:
            block *= np.float32(1 / 255)
        n = coarse.N
        small = block.reshape(n, 2, n, 2, len(chunk)).mean(axis=(1, 3))      # area average
        Pc = band_pqn(coarse.planner, coarse._polar(small, list(range(len(chunk)))),
                      iterations=its, **band)
        # Detector bins: linear interpolation between coarse bin centres.
        ns_c, ns_f = coarse.op.num_s, self.op.num_s
        u = np.clip((np.arange(ns_f) + 0.5) * ns_c / ns_f - 0.5, 0, ns_c - 1)
        lo = np.floor(u).astype(np.int64)
        hi = np.minimum(lo + 1, ns_c - 1)
        fr = (u - lo).astype(np.float32)[None, :, None]
        P0 = Pc[:, lo, :] * (1 - fr) + Pc[:, hi, :] * fr
        # Same scale convention as a fresh start: median in-part dose = 1.
        D = self.planner.dose(P0)
        inside = tp >= 0.95
        level = float(np.median(D[inside])) if inside.any() else 1.0
        P0 *= np.float32(1.0 / max(level, 1e-12))
        return P0

    def _polar(self, target: np.ndarray, layers: list[int]) -> np.ndarray:
        flat = target[:, :, layers].reshape(self.N * self.N, len(layers))
        flat = np.ascontiguousarray(flat, dtype=np.float32)
        if target.dtype == np.uint8:          # coverage stored as 0..255
            flat *= np.float32(1 / 255)
        tp = np.asarray(self.G @ flat)
        return tp.reshape(self.op.n_theta, self.op.grid.Q, len(layers))

    def _thin_features(self, target: np.ndarray, layers: list[int], boost: float, radius: float,
                       snap_void: bool, snap_solid: bool) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Polar target and per-cell weights that know how thick each feature is.

        Coverage alone cannot tell a voxel on the edge of a big body from one inside a
        hole narrower than two voxels: both are "partly solid", so the band puts them at an
        intermediate dose, and in a thin hole no cell is dark enough to anchor it — light
        from the walls closes it. Per layer, the distance transform gives every cell the
        half-width of the feature it belongs to (solid or void, local maximum of the
        distance to the other phase within ``radius`` voxels). Thin features then get
        ``1 + boost`` times the weight (linearly less up to ``radius``), and optionally a
        definite target: thin voids fully dark, thin solids fully cured — for them the
        topology matters more than a sub-voxel edge position.
        """
        block = np.asarray(target[:, :, layers], dtype=np.float32)
        if target.dtype == np.uint8:
            block *= np.float32(1 / 255)
        weights = np.ones_like(block)
        size = int(2 * np.ceil(radius) + 1)
        edt, local_max = _distance_tools(size)
        for i in range(block.shape[2]):
            t = block[:, :, i]
            solid = t >= 0.5
            if not solid.any() or solid.all():
                continue
            d_solid = edt(solid)                    # inside solid: distance to void
            d_void = edt(~solid)                    # inside void: distance to solid
            half = np.where(solid, local_max(d_solid), local_max(d_void))
            thin = np.clip((radius - half) / radius, 0.0, 1.0)
            # Only cells that belong to (or border) a thin feature: d <= radius from the boundary.
            near = np.where(solid, d_solid, d_void) <= radius
            thin = np.where(near, thin, 0.0)
            weights[:, :, i] = 1.0 + boost * thin
            if snap_void:
                t = np.where((~solid) & (thin > 0.5), 0.0, t)
            if snap_solid:
                t = np.where(solid & (thin > 0.5), 1.0, t)
            block[:, :, i] = t
        flat = block.reshape(self.N * self.N, len(layers))
        tp = np.asarray(self.G @ flat).reshape(self.op.n_theta, self.op.grid.Q, len(layers))
        wp = np.asarray(self.G @ weights.reshape(self.N * self.N, len(layers)))
        return tp, wp.reshape(tp.shape).astype(np.float32)

    def optimize_volume(self, target: np.ndarray, schedule=DEFAULT_SCHEDULE,
                        report: Optional[Callable[[float, str], None]] = None,
                        cancelled: Optional[Callable[[], None]] = None,
                        interp_tol: float = 0.02, warm_mask: bool = True,
                        **band) -> np.ndarray:
        """target [x, y, z] in [0, 1] -> projections [z, detector, angle] (max 1).

        `cancelled` is called between batches and may raise to abort.
        Frames are smoothed along the angle (smooth_theta, see band_pqn) unless the
        caller overrides it: no flicker, and on the 512 judge equal or better quality.
        """
        band.setdefault("smooth_theta", 0.03)
        thin_cfg = {k: band.pop(k) for k in ("thin_boost", "thin_radius", "snap_void", "snap_solid") if k in band}
        thin_boost = float(thin_cfg.get("thin_boost", THIN_BOOST))
        thin_radius = float(thin_cfg.get("thin_radius", THIN_RADIUS))
        snap_void = bool(thin_cfg.get("snap_void", SNAP_THIN_VOID))
        snap_solid = bool(thin_cfg.get("snap_solid", SNAP_THIN_SOLID))
        N, NZ = self.N, target.shape[2]
        op = self.op
        # float16 is ample for 8-bit frames and halves the largest array.
        out = np.zeros((NZ, op.n_theta, op.num_s), np.float16)     # [z, angle, s]
        done = np.zeros(NZ, bool)
        levels, seen = [], np.zeros(NZ, bool)
        for stride, iters in schedule:
            layers = [z for z in range(0, NZ, stride) if not seen[z]]
            seen[layers] = True
            levels.append((stride, iters, layers))
        full_iters = schedule[0][1]
        total = sum(len(lay) * (full_iters + 2) for _, _, lay in levels[:1]) + \
            sum(len(lay) * (it + 2) for _, it, lay in levels[1:])
        work = 0
        flat_target: Any = target.reshape(N * N, NZ)
        if target.dtype == np.uint8:
            flat_target = _ScaledColumns(flat_target)

        def neighbours(z: int) -> list[tuple[int, float]]:
            solved = np.flatnonzero(done)
            k = int(np.searchsorted(solved, z))
            lo = int(solved[k - 1]) if k > 0 else None
            hi = int(solved[k]) if k < len(solved) else None
            if lo is None:
                return [(hi, 1.0)]  # type: ignore[list-item]
            if hi is None:
                return [(lo, 1.0)]
            a = (z - lo) / (hi - lo)
            return [(lo, 1 - a), (hi, a)]

        import threading
        lock = threading.Lock()
        solved = 0
        self.stats = []
        # Per layer: 0 cold start, 1 warm start, 2 interpolated copy (diagnostics).
        self.layer_kind = np.zeros(NZ, np.int8)
        for level, (stride, iters, layers) in enumerate(levels):
            hard: list[int] = []
            easy: list[int] = []
            copy: list[int] = []
            start: dict[int, list[tuple[int, float]]] = {}
            if level == 0:
                hard = list(layers)
            else:
                for z in layers:
                    # Best start among: interpolation of both neighbours, or either one alone
                    # (a step in the geometry is usually identical to one side).
                    col = flat_target[:, z]
                    options = [neighbours(z)]
                    if len(options[0]) == 2:
                        options += [[(options[0][0][0], 1.0)], [(options[0][1][0], 1.0)]]
                    scored = []
                    for opt in options:
                        # Every neighbour of the mix must itself be close to this layer. Doses
                        # add linearly but the band objective is not linear in the target: the
                        # 50/50 mix of a full and an empty layer matches a half-covered layer
                        # (a top or bottom face inside the layer) perfectly as a target, yet its
                        # frames give half the dose and the layer would not cure.
                        diff = sum(w * np.abs(col - flat_target[:, zz]) for zz, w in opt)
                        scored.append((float(diff.sum()) / max(float(col.sum()), 1.0), opt))
                    change, best = min(scored, key=lambda item: item[0])
                    start[z] = best
                    if change > self.change_tol:
                        hard.append(z)
                    elif change <= interp_tol and stride == 1:
                        copy.append(z)      # dose is linear: interpolated frames give the interpolated dose
                    else:
                        easy.append(z)
            for z in copy:
                acc = np.zeros((op.n_theta, op.num_s), np.float32)
                for zz, w in start[z]:
                    acc += np.float32(w) * out[zz].astype(np.float32)
                out[z] = acc
            done[copy] = True
            solved += len(copy)
            self.layer_kind[copy] = 2
            self.layer_kind[easy] = 1
            for group, group_iters, warm in ((hard, full_iters, False), (easy, iters, True)):

                def solve(chunk: list[int], group_iters: int = group_iters, warm: bool = warm) -> None:
                    nonlocal work, solved
                    if cancelled is not None:
                        cancelled()
                    its = group_iters
                    weight = None
                    if thin_boost > 0 or snap_void or snap_solid:
                        tp, weight = self._thin_features(target, chunk, thin_boost, thin_radius,
                                                         snap_void, snap_solid)
                    else:
                        tp = self._polar(target, chunk)
                    P0 = None
                    if warm:
                        P0 = np.zeros((op.n_theta, op.num_s, len(chunk)), np.float32)
                        for i, z in enumerate(chunk):
                            for zz, w in start[z]:
                                P0[:, :, i] += np.float32(w) * out[zz].astype(np.float32)
                    elif self.coarse is not None:
                        # Two-grid start: solve at half resolution (1/4 of the work),
                        # then refine at full resolution from that solution.
                        P0 = self._coarse_start(target, chunk, its, tp, band)
                        its = self.refine_iters
                    P = band_pqn(self.planner, tp, iterations=its, P0=P0,
                                 mask_rays=warm_mask or not warm, cell_weight=weight, **band)
                    out[chunk] = P.transpose(2, 0, 1)
                    del P, P0, tp
                    with lock:
                        work += len(chunk) * (its + 2)
                        solved += len(chunk)
                        if report is not None:
                            report(min(1.0, work / max(total, 1)), f"FastCAL: {solved}/{NZ}")

                batches = [group[b0:b0 + self.batch] for b0 in range(0, len(group), self.batch)]
                run_batches(solve, batches, self.parallel_batches)
                done[group] = True
            self.stats.append((stride, len(hard), len(easy), len(copy)))
        peak = float(out.max())
        if peak > 0:
            out *= np.float16(1.0 / peak)
        return out.transpose(0, 2, 1)

    def layer_dose(self, sinogram: np.ndarray) -> np.ndarray:
        """Relative dose of one layer on the Cartesian grid, from P[detector, angle].

        Uses the same operator that planned the frames, interpolated to pixel
        centres; no full sparse system matrix is built for the preview.
        """
        P = np.ascontiguousarray(np.asarray(sinogram, np.float32).T)[:, :, None]
        D = self.planner.dose(P)[:, :, 0].ravel()
        H = polar_to_cartesian_matrix(self.op.grid, self.N, self.span)
        return np.asarray(H @ D).reshape(self.N, self.N)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if not values.size:
        return float("nan")
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order], dtype=np.float64)
    return float(values[order][min(int(np.searchsorted(cdf, q * cdf[-1])), len(order) - 1)])


def dose_statistics(engine: "FastVialEngine", target: np.ndarray, intensity,
                    max_samples: int = 2_000_000, max_layers: int = 96) -> dict:
    """Delivered relative dose of the whole volume, from projector intensities.

    intensity: [z, detector, angle] relative irradiance actually displayed
    (after gain, cut-off, quantisation and gamma round trip), 1 = full white;
    or a callable layers -> that array for those layers (saves memory).
    Returns robust in-part / out-of-part levels in Cartesian-pixel dose units
    (see VialDosePlanner), plus the best global threshold.
    """
    op, pl = engine.op, engine.planner
    NZ = target.shape[2]
    sample_layers = np.unique(np.linspace(0, NZ - 1, min(NZ, max_layers)).round().astype(int))
    rng = np.random.default_rng(0)
    din: list[np.ndarray] = []
    win: list[np.ndarray] = []
    dout: list[np.ndarray] = []
    wout: list[np.ndarray] = []
    keep = min(1.0, max_samples / max(1, len(sample_layers) * op.grid.n_cells))
    area = pl.w[None, :, None]
    for z0 in range(0, len(sample_layers), engine.batch):
        layers = [int(z) for z in sample_layers[z0:z0 + engine.batch]]
        tp = engine._polar(target, layers)
        block = intensity(layers) if callable(intensity) else intensity[layers]
        P = np.ascontiguousarray(np.asarray(block, np.float32).transpose(2, 1, 0))
        D = pl.dose(P)
        W = np.broadcast_to(area, D.shape)
        for mask, dv, wv in ((tp >= 0.95, din, win), (tp <= 0.05, dout, wout)):
            sel = mask & (rng.random(D.shape) < keep) if keep < 1 else mask
            dv.append(D[sel])
            wv.append(W[sel])
    d_in, w_in = np.concatenate(din), np.concatenate(win)
    d_out, w_out = np.concatenate(dout), np.concatenate(wout)
    lo = _weighted_quantile(d_in, w_in, 0.005)
    hi = _weighted_quantile(d_out, w_out, 0.995) if d_out.size else 0.0
    if lo > hi:
        threshold = 0.5 * (lo + hi)
    else:   # no clean window: threshold with the least misclassified area
        cand = np.quantile(np.concatenate([d_in, d_out]), np.linspace(0.01, 0.99, 197))
        err = [w_in[d_in < t].sum() + w_out[d_out >= t].sum() for t in cand]
        threshold = float(cand[int(np.argmin(err))])
    wrong = float(w_in[d_in < threshold].sum() + w_out[d_out >= threshold].sum())
    return {
        "in_part_low": lo, "out_of_part_high": hi, "threshold": threshold,
        "window": (lo - hi) / (lo + hi) if lo + hi > 0 else 0.0,
        "misprinted_fraction": wrong / max(float(w_in.sum() + w_out.sum()), 1e-30),
        "in_part_median": _weighted_quantile(d_in, w_in, 0.5),
        "background_per_unit_black": float(np.median(pl.dose(np.ones((op.n_theta, op.num_s, 1), np.float32)))),
    }


def optimize_projections_fast(
    target_vol: np.ndarray, angles_rad: np.ndarray, L_out: float, L_in: float,
    n_air: float = 1.0, n_glass: float = 1.51, n_liquid: float = 1.45,
    n_resin: float = 1.50, alpha: float = 0.0, grid_res: int = 128,
    grid_span_mm: float = 50.0, iterations: int = 30, container_mode: int = 0,
    report_cb: Optional[Callable[[float, str], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    num_s: Optional[int] = None, engine_out: Optional[list] = None,
    optics: Optional[OpticsOptions] = None,
) -> np.ndarray:
    """Drop-in replacement for optimize_projections_sirt_volume (round vials).

    `iterations` is used for layers solved from scratch. Warm-started layers
    use the shorter tail of DEFAULT_SCHEDULE.
    """
    target = np.asarray(target_vol)
    if target.ndim != 3 or target.shape[:2] != (grid_res, grid_res) or not target.shape[2]:
        raise ValueError("Target must have shape (grid_res, grid_res, nonzero layers).")
    if target.dtype != np.uint8 and (
        not np.isfinite(target).all() or np.any(target < 0) or np.any(target > 1)
    ):
        raise ValueError("Relative target dose must be finite and in [0, 1] (or uint8 coverage).")
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 1:
        raise ValueError("iterations must be a positive integer.")

    def check() -> None:
        if cancelled_cb and cancelled_cb():
            from slicing_engine import SliceCancelled
            raise SliceCancelled()

    check()
    if report_cb:
        report_cb(0.31, "FastCAL: трассировка лучей одного ракурса...")
    engine = FastVialEngine(angles_rad, grid_res if num_s is None else num_s, grid_res,
                            grid_span_mm, L_out, L_in, n_air, n_glass, n_liquid, n_resin,
                            alpha, container_mode, optics=optics)
    if engine_out is not None:
        engine_out.append(engine)
    schedule = ((DEFAULT_SCHEDULE[0][0], int(iterations)),) + DEFAULT_SCHEDULE[1:]

    def report(fraction: float, message: str) -> None:
        if report_cb:
            report_cb(0.35 + 0.43 * fraction, message)

    return engine.optimize_volume(target, schedule=schedule, report=report, cancelled=check)
