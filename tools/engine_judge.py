"""Engine judge: plan the same job with several CAL engines and judge them all the same way.

  python tools/engine_judge.py --cases nut,benchy,restarget --engines fastcal,fastcal_bin,osmo,pm,bclp,fbp
  python tools/engine_judge.py --cases nut --grid 128 --mc 8 --out results.json

Fairness rules (each one closes a bias found in the older tools/compare_osmo.py):

* Same physics for everyone: refraction through the vial wall and the bath (or air),
  Beer–Lambert attenuation, 360 equally spaced angles. The Siddon-based engines (OSMO,
  PM, BCLP, FBP) plan on SpinSlicer's exact sparse operator; FastCAL on its own
  rotation-symmetric operator. FastCAL is run with the physics the judge models
  (linear pixel profile; no Fresnel, no rotation smear), nothing more.
* Same information: OSMO/PM/BCLP/FBP need a binary target, so FastCAL is judged both
  with the sub-voxel coverage the app uses ("fastcal") and with the same binary target
  ("fastcal_bin").
* Competitors at their best: VAMToolbox update rules verbatim, plus sweeps of their
  settings (``--tune``) with the best result kept — chosen on the judge itself, which
  favours them, not us.
* Independent judge: a 2x finer voxel grid voxelised separately from the mesh, the exact
  Siddon operator at that grid (ray positions and voxels the planners never saw), frames
  quantised to 8 bits per engine and upsampled linearly to the judge's detector pitch,
  as a projector shows them.

Metrics per engine and case:
  iou            intersection over union at the best single exposure;
  surf_mean_mm   mean distance between the cured and the true surface at that exposure;
  surf_p95_mm    95th percentile of the same;
  latitude_pct   exposure error (± %) that keeps the IoU within 0.01 of the engine's best;
  latitude95_pct exposure error (± %) that keeps the IoU at or above 0.95 (a good part);
  window_pct     dose window between the dimmest 0.5 % part voxels and the brightest 0.5 %
                 resin voxels, away from partial edge voxels (> 0 means a clean threshold exists);
  robust_mean/robust_worst  IoU over --mc virtual prints of a real machine at the planned
                 exposure: exposure ±5 %, projector blur 0.5 row and 0.5 px, axis offset
                 ±0.1 mm, frame timing jitter ±1°;
  features       (resolution target only) smallest pin, hole and wall that print;
  seconds, peak_mb  planning time and peak traced memory of the engine.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optics.fastcal import FastVialEngine, OpticsOptions  # noqa: E402
from optics.optimization import _load_cpp_backend, _project, build_system_matrix  # noqa: E402
from voxelizer import voxelize  # noqa: E402

SPAN, L_OUT, L_IN, N_GLASS, N_LIQUID, N_RESIN = 60.0, 60.0, 56.0, 1.51, 1.45, 1.50
FRAMES = 360
JUDGE_OPTICS = OpticsOptions(fresnel=False, pixel_profile="linear", rotation_smear=False)


# ----------------------------------------------------------------------------- cases
@dataclass
class Case:
    name: str
    mesh: object
    mode: int = 2            # 2 = immersion bath, 0 = vial in air
    alpha: float = 0.0       # resin attenuation, 1/mm
    features: list = field(default_factory=list)   # (kind, size_mm, x, y) for the resolution target


def _nut():
    from threaded_nut import ThreadedNutParameters, create_threaded_nut
    return create_threaded_nut(ThreadedNutParameters(outer_diameter_mm=36, bore_diameter_mm=16, height_mm=24,
                                                     pitch_mm=3, thread_depth_mm=1.2))


def _benchy():
    import trimesh
    mesh = trimesh.load(str(ROOT.parent / "models" / "benchy.stl"), force="mesh")
    mesh.apply_scale(40.0 / float(max(mesh.extents[:2])))
    return mesh


def _restarget():
    """Pins, holes and walls from 0.4 to 3 mm. Holes are cylinders with inverted faces
    inside a plate: the non-zero winding voxeliser carves them out, no booleans needed."""
    import trimesh
    parts, feats = [], []
    sizes = [0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
    h = 10.0
    x = -21.0
    for d in sizes:                                   # pins, upper row
        cyl = trimesh.creation.cylinder(radius=d / 2, height=h, sections=48)
        cyl.apply_translation((x, 9.0, 0))
        parts.append(cyl)
        feats.append(("pin", d, x, 9.0))
        x += max(d, 1.0) + 4.5
    plate = trimesh.creation.box(extents=(44.0, 9.0, h))
    plate.apply_translation((0, -3.0, 0))
    parts.append(plate)
    x = -18.5
    for d in sizes:                                   # holes in the plate, middle row
        cyl = trimesh.creation.cylinder(radius=d / 2, height=h + 2, sections=48)
        cyl.apply_translation((x, -3.0, 0))
        cyl.invert()
        parts.append(cyl)
        feats.append(("hole", d, x, -3.0))
        x += max(d, 1.0) + 4.0
    x = -16.0
    for t in [0.4, 0.6, 0.8, 1.0, 1.5]:               # walls, lower row
        wall = trimesh.creation.box(extents=(t, 7.0, h))
        wall.apply_translation((x, -15.0, 0))
        parts.append(wall)
        feats.append(("wall", t, x, -15.0))
        x += 7.5
    return trimesh.util.concatenate(parts), feats


def make_case(name: str) -> Case:
    if name == "nut":
        return Case("nut", _nut())
    if name == "nut_air":
        return Case("nut_air", _nut(), mode=0)
    if name == "nut_absorbing":
        return Case("nut_absorbing", _nut(), alpha=0.03)
    if name == "benchy":
        return Case("benchy", _benchy())
    if name == "restarget":
        mesh, feats = _restarget()
        return Case("restarget", mesh, features=feats)
    raise SystemExit(f"unknown case {name}")


# ----------------------------------------------------------------------------- operators
def angles_rad() -> np.ndarray:
    return np.deg2rad((90.0 + np.linspace(0, 360, FRAMES, endpoint=False)) % 360)


class Siddon:
    """Exact sparse operator at grid n: fwd(x[N*N, Z]) -> sino[FRAMES*n, Z], bwd the adjoint."""

    def __init__(self, n: int, mode: int, alpha: float, num_s: Optional[int] = None):
        self.n = n
        self.ns = int(num_s or n)
        self.A, self.AT = build_system_matrix(angles_rad(), self.ns, n, SPAN, L_OUT, L_IN, 1.0, N_GLASS, N_LIQUID,
                                              N_RESIN, alpha, mode, memory_budget_bytes=12 * 2**30)
        self.lib = _load_cpp_backend()

    def fwd(self, x: np.ndarray) -> np.ndarray:
        return _project(self.A, np.ascontiguousarray(x, dtype=np.float32), self.lib)

    def bwd(self, b: np.ndarray) -> np.ndarray:
        return _project(self.AT, np.ascontiguousarray(b, dtype=np.float32), self.lib)


def resin_mask(n: int, margin_mm: float = 0.0) -> np.ndarray:
    c = (np.arange(n) + 0.5) * SPAN / n - SPAN / 2
    X, Y = np.meshgrid(c, c, indexing="ij")
    return (X**2 + Y**2 < (L_IN / 2 - margin_mm) ** 2)


# ----------------------------------------------------------------------------- engines
# Every engine returns frames [z, s, angle] (any positive scale).

def run_fastcal(target: np.ndarray, case: Case, n: int, binary: bool, **kw) -> np.ndarray:
    t = (target >= 0.5).astype(np.float32) if binary else target
    slope = kw.pop("edge_slope", None)
    if slope is not None:
        t = np.clip(0.5 + float(slope) * (t - 0.5), 0.0, 1.0).astype(np.float32)
    ctor = {k: kw.pop(k) for k in ("num_s", "angular_oversample", "multigrid") if k in kw}
    num_s = int(ctor.pop("num_s", n))
    eng = FastVialEngine(angles_rad(), num_s, n, SPAN, L_OUT, L_IN, 1.0, N_GLASS, N_LIQUID, N_RESIN, case.alpha,
                         case.mode, optics=JUDGE_OPTICS, **ctor)
    for key in ("change_tol", "refine_iters"):
        if key in kw:
            setattr(eng, key, kw.pop(key))
    if "schedule" in kw:
        kw["schedule"] = tuple(tuple(x) for x in kw["schedule"])
    return np.asarray(eng.optimize_volume(t, **kw), np.float32)


def _ramlak_target(T: np.ndarray) -> np.ndarray:
    import scipy.fft as sfft
    N = T.shape[0]
    fy, fx = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), indexing="ij")
    filt = np.sqrt(fx**2 + fy**2)
    out = np.empty_like(T)
    for z in range(T.shape[2]):
        out[:, :, z] = np.real(sfft.ifft2(sfft.ifftshift(sfft.fftshift(sfft.fft2(T[:, :, z])) * filt)))
    return out


def run_osmo(target: np.ndarray, op: Siddon, n_iter: int = 50, d_h: float = 0.8, d_l: float = 0.7) -> np.ndarray:
    """VAMToolbox minimizeOSMO, verbatim update rule."""
    N, _, NZ = target.shape
    circle = resin_mask(N)[:, :, None]
    T = (target >= 0.5).astype(np.float32)
    gel, void = (T > 0) & circle, (T == 0) & circle
    xm = _ramlak_target(T)
    flat = lambda a: a.reshape(N * N, NZ)  # noqa: E731
    b = np.clip(op.fwd(flat(xm)), 0, None)
    x = op.bwd(b).reshape(N, N, NZ)
    x /= x.max()
    for _ in range(1, n_iter):
        xm[void] -= np.maximum(x[void] - d_l, 0)
        b = op.fwd(flat(xm)); b /= b.max(); b[b < 0] = 0
        x = op.bwd(b).reshape(N, N, NZ); x /= x.max(); x[x < 0] = 0
        xm[gel] += np.maximum(d_h - x[gel], 0)
        b = op.fwd(flat(xm)); b /= b.max(); b[b < 0] = 0
        x = op.bwd(b).reshape(N, N, NZ); x /= x.max(); x[x < 0] = 0
    return b.reshape(FRAMES, op.ns, NZ).transpose(2, 1, 0)


def _ramlak_sino(op: Siddon, T: np.ndarray) -> np.ndarray:
    N, _, NZ = T.shape
    g = op.fwd(T.reshape(N * N, NZ)).reshape(FRAMES, op.ns, NZ)
    ramp = np.abs(np.fft.fftfreq(op.ns))[None, :, None]
    return np.maximum(np.real(np.fft.ifft(np.fft.fft(g, axis=1) * ramp, axis=1)), 0).astype(np.float32)


def run_fbp(target: np.ndarray, op: Siddon) -> np.ndarray:
    """Classic CAL start: ramp-filtered projections of the target, negatives clipped."""
    T = (target >= 0.5).astype(np.float32)
    return _ramlak_sino(op, T).transpose(2, 1, 0)


def run_bclp(target: np.ndarray, op: Siddon, n_iter: int = 200, lr: Optional[float] = None) -> np.ndarray:
    """VAMToolbox minimizeBCLP defaults (B 25, eps 0.1, p 2, q 1, projected gradient descent)."""
    N, _, NZ = target.shape
    lr = 0.01 * (N / 256) ** 2 if lr is None else lr
    w = resin_mask(N)[:, :, None].astype(np.float32)
    T = (target >= 0.5).astype(np.float32)
    eps, B, M = 0.1, 25.0, 0.5
    g = _ramlak_sino(op, T).reshape(FRAMES * op.ns, NZ)
    g *= np.float32(1.0 / np.median(op.bwd(g).reshape(N, N, NZ)[T > 0]))
    for _ in range(n_iter):
        m = 1.0 / (1.0 + np.exp(-B * (op.bwd(g).reshape(N, N, NZ) - M)))
        err = m - T
        band = np.abs(err) - eps
        v = band > 0
        loss = float(np.sqrt(np.sum(v * w * band**2)))
        if loss == 0:
            break
        operand = (v * w * band * np.sign(err) * B * m * (1 - m)).astype(np.float32)
        g = np.maximum(g - np.float32(lr / loss) * op.fwd(operand.reshape(N * N, NZ)), 0)
    return g.reshape(FRAMES, op.ns, NZ).transpose(2, 1, 0)


def run_pm(target: np.ndarray, op: Siddon, n_iter: int = 50, d_h: float = 0.8, d_l: float = 0.7,
           p: int = 1) -> np.ndarray:
    """VAMToolbox minimizePM: Lp penalty outside [d_l, d_h] bands, L-BFGS-B with 0 <= b <= 1.

    Dose is scaled like VAMToolbox's projector (angle integral in radians per unit length),
    so its default d_h / d_l mean the same thing here.
    """
    from scipy import optimize
    N, _, NZ = target.shape
    circle = resin_mask(N)[:, :, None]
    T = (target >= 0.5)
    core, background = (T & circle).reshape(N * N, NZ), (~T & circle).reshape(N * N, NZ)
    scale = np.float32(np.pi / 180.0 / (SPAN / N))
    b0 = _ramlak_sino(op, T.astype(np.float32)).reshape(FRAMES * op.ns, NZ)
    b0 /= b0.max()
    shape = b0.shape
    state: dict = {}

    def dose(b: np.ndarray) -> np.ndarray:
        return op.bwd(b.reshape(shape).astype(np.float32)) * scale

    def fun(bflat: np.ndarray) -> tuple[float, np.ndarray]:
        d = dose(bflat)
        over = background & (d > d_l)
        under = core & (d < d_h)
        lo, hi = np.where(over, d - d_l, 0), np.where(under, d_h - d, 0)
        loss = float(np.sum(lo**p) + np.sum(hi**p))
        gd = p * (lo ** (p - 1) * over - hi ** (p - 1) * under)
        grad = op.fwd(gd.astype(np.float32)) * scale
        state["n"] = state.get("n", 0) + 1
        return loss, grad.astype(np.float64).ravel()

    res = optimize.minimize(fun, b0.astype(np.float64).ravel(), jac=True, method="L-BFGS-B",
                            bounds=optimize.Bounds(np.zeros(b0.size), np.ones(b0.size)),
                            options={"maxiter": n_iter, "maxcor": 10, "maxls": 30, "ftol": 1e-12})
    return res.x.reshape(FRAMES, op.ns, NZ).astype(np.float32).transpose(2, 1, 0)


# ----------------------------------------------------------------------------- judge
def displayed(frames: np.ndarray, n_fine: int) -> np.ndarray:
    """8-bit frames upsampled linearly to the judge's detector pitch: [z, s_fine, angle]."""
    f = np.rint(np.clip(frames / max(float(frames.max()), 1e-30), 0, 1) * 255).astype(np.float32) / 255
    n = f.shape[1]
    u = np.clip((np.arange(n_fine) + 0.5) * n / n_fine - 0.5, 0, n - 1)
    lo = np.floor(u).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (u - lo).astype(np.float32)[None, :, None]
    return f[:, lo, :] * (1 - w) + f[:, hi, :] * w


def judge_dose(op: Siddon, shown: np.ndarray) -> np.ndarray:
    NZ, n, A = shown.shape
    return op.bwd(np.ascontiguousarray(shown.transpose(2, 1, 0).reshape(A * n, NZ)))      # [n*n, Z]


def perturb(shown: np.ndarray, rng: np.random.Generator, n_fine: int) -> tuple[np.ndarray, float]:
    """One virtual machine: blur 0.5 row / 0.5 planning px, axis offset ±0.1 mm, jitter ±1°, exposure ±5 %."""
    from scipy.ndimage import gaussian_filter1d, shift
    out = gaussian_filter1d(shown, 0.5, axis=0, mode="nearest")                    # rows (layers)
    out = gaussian_filter1d(out, 0.5 * n_fine / (n_fine // 2), axis=1, mode="constant")   # pixels
    offset_px = rng.uniform(-0.1, 0.1) / (SPAN / n_fine)
    out = shift(out, (0, offset_px, 0), order=1, mode="constant")
    step = 360.0 / FRAMES
    u = rng.uniform(-1.0, 1.0, FRAMES) / step                                     # jitter in frames
    k = np.arange(FRAMES)
    src = (k + u) % FRAMES
    lo = np.floor(src).astype(int)
    w = (src - lo).astype(np.float32)
    out = out[:, :, lo] * (1 - w) + out[:, :, (lo + 1) % FRAMES] * w
    return out.astype(np.float32), float(1.0 + rng.uniform(-0.05, 0.05))


def _surface_distances(pred: np.ndarray, truth: np.ndarray, spacing: tuple) -> np.ndarray:
    """Distances (mm) from cured-surface voxels to the true surface and back, per layer (2D)."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    out = []
    for z in range(truth.shape[2]):
        t, p = truth[:, :, z], pred[:, :, z]
        if not t.any() and not p.any():
            continue
        ts = t & ~binary_erosion(t)
        ps = p & ~binary_erosion(p)
        if not ts.any() or not ps.any():
            continue
        dt = distance_transform_edt(~ts, sampling=spacing[:2])
        dp = distance_transform_edt(~ps, sampling=spacing[:2])
        out.append(dt[ps])
        out.append(dp[ts])
    return np.concatenate(out) if out else np.array([0.0])


def judge(frames: np.ndarray, fine: np.ndarray, op2: Siddon, n_plan: int, mc: int,
          features: list) -> dict:
    N = fine.shape[0]
    NZ = fine.shape[2]
    shown = displayed(frames, N)
    D = judge_dose(op2, shown).reshape(N, N, NZ)
    resin = resin_mask(N, 0.5)[:, :, None] & np.ones((1, 1, NZ), bool)
    truth = fine >= 0.5
    T = truth[resin]
    Dr = D[resin]
    gel, void = Dr[T], Dr[~T]
    ths = np.linspace(np.percentile(void, 50), np.percentile(gel, 99.5), 400)
    inter = np.array([np.count_nonzero((Dr >= th) & T) for th in ths])
    union = np.array([np.count_nonzero((Dr >= th) | T) for th in ths])
    iou_curve = inter / np.maximum(union, 1)
    k = int(np.argmax(iou_curve))
    th = float(ths[k])
    pred = (D >= th) & resin
    spacing = (SPAN / N, SPAN / N, SPAN / N)
    dist = _surface_distances(pred, truth & resin, spacing)
    # Exposure latitude: how far the exposure may be off (± %, around the planned exposure)
    # while the part is still good (IoU >= 0.95), and while it stays within 0.01 of the
    # engine's own best IoU.
    exposures = np.linspace(0.70, 1.30, 241)
    curve = []
    for e in exposures:
        pr = Dr * e >= th
        curve.append(np.count_nonzero(pr & T) / max(np.count_nonzero(pr | T), 1))
    curve = np.array(curve)

    def bracket(ok: np.ndarray) -> float:
        i = int(np.argmin(np.abs(exposures - 1.0)))
        if not ok[i]:
            return 0.0
        lo = hi = i
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        while hi < len(ok) - 1 and ok[hi + 1]:
            hi += 1
        return 100.0 * min(1.0 - exposures[lo], exposures[hi] - 1.0)

    latitude95 = bracket(curve >= 0.95)
    latitude = bracket(curve >= float(iou_curve[k]) - 0.01)
    core_in = Dr[(fine[resin] > 0.95)]
    core_out = Dr[(fine[resin] < 0.05)]
    lo, hi = np.percentile(core_in, 0.5), np.percentile(core_out, 99.5)
    out = dict(iou=round(float(iou_curve[k]), 5), surf_mean_mm=round(float(dist.mean()), 4),
               surf_p95_mm=round(float(np.percentile(dist, 95)), 4), latitude_pct=round(latitude, 1),
               latitude95_pct=round(latitude95, 1),
               window_pct=round(float(100 * (lo - hi) / lo), 2))
    if mc:
        rng = np.random.default_rng(1234)                  # same machines for every engine
        ious = []
        for _ in range(mc):
            sh, e = perturb(shown, rng, N)
            Dp = judge_dose(op2, sh).reshape(N, N, NZ)[resin] * e
            pr = Dp >= th
            ious.append(np.count_nonzero(pr & T) / max(np.count_nonzero(pr | T), 1))
        out.update(robust_mean=round(float(np.mean(ious)), 4), robust_worst=round(float(np.min(ious)), 4))
    if features:
        out["features"] = feature_report(pred, truth, features, N)
    return out


def _surface_area(mask: np.ndarray, spacing: tuple) -> float:
    hx, hy, hz = spacing
    return (np.count_nonzero(mask[1:] != mask[:-1]) * hy * hz + np.count_nonzero(mask[:, 1:] != mask[:, :-1]) * hx * hz)


def feature_report(pred: np.ndarray, truth: np.ndarray, features: list, N: int) -> dict:
    """A feature prints when, in its own neighbourhood, IoU with the model is >= 0.5 (pins,
    walls) or the hole stays open (>= 50 % of its area clear) in the middle layers."""
    h = SPAN / N
    zs = slice(pred.shape[2] // 4, 3 * pred.shape[2] // 4)
    best: dict[str, Optional[float]] = {"pin": None, "hole": None, "wall": None}
    per: list = []
    for kind, size, x, y in features:
        i0 = int((x - size / 2 - 1.2 + SPAN / 2) / h)
        i1 = int((x + size / 2 + 1.2 + SPAN / 2) / h) + 1
        j0 = int((y - (3.6 if kind == "wall" else size / 2 + 1.2) + SPAN / 2) / h)
        j1 = int((y + (3.6 if kind == "wall" else size / 2 + 1.2) + SPAN / 2) / h) + 1
        p, t = pred[i0:i1, j0:j1, zs], truth[i0:i1, j0:j1, zs]
        if kind == "hole":
            c = (np.arange(N) + 0.5) * h - SPAN / 2
            X, Y = np.meshgrid(c[i0:i1], c[j0:j1], indexing="ij")
            inside = (X - x) ** 2 + (Y - y) ** 2 < (size / 2) ** 2
            if not inside.any():
                ok, score = False, 0.0
            else:
                score = float(1.0 - p[inside].mean())
                ok = score >= 0.5
        else:
            score = float(np.count_nonzero(p & t) / max(np.count_nonzero(p | t), 1))
            ok = score >= 0.5
        per.append((kind, size, round(score, 3)))
        if ok and (best[kind] is None or size < best[kind]):
            best[kind] = size
    return {"smallest": best, "scores": per}


# ----------------------------------------------------------------------------- driver
ENGINES: dict[str, Callable] = {}


def plan(engine: str, target: np.ndarray, case: Case, n: int, op: Optional[Siddon], cfg: dict) -> np.ndarray:
    if engine.startswith("fastcal:"):              # named variant, settings in --config
        return run_fastcal(target, case, n, False, **cfg)
    if engine == "fastcal":
        return run_fastcal(target, case, n, False, **cfg)
    if engine == "fastcal_bin":
        return run_fastcal(target, case, n, True, **cfg)
    assert op is not None
    if engine == "osmo":
        return run_osmo(target, op, **cfg)
    if engine == "pm":
        return run_pm(target, op, **cfg)
    if engine == "bclp":
        return run_bclp(target, op, **cfg)
    if engine == "fbp":
        return run_fbp(target, op)
    raise SystemExit(f"unknown engine {engine}")


def voxel_targets(case: Case, n: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = case.mesh
    shift = -np.asarray(mesh.bounds).mean(axis=0)
    mesh.apply_translation(shift)
    case.features = [(k, s, x + shift[0], y + shift[1]) for k, s, x, y in case.features]
    pitch = SPAN / n
    z = np.arange(mesh.bounds[0, 2] + pitch / 2, mesh.bounds[1, 2], pitch)
    # A projector row lights its layer evenly in height, so a layer cures or not as a whole:
    # both targets are sections through the layer centres (as the slicer voxelises), with
    # sub-pixel coverage only across the layer.
    coarse = voxelize(mesh, z, n, SPAN, sub_y=4).volume
    fine = voxelize(mesh, z, 2 * n, SPAN, sub_y=4).volume
    return coarse.astype(np.float32), fine.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="nut,benchy,restarget")
    ap.add_argument("--engines", default="fastcal,fastcal_bin,osmo,pm,bclp,fbp")
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--mc", type=int, default=8, help="virtual machines for the robustness test (0 = skip)")
    ap.add_argument("--config", default="{}", help='JSON per engine, e.g. {"osmo": {"n_iter": 150}}')
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    configs = json.loads(args.config)
    n = args.grid
    report: dict = {"grid": n, "judge_grid": 2 * n, "frames": FRAMES, "cases": {}}
    for case_name in [c for c in args.cases.split(",") if c]:
        case = make_case(case_name)
        target, fine = voxel_targets(case, n)
        engines = [e for e in args.engines.split(",") if e]
        need_siddon = any(not e.startswith("fastcal") and not e.endswith("2") for e in engines)
        need_siddon2 = any(not e.startswith("fastcal") and e.endswith("2") for e in engines)
        op = Siddon(n, case.mode, case.alpha) if need_siddon else None
        op_ns2 = Siddon(n, case.mode, case.alpha, num_s=2 * n) if need_siddon2 else None
        frames = {}
        res: dict = {}
        for eng in engines:
            cfg = dict(configs.get(eng, {}))
            tracemalloc.start()
            t0 = time.perf_counter()
            if eng.endswith("2") and not eng.startswith("fastcal"):
                frames[eng] = plan(eng[:-1], target, case, n, op_ns2, cfg)     # same engine, 2x detector
            else:
                frames[eng] = plan(eng, target, case, n, op, cfg)
            secs = time.perf_counter() - t0
            peak = tracemalloc.get_traced_memory()[1] / 2**20
            tracemalloc.stop()
            res[eng] = {"seconds": round(secs, 1), "peak_mb": round(peak)}
            print(f"[{case_name}] planned {eng} in {secs:.1f} s", file=sys.stderr, flush=True)
        del op, op_ns2
        op2 = Siddon(2 * n, case.mode, case.alpha)
        for eng, f in frames.items():
            res[eng].update(judge(f, fine, op2, n, args.mc, case.features))
            print(f"[{case_name}] {eng}: {json.dumps(res[eng])}", file=sys.stderr, flush=True)
        del op2
        report["cases"][case_name] = {"layers": int(target.shape[2]), "results": res}
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
