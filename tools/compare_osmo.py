"""FastCAL vs OSMO vs BCLP, head to head, judged by none of the engines' own models.

  python tools/compare_osmo.py                     # built-in threaded nut, grid 128, immersion
  python tools/compare_osmo.py model.stl --grid 256 --air --scale 0.75
  python tools/compare_osmo.py --engines fastcal,bclp --bclp-iters 200 --bclp-lr 0.005

Both engines plan the same voxel target with the same optics (refraction, attenuation,
no extras: LEGACY_OPTICS, which is what the exact sparse operator models):
  * FastCAL: optics/fastcal.py with its default settings;
  * OSMO: the VAMToolbox `minimizeOSMO` update rule verbatim (50 iterations, d_h 0.8,
    d_l 0.7, ram-lak filtered target), with SpinSlicer's exact Siddon operator in place
    of ASTRA, so refraction is included for it too;
  * BCLP: the VAMToolbox `minimizeBCLP` defaults (generalized logistic response B 25,
    eps 0.1, p 2, q 1, projected gradient descent from a ram-lak start), 50 iterations
    with step 0.01 * (grid/256)^2 (the best fixed step found), on the same exact operator.

Judge: the exact Cartesian Siddon operator on a 2x finer grid, applied to 8-bit frames,
against a target voxelised independently at that resolution. Reported per engine:
time, IoU and wrong voxels at the best single exposure, the robust exposure window,
VAMToolbox VER, and the same worst-voxel numbers away from partial edge voxels ("core").
Frames are also compared angle by angle (correlation), to show how alike they are.

The 2x judge grid needs memory: about 1 GB at --grid 128, 6+ GB at --grid 256.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optics.fastcal import LEGACY_OPTICS, FastVialEngine  # noqa: E402
from optics.optimization import _load_cpp_backend, _project, build_system_matrix  # noqa: E402
from voxelizer import voxelize  # noqa: E402

SPAN, L_OUT, L_IN, N_GLASS, N_LIQUID, N_RESIN = 60.0, 60.0, 56.0, 1.51, 1.45, 1.50
FRAMES = 360


def load_mesh(path: str | None, scale: float):
    import trimesh
    if path:
        mesh = trimesh.load(path, force="mesh")
    else:
        from threaded_nut import ThreadedNutParameters, create_threaded_nut
        mesh = create_threaded_nut(ThreadedNutParameters(outer_diameter_mm=36, bore_diameter_mm=16, height_mm=24,
                                                         pitch_mm=3, thread_depth_mm=1.2))
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


def operator(n: int, mode: int):
    angles = np.deg2rad((90.0 + np.linspace(0, 360, FRAMES, endpoint=False)) % 360)
    A, AT = build_system_matrix(angles, n, n, SPAN, L_OUT, L_IN, 1.0, N_GLASS, N_LIQUID, N_RESIN, 0.0, mode,
                                memory_budget_bytes=8 * 2**30)
    return angles, A, AT


def run_osmo(target: np.ndarray, A, AT, lib) -> np.ndarray:
    import scipy.fft as sfft
    N, _, NZ = target.shape
    fwd = lambda x: _project(A, np.ascontiguousarray(x.reshape(N * N, NZ), dtype=np.float32), lib)  # noqa: E731
    bwd = lambda b: _project(AT, np.ascontiguousarray(b, dtype=np.float32), lib).reshape(N, N, NZ)  # noqa: E731
    c = (np.arange(N) + 0.5) * SPAN / N - SPAN / 2
    X, Y = np.meshgrid(c, c, indexing="ij")
    circle = (X**2 + Y**2 < (L_IN / 2) ** 2)[:, :, None]
    T = (target >= 0.5).astype(np.float32)
    gel, void = (T > 0) & circle, (T == 0) & circle
    fy, fx = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), indexing="ij")
    filt = np.sqrt(fx**2 + fy**2)
    xm = np.empty_like(T)
    for z in range(NZ):
        xm[:, :, z] = np.real(sfft.ifft2(sfft.ifftshift(sfft.fftshift(sfft.fft2(T[:, :, z])) * filt)))
    b = np.clip(fwd(xm), 0, None)
    x = bwd(b)
    x /= x.max()
    for _ in range(1, 50):
        diff = np.maximum(x[void] - 0.7, 0)
        xm[void] -= diff
        b = fwd(xm); b /= b.max(); b[b < 0] = 0
        x = bwd(b); x /= x.max(); x[x < 0] = 0
        diff = np.maximum(0.8 - x[gel], 0)
        xm[gel] += diff
        b = fwd(xm); b /= b.max(); b[b < 0] = 0
        x = bwd(b); x /= x.max(); x[x < 0] = 0
    return b.reshape(FRAMES, N, NZ).transpose(2, 1, 0)          # [z, s, angle]


def run_bclp(target: np.ndarray, A, AT, lib, iterations: int, lr: float) -> np.ndarray:
    N, _, NZ = target.shape
    fwd = lambda x: _project(A, np.ascontiguousarray(x.reshape(N * N, NZ), dtype=np.float32), lib)  # noqa: E731
    bwd = lambda b: _project(AT, np.ascontiguousarray(b, dtype=np.float32), lib).reshape(N, N, NZ)  # noqa: E731
    c = (np.arange(N) + 0.5) * SPAN / N - SPAN / 2
    X, Y = np.meshgrid(c, c, indexing="ij")
    w = (X**2 + Y**2 < (L_IN / 2) ** 2)[:, :, None].astype(np.float32)
    T = (target >= 0.5).astype(np.float32)
    eps, B, M = 0.1, 25.0, 0.5
    g = fwd(T).reshape(FRAMES, N, NZ)
    ramp = np.abs(np.fft.fftfreq(N))[None, :, None]
    g = np.maximum(np.real(np.fft.ifft(np.fft.fft(g, axis=1) * ramp, axis=1)), 0).astype(np.float32)
    g = g.reshape(FRAMES * N, NZ)
    g *= np.float32(1.0 / np.median(bwd(g)[T > 0]))
    for _ in range(iterations):
        m = 1.0 / (1.0 + np.exp(-B * (bwd(g) - M)))
        err = m - T
        band = np.abs(err) - eps
        v = band > 0
        loss = float(np.sqrt(np.sum(v * w * band**2)))
        if loss == 0:
            break
        operand = (v * w * band * np.sign(err) * B * m * (1 - m)).astype(np.float32)
        g = np.maximum(g - np.float32(lr / loss) * fwd(operand), 0)
    return g.reshape(FRAMES, N, NZ).transpose(2, 1, 0)


def judge(frames: np.ndarray, fine: np.ndarray, AT, lib) -> dict:
    N, _, NZ = fine.shape
    s = np.rint(np.clip(frames / frames.max(), 0, 1) * 255) / 255
    s = s[:, (np.arange(N) * s.shape[1]) // N, :]
    D = _project(AT, np.ascontiguousarray(s.transpose(2, 1, 0).reshape(FRAMES * N, NZ), dtype=np.float32), lib)
    c = (np.arange(N) + 0.5) * SPAN / N - SPAN / 2
    X, Y = np.meshgrid(c, c, indexing="ij")
    resin = (X**2 + Y**2 < (L_IN / 2 - 0.5) ** 2).ravel()
    T = fine.reshape(N * N, NZ)
    B = T >= 0.5
    din, dout = D[(T > 0.95) & resin[:, None]], D[(T < 0.05) & resin[:, None]]
    gel, void = D[B & resin[:, None]], D[~B & resin[:, None]]
    Dr, Br = D[resin], B[resin]
    best = (0.0, 0.0)
    for th in np.linspace(np.percentile(void, 50), np.percentile(gel, 99), 300):
        pr = Dr >= th
        iou = np.count_nonzero(pr & Br) / np.count_nonzero(pr | Br)
        if iou > best[0]:
            best = (iou, np.count_nonzero(pr ^ Br) / Br.size)
    lo, hi = np.percentile(din, 0.5), np.percentile(dout, 99.5)
    return dict(iou=round(best[0], 5), wrong_voxels_pct=round(100 * best[1], 3),
                window_pct=round(float(100 * (lo - hi) / lo), 2),
                VER=round(float(np.count_nonzero(void >= gel.min()) / (gel.size + void.size)), 4),
                VER_core=round(float(np.count_nonzero(dout >= din.min()) / (din.size + dout.size)), 5))


def similarity(a: np.ndarray, b: np.ndarray) -> dict:
    a = a / a.max()
    b = b / b.max()
    r = []
    for k in range(a.shape[2]):
        lit = (a[:, :, k] > 0.01) | (b[:, :, k] > 0.01)
        r.append(np.corrcoef(a[:, :, k][lit], b[:, :, k][lit])[0, 1] if lit.sum() > 2 else 1.0)
    return dict(mean=round(float(np.mean(r)), 3), min=round(float(np.min(r)), 3))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stl", nargs="?")
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--air", action="store_true", help="vial in air instead of an immersion bath")
    ap.add_argument("--engines", default="fastcal,osmo,bclp")
    ap.add_argument("--bclp-iters", type=int, default=50)
    ap.add_argument("--bclp-lr", type=float, default=None,
                    help="BCLP step; default 0.01 * (grid/256)^2, the best fixed step found")
    args = ap.parse_args()
    mode = 0 if args.air else 2
    N = args.grid
    mesh = load_mesh(args.stl, args.scale)
    pitch = SPAN / N
    z = np.arange(mesh.bounds[0, 2] + pitch / 2, mesh.bounds[1, 2], pitch)
    target = voxelize(mesh, z, N, SPAN, sub_z=2, z_pitch=pitch).volume
    fine = voxelize(mesh, z, 2 * N, SPAN).volume
    lib = _load_cpp_backend()
    angles, A, AT = operator(N, mode)
    results: dict = {}

    engines = [e.strip().lower() for e in args.engines.split(",") if e.strip()]
    frames: dict[str, np.ndarray] = {}
    for name in engines:
        t0 = time.perf_counter()
        if name == "fastcal":
            eng = FastVialEngine(angles, N, N, SPAN, L_OUT, L_IN, 1.0, N_GLASS, N_LIQUID, N_RESIN, 0.0, mode,
                                 optics=LEGACY_OPTICS)
            frames["FastCAL"] = np.asarray(eng.optimize_volume(target), np.float32)
        elif name == "osmo":
            frames["OSMO"] = run_osmo(target, A, AT, lib)
        elif name == "bclp":
            lr = args.bclp_lr if args.bclp_lr is not None else 0.01 * (N / 256) ** 2
            frames["BCLP"] = run_bclp(target, A, AT, lib, args.bclp_iters, lr)
        else:
            raise SystemExit(f"unknown engine {name}")
        results[list(frames)[-1]] = {"seconds": round(time.perf_counter() - t0, 1)}
    del A, AT
    _, A2, AT2 = operator(2 * N, mode)
    del A2
    for name, f in frames.items():
        results[name].update(judge(f, fine, AT2, lib))
    if "FastCAL" in frames:
        results["frame_similarity_to_FastCAL"] = {k: similarity(frames["FastCAL"], f)
                                                  for k, f in frames.items() if k != "FastCAL"}
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
