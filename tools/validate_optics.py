"""Compare relative-dose phantoms against an optional historical Git version.

This tests numerical shape error, not a calibrated resin curing outcome.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optics.optimization import build_system_matrix, optimize_projections_sirt_volume
from slicing_engine import _relative_dose_fit


def phantom():
    coordinates = (np.arange(64) + 0.5) * 60 / 64 - 30
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius_squared = (x-4)**2 + (y+2)**2
    ring = (radius_squared < 11**2) & (radius_squared > 3**2)
    rectangle = (x > -15) & (x < -9) & (y > 3) & (y < 13)
    return np.stack([ring, rectangle, np.zeros_like(ring)], axis=2).astype(np.float32)


def parameters():
    return dict(L_out=60., L_in=56., grid_res=64, grid_span_mm=60.,
                n_air=1., n_glass=1., n_liquid=1., n_resin=1., iterations=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", help="Example: v0.3.0. Runs its Python backend in an isolated process.")
    arguments = parser.parse_args()
    target = phantom()
    angles = np.linspace(0, 2*np.pi, 90, endpoint=False)
    a, at = build_system_matrix(angles, 64, 64, 60., 60., 56., n_air=1., n_glass=1., n_liquid=1., n_resin=1.)

    def evaluate(projections):
        rays = projections.transpose(2, 1, 0).reshape(a.shape[0], -1)
        dose = (at @ rays).reshape(target.shape)
        relative, rmse, scale = _relative_dose_fit(dose, target)
        predicted = relative >= 0.5
        truth = target >= 0.5
        union = np.count_nonzero(predicted | truth)
        return dict(relative_dose_rmse=rmse, fitted_scale=scale,
                    shape_iou_at_half_unit=float(np.count_nonzero(predicted & truth) / union))

    report = {"grid": 64, "angles": 90, "iterations": 8, "layers": 3,
              "model": "linear relative dose; a common least-squares scale; no cure model"}
    if arguments.baseline_ref:
        with tempfile.TemporaryDirectory(prefix="spinslicer-reference-") as directory:
            folder = Path(directory)
            (folder / "optics").mkdir()
            (folder / "optics" / "__init__.py").write_text("")
            for name in ("optimization.py", "ray_tracing.py"):
                content = subprocess.run(
                    ["git", "show", f"{arguments.baseline_ref}:optics/{name}"],
                    cwd=ROOT, check=True, capture_output=True,
                ).stdout
                (folder / "optics" / name).write_bytes(content)
            np.save(folder / "target.npy", target)
            script = (
                "import numpy as np; from optics import optimization as o; "
                "o._load_cpp_backend=lambda:None; "
                f"p=o.optimize_projections_sirt_volume(np.load('target.npy'),np.linspace(0,2*np.pi,90,endpoint=False),**{parameters()!r}); "
                "np.save('projections.npy',p)"
            )
            subprocess.run([sys.executable, "-c", script], cwd=folder, check=True,
                           env=os.environ | {"PYTHONPATH": str(ROOT)})
            report[arguments.baseline_ref] = evaluate(np.load(folder / "projections.npy"))
    report["corrected"] = evaluate(optimize_projections_sirt_volume(target, angles, **parameters()))
    print(json.dumps(report, indent=2))
