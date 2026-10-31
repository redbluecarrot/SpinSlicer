"""Virtual print: cured volume, exposure sweep, print-plan wiring and the 3D surface."""
from __future__ import annotations

import os

import numpy as np
import pytest
import trimesh

import virtual_print
from print_plan import compute_print_plan, format_print_plan
from slicing_engine import ResinSettings, SliceParams, SlicingEngine


def _ball(n: int = 40, radius: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """Target ball and a smooth "dose" that crosses the gel level exactly on its surface."""
    c = np.arange(n) - (n - 1) / 2
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)
    target = (r <= radius).astype(np.float32)
    dose = 2.0 / (1.0 + np.exp((r - radius) / 2.0))                 # = 1 on the surface
    cured = np.clip(dose * virtual_print.SCALE, 0, 255).astype(np.uint8)
    return target, cured


def test_exposure_sweep_finds_the_size_true_exposure_and_a_tolerance():
    target, cured = _ball()
    m = virtual_print.match_metrics(cured, target)
    assert m["iou"] > 0.97
    assert m["best_exposure"] == pytest.approx(1.0, abs=0.03)
    assert m["exposure_low"] < m["best_exposure"] < m["exposure_high"]
    # more light grows the part, less light shrinks it
    assert m["plus5"]["offset_mm"] > 0 > m["minus5"]["offset_mm"]
    assert m["verdict"] in {"ok", "tight"}


def test_print_plan_takes_its_exposure_from_the_virtual_print():
    stats = dict(threshold=10.0, window=0.1, misprinted_fraction=0.0, background_per_unit_black=5.0)
    common = dict(alpha_per_mm=0.02, irradiance_mw_cm2=5.0, gel_dose_mj_cm3=50.0, detector_bin_mm=0.2,
                  voxel_pitch_mm=0.2, frame_rate_hz=24.0, num_frames=360, coverage_radius_mm=20.0,
                  part_radius_mm=10.0, part_fraction_outside=0.0, projector_contrast=1000.0)
    plain = compute_print_plan(stats, **common)
    exposure = dict(best_exposure=0.9, exposure_low=0.85, exposure_high=0.96, tolerance_pct=5.6)
    tuned = compute_print_plan(stats, exposure=exposure, **common)
    assert tuned["print_time_s"] == pytest.approx(0.9 * plain["print_time_s"])
    assert tuned["print_time_min_s"] == pytest.approx(0.85 * plain["print_time_s"])
    assert tuned["exposure_tolerance_pct"] == pytest.approx(5.6)


def test_slicing_writes_a_virtual_print_that_matches_the_part(tmp_path):
    mesh = trimesh.creation.box(extents=(14.0, 10.0, 4.0))
    params = SliceParams(diameter_mm=60.0, grid_res=64, output_res=64, num_frames=72, fill_holes=True,
                         resin=ResinSettings(), output_dir=str(tmp_path), projection_backend="physical",
                         optimizer_iterations=15, container_mode=2, cuvette_outer_mm=60.0,
                         cuvette_inner_mm=56.0, alpha=0.01, resin_gel_dose_mj_cm3=100.0,
                         projector_irradiance_mw_cm2=10.0)
    _, out_dir = SlicingEngine.run(mesh, params)
    path = virtual_print.mesh_path(out_dir)
    assert path is not None
    assert os.path.isfile(os.path.join(out_dir, "virtual_print_slices.png"))
    import json
    with open(os.path.join(out_dir, "print_plan.json"), encoding="utf-8") as handle:
        plan = json.load(handle)
    assert plan["virtual_print"]["iou"] > 0.9
    assert "Виртуальная печать" in format_print_plan(plan)
    # the cured surface is a closed, outward-facing shell where the box is
    cured = trimesh.load(path, force="mesh")
    assert cured.volume > 0
    np.testing.assert_allclose(cured.extents, (14.0, 10.0, 4.0), atol=2.0)
    assert 0.5 * 14.0 * 10.0 * 4.0 < cured.volume < 1.2 * 14.0 * 10.0 * 4.0   # coarse 64 grid
