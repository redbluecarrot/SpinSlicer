import numpy as np
import pytest
import trimesh

from frame_io import load_manifest
from print_plan import compute_print_plan
from slicing_engine import (
    ResinSettings,
    SliceParams,
    SlicingEngine,
    _encode_gray,
    _exported_detector_row,
)


def test_gamma_encoding_round_trips_displayed_light():
    light = np.linspace(0, 1, 1001, dtype=np.float32)
    for gamma in (1.0, 2.2):
        gray = _encode_gray(light, gamma)
        shown = (gray / 255.0) ** gamma
        assert np.max(np.abs(shown - light)) < 0.02
    frame = _encode_gray(np.full((2, 8), 0.25, np.float32), 2.2)
    np.testing.assert_allclose(_exported_detector_row(frame, 0, 4, gamma=2.2), 0.25, atol=0.01)


def test_print_time_follows_absorbed_energy_balance():
    stats = dict(threshold=10.0, window=0.1, misprinted_fraction=0.0, background_per_unit_black=5.0)
    common = dict(alpha_per_mm=0.02, irradiance_mw_cm2=5.0, detector_bin_mm=0.2, voxel_pitch_mm=0.2,
                  frame_rate_hz=24.0, num_frames=360, coverage_radius_mm=20.0, part_radius_mm=10.0,
                  part_fraction_outside=0.0, projector_contrast=1000.0)
    plan = compute_print_plan(stats, gel_dose_mj_cm3=50.0, **common)
    per_second = 1000 * 0.02 * 0.05 * (0.2 / 0.04)          # mJ/cm^3 per s per unit dose
    assert plan["frame_exposure_total_s"] == pytest.approx(50.0 / (per_second * 10.0))
    assert plan["print_time_s"] == pytest.approx(plan["revolutions"] * 15.0)
    doubled = compute_print_plan(stats, gel_dose_mj_cm3=100.0, **common)
    assert doubled["print_time_s"] == pytest.approx(2 * plan["print_time_s"])
    unknown = compute_print_plan(stats, gel_dose_mj_cm3=0.0, **common)
    assert "print_time_s" not in unknown and unknown["warnings"]


def test_generation_writes_a_print_plan(tmp_path):
    mesh = trimesh.creation.cylinder(6, 8, sections=64)
    params = SliceParams(
        diameter_mm=40.0, grid_res=32, output_res=64, num_frames=24, fill_holes=True,
        resin=ResinSettings(), output_dir=str(tmp_path), projection_backend="physical",
        optimizer_iterations=4, container_mode=2, cuvette_outer_mm=40.0, cuvette_inner_mm=36.0,
        alpha=0.02, projector_gamma=2.2, projector_irradiance_mw_cm2=5.0,
        projector_contrast=1000.0, resin_gel_dose_mj_cm3=50.0,
    )
    _, directory = SlicingEngine.run(mesh, params)
    manifest = load_manifest(directory)
    assert manifest is not None
    plan = manifest.slice_parameters["print_plan"]
    assert plan["print_time_s"] > 0 and plan["revolutions"] > 0
    assert manifest.slice_parameters["optics"]["projector_gamma"] == 2.2
    assert (tmp_path / directory / "print_plan.json").exists()
