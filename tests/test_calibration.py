"""Exposure ladder: planned pillar doses and reading the print result."""
import json
import os

import numpy as np
import pytest

from calibration import PILLARS, REFERENCE, Ladder, gel_dose_factor
from print_plan import format_print_plan
from slicing_engine import ResinSettings, SliceParams, SlicingEngine


def test_factor_sits_between_the_last_liquid_and_first_solid_pillar():
    cal = {"relative_dose": list(np.linspace(0.88, 1.12, PILLARS))}
    assert gel_dose_factor(cal, 6) == pytest.approx(0.5 * (0.97 + 1.00) + 0.03)   # pillars 5 and 6
    assert gel_dose_factor(cal, 5) == pytest.approx(0.985)
    assert gel_dose_factor(cal, 1) < 0.88 < 1.12 < gel_dose_factor(cal, None)
    with pytest.raises(ValueError):
        gel_dose_factor(cal, 10)


def test_ladder_geometry_fits_and_pillars_are_told_apart_by_height():
    ladder = Ladder()
    mesh = ladder.mesh()
    assert np.hypot(*mesh.vertices[:, :2].T).max() < 20.0      # inside full ray coverage, even in air
    assert len(set(ladder.heights())) == PILLARS


def test_ladder_job_plans_increasing_pillar_doses(tmp_path):
    params = SliceParams(diameter_mm=60.0, grid_res=96, output_res=96, num_frames=120, fill_holes=True,
                         resin=ResinSettings(), output_dir=str(tmp_path), projection_backend="physical",
                         optimizer_iterations=20, container_mode=2, cuvette_outer_mm=60.0, cuvette_inner_mm=56.0,
                         alpha=0.01, resin_gel_dose_mj_cm3=100.0, projector_irradiance_mw_cm2=10.0)
    ladder = Ladder()
    _, out_dir = SlicingEngine.run(ladder.mesh(), params, calibration=ladder)
    with open(os.path.join(out_dir, "print_plan.json"), encoding="utf-8") as handle:
        plan = json.load(handle)
    rel = np.array(plan["calibration"]["relative_dose"])
    assert rel[REFERENCE] == pytest.approx(1.0)
    assert np.all(np.diff(rel) > 0)
    # coarse test grid: the planned +-12 % ladder is compressed a little, but it is
    # measured from the frames, so the read-back uses the doses actually delivered
    assert rel[-1] - rel[0] > 0.12
    assert "print_time_s" in plan and "virtual_print" not in plan
    assert "Калибровочная печать" in format_print_plan(plan)
