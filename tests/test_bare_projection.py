import os
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from frame_io import FrameRepository, load_manifest
from optics.optimization import project_volume
from slicing_engine import BARE_PROJECTION_DIR, ResinSettings, SliceParams, SlicingEngine


def _params(tmp_path, **overrides) -> SliceParams:
    values = dict(
        diameter_mm=40.0, grid_res=32, output_res=64, num_frames=24,
        fill_holes=True, resin=ResinSettings(), output_dir=str(tmp_path),
        projection_backend="physical", optimizer_iterations=3,
        cuvette_outer_mm=40.0, cuvette_inner_mm=36.0,
        n_air=1.0, n_glass=1.51, n_liquid=1.45, n_resin=1.6,
    )
    values.update(overrides)
    return SliceParams(**values)


def test_bare_projection_is_the_plain_straight_ray_projection() -> None:
    grid = 32
    target = np.zeros((grid, grid, 2), dtype=np.float32)
    target[10:22, 8:20, :] = 1.0
    angles = np.deg2rad(np.linspace(0.0, 360.0, 16, endpoint=False))
    bare = project_volume(
        target, angles, L_out=40.0, L_in=40.0, grid_res=grid, grid_span_mm=40.0
    )
    assert bare.shape == (2, grid, 16)
    assert (bare >= 0).all() and bare.max() > 0
    # Every angle integrates the same object, so total ray length is conserved.
    totals = bare[0].sum(axis=0)
    assert totals.max() / totals.min() < 1.1
    # Layers with identical content give identical projections.
    np.testing.assert_allclose(bare[0], bare[1])


def test_engine_writes_bare_frames_next_to_the_compensated_ones(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(8.0, 8.0, 8.0))
    _, run_dir = SlicingEngine.run(mesh, _params(tmp_path, compute_mode="maximum"))
    run = Path(run_dir)
    bare = run / BARE_PROJECTION_DIR
    main_frames = sorted(run.glob("frame_*.png"))
    bare_frames = sorted(bare.glob("frame_*.png"))
    assert len(main_frames) == len(bare_frames) == 24
    first = Image.open(main_frames[0])
    assert Image.open(bare_frames[0]).size == first.size
    manifest = load_manifest(run_dir)
    assert manifest is not None
    assert manifest.slice_parameters["bare_projection_dir"] == BARE_PROJECTION_DIR
    # The extra folder must not confuse frame-set validation.
    assert FrameRepository.validate(run_dir).frame_count == 24
    # With refraction in the vial the compensated frames differ from the bare ones.
    a = np.asarray(first, dtype=float)
    b = np.asarray(Image.open(bare_frames[0]), dtype=float)
    assert np.abs(a - b).mean() > 0.5


def test_bare_projection_can_be_switched_off(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(8.0, 8.0, 8.0))
    _, run_dir = SlicingEngine.run(mesh, _params(tmp_path, compute_mode="maximum", save_bare_projection=False))
    assert not os.path.exists(os.path.join(run_dir, BARE_PROJECTION_DIR))
    manifest = load_manifest(run_dir)
    assert manifest is not None and manifest.slice_parameters["bare_projection_dir"] == ""


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_projector_tab_offers_bare_mode_only_when_available(tmp_path, qapp) -> None:
    from video_tab import ProjectorTab

    mesh = trimesh.creation.box(extents=(8.0, 8.0, 8.0))
    _, with_bare = SlicingEngine.run(mesh, _params(tmp_path / "a", compute_mode="maximum"))
    _, without = SlicingEngine.run(mesh, _params(tmp_path / "b", compute_mode="maximum", save_bare_projection=False))

    tab = ProjectorTab()
    tab.set_output_dir(with_bare)
    assert tab.mode_bare_btn.isEnabled() and tab.mode_corrected_btn.isChecked()
    tab.mode_bare_btn.setChecked(True)
    assert tab._bare_selected()
    tab.set_output_dir(without)
    assert not tab.mode_bare_btn.isEnabled()
    assert tab.mode_corrected_btn.isChecked() and not tab._bare_selected()


def test_assemble_worker_loads_either_frame_set(tmp_path) -> None:
    from workers import VideoAssembleWorker

    mesh = trimesh.creation.box(extents=(8.0, 8.0, 8.0))
    _, run_dir = SlicingEngine.run(mesh, _params(tmp_path, compute_mode="maximum"))

    def load(subdir: str):
        loaded: list = []
        errors: list = []
        worker = VideoAssembleWorker(run_dir, subdir=subdir)
        worker.finished_ok.connect(loaded.append)
        worker.failed.connect(errors.append)
        worker.run()
        assert not errors, errors
        return loaded[0]

    corrected, bare = load(""), load(BARE_PROJECTION_DIR)
    assert len(corrected) == len(bare) == 24
    assert corrected[0].shape == bare[0].shape
    assert not np.array_equal(corrected[0], bare[0])
