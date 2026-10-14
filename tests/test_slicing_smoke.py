from pathlib import Path

import numpy as np
import trimesh

import slicing_engine
from frame_io import FrameRepository, load_manifest, sha256_file
from slicing_engine import (
    ResinSettings,
    SliceParams,
    SlicingEngine,
    _exported_detector_row,
    _dose_response_and_rmse,
)


def test_dose_rmse_is_comparable_across_grid_resolutions() -> None:
    dose = np.array([[0.0, 0.3], [0.7, 1.0]], dtype=np.float32)
    target = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    response, rmse = _dose_response_and_rmse(dose, target)
    tiled_response, tiled_rmse = _dose_response_and_rmse(
        np.tile(dose, (2, 2)), np.tile(target, (2, 2))
    )

    assert response.shape == dose.shape
    assert np.all((0.0 <= response) & (response <= 1.0))
    assert tiled_response.shape == (4, 4)
    assert np.isclose(rmse, tiled_rmse)


def test_exported_detector_row_preserves_left_to_right_ray_order() -> None:
    frame = np.tile(
        np.array([0, 0, 64, 64, 128, 128, 255, 255], dtype=np.uint8),
        (2, 1),
    )

    detector_row = _exported_detector_row(frame, row_index=0, grid_res=4)

    np.testing.assert_allclose(detector_row, [0.0, 64 / 255, 128 / 255, 1.0])


def test_rasterized_mesh_keeps_physical_y_x_voxel_order(monkeypatch, tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(8.0, 4.0, 8.0))
    mesh.apply_translation((10.0, -8.0, 0.0))
    captured = {}

    def capture_slices(slices, angles, grid_res, num_frames, report, cancelled):
        captured["slices"] = slices.copy()
        return np.zeros((slices.shape[2], grid_res, num_frames), dtype=np.float32)

    monkeypatch.setattr(slicing_engine, "_radon_sinograms", capture_slices)
    params = SliceParams(
        diameter_mm=60.0,
        grid_res=32,
        output_res=64,
        num_frames=4,
        fill_holes=True,
        resin=ResinSettings(),
        output_dir=str(tmp_path / "output_frames"),
    )

    SlicingEngine.run(mesh, params)

    middle = captured["slices"][:, :, captured["slices"].shape[2] // 2]
    yy, xx = np.nonzero(middle > 0.5)
    # The box center maps to row/y ~= 11, column/x ~= 21 at this resolution.
    assert 10 <= float(yy.mean()) <= 12
    assert 20 <= float(xx.mean()) <= 22


def test_repeated_generation_keeps_completed_runs_and_manifest(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    source = tmp_path / "source.stl"
    mesh.export(source)
    output_root = tmp_path / "output_frames"
    params = SliceParams(
        diameter_mm=60.0,
        grid_res=16,
        output_res=32,
        num_frames=8,
        fill_holes=True,
        resin=ResinSettings(),
        output_dir=str(output_root),
    )
    context = {
        "source_path": str(source),
        "transform_matrix": np.eye(4, dtype=np.float64).tolist(),
        "units": "mm",
    }

    _, first_run = SlicingEngine.run(mesh, params, manifest_context=context)
    _, second_run = SlicingEngine.run(mesh, params, manifest_context=context)

    dose_stats = (Path(first_run) / "dose_stats.txt").read_text(encoding="utf-8")
    rmse = float(dose_stats.split("rmse=", 1)[1].splitlines()[0])
    assert 0.0 <= rmse <= 1.0
    assert "loss=" in dose_stats

    runs = sorted(path for path in output_root.iterdir() if path.is_dir())
    assert len(runs) == 2
    assert first_run != second_run
    assert all((path / "manifest.json").exists() for path in runs)

    info = FrameRepository.validate(str(output_root))
    manifest = load_manifest(info.resolved_dir)
    assert info.frame_count == 8
    assert manifest is not None and manifest.complete
    assert manifest.source_name == Path(source).name
    assert manifest.source_sha256 == sha256_file(str(source))
    assert manifest.transform_matrix[0][0] == 1.0
    assert manifest.machine_profile["name"] == "offline-unbound"
    assert manifest.resin_profile["base_exposure_s"] == 1.0
    assert manifest.frame_schedule["angles_deg"] == [90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 0.0, 45.0]
