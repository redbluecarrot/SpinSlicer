from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from optics.optimization import build_system_matrix
from slicing_engine import (
    ResinSettings,
    SliceParams,
    SlicingEngine,
    _exported_detector_row,
    _projection_gray,
    _relative_dose_fit,
)


def parameters(tmp_path):
    return SliceParams(
        diameter_mm=50., grid_res=24, output_res=48, num_frames=16,
        fill_holes=True, resin=ResinSettings(), output_dir=str(tmp_path),
        projection_backend="physical", optimizer_iterations=3,
        cuvette_outer_mm=50., cuvette_inner_mm=46.,
        n_air=1., n_glass=1., n_liquid=1., n_resin=1.,
    )


def test_export_gain_is_global_linear_below_saturation():
    frame = np.array([0, 0.25, 0.5, 1], dtype=np.float32)
    np.testing.assert_allclose(_projection_gray(frame, 1., ResinSettings(intensity=50)), frame * 0.5)
    np.testing.assert_allclose(_projection_gray(frame, 1., ResinSettings(intensity=1)), frame * 0.01)
    assert not _projection_gray(frame, 1., ResinSettings(threshold=100)).any()


def test_relative_rmse_has_no_hidden_sigmoid_and_does_not_depend_on_resolution():
    target = np.array([[0, 0.5], [1, 0]], dtype=np.float32)
    dose, error, scale = _relative_dose_fit(target * 20, target)
    np.testing.assert_allclose(dose, target)
    assert error == pytest.approx(0) and scale == pytest.approx(0.05)
    _, dark_error, _ = _relative_dose_fit(np.zeros_like(target), target)
    assert dark_error == pytest.approx(np.sqrt(np.mean(target**2)))
    _, enlarged_error, _ = _relative_dose_fit(np.zeros((4, 4)), np.tile(target, (2, 2)))
    assert enlarged_error == pytest.approx(dark_error)


def test_preview_and_rmse_are_computed_from_actual_exported_pngs(tmp_path):
    mesh = trimesh.creation.box(extents=(5, 7, 6))
    mesh.apply_translation((7, -3, 0))
    params = parameters(tmp_path)
    _, directory = SlicingEngine.run(mesh, params)
    directory = Path(directory)
    columns = []
    for file in sorted(directory.glob("frame_*.png")):
        gray = np.asarray(Image.open(file))
        # Constant rectangular section in Z: either middle row is the target.
        columns.append(_exported_detector_row(gray, gray.shape[0] // 2, params.grid_res))
    sinogram = np.stack(columns, axis=1)
    angles = np.deg2rad((params.angle_offset_deg + np.arange(params.num_frames) * 360 / params.num_frames) % 360)
    _, adjoint = build_system_matrix(
        angles, params.grid_res, params.grid_res, params.diameter_mm,
        params.cuvette_outer_mm, params.cuvette_inner_mm,
        n_air=1., n_glass=1., n_liquid=1., n_resin=1.,
    )
    dose = (adjoint @ sinogram.T.ravel()).reshape(params.grid_res, params.grid_res)
    np.testing.assert_allclose(np.load(directory / "reconstruction_dose.npy"), dose, atol=2e-6)
    target = np.load(directory / "reconstruction_target.npy")
    fitted, expected_rmse, _ = _relative_dose_fit(dose, target)
    stats = dict(line.split("=", 1) for line in (directory / "dose_stats.txt").read_text().splitlines())
    assert stats["source"] == "exported_png"
    assert float(stats["rmse"]) == pytest.approx(expected_rmse, abs=1e-7)
    assert float(stats["loss"]) == pytest.approx(np.linalg.norm(fitted-target), abs=1e-6)

    # The preview must report failure to deliver the target when all PNGs are black.
    _, dark_directory = SlicingEngine.run(mesh, replace(params, resin=ResinSettings(threshold=100)))
    assert not np.load(Path(dark_directory) / "reconstruction_dose.npy").any()
    dark_stats = dict(line.split("=", 1) for line in (Path(dark_directory) / "dose_stats.txt").read_text().splitlines())
    assert float(dark_stats["rmse"]) > 0


def test_ring_rasterization_preserves_area_and_bore_without_half_mm_outline(tmp_path):
    mesh = trimesh.creation.annulus(r_min=2., r_max=6., height=5., sections=128)
    mesh.apply_translation((2., -1., 0))
    params = replace(parameters(tmp_path), diameter_mm=20., grid_res=64, output_res=64,
                     cuvette_outer_mm=20., cuvette_inner_mm=18.)
    _, directory = SlicingEngine.run(mesh, params)
    target = np.load(Path(directory) / "reconstruction_target.npy")
    pitch = params.diameter_mm / params.grid_res
    area = float(target.sum()) * pitch**2
    assert area == pytest.approx(np.pi * (6**2 - 2**2), rel=0.05)
    centres = (np.arange(params.grid_res) + 0.5) * pitch - params.diameter_mm / 2
    assert target[np.argmin(abs(centres-2)), np.argmin(abs(centres+1))] == 0
