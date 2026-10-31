from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from frame_io import FrameRepository, load_manifest
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
    manifest = load_manifest(str(directory))
    assert manifest is not None
    # FastCAL plans with up to 2x finer detector bins than voxels (optics.fastcal.detector_bins).
    bins = int(manifest.slice_parameters["optics"].get("detector_bins", params.grid_res))
    columns = []
    for file in sorted(directory.glob("frame_*.png")):
        gray = np.asarray(Image.open(file))
        # Constant rectangular section in Z: either middle row is the target.
        columns.append(_exported_detector_row(gray, gray.shape[0] // 2, bins))
    sinogram = np.stack(columns, axis=1)
    angles = np.deg2rad((params.angle_offset_deg + np.arange(params.num_frames) * 360 / params.num_frames) % 360)
    _, adjoint = build_system_matrix(
        angles, bins, params.grid_res, params.diameter_mm,
        params.cuvette_outer_mm, params.cuvette_inner_mm,
        n_air=1., n_glass=1., n_liquid=1., n_resin=1.,
    )
    exact = (adjoint @ sinogram.T.ravel()).reshape(params.grid_res, params.grid_res)
    dose = np.load(directory / "reconstruction_dose.npy")
    if manifest.slice_parameters["optics"]["optimizer"] == "fastcal-band":
        # FastCAL previews with its own polar operator; it must agree with the
        # exact pixel-basis operator applied to the same exported PNG values.
        # The two discretisations differ most at very few angles (16 here:
        # ~14 % relative L2; ~7 % at 72 angles), where streaks are sharp.
        assert np.linalg.norm(dose - exact) / np.linalg.norm(exact) < 0.2
    else:
        np.testing.assert_allclose(dose, exact, atol=2e-6)
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


def test_square_corner_export_has_diagonal_field_and_correct_physical_aspect(tmp_path):
    mesh = trimesh.creation.box(extents=(2., 2., 14.))
    mesh.apply_translation((21., -21., 0.))
    params = replace(parameters(tmp_path), grid_res=64, output_res=128,
                     container_mode=1, angle_offset_deg=0.)
    _, directory = SlicingEngine.run(mesh, params)
    info = FrameRepository.validate(directory)
    manifest = load_manifest(directory)
    field = 50 * np.sqrt(2)
    assert info.width == 128
    assert info.height == round(128 * 14 / field)
    assert manifest is not None
    assert info.nz == manifest.slice_parameters["raster_layers"]
    assert manifest.slice_parameters["voxel_pitch_z_mm"] == pytest.approx(field / 128)
    geometry = manifest.slice_parameters["optics"]
    assert geometry["detector_span_mm"] == pytest.approx(field)
    assert geometry["detector_spacing_mm"] == pytest.approx(field / 64)
    assert geometry["output_pixel_pitch_mm"] == pytest.approx(field / 128)
    assert geometry["voxel_grid_span_mm"] == 50
    for index, theta in ((0, 0.), (2, np.pi/4)):
        with Image.open(Path(directory) / f"frame_{index:04d}.png") as image:
            gray = np.asarray(image)
        assert gray.any()  # the corner was completely absent at 45 degrees
        peak_column = np.argmax(gray.sum(axis=0))
        expected_position = 21 * np.cos(theta) + 21 * np.sin(theta)
        expected_column = (expected_position / field + 0.5) * params.output_res - 0.5
        assert abs(peak_column - expected_column) < 4


@pytest.mark.parametrize("height", [6.0, 8.8542, 9.8958])
def test_one_layer_is_exactly_one_projector_row(tmp_path, height):
    """Layers are sliced at the projector row pitch, so frames must have exactly one
    row per layer: rounding the part height instead stretched every frame by a row
    for some heights (10.25 mm here: 49.2 rows of 0.2 mm, 50 layers)."""
    mesh = trimesh.creation.box(extents=(5, 7, height))
    params = parameters(tmp_path)
    pitch = params.diameter_mm / params.output_res
    _, directory = SlicingEngine.run(mesh, params)
    layers = len(np.arange(-height / 2 + pitch / 2, height / 2, pitch))
    frame = np.asarray(Image.open(sorted(Path(directory).glob("frame_*.png"))[0]))
    assert frame.shape == (layers, params.output_res)
