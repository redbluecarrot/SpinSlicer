from __future__ import annotations

import numpy as np
import pytest

from frame_io import load_manifest
from slicing_engine import ResinSettings, SliceParams, SlicingEngine
from threaded_nut import ThreadedNutParameters, create_threaded_nut
from vam_backend import normalize_sinogram_shape


def test_threaded_nut_is_closed_and_centered() -> None:
    mesh = create_threaded_nut(ThreadedNutParameters())

    assert mesh.is_watertight
    assert mesh.is_volume
    assert np.allclose(mesh.bounding_box.centroid, 0.0, atol=1e-9)
    assert mesh.extents == pytest.approx([24.0, 24.0, 10.0])


def test_threaded_nut_thread_relief_changes_inner_radius() -> None:
    params = ThreadedNutParameters(
        outer_diameter_mm=30.0,
        bore_diameter_mm=16.0,
        height_mm=12.0,
        pitch_mm=2.0,
        thread_depth_mm=0.8,
        clearance_mm=0.2,
    )
    mesh = create_threaded_nut(params)
    radii = np.linalg.norm(mesh.vertices[:, :2], axis=1)

    # The inner surface contains both the valley and crest radii; a plain tube
    # would have only one inner radius.
    inner_radii = radii[radii < params.bore_diameter_mm / 2 + params.clearance_mm + 0.01]
    assert inner_radii.max() - inner_radii.min() > params.thread_depth_mm * 0.5


def test_threaded_nut_rejects_impossible_wall() -> None:
    params = ThreadedNutParameters(outer_diameter_mm=12.0, bore_diameter_mm=12.0)

    with pytest.raises(ValueError, match="outer diameter"):
        create_threaded_nut(params)


def test_vam_sinogram_layout_is_normalized() -> None:
    raw = np.zeros((8, 12, 5), dtype=np.float32)
    normalized = normalize_sinogram_shape(raw, grid_res=8, num_frames=12, num_layers=5)

    assert normalized.shape == (5, 8, 12)


def test_slice_params_preserve_internal_voids_by_default() -> None:
    params = SliceParams(
        diameter_mm=60.0,
        grid_res=16,
        output_res=24,
        num_frames=4,
        fill_holes=True,
        resin=ResinSettings(),
        output_dir="output_frames",
    )

    assert params.preserve_internal_voids is True


def test_threaded_nut_pipeline_records_void_preservation(tmp_path) -> None:
    params = SliceParams(
        diameter_mm=60.0,
        grid_res=16,
        output_res=24,
        num_frames=4,
        fill_holes=True,
        resin=ResinSettings(),
        output_dir=str(tmp_path / "frames"),
    )
    _, out_dir = SlicingEngine.run(
        create_threaded_nut(
            ThreadedNutParameters(
                outer_diameter_mm=18.0,
                bore_diameter_mm=8.0,
                height_mm=8.0,
                pitch_mm=2.0,
                thread_depth_mm=0.5,
                segments_per_turn=48,
                axial_segments_per_pitch=8,
            )
        ),
        params,
        manifest_context={
            "model_type": "threaded_nut",
            "source_name": "threaded-nut (parametric)",
        },
    )

    manifest = load_manifest(out_dir)
    assert manifest is not None
    assert manifest.source_name == "threaded-nut (parametric)"
    assert manifest.slice_parameters["model_type"] == "threaded_nut"
    assert manifest.slice_parameters["preserve_internal_voids"] is True
