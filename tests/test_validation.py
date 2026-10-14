from types import SimpleNamespace

import numpy as np
import pytest

from slicing_engine import ResinSettings, SliceParams
from validation import (
    ValidationError,
    estimate_layer_count,
    preflight_mesh,
    validate_stl_path,
)


def make_mesh(vertices: np.ndarray, faces: np.ndarray | None = None) -> SimpleNamespace:
    if faces is None:
        faces = np.array([[0, 1, 2]], dtype=np.int64)
    return SimpleNamespace(
        vertices=vertices,
        faces=faces,
        bounds=np.vstack([vertices.min(axis=0), vertices.max(axis=0)]),
    )


def valid_params() -> SliceParams:
    return SliceParams(
        diameter_mm=60.0,
        grid_res=64,
        output_res=128,
        num_frames=30,
        fill_holes=True,
        resin=ResinSettings(),
        output_dir="output_frames",
    )


def test_slice_params_reject_zero_frames() -> None:
    params = valid_params()
    params.num_frames = 0

    with pytest.raises(ValidationError, match="num_frames"):
        params.validate()


def test_preflight_rejects_non_finite_vertices() -> None:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, np.inf, 1]], dtype=np.float64)
    report = preflight_mesh(make_mesh(vertices), valid_params())

    assert not report.ok
    assert any("NaN" in error or "infinite" in error for error in report.errors)


def test_preflight_rejects_model_outside_vat() -> None:
    vertices = np.array(
        [[-40, -1, -1], [40, -1, -1], [0, 40, 1]],
        dtype=np.float64,
    )
    report = preflight_mesh(make_mesh(vertices), valid_params(), vat_height_mm=96.0)

    assert not report.ok
    assert any("radius" in error for error in report.errors)


def test_preflight_accepts_finite_mesh_inside_vat() -> None:
    vertices = np.array(
        [[-10, -10, -10], [10, -10, -10], [0, 10, 10]],
        dtype=np.float64,
    )
    report = preflight_mesh(make_mesh(vertices), valid_params(), vat_height_mm=96.0)

    assert report.ok
    assert report.triangle_count == 1
    assert report.estimated_layers == estimate_layer_count(-10, 10, 60 / 64)


def test_preflight_rejects_unreasonable_render_dimensions() -> None:
    params = valid_params()
    params.grid_res = 16
    params.output_res = 4096
    vertices = np.array(
        [[-1, -1, -250], [1, -1, 250], [0, 1, 0]],
        dtype=np.float64,
    )

    report = preflight_mesh(make_mesh(vertices), params)

    assert not report.ok
    assert any("dimensions" in error for error in report.errors)


def test_stl_file_size_budget_is_checked_before_parsing(monkeypatch, tmp_path) -> None:
    import validation

    path = tmp_path / "too-large.stl"
    path.write_bytes(b"123")
    monkeypatch.setattr(validation, "MAX_STL_FILE_BYTES", 2)

    with pytest.raises(ValidationError, match="too large"):
        validate_stl_path(str(path))


def test_binary_stl_triangle_budget_is_checked_from_header(monkeypatch, tmp_path) -> None:
    import validation

    path = tmp_path / "too-many.stl"
    triangle_count = 2
    payload = bytearray(84 + 50 * triangle_count)
    payload[80:84] = triangle_count.to_bytes(4, "little")
    path.write_bytes(payload)
    monkeypatch.setattr(validation, "MAX_MESH_TRIANGLES", 1)

    with pytest.raises(ValidationError, match="triangles"):
        validate_stl_path(str(path))
