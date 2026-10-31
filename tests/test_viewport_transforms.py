from __future__ import annotations

import numpy as np
import trimesh.transformations as tf
import vtk

from constants import (
    VIEWPORT_ORIENTATION_AXIS_CONE_RESOLUTION,
    VIEWPORT_ORIENTATION_AXIS_LENGTH,
    VIEWPORT_ORIENTATION_AXIS_LINE_WIDTH,
    VIEWPORT_ORIENTATION_AXIS_ORIGINS,
    VIEWPORT_ORIENTATION_AXIS_TIP_LENGTH,
    VIEWPORT_ORIENTATION_CUBE_SCALE,
)
from viewport import (
    Viewport3D,
    _camera_up_vector,
    _interpolate_camera_pose,
    _scale_matrix_from_box,
    _slerp_unit_vectors,
    _transformed_bounds,
)


def test_orientation_arrows_start_at_distinct_scaled_cube_vertices() -> None:
    half = 0.5 * VIEWPORT_ORIENTATION_CUBE_SCALE
    expected_vertices = (
        (half, -half, -half),
        (-half, half, -half),
        (-half, -half, half),
    )

    assert np.allclose(VIEWPORT_ORIENTATION_AXIS_ORIGINS, expected_vertices)


def test_orientation_arrows_match_the_reference_style() -> None:
    arrow = Viewport3D._create_orientation_arrow(
        axis_index=0,
        origin=VIEWPORT_ORIENTATION_AXIS_ORIGINS[0],
        direction=(1.0, 0.0, 0.0),
        color=(1.0, 0.0, 0.0),
        label="X",
    )
    parts = arrow.GetParts()
    line_actor = parts.GetItemAsObject(0)
    line_source = line_actor.GetMapper().GetInputConnection(0, 0).GetProducer()
    expected_shaft_end = (
        VIEWPORT_ORIENTATION_AXIS_ORIGINS[0][0]
        + VIEWPORT_ORIENTATION_AXIS_LENGTH
        * (1.0 - VIEWPORT_ORIENTATION_AXIS_TIP_LENGTH),
        VIEWPORT_ORIENTATION_AXIS_ORIGINS[0][1],
        VIEWPORT_ORIENTATION_AXIS_ORIGINS[0][2],
    )

    assert parts.GetNumberOfItems() == 3
    assert np.allclose(line_source.GetPoint1(), VIEWPORT_ORIENTATION_AXIS_ORIGINS[0])
    assert np.allclose(line_source.GetPoint2(), expected_shaft_end)
    assert np.isclose(line_actor.GetProperty().GetLineWidth(), VIEWPORT_ORIENTATION_AXIS_LINE_WIDTH)
    cone_actor = parts.GetItemAsObject(1)
    cone_source = cone_actor.GetMapper().GetInputConnection(0, 0).GetProducer()
    assert cone_source.GetResolution() == VIEWPORT_ORIENTATION_AXIS_CONE_RESOLUTION
    caption = parts.GetItemAsObject(2)
    assert caption.GetCaption() == "X"
    assert caption.GetCaptionTextProperty().GetShadow() == 0


def test_every_orientation_arrow_line_starts_at_its_cube_vertex() -> None:
    arrows = Viewport3D._create_orientation_axes().GetParts()

    for index, expected_origin in enumerate(VIEWPORT_ORIENTATION_AXIS_ORIGINS):
        arrow = arrows.GetItemAsObject(index).GetParts()
        line_actor = arrow.GetItemAsObject(0)
        line_source = line_actor.GetMapper().GetInputConnection(0, 0).GetProducer()

        assert np.allclose(line_source.GetPoint1(), expected_origin)


def test_orientation_cube_disables_face_text_edges() -> None:
    cube = Viewport3D._configure_orientation_cube(vtk.vtkAnnotatedCubeActor())

    assert cube.GetTextEdgesVisibility() == 0
    assert np.allclose(
        cube.GetAssembly().GetScale(),
        (VIEWPORT_ORIENTATION_CUBE_SCALE,) * 3,
    )


def test_orientation_bounds_proxy_keeps_rotation_center_at_cube_origin() -> None:
    proxy = Viewport3D._create_orientation_bounds_proxy()
    half_extent = 0.5 * VIEWPORT_ORIENTATION_CUBE_SCALE + VIEWPORT_ORIENTATION_AXIS_LENGTH

    assert np.allclose(
        proxy.GetBounds(),
        (-half_extent, half_extent, -half_extent, half_extent, -half_extent, half_extent),
        atol=1e-6,
    )
    assert proxy.GetProperty().GetOpacity() == 0.0
    assert proxy.GetPickable() == 0


def test_camera_up_vector_stays_orthogonal_to_view_direction() -> None:
    direction = np.array([1.0, -2.0, -3.0])

    up = _camera_up_vector(direction)

    assert np.isclose(np.linalg.norm(up), 1.0)
    assert np.isclose(np.dot(up, direction), 0.0)
    assert np.dot(up, [0.0, 0.0, 1.0]) > 0.0


def test_camera_up_vector_uses_stable_fallback_when_looking_along_z() -> None:
    up = _camera_up_vector(np.array([0.0, 0.0, -1.0]))

    assert np.allclose(up, [0.0, 1.0, 0.0])


def test_camera_slerp_follows_a_smooth_shortest_arc() -> None:
    halfway = _slerp_unit_vectors(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        0.5,
    )

    assert np.allclose(halfway, np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0))
    assert np.isclose(np.linalg.norm(halfway), 1.0)


def test_camera_pose_interpolation_keeps_an_orbit_radius() -> None:
    start = (
        np.array([10.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        12.0,
    )
    target = (
        np.array([0.0, 10.0, 0.0]),
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        20.0,
    )

    position, focal_point, up, parallel_scale = _interpolate_camera_pose(start, target, 0.5)

    assert np.allclose(focal_point, np.zeros(3))
    assert np.isclose(np.linalg.norm(position - focal_point), 10.0)
    assert np.allclose(position, np.array([1.0, 1.0, 0.0]) * (10.0 / np.sqrt(2.0)))
    assert np.allclose(up, np.array([0.0, 0.0, 1.0]))
    assert np.isclose(parallel_scale, 16.0)


def test_transformed_bounds_follow_model_matrix() -> None:
    bounds = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]])
    matrix = tf.translation_matrix([4.0, 5.0, 6.0]) @ np.diag([2.0, 3.0, 4.0, 1.0])

    minimum, maximum = _transformed_bounds(bounds, matrix)

    assert np.allclose(minimum, [2.0, -1.0, -6.0])
    assert np.allclose(maximum, [6.0, 11.0, 18.0])


def test_uniform_box_scale_keeps_rotation_and_proportions() -> None:
    rotation = tf.euler_matrix(0.2, 0.3, 0.4, axes="sxyz")
    baseline = tf.translation_matrix([1.0, 2.0, 3.0]) @ rotation @ np.diag([2.0, 3.0, 4.0, 1.0])
    candidate = tf.translation_matrix([1.5, 2.0, 3.0]) @ rotation @ np.diag([2.5, 3.0, 4.0, 1.0])

    result = _scale_matrix_from_box(candidate, baseline, uniform=True)
    scale, shear, angles, translation, _ = tf.decompose_matrix(result)

    assert np.allclose(scale, [2.5, 3.75, 5.0])
    assert np.allclose(shear, 0.0, atol=1e-9)
    assert np.allclose(angles, [0.2, 0.3, 0.4])
    assert np.allclose(translation, [1.5, 2.0, 3.0])


def test_box_scale_does_not_accept_inverted_model() -> None:
    baseline = np.eye(4)
    candidate = np.diag([-1.0, 1.0, 1.0, 1.0])

    result = _scale_matrix_from_box(candidate, baseline, uniform=False)

    assert np.allclose(result, baseline)
