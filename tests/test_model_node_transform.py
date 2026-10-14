import numpy as np
import trimesh

from model_node import ModelNode


def test_matrix_from_viewport_round_trips_to_model_transform() -> None:
    node = ModelNode(trimesh.creation.box(extents=(10.0, 20.0, 30.0)))
    node.set_size_mm(x=12.0, uniform=False)
    node.set_rotation_deg(x=20.0, y=-15.0, z=35.0)
    node.set_translation_mm(x=4.0, y=-3.0, z=2.0)
    matrix = node.matrix().copy()

    restored = ModelNode(node.original_mesh)
    restored.set_matrix(matrix)

    assert np.allclose(restored.matrix(), matrix, atol=1e-8)


def test_viewport_matrix_can_keep_scale_uniform() -> None:
    node = ModelNode(trimesh.creation.box(extents=(10.0, 20.0, 30.0)))
    matrix = np.diag([2.0, 3.0, 4.0, 1.0])

    node.set_matrix(matrix, uniform=True)

    assert np.allclose(node.transform.scale, [4.0, 4.0, 4.0])


def test_uniform_matrix_resize_preserves_existing_axis_ratios() -> None:
    node = ModelNode(trimesh.creation.box(extents=(10.0, 20.0, 30.0)))
    node.transform.scale = np.array([2.0, 3.0, 4.0])
    candidate = np.diag([2.5, 3.0, 4.0, 1.0])

    node.set_matrix(candidate, uniform=True)

    assert np.allclose(node.transform.scale, [2.5, 3.75, 5.0])
