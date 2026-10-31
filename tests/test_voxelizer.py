import numpy as np
import trimesh

from voxelizer import voxelize

SPAN, N = 40.0, 80
PIX = (SPAN / N) ** 2


def _layers(mesh, z):
    return voxelize(mesh, np.atleast_1d(z), N, SPAN, sub_y=8)


def test_rotated_box_area_is_exact():
    box = trimesh.creation.box(extents=(12, 7, 10))
    box.apply_transform(trimesh.transformations.rotation_matrix(0.4, [0, 0, 1]))
    result = _layers(box, [0.0, 2.0])
    assert not result.open_layers.size
    np.testing.assert_allclose(result.volume.sum(axis=(0, 1)) * PIX, 84.0, rtol=2e-3)


def test_cavity_stays_empty_and_overlapping_shells_form_a_union():
    outer = trimesh.creation.cylinder(10, 10, sections=128)
    bore = trimesh.creation.cylinder(4, 10, sections=128)
    bore.invert()                                     # cavity: normals point into the hole
    ring = trimesh.util.concatenate([outer, bore])
    ring_area = _layers(ring, 0.0).volume.sum() * PIX
    polygon_area = 0.5 * 128 * np.sin(2 * np.pi / 128) * (10**2 - 4**2)
    assert abs(ring_area - polygon_area) < 0.01 * polygon_area
    a = trimesh.creation.box(extents=(10, 10, 4))
    b = trimesh.creation.box(extents=(10, 10, 4)).apply_translation([5, 0, 0])
    union = _layers(trimesh.util.concatenate([a, b]), 0.0).volume
    assert union.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(union.sum() * PIX, 150.0, rtol=2e-3)


def test_axes_follow_the_slicer_convention():
    box = trimesh.creation.box(extents=(4, 4, 4)).apply_translation([10, -6, 0])
    vol = _layers(box, 0.0).volume[:, :, 0]
    i, j = np.unravel_index(np.argmax(vol), vol.shape)
    centre = (np.array([i, j]) + 0.5) * SPAN / N - SPAN / 2
    assert abs(centre[0] - 10) < 2.1 and abs(centre[1] + 6) < 2.1


def test_open_mesh_layers_are_reported():
    cylinder = trimesh.creation.cylinder(8, 10, sections=64)
    holed = cylinder.copy()
    holed.update_faces(np.arange(len(holed.faces)) != 70)
    z = np.linspace(-4, 4, 9)
    assert not _layers(cylinder, z).open_layers.size
    assert _layers(holed, z).open_layers.size
