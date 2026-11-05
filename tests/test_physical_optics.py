from __future__ import annotations

import numpy as np
import pytest

from spinslicer.optics import system_matrix as optics
from spinslicer.optics.ray_tracing import refract_rays, trace_immersion_vial


def matrix(angles=(0.0,), alpha=0.0, mode=0, n_resin=1.0):
    return optics.build_system_matrix(
        np.asarray(angles), num_s=33, grid_res=32, grid_span_mm=50.0,
        L_out=50.0, L_in=46.0, n_air=1.0, n_glass=1.0, n_liquid=1.0,
        n_resin=n_resin, alpha=alpha, container_mode=mode,
    )


def test_uniform_disk_integral_matches_analytic_chord_at_every_angle():
    angles = (0.0, 0.3, 1.1, np.pi)
    a, _ = matrix(angles)
    positions = (np.arange(33) + 0.5) * 50.0 / 33 - 25.0
    lengths = 2 * np.sqrt(np.maximum(23.0**2 - positions**2, 0.0))
    np.testing.assert_allclose(a @ np.ones(32**2), np.tile(lengths, len(angles)), atol=2e-5)


def test_rays_cover_both_halves_and_use_volume_xy_order():
    a, _ = matrix()
    target = np.zeros((32, 32))
    target[22, 10] = 1.0
    projection = a @ target.ravel()
    assert projection.max() > 0.0
    assert np.argmax(projection) > 16  # positive X; negative Y must be reachable
    target[:, :] = 0
    target[10, 22] = 1.0
    assert np.argmax(a @ target.ravel()) < 16


def test_beer_lambert_integral_is_from_resin_entry_not_grid_boundary():
    alpha = 0.07
    a, _ = matrix(alpha=alpha)
    expected = -np.expm1(-alpha * 46.0) / alpha
    assert float(a[16].sum()) == pytest.approx(expected, rel=2e-6)


def test_transpose_is_the_exact_adjoint():
    a, at = matrix(angles=(0.0, 0.7, 2.0), alpha=0.03, n_resin=1.5)
    rng = np.random.default_rng(72)
    x, b = rng.random(a.shape[1]), rng.random(a.shape[0])
    assert np.dot(a @ x, b) == pytest.approx(np.dot(x, at @ b), rel=1e-6)


def test_vial_in_air_actually_refracts():
    straight, _ = matrix()
    refracted, _ = matrix(n_resin=1.5)
    assert (straight != refracted).nnz > 0


def test_small_absorption_changes_are_not_lost_in_cache():
    a, _ = matrix(alpha=0.0)
    attenuated, _ = matrix(alpha=1e-5)
    assert a is not attenuated
    assert float(attenuated[16].sum()) < float(a[16].sum())


def test_snell_invariant_and_total_internal_reflection():
    angles = np.deg2rad([0, 20, 50])
    incident = np.column_stack([np.sin(angles), -np.cos(angles)])
    normals = np.tile([0.0, 1.0], (3, 1))
    transmitted, valid = refract_rays(incident, normals, 1.0, 1.5)
    assert valid.all()
    np.testing.assert_allclose(1.5 * transmitted[:, 0], incident[:, 0], atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(transmitted, axis=1), 1, atol=1e-14)
    transmitted, valid = refract_rays(incident, normals, 1.5, 1.0)
    assert not valid[-1]
    np.testing.assert_array_equal(transmitted[-1], 0)


def test_missed_rays_are_finite_and_zero_thickness_wall_is_supported():
    positions = np.array([-30.0, 0.0, 30.0])
    p, v, hit = trace_immersion_vial(positions, 0.0, 50.0, 50.0, 1.0, 1.0, 1.0, 1.0)
    np.testing.assert_array_equal(hit, [False, True, False])
    assert np.isfinite(p).all() and np.isfinite(v).all()


def test_cancelled_cache_hit_does_not_start_work():
    from spinslicer.slicing_engine import SliceCancelled
    matrix()
    with pytest.raises(SliceCancelled):
        optics.build_system_matrix(np.array([0.0]), 33, 32, 50., 50., 46., cancelled_cb=lambda: True)


@pytest.mark.parametrize("key,value", [("alpha", -1), ("n_resin", 0), ("L_in", 70), ("grid_span_mm", 20),
                                       ("container_mode", 1)])
def test_invalid_optics_are_rejected_before_allocation(key, value):
    parameters = dict(num_s=32, grid_res=32, grid_span_mm=50., L_out=50., L_in=46.)
    parameters[key] = value
    with pytest.raises(ValueError):
        optics.build_system_matrix(np.array([0.0]), **parameters)
