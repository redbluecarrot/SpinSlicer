from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from optics import optimization as optics
from optics.ray_tracing import refract_rays, trace_cuvette, trace_immersion_vial


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


def test_square_chord_and_transpose_dot_product():
    a, at = matrix(mode=1)
    assert float(a[16].sum()) == pytest.approx(46.0, abs=2e-5)
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
    for tracer, extra in ((trace_cuvette, ()), (trace_immersion_vial, (1.0,))):
        p, v, hit = tracer(positions, 0.0, 50.0, 50.0, 1.0, 1.0, *extra, 1.0)
        np.testing.assert_array_equal(hit, [False, True, False])
        assert np.isfinite(p).all() and np.isfinite(v).all()


def test_rotated_square_is_not_clipped_to_inscribed_circle():
    angle = np.pi / 4
    a, _ = matrix((angle,), mode=1)
    assert float(a[16].sum()) == pytest.approx(46 * np.sqrt(2), abs=2e-5)


def test_all_four_square_corner_patches_receive_rays_at_every_view():
    angles = np.deg2rad(np.arange(0, 360, 5))
    a, _ = optics.build_system_matrix(
        angles, 128, 128, 50., 50., 46., n_air=1., n_glass=1., n_resin=1., container_mode=1,
    )
    centres = (np.arange(128) + 0.5) * 50 / 128 - 25
    x, y = np.meshgrid(centres, centres, indexing="ij")
    for sign_x, sign_y in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        target = ((x * sign_x > 20) & (x * sign_x < 22)
                  & (y * sign_y > 20) & (y * sign_y < 22))
        projections = (a @ target.ravel()).reshape(len(angles), 128)
        assert (projections.max(axis=1) > 0).all(), (sign_x, sign_y)


def test_square_chords_match_independent_slab_reference_across_full_detector():
    angles = np.array([0., 0.17, np.pi/4, 1.2, np.pi/2, 2.1])
    a, _ = optics.build_system_matrix(
        angles, 129, 64, 50., 50., 46., n_air=1., n_glass=1., n_resin=1., container_mode=1,
    )
    field = 50 * np.sqrt(2)
    positions = (np.arange(129) + 0.5) * field / 129 - field/2
    expected = []
    for theta in angles:
        normal = np.array([np.cos(theta), -np.sin(theta)])
        direction = np.array([-np.sin(theta), -np.cos(theta)])
        for position in positions:
            point = position * normal
            entry, exit_ = -np.inf, np.inf
            for axis in (0, 1):
                if abs(direction[axis]) < 1e-12:
                    if abs(point[axis]) > 23:
                        exit_ = -np.inf
                else:
                    bounds = sorted(((-23-point[axis])/direction[axis], (23-point[axis])/direction[axis]))
                    entry, exit_ = max(entry, bounds[0]), min(exit_, bounds[1])
            expected.append(max(0., exit_-entry))
    np.testing.assert_allclose(a @ np.ones(64**2), expected, atol=2e-5)


def test_linear_sirt_step_matches_formula_and_reduces_weighted_residual():
    rng = np.random.default_rng(16)
    a = sp.csr_matrix(rng.uniform(0.1, 1, (8, 6)).astype(np.float32))
    at = a.T.tocsr()
    target = at @ rng.random((8, 3), dtype=np.float32)
    row_norm = 1 / np.asarray(a.sum(axis=1)).ravel()
    voxel_norm = 1 / np.asarray(at.sum(axis=1)).ravel()
    values = np.zeros((8, 3), dtype=np.float32)
    expected = row_norm[:, None] * (a @ (voxel_norm[:, None] * target))
    workspace = np.empty_like(target)
    hits = np.ones_like(values, dtype=bool)
    before = float(np.sum(voxel_norm[:, None] * (target - at @ values)**2))
    for step in range(10):
        optics._python_sirt_step(a, at, values, target, row_norm, voxel_norm, hits, workspace)
        if step == 0:
            np.testing.assert_allclose(values, expected, rtol=1e-6)
        after = float(np.sum(voxel_norm[:, None] * (target - at @ values)**2))
        assert after <= before + 1e-6
        before = after


def volume_result(target, **kwargs):
    return optics.optimize_projections_sirt_volume(
        target, np.linspace(0, 2 * np.pi, 32, endpoint=False),
        L_out=50.0, L_in=46.0, grid_res=target.shape[0], grid_span_mm=50.0,
        n_air=1.0, n_glass=1.0, n_liquid=1.0, n_resin=1.0,
        iterations=4, **kwargs,
    )


def test_native_and_python_solver_and_z_batches_agree(monkeypatch):
    library = optics._load_cpp_backend()
    target = np.zeros((24, 24, 4), dtype=np.float32)
    target[7:13, 8:15, 0] = 1
    target[5:9, 12:18, 2] = 0.4
    monkeypatch.setattr(optics, "_load_cpp_backend", lambda: None)
    python = volume_result(target, batch_layers=1)
    np.testing.assert_allclose(python, volume_result(target, batch_layers=4), atol=2e-6)
    assert not python[1].any() and not python[3].any()
    if library is None:
        pytest.skip("Native DLL/SO not built; Python batch invariance was checked.")
    monkeypatch.setattr(optics, "_load_cpp_backend", lambda: library)
    np.testing.assert_allclose(python, volume_result(target, batch_layers=4), atol=3e-6)


def test_sparse_feature_is_not_discarded_or_overdriven_by_percentile_normalization():
    target = np.zeros((16, 16, 128), dtype=np.float32)
    target[9, 10, 63] = 1
    result = volume_result(target)
    assert np.isfinite(result).all()
    assert float(result.max()) == pytest.approx(1.0)
    assert result[63].any()
    assert not result[:63].any() and not result[64:].any()


def test_cancelled_cache_hit_does_not_start_work():
    from slicing_engine import SliceCancelled
    matrix()
    with pytest.raises(SliceCancelled):
        optics.build_system_matrix(np.array([0.0]), 33, 32, 50., 50., 46., cancelled_cb=lambda: True)


@pytest.mark.parametrize("key,value", [("alpha", -1), ("n_resin", 0), ("L_in", 70), ("grid_span_mm", 20)])
def test_invalid_optics_are_rejected_before_allocation(key, value):
    parameters = dict(num_s=32, grid_res=32, grid_span_mm=50., L_out=50., L_in=46.)
    parameters[key] = value
    with pytest.raises(ValueError):
        optics.build_system_matrix(np.array([0.0]), **parameters)
