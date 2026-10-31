"""FastCAL is checked against the exact sparse Siddon operator, not against itself."""
import numpy as np
import pytest

from optics.fastcal import (
    CirculantVialProjector,
    FastVialEngine,
    cartesian_to_polar_matrix,
    optimize_projections_fast,
    supports,
)
from optics.optimization import build_system_matrix, optimize_projections_sirt_volume

N, NT, SPAN, L_OUT, L_IN = 64, 72, 60.0, 60.0, 56.0


def _target():
    c = (np.arange(N) + 0.5) * SPAN / N - SPAN / 2
    X, Y = np.meshgrid(c, c, indexing="ij")
    ring = ((X - 8) ** 2 + (Y + 4) ** 2 < 12**2) & ((X - 12) ** 2 + (Y + 2) ** 2 > 4**2)
    bar = (abs(X + 14) < 3) & (abs(Y - 10) < 8)
    return (ring | bar).astype(np.float32)


@pytest.mark.parametrize("mode", [0, 2])
@pytest.mark.parametrize("alpha", [0.0, 0.03])
@pytest.mark.parametrize("offset_deg,direction", [(0.0, 1), (90.0, 1), (33.0, -1)])
def test_projector_matches_exact_sparse_operator(mode, alpha, offset_deg, direction):
    angles = (np.deg2rad(offset_deg) + direction * np.linspace(0, 2 * np.pi, NT, endpoint=False)) % (2 * np.pi)
    A, _ = build_system_matrix(angles, N, N, SPAN, L_OUT, L_IN, 1.0, 1.51, 1.45, 1.50, alpha, mode)
    target = _target().ravel()
    reference = (A @ target).reshape(NT, N)
    op = CirculantVialProjector(NT, N, L_OUT, L_IN, 1.0, 1.51, 1.45, 1.50, alpha, mode,
                                pitch=SPAN / N, angle0=float(angles[0]), direction=direction)
    G = cartesian_to_polar_matrix(op.grid, N, SPAN)
    projected = op.forward(np.asarray(G @ target).reshape(NT, op.grid.Q, 1))[:, :, 0]
    # Different discretisations of the same physics: polar vs pixel basis.
    assert np.linalg.norm(projected - reference) / np.linalg.norm(reference) < 0.04


def test_adjoint_is_exact():
    op = CirculantVialProjector(NT, N, L_OUT, L_IN, 1.0, 1.51, 1.45, 1.50, 0.02, 2, pitch=SPAN / N)
    rng = np.random.default_rng(1)
    x = rng.random((NT, op.grid.Q, 3)).astype(np.float32)
    y = rng.random((NT, N, 3)).astype(np.float32)
    lhs = float(np.vdot(op.forward(x).astype(np.float64), y))
    rhs = float(np.vdot(x.astype(np.float64), op.adjoint(y)))
    assert abs(lhs - rhs) <= 1e-4 * abs(lhs)


def test_engine_beats_sirt_on_thresholded_print_metric():
    angles = np.linspace(0, 2 * np.pi, NT, endpoint=False)
    target = np.repeat(_target()[:, :, None], 6, axis=2)
    kwargs = dict(L_out=L_OUT, L_in=L_IN, n_air=1.0, n_glass=1.51, n_liquid=1.45, n_resin=1.50,
                  alpha=0.0, grid_res=N, grid_span_mm=SPAN, container_mode=2)
    fast = optimize_projections_fast(target, angles, iterations=15, **kwargs)
    sirt = optimize_projections_sirt_volume(target, angles, iterations=8, **kwargs)
    assert fast.shape == sirt.shape == (6, N, NT)
    assert np.isfinite(fast).all() and fast.min() >= 0 and np.isclose(fast.max(), 1)
    _, AT = build_system_matrix(angles, N, N, SPAN, L_OUT, L_IN, 1.0, 1.51, 1.45, 1.50, 0.0, 2)

    def best_iou(sino):
        dose = AT @ sino[0].T.ravel()
        part = target[:, :, 0].ravel() >= 0.5
        return max(np.count_nonzero((dose >= t) & part) / np.count_nonzero((dose >= t) | part)
                   for t in np.linspace(dose.max() * 0.05, dose.max(), 200))

    assert best_iou(fast) > best_iou(sirt)
    assert best_iou(fast) > 0.95


def test_angular_smoothing_removes_flicker_without_losing_the_part():
    angles = np.linspace(0, 2 * np.pi, NT, endpoint=False)
    engine = FastVialEngine(angles, N, N, SPAN, L_OUT, L_IN, container_mode=2, multigrid=False)
    target = np.repeat(_target()[:, :, None], 2, axis=2)
    _, AT = build_system_matrix(angles, N, N, SPAN, L_OUT, L_IN, 1.0, 1.51, 1.45, 1.50, 0.0, 2)
    part = target[:, :, 0].ravel() >= 0.5

    def flicker_and_iou(sino):
        frames = sino[0]                                   # [detector, angle]
        lit = frames > 0.02
        flicker = np.abs(np.roll(frames, -1, axis=1) - frames)[lit].mean() / frames[lit].mean()
        dose = AT @ frames.T.ravel()
        iou = max(np.count_nonzero((dose >= t) & part) / np.count_nonzero((dose >= t) | part)
                  for t in np.linspace(dose.max() * 0.05, dose.max(), 200))
        return flicker, iou

    rough = flicker_and_iou(engine.optimize_volume(target, smooth_theta=0.0))
    smooth = flicker_and_iou(engine.optimize_volume(target))       # default smoothing
    assert smooth[0] < 0.6 * rough[0]
    assert smooth[1] > rough[1] - 0.01


def test_rejects_unsupported_geometry():
    uniform = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    assert supports(0, uniform) and supports(2, uniform)
    assert not supports(1, uniform)
    assert not supports(0, uniform[[0, 1, 2, 4, 5, 6, 7, 3]])
    with pytest.raises(ValueError):
        FastVialEngine(uniform[:-1] ** 1.1, 16, 16, 60.0, 60.0, 56.0)


# ------------------------------------------------------------------- physics
from optics.fastcal import (  # noqa: E402
    LEGACY_OPTICS,
    OpticsOptions,
    fresnel_transmission,
    trace_lab_rays,
)


def test_fresnel_transmission_reference_values():
    n1, n2 = 1.0, 1.5
    assert fresnel_transmission(np.array([1.0]), n1, n2)[0] == pytest.approx(1 - ((n2 - n1) / (n2 + n1)) ** 2)
    brewster = np.arctan(n2 / n1)
    ci, ct = np.cos(brewster), np.sqrt(1 - (n1 / n2 * np.sin(brewster)) ** 2)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    assert fresnel_transmission(np.array([ci]), n1, n2)[0] == pytest.approx(1 - rs**2 / 2)
    assert fresnel_transmission(np.array([0.1]), 1.5, 1.0)[0] == 0.0    # total internal reflection


def test_divergent_rays_pass_through_their_axis_plane_points():
    s = np.linspace(-20, 20, 9)
    d = 300.0
    dirs = np.column_stack([s, np.full_like(s, -d)])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.column_stack([s * (d - 120) / d, np.full_like(s, 120.0)])
    p, v, ok, T = trace_lab_rays(origins, dirs, 60.0, 56.0, 1.0, 1.0, 1.0, fresnel=True)
    assert ok.all() and np.allclose(T, 1.0)
    t = -p[:, 1] / v[:, 1]                       # walk to y = 0
    np.testing.assert_allclose(p[:, 0] + t * v[:, 0], s, atol=1e-9)


def _forward(op, target):
    G = cartesian_to_polar_matrix(op.grid, N, SPAN)
    return op.forward(np.asarray(G @ target.ravel()).reshape(op.n_theta, op.grid.Q, 1))[:, :, 0]


def test_rotation_smear_is_the_average_of_rotated_bundles():
    steps = 4
    step = 2 * np.pi / NT
    common = dict(pitch=SPAN / N, direction=1)
    smear = CirculantVialProjector(NT, N, L_OUT, L_IN, 1.0, 1.51, 1.45, 1.50, 0.01, 0, angle0=0.3, **common,
                                   optics=OpticsOptions(fresnel=False, pixel_profile="ray", sub_rays=1,
                                                        rotation_smear=True, smear_steps=steps))
    target = _target()
    expected = np.mean([
        _forward(CirculantVialProjector(NT, N, L_OUT, L_IN, 1.0, 1.51, 1.45, 1.50, 0.01, 0,
                                        angle0=0.3 + ((k + 0.5) / steps - 0.5) * step, **common), target)
        for k in range(steps)], axis=0)
    # Same polar grid orientation; differences come only from resampling the target.
    got = _forward(smear, target)
    assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 0.02


def test_linear_pixel_profile_equals_hat_weighted_fine_rays():
    S = 4
    hat = CirculantVialProjector(NT, N, L_OUT, L_IN, 1.0, 1.51, 1.45, 1.50, 0.0, 2, pitch=SPAN / N,
                                 optics=OpticsOptions(fresnel=False, pixel_profile="linear", sub_rays=S,
                                                      rotation_smear=False))
    fine = CirculantVialProjector(NT, N * S, L_OUT, L_IN, 1.0, 1.51, 1.45, 1.50, 0.0, 2, pitch=SPAN / N,
                                  optics=LEGACY_OPTICS)
    assert fine.grid.Q == hat.grid.Q
    u = (np.arange(N * S) + 0.5) / S - 0.5
    lo = np.floor(u).astype(int)
    W = np.zeros((N, N * S))
    for b, w in ((np.clip(lo, 0, N - 1), 1 - (u - lo)), (np.clip(lo + 1, 0, N - 1), u - lo)):
        np.add.at(W, (b, np.arange(N * S)), w / S)
    target = _target()
    expected = _forward(fine, target) @ W.T
    np.testing.assert_allclose(_forward(hat, target), expected, rtol=1e-4, atol=1e-4 * expected.max())


def test_half_covered_face_layer_is_solved_not_mixed_from_neighbours():
    """A top or bottom face inside a layer (coverage 0.5) between a full and an empty layer
    must get its own frames. The 50/50 mix of the neighbours' frames matches it as a target
    but gives half the dose, and the face layer would not cure (found by tools/engine_judge.py)."""
    angles = np.linspace(0, 2 * np.pi, NT, endpoint=False)
    part = _target()
    layers = [0.0] * 3 + [0.5] + [1.0] * 13 + [0.5] + [0.0] * 3      # a slab with faces mid-layer
    target = np.stack([part * c for c in layers], axis=2).astype(np.float32)
    eng = FastVialEngine(angles, N, N, SPAN, L_OUT, L_IN, 1.0, 1.51, 1.45, 1.50, 0.0, 2)
    P = np.asarray(eng.optimize_volume(target), np.float32)
    assert eng.layer_kind[3] != 2 and eng.layer_kind[17] != 2      # not an interpolated copy
    _, AT = build_system_matrix(angles, N, N, SPAN, L_OUT, L_IN, 1.0, 1.51, 1.45, 1.50, 0.0, 2)
    inside = part.ravel() >= 0.5
    full = AT @ P[10].T.ravel()
    face = AT @ P[3].T.ravel()
    # The face layer is planned at the edge of the band (0.7 + 0.3 × 0.5 = 0.85 of a full layer),
    # not at half dose.
    assert np.median(face[inside]) > 0.75 * np.median(full[inside])


def test_thin_hole_stays_dark_with_thin_feature_awareness():
    """A hole about 1.5 voxels wide has no fully empty cell, so plain coverage puts it at an
    intermediate dose and light from the walls closes it. With thin-feature weights and a
    dark target for thin voids the hole gets clearly less dose than the plate around it."""
    angles = np.linspace(0, 2 * np.pi, NT, endpoint=False)
    sub = 8
    c = ((np.arange(N * sub) + 0.5) / sub) * SPAN / N - SPAN / 2
    X, Y = np.meshgrid(c, c, indexing="ij")
    hole_r = 0.75 * SPAN / N                          # 1.5 voxels across
    plate = (np.abs(X) < 9) & (np.abs(Y) < 9) & ((X - 1.3) ** 2 + (Y + 0.7) ** 2 > hole_r**2)
    cover = plate.reshape(N, sub, N, sub).mean(axis=(1, 3)).astype(np.float32)
    target = np.repeat(cover[:, :, None], 4, axis=2)
    _, AT = build_system_matrix(angles, N, N, SPAN, L_OUT, L_IN, 1.0, 1.51, 1.45, 1.50, 0.0, 2)
    cx = (np.arange(N) + 0.5) * SPAN / N - SPAN / 2
    CX, CY = np.meshgrid(cx, cx, indexing="ij")
    in_hole = ((CX - 1.3) ** 2 + (CY + 0.7) ** 2 < (0.5 * hole_r) ** 2).ravel()
    solid = (cover >= 0.95).ravel()

    def hole_contrast(**kw):
        eng = FastVialEngine(angles, N, N, SPAN, L_OUT, L_IN, 1.0, 1.51, 1.45, 1.50, 0.0, 2)
        P = np.asarray(eng.optimize_volume(target, **kw), np.float32)
        dose = AT @ P[1].T.ravel()
        return float(dose[in_hole].max() / np.median(dose[solid]))

    plain = hole_contrast(thin_boost=0.0, snap_void=False)
    aware = hole_contrast()                            # defaults: thin features on
    # Measured: 0.840 without, 0.807 with (hole peak / plate median). The cure level sits
    # near 0.85 of the plate, so without it the hole is on the edge of closing.
    assert aware < plain - 0.02
    assert aware < 0.82
