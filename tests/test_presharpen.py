"""Projector row blur: model and pre-compensation."""
import numpy as np
import pytest

from spinslicer.optics.presharpen import blur_rows, blurred_rows_getter, presharpen_rows, row_sigma


def _frames() -> np.ndarray:
    rng = np.random.default_rng(3)
    rows = np.full((60, 32, 12), 0.2, np.float32)          # lit background, as in a real frame
    rows[10:30] = 0.6 + 0.2 * rng.random((1, 32, 12))      # a block that starts and ends abruptly
    rows[30:45] = 0.3
    return rows


def test_presharpened_frames_blur_back_into_the_plan():
    plan = _frames()
    sigma = 1.5
    shown_plain = blur_rows(plan, sigma)
    shown_sharp = blur_rows(presharpen_rows(plan, sigma), sigma)
    err_plain = np.abs(shown_plain - plan).mean()
    err_sharp = np.abs(shown_sharp - plan).mean()
    assert err_sharp < 0.85 * err_plain
    assert presharpen_rows(plan, sigma).min() >= 0


def test_no_blur_changes_nothing_and_in_place_works():
    plan = _frames()
    np.testing.assert_array_equal(presharpen_rows(plan, 0.0), plan)
    store = plan.astype(np.float16)
    presharpen_rows(store, 1.0, out=store)
    assert store.dtype == np.float16 and np.isfinite(store).all()
    assert row_sigma(0.15, 0.075) == 2.0 and row_sigma(0.0, 0.1) == 0.0


def test_blurred_getter_matches_the_blur_model():
    plan = _frames()
    getter = blurred_rows_getter(lambda rows: plan[rows], plan.shape[0], 1.2)
    rows = [0, 5, 20, 59]
    np.testing.assert_allclose(getter(rows), blur_rows(plan, 1.2)[rows], atol=2e-3)


def test_presharpen_uses_the_exact_adjoint_of_the_clamped_blur():
    """With clamped edges the blur is not symmetric; the gradient needs its true transpose."""
    from spinslicer.optics.presharpen import _blur_last, _blur_last_adjoint, _lipschitz

    rng = np.random.default_rng(5)
    for sigma in (0.4, 1.5, 3.0):
        x, y = rng.random((4, 50)), rng.random((4, 50))
        assert np.vdot(_blur_last(x, sigma), y) == pytest.approx(np.vdot(x, _blur_last_adjoint(y, sigma)),
                                                                 rel=1e-12)
        dense = _blur_last(np.eye(50), sigma)
        assert np.linalg.norm(dense, 2) ** 2 <= _lipschitz(50, sigma) <= 1.02 * np.linalg.norm(dense, 2) ** 2
