"""The printer sketch follows the real settings, and every panel setting reaches the engine."""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sketch_scene import (  # noqa: E402
    SceneParams,
    _beam_profile,
    bath_lines,
    beam_mesh,
    projector_x,
    silhouette_lines,
    static_lines,
)


def test_vial_outline_scales_with_diameter() -> None:
    small = np.vstack(static_lines(SceneParams(diameter_mm=40)))
    large = np.vstack(static_lines(SceneParams(diameter_mm=80)))
    assert np.ptp(large[:, 1]) > np.ptp(small[:, 1]) * 1.6


def test_bath_is_drawn_only_with_a_bath() -> None:
    assert bath_lines(SceneParams(bath=False)) == []
    assert len(bath_lines(SceneParams(bath=True))) == 12


def test_beam_height_is_the_projector_field() -> None:
    p = SceneParams(field_mm=24, bath=True, n_resin=1.45, n_liquid=1.45)
    _w, h = _beam_profile(p, np.array([projector_x(p)]))
    assert np.isclose(2 * h[0], 24)


def test_refraction_without_bath_narrows_the_sheet() -> None:
    x = np.array([SceneParams().radius])
    _w, h_air = _beam_profile(SceneParams(bath=False, n_resin=1.5), x)
    _w, h_bath = _beam_profile(SceneParams(bath=True, n_resin=1.46, n_liquid=1.45), x)
    assert h_air[0] < h_bath[0]


def test_absorption_fades_the_beam_inside_the_vial() -> None:
    clear = beam_mesh(SceneParams(alpha_per_mm=0.0), "#FF0000", 0.2)
    murky = beam_mesh(SceneParams(alpha_per_mm=0.04), "#FF0000", 0.2)
    assert murky.point_data["rgba"][:, 3].mean() < clear.point_data["rgba"][:, 3].mean()


def test_silhouette_follows_the_camera() -> None:
    p = SceneParams()
    a = np.vstack(silhouette_lines(p, np.array([200.0, 0.0])))
    b = np.vstack(silhouette_lines(p, np.array([0.0, 200.0])))
    assert not np.allclose(a, b)


def test_panel_settings_reach_the_engine(tmp_path) -> None:
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    # Never touch the user's real saved settings.
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(tmp_path))
    from ui_panels import ProcessSettingsPanel

    panel = ProcessSettingsPanel()
    panel.placement.setValue(1)
    assert panel.container_mode_value() == 2
    panel.placement.setValue(0)
    assert panel.container_mode_value() == 0
    panel.rays.setValue(0)
    assert panel.projector_throw.value() == 0.0
    panel.rays.setValue(1)
    panel.throw_ratio.setValue(1.2)
    assert panel.projector_throw.value() > panel.vat_diameter_mm()
    panel.projector_blur.setValue(0.1)
    panel.blur_compensation.setChecked(False)
    assert panel.effective_blur_mm() == 0.0
    panel.blur_compensation.setChecked(True)
    assert np.isclose(panel.effective_blur_mm(), 0.1)
    panel.frames.setValue(720)
    panel.speed.setValue(10)
    assert np.isclose(panel.frame_rate_hz_value(), 20.0)
    panel.field_height.setValue(15)
    assert panel.printable_height_mm() == 15
    panel.diameter.setValue(70)
    assert panel.scene_params().diameter_mm == 70
    # The removed knobs are fixed at their neutral values.
    assert (panel.exposure.value(), panel.intensity.value(), panel.threshold.value()) == (1.0, 100.0, 0.0)
