from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt6.QtWidgets import QApplication, QCheckBox, QLabel

from constants import (
    VIEWPORT_ORIENTATION_AXIS_LENGTH,
    VIEWPORT_ORIENTATION_CUBE_SCALE,
    VIEWPORT_ORIENTATION_MARKER_EDGE_MARGIN,
    VIEWPORT_ORIENTATION_MARKER_PIXELS,
    VIEWPORT_ORIENTATION_MARKER_TOP_OFFSET,
    VIEWPORT_ORIENTATION_WIDGET_SIZE,
)
from SpinSlicer import CALSlicerMainWindow
from viewport import CAMERA_NAVIGATION_BINDINGS, _orientation_marker_viewport


def _viewport_pixel_bounds(viewport: tuple[float, float, float, float], width: int, height: int):
    left, bottom, right, top = viewport
    return (
        round(left * width),
        round((1.0 - top) * height),
        round(right * width),
        round((1.0 - bottom) * height),
    )


def test_camera_controls_replace_the_vat_sidebar() -> None:
    app = QApplication.instance() or QApplication([])
    window = CALSlicerMainWindow()
    try:
        slicer = window.slicer_tab
        labels = [label.text() for label in slicer.findChildren(QLabel)]

        assert not window.view_tabs.tabBar().drawBase()
        assert window.settings_tab.isAncestorOf(window.settings_tab.vat_controls)
        assert not slicer.isAncestorOf(window.settings_tab.vat_controls)
        viewport = slicer._viewport
        assert viewport._orientation_widget is not None
        assert viewport._camera_home_button.parent() is viewport.plotter.interactor
        assert not viewport._camera_home_button.isHidden()
        assert not viewport._camera_home_button.icon().isNull()
        assert viewport._camera_home_button.toolTip() == "Домашнее положение камеры"
        assert VIEWPORT_ORIENTATION_CUBE_SCALE == 1.8
        assert VIEWPORT_ORIENTATION_AXIS_LENGTH == 0.9
        assert not any("Инструмент:" in text for text in labels)
        assert not any("Пропорционально" in text for text in labels)
        assert not any("ЛКМ — вращение" in text for text in labels)
        assert not slicer.findChildren(QCheckBox)

        interactor = viewport.plotter.interactor
        interactor.resize(1000, 640)
        app.processEvents()
        viewport._update_orientation_widget_viewport()
        viewport._position_camera_home_button()

        marker_rect = viewport._orientation_screen_rect()
        if marker_rect is not None:
            assert marker_rect.width() == VIEWPORT_ORIENTATION_MARKER_PIXELS
            assert marker_rect.height() == VIEWPORT_ORIENTATION_MARKER_PIXELS
            assert marker_rect.top() == VIEWPORT_ORIENTATION_MARKER_TOP_OFFSET
            assert marker_rect.x() + marker_rect.width() == (
                interactor.width() - VIEWPORT_ORIENTATION_MARKER_EDGE_MARGIN
            )
        else:
            assert viewport._orientation_cube.GetSize() == (
                VIEWPORT_ORIENTATION_WIDGET_SIZE,
                VIEWPORT_ORIENTATION_WIDGET_SIZE,
            )
        assert viewport._camera_home_button.x() == (
            interactor.width() - viewport._camera_home_button.width()
            - VIEWPORT_ORIENTATION_MARKER_EDGE_MARGIN
        )
        assert viewport._camera_home_button.y() == 8
    finally:
        window.close()
        app.processEvents()


def test_middle_button_uses_the_same_pan_action_as_right_button() -> None:
    assert CAMERA_NAVIGATION_BINDINGS["middle"] == CAMERA_NAVIGATION_BINDINGS["right"] == "pan"


def test_orientation_marker_size_stays_fixed_when_viewport_height_changes() -> None:
    first = _orientation_marker_viewport(1600, 900)
    second = _orientation_marker_viewport(1600, 640)

    assert _viewport_pixel_bounds(first, 1600, 900) == (
        1600 - VIEWPORT_ORIENTATION_MARKER_EDGE_MARGIN - VIEWPORT_ORIENTATION_MARKER_PIXELS,
        VIEWPORT_ORIENTATION_MARKER_TOP_OFFSET,
        1600 - VIEWPORT_ORIENTATION_MARKER_EDGE_MARGIN,
        VIEWPORT_ORIENTATION_MARKER_TOP_OFFSET + VIEWPORT_ORIENTATION_MARKER_PIXELS,
    )
    assert _viewport_pixel_bounds(second, 1600, 640) == _viewport_pixel_bounds(first, 1600, 900)
