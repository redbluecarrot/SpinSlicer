from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt6.QtCore import Qt
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
from ui_panels import ProcessSettingsPanel, QuickSettingsSidebar
from viewport import CAMERA_NAVIGATION_BINDINGS, _orientation_marker_viewport


def _viewport_pixel_bounds(viewport: tuple[float, float, float, float], width: int, height: int):
    left, bottom, right, top = viewport
    return (
        round(left * width),
        round((1.0 - top) * height),
        round(right * width),
        round((1.0 - bottom) * height),
    )


def test_quick_settings_sidebar_and_camera_controls() -> None:
    app = QApplication.instance() or QApplication([])
    window = CALSlicerMainWindow()
    try:
        slicer = window.slicer_tab
        labels = [label.text() for label in slicer.findChildren(QLabel)]

        assert not window.view_tabs.tabBar().drawBase()
        assert slicer.quick_settings_sidebar.isAncestorOf(window.settings_tab.vat_controls)
        assert slicer.quick_settings_sidebar.isAncestorOf(window.settings_tab.resin_controls)
        assert slicer.quick_settings_sidebar.isAncestorOf(window.settings_tab.grid_controls)
        assert not window.settings_tab.isAncestorOf(window.settings_tab.vat_controls)
        assert window.settings_tab.isAncestorOf(window.settings_tab.glass_thickness)
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
        uniform_checkbox = slicer._transform_toolbar.scale_controls.uniform_checkbox
        assert isinstance(uniform_checkbox, QCheckBox)
        assert uniform_checkbox.text() == "Равномерное масштабирование"
        assert uniform_checkbox.isChecked()
        assert slicer._transform_toolbar.is_uniform()
        uniform_checkbox.setChecked(False)
        assert not slicer._transform_toolbar.is_uniform()
        assert not viewport._uniform_scale
        uniform_checkbox.setChecked(True)
        assert viewport._uniform_scale

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


def test_quick_settings_sidebar_fits_without_horizontal_scrolling() -> None:
    app = QApplication.instance() or QApplication([])
    settings = ProcessSettingsPanel()
    sidebar = QuickSettingsSidebar(settings)
    try:
        assert (
            sidebar.horizontalScrollBarPolicy()
            == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        for width in (280, 300, 320, 360):
            sidebar.resize(width, 800)
            sidebar.show()
            app.processEvents()

            assert sidebar.horizontalScrollBar().maximum() == 0
            assert sidebar.widget().width() <= sidebar.viewport().width()
            checkbox = settings.preserve_internal_voids
            assert checkbox.sizeHint().width() <= checkbox.parentWidget().contentsRect().width()
            sidebar.hide()
    finally:
        sidebar.close()
        settings.close()


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
