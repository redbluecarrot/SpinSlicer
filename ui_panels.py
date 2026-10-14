"""
ui_panels.py
============
Виджеты настроек и управления моделью.

  ProcessSettingsPanel — настройки принтера и процесса, вынесенные в
    отдельную вкладку приложения, включая опциональную установку VAMToolbox.
  TransformToolbar — компактное управление gizmo над 3D-вьюпортом.
  ScaleControls — компактная Fusion-style панель процентов и размеров.
  ObjectPanel — прежняя расширенная панель объекта, оставленная для обратной
    совместимости с внешними импортами.

Виджеты ничего не знают про PyVista напрямую — связь с ModelNode и сценой
делает SlicerTab.
"""
from __future__ import annotations

from contextlib import ExitStack
from typing import Optional

import numpy as np
from PyQt6.QtCore import QSettings, QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from constants import (
    APP_ORG,
    APP_TITLE,
    DEFAULT_DIAMETER_MM,
    DEFAULT_GRID_RESOLUTION,
    DEFAULT_NUM_FRAMES,
    DEFAULT_OUTPUT_RESOLUTION,
)
from i18n import tr
from model_node import ModelNode
from theme import AXIS_X, AXIS_Y, AXIS_Z
from vam_backend import (
    DEFAULT_VAM_ENV_NAME,
    find_conda_executable,
    is_vamtoolbox_available,
)
from widgets import LabeledSlider


def _scroll_wrap(inner: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(inner)
    return scroll



class NoScrollComboBox(QComboBox):
    def wheelEvent(self, e):
        e.ignore()


class ObjectTransformPanel(QGroupBox):
    positionChanged = pyqtSignal(int, float)
    rotationChanged = pyqtSignal(int, float)
    scaleSizeChanged = pyqtSignal(int, float)
    centerRequested = pyqtSignal()
    autoFitRequested = pyqtSignal()
    resetRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Трансформация модели", parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setHorizontalSpacing(8)

        # Headers
        layout.addWidget(QLabel(""), 0, 0)
        lbl_x = QLabel("X")
        lbl_x.setStyleSheet("color: #FF4444; font-weight: bold;")
        layout.addWidget(lbl_x, 0, 1)
        lbl_y = QLabel("Y")
        lbl_y.setStyleSheet("color: #44FF44; font-weight: bold;")
        layout.addWidget(lbl_y, 0, 2)
        lbl_z = QLabel("Z")
        lbl_z.setStyleSheet("color: #4444FF; font-weight: bold;")
        layout.addWidget(lbl_z, 0, 3)

        # Position
        layout.addWidget(QLabel("Позиция, мм"), 1, 0)
        self.pos_x = self._make_value_field(-1000, 1000)
        layout.addWidget(self.pos_x, 1, 1)
        self.pos_y = self._make_value_field(-1000, 1000)
        layout.addWidget(self.pos_y, 1, 2)
        self.pos_z = self._make_value_field(-1000, 1000)
        layout.addWidget(self.pos_z, 1, 3)

        # Rotation
        layout.addWidget(QLabel("Вращение, °"), 2, 0)
        self.rot_x = self._make_value_field(-360, 360)
        layout.addWidget(self.rot_x, 2, 1)
        self.rot_y = self._make_value_field(-360, 360)
        layout.addWidget(self.rot_y, 2, 2)
        self.rot_z = self._make_value_field(-360, 360)
        layout.addWidget(self.rot_z, 2, 3)

        # Size
        layout.addWidget(QLabel("Размер, мм"), 3, 0)
        self.size_x = self._make_value_field(0.01, 1000)
        layout.addWidget(self.size_x, 3, 1)
        self.size_y = self._make_value_field(0.01, 1000)
        layout.addWidget(self.size_y, 3, 2)
        self.size_z = self._make_value_field(0.01, 1000)
        layout.addWidget(self.size_z, 3, 3)

        # Action Buttons
        btn_row = QHBoxLayout()
        self.center_btn = QPushButton("Центр")
        self.autofit_btn = QPushButton("Авто-фит")
        self.reset_btn = QPushButton("Сброс")
        btn_row.addWidget(self.center_btn)
        btn_row.addWidget(self.autofit_btn)
        btn_row.addWidget(self.reset_btn)
        layout.addLayout(btn_row, 4, 0, 1, 4)

        self._wire_signals()
        self.setEnabled(False)

    @staticmethod
    def _make_value_field(minimum: float, maximum: float) -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setRange(minimum, maximum)
        field.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        return field

    def _wire_signals(self):
        self.pos_x.valueChanged.connect(lambda v: self.positionChanged.emit(0, v))
        self.pos_y.valueChanged.connect(lambda v: self.positionChanged.emit(1, v))
        self.pos_z.valueChanged.connect(lambda v: self.positionChanged.emit(2, v))

        self.rot_x.valueChanged.connect(lambda v: self.rotationChanged.emit(0, v))
        self.rot_y.valueChanged.connect(lambda v: self.rotationChanged.emit(1, v))
        self.rot_z.valueChanged.connect(lambda v: self.rotationChanged.emit(2, v))

        self.size_x.valueChanged.connect(lambda v: self.scaleSizeChanged.emit(0, v))
        self.size_y.valueChanged.connect(lambda v: self.scaleSizeChanged.emit(1, v))
        self.size_z.valueChanged.connect(lambda v: self.scaleSizeChanged.emit(2, v))

        self.center_btn.clicked.connect(self.centerRequested.emit)
        self.autofit_btn.clicked.connect(self.autoFitRequested.emit)
        self.reset_btn.clicked.connect(self.resetRequested.emit)

    def sync_from_model(self, node):
        t = node.transform.translation
        r = node.transform.rotation_deg
        s = node.current_size_mm()
        controls = (
            self.pos_x,
            self.pos_y,
            self.pos_z,
            self.rot_x,
            self.rot_y,
            self.rot_z,
            self.size_x,
            self.size_y,
            self.size_z,
        )
        with ExitStack() as blockers:
            for control in controls:
                blockers.enter_context(QSignalBlocker(control))
            self.pos_x.setValue(t[0])
            self.pos_y.setValue(t[1])
            self.pos_z.setValue(t[2])

            self.rot_x.setValue(r[0])
            self.rot_y.setValue(r[1])
            self.rot_z.setValue(r[2])

            self.size_x.setValue(s[0])
            self.size_y.setValue(s[1])
            self.size_z.setValue(s[2])
        self.setEnabled(True)


class ProcessSettingsPanel(QWidget):
    """Параметры печати, геометрия колбы и оптика."""

    diameterChanged = pyqtSignal(float)
    settingsChanged = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._settings = QSettings(APP_ORG, APP_TITLE)
        self._vam_conda_executable = str(
            self._settings.value("vam/conda_executable", "") or ""
        )
        if not self._vam_conda_executable:
            self._vam_conda_executable = find_conda_executable() or ""
        self._vam_environment_name = str(
            self._settings.value("vam/environment_name", DEFAULT_VAM_ENV_NAME)
            or DEFAULT_VAM_ENV_NAME
        )
        ready_value = self._settings.value("vam/ready", False)
        self._vam_ready = str(ready_value).lower() in {"1", "true", "yes"}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        self._content_layout = layout

        title = QLabel("Дополнительные настройки процесса")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        # Геометрия колбы находится в настройках, рядом с параметрами печати.
        self.vat_controls = QGroupBox("Печатная колба", self)
        self.vat_controls.setObjectName("quickVatSettings")
        vat_layout = QVBoxLayout(self.vat_controls)
        self.container_mode = NoScrollComboBox()
        self.container_mode.addItem("Колба (в воздухе)", 0)
        self.container_mode.addItem("Виала (в иммерсионной ванне)", 2)
        vat_layout.addWidget(QLabel("Оптический режим:"))
        vat_layout.addWidget(self.container_mode)

        self.diameter = LabeledSlider("Диаметр колбы, мм", 20.0, 200.0, DEFAULT_DIAMETER_MM, decimals=1)
        vat_layout.addWidget(self.diameter)
        self.container_mode.currentIndexChanged.connect(lambda: self.diameterChanged.emit(self.diameter.value()))
        layout.addWidget(self.vat_controls)

        # Детальные оптические свойства остаются в Settings.
        optics_box = QGroupBox("Оптические параметры колбы")
        optics_layout = QVBoxLayout(optics_box)
        self.glass_thickness = LabeledSlider("Толщина стенки колбы, мм", 1.0, 10.0, 2.0, decimals=1)
        optics_layout.addWidget(self.glass_thickness)
        self.n_glass = LabeledSlider("Показатель преломления (Стекло)", 1.0, 2.0, 1.51, decimals=2)
        optics_layout.addWidget(self.n_glass)
        self.n_liquid = LabeledSlider("Показатель преломления (Жидкость)", 1.0, 2.0, 1.45, decimals=2)
        optics_layout.addWidget(self.n_liquid)
        self.n_resin = LabeledSlider("Показатель преломления (Смола)", 1.0, 2.0, 1.50, decimals=2)
        optics_layout.addWidget(self.n_resin)
        self.alpha_absorption = LabeledSlider("Затухание (Бугер-Ламберт) 1/мм", 0.0, 0.10, 0.0, decimals=3)
        self.alpha_absorption.setToolTip("Коэффициент оптического затухания света смолой (для CAL обычно 0.000 - 0.030 мм⁻¹)")
        optics_layout.addWidget(self.alpha_absorption)
        layout.addWidget(optics_box)

        # --- Смола -----------------------------------------------------------
        resin_box = QGroupBox("Параметры фотополимера")
        resin_box.setObjectName("quickExposureSettings")
        self.resin_controls = resin_box
        resin_layout = QVBoxLayout(resin_box)
        self.exposure = LabeledSlider("Базовое время засветки, с", 0.1, 20.0, 1.0, decimals=2)
        self.intensity = LabeledSlider("Интенсивность источника, %", 1.0, 200.0, 100.0, decimals=1)
        self.threshold = LabeledSlider("Порог полимеризации, %", 0.0, 100.0, 0.0, decimals=1)
        resin_layout.addWidget(self.exposure)
        resin_layout.addWidget(self.intensity)
        resin_layout.addWidget(self.threshold)
        hint = QLabel("Порог и интенсивность определяют итоговую\nконтрастность и яркость кадров.")
        hint.setObjectName("hintLabel")
        resin_layout.addWidget(hint)
        layout.addWidget(resin_box)

        # --- Сетка и рендер -----------------------------------------------
        grid_box = QGroupBox("Разрешение и кадры")
        grid_box.setObjectName("quickProjectionSettings")
        self.grid_controls = grid_box
        grid_layout = QVBoxLayout(grid_box)
        self.grid_res = LabeledSlider("Разрешение сетки (Voxel Grid)", 64, 256, DEFAULT_GRID_RESOLUTION, decimals=0, step=1)
        self.output_res = LabeledSlider("Разрешение кадра, px", 128, 2048, DEFAULT_OUTPUT_RESOLUTION, decimals=0, step=1)
        self.frames = LabeledSlider("Количество кадров", 30, 720, DEFAULT_NUM_FRAMES, decimals=0, step=1)
        grid_layout.addWidget(self.grid_res)
        grid_layout.addWidget(self.output_res)
        grid_layout.addWidget(self.frames)

        self.fill_holes = QCheckBox("Сплошная заливка (ремонт сетки)")
        self.fill_holes.setChecked(True)
        grid_layout.addWidget(self.fill_holes)
        self.preserve_internal_voids = QCheckBox("Сохранять полости")
        self.preserve_internal_voids.setToolTip(
            "Сохраняет отверстия и резьбу при растеризации STL-сетки."
        )
        # Threaded STL models need their bore and thread relief to survive
        # rasterization; keep the safe behavior enabled for every model.
        self.preserve_internal_voids.setChecked(True)
        grid_layout.addWidget(self.preserve_internal_voids)
        layout.addWidget(grid_box)

        # --- Projection engine --------------------------------------------
        backend_box = QGroupBox("Движок проекций (Собственный C++ SIRT)")
        backend_layout = QVBoxLayout(backend_box)

        self.optimizer_iterations = LabeledSlider(
            "Итерации оптимизатора", 1, 100, 8, decimals=0, step=1,
        )
        backend_layout.addWidget(self.optimizer_iterations)
        backend_hint = QLabel(
            "Встроенный физический движок SIRT на C++ (OpenMP в Linux). "
            "Учитывает преломление по Снеллиусу и поглощение по Бугеру-Ламберту."
        )
        backend_hint.setObjectName("hintLabel")
        backend_hint.setWordWrap(True)
        backend_layout.addWidget(backend_hint)
        layout.addWidget(backend_box)

        layout.addStretch(1)
        content.setMaximumWidth(560)
        scroll = _scroll_wrap(content)
        scroll.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        root.addWidget(scroll)

        self.diameter.valueChanged.connect(self.diameterChanged.emit)
        self.diameter.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.glass_thickness.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.n_glass.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.n_liquid.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.n_resin.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.alpha_absorption.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.exposure.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.intensity.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.threshold.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.fill_holes.toggled.connect(lambda _v: self.settingsChanged.emit())
        self.preserve_internal_voids.toggled.connect(lambda _v: self.settingsChanged.emit())
        self.optimizer_iterations.valueChanged.connect(lambda _v: self.settingsChanged.emit())

    def vat_diameter_mm(self) -> float:
        return self.diameter.value()

    def projection_backend_name(self) -> str:
        return "physical"

    def optimizer_iterations_value(self) -> int:
        return self.optimizer_iterations.value_int()

    def vam_runtime_config(self) -> tuple[str, str]:
        """Return the configured isolated runtime, if it has been verified."""

        if is_vamtoolbox_available():
            return "", ""
        if self._vam_ready and self._vam_conda_executable:
            return self._vam_conda_executable, self._vam_environment_name
        return "", ""

    def container_mode_value(self) -> int:
        return self.container_mode.currentData()

    def glass_thickness_mm(self) -> float:
        return self.glass_thickness.value()

    def n_glass_value(self) -> float:
        return self.n_glass.value()

    def n_liquid_value(self) -> float:
        return self.n_liquid.value()

    def n_resin_value(self) -> float:
        return self.n_resin.value()

    def alpha_absorption_value(self) -> float:
        return self.alpha_absorption.value()


class QuickSettingsSidebar(QScrollArea):
    """Frequently adjusted print settings shown beside the 3D workspace."""

    def __init__(
        self,
        settings_panel: ProcessSettingsPanel,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("quickSettingsSidebar")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(280)
        self.setMaximumWidth(360)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        content = QWidget()
        content.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Быстрые настройки")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        for group in (
            settings_panel.vat_controls,
            settings_panel.resin_controls,
            settings_panel.grid_controls,
        ):
            settings_panel._content_layout.removeWidget(group)
            layout.addWidget(group)

        layout.addStretch(1)
        self.setWidget(content)


class ScaleControls(QFrame):
    """Compact Fusion-style percentage/size controls for the scale gizmo."""

    scalePercentChanged = pyqtSignal(int, float)
    sizeChanged = pyqtSignal(int, float)
    uniformScaleChanged = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("scaleControls")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.setLineWidth(1)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFixedWidth(374)

        layout = QGridLayout(self)
        layout.setContentsMargins(9, 4, 9, 7)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)
        layout.setColumnMinimumWidth(0, 54)

        layout.addWidget(QLabel(""), 0, 0)
        self.axis_labels: list[QLabel] = []
        for column, (axis_name, color) in enumerate(
            (("X", AXIS_X), ("Y", AXIS_Y), ("Z", AXIS_Z)),
            start=1,
        ):
            label = QLabel(axis_name)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"color: {color}; font-weight: 700;")
            self.axis_labels.append(label)
            layout.addWidget(label, 0, column)

        layout.addWidget(QLabel("Масштаб"), 1, 0)
        layout.addWidget(QLabel("Размер"), 2, 0)
        self.percent_fields: list[QDoubleSpinBox] = []
        self.size_fields: list[QDoubleSpinBox] = []
        for axis in range(3):
            percent = self._make_field(0.01, 10000.0, 100.0, 2, " %")
            size = self._make_field(0.01, 100000.0, 1.0, 2, " mm")
            self.percent_fields.append(percent)
            self.size_fields.append(size)
            layout.addWidget(percent, 1, axis + 1)
            layout.addWidget(size, 2, axis + 1)
            percent.valueChanged.connect(
                lambda value, index=axis: self.scalePercentChanged.emit(index, value)
            )
            size.valueChanged.connect(
                lambda value, index=axis: self.sizeChanged.emit(index, value)
            )

        self.uniform_checkbox = QCheckBox("Равномерное масштабирование")
        self.uniform_checkbox.setChecked(True)
        layout.addWidget(self.uniform_checkbox, 3, 0, 1, 4)
        self.uniform_checkbox.toggled.connect(self.uniformScaleChanged.emit)

    @staticmethod
    def _make_field(
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
        suffix: str,
    ) -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        field.setRange(minimum, maximum)
        field.setDecimals(decimals)
        field.setSingleStep(1.0)
        field.setSuffix(suffix)
        field.setFixedWidth(86)
        field.setValue(value)
        return field

    def sync_from_model(self, node: ModelNode) -> None:
        """Update fields without feeding the values back into the model."""

        scale = np.asarray(node.transform.scale, dtype=np.float64)
        size = np.asarray(node.current_size_mm(), dtype=np.float64)
        for axis in range(3):
            with QSignalBlocker(self.percent_fields[axis]):
                self.percent_fields[axis].setValue(max(float(scale[axis] * 100.0), 0.01))
            with QSignalBlocker(self.size_fields[axis]):
                self.size_fields[axis].setValue(max(float(size[axis]), 0.01))

    def set_enabled_state(self, enabled: bool) -> None:
        self.setEnabled(enabled)


class TransformToolbar(QWidget):
    """Compact transform controls displayed above the 3D viewport."""

    modeChanged = pyqtSignal(str)
    centerRequested = pyqtSignal()
    autoFitRequested = pyqtSignal()
    resetRequested = pyqtSignal()
    uniformScaleChanged = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("transformToolbar")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(6)
        layout.addLayout(tool_row)

        self.move_mode_btn = QPushButton("Перемещение")
        self.rotate_mode_btn = QPushButton("Вращение")
        self.scale_mode_btn = QPushButton("Масштаб")
        self._mode_buttons = (
            self.move_mode_btn,
            self.rotate_mode_btn,
            self.scale_mode_btn,
        )
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for index, button in enumerate(self._mode_buttons):
            button.setCheckable(True)
            button.setObjectName("transformModeButton")
            self._mode_group.addButton(button, index)
            tool_row.addWidget(button)
        self.move_mode_btn.setChecked(True)

        tool_row.addStretch(1)

        self.center_btn = QPushButton("Центрировать")
        self.center_btn.setToolTip("Переместить модель в центр XY")
        self.autofit_btn = QPushButton("Авто-фит под колбу")
        self.autofit_btn.setToolTip("Вписать модель в текущий размер колбы")
        self.reset_btn = QPushButton("Сбросить")
        self.reset_btn.setToolTip("Вернуть исходное положение и масштаб модели")
        tool_row.addWidget(self.center_btn)
        tool_row.addWidget(self.autofit_btn)
        tool_row.addWidget(self.reset_btn)

        self.scale_controls = ScaleControls(self)
        self.scale_controls.setVisible(False)
        self.scale_controls.uniformScaleChanged.connect(self.uniformScaleChanged)
        layout.addWidget(self.scale_controls, 0, Qt.AlignmentFlag.AlignLeft)

        self._wire_signals()
        self.set_enabled_state(False)

    def _wire_signals(self) -> None:
        for button, mode in zip(self._mode_buttons, ("move", "rotate", "scale")):
            button.clicked.connect(lambda _checked=False, m=mode: self._on_mode_clicked(m))
        self.center_btn.clicked.connect(self.centerRequested.emit)
        self.autofit_btn.clicked.connect(self.autoFitRequested.emit)
        self.reset_btn.clicked.connect(self.resetRequested.emit)

    def _on_mode_clicked(self, mode: str) -> None:
        self.scale_controls.setVisible(mode == "scale")
        self.modeChanged.emit(mode)

    def is_uniform(self) -> bool:
        return self.scale_controls.uniform_checkbox.isChecked()

    def sync_from_model(self, node: ModelNode) -> None:
        self.scale_controls.sync_from_model(node)

    def set_enabled_state(self, enabled: bool) -> None:
        for widget in (
            *self._mode_buttons,
            self.center_btn,
            self.autofit_btn,
            self.reset_btn,
        ):
            widget.setEnabled(enabled)
        self.scale_controls.set_enabled_state(enabled)


class ObjectPanel(QWidget):
    """Legacy expanded object panel kept for API compatibility."""

    modeChanged = pyqtSignal(str)
    centerRequested = pyqtSignal()
    autoFitRequested = pyqtSignal()
    resetRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        title = QLabel("Объект")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.transform_panel = ObjectTransformPanel(self)
        layout.addWidget(self.transform_panel)

        info_box = QGroupBox("Модель")
        info_layout = QVBoxLayout(info_box)
        self.file_label = QLabel("Файл не выбран")
        self.file_label.setObjectName("hintLabel")
        self.info_label = QLabel("")
        self.info_label.setObjectName("hintLabel")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.file_label)
        info_layout.addWidget(self.info_label)
        layout.addWidget(info_box)

        # --- Bambu-style transform modes -----------------------------------
        transform_box = QGroupBox("Трансформация")
        transform_layout = QVBoxLayout(transform_box)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self.move_mode_btn = QPushButton("Перемещение")
        self.rotate_mode_btn = QPushButton("Вращение")
        self.scale_mode_btn = QPushButton("Масштаб")
        self._mode_buttons = (
            self.move_mode_btn,
            self.rotate_mode_btn,
            self.scale_mode_btn,
        )
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for index, button in enumerate(self._mode_buttons):
            button.setCheckable(True)
            button.setObjectName("transformModeButton")
            self._mode_group.addButton(button, index)
            mode_row.addWidget(button, 1)
        self.move_mode_btn.setChecked(True)
        transform_layout.addLayout(mode_row)

        hint = QLabel("Тяни цветные стрелки, кольца или квадратные ручки прямо на модели.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        transform_layout.addWidget(hint)

        self.position_value = QLabel("X 0.00 · Y 0.00 · Z 0.00 мм")
        self.rotation_value = QLabel("X 0.0 · Y 0.0 · Z 0.0°")
        self.size_value = QLabel("X 0.00 · Y 0.00 · Z 0.00 мм")
        for value_label in (self.position_value, self.rotation_value, self.size_value):
            value_label.setObjectName("transformValue")
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        transform_layout.addWidget(QLabel("Позиция"))
        transform_layout.addWidget(self.position_value)
        transform_layout.addWidget(QLabel("Вращение"))
        transform_layout.addWidget(self.rotation_value)
        transform_layout.addWidget(QLabel("Размер"))
        transform_layout.addWidget(self.size_value)

        layout.addWidget(transform_box)

        # --- Быстрые действия -------------------------------------------
        actions = QHBoxLayout()
        self.center_btn = QPushButton("Центрировать")
        self.autofit_btn = QPushButton("Авто-фит под колбу")
        self.reset_btn = QPushButton("Сбросить")
        actions.addWidget(self.center_btn)
        actions.addWidget(self.autofit_btn)
        actions.addWidget(self.reset_btn)
        layout.addLayout(actions)

        layout.addStretch(1)
        root.addWidget(_scroll_wrap(content))

        self._wire_signals()
        self.set_enabled_state(False)

    # --- сигналы -------------------------------------------------------------
    def _wire_signals(self) -> None:
        for button, mode in zip(self._mode_buttons, ("move", "rotate", "scale")):
            button.clicked.connect(lambda _checked=False, m=mode: self.modeChanged.emit(m))

        self.center_btn.clicked.connect(self.centerRequested.emit)
        self.autofit_btn.clicked.connect(self.autoFitRequested.emit)
        self.reset_btn.clicked.connect(self.resetRequested.emit)

    def is_uniform(self) -> bool:
        return True

    # --- состояние -------------------------------------------------------------
    def set_enabled_state(self, enabled: bool) -> None:
        for w in (*self._mode_buttons, self.center_btn, self.autofit_btn, self.reset_btn):
            w.setEnabled(enabled)

    def show_model_info(self, filename: str, node: ModelNode) -> None:
        self.file_label.setText(filename)
        extents = node.base_extents
        self.info_label.setText(
            f"{tr('Вершин:')} {node.vertex_count:,}\n"
            f"{tr('Граней:')} {node.face_count:,}\n"
            f"{tr('Исходные размеры:')} "
            f"{extents[0]:.2f} × {extents[1]:.2f} × {extents[2]:.2f} {tr('мм')}"
        )

    # --- синхронизация с ModelNode ---------------------------------------------
    def sync_from_model(self, node: ModelNode) -> None:
        size = node.current_size_mm()
        rot = node.transform.rotation_deg
        pos = node.transform.translation
        self.position_value.setText(
            f"X {pos[0]:+.2f} · Y {pos[1]:+.2f} · Z {pos[2]:+.2f} мм"
        )
        self.rotation_value.setText(
            f"X {rot[0]:+.1f} · Y {rot[1]:+.1f} · Z {rot[2]:+.1f}°"
        )
        self.size_value.setText(
            f"X {size[0]:.2f} · Y {size[1]:.2f} · Z {size[2]:.2f} мм"
        )
