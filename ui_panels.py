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

import math
import subprocess
from typing import Callable, Optional, Sequence

import numpy as np
from PyQt6.QtCore import QProcess, QSettings, QSignalBlocker, Qt, QTimer, pyqtSignal
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
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import theme
from constants import (
    APP_ORG,
    APP_TITLE,
    DEFAULT_DIAMETER_MM,
    DEFAULT_FRAME_RATE_HZ,
    DEFAULT_NUM_FRAMES,
    DEFAULT_OUTPUT_RESOLUTION,
    FILL_FRACTION,
    VAT_HEIGHT_RATIO,
)
from i18n import num, tr
from model_node import ModelNode
from pixel_game import PixelGame
from projector_canvas import RESOLUTIONS, CanvasGeometry, cone_smear_mm, throw_distance_mm
from sketch_scene import SceneParams
from theme import AXIS_X, AXIS_Y, AXIS_Z
from ui_kit import Metric, Segmented, SettingRow, Stepper, Switch, section_head
from vam_backend import (
    DEFAULT_VAM_ENV_NAME,
    build_vam_install_command,
    build_vam_probe_command,
    find_conda_executable,
    is_vamtoolbox_available,
)


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
        super().__init__("", parent)
        self.setFlat(True)
        self.setStyleSheet("QGroupBox { border: none; background: transparent; margin: 0; padding: 0; }")
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 4, 0, 8)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)

        # Headers in the colours of the scene axes.
        for col, (name, color) in enumerate((("X", AXIS_X), ("Y", AXIS_Y), ("Z", AXIS_Z)), start=1):
            head = QLabel(name)
            head.setAlignment(Qt.AlignmentFlag.AlignCenter)
            head.setStyleSheet(f"color: {color}; font-weight: 700;")
            layout.addWidget(head, 0, col)

        def box(lo: float, hi: float) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(1)
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            spin.setMinimumWidth(0)
            spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return spin

        rows = ((tr("Сдвиг, мм"), -1000, 1000, ("pos_x", "pos_y", "pos_z")),
                (tr("Поворот, °"), -360, 360, ("rot_x", "rot_y", "rot_z")),
                (tr("Размер, мм"), 0.01, 1000, ("size_x", "size_y", "size_z")))
        for r, (label, lo, hi, names) in enumerate(rows, start=1):
            caption = QLabel(label)
            caption.setObjectName("rowEffect")
            layout.addWidget(caption, r, 0)
            for c, name in enumerate(names, start=1):
                spin = box(lo, hi)
                setattr(self, name, spin)
                layout.addWidget(spin, r, c)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.center_btn = QPushButton(tr("Центр"))
        self.autofit_btn = QPushButton(tr("Вписать"))
        self.reset_btn = QPushButton(tr("Сброс"))
        for b in (self.center_btn, self.autofit_btn, self.reset_btn):
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn_row.addWidget(b)
        layout.addLayout(btn_row, 4, 0, 1, 4)

        self._wire_signals()
        self.setEnabled(False)

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
            self.blockSignals(True)
            t = node.transform.translation
            r = node.transform.rotation_deg
            s = node.current_size_mm()

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
            self.blockSignals(False)


class _Fixed:
    """A value the engine still accepts but the interface no longer exposes.

    Base exposure, source intensity and the gray cut-off only multiplied or
    clipped the frame gray levels (values above 1 even saturated frames). Print
    time now comes from the gel dose and irradiance; the engine stops its
    optimiser on its own; repair is always on.
    """

    def __init__(self, value: float | bool) -> None:
        self._value = value

    def value(self) -> float:
        return float(self._value)

    def value_int(self) -> int:
        return int(self._value)

    def isChecked(self) -> bool:  # noqa: N802 (Qt-like API)
        return bool(self._value)


class _Computed:
    """Read-only value derived from other settings (keeps the old widget API)."""

    def __init__(self, fn: Callable[[], float]) -> None:
        self._fn = fn

    def value(self) -> float:
        return float(self._fn())


PAGES: dict[str, tuple[str, str]] = {
    "setup": ("НАСТРОЙКА · ОДИН РАЗ", "Принтер и смола"),
    "model": ("ШАГ 1 ИЗ 3", "Модель"),
    "light": ("ШАГ 2 ИЗ 3", "Свет"),
    "print": ("ШАГ 3 ИЗ 3", "Печать"),
}


class ProcessSettingsPanel(QWidget):
    """Side panel: setup (printer + resin) and the three steps of a part."""

    diameterChanged = pyqtSignal(float)
    settingsChanged = pyqtSignal()
    sceneChanged = pyqtSignal()
    canvasChanged = pyqtSignal()
    pageChanged = pyqtSignal(str)
    actionRequested = pyqtSignal(str)
    scaleRequested = pyqtSignal(float)
    orientationRequested = pyqtSignal(int)
    virtualToggled = pyqtSignal(bool)
    hintsToggled = pyqtSignal(bool)
    languageChanged = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("sidePanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._settings = QSettings(APP_ORG, APP_TITLE)
        self._vam_process: Optional[QProcess] = None
        self._vam_process_stage = ""
        self._vam_conda_executable = str(self._settings.value("vam/conda_executable", "") or "")
        if not self._vam_conda_executable:
            self._vam_conda_executable = find_conda_executable() or ""
        self._vam_environment_name = str(
            self._settings.value("vam/environment_name", DEFAULT_VAM_ENV_NAME) or DEFAULT_VAM_ENV_NAME)
        ready_value = self._settings.value("vam/ready", False)
        self._vam_ready = str(ready_value).lower() in {"1", "true", "yes"}
        self.vam_status = QLabel()
        self.vam_install_btn = QPushButton()

        self._page = "model"
        self._has_model = False
        self._model_info_args: Optional[tuple] = None
        self._fit_text = ""
        self._model_height_mm: Optional[float] = None
        self._fits = True
        self._running = False
        self._done = False
        self._plan: dict | None = None
        self._output_dir = ""
        self._frames_loaded = False
        self._stale = False

        # Accepted by the engine, no longer shown (see _Fixed).
        self.exposure = _Fixed(1.0)
        self.intensity = _Fixed(100.0)
        self.threshold = _Fixed(0.0)
        self.fill_holes = _Fixed(True)
        self.optimizer_iterations = _Fixed(30)
        self.projector_throw = _Computed(self.throw_mm)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        head = QWidget()
        head_l = QVBoxLayout(head)
        head_l.setContentsMargins(24, 22, 24, 8)
        head_l.setSpacing(3)
        self.step_label = QLabel()
        self.step_label.setObjectName("sectionHead")
        self.step_label.setProperty("_spinslicer_skip_text_translation", True)
        self.title = QLabel()
        self.title.setObjectName("panelTitle")
        self.title.setProperty("_spinslicer_skip_text_translation", True)
        self.todo = QLabel()
        self.todo.setObjectName("todoLabel")
        self.todo.setWordWrap(True)
        self.todo.setProperty("_spinslicer_skip_text_translation", True)
        head_l.addWidget(self.step_label)
        head_l.addWidget(self.title)
        head_l.addSpacing(4)
        head_l.addWidget(self.todo)
        root.addWidget(head)

        self._stack = QStackedWidget()
        self._pages: dict[str, QWidget] = {}
        root.addWidget(self._stack, 1)

        foot = QWidget()
        foot_l = QVBoxLayout(foot)
        foot_l.setContentsMargins(24, 10, 24, 18)
        foot_l.setSpacing(8)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setVisible(False)
        self.action_btn = QPushButton()
        self.action_btn.setObjectName("primaryAction")
        self.action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.action_btn.clicked.connect(self._on_action)
        self.action_sub = QLabel()
        self.action_sub.setObjectName("actionSub")
        self.action_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_sub.setProperty("_spinslicer_skip_text_translation", True)
        foot_l.addWidget(self.progress_bar)
        foot_l.addWidget(self.action_btn)
        foot_l.addWidget(self.action_sub)
        root.addWidget(foot)

        self._build_setup()
        self._build_model()
        self._build_light()
        self._build_print()
        self._restore()
        self._wire()
        self.show_page("model")
        self._refresh()

    # ------------------------------------------------------------------ build helpers
    def _add_page(self, key: str, rows: list[QWidget]) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 0, 24, 12)
        layout.setSpacing(0)
        for row in rows:
            layout.addWidget(row)
        layout.addStretch(1)
        scroll = _scroll_wrap(content)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._pages[key] = scroll
        self._stack.addWidget(scroll)

    @staticmethod
    def _row(label: str, control: Optional[QWidget], effect: str = "", hint: str = "",
             stacked: bool = False) -> SettingRow:
        row = SettingRow(label, control, effect, hint, stacked)
        row.setObjectName("settingRow")
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.label.setProperty("_spinslicer_skip_text_translation", False)
        row.effect.setProperty("_spinslicer_skip_text_translation", True)
        row.hint.setProperty("_spinslicer_skip_text_translation", True)
        return row

    def _link(self, text: str, action: str) -> QPushButton:
        b = QPushButton(tr(text))
        b.setObjectName("linkButton")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(lambda: self.actionRequested.emit(action))
        return b

    def _expander(self, title: str, body: QWidget) -> QWidget:
        toggle = QPushButton(tr("показать ▾"))
        toggle.setObjectName("linkButton")
        toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        row = self._row(title, toggle)
        body.setVisible(False)

        def flip() -> None:
            body.setVisible(not body.isVisible())
            toggle.setText(tr("скрыть ▴") if body.isVisible() else tr("показать ▾"))
        toggle.clicked.connect(flip)
        return self._group([row, body])

    @staticmethod
    def _group(rows: list[QWidget], top: int = 0) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, top, 0, 0)
        lay.setSpacing(0)
        for r in rows:
            lay.addWidget(r)
        return box

    # ------------------------------------------------------------------ pages
    def _build_setup(self) -> None:
        self.diameter = Stepper(20.0, 200.0, DEFAULT_DIAMETER_MM, 1.0, 1, "мм")
        self.glass_thickness = Stepper(0.5, 10.0, 2.0, 0.5, 1, "мм")
        self.placement = Segmented([("в ванне", 1), ("в воздухе", 0)], 0)
        self.bath = self.placement   # back-compat name; use container_mode_value()
        self.field_height = Stepper(5.0, 120.0, 40.0, 1.0, 0, "мм")
        self.resolution = Segmented([(f"{w}×{h}", i) for i, (w, h) in enumerate(RESOLUTIONS)], 1)
        self.output_res = Stepper(128, 3840, DEFAULT_OUTPUT_RESOLUTION, 8, 0, "px")
        self.rays = Segmented([("параллельные", 0), ("конусом", 1)], 0)
        self.throw_ratio = Stepper(0.2, 5.0, 1.2, 0.05, 2, ": 1")
        self.projector_blur = Stepper(0.0, 0.5, 0.0, 0.01, 2, "мм")
        self.offset_x = Stepper(-2000, 2000, 0, 1, 0, "px")
        self.offset_y = Stepper(-2000, 2000, 0, 1, 0, "px")
        self.projector_gamma = Stepper(0.5, 3.0, 2.2, 0.1, 2)
        self.projector_contrast = Stepper(0.0, 10000.0, 1000.0, 100.0, 0, ":1")
        self.projector_irradiance = Stepper(0.0, 200.0, 0.0, 0.5, 2, "мВт/см²")
        self.n_glass = Stepper(1.0, 2.0, 1.51, 0.01, 2)
        self.n_liquid = Stepper(1.0, 2.0, 1.45, 0.01, 2)
        self.alpha_absorption = Stepper(0.0, 0.1, 0.0, 0.001, 3, "мм⁻¹")
        self.n_resin = Stepper(1.0, 2.0, 1.50, 0.01, 2)
        self.gel_dose = Stepper(0.0, 5000.0, 0.0, 10.0, 1, "мДж/см³")
        self.hints_switch = Switch(True)
        self.language_box = NoScrollComboBox()

        self.r_diameter = self._row("Диаметр колбы", self.diameter)
        self.r_wall = self._row("Стенка", self.glass_thickness)
        self.r_bath = self._row("Колба стоит", self.placement, stacked=True)
        self.r_field = self._row("Высота поля", self.field_height)
        self.r_resolution = self._row("Разрешение проектора", self.resolution, stacked=True)
        self.r_output = self._row("Колба на экране", self.output_res,
                                  hint="сколько пикселей проектора по ширине занимает колба — проверьте разметкой")
        self.r_rays = self._row("Лучи проектора", self.rays, stacked=True)
        self.r_throw = self._row("Throw ratio", self.throw_ratio, hint="есть в паспорте проектора, обычно 1,1–1,6")
        self.r_blur = self._row("Размытие пятна", self.projector_blur, hint="не знаете? оставьте 0")
        self.r_alpha = self._row("Поглощение смолы", self.alpha_absorption)
        self.r_nresin = self._row("Преломление смолы", self.n_resin)
        self.r_gel = self._row("Доза геля", self.gel_dose, hint="не знаете? напечатайте шкалу ↓")
        self.r_calib = self._row("Калибровка", self._link("напечатать шкалу →", "calibrate"),
                                 "9 столбиков с дозой 88–112 %: выбираете лучший, доза обновится сама")
        self.r_offx = self._row("Сдвиг картинки по X", self.offset_x)
        self.r_offy = self._row("Сдвиг картинки по Y", self.offset_y)
        self.r_gamma = self._row("Гамма", self.projector_gamma)
        self.r_contrast = self._row("Контраст", self.projector_contrast)
        self.r_irr = self._row("Облучённость", self.projector_irradiance)
        self.r_nglass = self._row("n стекла", self.n_glass)
        self.r_nliquid = self._row("n жидкости ванны", self.n_liquid)
        self.r_hints = self._row("Подсказки маркером", self.hints_switch, "для тех, кто печатает впервые")
        self.r_language = self._row("Язык", self.language_box)
        expert = self._group([self.r_offx, self.r_offy, self.r_gamma, self.r_contrast, self.r_irr,
                              self.r_nglass, self.r_nliquid])
        self._add_page("setup", [
            section_head("Колба"), self.r_diameter, self.r_wall, self.r_bath,
            section_head("Проектор"), self.r_resolution, self.r_output, self.r_field, self.r_rays, self.r_throw,
            self.r_blur,
            section_head("Смола"), self.r_alpha, self.r_nresin, self.r_gel, self.r_calib,
            section_head("Ещё"), self._expander("Для специалистов", expert), self.r_hints, self.r_language,
        ])

    def _build_model(self) -> None:
        self.load_link = self._link("загрузить STL", "load")
        self.r_file = self._row("Модель не загружена", self.load_link, "STL-файл детали")
        self.r_file.label.setProperty("_spinslicer_skip_text_translation", True)
        self.scale = Stepper(1.0, 2000.0, 100.0, 5.0, 0, "%")
        self.r_scale = self._row("Масштаб", self.scale)
        self.orientation = Segmented([("как в файле", 0), ("на бок", 1), ("на торец", 2)])
        self.r_orient = self._row("Ориентация", self.orientation, stacked=True,
                                  hint="ниже деталь — меньше слоёв и точнее")
        self.preserve_internal_voids = Switch(True)
        self.r_voids = self._row("Отверстия и резьба", self.preserve_internal_voids)
        places = QWidget()
        pl = QHBoxLayout(places)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(16)
        pl.addWidget(self._link("в центр", "center"))
        pl.addWidget(self._link("вписать в колбу", "autofit"))
        self.r_place = self._row("Положение", places)
        self.fit_label = QLabel()
        self.fit_label.setObjectName("fitLabel")
        self.fit_label.setWordWrap(True)
        self.fit_label.setProperty("_spinslicer_skip_text_translation", True)
        fit_box = QWidget()
        fl = QVBoxLayout(fit_box)
        fl.setContentsMargins(0, 14, 0, 6)
        fl.addWidget(self.fit_label)
        self.transform_panel = ObjectTransformPanel(self)
        self._model_rows = [self.r_scale, self.r_orient, self.r_voids, self.r_place, fit_box]
        self._model_expander = self._expander("Точная позиция", self.transform_panel)
        self._add_page("model", [self.r_file, *self._model_rows, self._model_expander])

    def _build_light(self) -> None:
        self.grid_res = Segmented([("Быстро", 128), ("Норма", 256), ("Точно", 384)], 1)
        self.frames = Segmented([("180", 180), ("360", 360), ("720", 720)], 1)
        self.blur_compensation = Switch(True)
        self.virtual = Switch(False)
        self._virtual_ready = False
        self.r_grid = self._row("Качество", self.grid_res, stacked=True, hint="для первой печати хватит «Норма»")
        self.r_frames = self._row("Кадров на оборот", self.frames, stacked=True)
        self.r_comp = self._row("Компенсация размытия", self.blur_compensation)
        self.r_virtual = self._row("Виртуальная печать", self.virtual, "появится после расчёта света")
        self.metric_match = Metric("совпадение")
        self.metric_tol = Metric("запас экспоз.")
        self.metric_time = Metric("печать")
        for m in (self.metric_match, self.metric_tol, self.metric_time):
            m.value.setProperty("_spinslicer_skip_text_translation", True)
            m.note.setProperty("_spinslicer_skip_text_translation", True)
        results = QWidget()
        rl = QVBoxLayout(results)
        rl.setContentsMargins(0, 16, 0, 4)
        rl.setSpacing(8)
        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        for m in (self.metric_match, self.metric_tol, self.metric_time):
            tiles.addWidget(m, 1, Qt.AlignmentFlag.AlignTop)
        rl.addLayout(tiles)
        self.warnings_label = QLabel()
        self.warnings_label.setObjectName("rowEffect")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setProperty("_spinslicer_skip_text_translation", True)
        rl.addWidget(self.warnings_label)
        self.results_box = results
        results.setVisible(False)

        self.game = PixelGame()
        self.game_title = QLabel(tr("Пока считается — засветите деталь сами"))
        self.game_title.setObjectName("rowLabel")
        self.game_title.setProperty("_spinslicer_skip_text_translation", True)
        game_box = QWidget()
        gl = QVBoxLayout(game_box)
        gl.setContentsMargins(0, 16, 0, 0)
        gl.setSpacing(6)
        gl.addWidget(self.game_title)
        gl.addWidget(self._arcade(self.game))
        self.game_box = game_box
        game_box.setVisible(False)
        self._add_page("light", [self.r_grid, self.r_frames, self.r_comp, self.r_virtual, results, game_box])

    def _build_print(self) -> None:
        self.output_kind = Segmented([("на проектор", 0), ("видео MP4", 1), ("кадры PNG", 2)])
        self.speed = Stepper(1.0, 240.0, DEFAULT_FRAME_RATE_HZ * 360.0 / DEFAULT_NUM_FRAMES, 1.0, 0, "°/с")
        self.brightness = Stepper(10.0, 100.0, 100.0, 5.0, 0, "%")
        self.r_output_kind = self._row("Куда выводить", self.output_kind, stacked=True)
        self.r_speed = self._row("Скорость вращения", self.speed)
        self.r_brightness = self._row("Яркость проектора", self.brightness)
        tools = QWidget()
        tl = QHBoxLayout(tools)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(16)
        tl.addWidget(self._link("разметка колбы", "pattern"))
        tl.addWidget(self._link("тест серого", "ramp"))
        self.r_tools = self._row("Настроить проектор", tools,
                                 "разметка покажет, где должны быть стенки колбы; тест серого — все уровни яркости")
        self.plan_label = QLabel(tr("Сначала рассчитайте свет."))
        self.plan_label.setObjectName("planLabel")
        self.plan_label.setWordWrap(True)
        self.plan_label.setProperty("_spinslicer_skip_text_translation", True)
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(0, 10, 0, 12)
        bl.addWidget(self.plan_label)
        # The dose card (video_tab.DoseCard) is put here by the main window: its slider
        # is the print exposure, and the plan above follows it.
        self.dose_slot = QWidget()
        self._dose_layout = QVBoxLayout(self.dose_slot)
        self._dose_layout.setContentsMargins(0, 12, 0, 4)
        self._dose_card: Optional[QWidget] = None
        self._exposure = 1.0
        self._projecting: Optional[tuple[float, float]] = None
        self.game_print = PixelGame()
        self._add_page("print", [self.r_output_kind, self.r_speed, self.r_brightness, self.dose_slot, box,
                                 self.r_tools, self._expander("Мини-игры", self._arcade(self.game_print))])

    def attach_dose_card(self, card: QWidget) -> None:
        """Show the dose card on the print page; its slider sets the print exposure."""
        self._dose_card = card
        self._dose_layout.addWidget(card)
        stored = self._settings.value("print/exposure", None)
        if stored is not None and hasattr(card, "set_exposure"):
            try:
                card.set_exposure(float(stored))
            except (TypeError, ValueError):
                pass
        if hasattr(card, "exposure"):
            self._exposure = float(card.exposure())
        if hasattr(card, "exposureChanged"):
            card.exposureChanged.connect(self._on_exposure)

    def _on_exposure(self, value: float) -> None:
        self._exposure = float(value)
        self._settings.setValue("print/exposure", self._exposure)
        self._refresh()

    def exposure_factor(self) -> float:
        return self._exposure

    def brightness_factor(self) -> float:
        return self.brightness.value() / 100.0

    def print_duration_s(self) -> float:
        """Projection time that delivers the planned dose × exposure at this brightness (0 = unknown)."""
        plan = self._plan
        if not plan or "frame_exposure_total_s" not in plan:
            return 0.0
        revs = float(plan["frame_exposure_total_s"]) * self.frame_rate_hz_value()
        base = revs * self.frames.value_int() / self.frame_rate_hz_value()
        return base * self._exposure / max(self.brightness_factor(), 0.05)

    def set_projecting(self, elapsed: Optional[float], total: float = 0.0) -> None:
        """Projection state from the screen: None = stopped, else elapsed / planned seconds."""
        self._projecting = None if elapsed is None else (float(elapsed), float(total))
        # Speed and brightness are baked into a running projection: locked until it stops.
        running = self._projecting is not None
        for row in (self.r_speed, self.r_brightness):
            row.setEnabled(not running)
        if self._dose_card is not None:
            self._dose_card.setEnabled(not running)
        self._refresh_action()

    def _arcade(self, game: PixelGame) -> QWidget:
        """A game with its picker and a one-line rule."""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 4, 0, 8)
        lay.setSpacing(6)
        picker = Segmented([(g.title, i) for i, g in enumerate(game.games)])
        rule = QLabel()
        rule.setObjectName("rowEffect")
        rule.setWordWrap(True)
        rule.setProperty("_spinslicer_skip_text_translation", True)
        controls = QLabel()
        controls.setObjectName("rowEffect")
        controls.setProperty("_spinslicer_skip_text_translation", True)
        hints = {"cure": "мышь — двигать, клик — старт", "layers": "←/→ — двигать, ↑ — повернуть, ↓ — быстрее",
                 "photon": "клик или пробел — подлететь вверх"}

        def choose(index: int) -> None:
            game.set_game(index)
            rule.setText(tr(game.game().hint))
            controls.setText(tr(hints[game.game().key]))
            self._settings.setValue("game/last", index)
        picker.changed.connect(choose)
        self._arcade_refresh = getattr(self, "_arcade_refresh", []) + [lambda: choose(picker.currentIndex())]
        last = int(self._settings.value("game/last", 0) or 0)
        picker.setCurrentIndex(last)
        choose(last)
        lay.addWidget(picker)
        lay.addWidget(game)
        lay.addWidget(rule)
        lay.addWidget(controls)
        return box

    # ------------------------------------------------------------- settings
    def _persisted(self) -> dict[str, QWidget]:
        return {
            "ui/diameter": self.diameter, "ui/wall": self.glass_thickness, "ui/placement": self.placement,
            "ui/field": self.field_height, "ui/resolution": self.resolution, "ui/output_res": self.output_res,
            "ui/rays": self.rays, "ui/throw_ratio": self.throw_ratio,
            "physics/blur": self.projector_blur, "ui/offset_x": self.offset_x, "ui/offset_y": self.offset_y,
            "physics/gamma": self.projector_gamma, "physics/contrast": self.projector_contrast,
            "physics/irradiance": self.projector_irradiance, "ui/n_glass": self.n_glass,
            "ui/n_liquid": self.n_liquid, "physics/alpha": self.alpha_absorption, "ui/n_resin": self.n_resin,
            "physics/gel_dose": self.gel_dose, "ui/grid": self.grid_res, "ui/frames": self.frames,
            "ui/blur_comp": self.blur_compensation, "ui/speed": self.speed, "ui/voids": self.preserve_internal_voids,
            "ui/output_kind": self.output_kind, "print/brightness": self.brightness,
        }

    def _physics_widgets(self) -> dict[str, Stepper]:
        return {k: w for k, w in self._persisted().items() if k.startswith("physics/") and isinstance(w, Stepper)}

    def _restore(self) -> None:
        for key, widget in self._persisted().items():
            stored = self._settings.value(key, None)
            if stored is None:
                continue
            try:
                if isinstance(widget, Switch):
                    widget.setChecked(str(stored).lower() in {"1", "true", "yes"})
                else:
                    widget.setValue(float(stored))  # type: ignore[attr-defined]
            except (TypeError, ValueError):
                pass
        hints = str(self._settings.value("ui/hints", True)).lower() in {"1", "true", "yes"}
        self.hints_switch.setChecked(hints)

    def _save(self, key: str, widget: QWidget) -> None:
        if isinstance(widget, Switch):
            self._settings.setValue(key, widget.isChecked())
        else:
            self._settings.setValue(key, widget.value())  # type: ignore[attr-defined]

    def _wire(self) -> None:
        scene_keys = {"ui/diameter", "ui/wall", "ui/placement", "ui/field", "ui/rays", "ui/throw_ratio",
                      "ui/resolution", "ui/output_res", "physics/alpha", "ui/n_resin", "ui/n_liquid", "ui/frames"}
        canvas_keys = {"ui/resolution", "ui/offset_x", "ui/offset_y", "ui/speed", "ui/frames"}
        for key, widget in self._persisted().items():
            signal = widget.toggled if isinstance(widget, Switch) else widget.valueChanged  # type: ignore[attr-defined]
            signal.connect(lambda _v, k=key, w=widget: self._save(k, w))
            signal.connect(lambda _v, k=key: self._on_setting_changed(k))
            if key in scene_keys:
                signal.connect(lambda _v: self.sceneChanged.emit())
            if key in canvas_keys:
                signal.connect(lambda _v: self.canvasChanged.emit())
        self.diameter.valueChanged.connect(self.diameterChanged.emit)
        self.placement.valueChanged.connect(lambda _v: self.diameterChanged.emit(self.diameter.value()))
        self.scale.valueChanged.connect(self.scaleRequested.emit)
        self.orientation.changed.connect(self.orientationRequested.emit)
        self.virtual.toggled.connect(self._on_virtual_toggled)
        self.hints_switch.toggled.connect(self.hintsToggled.emit)
        self.hints_switch.toggled.connect(lambda v: self._settings.setValue("ui/hints", v))
        self.language_box.currentIndexChanged.connect(self.languageChanged.emit)
        self.output_kind.changed.connect(lambda _i: self._refresh_action())

    def _on_virtual_toggled(self, on: bool) -> None:
        if not on and not self._virtual_ready:
            self.virtualToggled.emit(False)
            return
        if on and not self._virtual_ready:
            # Always clickable: say why it cannot show anything yet instead of looking dead.
            QTimer.singleShot(180, lambda: self.virtual.setChecked(False))
            self.r_virtual.set_effect("появится после расчёта света" if not self._stale
                                      else "результат устарел — пересчитайте свет", warn=True)
            return
        self.virtualToggled.emit(on)

    # Settings that only change how finished frames are shown, never the frames.
    _SHOW_ONLY_KEYS = frozenset({"ui/speed", "ui/output_kind", "print/brightness", "ui/offset_x",
                                 "ui/offset_y", "ui/resolution"})

    def _on_setting_changed(self, key: str = "") -> None:
        # Changing physics after a run makes the old result stale.
        if self._done and not self._running and key not in self._SHOW_ONLY_KEYS:
            self._done = False
            self._plan = None
            self.set_virtual_available(False)
            self._stale = True
        self._refresh()
        self.settingsChanged.emit()

    # ------------------------------------------------------------- accessors
    def vat_diameter_mm(self) -> float:
        return self.diameter.value()

    def projection_backend_name(self) -> str:
        return "physical"

    def optimizer_iterations_value(self) -> int:
        return self.optimizer_iterations.value_int()

    def vam_runtime_config(self) -> tuple[str, str]:
        if is_vamtoolbox_available():
            return "", ""
        if self._vam_ready and self._vam_conda_executable:
            return self._vam_conda_executable, self._vam_environment_name
        return "", ""

    def has_bath(self) -> bool:
        return self.placement.value_int() == 1

    def container_mode_value(self) -> int:
        return 2 if self.has_bath() else 0

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

    def field_height_mm(self) -> float:
        return self.field_height.value()

    def printable_height_mm(self) -> float:
        return min(self.field_height.value(), self.diameter.value() * VAT_HEIGHT_RATIO)

    def effective_blur_mm(self) -> float:
        return self.projector_blur.value() if self.blur_compensation.isChecked() else 0.0

    def frame_rate_hz_value(self) -> float:
        return max(0.1, self.speed.value() * self.frames.value_int() / 360.0)

    def screen_size(self) -> tuple[int, int]:
        return RESOLUTIONS[self.resolution.value_int()]

    def canvas_geometry(self) -> CanvasGeometry:
        w, h = self.screen_size()
        return CanvasGeometry(w, h, int(self.offset_x.value()), int(self.offset_y.value()))

    def throw_mm(self) -> float:
        """Projector pupil → vial axis for the engine; 0 = parallel rays."""
        if self.rays.value_int() == 0:
            return 0.0
        d = throw_distance_mm(self.throw_ratio.value(), self.diameter.value(),
                              self.output_res.value_int(), self.screen_size()[0])
        # The engine needs the pupil outside the vial.
        return max(d, self.diameter.value() * 1.05)

    def throw_too_close(self) -> bool:
        if self.rays.value_int() == 0:
            return False
        d = throw_distance_mm(self.throw_ratio.value(), self.diameter.value(),
                              self.output_res.value_int(), self.screen_size()[0])
        return d <= self.diameter.value() * 1.05

    def scene_params(self) -> SceneParams:
        return SceneParams(
            diameter_mm=self.diameter.value(), wall_mm=self.glass_thickness.value(), bath=self.has_bath(),
            field_mm=self.field_height.value(), throw_mm=self.throw_mm(),
            alpha_per_mm=self.alpha_absorption.value(), n_resin=self.n_resin.value(),
            n_liquid=self.n_liquid.value(), height_ratio=VAT_HEIGHT_RATIO,
        )

    def _compute_estimate_s(self, grid: int, frames: int) -> float:
        """Compute time on an ordinary 4-core computer.

        Fitted to benchmarks (docs/validation/resources-2026-10-31.md): the engine
        solves one layer per projector row, so time grows with the part's height,
        roughly with grid² and with frames^0.65; a 2-core 2.1 GHz VM takes
        3 s + 0.03 s × layers × (grid/256)² × (frames/360)^0.65, a typical desktop ~0.6 of that.
        """
        height = self._model_height_mm or 20.0
        layers = max(1.0, height * self.output_res.value_int() / max(self.diameter.value(), 1.0))
        vm = 3.0 + 0.03 * layers * (grid / 256.0) ** 2 * (frames / 360.0) ** 0.65
        # Normal and Fine plan with 1.5 detector bins per voxel (optics.fastcal.detector_bins):
        # measured 1.45–1.6x the time of one bin (Benchy: 21 -> 34 s, 48 -> 69 s).
        from optics.fastcal import detector_bins
        vm *= detector_bins(grid, self.output_res.value_int()) / grid
        return 0.6 * vm

    def derived(self) -> dict[str, float]:
        d = self.diameter.value()
        inner = max(d - 2 * self.glass_thickness.value(), 1.0)
        grid = self.grid_res.value_int()
        frames = self.frames.value_int()
        zone = inner * FILL_FRACTION
        if not self.has_bath():
            # In air the vial is a lens: only the centre receives light from every side.
            zone = min(zone, 2 * (d / 2) * 1.0 / max(self.n_resin.value(), 1.0) * 0.98)
        return {
            "inner": inner, "zone": zone, "height": self.printable_height_mm(),
            "voxel": d / grid, "grid": grid, "frames": frames, "step": 360.0 / frames,
            "calc_s": self._compute_estimate_s(grid, frames),
            "far": math.exp(-self.alpha_absorption.value() * inner),
            "fps": self.frame_rate_hz_value(),
            "smear": cone_smear_mm(self.throw_mm(), self.printable_height_mm() / 2, inner / 2),
        }

    # ------------------------------------------------------------- refresh
    @staticmethod
    def _f(x: float, digits: int = 1) -> str:
        return num(x, digits)

    def _refresh(self) -> None:
        v = self.derived()
        f = self._f
        bath = self.has_bath()
        outside = self.n_liquid.value() if bath else 1.0
        bend = abs(self.n_resin.value() - outside) * 20.0
        self.r_diameter.set_effect(f"зона печати Ø {f(v['zone'], 0)} мм · воксель {f(v['voxel'], 2)} мм")
        self.r_wall.set_effect(f"внутри Ø {f(v['inner'], 0)} мм — столько занимает смола")
        self.r_bath.set_effect(
            f"в жидкости n = {f(self.n_liquid.value(), 2)} луч почти не гнётся: кадры без искажений" if bath
            else f"в воздухе колба работает как линза: кадры специально растянуты, печатается только центр "
                 f"Ø {f(v['zone'], 0)} мм. Ванна лучше.", warn=not bath)
        vial_h = self.diameter.value() * VAT_HEIGHT_RATIO
        self.r_field.set_effect(
            f"печатать можно до {f(v['height'], 0)} мм в высоту"
            + (f" (ограничивает колба {f(vial_h, 0)} мм)" if vial_h < self.field_height.value() else ""))
        sw, sh = self.screen_size()
        g = self.canvas_geometry()
        fits = self.output_res.value_int() <= sw
        self.r_resolution.set_effect(f"видео и вывод на проектор — {sw}×{sh}, пиксель в пиксель")
        self.r_output.set_effect(
            f"пиксель {f(self.diameter.value() / max(self.output_res.value(), 1), 3)} мм на оси колбы"
            + ("" if fits else " — шире экрана проектора!"), warn=not fits)
        cone = self.rays.value_int() == 1
        self.r_throw.setVisible(cone)
        if cone:
            dist = throw_distance_mm(self.throw_ratio.value(), self.diameter.value(),
                                     self.output_res.value_int(), sw)
            smear = v["smear"]
            if self.throw_too_close():
                self.r_throw.set_effect(f"до оси {f(dist, 0)} мм — ближе колбы, так не бывает: проверьте числа", warn=True)
            else:
                self.r_throw.set_effect(
                    f"до оси ≈ {f(dist, 0)} мм · по высоте у краёв детали размытие до {f(smear, 2)} мм"
                    + (" — поставьте проектор дальше" if smear > v["voxel"] * 2 else ""),
                    warn=smear > v["voxel"] * 2)
        self.r_rays.set_effect("движок трассирует расходящиеся лучи (как у обычного объектива)" if cone
                               else "телецентрический объектив или проектор далеко")
        blur = self.projector_blur.value()
        self.r_blur.set_effect("проектор резкий — кадры как есть" if blur <= 0
                               else f"кадры подрезкиваются под размытие {f(blur, 2)} мм")
        self.r_offx.set_effect(f"картинка на экране: x = {g.placement(self.output_res.value_int(), 1)[0]} px")
        self.r_offy.set_effect("двигает картинку вверх/вниз, чтобы деталь попала в колбу")
        self.r_gamma.set_effect(f"серый → свет по степени {f(self.projector_gamma.value(), 2)}; кадры пересчитываются")
        con = self.projector_contrast.value()
        self.r_contrast.set_effect("0 — идеальный чёрный" if con <= 0
                                   else f"чёрный пиксель светит {f(100.0 / con, 2)} % от белого")
        irr = self.projector_irradiance.value()
        self.r_irr.set_effect("0 — время печати не считается" if irr <= 0 else "время печати считается из неё",
                              warn=irr <= 0)
        self.r_nglass.set_effect("преломление на стенке колбы")
        self.r_nliquid.set_effect("преломление на стенке ванны")
        alpha = self.alpha_absorption.value()
        self.r_alpha.set_effect(
            "0 — свет не слабеет, время печати не считается" if alpha <= 0
            else f"до дальней стенки доходит {round(v['far'] * 100)} % света", warn=alpha <= 0)
        self.r_nresin.set_effect(f"луч гнётся на стенке на {f(bend)}°" + (" (с ванной)" if bath else " (без ванны)"))
        gel = self.gel_dose.value()
        self.r_gel.set_effect("0 — неизвестно, время печати не считается" if gel <= 0
                              else "из неё и мощности проектора считается время печати", warn=gel <= 0)
        self.r_grid.set_effect(f"сетка {v['grid']:.0f} · воксель {f(v['voxel'], 2)} мм · расчёт ≈ {self._secs(v['calc_s'])}")
        self.r_frames.set_effect(f"шаг {f(v['step'])}° между кадрами")
        self.r_comp.setVisible(blur > 0)
        self.r_comp.set_effect(("подрезка кадров под " + f(blur, 2) + " мм") if self.blur_compensation.isChecked()
                               else "без подрезки мелкие детали размоет", warn=not self.blur_compensation.isChecked())
        fps = v["fps"]
        self.r_speed.set_effect(f"кадр каждые {round(1000 / fps)} мс · проектор {f(fps)} Гц"
                                + (" — больше 60 Гц, проверьте проектор" if fps > 60 else ""), warn=fps > 60)
        b = self.brightness_factor()
        self.r_brightness.set_effect(
            "полная яркость — самая короткая печать" if b >= 0.999 else
            f"кадры в {f(1 / b, 1)} раза тусклее, засветка во столько же раз дольше — "
            "доза та же; полезно для быстрой смолы")
        kind = self.output_kind.currentIndex()
        self.r_output_kind.set_effect(
            ["полноэкранно на второй экран, пиксель в пиксель", f"файл {sw}×{sh} — показывать во весь экран без масштабирования",
             "папка PNG для своего софта показа"][kind])
        self._refresh_action()
        self._refresh_plan()

    @staticmethod
    def _secs(s: float) -> str:
        return f"{max(1, round(s))} с" if s < 60 else f"{int(s // 60)}:{int(s % 60):02d}"

    def _todo(self) -> str:
        page = self._page
        if page == "setup":
            return "Опишите свой принтер и смолу. Это делается один раз — дальше только детали."
        if page == "model":
            if not self._has_model:
                return "Загрузите STL: перетащите файл на сцену слева или нажмите кнопку внизу."
            if not self._fits:
                return "Деталь не помещается — уменьшите масштаб или нажмите «вписать в колбу»."
            return "Покрутите деталь мышью и проверьте, как она стоит. Если всё ок — дальше."
        if page == "light":
            if self._running:
                return "Считаю кадры для каждого угла. Можно пока поиграть ↓"
            if self._done:
                return "Готово. Посмотрите совпадение и запас — и переходите к печати."
            return "Выберите качество и нажмите «Рассчитать свет»."
        if not self._output_dir:
            return "Сначала рассчитайте свет на шаге 2."
        if self._frames_outdated():
            return "Настройки или деталь изменились — кадры устарели. Пересчитайте свет."
        return "Проверьте разметку колбы на проекторе и запускайте."

    def _refresh_action(self) -> None:
        page = self._page
        self.todo.setText(tr(self._todo()))
        self.progress_bar.setVisible(self._running)
        self.action_btn.setEnabled(True)
        if page == "setup":
            self.action_btn.setText(tr("Готово — к модели"))
            self.action_sub.setText(tr("всё сохраняется само"))
        elif page == "model":
            if not self._has_model:
                self.action_btn.setText(tr("Загрузить STL"))
                self.action_sub.setText(tr("или перетащите файл на сцену"))
            elif not self._fits:
                self.action_btn.setText(tr("Сначала уместите деталь"))
                self.action_btn.setEnabled(False)
                self.action_sub.setText(tr("уменьшите масштаб или возьмите колбу шире"))
            else:
                self.action_btn.setText(tr("Дальше — свет →"))
                self.action_sub.setText(tr("проверка модели пройдена"))
        elif page == "light":
            if self._running:
                self.action_btn.setText(tr("Остановить"))
                self.action_sub.setText(tr("считаю кадры…"))
            elif self._done:
                self.action_btn.setText(tr("Дальше — печать →"))
                self.action_sub.setText(tr(f"готово · {self.frames.value_int()} кадров"))
            else:
                self.action_btn.setText(tr("Рассчитать свет"))
                self.action_btn.setEnabled(self._has_model)
                self.action_sub.setText(tr(f"≈ {self._secs(self.derived()['calc_s'])} на обычном компьютере"
                                           if self._has_model else "сначала загрузите модель"))
        elif self._frames_outdated() and self._projecting is None:
            self.action_btn.setText(tr("Пересчитать свет"))
            self.action_sub.setText(tr("старые кадры не подходят к новым настройкам"))
        elif self._projecting is not None:
            elapsed, total = self._projecting
            self.action_btn.setText(tr("Остановить засветку"))
            self.action_sub.setText(tr(f"идёт засветка {self._secs(elapsed)} из {self._secs(total)}") if total
                                    else tr("идёт засветка · Esc на проекторе — стоп"))
        else:
            ready = bool(self._output_dir)
            kind = self.output_kind.currentIndex()
            self.action_btn.setText(tr(["Показать на проекторе", "Сохранить видео", "Открыть папку кадров"][kind]))
            self.action_btn.setEnabled(ready)
            self.action_sub.setText(tr(("Esc на проекторе — остановить" if kind == 0 else "кадры готовы") if ready
                                       else "сначала рассчитайте свет"))
        for row in self._model_rows:
            row.setVisible(self._has_model)
        self._model_expander.setVisible(self._has_model)

    def _refresh_plan(self) -> None:
        plan = self._plan
        f = self._f
        if not plan:
            self.results_box.setVisible(False)
            self.plan_label.setText(tr("Сначала рассчитайте свет."))
            return
        self.results_box.setVisible(self._done)
        vp = plan.get("virtual_print") or {}
        iou = vp.get("iou")
        self.metric_match.set(f"{f(100 * iou)} %" if iou is not None else "—",
                              ("почти идеально" if iou > 0.98 else "можно точнее") if iou is not None else "")
        tol = float(plan.get("exposure_tolerance_pct", 0.0))
        self.metric_tol.set(f"± {f(tol)} %" if tol > 0 else "нет",
                            "ошибиться сложно" if tol >= 4 else ("нужна калибровка" if tol > 0 else "увеличьте сетку"))
        fps = self.frame_rate_hz_value()
        frames = self.frames.value_int()
        rows = []
        if "frame_exposure_total_s" in plan:
            revs = float(plan["frame_exposure_total_s"]) * fps
            total = revs * frames / fps
            self.metric_time.set(self._secs(total), f"{f(revs)} оборота")
            # On the print page: the time the projector will actually run.
            scale = self._exposure / max(self.brightness_factor(), 0.05)
            rows += [("Время засветки", self._secs(total * scale)), ("Обороты", f(revs * scale))]
            if abs(self._exposure - 1.0) > 1e-3:
                rows.append(("Экспозиция", f"{round(self._exposure * 100)} % от плана"))
            if self.brightness_factor() < 0.999:
                rows.append(("Яркость", f"{round(self.brightness_factor() * 100)} %"))
        else:
            self.metric_time.set("—", "нужны доза и мощность")
            rows.append(("Время", "задайте поглощение, облучённость и дозу"))
        rows.append(("Частота кадров", f"{f(fps)} Гц"))
        if tol > 0:
            rows.append(("Запас экспозиции", f"± {f(tol)} %"))
        warnings = plan.get("warnings") or []
        self.warnings_label.setText("\n".join(f"• {tr(w)}" for w in warnings[:3]))
        self.warnings_label.setVisible(bool(warnings))
        p = theme.current()
        html = "".join(f"<tr><td style='color:{p.muted}; padding:3px 18px 3px 0'>{tr(k)}</td>"
                       f"<td style='color:{p.text}'>{tr(val)}</td></tr>" for k, val in rows)
        self.plan_label.setText(f"<table>{html}</table>")

    # ------------------------------------------------------------- state from the app
    def show_page(self, key: str) -> None:
        if key == "printer" or key == "resin":
            key = "setup"
        if key not in PAGES:
            return
        self._page = key
        self._stack.setCurrentWidget(self._pages[key])
        step, title = PAGES[key]
        self.step_label.setText(tr(step))
        self.title.setText(tr(title))
        self._refresh_action()
        self.pageChanged.emit(key)

    def current_page(self) -> str:
        return self._page

    def refresh_language(self) -> None:
        """Rebuild every text the panel composes itself, after the language changed."""
        if self._model_info_args is not None:
            self.set_model_info(*self._model_info_args)
        self.show_page(self._page)
        if self._fit_text:
            self.set_fit(self._fits, self._fit_text)
        self._refresh()
        for refresh in getattr(self, "_arcade_refresh", []):
            refresh()

    def set_model_info(self, name: str, size_mm: Sequence[float] | None, faces: int, scale_pct: float) -> None:
        self._model_info_args = (name, size_mm, faces, scale_pct)
        self._has_model = size_mm is not None
        self._model_height_mm = float(size_mm[2]) if size_mm is not None else None
        f = self._f
        if size_mm is None:
            self.r_file.label.setText(tr("Модель не загружена"))
            self.r_file.set_effect("STL-файл детали")
            self.load_link.setText(tr("загрузить STL"))
        else:
            self.r_file.label.setText(name)
            faces_text = f"{faces:,}".replace(",", " ")
            self.r_file.set_effect(f"{f(size_mm[0])} × {f(size_mm[1])} × {f(size_mm[2])} мм · {faces_text} треугольников")
            self.load_link.setText(tr("заменить"))
            with QSignalBlocker(self.scale):
                self.scale.setValue(scale_pct)
            self.r_scale.set_effect(f"{f(size_mm[0])} × {f(size_mm[1])} × {f(size_mm[2])} мм")
            self.r_orient.set_effect(f"по высоте {f(size_mm[2])} мм из {f(self.printable_height_mm(), 0)}")
        self._refresh_action()

    def set_fit(self, ok: bool, text: str) -> None:
        self._fits = ok
        self._fit_text = text
        p = theme.current()
        self.fit_label.setText(("✓  " if ok else "!  ") + tr(text))
        self.fit_label.setStyleSheet(f"color: {p.text if ok else p.warn}; font-weight: 600;")
        self._refresh_action()

    def set_running(self, running: bool, progress: float = 0.0) -> None:
        was = self._running
        self._running = running
        self.progress_bar.setValue(int(max(0.0, min(1.0, progress)) * 1000))
        if running:
            self._done = False
            if not was:
                self.game_box.setVisible(True)
                self.game_title.setText(tr("Пока считается — засветите деталь сами"))
        # While computing the settings cannot change: make room for the game.
        for row in (self.r_grid, self.r_frames, self.r_virtual):
            row.setVisible(not running)
        self.r_comp.setVisible(not running and self.projector_blur.value() > 0)
        self._refresh_action()

    def set_results(self, plan: dict | None, output_dir: str) -> None:
        self._plan = plan
        self._output_dir = output_dir
        self._done = True
        self.game_title.setText(tr("Свет готов. Доиграйте или идите печатать"))
        self._refresh_plan()
        self._refresh_action()

    def set_virtual_available(self, available: bool) -> None:
        self._virtual_ready = available
        if available:
            self._stale = False
            self.r_virtual.set_effect("показать в 3D, что затвердеет")
        else:
            if self._done and not self._running:
                # The part moved after the run: the frames no longer match it.
                self._done = False
                self._plan = None
                self._stale = True
                self._refresh_plan()
                self._refresh_action()
            if self.virtual.isChecked():
                with QSignalBlocker(self.virtual):
                    self.virtual.setChecked(False)
                self.virtualToggled.emit(False)
            self.r_virtual.set_effect("результат устарел — пересчитайте свет" if self._stale
                                      else "появится после расчёта света")

    def set_photon_frames(self, frames: list) -> None:
        for game in (self.game, self.game_print):
            game.photon.set_level_from_frames(frames)
            game.update()
        for refresh in getattr(self, "_arcade_refresh", []):
            refresh()

    def set_game_shape(self, grid: object) -> None:
        import numpy as _np
        shape = _np.asarray(grid, dtype=bool)
        self.game.set_shape(shape)
        self.game_print.set_shape(shape)

    def is_done(self) -> bool:
        return self._done

    def _frames_outdated(self) -> bool:
        """Frames exist but the settings or the part changed since they were computed."""
        return bool(self._output_dir) and self._stale and not self._done and not self._running

    def output_dir(self) -> str:
        return self._output_dir

    def _on_action(self) -> None:
        page = self._page
        if page == "setup":
            self.actionRequested.emit("goto:model")
        elif page == "model":
            self.actionRequested.emit("load" if not self._has_model else "goto:light")
        elif page == "light":
            self.actionRequested.emit("goto:print" if (self._done and not self._running) else "generate")
        elif self._projecting is not None:
            self.actionRequested.emit("stop_projector")
        elif self._frames_outdated():
            self.actionRequested.emit("goto:light")
        else:
            self.actionRequested.emit(["projector", "export", "open_frames"][self.output_kind.currentIndex()])

    def refresh_theme(self) -> None:
        self._refresh()
        if self.fit_label.text():
            self.set_fit(self._fits, self._fit_text)
        self.game.update()
        self.game_print.update()

    def shutdown(self) -> None:
        """Stop an in-progress optional dependency installation."""

        process = self._vam_process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            return
        process.terminate()
        if not process.waitForFinished(1000):
            process.kill()
            process.waitForFinished(1000)

    def _refresh_vam_status(self) -> None:
        if is_vamtoolbox_available():
            self._vam_ready = True
            self.vam_status.setText("VAMToolbox доступен в текущем Python-окружении.")
            self.vam_install_btn.setText("VAMToolbox уже доступен")
            self.vam_install_btn.setEnabled(False)
            return

        if self._vam_ready and self._vam_conda_executable:
            self.vam_status.setText(
                f"Проверенное окружение: {self._vam_environment_name}. "
                "Его можно выбрать как движок проекций."
            )
            self.vam_install_btn.setText("Проверить окружение VAMToolbox")
            self.vam_install_btn.setEnabled(False)
            QTimer.singleShot(0, self._start_vam_probe)
            return

        if self._vam_conda_executable:
            self.vam_status.setText(
                "Conda найдена. Установщик создаст отдельное окружение "
                f"«{self._vam_environment_name}» и не изменит внутренний Radon."
            )
            self.vam_install_btn.setEnabled(True)
        else:
            self.vam_status.setText(
                "Conda не найдена. Установите Miniconda или Anaconda, "
                "перезапустите приложение и повторите установку."
            )
            self.vam_install_btn.setEnabled(False)

    def _qprocess_argv(self, command: list[str]) -> tuple[str, list[str]]:
        executable = command[0]
        if executable.lower().endswith((".bat", ".cmd")):
            return "cmd.exe", ["/d", "/s", "/c", subprocess.list2cmdline(command)]
        return executable, command[1:]

    def _run_vam_process(self, command: list[str], stage: str) -> None:
        if self._vam_process is not None:
            return
        program, arguments = self._qprocess_argv(command)
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_vam_output)
        process.finished.connect(self._on_vam_process_finished)
        self._vam_process = process
        self._vam_process_stage = stage
        process.start(program, arguments)

    def _on_vam_output(self) -> None:
        process = self._vam_process
        if process is None:
            return
        raw = process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if lines:
            self.vam_status.setText(lines[-1][-240:])

    def _on_vam_process_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        process = self._vam_process
        stage = self._vam_process_stage
        self._vam_process = None
        self._vam_process_stage = ""
        if process is not None:
            process.deleteLater()

        if stage == "install":
            if exit_code != 0:
                self._vam_ready = False
                self.vam_status.setText(
                    "Установка VAMToolbox завершилась с ошибкой. "
                    "Проверьте вывод команды и каналы Conda."
                )
                self.vam_install_btn.setEnabled(bool(self._vam_conda_executable))
                return
            self.vam_status.setText("Проверяю установленный VAMToolbox...")
            self._start_vam_probe()
            return

        if stage == "probe":
            self._vam_ready = exit_code == 0
            self._settings.setValue("vam/ready", self._vam_ready)
            self._settings.setValue("vam/conda_executable", self._vam_conda_executable)
            self._settings.setValue("vam/environment_name", self._vam_environment_name)
            if self._vam_ready:
                self.vam_status.setText(
                    f"VAMToolbox установлен в окружении «{self._vam_environment_name}»."
                )
                self.vam_install_btn.setText("VAMToolbox установлен")
                self.vam_install_btn.setEnabled(False)
                self.settingsChanged.emit()
            else:
                self.vam_status.setText(
                    "Не удалось импортировать VAMToolbox из окружения. "
                    "Переустановите его или используйте внутренний Radon."
                )
                self.vam_install_btn.setEnabled(bool(self._vam_conda_executable))

    def _start_vam_install(self) -> None:
        if self._vam_process is not None:
            return
        self._vam_conda_executable = find_conda_executable() or self._vam_conda_executable
        if not self._vam_conda_executable:
            QMessageBox.information(
                self,
                tr("VAMToolbox — альтернативный движок"),
                "Установите Miniconda или Anaconda с официального сайта, "
                "добавьте Conda в PATH и перезапустите приложение.",
            )
            return

        self._vam_ready = False
        self.vam_install_btn.setEnabled(False)
        self.vam_status.setText(
            f"Устанавливаю VAMToolbox в окружение «{self._vam_environment_name}». "
            "Это может занять несколько минут..."
        )
        self._run_vam_process(
            build_vam_install_command(
                self._vam_conda_executable,
                self._vam_environment_name,
            ),
            "install",
        )

    def _start_vam_probe(self) -> None:
        if self._vam_process is not None or not self._vam_conda_executable:
            return
        self._run_vam_process(
            build_vam_probe_command(
                self._vam_conda_executable,
                self._vam_environment_name,
            ),
            "probe",
        )


class ScaleControls(QFrame):
    """Compact Fusion-style percentage/size controls for the scale gizmo."""

    scalePercentChanged = pyqtSignal(int, float)
    sizeChanged = pyqtSignal(int, float)

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

        self.uniform_scale = QCheckBox("Равномерное масштабирование")
        self.uniform_scale.setChecked(True)
        self.uniform_scale.setToolTip("Сохранять соотношение сторон модели")
        layout.addWidget(self.uniform_scale, 3, 0, 1, 4)

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

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("transformToolbar")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(6)
        layout.addLayout(tool_row)

        tool_label = QLabel("Инструмент:")
        tool_label.setObjectName("fieldLabel")
        tool_row.addWidget(tool_label)

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

        self.uniform_scale = QCheckBox("Пропорционально")
        self.uniform_scale.setChecked(True)
        self.uniform_scale.setToolTip("Сохранять пропорции при масштабировании")
        tool_row.addWidget(self.uniform_scale)

        tool_row.addStretch(1)

        self.center_btn = QPushButton("Центрировать")
        self.center_btn.setToolTip("Переместить модель в центр XY")
        self.autofit_btn = QPushButton("Авто-фит под колбу")
        self.autofit_btn.setToolTip("Вписать модель в текущий размер колбы")
        tool_row.addWidget(self.center_btn)
        tool_row.addWidget(self.autofit_btn)

        self.scale_controls = ScaleControls(self)
        self.scale_controls.setVisible(False)
        layout.addWidget(self.scale_controls, 0, Qt.AlignmentFlag.AlignRight)

        self._wire_signals()
        self.set_enabled_state(False)

    def _wire_signals(self) -> None:
        for button, mode in zip(self._mode_buttons, ("move", "rotate", "scale")):
            button.clicked.connect(lambda _checked=False, m=mode: self._on_mode_clicked(m))
        self.center_btn.clicked.connect(self.centerRequested.emit)
        self.autofit_btn.clicked.connect(self.autoFitRequested.emit)
        self.scale_controls.uniform_scale.toggled.connect(self._sync_uniform_checkboxes)
        self.uniform_scale.toggled.connect(self._sync_detail_uniform_checkbox)

    def _on_mode_clicked(self, mode: str) -> None:
        self.scale_controls.setVisible(mode == "scale")
        self.modeChanged.emit(mode)

    def _sync_uniform_checkboxes(self, checked: bool) -> None:
        with QSignalBlocker(self.uniform_scale):
            self.uniform_scale.setChecked(checked)

    def _sync_detail_uniform_checkbox(self, checked: bool) -> None:
        with QSignalBlocker(self.scale_controls.uniform_scale):
            self.scale_controls.uniform_scale.setChecked(checked)

    def is_uniform(self) -> bool:
        return self.uniform_scale.isChecked()

    def sync_from_model(self, node: ModelNode) -> None:
        self.scale_controls.sync_from_model(node)

    def set_enabled_state(self, enabled: bool) -> None:
        for widget in (*self._mode_buttons, self.uniform_scale, self.center_btn, self.autofit_btn):
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

        self.uniform_scale = QCheckBox("Uniform Scale (сохранять пропорции)")
        self.uniform_scale.setChecked(True)
        transform_layout.addWidget(self.uniform_scale)
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
        return self.uniform_scale.isChecked()

    # --- состояние -------------------------------------------------------------
    def set_enabled_state(self, enabled: bool) -> None:
        for w in (*self._mode_buttons, self.uniform_scale,
                  self.center_btn, self.autofit_btn, self.reset_btn):
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
