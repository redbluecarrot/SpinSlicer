"""
ui_panels.py
============
Виджеты настроек и управления моделью.

  ProcessSettingsPanel — настройки принтера и процесса, вынесенные в
    отдельную вкладку приложения.
  TransformToolbar — компактное управление gizmo над 3D-вьюпортом.
  ScaleControls — компактная Fusion-style панель процентов и размеров.
  ObjectPanel — прежняя расширенная панель объекта, оставленная для обратной
    совместимости с внешними импортами.

Виджеты ничего не знают про PyVista напрямую — связь с ModelNode и сценой
делает SlicerTab.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np
from PyQt6.QtCore import QSettings, QSignalBlocker, Qt, QTimer, pyqtSignal
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
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from spinslicer import theme
from spinslicer.constants import (
    APP_ORG,
    APP_TITLE,
    DEFAULT_DIAMETER_MM,
    DEFAULT_FRAME_RATE_HZ,
    DEFAULT_NUM_FRAMES,
    DEFAULT_OUTPUT_RESOLUTION,
    FILL_FRACTION,
    VAT_HEIGHT_RATIO,
)
from spinslicer.i18n import num, tr
from spinslicer.model_node import ModelNode
from spinslicer.pixel_game import PixelGame
from spinslicer.projector_canvas import (
    RESOLUTIONS,
    CanvasGeometry,
    cone_smear_mm,
    throw_distance_mm,
)
from spinslicer.sketch_scene import SceneParams
from spinslicer.theme import AXIS_X, AXIS_Y, AXIS_Z
from spinslicer.ui_kit import Segmented, SettingRow, Stepper, Switch, section_head


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
    "setup": ("НАСТРОЙКА · ОДИН РАЗ", "Настройки"),
    "model": ("ШАГ 1 ИЗ 3", "Модель"),
    "light": ("ШАГ 2 ИЗ 3", "Свет"),
    "print": ("ШАГ 3 ИЗ 3", "Печать"),
}


_PRINTER_STATE_COLORS = {
    "off": "#6b6f76",     # not connected
    "busy": "#f0b429",    # connecting
    "ok": "#34c759",      # connected
    "error": "#ff453a",   # the printer does not answer
}


class _StatusDot(QLabel):
    """A round colour light: grey off, amber connecting, green connected, red error."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("printerDot")
        self.setFixedSize(10, 10)
        self.set_state("off")

    def set_state(self, state: str) -> None:
        color = _PRINTER_STATE_COLORS.get(state, _PRINTER_STATE_COLORS["off"])
        self.setStyleSheet(f"QLabel#printerDot {{ background-color: {color}; border-radius: 5px; }}")
        self.setProperty("state", state)


class _Tile(QFrame):
    """A number with its name: the result of a run, the size of a print. Easier to read than a sentence."""

    def __init__(self, key: str) -> None:
        super().__init__()
        self.setObjectName("statTile")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(1)
        self.key = QLabel(tr(key))
        self.key.setObjectName("tileKey")
        self.value = QLabel("—")
        self.value.setObjectName("tileValue")
        self.value.setProperty("_spinslicer_skip_text_translation", True)
        self.note = QLabel("")
        self.note.setObjectName("tileNote")
        self.note.setWordWrap(True)
        self.note.setProperty("_spinslicer_skip_text_translation", True)
        for w in (self.key, self.value, self.note):
            lay.addWidget(w)
        lay.addStretch(1)              # the numbers of a row of tiles on one line

    def set(self, value: str, note: str = "", warn: bool = False) -> None:
        self.value.setText(value)
        self.note.setText(tr(note) if note else "")
        self.note.setVisible(bool(note))
        self.value.setProperty("warn", warn)
        style = self.value.style()
        if style is not None:            # re-read the [warn] rule of the stylesheet
            style.unpolish(self.value)
            style.polish(self.value)


def _tiles(tiles: Sequence[QWidget], columns: int = 2) -> QWidget:
    box = QWidget()
    grid = QGridLayout(box)
    grid.setContentsMargins(0, 2, 0, 2)
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(8)
    for i, tile in enumerate(tiles):
        grid.addWidget(tile, i // columns, i % columns)
    for c in range(columns):
        grid.setColumnStretch(c, 1)
    return box


SETUP_TABS: tuple[tuple[str, str], ...] = (
    ("interface", "Интерфейс"), ("printer", "Принтер"), ("projector", "Проектор"), ("resin", "Смола"))


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
    lineStyleChanged = pyqtSignal(str)
    languageChanged = pyqtSignal(int)
    themeChanged = pyqtSignal(bool)          # Settings → Интерфейс → Тема: True = dark

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("sidePanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._settings = QSettings(APP_ORG, APP_TITLE)

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
        self.todo.setVisible(False)        # the step hints are gone; the label only keeps old callers working
        root.addWidget(head)

        self._stack = QStackedWidget()
        self._pages: dict[str, QWidget] = {}
        root.addWidget(self._stack, 1)

        foot = QWidget()
        foot_l = QVBoxLayout(foot)
        foot_l.setContentsMargins(20, 10, 20, 16)      # the same edges as the blocks above
        foot_l.setSpacing(8)
        self._foot = foot
        self._foot_l = foot_l
        # The footer never gets less height than its content: the print buttons stay whole.
        foot.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
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
        self._layout_setup()
        self._restore()
        self._wire()
        self.show_page("model")
        self._refresh()

    # ------------------------------------------------------------------ build helpers
    def _add_page(self, key: str, rows: list[QWidget], footer: Optional[QWidget] = None) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 6, 20, 18)
        layout.setSpacing(12)            # air between the blocks
        for row in rows:
            layout.addWidget(row)
        layout.addStretch(1)
        if footer is not None:           # pinned to the bottom of the page, inside its scroll area
            layout.addWidget(footer)
        # The page is wrapped so the window can be narrower than its widest row: a row that is
        # too wide scrolls sideways instead of pushing the buttons off the screen. Only Settings
        # (long, about 1300 px) scrolls down; the other steps show their whole page.
        scroll = _scroll_wrap(content)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # The print step carries the buttons at its bottom edge: when the window is short it
        # scrolls, so those buttons are never cut off.
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if key in ("setup", "print", "light")
                                          else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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

    def _button(self, text: str, action: str) -> QPushButton:
        """A filled button for the main action of a row."""
        b = QPushButton(tr(text))
        b.setObjectName("primaryAction")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(lambda: self.actionRequested.emit(action))
        return b

    def _secondary(self, text: str, action: str) -> QPushButton:
        """An outlined button: the second choice next to a filled one."""
        b = QPushButton(tr(text))
        b.setObjectName("secondaryAction")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(lambda: self.actionRequested.emit(action))
        return b

    def _small_link(self, text: str, action: str) -> QPushButton:
        """A small grey link: things that are there, but not the main way."""
        b = QPushButton(tr(text))
        b.setObjectName("smallLink")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(lambda: self.actionRequested.emit(action))
        return b

    def _link(self, text: str, action: str) -> QPushButton:
        b = QPushButton(tr(text))
        b.setObjectName("linkButton")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(lambda: self.actionRequested.emit(action))
        return b

    def _expander(self, title: str, body: QWidget) -> QWidget:
        toggle = QPushButton(tr("показать"))
        toggle.setObjectName("linkButton")
        toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        row = self._row(title, toggle)
        body.setVisible(False)

        def flip() -> None:
            body.setVisible(not body.isVisible())
            toggle.setText(tr("скрыть") if body.isVisible() else tr("показать"))
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

    @staticmethod
    def _section(title: str, rows: list[QWidget]) -> QWidget:
        """A block of the menu: a rounded container with its heading and rows inside."""
        box = QFrame()
        box.setObjectName("settingsBlock")
        box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(18, 14, 18, 12)
        lay.setSpacing(0)
        head = section_head(title)
        head.setStyleSheet("padding-top: 0px; padding-bottom: 6px;")
        lay.addWidget(head)
        for row in rows:
            lay.addWidget(row)
        return box

    # ------------------------------------------------------------------ pages
    def _build_setup(self) -> None:
        self.diameter = Stepper(20.0, 200.0, DEFAULT_DIAMETER_MM, 1.0, 1, "мм")
        self.glass_thickness = Stepper(0.5, 10.0, 2.0, 0.5, 1, "мм")
        self.placement = Segmented([("в ванне", 1), ("в воздухе", 0)], 0)
        self.bath = self.placement   # back-compat name; use container_mode_value()
        self.field_height = Stepper(5.0, 400.0, 40.0, 1.0, 0, "мм")
        self._field_auto = True
        self.resolution = Segmented([(f"{w}×{h}", i) for i, (w, h) in enumerate(RESOLUTIONS)], 2)
        self.output_res = Stepper(128, 3840, DEFAULT_OUTPUT_RESOLUTION, 8, 0, "px")
        self.rays = Segmented([("параллельные", 0), ("конусом", 1)], 0)
        self.throw_ratio = Stepper(0.2, 5.0, 1.2, 0.05, 2, ": 1")
        self.projector_blur = Stepper(0.0, 0.5, 0.0, 0.01, 2, "мм")
        self.offset_x = Stepper(-2000, 2000, 0, 1, 0, "px")
        self.offset_y = Stepper(-2000, 2000, 0, 1, 0, "px")
        self.projector_hz = Stepper(24.0, 360.0, 60.0, 1.0, 0, "Гц")
        self.projector_gamma = Stepper(0.5, 3.0, 2.2, 0.1, 2)
        self.projector_contrast = Stepper(0.0, 10000.0, 1000.0, 100.0, 0, ":1")
        self.projector_irradiance = Stepper(0.0, 200.0, 0.0, 0.5, 2, "мВт/см²")
        self.n_glass = Stepper(1.0, 2.0, 1.51, 0.01, 2)
        self.n_liquid = Stepper(1.0, 2.0, 1.45, 0.01, 2)
        self.alpha_absorption = Stepper(0.0, 0.1, 0.0, 0.001, 3, "/мм")
        self.n_resin = Stepper(1.0, 2.0, 1.50, 0.01, 2)
        self.gel_dose = Stepper(0.0, 5000.0, 0.0, 10.0, 1, "мДж/см³")
        self.hints_switch = Switch(False)
        self.blur_compensation = Switch(True)
        self.line_style = Segmented([("прямые", "straight"), ("от руки", "sketch"), ("живые", "live")], 0)
        self.language_box = NoScrollComboBox()
        # The printer on the network (printer_link): its profile fills the settings below.
        self.printer_box = NoScrollComboBox()
        self.printer_box.setMinimumWidth(150)
        self.printer_box.addItem(tr("не выбран"), None)
        self._printer_profile_id = ""

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
        self.r_comp = self._row("Компенсация размытия", self.blur_compensation)
        self.r_alpha = self._row("Поглощение смолы", self.alpha_absorption,
                                 hint="натуральный логарифм: α = 2,303·A на 1 мм = 1 / Dp")
        self.r_nresin = self._row("Преломление смолы", self.n_resin)
        self.r_gel = self._row("Доза геля", self.gel_dose, hint="не знаете? напечатайте шкалу ↓")
        self.r_calib = self._row("Шкала дозы", self._link("напечатать →", "calibrate"),
                                 "9 столбиков с дозой 88–112 %: выбираете лучший, доза обновится сама")
        self.r_hz = self._row("Частота проектора", self.projector_hz,
                              hint="герцовка экрана: кадры не могут сменяться чаще")
        self.r_offx = self._row("Сдвиг картинки по X", self.offset_x)
        self.r_offy = self._row("Сдвиг картинки по Y", self.offset_y)
        self.r_gamma = self._row("Гамма", self.projector_gamma)
        self.r_contrast = self._row("Контраст", self.projector_contrast)
        self.r_irr = self._row("Облучённость", self.projector_irradiance)
        self.r_nglass = self._row("n стекла", self.n_glass)
        self.r_nliquid = self._row("n жидкости ванны", self.n_liquid)
        self.r_hints = self._row("Подсказки маркером", self.hints_switch, "для тех, кто печатает впервые")
        self.r_draw = self._row("Линии", self.line_style, stacked=True,
                                effect="прямые — по линейке, модель без точек; от руки — неподвижный набросок; живые — набросок «дышит»")
        self.r_language = self._row("Язык", self.language_box)
        self.theme_choice = Segmented([("светлая", 0), ("тёмная", 1)], 1 if theme.current().dark else 0)
        self.theme_choice.changed.connect(lambda i: self.themeChanged.emit(i == 1))
        self.r_theme = self._row("Тема", self.theme_choice, stacked=True)
        self.r_printer = self._row("Принтер", self.printer_box,
                                   effect="SpinPrinter в сети: его профиль заполнит колбу и проектор")
        # The state of the printer, in one line: a coloured light and what it means.
        status = QWidget()
        stl = QHBoxLayout(status)
        stl.setContentsMargins(0, 2, 0, 8)
        stl.setSpacing(10)
        self.printer_dot = _StatusDot()
        self.printer_state = QLabel(tr("не подключён"))
        self.printer_state.setObjectName("printerState")
        self.printer_state.setWordWrap(True)
        self.printer_state.setProperty("_spinslicer_skip_text_translation", True)
        stl.addWidget(self.printer_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        stl.addWidget(self.printer_state, 1)
        self.printer_status = status
        printer_tools = QWidget()
        ptl = QHBoxLayout(printer_tools)
        ptl.setContentsMargins(0, 0, 0, 0)
        ptl.setSpacing(16)
        self.printer_toggle = QPushButton()
        self.printer_toggle.setObjectName("secondaryAction")
        self.printer_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.printer_toggle.clicked.connect(self._on_printer_toggle)
        self.printer_toggle.setStyleSheet("min-height: 38px; border-radius: 14px; font-size: 13px; padding: 0 20px;")
        ptl.addWidget(self.printer_toggle)
        # Searching and typing an address are for choosing a printer: hidden while one is connected.
        self.printer_find_link = self._link("найти в сети", "printer_find")
        self.printer_manual_link = self._link("ввести адрес", "printer_manual")
        ptl.addWidget(self.printer_find_link)
        ptl.addWidget(self.printer_manual_link)
        ptl.addStretch(1)
        printer_tools.setContentsMargins(0, 0, 0, 6)
        self.r_printer_tools = printer_tools
        self.printer_write = self._link("записать в принтер", "printer_write")
        self.r_printer_write = self._row("Настройки этого слайсера", self.printer_write,
                                         "колба, проектор и калибровка, изменённые здесь, сохранятся в принтере")
        self.r_printer_write.setVisible(False)
        tools = QWidget()
        tl = QHBoxLayout(tools)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(16)
        tl.addWidget(self._link("разметка колбы", "pattern"))
        tl.addWidget(self._link("тест серого", "ramp"))
        tl.addStretch(1)
        self.r_tools = self._row("Проверить на проекторе", tools,
                                 "разметка покажет, где должны быть стенки колбы; тест серого — все уровни яркости",
                                 stacked=True)
        self._expert_group = self._group([self.r_blur, self.r_comp, self.r_gamma, self.r_nglass, self.r_nliquid])

    def _layout_setup(self) -> None:
        """Settings, in four tabs: what the window looks like, the printer (its connection, vial,
        calibration and motor), the projector, the resin. One tab on screen at a time."""
        pages = {
            "interface": [self._section("Внешний вид", [self.r_theme, self.r_language, self.r_draw, self.r_hints])],
            "printer": [
                self._section("Подключение", [self.printer_status, self.r_printer, self.r_printer_tools,
                                              self.r_printer_write]),
                self._section("Колба", [self.r_diameter, self.r_wall, self.r_bath, self.r_field]),
                self._section("Калибровка на проекторе", [self.r_output, self.r_offx, self.r_offy, self.r_tools]),
                self._section("Мотор", [self.r_speed]),
            ],
            "projector": [
                self._section("Проектор", [self.r_resolution, self.r_hz, self.r_rays, self.r_throw, self.r_brightness]),
                self._section("Свет", [self.r_irr, self.r_contrast,
                                       self._expander("Тонкая настройка", self._expert_group)]),
            ],
            "resin": [
                self._section("Смола", [self.r_alpha, self.r_nresin, self.r_gel]),
                self._section("Калибровка смолы", [self.r_calib]),
            ],
        }
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(20, 2, 20, 0)
        lay.setSpacing(10)
        self.setup_tabs = Segmented([(title, key) for key, title in SETUP_TABS])
        lay.addWidget(self.setup_tabs)
        self._setup_stack = QStackedWidget()
        self._setup_pages: dict[str, QWidget] = {}
        for key, blocks in pages.items():
            content = QWidget()
            cl = QVBoxLayout(content)
            cl.setContentsMargins(2, 4, 2, 18)
            cl.setSpacing(14)
            for block in blocks:
                cl.addWidget(block)
            cl.addStretch(1)
            scroll = _scroll_wrap(content)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._setup_pages[key] = scroll
            self._setup_stack.addWidget(scroll)
        lay.addWidget(self._setup_stack, 1)
        self.setup_tabs.changed.connect(lambda i: self.show_setup_tab(SETUP_TABS[i][0]))
        self._pages["setup"] = box
        self._stack.addWidget(box)
        stored = str(self._settings.value("ui/setup_tab", "printer"))
        self.show_setup_tab(stored if stored in self._setup_pages else "printer")

    def show_setup_tab(self, key: str) -> None:
        """Settings → one of interface / printer / projector / resin."""
        if key not in self._setup_pages:
            return
        index = [k for k, _t in SETUP_TABS].index(key)
        if self.setup_tabs.currentIndex() != index:
            with QSignalBlocker(self.setup_tabs):
                self.setup_tabs.setCurrentIndex(index)
        self._setup_stack.setCurrentWidget(self._setup_pages[key])
        self._settings.setValue("ui/setup_tab", key)

    def set_theme_silent(self, dark: bool) -> None:
        with QSignalBlocker(self.theme_choice):
            self.theme_choice.setCurrentIndex(1 if dark else 0)

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
        self._model_rows = [self.r_orient, self.r_scale, self.r_place, fit_box]
        self._model_expander = self._expander("Точная позиция", self.transform_panel)
        self._add_page("model", [self._section("Деталь", [self.r_file, self.r_voids]),
                                 self._section("Положение", [*self._model_rows, self._model_expander])])

    def _build_light(self) -> None:
        self.grid_res = Segmented([("Быстро", 128), ("Норма", 256), ("Точно", 384)], 1)
        self.frames = Segmented([("180", 180), ("360", 360), ("720", 720)], 1)
        self.virtual = Switch(False)
        self._virtual_ready = False
        self._review = False          # "пересчитать": back to the parameters while a result exists
        self.r_grid = self._row("Качество", self.grid_res, stacked=True, hint="для первой печати хватит «Норма»")
        self.r_frames = self._row("Кадров на оборот", self.frames, stacked=True)
        self.r_virtual = self._row("Виртуальная печать", self.virtual, "показать в 3D, что затвердеет")
        self.r_virtual.setVisible(False)          # only after a computation
        self.tile_match = _Tile("совпадение")
        self.tile_tol = _Tile("запас экспозиции")
        self.tile_time = _Tile("время печати")
        self.tile_frames = _Tile("кадров")
        self.stale_label = QLabel(tr("Деталь или настройки изменились: прошлый результат устарел."))
        self.stale_label.setObjectName("warnLine")
        self.stale_label.setWordWrap(True)
        self.stale_label.setVisible(False)
        # Notes from the engine: folded into one line, opened on demand.
        self.warnings_toggle = QPushButton()
        self.warnings_toggle.setObjectName("smallLink")
        self.warnings_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.warnings_toggle.setProperty("_spinslicer_skip_text_translation", True)
        self.warnings_toggle.clicked.connect(self._flip_warnings)
        self.warnings_label = QLabel()
        self.warnings_label.setObjectName("rowEffect")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setProperty("_spinslicer_skip_text_translation", True)
        self.warnings_label.setVisible(False)
        self.calc_settings = QLabel()
        self.calc_settings.setObjectName("rowEffect")
        self.calc_settings.setWordWrap(True)
        self.calc_settings.setProperty("_spinslicer_skip_text_translation", True)
        again = QPushButton(tr("пересчитать с другими параметрами"))
        again.setObjectName("smallLink")
        again.setCursor(Qt.CursorShape.PointingHandCursor)
        again.clicked.connect(self.review_parameters)
        results = QWidget()
        rl = QVBoxLayout(results)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        rl.addWidget(self.calc_settings)
        rl.addWidget(_tiles([self.tile_match, self.tile_tol, self.tile_time, self.tile_frames]))
        rl.addWidget(self.warnings_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        rl.addWidget(self.warnings_label)
        rl.addWidget(self.r_virtual)
        rl.addWidget(again, 0, Qt.AlignmentFlag.AlignLeft)
        self.results_box = results

        self.game = PixelGame()
        self.game_title = QLabel(tr("Пока ждёте — засветите деталь сами"))
        self.game_title.setObjectName("rowLabel")
        self.game_title.setProperty("_spinslicer_skip_text_translation", True)
        game_body = QWidget()
        gl = QVBoxLayout(game_body)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(6)
        gl.addWidget(self.game_title)
        gl.addWidget(self._arcade(self.game))
        # Before the run: the parameters and the game. During it: only the game. After it: the result.
        self.calc_block = self._section("Параметры расчёта", [self.stale_label, self.r_grid, self.r_frames])
        self.game_box = self._section("Игра", [game_body])
        self.result_block = self._section("Результат", [results])
        self._add_page("light", [self.calc_block, self.game_box, self.result_block])
        self._show_light_stage()

    def _flip_warnings(self) -> None:
        self.warnings_label.setVisible(not self.warnings_label.isVisible())
        self._refresh_plan()

    def review_parameters(self) -> None:
        """"Пересчитать": the parameters (and the game) again, the result stays until a new run."""
        self._review = True
        self._show_light_stage()
        self._refresh_action()

    def _build_print(self) -> None:
        # Kept for the old projector / MP4 / PNG paths (now small links on the print page) and settings.
        self.output_kind = Segmented([("на проектор", 0), ("видео MP4", 1), ("кадры PNG", 2)])
        self.speed = Stepper(1.0, 240.0, DEFAULT_FRAME_RATE_HZ * 360.0 / DEFAULT_NUM_FRAMES, 1.0, 0, "°/с")
        self.brightness = Stepper(10.0, 100.0, 100.0, 5.0, 0, "%")
        self.r_output_kind = self._row("Куда выводить", self.output_kind, stacked=True)
        self.r_speed = self._row("Скорость вращения", self.speed)
        self.r_brightness = self._row("Яркость при печати", self.brightness)

        # The printer, as on Settings → Принтер: its light and what it is doing.
        card = QFrame()
        card.setObjectName("settingsBlock")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        cl = QHBoxLayout(card)
        cl.setContentsMargins(16, 12, 16, 12)
        cl.setSpacing(10)
        self.print_dot = _StatusDot()
        self.print_printer = QLabel(tr("Принтер не подключён"))
        self.print_printer.setObjectName("printerState")
        self.print_printer.setWordWrap(True)
        self.print_printer.setProperty("_spinslicer_skip_text_translation", True)
        cl.addWidget(self.print_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        cl.addWidget(self.print_printer, 1)
        to_printer = self._link("настройки принтера", "goto:setup")
        to_printer.clicked.connect(lambda: self.show_setup_tab("printer"))
        cl.addWidget(to_printer, 0, Qt.AlignmentFlag.AlignVCenter)
        self.printer_card = card

        self.ptile_time = _Tile("время")
        self.ptile_revs = _Tile("оборотов")
        self.ptile_tol = _Tile("запас")
        self.ptile_fps = _Tile("кадры")
        self.plan_label = QLabel(tr("Сначала рассчитайте свет."))    # the old text plan, kept for callers
        self.plan_label.setVisible(False)
        # The dose card (video_tab.DoseCard) is put here by the main window: its slider
        # is the print exposure, and the numbers above follow it.
        self.dose_slot = QWidget()
        self._dose_layout = QVBoxLayout(self.dose_slot)
        self._dose_layout.setContentsMargins(0, 0, 0, 0)
        self._dose_card: Optional[QWidget] = None
        self._exposure = 1.0
        self._projecting: Optional[tuple[float, float]] = None
        self._build_dock()
        self._add_page("print", [self.printer_card,
                                 _tiles([self.ptile_time, self.ptile_revs, self.ptile_tol, self.ptile_fps], 4),
                                 self._section("Доза", [self.dose_slot, self.plan_label])], footer=self._dock)

    def _build_dock(self) -> None:
        """The two ways to print: over the network on the SpinPrinter, or a .spin on a USB stick.
        The others (the projector of this computer, MP4, PNG, a folder) are small links under them."""
        self._dock = QWidget()
        dock_l = QVBoxLayout(self._dock)
        dock_l.setContentsMargins(0, 0, 0, 0)
        dock_l.setSpacing(8)
        buttons = QWidget()
        bl = QHBoxLayout(buttons)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)
        bl.setSizeConstraint(QHBoxLayout.SizeConstraint.SetMinimumSize)
        self.print_button = self._button("Напечатать", "spin_print")
        self.stick_button = self._secondary("На флешку", "spin_stick")
        bl.addWidget(self.print_button, 3)
        bl.addWidget(self.stick_button, 2)
        self.r_spin = self._row("", None, "")
        self.r_spin.setObjectName("printStatus")
        self.stop_link = self._link("остановить печать", "spin_stop")
        self.stop_link.setVisible(False)
        more = QWidget()
        ml = QHBoxLayout(more)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(4)
        self.more_links = [self._small_link("на проектор этого компьютера", "projector"),
                           self._small_link("видео MP4", "export"),
                           self._small_link("кадры PNG", "open_frames"),
                           self._small_link(".spin в папку", "spin_export")]
        for i, link in enumerate(self.more_links):
            if i:
                dot = QLabel("·")
                dot.setObjectName("smallDot")
                ml.addWidget(dot)
            ml.addWidget(link)
        ml.addStretch(1)
        dock_l.addWidget(buttons)
        dock_l.addWidget(self.r_spin)
        dock_l.addWidget(self.stop_link, 0, Qt.AlignmentFlag.AlignLeft)
        dock_l.addWidget(more)
        self._dock.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        dock_l.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)
        self._printer_on = False
        self._print_status = ("", False)
        self.set_printer_connected(False)

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
        """A game with its picker."""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 4, 0, 8)
        lay.setSpacing(6)
        picker = Segmented([(g.title, i) for i, g in enumerate(game.games)])

        def choose(index: int) -> None:
            game.set_game(index)
            self._settings.setValue("game/last", index)
        picker.changed.connect(choose)
        self._arcade_refresh = getattr(self, "_arcade_refresh", []) + [lambda: choose(picker.currentIndex())]
        last = int(self._settings.value("game/last", 0) or 0)
        picker.setCurrentIndex(last)
        choose(last)
        lay.addWidget(picker)
        # centred when it is held narrow (the preview before a run), full width otherwise
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(game, 100)
        row.addStretch(1)
        lay.addLayout(row)
        return box

    # ------------------------------------------------------------- settings
    def _persisted(self) -> dict[str, QWidget]:
        return {
            "ui/diameter": self.diameter, "ui/wall": self.glass_thickness, "ui/placement": self.placement,
            "ui/resolution2": self.resolution, "ui/output_res": self.output_res,
            "ui/rays": self.rays, "ui/throw_ratio": self.throw_ratio,
            "physics/blur": self.projector_blur, "ui/offset_x": self.offset_x, "ui/offset_y": self.offset_y,
            "ui/refresh_hz": self.projector_hz, "physics/gamma": self.projector_gamma, "physics/contrast": self.projector_contrast,
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
        hints = str(self._settings.value("ui/hints", False)).lower() in {"1", "true", "yes"}
        self.hints_switch.setChecked(hints)
        self._field_auto = True
        self._sync_field()
        style = str(self._settings.value("ui/line_style", "straight"))
        index = {"straight": 0, "sketch": 1, "live": 2}.get(style, 0)
        self.line_style.setCurrentIndex(index)

    def _save(self, key: str, widget: QWidget) -> None:
        if isinstance(widget, Switch):
            self._settings.setValue(key, widget.isChecked())
        else:
            self._settings.setValue(key, widget.value())  # type: ignore[attr-defined]

    def _wire(self) -> None:
        for geometry in (self.diameter, self.glass_thickness, self.output_res, self.resolution):
            geometry.valueChanged.connect(lambda _v: self._sync_field(force=True))   # before the refresh below
        self.field_height.valueChanged.connect(self._on_field_edited)
        self.field_height.valueChanged.connect(lambda _v: self._on_setting_changed("ui/field"))
        self.field_height.valueChanged.connect(lambda _v: self.sceneChanged.emit())
        scene_keys = {"ui/diameter", "ui/wall", "ui/placement", "ui/field", "ui/rays", "ui/throw_ratio",
                      "ui/resolution2", "ui/output_res", "physics/alpha", "ui/n_resin", "ui/n_liquid", "ui/frames"}
        canvas_keys = {"ui/resolution2", "ui/offset_x", "ui/offset_y", "ui/speed", "ui/frames"}
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
        self.line_style.changed.connect(self._on_line_style)
        self.language_box.currentIndexChanged.connect(self.languageChanged.emit)
        self.output_kind.changed.connect(lambda _i: self._refresh_action())

    def _auto_field_mm(self) -> float:
        """Height of the projector picture in the vial: its rows times the pixel size."""
        sh = self.screen_size()[1]
        height = sh * self.diameter.value() / max(self.output_res.value(), 1.0)
        zone = (self.diameter.value() - 2 * self.glass_thickness.value()) * FILL_FRACTION
        return float(max(5.0, min(400.0, round(min(height, zone)), self.diameter.value() * VAT_HEIGHT_RATIO)))

    def _sync_field(self, force: bool = False) -> None:
        if force:
            self._field_auto = True
        if self._field_auto:
            with QSignalBlocker(self.field_height):
                self.field_height.setValue(self._auto_field_mm())

    def _on_field_edited(self, _value: float) -> None:
        self._field_auto = False

    def _on_line_style(self, _index: int) -> None:
        style = str(self.line_style.currentData())
        self._settings.setValue("ui/line_style", style)
        self.lineStyleChanged.emit(style)

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
                                 "ui/offset_y", "ui/resolution2"})

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

    def optimizer_iterations_value(self) -> int:
        return self.optimizer_iterations.value_int()

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

    # ------------------------------------------------------------- network printer
    def set_printers(self, printers: Sequence[object], keep: Optional[object] = None) -> None:
        """Fill the printer list (printer_link.PrinterAddress items); keeps the chosen one if listed."""
        current = keep if keep is not None else self.selected_printer()
        with QSignalBlocker(self.printer_box):
            self.printer_box.clear()
            self.printer_box.addItem(tr("не выбран"), None)
            for printer in printers:
                self.printer_box.addItem(getattr(printer, "label", str(printer)), printer)
            if current is not None:
                index = next((i for i in range(1, self.printer_box.count())
                              if self.printer_box.itemData(i) == current), -1)
                if index < 0:
                    self.printer_box.addItem(getattr(current, "label", str(current)), current)
                    index = self.printer_box.count() - 1
                self.printer_box.setCurrentIndex(index)

    def selected_printer(self) -> Optional[object]:
        return self.printer_box.currentData()

    def set_printer_connected(self, connected: bool, status: str = "", state: str = "") -> None:
        """The printer's light and state (Settings and the print page), the connect / disconnect
        button and the Print button follow the connection.

        state: "off" (not connected), "busy" (connecting), "ok" (connected), "error" (no answer).
        """
        state = state or ("ok" if connected else "off")
        self._printer_on = connected
        text = status or (tr("подключён") if connected else tr("не подключён"))
        for dot in (getattr(self, "printer_dot", None), getattr(self, "print_dot", None)):
            if dot is not None:
                dot.set_state(state)
        if hasattr(self, "printer_state"):
            self.printer_state.setText(text)
            self.print_printer.setText(text if connected or state in ("busy", "error")
                                       else tr("Принтер не подключён"))
        if hasattr(self, "r_printer_write"):
            self.r_printer_write.setVisible(connected)
        if hasattr(self, "printer_toggle"):
            self.printer_toggle.setText(tr("Отключить") if connected else tr("Подключить"))
            self.printer_toggle.setEnabled(state != "busy")
            self.printer_find_link.setVisible(not connected)
            self.printer_manual_link.setVisible(not connected)
        if hasattr(self, "_dock"):
            self._refresh_print_buttons()

    def _on_printer_toggle(self) -> None:
        self.actionRequested.emit("printer_disconnect" if self._printer_on else "printer_connect")

    def set_print_status(self, text: str, warn: bool = False, stoppable: bool = False) -> None:
        """What the print sent to the printer is doing, under the Print button ("" = the usual hint)."""
        self._print_status = (text, warn)
        self.stop_link.setVisible(stoppable)
        self._refresh_print_buttons()

    def _frames_ready(self) -> bool:
        return bool(self._output_dir) and not self._frames_outdated() and not self._running

    def _refresh_print_buttons(self) -> None:
        """Print needs frames and a connected printer, the stick only frames; the line under the
        buttons says what is missing instead of leaving a grey button to guess at."""
        ready = self._frames_ready()
        self.print_button.setEnabled(ready and self._printer_on)
        self.stick_button.setEnabled(ready)
        for link in self.more_links:
            link.setEnabled(ready)
        text, warn = self._print_status
        if not text:
            if not self._output_dir:
                text = "сначала рассчитайте свет"
            elif self._frames_outdated():
                text, warn = "кадры устарели — пересчитайте свет", True
            elif not self._printer_on:
                text = "принтер не подключён: Настройки → Принтер. На флешку можно и так"
            else:
                text = "файл уйдёт на принтер, его экран спросит «Печатать?»"
        self.r_spin.set_effect(text, warn)
        self.print_button.setToolTip("" if self._printer_on else tr("Подключите принтер: Настройки → Принтер"))

    def printer_values(self) -> dict[str, float]:
        """This slicer's printer settings, in printer_link.profile_settings keys (to write back)."""
        return {"vial_diameter_mm": float(self.diameter.value()), "vial_wall_mm": float(self.glass_thickness.value()),
                "n_glass": float(self.n_glass.value()), "irradiance_mw_cm2": float(self.projector_irradiance.value()),
                "contrast": float(self.projector_contrast.value()), "vial_width_px": int(self.output_res.value_int()),
                "offset_x_px": int(self.offset_x.value())}

    def printer_profile_id(self) -> str:
        return self._printer_profile_id

    def apply_printer_profile(self, settings: dict) -> list[str]:
        """Put a printer profile (printer_link.profile_settings) into the settings; returns what changed.

        Only values that differ count as changed: the list is shown as "from the printer: …", and
        it named every field the printer sent, even ones that were already the same."""
        self._printer_profile_id = str(settings.get("profile_id", ""))
        changed: list[str] = []

        def put(key: str, widget: Stepper, label: str) -> None:
            if key not in settings:
                return
            before = widget.value()
            widget.setValue(float(settings[key]))
            if abs(widget.value() - before) > 1e-9:
                changed.append(label)
        put("vial_diameter_mm", self.diameter, "диаметр колбы")
        put("vial_wall_mm", self.glass_thickness, "стенка")
        size = (settings.get("projector_width"), settings.get("projector_height"))
        if size in RESOLUTIONS:
            index = float(RESOLUTIONS.index(size))  # type: ignore[arg-type]
            if self.resolution.value() != index:
                self.resolution.setValue(index)
                changed.append("разрешение проектора")
        put("projector_hz", self.projector_hz, "частота проектора")
        put("irradiance_mw_cm2", self.projector_irradiance, "облучённость")
        put("contrast", self.projector_contrast, "контраст")
        put("n_glass", self.n_glass, "n стекла")
        put("vial_width_px", self.output_res, "колба на экране")
        put("offset_x_px", self.offset_x, "сдвиг по X")
        put("offset_y_px", self.offset_y, "сдвиг по Y")
        return changed

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
        from spinslicer.optics.fastcal import detector_bins
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
        cone = self.rays.value_int() == 1
        self.r_throw.setVisible(cone)
        self.r_nliquid.setVisible(bath)
        vial_h = self.diameter.value() * VAT_HEIGHT_RATIO
        self.r_field.set_effect(
            f"печатать можно до {f(v['height'], 0)} мм в высоту"
            + (f" (ограничивает колба {f(vial_h, 0)} мм)" if vial_h < self.field_height.value() else ""))
        sw, sh = self.screen_size()
        g = self.canvas_geometry()
        fits = self.output_res.value_int() <= sw
        hz = self.projector_hz.value()
        self.r_hz.set_effect(f"кадр на экране ≥ {f(1000 / max(hz, 1), 1)} мс")
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
                               else "конус проектора проходит линзу Френеля и становится параллельным")
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
                                + (f" — выше частоты проектора ({f(hz, 0)} Гц)" if fps > hz else ""), warn=fps > hz)
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
                self.action_sub.setText("")
        elif page == "light":
            if self._running:
                self.action_btn.setText(tr("Остановить"))
                self.action_sub.setText(tr("считаю кадры…"))
            elif self._done and not self._review:
                self.action_btn.setText(tr("Дальше — печать →"))
                self.action_sub.setText(tr(f"готово · {self.frames.value_int()} кадров"))
            else:
                self.action_btn.setText(tr("Рассчитать свет"))
                self.action_btn.setEnabled(self._has_model)
                self.action_sub.setText(tr(f"≈ {self._secs(self.derived()['calc_s'])} на обычном компьютере"
                                           if self._has_model else "сначала загрузите модель"))
        elif self._projecting is not None:
            elapsed, total = self._projecting
            self.action_btn.setText(tr("Остановить засветку"))
            self.action_sub.setText(tr(f"идёт засветка {self._secs(elapsed)} из {self._secs(total)}") if total
                                    else tr("идёт засветка · Esc на проекторе — стоп"))
        elif self._frames_outdated():
            self.action_btn.setText(tr("Пересчитать свет"))
            self.action_sub.setText(tr("старые кадры не подходят к новым настройкам"))
        else:
            self.action_btn.setText(tr("Рассчитать свет →"))
            self.action_sub.setText(tr("без кадров печатать нечего"))
        # The print page has its own buttons (network, stick); the big action shows there only to
        # stop this computer's projector or to send you back to compute the light.
        on_print = page == "print"
        print_ready = self._frames_ready() and self._projecting is None
        self._foot.setVisible(not (on_print and print_ready))
        if hasattr(self, "_dock"):
            self._dock.setVisible(on_print)
            self._refresh_print_buttons()
        for row in self._model_rows:
            row.setVisible(self._has_model)
        self._model_expander.setVisible(self._has_model)

    def _refresh_plan(self) -> None:
        plan = self._plan
        f = self._f
        fps = self.frame_rate_hz_value()
        frames = self.frames.value_int()
        self.ptile_fps.set(f"{f(fps)} {tr('Гц')}", f"{frames} {tr('на оборот')}")
        if not plan:
            self.plan_label.setText(tr("Сначала рассчитайте свет."))
            for tile in (self.ptile_time, self.ptile_revs, self.ptile_tol):
                tile.set("—")
            return
        vp = plan.get("virtual_print") or {}
        iou = vp.get("iou")
        self.tile_match.set(f"{f(100 * iou)} %" if iou is not None else "—",
                            ("почти идеально" if iou > 0.98 else "можно точнее") if iou is not None else "")
        tol = float(plan.get("exposure_tolerance_pct", 0.0))
        tol_note = "ошибиться сложно" if tol >= 4 else ("нужна калибровка" if tol > 0 else "увеличьте сетку")
        self.tile_tol.set(f"± {f(tol)} %" if tol > 0 else tr("нет"), tol_note, warn=0 < tol < 3)
        self.ptile_tol.set(f"± {f(tol)} %" if tol > 0 else tr("нет"), "", warn=0 < tol < 3)
        self.tile_frames.set(f"{frames}", f"{tr('шаг')} {f(360.0 / frames)}°")
        if "frame_exposure_total_s" in plan:
            revs = float(plan["frame_exposure_total_s"]) * fps
            total = revs * frames / fps
            self.tile_time.set(self._secs(total), f"{f(revs)} {tr('оборота')}")
            # On the print page: the time the projector will actually run.
            scale = self._exposure / max(self.brightness_factor(), 0.05)
            notes = []
            if abs(self._exposure - 1.0) > 1e-3:
                notes.append(f"{tr('экспозиция')} {round(self._exposure * 100)} %")
            if self.brightness_factor() < 0.999:
                notes.append(f"{tr('яркость')} {round(self.brightness_factor() * 100)} %")
            self.ptile_time.set(self._secs(total * scale), " · ".join(notes))
            self.ptile_revs.set(f(revs * scale))
        else:
            # Not a sentence in place of a number: a dash, and where to fix it.
            self.tile_time.set("—", "нужны поглощение, облучённость и доза: Настройки")
            self.ptile_time.set("—", "нужны данные смолы")
            self.ptile_revs.set("—")
        warnings = [tr(w) for w in (plan.get("warnings") or [])]
        self.warnings_label.setText("\n".join(f"• {w}" for w in warnings))
        n = len(warnings)
        self.warnings_toggle.setVisible(bool(n))
        shown = self.warnings_label.isVisible()
        self.warnings_toggle.setText(tr("скрыть замечания") if shown else f"{tr('замечаний')}: {n} · {tr('показать')}")
        self.plan_label.setText(tr("Готово."))

    # ------------------------------------------------------------- state from the app
    def show_page(self, key: str) -> None:
        if key in ("printer", "resin", "projector", "interface"):
            self.show_setup_tab(key)
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
            faces_text = f"{faces:,}".replace(",", " ")
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
            self._review = False
            if not was:
                self.game_title.setText(tr("Пока считается — засветите деталь сами"))
        elif was:
            self.game_title.setText(tr("Пока ждёте — засветите деталь сами"))
        self.r_virtual.setVisible(not running and self._virtual_ready)
        self.r_comp.setVisible(not running and self.projector_blur.value() > 0)
        self._show_light_stage()
        self._refresh_action()

    def _show_light_stage(self) -> None:
        """Before the run: the parameters and the game. During it: only the game. After: the result
        (and, after "пересчитать", the parameters and the game again)."""
        running = self._running
        result = self._done and not running and not self._review
        self.calc_block.setVisible(not running and not result)
        self.game_box.setVisible(not result)
        self.game.setMaximumWidth(16777215 if running else 380)
        self.result_block.setVisible(result)
        self.stale_label.setVisible(self._stale and not running)

    def set_results(self, plan: dict | None, output_dir: str) -> None:
        self._plan = plan
        self._output_dir = output_dir
        self._done = True
        self._review = False
        self._stale = False
        self._refresh_plan()
        # What this result was calculated with: quality, grid, frames per turn, step between frames.
        v = self.derived()
        quality = {128: tr("Быстро"), 256: tr("Норма"), 384: tr("Точно")}.get(self.grid_res.value_int(), "")
        self.calc_settings.setText(
            f"{tr('Качество')} {quality} · {tr('сетка')} {self.grid_res.value_int()} · "
            f"{tr('кадров')} {self.frames.value_int()} · {tr('шаг')} {self._f(v['step'])}°")
        self._show_light_stage()
        self._refresh_action()

    def set_virtual_available(self, available: bool) -> None:
        self._virtual_ready = available
        self.r_virtual.setVisible(available and not self._running)
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
        self._show_light_stage()

    def set_photon_frames(self, frames: list) -> None:
        self.game.photon.set_level_from_frames(frames)
        self.game.update()
        for refresh in getattr(self, "_arcade_refresh", []):
            refresh()

    def set_game_shape(self, grid: object) -> None:
        import numpy as _np
        shape = _np.asarray(grid, dtype=bool)
        self.game.set_shape(shape)

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
            self.actionRequested.emit("goto:print" if (self._done and not self._running and not self._review)
                                      else "generate")
        elif self._projecting is not None:
            self.actionRequested.emit("stop_projector")
        else:
            self.actionRequested.emit("goto:light")

    def refresh_theme(self) -> None:
        self._refresh()
        if self.fit_label.text():
            self.set_fit(self._fits, self._fit_text)
        self.game.update()


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
