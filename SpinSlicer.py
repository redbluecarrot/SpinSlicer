"""

Главное окно:

  шапка   — путь детали «1 Модель → 2 Свет → 3 Печать», справа «Принтер»
            (принтер и смола, настраиваются один раз) и день/ночь.
  слева   — 3D-сцена в стиле рисунка (slicer_tab.SlicerTab), а на шаге
            «Печать» — кадры проектора (video_tab.ProjectorTab);
            под сценой — строка значений (зона, воксель, кадры, время, запас).
  справа  — одна страница настроек (ui_panels.ProcessSettingsPanel).


"""
from __future__ import annotations

import os
import sys

if len(sys.argv) >= 4 and sys.argv[1] == "--engine-worker":
    # The frozen app re-launches itself as the compute process (see engine_process.py).
    from engine_process import run_job
    raise SystemExit(run_job(sys.argv[2], sys.argv[3]))
from datetime import datetime, timezone
from pathlib import Path

if sys.platform.startswith("linux"):
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        os.environ.pop("DISPLAY", None)
        os.environ["PYVISTA_OFF_SCREEN"] = "true"
    else:
        try:
            import ctypes
            _x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
            _x11_handler_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
            _x11_err_handler = _x11_handler_type(lambda _disp, _evt: 0)
            _x11.XSetErrorHandler(_x11_err_handler)
        except Exception:
            pass

from qt_runtime import configure_qt_dll_search_path

configure_qt_dll_search_path()

from PyQt6.QtCore import QSettings, Qt, QTimer, QUrl  # noqa: E402
from PyQt6.QtGui import QDesktopServices, QIcon  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from constants import APP_ORG, APP_TITLE  # noqa: E402
from i18n import LANGUAGES, apply_translations, language, num, set_language, tr  # noqa: E402
from job_controller import JobController  # noqa: E402
from pixel_game import silhouette_from_mesh  # noqa: E402
from slicer_tab import SlicerTab  # noqa: E402
from theme import apply_theme  # noqa: E402
from theme import current as current_palette  # noqa: E402
from ui_kit import Chip, NavButton, SunMoonToggle, set_hints_visible  # noqa: E402
from ui_panels import ProcessSettingsPanel  # noqa: E402
from video_tab import ProjectorTab  # noqa: E402


def resource_path(relative_path: str) -> str:
    """Resolve a bundled resource both from source and a PyInstaller build."""

    bundle_root = getattr(sys, "_MEIPASS", None)
    root = Path(bundle_root) if isinstance(bundle_root, str) else Path(__file__).resolve().parent
    return str(root / relative_path)


APP_ICON_PATH = resource_path("assets/spinslicer.svg")


class _LogoMark(QWidget):
    """The app logo (brain in a vial), drawn from the SVG so it stays sharp on any screen."""

    def __init__(self, path: str, side: int) -> None:
        super().__init__()
        from PyQt6.QtSvg import QSvgRenderer
        self._renderer = QSvgRenderer(path, self)
        self.setFixedSize(side, side)
        self.setToolTip("SpinSlicer")

    def paintEvent(self, _event) -> None:  # noqa: N802
        from PyQt6.QtCore import QRectF
        from PyQt6.QtGui import QPainter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._renderer.render(painter, QRectF(0, 0, self.width(), self.height()))
        painter.end()


class CALSlicerMainWindow(QMainWindow):
    """Header (steps · printer), scene on the left, one page of settings on the right."""

    STEPS = ("model", "light", "print")

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setWindowIcon(QIcon(APP_ICON_PATH))
        self.setAcceptDrops(True)
        self._settings = QSettings(APP_ORG, APP_TITLE)
        self._job_controller = JobController()
        self._scene_timer = QTimer(self)
        self._scene_timer.setSingleShot(True)
        self._scene_timer.setInterval(60)
        self._scene_timer.timeout.connect(self._apply_scene)
        self._frames_assembled_for = ""
        self._game_shape_for = ""

        self._build_central()
        self._build_status_bar()
        self._wire_signals()
        index = self.language_combo.findData(language())
        if index >= 0:
            self.language_combo.setCurrentIndex(index)
        apply_translations(self)
        self._select_page("model")
        self._apply_scene()
        self._apply_canvas()
        self._refresh_chips()
        self._refresh_nav()
        self._log(tr("Готово к работе. Загрузите STL-модель, чтобы начать."))

    @property
    def tabs(self):
        class _TabAdapter:
            def __init__(self, win):
                self._widgets = [win.slicer_tab, win.projector_tab, win.settings_tab]
            def count(self):
                return len(self._widgets)
            def widget(self, index):
                return self._widgets[index]
            def currentIndex(self):
                return 0
        return _TabAdapter(self)

    # =======================================================================
    # Построение интерфейса
    # =======================================================================
    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("appHeader")
        header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        header.setFixedHeight(64)
        row = QHBoxLayout(header)
        row.setContentsMargins(24, 0, 20, 0)
        row.setSpacing(4)

        ident = QVBoxLayout()
        ident.setSpacing(1)
        self.project_name = QLabel(tr("Новый проект"))
        self.project_name.setObjectName("projectName")
        self.project_name.setProperty("_spinslicer_skip_text_translation", True)
        self.project_sub = QLabel(tr("модель не загружена"))
        self.project_sub.setObjectName("projectSub")
        self.project_sub.setProperty("_spinslicer_skip_text_translation", True)
        ident.addStretch(1)
        ident.addWidget(self.project_name)
        ident.addWidget(self.project_sub)
        ident.addStretch(1)
        ident_box = QWidget()
        ident_box.setLayout(ident)
        ident_box.setFixedWidth(250)
        row.addWidget(_LogoMark(APP_ICON_PATH, 38))
        row.addSpacing(12)
        row.addWidget(ident_box)
        row.addStretch(1)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self.nav: dict[str, NavButton] = {
            "model": NavButton(tr("Модель"), badge="1"),
            "light": NavButton(tr("Свет"), badge="2"),
            "print": NavButton(tr("Печать"), badge="3"),
            "setup": NavButton(tr("Принтер"), sub="—"),
        }
        self._step_lines: list[QFrame] = []
        for key in self.STEPS:
            button = self.nav[key]
            self._nav_group.addButton(button)
            button.clicked.connect(lambda _c=False, k=key: self._select_page(k))
            row.addWidget(button)
            if key != self.STEPS[-1]:
                line = QFrame()
                line.setFixedSize(34, 1)
                self._step_lines.append(line)
                row.addWidget(line, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        setup = self.nav["setup"]
        self._nav_group.addButton(setup)
        setup.clicked.connect(lambda _c=False: self._select_page("setup"))
        setup.setToolTip(tr("Принтер и смола — настраиваются один раз"))
        setup_box = QWidget()
        sb = QHBoxLayout(setup_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(14)
        sb.addStretch(1)
        sb.addWidget(setup)
        self.theme_toggle = SunMoonToggle()
        self.theme_toggle.set_dark_silent(current_palette().dark)
        sb.addWidget(self.theme_toggle)
        setup_box.setFixedWidth(260)
        row.addWidget(setup_box)
        return header

    def _build_central(self) -> None:
        central = QWidget()
        central.setObjectName("centralRoot")
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.settings_tab = ProcessSettingsPanel()
        self.settings_tab.setMinimumWidth(360)
        self.settings_tab.setMaximumWidth(760)
        self.slicer_tab = SlicerTab(job_controller=self._job_controller, process_panel=self.settings_tab)
        self.projector_tab = ProjectorTab(job_controller=self._job_controller)
        self.right_panel = self.settings_tab
        self.language_combo = self.settings_tab.language_box
        self.language_label = QLabel("")
        for code, label in LANGUAGES.items():
            self.language_combo.addItem(label, code)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(0)
        self.view_tabs = QStackedWidget()
        self.view_tabs.setMinimumSize(320, 240)
        self.view_tabs.addWidget(self.slicer_tab)
        projector_wrap = QWidget()
        projector_wrap.setObjectName("printStage")
        projector_wrap.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        pw = QVBoxLayout(projector_wrap)
        pw.setContentsMargins(28, 22, 28, 14)
        pw.addWidget(self.projector_tab)
        self._projector_wrap = projector_wrap
        self.view_tabs.addWidget(projector_wrap)

        self._log_panel = QPlainTextEdit()
        self._log_panel.setObjectName("logPanel")
        self._log_panel.setReadOnly(True)
        self._log_panel.setMinimumHeight(60)
        # The journal is a drawer inside the left column: opening it takes space
        # from the scene and never makes the window taller than the screen.
        self._left_split = QSplitter(Qt.Orientation.Vertical)
        self._left_split.setChildrenCollapsible(True)
        self._left_split.addWidget(self.view_tabs)
        self._left_split.addWidget(self._log_panel)
        self._left_split.setCollapsible(0, False)
        self._left_split.setSizes([1, 0])
        self._log_panel.setVisible(False)
        left_l.addWidget(self._left_split, 1)

        chip_bar = QWidget()
        chip_bar.setObjectName("chipBar")
        chip_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        chips = QHBoxLayout(chip_bar)
        chips.setContentsMargins(20, 10, 20, 12)
        chips.setSpacing(8)
        self.chips = {key: Chip(tr(label)) for key, label in (
            ("zone", "зона"), ("voxel", "воксель"), ("frames", "кадров"), ("time", "печать"), ("tol", "запас"))}
        for chip in self.chips.values():
            chip.setProperty("_spinslicer_skip_text_translation", True)
            chips.addWidget(chip)
        chips.addStretch(1)
        self.log_toggle = QPushButton(tr("журнал ▴"))
        self.log_toggle.setObjectName("linkButton")
        self.log_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        chips.addWidget(self.log_toggle)
        left_l.addWidget(chip_bar)

        # The settings panel can be dragged wider (or narrower) by its left edge.
        self._body_split = QSplitter(Qt.Orientation.Horizontal)
        self._body_split.setChildrenCollapsible(False)
        self._body_split.setHandleWidth(5)
        self._body_split.addWidget(left)
        self._body_split.addWidget(self.settings_tab)
        self._body_split.setStretchFactor(0, 1)
        self._body_split.setStretchFactor(1, 0)
        width = int(self._settings.value("ui/panel_width", 410) or 410)
        self._body_split.setSizes([1200, max(360, min(width, 760))])
        self._body_split.splitterMoved.connect(
            lambda _pos, _i: self._settings.setValue("ui/panel_width", self.settings_tab.width()))
        body.addWidget(self._body_split, 1)
        root.addLayout(body, 1)
        self.setCentralWidget(central)
        self._style_lines()

    def _style_lines(self) -> None:
        for line in self._step_lines:
            line.setStyleSheet(f"background: {current_palette().border}; border: none;")

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        bar.setSizeGripEnabled(False)
        self.setStatusBar(bar)
        self._status_label = QLabel(tr("Готово к работе."))
        bar.addWidget(self._status_label, 1)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedWidth(220)
        bar.addPermanentWidget(self._progress_bar)

    def _wire_signals(self) -> None:
        for tab in (self.slicer_tab, self.projector_tab):
            tab.progress.connect(self._set_progress)
            tab.logMessage.connect(self._log)
        self.slicer_tab.outputGenerated.connect(self.projector_tab.set_output_dir)
        self.slicer_tab.outputGenerated.connect(self._on_output_generated)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)

        panel = self.settings_tab
        panel.actionRequested.connect(self._on_panel_action)
        panel.sceneChanged.connect(self._scene_timer.start)
        panel.canvasChanged.connect(self._apply_canvas)
        panel.settingsChanged.connect(self._refresh_chips)
        panel.settingsChanged.connect(self._refresh_nav)
        panel.settingsChanged.connect(self._apply_canvas)
        panel.scaleRequested.connect(self.slicer_tab.set_scale_pct)
        panel.orientationRequested.connect(self.slicer_tab.set_orientation)
        panel.virtualToggled.connect(self.slicer_tab.virtual_btn.setChecked)
        panel.settingsChanged.connect(self.slicer_tab.refresh_model_state)
        panel.hintsToggled.connect(set_hints_visible)
        set_hints_visible(panel.hints_switch.isChecked())

        st = self.slicer_tab
        st.modelInfo.connect(self._on_model_info)
        st.fitChanged.connect(panel.set_fit)
        st.fitChanged.connect(lambda ok, _t: self._refresh_nav())
        st.runningChanged.connect(self._on_running)
        st.generationProgress.connect(self._on_generation_progress)
        st.resultsReady.connect(self._on_results)
        st.virtualAvailable.connect(panel.set_virtual_available)
        # The Photon game flies through the light of the real print frames.
        self.projector_tab.framesReady.connect(lambda _n: panel.set_photon_frames(self.projector_tab._frames))
        # The dose card sits on the print page; its slider is the print exposure.
        panel.attach_dose_card(self.projector_tab.dose)
        self.projector_tab.projectionTick.connect(panel.set_projecting)
        self.projector_tab.projectionFinished.connect(self._on_projection_finished)

        self.theme_toggle.toggled.connect(self._on_theme)
        self.log_toggle.clicked.connect(self._toggle_log)

    # =======================================================================
    # Navigation and state
    # =======================================================================
    def _select_page(self, key: str) -> None:
        button = self.nav.get(key)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        self.settings_tab.show_page(key)
        on_print = key == "print"
        self.view_tabs.setCurrentWidget(self._projector_wrap if on_print else self.slicer_tab)
        if on_print:
            out_dir = self.settings_tab.output_dir()
            if out_dir and self._frames_assembled_for != out_dir:
                self._frames_assembled_for = out_dir
                self.projector_tab._on_assemble_clicked()

    def _on_panel_action(self, action: str) -> None:
        if action.startswith("goto:"):
            self._select_page(action.split(":", 1)[1])
        elif action == "load":
            self.slicer_tab._on_load_clicked()
        elif action == "generate":
            self.slicer_tab._on_generate_clicked()
        elif action == "calibrate":
            self.slicer_tab._on_calibrate_clicked()
        elif action == "center":
            self.slicer_tab._on_center()
        elif action == "autofit":
            self.slicer_tab._on_autofit()
        elif action == "export":
            self._apply_output_light()
            self.projector_tab._on_save_clicked()
        elif action == "ramp":
            self.projector_tab._on_ramp_clicked()
        elif action == "projector":
            self._apply_output_light()
            if self.projector_tab.show_on_projector(duration_s=self.settings_tab.print_duration_s()):
                self.settings_tab.set_projecting(0.0, self.settings_tab.print_duration_s())
        elif action == "stop_projector":
            self.projector_tab.stop_projector()
        elif action == "pattern":
            self.projector_tab.show_on_projector(pattern=True)
        elif action == "open_frames":
            out_dir = self.settings_tab.output_dir()
            if out_dir:
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(out_dir)))

    def _apply_output_light(self) -> None:
        panel = self.settings_tab
        self.projector_tab.set_output_light(panel.brightness_factor(), panel.projector_gamma.value())

    def _on_projection_finished(self, completed: bool) -> None:
        self.settings_tab.set_projecting(None)
        if completed:
            self._log(tr("Засветка завершена: доза набрана."))

    def _apply_scene(self) -> None:
        viewport = self.slicer_tab.viewport
        viewport.update_scene(self.settings_tab.scene_params())
        viewport.set_scene_frames(self.settings_tab.frames.value_int())

    def _apply_canvas(self) -> None:
        self.projector_tab.set_canvas(self.settings_tab.canvas_geometry())
        self.projector_tab.set_fps(self.settings_tab.frame_rate_hz_value())

    def _on_model_info(self, name: str, size: object, faces: int, scale_pct: float) -> None:
        self._model_info_args = (name, size, faces, scale_pct)
        self.settings_tab.set_model_info(name, size if isinstance(size, list) else None, faces, scale_pct)
        if not isinstance(size, list):
            self.project_name.setText(tr("Новый проект"))
            self.project_sub.setText(tr("модель не загружена"))
        else:
            sx, sy, sz = (float(v) for v in size)
            self.project_name.setText(os.path.splitext(name)[0])
            self.project_sub.setText(tr(f"{num(sx)} × {num(sy)} × {num(sz)} мм"))
            node = self.slicer_tab._model_node
            if node is not None and self._game_shape_for != node.source_path:
                self._game_shape_for = node.source_path
                self.settings_tab.set_game_shape(silhouette_from_mesh(node.original_mesh))
        self._refresh_nav()

    def _on_running(self, running: bool) -> None:
        self.settings_tab.set_running(running)
        self.slicer_tab.viewport.set_scene_activity("computing" if running else
                                                    ("done" if self.settings_tab.is_done() else "idle"))

    def _on_generation_progress(self, frac: float) -> None:
        self.settings_tab.set_running(True, frac)
        self.slicer_tab.viewport.set_scene_activity("computing", frac)

    def _on_results(self, plan: object, out_dir: str) -> None:
        self.settings_tab.set_results(plan if isinstance(plan, dict) else None, out_dir)
        self.slicer_tab.viewport.set_scene_activity("done")
        self._refresh_chips()
        self._refresh_nav()

    def _refresh_nav(self) -> None:
        panel = self.settings_tab
        w, h = panel.screen_size()
        gel = panel.gel_dose.value()
        self.nav["setup"].set_sub(f"Ø {panel.vat_diameter_mm():.0f} мм · {w}×{h}" + (" · ✓" if gel > 0 else ""))
        has_model = self.slicer_tab._model_node is not None
        fits = self.slicer_tab.model_fit()[0] if has_model else False
        self.nav["model"].set_badge("1", has_model and fits)
        self.nav["light"].set_badge("2", panel.is_done())
        self.nav["print"].set_badge("3", False)

    def _refresh_chips(self) -> None:
        panel = self.settings_tab
        v = panel.derived()
        f = num
        self.chips["zone"].set_value(f"Ø {f(v['zone'], 0)} × {f(v['height'], 0)} мм", warn=not panel.has_bath())
        self.chips["voxel"].set_value(f"{f(v['voxel'], 2)} мм")
        self.chips["frames"].set_value(f"{v['frames']:.0f} · шаг {f(v['step'])}°")
        plan = panel._plan if panel.is_done() else None
        if plan and "frame_exposure_total_s" in plan:
            total = float(plan["frame_exposure_total_s"]) * v["frames"]
            self.chips["time"].set_value(f"{int(total // 60)}:{int(total % 60):02d}")
        else:
            self.chips["time"].set_value("—")
        tol = float(plan.get("exposure_tolerance_pct", 0.0)) if plan else 0.0
        self.chips["tol"].set_value(f"± {f(tol)} %" if plan else "—", warn=bool(plan) and tol < 3)

    def _on_theme(self, dark: bool) -> None:
        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_theme(app, dark)
        self._settings.setValue("ui/dark", dark)
        self.slicer_tab.viewport.apply_palette()
        for chip in self.chips.values():
            chip.refresh()
        for button in self.nav.values():
            button.update()
        self._style_lines()
        self.settings_tab.refresh_theme()
        self.projector_tab.preview.update()
        self.projector_tab.dose.refresh_theme()

    def _toggle_log(self) -> None:
        shown = not self._log_panel.isVisible()
        self._log_panel.setVisible(shown)
        if shown:
            total = max(self._left_split.height(), 300)
            self._left_split.setSizes([total - 170, 170])
        self.log_toggle.setText(tr("журнал ▾") if shown else tr("журнал ▴"))

    def _on_language_changed(self, index: int) -> None:
        selected = self.language_combo.itemData(index)
        if not isinstance(selected, str) or selected == language():
            return
        set_language(selected)
        self._settings.setValue("language", selected)
        apply_translations(self)
        # Texts composed from live values are rebuilt by their owners.
        self.settings_tab.refresh_language()
        args = getattr(self, "_model_info_args", None)
        if args is not None:
            self._on_model_info(*args)
        else:
            self.project_name.setText(tr("Новый проект"))
            self.project_sub.setText(tr("модель не загружена"))
        self._refresh_chips()
        self._refresh_nav()
        for button in self.findChildren(NavButton):      # they paint their own text
            button.updateGeometry()
            button.update()
        self.projector_tab.update_language()
        self.projector_tab.preview.update()
        self.projector_tab.dose.refresh_language()
        self._log(tr("Язык изменён") + ".")

    def _on_output_generated(self, out_dir: str) -> None:
        self._frames_assembled_for = ""

    # --- drag & drop an STL anywhere on the window ------------------------------------
    @staticmethod
    def _dropped_stl(event) -> str:
        data = event.mimeData()
        if data is not None and data.hasUrls():
            for url in data.urls():
                if url.toLocalFile().lower().endswith(".stl"):
                    return url.toLocalFile()
        return ""

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if self._dropped_stl(event):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        path = self._dropped_stl(event)
        if path:
            event.acceptProposedAction()
            self._select_page("model")
            self.slicer_tab.load_path(path)

    # =======================================================================
    # Статус / лог (общие для всех вкладок)
    # =======================================================================
    def _set_progress(self, frac: float, message: str) -> None:
        self._progress_bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self._status_label.setText(tr(message))
        self._log(message)

    def _log(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
        text = tr(message)
        if text == message and "\n" in message:      # multi-line reports: line by line
            text = "\n".join(tr(line) for line in message.splitlines())
        self._log_panel.appendPlainText(f"[{stamp}] {text}")

    def closeEvent(self, event) -> None:  # noqa: N802 (имя метода задано Qt)
        self.settings_tab.shutdown()
        self.projector_tab.stop_projector()
        if self._job_controller.shutdown():
            self.slicer_tab.viewport.shutdown()
            event.accept()
        else:
            self._log("Невозможно безопасно закрыть приложение: фоновая задача ещё выполняется.")
            event.ignore()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    # Matches spinslicer.desktop, so the taskbar (Wayland) shows the app icon.
    app.setDesktopFileName("spinslicer")
    app.setWindowIcon(QIcon(APP_ICON_PATH))

    settings = QSettings(APP_ORG, APP_TITLE)
    saved_language = settings.value("language", "ru")
    if isinstance(saved_language, str) and saved_language in LANGUAGES:
        set_language(saved_language)

    dark = str(settings.value("ui/dark", True)).lower() in {"1", "true", "yes"}
    apply_theme(app, dark)

    window = CALSlicerMainWindow()
    if "--self-check" in sys.argv:
        from release_check import check_release

        output = Path(sys.argv[sys.argv.index("--self-check")+1])
        raise SystemExit(check_release(app, window, output))
    window.resize(1680, 980)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
