"""

Главное окно — тонкая оболочка над вкладками приложения:

  1. "Слайсер"         — генерация проекций (slicer_tab.SlicerTab).
  2. "Проектор (Видео)" — сборка/проигрывание/экспорт видео из кадров
                              (video_tab.ProjectorTab).
  3. "Настройки"          — параметры принтера, процесса и VAMToolbox.

Статус-бар и лог общие для всего приложения. Прогресс-бар показывается
только на главной вкладке "Слайсер".


"""
from __future__ import annotations

import os
import re
import sys
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

from PyQt6.QtCore import QSettings, Qt  # noqa: E402
from PyQt6.QtGui import QIcon  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from constants import APP_ORG, APP_TITLE  # noqa: E402
from i18n import LANGUAGES, apply_translations, language, set_language, tr  # noqa: E402
from job_controller import JobController  # noqa: E402
from slicer_tab import SlicerTab  # noqa: E402
from theme import apply_theme  # noqa: E402
from ui_panels import ProcessSettingsPanel  # noqa: E402
from video_tab import ProjectorTab  # noqa: E402


def resource_path(relative_path: str) -> str:
    """Resolve a bundled resource both from source and a PyInstaller build."""

    bundle_root = getattr(sys, "_MEIPASS", None)
    root = Path(bundle_root) if isinstance(bundle_root, str) else Path(__file__).resolve().parent
    return str(root / relative_path)


APP_ICON_PATH = resource_path("assets/spinslicer.svg")


class CALSlicerMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setWindowIcon(QIcon(APP_ICON_PATH))
        self._settings = QSettings(APP_ORG, APP_TITLE)
        self._last_progress_key = ""
        self._last_logged_fraction = 0.0
        self._job_controller = JobController()

        self._build_central()
        self._build_status_bar()
        self._wire_signals()
        index = self.language_combo.findData(language())
        if index >= 0:
            self.language_combo.setCurrentIndex(index)
        apply_translations(self)

        self._log(tr("Готово к работе. Загрузите STL-модель на вкладке «Слайсер», чтобы начать."))

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
    def _build_central(self) -> None:
        central = QWidget()
        central.setObjectName("centralRoot")
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 10)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel(APP_TITLE)
        title.setObjectName("appTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.language_label = QLabel("Язык:")
        header.addWidget(self.language_label)
        self.language_combo = QComboBox()
        for code, label in LANGUAGES.items():
            self.language_combo.addItem(label, code)
        header.addWidget(self.language_combo)
        root.addLayout(header)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(True)

        # Left side: Viewports (Slicer and Projector)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.view_tabs = QTabWidget()
        self.view_tabs.setDocumentMode(True)
        self.view_tabs.setObjectName("mainTabs")
        tab_bar = self.view_tabs.tabBar()
        if tab_bar is not None:
            tab_bar.setDrawBase(False)

        self.settings_tab = ProcessSettingsPanel()
        self.slicer_tab = SlicerTab(
            job_controller=self._job_controller,
            process_panel=self.settings_tab,
        )
        self.projector_tab = ProjectorTab(job_controller=self._job_controller)

        self.view_tabs.addTab(self.slicer_tab, "Слайсер (3D)")
        self.view_tabs.addTab(self.projector_tab, "Проектор (Кадры)")
        left_layout.addWidget(self.view_tabs)

        main_splitter.addWidget(left_panel)

        # Right side: Settings Sidebar
        self.right_panel = QWidget()
        self.right_panel.setMinimumWidth(200)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.settings_tab)

        main_splitter.addWidget(self.right_panel)

        # Set initial sizes (70% left, 30% right)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)
        main_splitter.setSizes([1040, 500])

        self._log_panel = QPlainTextEdit()
        self._log_panel.setObjectName("logPanel")
        self._log_panel.setReadOnly(True)
        self._log_panel.setMinimumHeight(40)

        # The log can be dragged taller or shorter, like the settings sidebar.
        self._log_splitter = QSplitter(Qt.Orientation.Vertical)
        self._log_splitter.setChildrenCollapsible(False)
        self._log_splitter.addWidget(main_splitter)
        self._log_splitter.addWidget(self._log_panel)
        self._log_splitter.setStretchFactor(0, 1)
        self._log_splitter.setStretchFactor(1, 0)
        saved_height = self._settings.value("log_height", 96, type=int)
        self._log_splitter.setSizes([700, max(40, min(int(saved_height), 600))])
        root.addWidget(self._log_splitter, 1)

        self.setCentralWidget(central)

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)

        self._status_label = QLabel("Готово к работе.")
        bar.addWidget(self._status_label, 1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedWidth(260)
        bar.addPermanentWidget(self._progress_bar)

    def _wire_signals(self) -> None:
        # Прогресс и лог рабочих вкладок стекаются в общий статус-бар/лог.
        for tab in (self.slicer_tab, self.projector_tab):
            tab.progress.connect(self._set_progress)
            tab.logMessage.connect(self._log)

        # Передать готовые кадры в просмотрщик без ручного выбора папки.
        self.slicer_tab.outputGenerated.connect(self.projector_tab.set_output_dir)
        self.slicer_tab.outputGenerated.connect(self._on_output_generated)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        self.view_tabs.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed(self.view_tabs.currentIndex())

    def _on_tab_changed(self, index: int) -> None:
        """Show progress bar and settings sidebar only on the 3D Slicer tab."""
        is_slicer = (index == self.view_tabs.indexOf(self.slicer_tab))
        self._progress_bar.setVisible(is_slicer)
        self.right_panel.setVisible(is_slicer)

    def _on_language_changed(self, index: int) -> None:
        selected = self.language_combo.itemData(index)
        if not isinstance(selected, str) or selected == language():
            return
        set_language(selected)
        self._settings.setValue("language", selected)
        apply_translations(self)
        if hasattr(self.projector_tab, "update_language"):
            self.projector_tab.update_language()
        self._log(tr("Язык изменён") + ". " + tr("Перезапустите приложение, чтобы применить язык."))

    def _on_output_generated(self, out_dir: str) -> None:
        projector_index = self.view_tabs.indexOf(self.projector_tab)
        self.view_tabs.setTabToolTip(projector_index, f"Кадры готовы: {out_dir}")
        self.view_tabs.setCurrentIndex(projector_index)
        # Автоматически загружаем и запускаем видео
        self.projector_tab._on_assemble_clicked()

    # =======================================================================
    # Статус / лог (общие для всех вкладок)
    # =======================================================================
    def _set_progress(self, frac: float, message: str) -> None:
        self._progress_bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self._status_label.setText(tr(message))
        # The status bar already shows every update. The log gets one line when a
        # phase starts and then at most one per 10 % of progress, instead of a
        # line for every iteration of every layer block.
        key = re.sub(r"\d+", "#", message)
        advanced = frac - self._last_logged_fraction >= 0.10
        if key != self._last_progress_key or advanced or frac >= 1.0 or frac < self._last_logged_fraction:
            self._last_progress_key = key
            self._last_logged_fraction = frac
            self._log(message)

    def _log(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
        self._log_panel.appendPlainText(f"[{stamp}] {tr(message)}")

    def closeEvent(self, event) -> None:  # noqa: N802 (имя метода задано Qt)
        self._settings.setValue("log_height", self._log_splitter.sizes()[1])
        self.settings_tab.shutdown()
        if self._job_controller.shutdown():
            event.accept()
        else:
            self._log("Невозможно безопасно закрыть приложение: фоновая задача ещё выполняется.")
            event.ignore()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setWindowIcon(QIcon(APP_ICON_PATH))

    settings = QSettings(APP_ORG, APP_TITLE)
    saved_language = settings.value("language", "ru")
    if isinstance(saved_language, str) and saved_language in LANGUAGES:
        set_language(saved_language)

    apply_theme(app)

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
