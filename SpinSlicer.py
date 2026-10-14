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
import sys
from datetime import datetime, timezone
from pathlib import Path

if os.environ.get("QT_QPA_PLATFORM") == "offscreen" and sys.platform.startswith("linux"):
    os.environ.pop("DISPLAY", None)

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
    def tabs(self) -> QTabWidget:
        return self.view_tabs

    # =======================================================================
    # Построение интерфейса
    # =======================================================================
    def _build_central(self) -> None:
        central = QWidget()
        central.setObjectName("centralRoot")
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 10)
        root.setSpacing(10)

        self.settings_tab = ProcessSettingsPanel()
        self.language_label = QLabel("Язык:")
        self.language_combo = QComboBox(self.settings_tab)
        for code, label in LANGUAGES.items():
            self.language_combo.addItem(label, code)
        language_row = QHBoxLayout()
        language_row.setContentsMargins(12, 8, 12, 0)
        language_row.addStretch(1)
        language_row.addWidget(self.language_label)
        language_row.addWidget(self.language_combo)
        settings_layout = self.settings_tab.layout()
        if not isinstance(settings_layout, QVBoxLayout):
            raise RuntimeError("Settings panel must provide a root layout.")
        settings_layout.insertLayout(0, language_row)

        self.view_tabs = QTabWidget()
        self.view_tabs.setDocumentMode(True)
        tab_bar = self.view_tabs.tabBar()
        if tab_bar is None:
            raise RuntimeError("Main tab widget did not create its tab bar.")
        tab_bar.setDrawBase(False)
        self.view_tabs.setObjectName("mainTabs")
        self.slicer_tab = SlicerTab(
            job_controller=self._job_controller,
            process_panel=self.settings_tab,
        )
        self.projector_tab = ProjectorTab(job_controller=self._job_controller)

        self.view_tabs.addTab(self.slicer_tab, "Слайсер (3D)")
        self.view_tabs.addTab(self.projector_tab, "Проектор (Кадры)")
        self.view_tabs.addTab(self.settings_tab, "Настройки")

        self._log_panel = QPlainTextEdit()
        self._log_panel.setObjectName("logPanel")
        self._log_panel.setReadOnly(True)
        self._log_panel.setMinimumHeight(0)
        self._log_panel.setMaximumHeight(16777215)

        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.setChildrenCollapsible(True)
        self.main_splitter.setHandleWidth(8)
        self.main_splitter.addWidget(self.view_tabs)
        self.main_splitter.addWidget(self._log_panel)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, True)
        self.main_splitter.setSizes([760, 120])
        root.addWidget(self.main_splitter, 1)

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
        """Show the progress bar only on the 3D Slicer tab."""
        is_slicer = (index == self.view_tabs.indexOf(self.slicer_tab))
        self._progress_bar.setVisible(is_slicer)

    def _on_language_changed(self, index: int) -> None:
        selected = self.language_combo.itemData(index)
        if not isinstance(selected, str) or selected == language():
            return
        set_language(selected)
        self._settings.setValue("language", selected)
        apply_translations(self)
        self.slicer_tab.update_language()
        if hasattr(self.projector_tab, "update_language"):
            self.projector_tab.update_language()
        self._update_projector_tab_tooltip()
        self._log(tr("Язык изменён") + ".")

    def _on_output_generated(self, out_dir: str) -> None:
        projector_index = self.view_tabs.indexOf(self.projector_tab)
        self.view_tabs.setTabToolTip(projector_index, f"{tr('Кадры готовы: ')}{out_dir}")
        self.view_tabs.setCurrentIndex(projector_index)
        # Автоматически загружаем и запускаем видео
        self.projector_tab._on_assemble_clicked()

    def _update_projector_tab_tooltip(self) -> None:
        projector_index = self.view_tabs.indexOf(self.projector_tab)
        out_dir = self.slicer_tab._last_output_dir
        tooltip = f"{tr('Кадры готовы: ')}{out_dir}" if out_dir else ""
        self.view_tabs.setTabToolTip(projector_index, tooltip)

    # =======================================================================
    # Статус / лог (общие для всех вкладок)
    # =======================================================================
    def _set_progress(self, frac: float, message: str) -> None:
        self._progress_bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self._status_label.setText(tr(message))
        self._log(message)

    def _log(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
        self._log_panel.appendPlainText(f"[{stamp}] {tr(message)}")

    def closeEvent(self, event) -> None:  # noqa: N802 (имя метода задано Qt)
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
