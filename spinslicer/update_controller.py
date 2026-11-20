"""Settings → Updates and the startup check, wired to the main window.

The network work runs in daemon threads: closing the window never waits for GitHub.
Their results come back to the GUI thread through this object's signals.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QSettings, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QMessageBox, QPushButton, QWidget

from spinslicer import __version__, updater
from spinslicer.i18n import tr

AUTO_CHECK_EVERY_S = 20 * 3600       # "once a day", a little early so a daily start always checks
AUTO_CHECK_DELAY_MS = 4000           # after the window is up


class UpdateController(QObject):
    _found = pyqtSignal(object, bool)            # Release, silent
    _latest = pyqtSignal(str, bool)              # latest version, silent
    _check_failed = pyqtSignal(str, str, bool)   # message, detail, silent
    _progress = pyqtSignal(float, str)
    _downloaded = pyqtSignal(str)
    _download_failed = pyqtSignal(str, str)

    def __init__(self, window: QWidget, panel, settings: QSettings, busy: Callable[[], bool],
                 log: Callable[[str], None], progress: Callable[[float, str], None]) -> None:
        super().__init__(window)
        self._window = window
        self._panel = panel
        self._settings = settings
        self._busy = busy
        self._log = log
        self._set_progress = progress
        self._release: Optional[updater.Release] = None
        self._file: Optional[Path] = None            # the downloaded update, ready to install
        self._checking = False
        self._cancel: Optional[threading.Event] = None
        self._found.connect(self._on_found)
        self._latest.connect(self._on_latest)
        self._check_failed.connect(self._on_check_failed)
        self._progress.connect(self._on_progress)
        self._downloaded.connect(self._on_downloaded)
        self._download_failed.connect(self._on_download_failed)

        # Shown in the status bar when the startup check found something.
        self.status_button = QPushButton()
        self.status_button.setObjectName("linkButton")
        self.status_button.setProperty("_spinslicer_skip_text_translation", True)
        self.status_button.setStyleSheet("padding: 0 12px;")
        self.status_button.clicked.connect(self.offer)
        self.status_button.setVisible(False)
        updater.cleanup_previous()

    def refresh_language(self) -> None:
        if self._release is not None:
            self.status_button.setText(tr("Доступна версия") + f" {self._release.version} — " + tr("установить"))

    # ------------------------------------------------------------------ entry points
    def handle(self, action: str) -> None:
        if action == "update_check":
            self.check(silent=False)
        elif action == "update_install":
            self.offer()
        elif action == "update_cancel" and self._cancel is not None:
            self._cancel.set()

    def start_auto(self) -> None:
        """The startup check: if switched on and the last one was a day ago."""
        if not self._panel.update_switch.isChecked():
            return
        try:
            last = float(self._settings.value("update/last_check", 0) or 0)
        except (TypeError, ValueError):
            last = 0.0
        if time.time() - last >= AUTO_CHECK_EVERY_S:
            QTimer.singleShot(AUTO_CHECK_DELAY_MS, lambda: self.check(silent=True))
        else:
            # Between daily checks, still remind about a version found earlier.
            known = str(self._settings.value("update/available", "") or "")
            if updater.is_newer(known):
                QTimer.singleShot(AUTO_CHECK_DELAY_MS, lambda: self.check(silent=True))

    def check(self, silent: bool) -> None:
        if self._checking or self._cancel is not None:
            return
        self._checking = True
        if not silent:
            self._panel.set_update_state("checking")

        def work() -> None:
            try:
                release = updater.fetch_latest()
            except updater.UpdateError as exc:
                self._check_failed.emit(exc.message, exc.detail, silent)
                return
            except Exception as exc:              # never let the thread die silently
                self._check_failed.emit("Не удалось проверить обновления.", str(exc), silent)
                return
            if updater.is_newer(release.version):
                self._found.emit(release, silent)
            else:
                self._latest.emit(release.version, silent)

        threading.Thread(target=work, name="update-check", daemon=True).start()

    # ------------------------------------------------------------------ check results
    def _on_found(self, release: updater.Release, silent: bool) -> None:
        self._checking = False
        self._settings.setValue("update/last_check", time.time())
        self._settings.setValue("update/available", release.version)
        if self._release is None or self._release.version != release.version:
            self._file = None
        self._release = release
        self._panel.set_update_state("found", release.version)
        self.status_button.setVisible(True)
        self.refresh_language()
        self._log(tr("Доступна версия") + f" {release.version} ({tr('сейчас')} {__version__}).")
        if not silent:
            self.offer()

    def _on_latest(self, version: str, silent: bool) -> None:
        self._checking = False
        self._settings.setValue("update/last_check", time.time())
        self._settings.setValue("update/available", "")
        self._panel.set_update_state("latest")
        self.status_button.setVisible(False)
        if not silent:
            self._log(tr("Установлена последняя версия") + f" {__version__}.")

    def _on_check_failed(self, message: str, detail: str, silent: bool) -> None:
        self._checking = False
        self._log(tr(message) + (f" {detail}" if detail else ""))
        # A failed background check (offline, say) is not worth a red line in Settings.
        self._panel.set_update_state("" if silent else "failed", tr(message))

    # ------------------------------------------------------------------ install
    def offer(self) -> None:
        """Ask, then download and install the found version (or open its page)."""
        release = self._release
        if release is None:
            self.check(silent=False)
            return
        kind = updater.install_kind()
        asset = updater.pick_asset(release, kind)
        box = QMessageBox(self._window)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle(tr("Обновление SpinSlicer"))
        box.setText(tr("Доступна версия") + f" {release.version}. " + tr("Сейчас установлена") + f" {__version__}.")
        if asset is None:
            box.setInformativeText(tr("Эту копию нужно обновить вручную: откроется страница релиза."))
            go = box.addButton(tr("Открыть страницу"), QMessageBox.ButtonRole.AcceptRole)
        else:
            size = f"{asset.size / (1024 * 1024):.0f} MB" if asset.size else ""
            box.setInformativeText(tr("SpinSlicer скачает обновление и перезапустится.")
                                   + (f" ({size})" if size else ""))
            go = box.addButton(tr("Установить"), QMessageBox.ButtonRole.AcceptRole)
        notes = box.addButton(tr("Что нового"), QMessageBox.ButtonRole.HelpRole)
        box.addButton(tr("Позже"), QMessageBox.ButtonRole.RejectRole)
        if release.notes:
            box.setDetailedText(release.notes)
        box.exec()
        clicked = box.clickedButton()
        if clicked is notes:
            QDesktopServices.openUrl(QUrl(release.page))
        elif clicked is go:
            if asset is None:
                QDesktopServices.openUrl(QUrl(release.page))
            elif self._file is not None and self._file.is_file():
                self._install(kind)
            else:
                self._download(asset)

    def _download(self, asset: updater.Asset) -> None:
        if self._cancel is not None:
            return
        cancel = threading.Event()
        self._cancel = cancel
        self._panel.set_update_state("downloading", "")
        self._log(tr("Скачиваю обновление") + f" {asset.name}…")

        def report(done: int, total: int) -> None:
            mb = 1024 * 1024
            text = f"{done / mb:.0f} / {total / mb:.0f} MB" if total else f"{done / mb:.0f} MB"
            self._progress.emit(done / total if total else 0.0, text)

        def work() -> None:
            try:
                path = updater.download(asset, updater.download_folder(), report, cancel.is_set)
            except updater.UpdateError as exc:
                self._download_failed.emit(exc.message, exc.detail)
                return
            except Exception as exc:
                self._download_failed.emit("Не удалось скачать обновление.", str(exc))
                return
            self._downloaded.emit(str(path))

        threading.Thread(target=work, name="update-download", daemon=True).start()

    def _on_progress(self, fraction: float, text: str) -> None:
        if self._cancel is None:
            return
        self._panel.set_update_state("downloading", text)
        self._set_progress(fraction, tr("Скачиваю обновление") + f" · {text}")

    def _on_download_failed(self, message: str, detail: str) -> None:
        self._cancel = None
        self._set_progress(0.0, tr(message))
        self._log(tr(message) + (f" {detail}" if detail else ""))
        if self._release is not None:
            self._panel.set_update_state("found", self._release.version)
        if message != "Загрузка отменена.":
            QMessageBox.warning(self._window, tr("Обновление SpinSlicer"), tr(message))

    def _on_downloaded(self, path: str) -> None:
        self._cancel = None
        self._file = Path(path)
        self._set_progress(1.0, tr("Обновление скачано."))
        if self._release is not None:
            self._panel.set_update_state("found", self._release.version)
        self._install(updater.install_kind())

    def _install(self, kind: str) -> None:
        if self._file is None:
            return
        if self._busy():
            QMessageBox.information(self._window, tr("Обновление SpinSlicer"),
                                    tr("Обновление скачано. Установи его, когда закончится расчёт или печать: "
                                       "Настройки → Интерфейс → Обновления."))
            return
        try:
            updater.install(self._file, kind)
        except updater.UpdateError as exc:
            self._log(tr(exc.message) + (f" {exc.detail}" if exc.detail else ""))
            QMessageBox.warning(self._window, tr("Обновление SpinSlicer"), tr(exc.message))
            return
        except OSError as exc:
            self._log(tr("Не удалось установить обновление.") + f" {exc}")
            QMessageBox.warning(self._window, tr("Обновление SpinSlicer"), tr("Не удалось установить обновление."))
            return
        self._log(tr("Запускаю новую версию") + "…")
        self._window.close()
