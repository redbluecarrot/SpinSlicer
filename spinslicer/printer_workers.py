"""Background threads for the printer: pack a ``.spin``, upload it, find printers, read a profile."""
from __future__ import annotations

import os
import threading
import traceback
from typing import Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from spinslicer import printer_link
from spinslicer.spin_format import PrintJob, write_spin


class SpinPackWorker(QThread):
    """Packs an encoded video into a ``.spin`` file and optionally uploads it.

    ``video_path`` is removed afterwards when ``remove_video`` (a temporary encode).
    """

    progress = pyqtSignal(float, str)
    packed = pyqtSignal(str)               # path of the .spin file
    uploaded = pyqtSignal(str, object)     # printer label, the printer's answer
    failed = pyqtSignal(str)

    def __init__(self, job: PrintJob, video_path: str, spin_path: str, *,
                 preview_png: Optional[bytes] = None, slicer_settings: Optional[dict[str, Any]] = None,
                 printer: Optional[printer_link.PrinterAddress] = None, remove_video: bool = True,
                 remove_spin: bool = False, parent: Any = None) -> None:
        super().__init__(parent)
        self.job = job
        self.video_path = video_path
        self.spin_path = spin_path
        self.preview_png = preview_png
        self.slicer_settings = slicer_settings
        self.printer = printer
        self.remove_video = remove_video
        self.remove_spin = remove_spin
        self._cancel = threading.Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            self.progress.emit(0.0, "Упаковка файла .spin...")
            write_spin(self.spin_path, self.job, self.video_path, preview_png=self.preview_png,
                       slicer_settings=self.slicer_settings)
            self.packed.emit(self.spin_path)
            if self.printer is not None:
                last = [-1]

                def report(sent: int, total: int) -> None:
                    percent = sent * 100 // max(total, 1)
                    if percent // 5 != last[0]:          # every 5 %: the log stays readable
                        last[0] = percent // 5
                        self.progress.emit(sent / max(total, 1), f"Отправка на принтер {percent}%")
                answer = printer_link.upload(self.printer, self.spin_path, os.path.basename(self.spin_path),
                                             progress=report, cancelled=self._cancel.is_set)
                self.uploaded.emit(self.printer.label, answer)
        except printer_link.PrinterError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=4)}")
        finally:
            for path, remove in ((self.video_path, self.remove_video), (self.spin_path, self.remove_spin)):
                if remove:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            if self.remove_spin:
                try:
                    os.rmdir(os.path.dirname(self.spin_path))   # the temporary folder of a send
                except OSError:
                    pass


class DiscoverWorker(QThread):
    found = pyqtSignal(object)             # list[PrinterAddress]

    def __init__(self, timeout_s: float = 2.5, parent: Any = None) -> None:
        super().__init__(parent)
        self.timeout_s = timeout_s

    def run(self) -> None:
        try:
            printers = printer_link.discover(self.timeout_s)
        except Exception:  # noqa: BLE001 - discovery is best effort
            printers = []
        self.found.emit(printers)


class ProfileWorker(QThread):
    """``GET /api/printer``: emits the profile (dict) or None when the printer has none."""

    loaded = pyqtSignal(object, object)    # address, profile or None
    failed = pyqtSignal(object, str)

    def __init__(self, address: printer_link.PrinterAddress, parent: Any = None) -> None:
        super().__init__(parent)
        self.address = address

    def run(self) -> None:
        try:
            self.loaded.emit(self.address, printer_link.get_printer_profile(self.address, timeout_s=5.0))
        except printer_link.PrinterError as exc:
            self.failed.emit(self.address, str(exc))


class CallWorker(QThread):
    """One call to the printer off the UI thread: ``done`` with its result or ``failed`` with the reason."""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Any, *args: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self.fn = fn
        self.args = args

    def run(self) -> None:
        try:
            self.done.emit(self.fn(*self.args))
        except printer_link.PrinterError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.failed.emit(str(exc) or exc.__class__.__name__)
