"""Glue between the main window and a SpinPrinter.

Settings → Printer: connect (the printer's profile fills the slicer), write the slicer's vial,
projector and calibration back to the printer. Print step: "Print" (upload the .spin, ask the
printer to print it; its screen asks "Print?", then the slicer follows the print and can stop
it) and "To USB stick". The panel (ui_panels.ProcessSettingsPanel) shows it all, the projector
tab (video_tab.ProjectorTab) encodes and packs, and every network call runs in a thread.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QObject, QSettings, QTimer
from PyQt6.QtWidgets import QInputDialog, QMessageBox, QWidget

from spinslicer import printer_link, removable
from spinslicer.constants import APP_ORG, APP_TITLE
from spinslicer.i18n import tr
from spinslicer.printer_workers import CallWorker, DiscoverWorker, ProfileWorker
from spinslicer.spin_format import PrintJob, preview_from_frame, render_preview, spin_name

SETTINGS_ADDRESS = "printer/address"


def load_json(path: str) -> Optional[dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def build_job(panel: Any, out_dir: str, name: str, frames: int, width: int, height: int,
              fps: float) -> PrintJob:
    """The print job for the frames in `out_dir` with the panel's current print settings."""
    plan = load_json(os.path.join(out_dir, "print_plan.json")) or {}
    manifest = load_json(os.path.join(out_dir, "manifest.json")) or {}
    slice_parameters = manifest.get("slice_parameters") or {}
    direction = int(slice_parameters.get("angle_direction", 1) or 1)
    optics = slice_parameters.get("optics") or {}
    resin = {
        "alpha_per_mm": float(optics.get("alpha_per_mm", panel.alpha_absorption.value())),
        "n_resin": float(optics.get("n_resin", panel.n_resin.value())),
        "gel_dose_mj_cm3": float(panel.gel_dose.value()),
    }
    printer: dict[str, Any] = {
        "vial_diameter_mm": float(slice_parameters.get("diameter_mm", panel.vat_diameter_mm())),
        "vial_wall_mm": float(panel.glass_thickness_mm()),
        "immersion": bool(panel.has_bath()),
        "projector": {
            "width": int(width), "height": int(height),
            "irradiance_mw_cm2": float(panel.projector_irradiance.value()),
            "refresh_hz": float(panel.projector_hz.value()),
            "gamma": float(panel.projector_gamma.value()),
            "contrast": float(panel.projector_contrast.value()),
            "vial_width_px": int(panel.output_res.value_int()),
            "offset_px": [int(panel.offset_x.value()), int(panel.offset_y.value())],
        },
    }
    if panel.printer_profile_id():
        printer["profile_id"] = panel.printer_profile_id()
    return PrintJob(name=name, frames=frames, fps=fps, width=width, height=height,
                    angle_direction=direction if direction in (1, -1) else 1, plan=plan,
                    exposure=float(panel.exposure_factor()), brightness=float(panel.brightness_factor()),
                    resin=resin, printer=printer)


def state_text(state: dict[str, Any]) -> tuple[str, str, bool, bool]:
    """What to show for the printer's /api/state: (kind, text, warn, still going)."""
    what, progress, turn, turns, left = printer_link.describe_state(state)
    msg = str(state.get("msg") or (state.get("request") or {}).get("error") or "").strip()
    final = {"cancelled": (tr("Печать отменена на экране принтера."), False),
             "expired": (tr("Принтер не дождался ответа на экране (10 минут)."), True),
             "replaced": (tr("На принтере запустили другую печать."), True),
             "done": (tr("Готово! Ответьте на принтере, как получилось."), False),
             "stopped": (f"{tr('Печать остановлена.')} {msg}".strip(), True),
             "error": (f"{tr('Ошибка печати:')} {msg}".strip(), True)}
    if what in final:
        text, warn = final[what]
        return what, text, warn, False
    if what == "waiting_confirm":
        return what, tr("Подтвердите печать на экране принтера: джойстик вправо."), False, True
    if what == "starting":
        return what, tr("Принтер готовится: разгон, прогрев, пауза до света…"), False, True
    mins, secs = divmod(int(left + 0.99), 60)
    return what, (f"{tr('Печать')}: {int(progress * 100)}% · {tr('оборот')} {turn} {tr('из')} {turns} · "
                  f"{tr('осталось')} {mins}:{secs:02d}"), False, True


class PrinterController(QObject):
    def __init__(self, window: QWidget, panel: Any, projector_tab: Any,
                 out_dir: Callable[[], str], model: Callable[[], Any], log: Callable[[str], None]) -> None:
        super().__init__(window)
        self._window = window
        self._panel = panel
        self._tab = projector_tab
        self._out_dir = out_dir
        self._model = model
        self._log = log
        self._settings = QSettings(APP_ORG, APP_TITLE)
        self._discover: Optional[DiscoverWorker] = None
        self._profile: Optional[ProfileWorker] = None
        self._connected: Optional[printer_link.PrinterAddress] = None
        self._printer_settings: dict[str, Any] = {}   # printer_link.profile_settings of the connected one
        self._calls: list[CallWorker] = []
        self._printing: Optional[printer_link.PrinterAddress] = None   # the print the slicer follows
        self._polling = False
        self._poll = QTimer(self)
        self._poll.setInterval(1000)
        self._poll.timeout.connect(self._poll_state)
        projector_tab.job_provider = self._job
        projector_tab.spin_extras = self._extras
        projector_tab.spinFailed.connect(self._on_spin_failed)
        panel.printer_box.currentIndexChanged.connect(self._on_printer_chosen)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)
        stored = str(self._settings.value(SETTINGS_ADDRESS, "") or "")
        if stored:
            try:
                panel.set_printers([], keep=printer_link.parse_address(stored))
                QTimer.singleShot(0, self._connect_stored)   # the printer from last time
            except printer_link.PrinterError:
                pass

    def shutdown(self) -> None:
        """Before the app goes: let the printer requests in flight finish (a QThread destroyed
        while running takes the whole process down). Each gives up after its own timeout."""
        self._poll.stop()
        for worker in [self._profile, self._discover, *self._calls]:
            if worker is not None and not sip.isdeleted(worker) and worker.isRunning():
                worker.wait(12000)

    # ------------------------------------------------------------------ job
    def _name(self) -> str:
        node = self._model()
        source = getattr(node, "source_path", "") if node is not None else ""
        return os.path.splitext(os.path.basename(source))[0] if source else "print"

    def _job(self, frames: int, width: int, height: int, fps: float) -> Optional[PrintJob]:
        out_dir = self._out_dir()
        if not out_dir:
            return None
        return build_job(self._panel, out_dir, self._name(), frames, width, height, fps)

    def _extras(self) -> tuple[Optional[bytes], Optional[dict[str, Any]]]:
        out_dir = self._out_dir()
        settings = load_json(os.path.join(out_dir, "manifest.json")) if out_dir else None
        preview: Optional[bytes] = None
        try:
            node = self._model()
            if node is not None:
                mesh = node.get_transformed_mesh()
                preview = render_preview(mesh.vertices, mesh.faces)
        except Exception:  # noqa: BLE001 - the preview is optional
            preview = None
        if preview is None:
            frame = self._tab.composed_frame(0)
            if frame is not None:
                preview = preview_from_frame(frame)
        return preview, settings

    # ------------------------------------------------------------------ actions
    def handle(self, action: str) -> bool:
        actions: dict[str, Callable[[], Any]] = {
            "spin_export": lambda: self._tab.export_spin(), "spin_print": self.print_now,
            "spin_stick": self.to_stick, "spin_stop": self.stop, "printer_find": self.find,
            "printer_manual": self.ask_address, "printer_connect": lambda: self.connect_printer(),
            "printer_write": self.write_to_printer,
        }
        if action not in actions:
            return False
        actions[action]()
        return True

    def _call(self, fn: Callable[..., Any], *args: Any, done: Callable[[Any], Any],
              failed: Callable[[str], Any]) -> None:
        worker = CallWorker(fn, *args, parent=self)

        def alive(callback: Callable[[Any], Any]) -> Callable[[Any], None]:
            # the window may be gone by the time the printer answers (closed mid-request)
            return lambda value: None if sip.isdeleted(self) else callback(value)
        worker.done.connect(alive(done))
        worker.failed.connect(alive(failed))
        worker.finished.connect(lambda: self._calls.remove(worker) if worker in self._calls else None)
        self._calls.append(worker)
        worker.start()

    def _not_connected(self) -> None:
        QMessageBox.information(self._window, tr("Принтер не подключён"),
                                tr("Подключите принтер: Настройки → Принтер → «Подключить»."))

    # ------------------------------------------------------------------ print
    def print_now(self) -> None:
        """Upload the .spin and ask the printer to print it (its screen asks first)."""
        printer = self._connected
        if printer is None:
            self._not_connected()
            return
        if self._printing is not None:
            QMessageBox.information(self._window, tr("Внимание"), tr("Принтер уже печатает отправленную деталь."))
            return
        if self._tab.export_spin(printer, on_uploaded=lambda _label, answer: self._after_upload(printer, answer)):
            self._panel.set_print_status(tr("Готовлю файл и отправляю на принтер…"))
            self._log(f"{tr('Отправка на принтер')} {printer.label}…")

    def _after_upload(self, printer: printer_link.PrinterAddress, answer: object) -> None:
        name = answer.get("name", "") if isinstance(answer, dict) else ""
        if not name:
            self._panel.set_print_status(tr("Принтер не назвал загруженный файл."), warn=True)
            return
        self._log(f"{tr('Отправлено на принтер')} {printer.label}: {name}")
        self._call(printer_link.start_print, printer, name,
                   done=lambda _request: self._start_following(printer),
                   failed=lambda msg: self._panel.set_print_status(f"{tr('Принтер не принял печать:')} {msg}",
                                                                   warn=True))

    def _start_following(self, printer: printer_link.PrinterAddress) -> None:
        self._printing = printer
        self._panel.set_print_status(tr("Подтвердите печать на экране принтера: джойстик вправо."))
        self._poll.start()

    def _poll_state(self) -> None:
        printer = self._printing
        if printer is None or self._polling:
            return
        self._polling = True

        def done(state: Any) -> None:
            self._polling = False
            self._show_state(state if isinstance(state, dict) else {})

        def failed(msg: str) -> None:
            self._polling = False
            self._panel.set_print_status(f"{tr('Нет связи с принтером:')} {msg}", warn=True, stoppable=True)
        self._call(printer_link.get_state, printer, done=done, failed=failed)

    def _show_state(self, state: dict[str, Any]) -> None:
        what, text, warn, going = state_text(state)
        if not going:
            self._poll.stop()
            self._printing = None
            self._log(text)
        self._panel.set_print_status(text, warn=warn, stoppable=going and what != "waiting_confirm")

    def stop(self) -> None:
        printer = self._printing
        if printer is None:
            return
        if QMessageBox.question(self._window, tr("Остановить печать?"),
                                tr("Свет погаснет, деталь останется недосвеченной.")) != QMessageBox.StandardButton.Yes:
            return
        self._call(printer_link.stop_print, printer, done=lambda _r: self._poll_state(),
                   failed=lambda msg: QMessageBox.warning(self._window, tr("Не удалось остановить"), msg))

    def _on_spin_failed(self, msg: str) -> None:
        if self._printing is None:
            self._panel.set_print_status(tr("Файл не отправлен.") if msg else "", warn=bool(msg))

    # ------------------------------------------------------------------ USB stick
    def to_stick(self) -> None:
        """Write the .spin onto a USB stick; no stick: the usual "Save as"."""
        sticks = removable.find_sticks()
        if not sticks:
            QMessageBox.information(self._window, tr("Флешка не найдена"),
                                    tr("Вставьте флешку в компьютер. Пока сохраню файл туда, куда скажете."))
            self._tab.export_spin()
            return
        stick = sticks[0]
        if len(sticks) > 1:
            labels = [s.label for s in sticks]
            choice, ok = QInputDialog.getItem(self._window, tr("На какую флешку?"), tr("Флешка"), labels, 0, False)
            if not ok:
                return
            stick = sticks[labels.index(choice)]
        path = os.path.join(stick.path, spin_name(self._name()))
        if os.path.exists(path) and QMessageBox.question(
                self._window, tr("Файл уже есть"),
                f"{tr('На флешке уже есть')} {os.path.basename(path)}. {tr('Заменить?')}") != QMessageBox.StandardButton.Yes:
            return
        if self._tab.export_spin(path=path, on_saved=lambda saved: self._offer_eject(stick, saved)):
            self._panel.set_print_status(f"{tr('Записываю на флешку')} {stick.label}…")

    def _offer_eject(self, stick: removable.Stick, path: str) -> None:
        self._panel.set_print_status(f"{tr('Записано на флешку')} {stick.label}: {os.path.basename(path)}")
        self._log(f".spin: {path}")
        answer = QMessageBox.question(self._window, tr("Готово"),
                                      f"{tr('Файл записан на флешку')} {stick.label}.\n"
                                      f"{tr('Извлечь флешку? Потом вставьте её в Raspberry Pi принтера.')}")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._call(removable.eject, stick,
                   done=lambda ok: self._panel.set_print_status(
                       tr("Флешку можно вынимать.") if ok else tr("Не удалось извлечь: извлеките её сами."),
                       warn=not ok),
                   failed=lambda msg: self._panel.set_print_status(msg, warn=True))

    # ------------------------------------------------------------------ finding and connecting
    def find(self) -> None:
        if self._discover is not None and self._discover.isRunning():
            return
        if not printer_link.zeroconf_available():
            self._log(tr("Поиск по mDNS недоступен (pip install zeroconf): проверяю spinslicer.local."))
        else:
            self._log(tr("Ищу принтеры в сети…"))
        self._discover = DiscoverWorker(parent=self)
        self._discover.found.connect(self._on_found)
        self._discover.start()

    def _on_found(self, printers: list) -> None:
        self._panel.set_printers(printers)
        if printers:
            self._log(f"{tr('Найдено принтеров:')} {len(printers)}")
            if self._panel.selected_printer() is None:
                self._panel.printer_box.setCurrentIndex(1)
        else:
            self._log(tr("Принтер в сети не найден. Введите его адрес."))

    def ask_address(self) -> Optional[printer_link.PrinterAddress]:
        current = self._panel.selected_printer()
        default = current.host if current is not None else str(
            self._settings.value(SETTINGS_ADDRESS, "") or printer_link.DEFAULT_HOST)
        text, ok = QInputDialog.getText(self._window, tr("Адрес принтера"),
                                        tr("IP-адрес или имя принтера, например spinslicer.local"),
                                        text=default)
        if not ok or not text.strip():
            return None
        try:
            address = printer_link.parse_address(text)
        except printer_link.PrinterError as exc:
            QMessageBox.warning(self._window, tr("Адрес принтера"), str(exc))
            return None
        self._panel.set_printers([self._panel.printer_box.itemData(i)
                                  for i in range(1, self._panel.printer_box.count())], keep=address)
        self.connect_printer()
        return address

    def _on_printer_chosen(self, _index: int = 0) -> None:
        if self._panel.selected_printer() is None:
            self._connected = None
            self._panel.set_printer_connected(False, tr("не подключён"))
            return
        self.connect_printer()

    def _connect_stored(self) -> None:
        self.connect_printer()

    def connect_printer(self, printer: Optional[printer_link.PrinterAddress] = None) -> None:
        """Talk to the chosen printer (or `printer`) and take its profile into the slicer."""
        printer = printer or self._panel.selected_printer()
        if printer is None:
            self.ask_address()          # chooses one, and connects to it
            return
        self._settings.setValue(SETTINGS_ADDRESS, f"{printer.host}:{printer.port}")
        self._connected = None
        self._panel.set_printer_connected(False, f"{tr('Подключаюсь к')} {printer.label}…")
        self._profile = ProfileWorker(printer, parent=self)
        self._profile.loaded.connect(self._on_profile)
        self._profile.failed.connect(self._on_profile_failed)
        self._profile.start()

    def _on_profile_failed(self, address: printer_link.PrinterAddress, msg: str) -> None:
        self._log(f"{tr('Принтер недоступен:')} {msg}")
        self._panel.set_printer_connected(False, f"{tr('Не подключён:')} {address.label} {tr('не отвечает')}")

    def _on_profile(self, address: printer_link.PrinterAddress, profile: Optional[dict]) -> None:
        self._connected = address
        if profile is None:
            self._printer_settings = {}
            self._panel.set_printer_connected(True, f"{tr('Подключён:')} {address.label}")
            self._log(f"{address.label}: {tr('принтер не отдаёт профиль, настройки остаются как есть.')}")
            return
        settings = printer_link.profile_settings(profile)
        self._printer_settings = settings
        changed = self._panel.apply_printer_profile(settings)
        software: dict[str, Any] = profile.get("software") or {} if isinstance(profile.get("software"), dict) else {}
        status = f"{tr('Подключён:')} {settings.get('name') or address.label}"
        if software.get("spinpi"):
            status += f" · SpinPi {software['spinpi']}"
        status += f" · {tr('из принтера:')} " + (", ".join(tr(c) for c in changed) if changed else tr("ничего нового"))
        self._panel.set_printer_connected(True, status)
        size = (settings.get("projector_width"), settings.get("projector_height"))
        if None not in size and "разрешение проектора" not in changed:
            self._log(f"{tr('Разрешения проектора нет в списке:')} {size[0]}×{size[1]}")
        name = settings.get("name") or address.label
        if changed:
            self._log(f"{tr('Профиль принтера')} {name}: " + ", ".join(tr(c) for c in changed))
        else:
            self._log(f"{tr('Профиль принтера')} {name}: {tr('ничего не изменилось')}")

    def write_to_printer(self) -> None:
        """Store the slicer's vial, projector and calibration on the printer, after showing what changes."""
        printer = self._connected
        if printer is None:
            self._not_connected()
            return
        diff = printer_link.profile_diff(self._printer_settings, self._panel.printer_values())
        if not diff:
            QMessageBox.information(self._window, tr("Записать в принтер"), tr("В принтере уже те же настройки."))
            return

        def fmt(value: Any) -> str:
            return "—" if value is None else (f"{value:g}" if isinstance(value, float) else str(value))
        lines = "\n".join(f"{tr(label)}: {fmt(theirs)} → {fmt(mine)}" for _key, label, theirs, mine in diff)
        if QMessageBox.question(self._window, tr("Записать в принтер"),
                                f"{tr('В принтере изменится:')}\n\n{lines}") != QMessageBox.StandardButton.Yes:
            return
        values = {key: mine for key, _label, _theirs, mine in diff}

        def written(_result: Any) -> None:
            self._log(tr("Настройки записаны в принтер."))
            self.connect_printer(printer)
        self._call(printer_link.write_profile, printer, values, done=written,
                   failed=lambda msg: QMessageBox.warning(self._window, tr("Не удалось записать в принтер"), msg))
