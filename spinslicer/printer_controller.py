"""Glue between the main window and a SpinPrinter: .spin export, send, printer choice.

The panel (ui_panels.ProcessSettingsPanel) shows the printer list and the links; the
projector tab (video_tab.ProjectorTab) encodes and packs. This object builds the print
job from the settings and talks to the printer in background threads.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

from PyQt6.QtCore import QObject, QSettings
from PyQt6.QtWidgets import QInputDialog, QMessageBox, QWidget

from spinslicer import printer_link
from spinslicer.constants import APP_ORG, APP_TITLE
from spinslicer.i18n import tr
from spinslicer.printer_workers import DiscoverWorker, ProfileWorker
from spinslicer.spin_format import PrintJob, preview_from_frame, render_preview

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
        self._send_after_discovery = False
        projector_tab.job_provider = self._job
        projector_tab.spin_extras = self._extras
        panel.printer_box.currentIndexChanged.connect(self._on_printer_chosen)
        stored = str(self._settings.value(SETTINGS_ADDRESS, "") or "")
        if stored:
            try:
                panel.set_printers([], keep=printer_link.parse_address(stored))
            except printer_link.PrinterError:
                pass

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
        if action == "spin_export":
            self._tab.export_spin()
        elif action == "spin_send":
            self.send()
        elif action == "printer_find":
            self.find()
        elif action == "printer_manual":
            self.ask_address()
        else:
            return False
        return True

    def send(self) -> None:
        printer = self._panel.selected_printer()
        if printer is None:
            printer = self.ask_address()
            if printer is None:
                return
        if self._tab.export_spin(printer):
            self._log(f"{tr('Отправка на принтер')} {printer.label}…")

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
        self._on_printer_chosen()
        return address

    def _on_printer_chosen(self, _index: int = 0) -> None:
        printer = self._panel.selected_printer()
        if printer is None:
            return
        self._settings.setValue(SETTINGS_ADDRESS, f"{printer.host}:{printer.port}")
        self._profile = ProfileWorker(printer, parent=self)
        self._profile.loaded.connect(self._on_profile)
        self._profile.failed.connect(lambda _a, msg: self._log(f"{tr('Принтер недоступен:')} {msg}"))
        self._profile.start()

    def _on_profile(self, address: printer_link.PrinterAddress, profile: Optional[dict]) -> None:
        if profile is None:
            self._log(f"{address.label}: {tr('принтер не отдаёт профиль, настройки остаются как есть.')}")
            return
        settings = printer_link.profile_settings(profile)
        changed = self._panel.apply_printer_profile(settings)
        size = (settings.get("projector_width"), settings.get("projector_height"))
        if None not in size and "разрешение проектора" not in changed:
            self._log(f"{tr('Разрешения проектора нет в списке:')} {size[0]}×{size[1]}")
        name = settings.get("name") or address.label
        if changed:
            self._log(f"{tr('Профиль принтера')} {name}: " + ", ".join(tr(c) for c in changed))
        else:
            self._log(f"{tr('Профиль принтера')} {name}: {tr('ничего не изменилось')}")
