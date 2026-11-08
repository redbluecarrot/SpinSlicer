"""The .spin print file, the MP4 comment tag and the upload to a SpinPrinter."""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from spinslicer import printer_link
from spinslicer.spin_format import (
    MP4_TAG_FORMAT,
    PrintJob,
    SpinFormatError,
    mp4_comment,
    read_spin,
    render_preview,
    rotation_name,
    spin_name,
    write_spin,
)

PLAN = {
    "relative_threshold_dose": 0.5, "process_window": 0.06, "exposure_tolerance_pct": 6.0,
    "frame_exposure_total_s": 0.25, "revolutions": 7.5, "revolution_s": 40.0, "print_time_s": 300.0,
    "warnings": ["Меньше 3 оборотов: тест"],
}


def _job(**kwargs) -> PrintJob:
    values = dict(name="lattice_cube", frames=1200, fps=30.0, width=1280, height=800, plan=PLAN,
                  resin={"alpha_per_mm": 0.08, "gel_dose_mj_cm3": 12.0},
                  printer={"profile_id": "spinslicer-3f2a", "vial_diameter_mm": 25.0,
                           "projector": {"width": 1280, "height": 800, "irradiance_mw_cm2": 4.1}})
    values.update(kwargs)
    return PrintJob(**values)


def _video(tmp_path, size: int = 300_000):
    path = tmp_path / "video.mp4"
    path.write_bytes(np.random.default_rng(1).integers(0, 256, size, dtype=np.uint8).tobytes())
    return path


def test_spin_round_trip(tmp_path):
    video = _video(tmp_path)
    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    faces = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
             [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    preview = render_preview(cube, faces)
    path = tmp_path / "part.spin"
    written = write_spin(path, _job(), video, preview_png=preview, slicer_settings={"grid_res": 256},
                         created="2026-11-07T21:40:00Z")

    with zipfile.ZipFile(path) as archive:                    # any ZIP tool opens it
        infos = archive.infolist()
        assert [i.filename for i in infos] == ["spin.json", "video.mp4", "preview.png", "slicer.json"]
        assert all(i.compress_type == zipfile.ZIP_STORED for i in infos)
        assert archive.read("video.mp4") == video.read_bytes()
        assert json.loads(archive.read("slicer.json")) == {"grid_res": 256}

    spin = read_spin(path)
    m = spin.manifest
    assert m == written
    assert m["format"] == "spin" and m["version"] == 1 and m["name"] == "lattice_cube"
    assert m["created"] == "2026-11-07T21:40:00Z" and m["generator"].startswith("SpinSlicer")
    v = m["video"]
    assert v["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert (v["file"], v["frames"], v["fps"], v["width"], v["height"]) == ("video.mp4", 1200, 30.0, 1280, 800)
    assert v["frames_per_rev"] == 1200 and v["rotation"] == "ccw_from_top" and v["frame0_angle_deg"] == 0.0
    p = m["plan"]
    assert p["revolution_s"] == pytest.approx(40.0)
    assert p["rpm"] == pytest.approx(1.5)
    assert p["revolutions"] == pytest.approx(0.25 * 30.0)
    assert p["print_time_s"] == pytest.approx(300.0)
    assert p["tolerance_pct"] == 6.0 and p["warnings"] == PLAN["warnings"]
    assert m["resin"]["gel_dose_mj_cm3"] == 12.0 and m["printer"]["profile_id"] == "spinslicer-3f2a"
    png = spin.read("preview.png")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    from io import BytesIO

    from PIL import Image
    image = Image.open(BytesIO(png))
    assert image.size == (320, 320) and np.asarray(image)[..., 3].any()


def test_exposure_and_brightness_scale_the_revolutions(tmp_path):
    plan = _job(exposure=1.1, brightness=0.5).plan_section()
    assert plan["revolutions"] == pytest.approx(0.25 * 30.0 * 1.1 / 0.5)
    assert plan["print_time_s"] == pytest.approx(plan["revolutions"] * 40.0)
    unknown = _job(plan={"exposure_tolerance_pct": 0.0, "warnings": []}).plan_section()
    assert unknown["revolutions"] is None and unknown["revolution_s"] == pytest.approx(40.0)


def test_rotation_follows_the_engine_direction():
    assert rotation_name(1) == "ccw_from_top" and rotation_name(-1) == "cw_from_top"
    assert _job(angle_direction=-1).video_section()["rotation"] == "cw_from_top"


def test_damaged_video_is_detected(tmp_path):
    path = tmp_path / "part.spin"
    write_spin(path, _job(), _video(tmp_path))
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF                             # one flipped byte inside video.mp4
    path.write_bytes(bytes(data))
    with pytest.raises(SpinFormatError):
        read_spin(path)
    assert read_spin(path, verify=False).manifest["format"] == "spin"


def test_reader_ignores_unknown_fields_and_rejects_other_files(tmp_path):
    path = tmp_path / "future.spin"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("spin.json", json.dumps({"format": "spin", "version": 1, "hologram": {"x": 1},
                                                  "video": {"file": "video.mp4"}}))
        archive.writestr("video.mp4", b"x")
        archive.writestr("extra.bin", b"y")
    assert read_spin(path).manifest["hologram"] == {"x": 1}
    other = tmp_path / "other.spin"
    with zipfile.ZipFile(other, "w") as archive:
        archive.writestr("video.mp4", b"x")
        archive.writestr("spin.json", "{}")
    with pytest.raises(SpinFormatError):
        read_spin(other)
    (tmp_path / "junk.spin").write_bytes(b"not a zip")
    with pytest.raises(SpinFormatError):
        read_spin(tmp_path / "junk.spin")


def test_spin_name_is_safe():
    assert spin_name("C:\\models\\my part.stl") == "my_part.stl.spin"
    assert spin_name("../../etc/passwd") == "passwd.spin"
    assert spin_name("cube.spin") == "cube.spin"
    assert spin_name("") == "print.spin"


def test_mp4_tag_is_the_flat_plan_spinpi_reads():
    tag = json.loads(mp4_comment(_job()))
    assert tag["format"] == MP4_TAG_FORMAT and tag["format"].startswith("spinslicer-print")
    assert tag["revolutions"] == pytest.approx(7.5) and tag["tolerance_pct"] == 6.0
    assert tag["frames"] == 1200 and tag["rotation"] == "ccw_from_top"
    assert mp4_comment(_job()).isascii()


def test_mp4_export_carries_the_comment_tag(tmp_path):
    pytest.importorskip("cv2")
    from spinslicer import workers
    from spinslicer.workers import VideoExportWorker
    ffmpeg = workers._ffmpeg_executable()
    if ffmpeg is None:
        pytest.skip("no ffmpeg")
    path = tmp_path / "part.mp4"
    comment = mp4_comment(_job(frames=4, width=32, height=32))
    worker = VideoExportWorker([np.full((32, 32), 120, np.uint8)] * 4, 30.0, str(path), comment=comment)
    errors: list[str] = []
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors, errors
    out = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "ffmetadata", "-"],
                         capture_output=True, text=True, timeout=60).stdout
    line = next(x for x in out.splitlines() if x.startswith("comment="))
    tag = json.loads(line[len("comment="):].replace("\\=", "=").replace("\\;", ";").replace("\\#", "#"))
    assert tag["format"] == "spinslicer-print" and tag["revolutions"] == pytest.approx(7.5)


def test_mp4v_fallback_writes_the_sidecar(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from spinslicer import workers
    from spinslicer.workers import VideoExportWorker
    monkeypatch.setattr(workers, "_ffmpeg_executable", lambda: None)
    path = tmp_path / "part.mp4"
    worker = VideoExportWorker([np.full((32, 32), 120, np.uint8)] * 4, 30.0, str(path), comment=mp4_comment(_job()))
    worker.run()
    assert json.loads((tmp_path / "part.spin.json").read_text())["format"] == "spinslicer-print"


# ------------------------------------------------------------------ the printer over HTTP
class _Printer(BaseHTTPRequestHandler):
    uploads: dict[str, bytes] = {}
    profile: dict | None = None

    def log_message(self, *_args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/api/printer" and self.profile is not None:
            return self._json(self.profile)
        self._json({"error": "not found"}, 404)

    def do_PUT(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        if not self.path.startswith("/api/files/") or n <= 0:
            return self._json({"error": "bad size"}, 400)
        name = self.path[len("/api/files/"):]
        self.uploads[name] = self.rfile.read(n)
        self._json({"ok": True, "name": name})


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):  # a cancelled upload breaks off mid-body
        pass


@pytest.fixture
def printer():
    _Printer.uploads = {}
    _Printer.profile = None
    server = _QuietServer(("127.0.0.1", 0), _Printer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield printer_link.PrinterAddress("127.0.0.1", server.server_address[1], name="test")
    finally:
        server.shutdown()
        server.server_close()


def test_upload_sends_the_spin_file(tmp_path, printer):
    path = tmp_path / "part.spin"
    write_spin(path, _job(), _video(tmp_path, 1_000_000))
    seen: list[tuple[int, int]] = []
    answer = printer_link.upload(printer, str(path), progress=lambda s, t: seen.append((s, t)))
    assert answer == {"ok": True, "name": "part.spin"}
    assert _Printer.uploads["part.spin"] == path.read_bytes()
    assert seen[0] == (0, path.stat().st_size) and seen[-1] == (path.stat().st_size,) * 2
    with pytest.raises(printer_link.PrinterError):
        printer_link.upload(printer, str(path), cancelled=lambda: True)


def test_upload_reports_errors(tmp_path):
    path = tmp_path / "part.spin"
    path.write_bytes(b"x")
    closed = printer_link.PrinterAddress("127.0.0.1", 9)      # nothing listens there
    with pytest.raises(printer_link.PrinterError):
        printer_link.upload(closed, str(path), timeout_s=2.0)


def test_printer_profile(printer):
    assert printer_link.get_printer_profile(printer) is None  # older SpinPi: no /api/printer
    _Printer.profile = {
        "id": "spinslicer-3f2a", "name": "Bench printer",
        "projector": {"width": 1920, "height": 1080, "hz": 60, "irradiance_mw_cm2": 4.1, "contrast": 1500},
        "vial": {"diameter_mm": 25.0, "wall_mm": 1.2, "n": 1.47},
        "calibration": {"center_offset_px": [12, -3], "vial_width_px": 860},
        "motor": {"direction": "ccw_from_top"},
        "versions": {"spinpi": "1.0"},
    }
    settings = printer_link.profile_settings(printer_link.get_printer_profile(printer))
    assert settings == {
        "profile_id": "spinslicer-3f2a", "name": "Bench printer", "projector_width": 1920,
        "projector_height": 1080, "projector_hz": 60.0, "irradiance_mw_cm2": 4.1, "contrast": 1500.0,
        "vial_diameter_mm": 25.0, "vial_wall_mm": 1.2, "n_glass": 1.47, "vial_width_px": 860,
        "offset_x_px": 12, "offset_y_px": -3, "motor_direction": 1,
    }
    assert printer_link.profile_settings({"projector": {"resolution": "1280x800"}}) == {
        "projector_width": 1280, "projector_height": 800}


def test_parse_address():
    assert printer_link.parse_address("spinprinter.local") == printer_link.PrinterAddress("spinprinter.local", 80)
    assert printer_link.parse_address("http://192.168.1.7:8080/").port == 8080
    assert printer_link.parse_address("[fe80::1]:81").host == "fe80::1"
    with pytest.raises(printer_link.PrinterError):
        printer_link.parse_address("")


def test_discovery_without_zeroconf_still_offers_the_default(monkeypatch):
    monkeypatch.setattr(printer_link, "discover_mdns", lambda timeout_s=0: [])
    monkeypatch.setattr(printer_link, "resolve_default",
                        lambda host=printer_link.DEFAULT_HOST: printer_link.PrinterAddress(host, source="default"))
    # the new name, and the one the Pi had until now
    assert [p.host for p in printer_link.discover()] == ["spinprinter.local", "spinslicer.local"]


def test_panel_applies_a_printer_profile(qapp=None):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from spinslicer.ui_panels import ProcessSettingsPanel
    _app = QApplication.instance() or QApplication([])
    panel = ProcessSettingsPanel()
    changed = panel.apply_printer_profile({"profile_id": "p1", "vial_diameter_mm": 30.0,
                                           "projector_width": 1920, "projector_height": 1080,
                                           "irradiance_mw_cm2": 5.5, "vial_width_px": 900})
    assert panel.vat_diameter_mm() == 30.0 and panel.screen_size() == (1920, 1080)
    assert panel.projector_irradiance.value() == 5.5 and panel.output_res.value_int() == 900
    assert panel.printer_profile_id() == "p1" and "диаметр колбы" in changed
    address = printer_link.PrinterAddress("10.0.0.5", name="bench")
    panel.set_printers([address])
    panel.printer_box.setCurrentIndex(1)
    assert panel.selected_printer() == address
