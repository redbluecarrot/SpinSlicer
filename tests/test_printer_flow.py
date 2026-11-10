"""Printing on a SpinPrinter from the slicer: start (confirmed on the printer), follow, stop,
write the profile back, and the USB stick."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from spinslicer import printer_link, removable


class _Pi(BaseHTTPRequestHandler):
    posts: list[tuple[str, dict]] = []
    state: dict = {}
    screen = True

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
        if self.path == "/api/state":
            return self._json(self.state)
        self._json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        self.posts.append((self.path, body))
        if self.path == "/api/print":
            if not self.screen:
                return self._json({"ok": False, "error": "экран принтера не подключён", "request": None}, 409)
            if body.get("id") == "lib/missing.spin":
                return self._json({"ok": False, "error": "no such video", "request": None})
            return self._json({"ok": True, "error": None,
                               "request": {"state": "waiting_confirm", "id": body["id"], "name": "part.spin"}})
        if self.path == "/api/result":
            return self._json({"ok": True, "factor_pct": 6.0})
        if self.path in ("/api/stop", "/api/printer", "/api/calibration", "/api/settings"):
            return self._json({"ok": True})
        self._json({"error": "not found"}, 404)


@pytest.fixture
def pi():
    _Pi.posts, _Pi.state, _Pi.screen = [], {}, True
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Pi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield printer_link.PrinterAddress("127.0.0.1", server.server_address[1], name="bench")
    finally:
        server.shutdown()
        server.server_close()


def test_print_is_sent_to_wait_for_the_printers_screen(pi):
    request = printer_link.start_print(pi, "part.spin")
    assert _Pi.posts == [("/api/print", {"id": "lib/part.spin", "confirm": True})]
    assert request["state"] == "waiting_confirm"


def test_print_refused(pi):
    with pytest.raises(printer_link.PrinterError, match="no such video"):
        printer_link.start_print(pi, "missing.spin")
    _Pi.screen = False
    with pytest.raises(printer_link.PrinterError, match="экран"):
        printer_link.start_print(pi, "part.spin")


def test_state_and_stop(pi):
    _Pi.state = {"state": "print", "request": {"state": "started"}, "job": {"progress": 0.5, "turn": 3, "revs": 6,
                                                                            "left_s": 40}}
    st = printer_link.get_state(pi)
    assert printer_link.describe_state(st) == ("printing", 0.5, 3, 6, 40)
    _Pi.state = {"state": "idle", "request": {"state": "waiting_confirm"}}
    assert printer_link.describe_state(printer_link.get_state(pi))[0] == "waiting_confirm"
    _Pi.state = {"state": "idle", "request": {"state": "cancelled"}}
    assert printer_link.describe_state(printer_link.get_state(pi))[0] == "cancelled"
    _Pi.state = {"state": "done", "request": {"state": "started"}}
    assert printer_link.describe_state(printer_link.get_state(pi))[0] == "done"
    _Pi.state = {"state": "error", "msg": "ESP не отвечает", "request": {"state": "started"}}
    assert printer_link.describe_state(printer_link.get_state(pi))[0] == "error"
    printer_link.stop_print(pi)
    assert _Pi.posts[-1][0] == "/api/stop"


def test_state_of_our_own_print_only():
    """Right after the yes the printer still says "done" for its last print: that is not ours."""
    describe = printer_link.describe_state
    assert describe({"state": "done", "request": {"state": "starting"}})[0] == "starting"
    old = {"state": "done", "job": {"seq": 4, "progress": 1.0}, "request": {"state": "started", "job_seq": 5}}
    assert describe(old)[0] == "starting"
    ours = {"state": "print", "job": {"seq": 5, "progress": 0.25, "turn": 2, "revs": 8, "left_s": 90},
            "request": {"state": "started", "job_seq": 5}}
    assert describe(ours) == ("printing", 0.25, 2, 8, 90)
    assert describe(dict(ours, state="done"))[0] == "done"
    later = {"state": "print", "job": {"seq": 6}, "request": {"state": "started", "job_seq": 5}}
    assert describe(later)[0] == "replaced"   # someone started another print after ours


def test_can_confirm():
    assert printer_link.can_confirm({"features": ["confirm", "job_seq"]})
    assert not printer_link.can_confirm({"features": ["job_seq"]})
    assert printer_link.can_confirm({"software": {"spinpi": "0.8.0"}})       # confirms, no features list yet
    assert not printer_link.can_confirm({"software": {"spinpi": "0.6.0"}})   # would start at once
    assert not printer_link.can_confirm({})
    assert not printer_link.can_confirm(None)                                 # no /api/printer at all


def test_profile_offset_x_only():
    """SpinPi sends center_offset_px [x, 0]: it has no vertical offset, the slicer's Y stays."""
    s = printer_link.profile_settings({"calibration": {"center_offset_px": [7, 0], "vial_width_px": 812}})
    assert s["offset_x_px"] == 7 and "offset_y_px" not in s and s["vial_width_px"] == 812
    s = printer_link.profile_settings({"calibration": {"offset_x_px": 3, "center_offset_px": [7, 0]}})
    assert s["offset_x_px"] == 3
    uncalibrated = {"calibration": {"offset_px": None, "offset_x_px": None, "center_offset_px": None,
                                    "vial_width_px": None}}
    assert printer_link.profile_settings(uncalibrated) == {}


def test_profile_written_back_to_the_printer(pi):
    printer = {"vial_diameter_mm": 25.0, "vial_wall_mm": 1.2, "n_glass": 1.47, "irradiance_mw_cm2": 4.1,
               "contrast": 1000.0, "vial_width_px": 400, "offset_x_px": 0}
    slicer = dict(printer, vial_diameter_mm=30.0, vial_width_px=860, offset_x_px=12, irradiance_mw_cm2=0.0)
    diff = printer_link.profile_diff(printer, slicer)
    assert [d[0] for d in diff] == ["vial_diameter_mm", "vial_width_px", "offset_x_px"]  # 0 irradiance = unknown
    printer_link.write_profile(pi, {k: slicer[k] for k, *_ in diff})
    assert _Pi.posts == [("/api/printer", {"vial_diameter_mm": 30.0}),
                         ("/api/calibration", {"offset_px": 12, "vial_px": 860})]
    _Pi.posts.clear()
    printer_link.write_profile(pi, {"n_glass": 1.5, "contrast": 900.0})
    assert _Pi.posts == [("/api/printer", {"vial_n": 1.5, "contrast": 900.0})]


# ------------------------------------------------------------------ USB sticks
def test_sticks_on_linux():
    mounts = ("/dev/sda2 / ext4 rw 0 0\n"
              "/dev/sdb1 /media/anna/SPIN\\040STICK vfat rw 0 0\n"
              "/dev/sdc1 /run/media/anna/KINGSTON exfat rw 0 0\n"
              "tmpfs /run/user/1000 tmpfs rw 0 0\n")
    sticks = removable.find_sticks(system="Linux", mounts=mounts)
    assert [(s.path, s.label) for s in sticks] == [("/media/anna/SPIN STICK", "SPIN STICK"),
                                                    ("/run/media/anna/KINGSTON", "KINGSTON")]


def test_sticks_on_windows():
    types = {"C:\\": 3, "E:\\": 2, "F:\\": 2, "D:\\": 5}
    labels = {"E:\\": "SPIN", "F:\\": ""}
    sticks = removable.find_sticks(system="Windows", drives=list(types), drive_type=types.get,
                                   volume_label=labels.get)
    assert [(s.path, s.label) for s in sticks] == [("E:\\", "SPIN (E:)"), ("F:\\", "F:")]


def test_sticks_on_mac(tmp_path):
    (tmp_path / "Macintosh HD").mkdir()
    (tmp_path / "SPIN").mkdir()
    sticks = removable.find_sticks(system="Darwin", volumes=str(tmp_path),
                                   is_system=lambda p: p.endswith("Macintosh HD"))
    assert [s.label for s in sticks] == ["SPIN"]


# ------------------------------------------------------------------ the slicer's side, with Qt
_APP = []


def _qt():
    """The QApplication, kept alive for the whole run (a collected one takes Qt down with it)."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    return _APP[0]


def _wait(cond, timeout=5.0):
    import time

    from PyQt6.QtWidgets import QApplication
    end = time.time() + timeout
    while time.time() < end:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_state_text():
    from spinslicer.printer_controller import state_text
    assert state_text({"request": {"state": "waiting_confirm"}})[0::3] == ("waiting_confirm", True)
    kind, text, warn, going = state_text({"state": "print", "request": {"state": "started"},
                                          "job": {"progress": 0.42, "turn": 3, "revs": 6, "left_s": 61}})
    assert (kind, going, warn) == ("printing", True, False) and "42%" in text and "1:01" in text
    assert state_text({"state": "done", "request": {"state": "started"}})[3] is False
    assert state_text({"state": "idle", "msg": "остановлено с экрана принтера",
                       "request": {"state": "started"}})[0:3:2] == ("stopped", True)


class _FakeTab:
    """What the controller needs from the projector tab: export_spin, uploading at once."""

    def __init__(self):
        from PyQt6.QtCore import QObject, pyqtSignal

        class Signals(QObject):
            spinFailed = pyqtSignal(str)
        self._signals = Signals()
        self.spinFailed = self._signals.spinFailed
        self.job_provider = self.spin_extras = None
        self.exports: list[tuple] = []

    def export_spin(self, printer=None, *, path=None, on_saved=None, on_uploaded=None):
        self.exports.append((printer, path))
        if printer is not None and on_uploaded:
            on_uploaded(printer.label, {"ok": True, "name": "part.spin"})
        if path is not None and on_saved:
            on_saved(path)
        return True


def test_print_from_the_slicer_follows_the_printer(pi, monkeypatch):
    _qt()
    from PyQt6.QtWidgets import QMessageBox, QWidget

    from spinslicer.printer_controller import PrinterController
    from spinslicer.ui_panels import ProcessSettingsPanel
    window, panel, tab = QWidget(), ProcessSettingsPanel(), _FakeTab()
    ctl = PrinterController(window, panel, tab, lambda: "", lambda: None, lambda _m: None)
    assert not panel.print_button.isEnabled()                       # no printer, no frames
    assert not panel.stick_button.isEnabled()                       # no frames yet
    ctl._on_profile(pi, {"id": "spinpi-1", "name": "Bench", "vial": {"diameter_mm": 25.0},
                         "software": {"spinpi": "0.8.2"}, "features": ["confirm", "job_seq"]})
    assert not panel.print_button.isEnabled()                       # connected, but no frames
    panel.set_results({}, "frames")                                 # the light is computed
    assert panel.print_button.isEnabled() and not panel.r_printer_write.isHidden()
    assert "Bench" in panel.printer_state.text() and "0.8.2" in panel.printer_state.text()

    _Pi.state = {"state": "idle", "request": {"state": "waiting_confirm"}}
    ctl.print_now()
    assert _wait(lambda: ("/api/print", {"id": "lib/part.spin", "confirm": True}) in _Pi.posts)
    assert _wait(lambda: ctl._printing is not None)
    # the progress card: the stage, then a bar with turn and time, and its own Stop
    assert _wait(lambda: not panel.prog_card.isHidden())
    assert "подтвержд" in panel.prog_stage.text() or "confirm" in panel.prog_stage.text().lower()
    _Pi.state = {"state": "print", "request": {"state": "started"},
                 "job": {"progress": 0.5, "turn": 3, "revs": 6, "left_s": 30}}
    assert _wait(lambda: panel.prog_bar.value() == 500, timeout=4)
    assert "3" in panel.prog_detail.text() and "0:30" in panel.prog_detail.text()
    assert not panel.prog_stop.isHidden()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    ctl.stop()
    assert _wait(lambda: any(p[0] == "/api/stop" for p in _Pi.posts))
    asked = []
    monkeypatch.setattr(PrinterController, "_ask_result", lambda self, printer: asked.append(printer))
    _Pi.state = {"state": "done", "request": {"state": "started"}}
    assert _wait(lambda: ctl._printing is None, timeout=4)
    assert panel.prog_card.isHidden() and panel.stop_link.isHidden()
    assert asked == [pi]  # "how did it come out?" is asked in the slicer, not only on the printer

    # the slicer's vial goes back to the printer, after the question
    panel.diameter.setValue(30.0)
    ctl.write_to_printer()
    assert _wait(lambda: any(path == "/api/printer" and body.get("vial_diameter_mm") == 30.0
                             for path, body in _Pi.posts))
    assert _wait(lambda: ctl._connected is not None and not ctl._calls)   # reconnected, nothing left running


def _controller():
    _qt()
    from PyQt6.QtWidgets import QWidget

    from spinslicer.printer_controller import PrinterController
    from spinslicer.ui_panels import ProcessSettingsPanel
    window, panel, tab = QWidget(), ProcessSettingsPanel(), _FakeTab()
    ctl = PrinterController(window, panel, tab, lambda: "", lambda: None, lambda _m: None)
    return window, panel, tab, ctl


def test_old_printer_is_not_sent_a_print(pi, monkeypatch):
    """An older SpinPi ignores "confirm" and would start the motor and the light at once."""
    from PyQt6.QtWidgets import QMessageBox
    window, panel, tab, ctl = _controller()
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[1]))
    ctl._on_profile(pi, {"id": "spinpi-1", "software": {"spinpi": "0.6.0"}})
    ctl.print_now()
    assert warned and tab.exports == [] and _Pi.posts == []


def test_busy_printer_is_not_sent_a_print(pi):
    """Checked before the file is packed and uploaded, not after."""
    window, panel, tab, ctl = _controller()
    ctl._on_profile(pi, {"id": "spinpi-1", "features": ["confirm", "job_seq"]})
    _Pi.state = {"state": "print", "job": {"seq": 3}}
    ctl.print_now()
    assert _wait(lambda: "печатает" in panel.r_spin.effect.text() or "printing" in panel.r_spin.effect.text())
    assert tab.exports == [] and not any(p[0] == "/api/print" for p in _Pi.posts)
    assert _wait(lambda: not ctl._calls)


def test_profile_marks_only_what_changed(pi):
    window, panel, tab, ctl = _controller()
    panel.offset_y.setValue(5)
    panel.diameter.setValue(25.0)
    changed = panel.apply_printer_profile(printer_link.profile_settings(
        {"vial": {"diameter_mm": 25.0, "wall_mm": 1.7}, "calibration": {"center_offset_px": [4, 0]}}))
    assert "диаметр колбы" not in changed               # was already 25
    assert "стенка" in changed and "сдвиг по X" in changed
    assert panel.offset_y.value() == 5                  # the printer has no Y offset: the slicer's stays


def test_to_usb_stick(tmp_path, monkeypatch):
    _qt()
    from PyQt6.QtWidgets import QMessageBox, QWidget

    from spinslicer.printer_controller import PrinterController
    from spinslicer.ui_panels import ProcessSettingsPanel
    window, panel, tab = QWidget(), ProcessSettingsPanel(), _FakeTab()
    ctl = PrinterController(window, panel, tab, lambda: "", lambda: None, lambda _m: None)
    stick = removable.Stick(str(tmp_path), "SPIN (E:)")
    monkeypatch.setattr(removable, "find_sticks", lambda: [stick])
    ejected = []
    monkeypatch.setattr(removable, "eject", lambda s: ejected.append(s) or True)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    ctl.to_stick()
    assert tab.exports == [(None, str(tmp_path / "print.spin"))]
    assert _wait(lambda: ejected == [stick])
    assert _wait(lambda: "вынимать" in panel.r_spin.effect.text() or "removed" in panel.r_spin.effect.text())


# ---------------------------------------------------------------- the seams with SpinPi
def test_the_baked_centering_goes_into_the_file():
    """SpinPi shifts by its calibration minus video.offset_px: the shift is not applied twice."""
    from spinslicer.spin_format import PrintJob
    job = PrintJob(name="p", frames=360, fps=24.0, width=1280, height=800, offset_px=(12, -3))
    assert job.manifest("0" * 64)["video"]["offset_px"] == [12, -3]
    assert job.mp4_tag()["offset_px"] == [12, -3]
    assert PrintJob(name="p", frames=360, fps=24.0, width=1280, height=800).video_section()["offset_px"] == [0, 0]


def test_spinprinter_projector_size_is_in_the_list():
    from spinslicer.projector_canvas import RESOLUTIONS, RESOLUTIONS_V2
    assert (1280, 800) in RESOLUTIONS
    assert set(RESOLUTIONS_V2) <= set(RESOLUTIONS)  # an old saved choice still has its size


def test_the_address_prompt_offers_a_name_that_exists(monkeypatch):
    resolves = {"spinslicer.local"}
    monkeypatch.setattr(printer_link, "resolve_default",
                        lambda host=printer_link.DEFAULT_HOST: printer_link.PrinterAddress(host) if host in resolves else None)
    assert printer_link.first_known_host() == "spinslicer.local"
    resolves.clear()
    assert printer_link.first_known_host() == "spinprinter.local"  # the printer's own name
    resolves.add("spinprinter.local")
    assert printer_link.first_known_host() == "spinprinter.local"



# ---------------------------------------------------------------- the print step talks to the printer
def test_the_answer_goes_to_the_printer(pi):
    printer_link.send_result(pi, "under")
    assert ("/api/result", {"result": "under"}) in _Pi.posts
    with pytest.raises(printer_link.PrinterError):
        printer_link.send_result(pi, "great")


def test_refusals_in_the_slicers_words():
    from spinslicer import i18n
    assert printer_link.error_from({"ok": False, "error": "busy", "code": "busy"}) == i18n.tr("Принтер занят другой печатью.")
    assert printer_link.error_from({"ok": False, "error": "что-то новое", "code": "later"}) == "что-то новое"


def test_checks_before_sending():
    _qt()
    from spinslicer.ui_panels import ProcessSettingsPanel
    panel = ProcessSettingsPanel()
    panel.apply_printer_profile(printer_link.profile_settings({
        "id": "p1", "projector": {"width": 1280, "height": 800},
        "software": {"engine": "angle"}, "storage": {"free_gb": 12.4}}))
    panel._print_revs = 5.35
    checks = dict((text, ok) for ok, text in panel.print_checks())
    size_ok = [ok for text, ok in checks.items() if "1280×800" in text]
    assert size_ok and size_ok[0] == (panel.screen_size() == (1280, 800))
    assert any("5" in text and ok for text, ok in checks.items())         # whole turns, the dose as planned
    assert any("12,4" in text and ok for text, ok in checks.items())      # room on the printer


# ---------------------------------------------------------------- step 5b: the answers reach the resin's dose
def test_resin_id_is_the_printers():
    # SpinPi makes the same id for files without one (spinpi.resin_id)
    import hashlib
    assert printer_link.resin_id(0.08, 1.5) == "r-" + hashlib.sha1(b"0.0800/1.5000").hexdigest()[:10]


def test_dose_suggestion():
    rid = printer_link.resin_id(0.08, 1.5)

    def p(result, factor=1.0, resin_factor=1.0, gel=12.0, r=rid):
        return {"result": result, "factor": factor, "resin_factor": resin_factor, "resin_id": r,
                "resin": {"gel_dose_mj_cm3": gel}}
    # under, under, then ok at +12 %: the dose that was right is 12 x 1.12
    hist = {"prints": [p("under"), p("under", resin_factor=1.06), p("ok", resin_factor=1.12)]}
    dose, n = printer_link.dose_suggestion(hist, rid, 12.0)
    assert abs(dose - 13.44) < 1e-6 and n == 3
    # under and over without an ok: in the middle
    hist = {"prints": [p("under", resin_factor=1.0), p("over", resin_factor=1.2)]}
    dose, _n = printer_link.dose_suggestion(hist, rid, 12.0)
    assert abs(dose - 12.0 * 1.2 ** 0.5) < 1e-6
    # one direction only, or another resin: nothing to say
    assert printer_link.dose_suggestion({"prints": [p("under"), p("under")]}, rid, 12.0) is None
    assert printer_link.dose_suggestion({"prints": [p("ok", r="other")]}, rid, 12.0) is None


def test_applying_resets_the_printers_correction(pi):
    printer_link.reset_resin_factor(pi, "r-1")
    assert ("/api/settings", {"resin_factor": {"resin_id": "r-1", "value": 1.0}}) in _Pi.posts
