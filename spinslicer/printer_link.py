"""Talking to a SpinPrinter (SpinPi) over the network.

* Discovery: SpinPi announces ``_spinprinter._tcp`` over mDNS with the TXT fields
  ``name``, ``id`` and ``api`` (``zeroconf``). The host names ``spinprinter.local`` (the
  printer's name; SpinPi 0.8.9+ answers to it on any Raspberry) and ``spinslicer.local`` (what
  older SD cards named it) are tried as well, and an address typed by hand always works.
* ``GET /api/printer``: the printer profile (projector, vial, calibration). Older
  SpinPi versions do not have it yet, so its absence is not an error.
* ``PUT /api/files/<name>.spin``: upload of a print file, with progress.

Only the standard library is needed (``http.client``); nothing here touches Qt.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import threading
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable, Optional

SERVICE_TYPE = "_spinprinter._tcp.local."
DEFAULT_HOST = "spinprinter.local"
# The name SpinPrinter's install guide gave the Raspberry Pi until now: still looked for, so an
# existing printer is found without renaming it.
LEGACY_HOSTS = ("spinslicer.local",)
DEFAULT_PORT = 80
TIMEOUT_S = 10.0
CHUNK = 256 * 1024


class PrinterError(RuntimeError):
    """The printer could not be reached or refused the request."""


@dataclass(frozen=True)
class PrinterAddress:
    host: str
    port: int = DEFAULT_PORT
    name: str = ""
    printer_id: str = ""
    source: str = "manual"          # mdns | default | manual

    @property
    def label(self) -> str:
        where = self.host if self.port == DEFAULT_PORT else f"{self.host}:{self.port}"
        return f"{self.name} ({where})" if self.name and self.name != self.host else where

    def url(self, path: str) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}{path}"


def parse_address(text: str) -> PrinterAddress:
    """``host``, ``host:port``, ``http://host:port/…`` or ``[v6]:port`` → address."""
    text = (text or "").strip()
    if not text:
        raise PrinterError("Empty printer address.")
    if "://" not in text:
        text = "http://" + text
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in ("http", ""):
        raise PrinterError("Only http:// printer addresses are supported.")
    host = parts.hostname
    if not host:
        raise PrinterError(f"Not a printer address: {text}")
    try:
        port = parts.port or DEFAULT_PORT
    except ValueError as exc:
        raise PrinterError(f"Bad port in {text}") from exc
    return PrinterAddress(host=host, port=port, source="manual")


# ------------------------------------------------------------------ discovery
def zeroconf_available() -> bool:
    try:
        import zeroconf  # noqa: F401
    except ImportError:
        return False
    return True


def _txt(properties: dict[Any, Any], key: str) -> str:
    value = properties.get(key.encode()) if properties else None
    if value is None and properties:
        value = properties.get(key)
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return "" if value is None else str(value)


def discover_mdns(timeout_s: float = 2.5) -> list[PrinterAddress]:
    """Printers announcing ``_spinprinter._tcp``; [] when zeroconf is missing."""
    try:
        from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
    except ImportError:
        return []
    found: dict[str, PrinterAddress] = {}
    lock = threading.Lock()

    class Listener(ServiceListener):
        def _resolve(self, zc: Any, type_: str, name: str) -> None:
            info = zc.get_service_info(type_, name, timeout=int(timeout_s * 1000))
            if info is None:
                return
            addresses = info.parsed_addresses() if hasattr(info, "parsed_addresses") else []
            host = next((a for a in addresses if ":" not in a), addresses[0] if addresses else "")
            if not host:
                host = (info.server or "").rstrip(".")
            if not host:
                return
            props = info.properties or {}
            label = _txt(props, "name") or name.split(".")[0]
            with lock:
                found[name] = PrinterAddress(host=host, port=int(info.port or DEFAULT_PORT), name=label,
                                             printer_id=_txt(props, "id"), source="mdns")

        def add_service(self, zc: Any, type_: str, name: str) -> None:
            self._resolve(zc, type_, name)

        def update_service(self, zc: Any, type_: str, name: str) -> None:
            self._resolve(zc, type_, name)

        def remove_service(self, zc: Any, type_: str, name: str) -> None:
            with lock:
                found.pop(name, None)

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, SERVICE_TYPE, Listener())
        threading.Event().wait(timeout_s)
    finally:
        zc.close()
    with lock:
        return sorted(found.values(), key=lambda p: p.label.lower())


def resolve_default(host: str = DEFAULT_HOST) -> Optional[PrinterAddress]:
    """``spinprinter.local`` when the system resolver knows it (Bonjour, avahi, nss-mdns)."""
    try:
        socket.getaddrinfo(host, DEFAULT_PORT, proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError):
        return None
    return PrinterAddress(host=host, port=DEFAULT_PORT, name=host, source="default")


def known_hosts() -> tuple[str, ...]:
    """Every host name a SpinPrinter may have, the usual one first."""
    return (DEFAULT_HOST, *LEGACY_HOSTS)


def first_known_host() -> str:
    """The first of known_hosts() the system resolves; else the printer's own name, spinprinter.local."""
    for host in known_hosts():
        if resolve_default(host) is not None:
            return host
    return DEFAULT_HOST


def discover(timeout_s: float = 2.5) -> list[PrinterAddress]:
    """mDNS printers, then ``spinprinter.local`` if it resolves and is not already listed."""
    printers = discover_mdns(timeout_s)
    for host in (DEFAULT_HOST, *LEGACY_HOSTS):
        if any(p.host.rstrip(".").lower() == host for p in printers):
            continue
        fallback = resolve_default(host)
        if fallback is not None:
            printers.append(fallback)
    return printers


# ------------------------------------------------------------------ HTTP
def _connection(address: PrinterAddress, timeout_s: float) -> http.client.HTTPConnection:
    return http.client.HTTPConnection(address.host, address.port, timeout=timeout_s)


# SpinPi 0.8.9+ answers a refusal with a code too: the text is then the slicer's, in its language.
ERROR_CODES: dict[str, str] = {
    "busy": "Принтер занят другой печатью.",
    "no_file": "На принтере нет такого файла.",
    "bad_file": "Файл на принтере повреждён: отправь его ещё раз.",
    "no_screen": "Экран принтера не подключён: подтвердить печать негде.",
    "no_hdmi": "Проектор не подключён к принтеру (HDMI).",
    "reversed_mpv": "Видео для другого направления вращения: на принтере нужен движок angle.",
    "old_board": "Прошивка платы принтера не спрашивает «Печатать?»: обнови её.",
    "bad_request": "Принтер не понял запрос.",
    "foreign": "Принтер отказал запросу из этой программы.",
}


def error_from(data: Any) -> str:
    """The text for a refusal: the code's, else what the printer wrote."""
    if not isinstance(data, dict):
        return ""
    code = str(data.get("code") or "")
    if code in ERROR_CODES:
        from spinslicer.i18n import tr
        return tr(ERROR_CODES[code])
    return str(data.get("error") or "")


def _error_text(status: int, reason: str, body: bytes) -> str:
    detail = ""
    try:
        data = json.loads(body.decode("utf-8"))
        if isinstance(data, dict) and (data.get("error") or data.get("code")):
            detail = error_from(data)
    except (ValueError, UnicodeDecodeError):
        detail = body[:200].decode("utf-8", "replace").strip()
    return f"HTTP {status} {reason}" + (f": {detail}" if detail else "")


def get_json(address: PrinterAddress, path: str, timeout_s: float = TIMEOUT_S) -> Any:
    conn = _connection(address, timeout_s)
    try:
        conn.request("GET", path, headers={"Accept": "application/json"})
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise PrinterError(_error_text(response.status, response.reason, body))
        return json.loads(body.decode("utf-8"))
    except (OSError, http.client.HTTPException) as exc:
        raise PrinterError(f"{address.label}: {exc}") from exc
    except ValueError as exc:
        raise PrinterError(f"{address.label}{path}: not JSON") from exc
    finally:
        conn.close()


def get_printer_profile(address: PrinterAddress, timeout_s: float = TIMEOUT_S) -> Optional[dict[str, Any]]:
    """``GET /api/printer``; None when the printer's software does not have it yet (404)."""
    try:
        data = get_json(address, "/api/printer", timeout_s)
    except PrinterError as exc:
        if "HTTP 404" in str(exc) or "HTTP 501" in str(exc):
            return None
        raise
    return data if isinstance(data, dict) else None


def upload(address: PrinterAddress, file_path: str, name: Optional[str] = None, *,
           progress: Optional[Callable[[int, int], None]] = None,
           cancelled: Optional[Callable[[], bool]] = None,
           timeout_s: float = 60.0) -> dict[str, Any]:
    """``PUT /api/files/<name>`` with the file as the body; returns the printer's JSON answer."""
    from spinslicer.spin_format import spin_name

    name = spin_name(name or os.path.basename(file_path))
    total = os.path.getsize(file_path)
    if total <= 0:
        raise PrinterError("The file is empty.")
    path = "/api/files/" + urllib.parse.quote(name)
    conn = _connection(address, timeout_s)
    try:
        conn.putrequest("PUT", path)
        conn.putheader("Content-Type", "application/zip")
        conn.putheader("Content-Length", str(total))
        conn.endheaders()
        sent = 0
        if progress:
            progress(0, total)
        with open(file_path, "rb") as handle:
            while True:
                if cancelled is not None and cancelled():
                    raise PrinterError("Upload cancelled.")
                block = handle.read(CHUNK)
                if not block:
                    break
                conn.send(block)
                sent += len(block)
                if progress:
                    progress(sent, total)
        response = conn.getresponse()
        body = response.read()
        if response.status not in (200, 201):
            raise PrinterError(_error_text(response.status, response.reason, body))
        try:
            answer = json.loads(body.decode("utf-8")) if body else {}
        except ValueError:
            answer = {}
        if isinstance(answer, dict) and answer.get("ok") is False:
            raise PrinterError(str(answer.get("error") or "the printer refused the file"))
        return answer if isinstance(answer, dict) else {}
    except (OSError, http.client.HTTPException) as exc:
        raise PrinterError(f"{address.label}: {exc}") from exc
    finally:
        conn.close()


# ------------------------------------------------------------------ printer profile
def _pick(data: Any, *paths: str) -> Any:
    """First value found under any of the dotted paths."""
    for path in paths:
        node = data
        for key in path.split("."):
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                node = None
                break
        if node is not None:
            return node
    return None


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def profile_settings(profile: dict[str, Any]) -> dict[str, Any]:
    """The slicer settings a ``GET /api/printer`` profile defines (only those present).

    Keys: ``profile_id``, ``name``, ``projector_width``, ``projector_height``,
    ``projector_hz``, ``irradiance_mw_cm2``, ``contrast``, ``vial_diameter_mm``,
    ``vial_wall_mm``, ``n_glass``, ``vial_width_px``, ``offset_x_px``, ``offset_y_px``,
    ``motor_direction``.
    """
    out: dict[str, Any] = {}

    def put(key: str, value: Any, *, integer: bool = False, positive: bool = True) -> None:
        number = _number(value)
        if number is None or (positive and number <= 0):
            return
        out[key] = int(round(number)) if integer else number

    pid = _pick(profile, "id", "profile_id")
    if pid:
        out["profile_id"] = str(pid)
    name = _pick(profile, "name")
    if name:
        out["name"] = str(name)
    width = _pick(profile, "projector.width", "projector.width_px")
    height = _pick(profile, "projector.height", "projector.height_px")
    resolution = _pick(profile, "projector.resolution")
    if isinstance(resolution, str) and "x" in resolution.lower():
        width, height = resolution.lower().split("x", 1)
    elif isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        width, height = resolution
    put("projector_width", width, integer=True)
    put("projector_height", height, integer=True)
    put("projector_hz", _pick(profile, "projector.hz", "projector.refresh_hz", "projector.fps"))
    put("irradiance_mw_cm2", _pick(profile, "projector.irradiance_mw_cm2", "projector.irradiance"))
    put("contrast", _pick(profile, "projector.contrast"))
    put("vial_diameter_mm", _pick(profile, "vial.diameter_mm", "vial_diameter_mm", "vial.diameter"))
    put("vial_wall_mm", _pick(profile, "vial.wall_mm", "vial.wall"))
    put("n_glass", _pick(profile, "vial.n_glass", "vial.refractive_index", "vial.n"))
    put("vial_width_px", _pick(profile, "calibration.vial_width_px", "calibration.width_px"), integer=True)
    put("offset_x_px", _pick(profile, "calibration.offset_x_px", "calibration.center_offset_x_px",
                             "calibration.offset_x"),
        integer=True, positive=False)
    put("offset_y_px", _pick(profile, "calibration.offset_y_px", "calibration.center_offset_y_px",
                             "calibration.offset_y"),
        integer=True, positive=False)
    offset = _pick(profile, "calibration.center_offset_px")
    if isinstance(offset, (list, tuple)) and len(offset) == 2:
        if "offset_x_px" not in out:
            put("offset_x_px", offset[0], integer=True, positive=False)
        # SpinPi has no vertical offset and always sends [x, 0]: taking that 0 reset the slicer's
        # own Y offset on every connect. A Y only when the printer really has one.
        if "offset_y_px" not in out and _number(offset[1]):
            put("offset_y_px", offset[1], integer=True, positive=False)
    direction = _pick(profile, "motor.direction")
    if direction in (1, -1, "1", "-1"):
        out["motor_direction"] = int(direction)
    elif isinstance(direction, str) and direction in ("ccw_from_top", "cw_from_top"):
        out["motor_direction"] = 1 if direction == "ccw_from_top" else -1
    # for the checks before sending (not settings of this slicer)
    engine = _pick(profile, "software.engine")
    if isinstance(engine, str):
        out["engine"] = engine   # "angle": the printer makes up a fraction of a turn with the speed
    free = _pick(profile, "storage.free_gb")
    if _number(free) is not None:
        out["free_gb"] = float(free)
    return out


# ------------------------------------------------------------------ printing
def post_json(address: PrinterAddress, path: str, body: dict[str, Any],
              timeout_s: float = TIMEOUT_S) -> dict[str, Any]:
    """``POST`` a JSON body; the printer's JSON answer. ``"ok": false`` or an HTTP error → PrinterError."""
    data = json.dumps(body).encode("utf-8")
    conn = _connection(address, timeout_s)
    try:
        conn.request("POST", path, body=data, headers={"Content-Type": "application/json",
                                                         "Accept": "application/json"})
        response = conn.getresponse()
        raw = response.read()
        if response.status not in (200, 201):
            raise PrinterError(_error_text(response.status, response.reason, raw))
        try:
            answer = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            answer = {}
        if isinstance(answer, dict) and answer.get("ok") is False:
            raise PrinterError(error_from(answer) or "the printer refused")
        return answer if isinstance(answer, dict) else {}
    except (OSError, http.client.HTTPException) as exc:
        raise PrinterError(f"{address.label}: {exc}") from exc
    finally:
        conn.close()


def version_tuple(text: Any) -> Optional[tuple[int, int, int]]:
    """``"0.8.1"`` / ``"v0.8.1-dev"`` → ``(0, 8, 1)``; None when there is no version in it."""
    import re
    match = re.match(r"\s*v?(\d+)\.(\d+)(?:\.(\d+))?", str(text or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)


CONFIRM_SPINPI = (0, 8, 0)   # the first SpinPi that waits for a yes on the printer's screen


def can_confirm(profile: Optional[dict[str, Any]]) -> bool:
    """Does this printer wait for "Print?" on its screen before it starts?

    An older SpinPi ignores ``"confirm": true`` and starts at once: the motor and the light,
    with nobody checking that the vial is in place. SpinPi 0.8.2+ lists ``features``; 0.8.0
    and 0.8.1 already confirm but only say their version."""
    if not isinstance(profile, dict):
        return False
    features = profile.get("features")
    if isinstance(features, list):
        return "confirm" in features
    software = profile.get("software")
    version = version_tuple(software.get("spinpi") if isinstance(software, dict) else None)
    return version is not None and version >= CONFIRM_SPINPI


def is_busy(state: dict[str, Any]) -> bool:
    """The printer is running a print (spin-up, warm-up, dark pause or light)."""
    return str(state.get("state")) in ("wait", "print")


def start_print(address: PrinterAddress, name: str) -> dict[str, Any]:
    """Ask the printer to print an uploaded ``.spin``. It does not start by itself: the printer's
    screen asks "Print?" and a yes there starts it (the vial has to be in place). Returns the
    request (``state`` = ``waiting_confirm``)."""
    answer = post_json(address, "/api/print", {"id": "lib/" + name, "confirm": True})
    return answer.get("request") or {"state": "waiting_confirm"}


def get_state(address: PrinterAddress, timeout_s: float = 5.0) -> dict[str, Any]:
    data = get_json(address, "/api/state", timeout_s)
    return data if isinstance(data, dict) else {}


def stop_print(address: PrinterAddress) -> None:
    post_json(address, "/api/stop", {})


RESULTS = ("under", "ok", "over")


def resin_id(alpha_per_mm: float, n_resin: float) -> str:
    """Which resin a print is for, from what defines it optically. SpinPi makes the same id for files
    without one, so its corrections and the slicer's suggestion talk about the same resin."""
    import hashlib
    return "r-" + hashlib.sha1(f"{float(alpha_per_mm):.4f}/{float(n_resin):.4f}".encode()).hexdigest()[:10]


def get_history(address: PrinterAddress, timeout_s: float = TIMEOUT_S) -> dict[str, Any]:
    data = get_json(address, "/api/history", timeout_s)
    return data if isinstance(data, dict) else {}


def dose_suggestion(history: dict[str, Any], rid: str, gel_now: float) -> Optional[tuple[float, int]]:
    """(gel dose, prints it is based on) from this printer's answers for this resin, or None.

    Each answered print says what dose was right for it: the gel dose it was sliced with times the
    correction the printer used (general x resin). An "ok" gives it directly (the last one counts);
    an "under" and an "over" without an "ok" meet in the middle (geometric mean); answers in one
    direction only are not enough to say how far."""
    rows = []
    for h in history.get("prints") or []:
        if not isinstance(h, dict) or h.get("resin_id") != rid or h.get("result") not in RESULTS:
            continue
        raw = h.get("resin")
        resin: dict[str, Any] = raw if isinstance(raw, dict) else {}
        gel = _number(resin.get("gel_dose_mj_cm3")) or gel_now
        k = (_number(h.get("factor")) or 1.0) * (_number(h.get("resin_factor")) or 1.0)
        if gel and gel > 0 and k > 0:
            rows.append((h["result"], gel * k))
    oks = [d for r, d in rows if r == "ok"]
    if oks:
        return oks[-1], len(rows)
    under = [d for r, d in rows if r == "under"]
    over = [d for r, d in rows if r == "over"]
    if under and over:
        return (under[-1] * over[-1]) ** 0.5, len(rows)
    return None


def reset_resin_factor(address: PrinterAddress, rid: str) -> dict[str, Any]:
    """The suggestion went into the resin's dose: the printer's correction for it starts again at 1."""
    return post_json(address, "/api/settings", {"resin_factor": {"resin_id": rid, "value": 1.0}})


def send_result(address: PrinterAddress, result: str) -> dict[str, Any]:
    """How the print came out (under / ok / over): the printer moves its exposure correction."""
    if result not in RESULTS:
        raise PrinterError(f"result: {' / '.join(RESULTS)}")
    return post_json(address, "/api/result", {"result": result})


def print_stage(state: dict[str, Any]) -> tuple[str, Optional[float], str]:
    """For the progress card: (stage, progress 0..1 or None, detail) of the print the slicer sent."""
    what, progress, turn, turns, left = describe_state(state)
    if what == "waiting_confirm":
        return "Ждёт подтверждения на экране принтера", None, "джойстик вправо — печатать"
    if what == "starting":
        wait = state.get("wait")
        if isinstance(wait, (list, tuple)) and len(wait) == 2:
            secs, kind = wait
            name = {"spin": "Разгон", "warm": "Прогрев проектора", "dark": "Пауза до света"}.get(str(kind), "Подготовка")
            m, s = divmod(int(float(secs) + 0.99), 60)
            return name, None, f"{m}:{s:02d}"
        return "Подготовка", None, "разгон, прогрев, пауза до света"
    if what == "printing":
        from spinslicer.i18n import tr
        m, s = divmod(int(left + 0.99), 60)
        return "Печать", progress, f"{tr('оборот')} {turn} {tr('из')} {turns} · {tr('осталось')} {m}:{s:02d}"
    return "", None, ""


def describe_state(state: dict[str, Any]) -> tuple[str, float, int, int, float]:
    """``(what, progress 0..1, turn, turns, seconds left)`` of the print the slicer sent.

    what: ``waiting_confirm`` (the printer's screen asks), ``cancelled`` / ``expired`` /
    ``replaced`` (no print came of it, or another print followed it), ``starting`` (the yes
    was given; spin-up, warm-up, dark pause), ``printing``, ``done``, ``stopped``, ``error``.

    The printer's ``state`` belongs to its latest print, which may not be ours: right after the
    yes the last print can still say ``done``. SpinPi 0.8.2+ numbers its prints (``job.seq``,
    ``request.job_seq``), so only our own print's state is taken.
    """
    request = state.get("request") or {}
    asked = request.get("state")
    job = state.get("job") or {}
    progress = float(job.get("progress") or 0.0)
    turn, turns = int(job.get("turn") or 0), int(job.get("revs") or 0)
    left = float(job.get("left_s") or 0.0)
    if asked in ("waiting_confirm", "cancelled", "expired", "replaced"):
        return asked, 0.0, 0, 0, 0.0
    if asked == "error":
        return "error", 0.0, 0, 0, 0.0
    if asked == "starting":   # the yes is in, the printer is starting the job
        return "starting", 0.0, 0, 0, 0.0
    ours = request.get("job_seq")
    if ours is not None and job.get("seq") != ours:
        if job.get("seq") is None or int(job["seq"]) < int(ours):
            return "starting", 0.0, 0, 0, 0.0
        return "replaced", 0.0, 0, 0, 0.0   # a later print took the printer
    what = {"wait": "starting", "print": "printing", "done": "done", "error": "error"}.get(
        str(state.get("state")), "stopped")
    return what, progress, turn, turns, left


# ------------------------------------------------------------------ the profile, back to the printer
# slicer setting → (label for people, where it goes on the printer)
PROFILE_FIELDS: dict[str, tuple[str, str]] = {
    "vial_diameter_mm": ("диаметр колбы", "vial_diameter_mm"),
    "vial_wall_mm": ("стенка", "vial_wall_mm"),
    "n_glass": ("n стекла", "vial_n"),
    "irradiance_mw_cm2": ("облучённость", "irradiance_mw_cm2"),
    "contrast": ("контраст", "contrast"),
    "vial_width_px": ("колба на экране", "calibration:vial_px"),
    "offset_x_px": ("сдвиг по X", "calibration:offset_px"),
}


def profile_diff(printer: dict[str, Any], slicer: dict[str, Any]) -> list[tuple[str, str, Any, Any]]:
    """What the slicer would change on the printer: ``(key, label, on the printer, in the slicer)``.

    ``printer`` is :func:`profile_settings` of the printer's profile. A value the slicer does
    not know (0 irradiance) is never written over the printer's.
    """
    out = []
    for key, (label, _target) in PROFILE_FIELDS.items():
        if key not in slicer:
            continue
        mine = slicer[key]
        if key == "irradiance_mw_cm2" and not mine:
            continue
        theirs = printer.get(key)
        if theirs is None or abs(float(theirs) - float(mine)) > 1e-6:
            out.append((key, label, theirs, mine))
    return out


def write_profile(address: PrinterAddress, values: dict[str, Any]) -> None:
    """Store slicer values on the printer: the profile (``POST /api/printer``) and the vial
    calibration (``POST /api/calibration``)."""
    profile: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    for key, value in values.items():
        target = PROFILE_FIELDS[key][1]
        if target.startswith("calibration:"):
            calibration[target.split(":", 1)[1]] = int(round(float(value)))
        else:
            profile[target] = float(value)
    if profile:
        post_json(address, "/api/printer", profile)
    if calibration:
        post_json(address, "/api/calibration", calibration)
