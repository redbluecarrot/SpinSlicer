"""USB sticks the computer has mounted, to save a ``.spin`` straight onto one.

Windows: drives of type "removable"; macOS: volumes under ``/Volumes`` other than the system
disk; Linux: what the desktop mounted under ``/media/<user>`` or ``/run/media/<user>``.
The system calls can be handed in, so the tests run on any computer. Nothing here touches Qt.
"""
from __future__ import annotations

import os
import platform
import string
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

DRIVE_REMOVABLE = 2  # GetDriveTypeW


@dataclass(frozen=True)
class Stick:
    path: str   # where to write: "E:\\", "/Volumes/SPIN", "/media/anna/SPIN"
    label: str  # for people: "SPIN (E:)"


# ------------------------------------------------------------------ Windows
def _windows_drives() -> list[str]:
    import ctypes

    mask = ctypes.windll.kernel32.GetLogicalDrives()  # type: ignore[attr-defined]
    return [f"{letter}:\\" for i, letter in enumerate(string.ascii_uppercase) if mask >> i & 1]


def _windows_drive_type(root: str) -> int:
    import ctypes

    return int(ctypes.windll.kernel32.GetDriveTypeW(root))  # type: ignore[attr-defined]


def _windows_label(root: str) -> str:
    import ctypes

    name = ctypes.create_unicode_buffer(261)
    ok = ctypes.windll.kernel32.GetVolumeInformationW(  # type: ignore[attr-defined]
        root, name, len(name), None, None, None, None, 0)
    return name.value if ok else ""


# ------------------------------------------------------------------ Linux
def _unescape(path: str) -> str:
    """/proc/mounts writes spaces and tabs as \\040 and \\011."""
    return path.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def _linux_sticks(mounts: str) -> list[Stick]:
    out = []
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].startswith("/dev/"):
            continue
        point = _unescape(parts[1])
        if point.startswith(("/media/", "/run/media/")) and point.count("/") >= 3:
            out.append(Stick(point, os.path.basename(point)))
    return out


def _read_mounts() -> str:
    try:
        with open("/proc/mounts", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


# ------------------------------------------------------------------ macOS
def _mac_is_system(path: str) -> bool:
    return os.path.realpath(path) == "/"


def find_sticks(*, system: Optional[str] = None, mounts: Optional[str] = None,
                drives: Optional[Sequence[str]] = None,
                drive_type: Optional[Callable[[str], Optional[int]]] = None,
                volume_label: Optional[Callable[[str], Optional[str]]] = None,
                volumes: str = "/Volumes", is_system: Optional[Callable[[str], bool]] = None) -> list[Stick]:
    """USB sticks mounted right now, in a stable order. [] when there are none (or on error)."""
    system = system or platform.system()
    try:
        if system == "Windows":
            drives = _windows_drives() if drives is None else list(drives)
            drive_type = drive_type or _windows_drive_type
            volume_label = volume_label or _windows_label
            out = []
            for root in drives:
                if drive_type(root) == DRIVE_REMOVABLE:
                    letter = root.rstrip("\\")
                    label = volume_label(root) or ""
                    out.append(Stick(root, f"{label} ({letter})" if label else letter))
            return out
        if system == "Darwin":
            is_system = is_system or _mac_is_system
            names = sorted(os.listdir(volumes)) if os.path.isdir(volumes) else []
            return [Stick(os.path.join(volumes, n), n) for n in names
                    if not n.startswith(".") and not is_system(os.path.join(volumes, n))
                    and os.path.isdir(os.path.join(volumes, n))]
        return _linux_sticks(_read_mounts() if mounts is None else mounts)
    except OSError:
        return []


def eject(stick: Stick) -> bool:
    """Safely remove the stick (best effort): True when the system says it is done."""
    system = platform.system()
    if system == "Windows":
        letter = stick.path.rstrip("\\")
        cmd = ["powershell", "-NoProfile", "-Command",
               f"(New-Object -ComObject Shell.Application).Namespace(17).ParseName('{letter}').InvokeVerb('Eject')"]
    elif system == "Darwin":
        cmd = ["diskutil", "eject", stick.path]
    else:
        cmd = ["gio", "mount", "-e", stick.path]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False
