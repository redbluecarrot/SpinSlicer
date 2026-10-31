"""Keep PyQt6 paired with the Qt DLLs from the same installation."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_dll_directory_handles: list[object] = []


def _qt_bin_candidates() -> list[Path]:
    candidates: list[Path] = []

    bundle_root = getattr(sys, "_MEIPASS", None)
    if isinstance(bundle_root, str):
        candidates.append(Path(bundle_root) / "PyQt6" / "Qt6" / "bin")

    try:
        spec = importlib.util.find_spec("PyQt6")
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec is not None:
        locations = spec.submodule_search_locations or []
        candidates.extend(Path(location) / "Qt6" / "bin" for location in locations)
        if spec.origin:
            candidates.append(Path(spec.origin).resolve().parent / "Qt6" / "bin")

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = os.path.normcase(os.path.abspath(str(candidate)))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(candidate)
    return unique


def configure_qt_dll_search_path() -> None:
    """Prepend the matching PyQt6 Qt directory before importing Qt modules."""

    if not sys.platform.startswith("win"):
        return

    add_dll_directory = getattr(os, "add_dll_directory", None)
    for directory in _qt_bin_candidates():
        if not directory.is_dir():
            continue
        directory_text = str(directory)
        if callable(add_dll_directory):
            _dll_directory_handles.append(add_dll_directory(directory_text))
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if directory_text not in path_entries:
            os.environ["PATH"] = directory_text + os.pathsep + os.environ.get("PATH", "")
        break
