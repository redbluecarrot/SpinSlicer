"""Keep the test runs away from the user's real settings (Windows registry / ~/.config).

QSettings(organization, application) always uses the native store, whatever the default
format is, so the class itself is replaced by one that reads and writes an ini file in a
temporary folder. This runs before the test modules import it.
"""
from __future__ import annotations

import os
import tempfile

import PyQt6.QtCore as QtCore

_SETTINGS_DIR = tempfile.mkdtemp(prefix="spinslicer-test-settings-")
_RealSettings = QtCore.QSettings


class _IsolatedSettings(_RealSettings):  # type: ignore[misc, valid-type]
    def __init__(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str):
            path = os.path.join(_SETTINGS_DIR, f"{args[0]}-{args[1]}.ini")
            super().__init__(path, _RealSettings.Format.IniFormat)
        else:
            super().__init__(*args, **kwargs)


QtCore.QSettings = _IsolatedSettings
