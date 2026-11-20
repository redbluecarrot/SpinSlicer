"""OpenCAL Resin Clear as the default resin; the installer's shortcut carries the taskbar ID."""
from __future__ import annotations

import os
import re
from pathlib import Path

from spinslicer.constants import DEFAULT_RESIN_ALPHA_PER_MM, DEFAULT_RESIN_N

ROOT = Path(__file__).resolve().parents[1]


def test_installer_shortcuts_carry_the_apps_taskbar_id() -> None:
    iss = (ROOT / "packaging" / "SpinSlicer.iss").read_text(encoding="utf-8")
    app = (ROOT / "spinslicer" / "app.py").read_text(encoding="utf-8")
    in_app = re.search(r'APP_USER_MODEL_ID = "([^"]+)"', app)
    in_iss = re.search(r'#define MyAppUserModelID "([^"]+)"', iss)
    assert in_app and in_iss and in_app[1] == in_iss[1]
    icons = iss.split("[Icons]", 1)[1].split("[", 1)[0]
    shortcuts = [line for line in icons.splitlines() if line.startswith("Name:")]
    assert shortcuts and all('AppUserModelID: "{#MyAppUserModelID}"' in line for line in shortcuts)


def _panel(settings):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from spinslicer.ui_panels import ProcessSettingsPanel

    QApplication.instance() or QApplication([])
    panel = ProcessSettingsPanel.__new__(ProcessSettingsPanel)
    panel._settings = settings
    return panel


class _Settings(dict):
    def value(self, key, default=None):
        return self.get(key, default)

    def setValue(self, key, value):  # noqa: N802 (QSettings API)
        self[key] = value

    def remove(self, key):
        self.pop(key, None)


def test_old_placeholders_give_way_to_the_resin_defaults_once() -> None:
    settings = _Settings({"physics/alpha": "0", "ui/n_resin": "1.5"})
    _panel(settings)._migrate_resin_defaults()
    assert "physics/alpha" not in settings and "ui/n_resin" not in settings

    settings.update({"physics/alpha": "0"})                 # set to 0 by hand later: stays
    _panel(settings)._migrate_resin_defaults()
    assert settings["physics/alpha"] == "0"


def test_hand_set_values_survive_the_migration() -> None:
    settings = _Settings({"physics/alpha": "0.05", "ui/n_resin": "1.52"})
    _panel(settings)._migrate_resin_defaults()
    assert settings["physics/alpha"] == "0.05" and settings["ui/n_resin"] == "1.52"


def test_defaults_are_opencal_resin_clear() -> None:
    assert abs(DEFAULT_RESIN_ALPHA_PER_MM - 0.020) < 1e-9
    assert abs(DEFAULT_RESIN_N - 1.48) < 1e-9
