"""
theme.py
========
SpinSlicer visual theme: two palettes (night and day) that share one layout.

Night: graphite with a soft pink marker and a pink "soap" main button.
Day:   warm paper with a red-pen marker and an ink-black main button.

Everything that is a colour lives here: Qt style sheets, custom-painted widgets
(they read ``current()``) and the 3D sketch scene (``scene_palette()``).
Fonts (Onest, JetBrains Mono, Caveat; SIL OFL) are bundled in assets/fonts.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt6.QtWidgets import QApplication

from sketch_scene import ScenePalette


@dataclass(frozen=True)
class Palette:
    dark: bool
    bg: str
    view: str
    surface: str
    surface_alt: str
    seg_on: str
    input: str
    border: str
    border_soft: str
    text: str
    muted: str
    dim: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_soft: str
    action: str
    on_action: str
    warn: str
    sketch: str
    resin: str
    liquid: str
    track: str
    sun: str


DARK = Palette(
    dark=True, bg="#0E0D10", view="#09080B", surface="#121115", surface_alt="#1C191F", seg_on="#302A33",
    input="#121115", border="#2A262E", border_soft="#1F1C23", text="#F1ECEE", muted="#9D969A", dim="#6E676B",
    accent="#FF8DB5", accent_hover="#FFA6C6", accent_pressed="#E9799F", accent_soft="#2A1A22",
    action="#F4A6BE", on_action="#3A1022", warn="#FFB86B", sketch="#D9D1D5", resin="#2A2230", liquid="#5FA8C8",
    track="#2C2630", sun="#F1ECEE",
)
LIGHT = Palette(
    dark=False, bg="#F6F4F0", view="#EDEAE4", surface="#F6F4F0", surface_alt="#E7E3DC", seg_on="#FFFFFF",
    input="#FFFFFF", border="#DDD8D1", border_soft="#E7E3DC", text="#19171A", muted="#6C686B", dim="#A39E98",
    accent="#E5452B", accent_hover="#EE5C43", accent_pressed="#C93A22", accent_soft="#F7DDD5",
    action="#19171A", on_action="#F6F4F0", warn="#B86A00", sketch="#3A3537", resin="#C9C2BA", liquid="#7FA4B8",
    track="#F3D3C6", sun="#F09A1E",
)

# Legacy names still imported by older widgets.
AXIS_X = "#E0715F"
AXIS_Y = "#86B574"
AXIS_Z = "#6F9FD6"

FONT_UI = "Onest"
FONT_MONO = "JetBrains Mono"
FONT_HAND = "Caveat"
RADIUS = 10
RADIUS_SM = 8

_current: Palette = DARK
_fonts_loaded = False


def current() -> Palette:
    return _current


def set_dark(dark: bool) -> Palette:
    global _current
    _current = DARK if dark else LIGHT
    return _current


def scene_palette(p: Palette | None = None) -> ScenePalette:
    p = p or _current
    return ScenePalette(
        background=p.view, sketch=p.sketch, accent=p.accent, muted=p.dim, resin=p.resin,
        liquid=p.liquid, dither_ink=p.sketch, dither_paper=p.view, warn=p.warn, hatch=not p.dark,
    )


def _assets_dir() -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "assets", "fonts")


def load_fonts() -> None:
    """Register the bundled fonts once (silently falls back to system fonts)."""
    global _fonts_loaded
    if _fonts_loaded:
        return
    _fonts_loaded = True
    folder = _assets_dir()
    if not os.path.isdir(folder):
        return
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith((".ttf", ".otf")):
            QFontDatabase.addApplicationFont(os.path.join(folder, name))


def hand_font(size: int = 20) -> QFont:
    font = QFont(FONT_HAND)
    font.setPixelSize(size)
    font.setWeight(QFont.Weight.Medium)
    return font


def mono_font(size: int = 11) -> QFont:
    font = QFont(FONT_MONO)
    font.setPixelSize(size)
    return font


def build_qss(p: Palette | None = None) -> str:
    p = p or _current
    hover_bg = p.seg_on if p.dark else "#EFEBE5"
    return f"""
* {{
    font-family: "{FONT_UI}", "Inter", "Segoe UI", sans-serif;
    font-size: 13px;
    color: {p.text};
}}
QMainWindow, QDialog, QMessageBox {{ background-color: {p.bg}; }}
QWidget#centralRoot, QWidget#appHeader, QWidget#sidePanel, QWidget#chipBar {{ background-color: {p.bg}; }}
QWidget#appHeader {{ border-bottom: 1px solid {p.border}; }}
QWidget#sidePanel {{ border-left: 1px solid {p.border}; }}
QWidget#settingRow {{ border-bottom: 1px solid {p.border_soft}; }}
QToolTip {{
    background-color: {p.surface_alt}; color: {p.text};
    border: 1px solid {p.border}; padding: 6px 9px; border-radius: {RADIUS_SM}px;
}}

QLabel {{ color: {p.muted}; background: transparent; }}
QLabel#appTitle, QLabel#projectName {{ font-size: 15px; font-weight: 600; color: {p.text}; }}
QLabel#projectSub {{ font-family: "{FONT_MONO}"; font-size: 11px; color: {p.muted}; }}
QLabel#panelTitle {{ font-size: 19px; font-weight: 700; color: {p.text}; }}
QLabel#panelLead {{ color: {p.muted}; }}
QLabel#todoLabel {{ color: {p.text}; font-size: 14px; padding: 10px 12px; border-radius: 10px; background: {p.surface_alt}; }}
QLabel#emptyTitle {{ font-size: 26px; font-weight: 700; color: {p.text}; }}
QLabel#planLabel {{ font-family: "{FONT_MONO}"; font-size: 12px; color: {p.text}; }}
QWidget#printStage {{ background-color: {p.view}; }}
QSplitter::handle:vertical {{ height: 6px; background: {p.border_soft}; }}
QLabel#sectionHead {{ font-family: "{FONT_MONO}"; font-size: 11px; letter-spacing: 1px; color: {p.muted}; padding-top: 8px; }}
QLabel#rowLabel {{ color: {p.text}; font-weight: 600; }}
QLabel#rowValue {{ font-family: "{FONT_MONO}"; color: {p.text}; font-weight: 500; }}
QLabel#rowEffect {{ font-family: "{FONT_MONO}"; font-size: 11px; color: {p.muted}; }}
QLabel#rowEffect[warn="true"] {{ color: {p.warn}; }}
QLabel#rowHint {{ font-family: "{FONT_HAND}"; font-size: 21px; color: {p.accent}; }}
QLabel#hintLabel {{ color: {p.dim}; font-size: 11px; }}
QLabel#fieldLabel {{ color: {p.muted}; font-size: 12px; }}
QLabel#actionSub {{ font-family: "{FONT_MONO}"; font-size: 11px; color: {p.muted}; }}
QLabel#chip {{
    font-family: "{FONT_MONO}"; font-size: 11px; color: {p.text};
    border: 1px solid {p.border}; border-radius: 9px; padding: 6px 10px; background: {p.bg};
}}
QLabel#chip[warn="true"] {{ color: {p.warn}; border-color: {p.warn}; }}
QLabel#metricValue {{ font-size: 22px; font-weight: 700; color: {p.text}; }}
QLabel#metricKey {{ font-size: 11px; color: {p.muted}; }}
QLabel#transformValue {{
    color: {p.text}; background-color: {p.input}; border: 1px solid {p.border_soft};
    border-radius: {RADIUS_SM}px; padding: 4px 8px; font-family: "{FONT_MONO}";
}}
QLabel#videoPreview {{
    background-color: {p.view}; border: 1px solid {p.border_soft};
    border-radius: {RADIUS}px; color: {p.dim};
}}
QLabel a {{ color: {p.accent}; }}

QGroupBox {{
    background-color: {p.surface}; border: 1px solid {p.border_soft};
    border-radius: {RADIUS}px; margin-top: 22px; padding: 14px 12px 12px 12px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left; left: 6px; top: 0px; padding: 0 4px;
    color: {p.muted}; font-size: 12px; font-weight: 600;
}}

QPushButton {{
    background-color: {p.surface_alt}; border: 1px solid {p.border};
    border-radius: {RADIUS_SM}px; padding: 7px 16px; font-weight: 500; color: {p.text};
}}
QPushButton:hover {{ background-color: {hover_bg}; }}
QPushButton:pressed {{ background-color: {p.input}; }}
QPushButton:disabled {{ color: {p.dim}; border-color: {p.border_soft}; }}
QPushButton#stepButton {{
    background-color: {p.surface_alt}; border: none; border-radius: 8px;
    min-width: 28px; max-width: 28px; min-height: 28px; max-height: 28px; padding: 0; font-size: 16px;
}}
QPushButton#stepButton:hover {{ background-color: {p.seg_on}; }}
QFrame#segFrame {{ background-color: {p.surface_alt}; border-radius: 10px; }}
QPushButton#segButton {{
    background: transparent; border: none; border-radius: 7px; padding: 6px 4px;
    font-weight: 600; font-size: 12px; color: {p.text};
}}
QPushButton#segButton:checked {{ background-color: {p.seg_on}; }}
QPushButton#segButton:hover:!checked {{ color: {p.accent}; }}
QPushButton#linkButton {{
    background: transparent; border: none; padding: 0; color: {p.accent};
    font-family: "{FONT_MONO}"; font-size: 12px;
}}
QPushButton#linkButton:hover {{ color: {p.accent_hover}; }}
QPushButton#primaryAction, QPushButton#generateButton {{
    background-color: {p.action}; color: {p.on_action}; border: none; border-radius: 18px;
    min-height: 54px; font-size: 16px; font-weight: 700;
}}
QPushButton#primaryAction:hover, QPushButton#generateButton:hover {{ background-color: {p.accent_hover if p.dark else "#2A272B"}; }}
QPushButton#primaryAction:disabled {{ background-color: {p.surface_alt}; color: {p.dim}; }}
QPushButton#transformModeButton {{
    background-color: transparent; border: 1px solid transparent; color: {p.muted}; padding: 6px 12px;
}}
QPushButton#transformModeButton:checked {{ background-color: {p.accent_soft}; border: 1px solid {p.accent}; color: {p.accent}; }}
QPushButton#nudgeButton {{ padding: 2px 8px; font-size: 15px; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {p.input}; border: 1px solid {p.border};
    border-radius: {RADIUS_SM}px; padding: 5px 9px; min-height: 18px;
    selection-background-color: {p.accent}; selection-color: {p.on_action};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {p.accent}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {p.surface}; border: 1px solid {p.border}; outline: none;
    selection-background-color: {p.accent_soft}; selection-color: {p.accent}; padding: 4px;
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0px; border: none; }}

QSlider::groove:horizontal {{ height: 4px; background: {p.border}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {p.text}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {p.text}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}

QCheckBox {{ color: {p.text}; spacing: 8px; background: transparent; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {p.border}; background-color: {p.input}; }}
QCheckBox::indicator:checked {{ background-color: {p.accent}; border-color: {p.accent}; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p.border}; min-height: 30px; border-radius: 3px; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {p.border}; min-width: 30px; border-radius: 3px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QPlainTextEdit#logPanel {{
    background-color: {p.view}; border: 1px solid {p.border_soft}; border-radius: {RADIUS}px;
    padding: 8px; color: {p.muted}; font-family: "{FONT_MONO}"; font-size: 11px;
}}
QProgressBar {{
    background-color: {p.border}; border: none; border-radius: 2px;
    max-height: 4px; min-height: 4px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background-color: {p.accent}; border-radius: 2px; }}
QStatusBar {{ background-color: {p.bg}; border-top: 1px solid {p.border_soft}; color: {p.muted}; }}
QStatusBar QLabel {{ font-family: "{FONT_MONO}"; font-size: 11px; }}
QStatusBar::item {{ border: none; }}

QFrame#scaleControls {{ background-color: {p.surface}; border: 1px solid {p.border}; border-radius: {RADIUS}px; }}
QWidget#transformToolbar {{ background: transparent; }}
QTabWidget::pane {{ border: none; background: transparent; top: 0px; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: transparent; color: {p.dim}; padding: 9px 20px; border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {p.text}; border-bottom: 2px solid {p.accent}; }}
QSplitter::handle {{ background-color: {p.border_soft}; }}
QSplitter::handle:horizontal:hover, QSplitter::handle:vertical:hover {{ background-color: {p.accent}; }}
"""


def apply_theme(app: QApplication, dark: bool | None = None) -> Palette:
    """Apply a palette to the whole application (Fusion + QPalette + QSS)."""
    load_fonts()
    p = set_dark(dark) if dark is not None else _current
    app.setStyle("Fusion")
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: p.bg, QPalette.ColorRole.WindowText: p.text,
        QPalette.ColorRole.Base: p.input, QPalette.ColorRole.AlternateBase: p.surface,
        QPalette.ColorRole.Text: p.text, QPalette.ColorRole.Button: p.surface_alt,
        QPalette.ColorRole.ButtonText: p.text, QPalette.ColorRole.ToolTipBase: p.surface_alt,
        QPalette.ColorRole.ToolTipText: p.text, QPalette.ColorRole.Highlight: p.accent,
        QPalette.ColorRole.HighlightedText: p.on_action, QPalette.ColorRole.PlaceholderText: p.dim,
        QPalette.ColorRole.Link: p.accent,
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    font = QFont(FONT_UI)
    font.setPixelSize(13)
    app.setFont(font)
    app.setStyleSheet(build_qss(p))
    return p
