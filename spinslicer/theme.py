"""
theme.py
========
SpinSlicer visual theme: two palettes (night and day) that share one layout.

Night: graphite with a soft pink marker and a pink "soap" main button.
Day:   warm paper with a red-pen marker and an ink-black main button.

Everything that is a colour lives here: Qt style sheets, custom-painted widgets
(they read ``current()``) and the 3D sketch scene (``scene_palette()``).
Two fonts (SIL OFL, bundled in assets/fonts): Syne ExtraBold, the letter of the logo, for
titles and tabs, and Onest for everything else.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt6.QtWidgets import QApplication

from spinslicer.sketch_scene import ScenePalette


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
    # depth without shadows and without boxes in boxes: the panel is the floor, a group is a
    # hairline and a heading on it, only a single special thing is a card (a step lighter, with
    # a border),
    # fields are sunk into the floor, the chosen option stands up out of its field
    card: str = "#FFFFFF"
    card_border: str = "#E9E5DF"
    well: str = "#F2EFEA"
    well_border: str = "#E2DDD6"
    raised: str = "#FFFFFF"
    raised_border: str = "#D6D0C8"


DARK = Palette(
    dark=True, bg="#0E0D10", view="#09080B", surface="#19161C", surface_alt="#26222B", seg_on="#3B3441",
    input="#0F0E12", border="#403A46", border_soft="#2E2A33", text="#F6F2F4", muted="#C2BBBF", dim="#938C91",
    accent="#FF8DB5", accent_hover="#FFA6C6", accent_pressed="#E9799F", accent_soft="#3A1F2D",
    action="#FF8DB5", on_action="#3A1022", warn="#FFB86B", sketch="#D9D1D5", resin="#2A2230", liquid="#5FA8C8",
    track="#2C2630", sun="#F1ECEE",
    card="#25212A", card_border="#38323E", well="#131116", well_border="#2B2630",
    raised="#3A3340", raised_border="#4E4656",
)
LIGHT = Palette(
    dark=False, bg="#F6F4F0", view="#EDEAE4", surface="#F6F4F0", surface_alt="#E7E3DC", seg_on="#FFFFFF",
    input="#FFFFFF", border="#DDD8D1", border_soft="#E7E3DC", text="#19171A", muted="#6C686B", dim="#A39E98",
    accent="#D63A75", accent_hover="#E0558A", accent_pressed="#B92E62", accent_soft="#F8DCE6",
    action="#19171A", on_action="#F6F4F0", warn="#B86A00", sketch="#3A3537", resin="#C9C2BA", liquid="#7FA4B8",
    track="#F3CFDC", sun="#F09A1E",
)

# Legacy names still imported by older widgets.
AXIS_X = "#E0715F"
AXIS_Y = "#86B574"
AXIS_Z = "#6F9FD6"

FONT_UI = "Onest"
FONT_DISPLAY = "Syne"
FONT_MONO = FONT_UI          # old names: one text face, no separate mono or handwriting
FONT_HAND = FONT_UI
DISPLAY_WEIGHT = 800
RADIUS = 10          # controls and containers
RADIUS_SM = 6        # parts inside a control (a segment, a step button)
RADIUS_LG = 14       # the big buttons

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
        model="#E7DFCF",
    )


def _assets_dir() -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def display_font(size: int = 14) -> QFont:
    font = QFont(FONT_DISPLAY)
    font.setFamilies([FONT_DISPLAY, FONT_UI])
    font.setPixelSize(size)
    font.setWeight(QFont.Weight(DISPLAY_WEIGHT))
    return font


def hand_font(size: int = 20) -> QFont:
    font = QFont(FONT_UI)
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
    # the chosen option is a raised chip in its sunken track (no outline-only state: it read flat)
    seg_checked = (f"background-color: {p.raised}; border: none; "
                   f"color: {p.accent if p.dark else p.text}; font-weight: 700;")
    return f"""
* {{
    font-family: "{FONT_UI}", "Inter", "Segoe UI", sans-serif;
    font-size: 13px;
    color: {p.text};
}}
QMainWindow, QDialog, QMessageBox {{ background-color: {p.bg}; }}
QWidget#centralRoot, QWidget#appHeader, QWidget#chipBar {{ background-color: {p.bg}; }}
QWidget#sidePanel {{ background-color: {p.surface}; }}
QWidget#appHeader {{ border-bottom: 1px solid {p.border}; }}
QWidget#sidePanel {{ border-left: 1px solid {p.border}; }}
QWidget#settingRow {{ border: none; }}
QFrame#settingsBlock {{ background: transparent; border: none; border-top: 1px solid {p.card_border}; border-radius: 0px; }}
QFrame#statTile {{ background-color: {p.card}; border: 1px solid {p.card_border}; border-radius: {RADIUS}px; }}
QFrame#settingsBlock QFrame#statTile {{ background-color: {p.well}; border: none; }}
QWidget#stepper {{ background-color: {p.well}; border: none; border-radius: {RADIUS}px; }}
QLabel#tileKey {{ font-size: 11px; color: {p.muted}; }}
QLabel#tileValue {{ font-family: "{FONT_UI}"; font-weight: 800; font-size: 19px; color: {p.text}; }}
QLabel#tileValue[warn="true"] {{ color: {p.warn}; }}
QLabel#tileNote {{ font-size: 11px; color: {p.muted}; }}
QLabel#printerState {{ font-weight: 600; color: {p.text}; }}
QLabel#warnLine {{ font-size: 12px; color: {p.warn}; padding-bottom: 6px; }}
QLabel#smallDot {{ font-size: 11px; color: {p.dim}; }}
QPushButton#smallLink {{
    background: transparent; border: none; padding: 0; color: {p.muted}; font-size: 11px; text-align: left;
}}
QPushButton#smallLink:hover {{ color: {p.accent}; }}
QPushButton#smallLink:disabled {{ color: {p.dim}; }}
QPushButton#secondaryAction {{
    background-color: transparent; color: {p.text}; border: 1.5px solid {p.border}; border-radius: {RADIUS_LG}px;
    min-height: 44px; padding: 0 18px; font-family: "{FONT_UI}"; font-size: 15px; font-weight: 800;
}}
QPushButton#secondaryAction:hover {{ border-color: {p.accent}; color: {p.accent}; }}
QPushButton#secondaryAction:disabled {{ color: {p.dim}; border-color: {p.border_soft}; }}
QToolTip {{
    background-color: {p.surface_alt}; color: {p.text};
    border: 1px solid {p.border}; padding: 6px 9px; border-radius: {RADIUS}px;
}}

QLabel {{ color: {p.muted}; background: transparent; }}
QLabel#appTitle, QLabel#projectName {{ font-family: "{FONT_DISPLAY}", "{FONT_UI}"; font-weight: {DISPLAY_WEIGHT}; font-size: 15px; color: {p.text}; }}
QLabel#projectSub {{ font-family: "{FONT_MONO}"; font-size: 11px; color: {p.muted}; }}
QLabel#panelTitle {{ font-family: "{FONT_UI}"; font-weight: 800; font-size: 20px; color: {p.text}; }}
QLabel#panelLead {{ color: {p.muted}; }}
QLabel#todoLabel {{ color: {p.text}; font-size: 14px; padding: 10px 12px; border-radius: {RADIUS}px; background: {p.surface_alt}; border: 1px solid {p.border_soft}; }}
QLabel#emptyTitle {{ font-family: "{FONT_UI}"; font-weight: 800; font-size: 26px; color: {p.text}; }}
QLabel#planLabel {{ font-family: "{FONT_MONO}"; font-size: 12px; color: {p.text}; }}
QWidget#printStage {{ background-color: {p.view}; }}
QSplitter::handle:vertical {{ height: 6px; background: {p.border_soft}; }}
QLabel#sectionHead {{ font-size: 11px; font-weight: 700; letter-spacing: 1.2px; color: {p.muted}; padding-top: 14px; padding-bottom: 2px; }}
QLabel#stepLabel {{ font-size: 11px; font-weight: 700; letter-spacing: 1.2px; color: {p.accent if p.dark else p.muted}; padding-top: 14px; padding-bottom: 2px; }}
QLabel#progStage {{ font-size: 15px; font-weight: 700; color: {p.text}; }}
QLabel#printChecks {{ font-size: 12px; }}
QPushButton#dangerAction {{ min-height: 32px; padding: 0 16px; border-radius: {RADIUS}px; font-weight: 700;
    background-color: transparent; color: {p.warn}; border: 1px solid {p.warn}; }}
QPushButton#dangerAction:hover {{ background-color: {p.warn}; color: {p.bg}; }}
QLabel#rowLabel {{ color: {p.text}; font-weight: 600; }}
QLabel#rowValue {{ font-family: "{FONT_MONO}"; color: {p.text}; font-weight: 500; }}
QLabel#rowEffect {{ font-size: 12px; color: {p.muted}; }}
QLabel#rowEffect[warn="true"] {{ color: {p.warn}; }}
QLabel#rowHint {{ font-size: 12px; font-weight: 500; color: {p.accent}; }}
QLabel#hintLabel {{ color: {p.dim}; font-size: 11px; }}
QLabel#fieldLabel {{ color: {p.muted}; font-size: 12px; }}
QLabel#actionSub {{ font-size: 12px; color: {p.muted}; }}
QLabel#chip {{
    font-family: "{FONT_MONO}"; font-size: 11px; color: {p.text};
    border: 1px solid {p.card_border}; border-radius: {RADIUS}px; padding: 6px 10px; background: {p.card};
}}
QLabel#chip[warn="true"] {{ color: {p.warn}; border-color: {p.warn}; }}
QLabel#metricValue {{ font-size: 22px; font-weight: 700; color: {p.text}; }}
QLabel#metricKey {{ font-size: 11px; color: {p.muted}; }}
QLabel#transformValue {{
    color: {p.text}; background-color: {p.input}; border: 1px solid {p.border_soft};
    border-radius: {RADIUS}px; padding: 4px 8px; font-family: "{FONT_MONO}";
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
    border-radius: {RADIUS}px; padding: 7px 16px; font-weight: 500; color: {p.text};
}}
QPushButton:hover {{ background-color: {hover_bg}; }}
QPushButton:pressed {{ background-color: {p.input}; }}
QPushButton:disabled {{ color: {p.dim}; border-color: {p.border_soft}; }}
QPushButton#stepButton {{
    background-color: {p.raised}; border: none; border-radius: {RADIUS_SM}px;
    min-width: 26px; max-width: 26px; min-height: 26px; max-height: 26px; padding: 0; font-size: 15px;
}}
QPushButton#stepButton:hover {{ background-color: {p.accent_soft}; color: {p.accent}; }}
QPushButton#stepButton:pressed {{ background-color: {p.well}; }}
QFrame#segFrame {{ background-color: {p.well}; border: none; border-radius: {RADIUS}px; }}
QPushButton#segButton {{
    background: transparent; border: none; border-radius: {RADIUS_SM}px; padding: 8px 4px;
    font-weight: 500; font-size: 12px; color: {p.muted};
}}
QPushButton#segButton:checked {{ {seg_checked} }}
QPushButton#segButton:hover:!checked {{ color: {p.text}; background-color: {p.card}; }}
QPushButton#linkButton {{
    background: transparent; border: none; padding: 0; color: {p.accent};
    font-family: "{FONT_MONO}"; font-size: 12px;
}}
QPushButton#linkButton:hover {{ color: {p.accent_hover}; }}
QPushButton#primaryAction, QPushButton#generateButton {{
    background-color: {p.action}; color: {p.on_action}; border: none; border-radius: {RADIUS_LG}px;
    min-height: 44px; font-family: "{FONT_UI}"; font-size: 15px; font-weight: 800;
}}
QPushButton#primaryAction:hover, QPushButton#generateButton:hover {{ background-color: {p.accent_hover if p.dark else "#2A272B"}; }}
QPushButton#primaryAction:disabled {{ background-color: {p.surface_alt}; color: {p.dim}; }}
QPushButton#transformModeButton {{
    background-color: transparent; border: 1px solid transparent; color: {p.muted}; padding: 6px 12px;
}}
QPushButton#transformModeButton:checked {{ background-color: {p.accent_soft}; border: 1px solid {p.accent}; color: {p.accent}; }}
QPushButton#nudgeButton {{ padding: 2px 8px; font-size: 15px; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {p.well}; border: 1px solid {p.well_border};
    border-radius: {RADIUS}px; padding: 5px 9px; min-height: 18px;
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

QFrame#scaleControls {{ background-color: {p.card}; border: 1px solid {p.card_border}; border-radius: {RADIUS}px; }}
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
    # Grey-scale smoothing without hinting: heavy wide letters (Syne) otherwise get colour
    # fringes and stair-stepped edges that read as "pixelated".
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias)
    app.setFont(font)
    app.setStyleSheet(build_qss(p))
    return p
