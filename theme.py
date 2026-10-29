"""
theme.py
========
Единая визуальная тема SpinSlicer: тёплая тёмная палитра с терракотовым
акцентом (в духе Claude), спокойные поверхности, тонкие границы и
единая шкала скруглений.

Всё, что относится к цвету, живёт здесь: QSS для виджетов Qt и цвета
для 3D-сцены (viewport.py). Менять оформление нужно только в этом файле.
"""
from __future__ import annotations

from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication

# --- Палитра ---------------------------------------------------------------
BG = "#262624"            # фон окна
SURFACE = "#2F2E2B"       # панели, карточки
SURFACE_ALT = "#363531"   # приподнятые элементы
INPUT = "#222120"         # поля ввода, лог
BORDER = "#45433E"        # тонкие границы
BORDER_SOFT = "#3A3935"   # разделители внутри панелей
TEXT = "#F0EEE6"          # основной текст
TEXT_MUTED = "#A9A69B"    # вторичный текст
TEXT_DIM = "#7B796F"      # подсказки, отключённое
ACCENT = "#D97757"        # терракота
ACCENT_HOVER = "#E48868"
ACCENT_PRESSED = "#C46646"
ACCENT_SOFT = "#3C2F29"   # фон выбранного состояния

AXIS_X = "#E0715F"
AXIS_Y = "#86B574"
AXIS_Z = "#6F9FD6"

# --- 3D-сцена --------------------------------------------------------------
SCENE_BG_TOP = "#1D1C1B"
SCENE_BG_BOTTOM = "#2D2C29"
SCENE_MODEL = "#E7DFCF"
SCENE_VAT_GLASS = "#D8D2C4"
SCENE_GRID_FILL = "#1F1E1C"
SCENE_GRID_EDGE = "#3F3D38"

RADIUS = 8
RADIUS_SM = 6

_CHECK_ICON = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdib3g9IjAgMCAxNiAxNiI+"
    "PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEzLjQ4NSA0LjM0M2wxLjQxNCAxLjQxNEw2LjQxNCAxNC4yNDMg"
    "MS4xMDEgOC45M2wxLjQxNC0xLjQxNCAzLjg5OSAzLjkgOC40ODYtOC40ODV6Ii8+PC9zdmc+"
)


def build_qss() -> str:
    return f"""
* {{
    font-family: "Segoe UI", "Inter", -apple-system, "SF Pro Text", Roboto, sans-serif;
    font-size: 13px;
    color: {TEXT};
}}
QMainWindow, QDialog, QMessageBox {{ background-color: {BG}; }}
QWidget#centralRoot {{ background-color: {BG}; }}
QToolTip {{
    background-color: {SURFACE_ALT}; color: {TEXT};
    border: 1px solid {BORDER}; padding: 5px 8px; border-radius: {RADIUS_SM}px;
}}

/* --- Текст ----------------------------------------------------------- */
QLabel {{ color: {TEXT_MUTED}; background: transparent; }}
QLabel#appTitle {{ font-size: 17px; font-weight: 600; color: {TEXT}; letter-spacing: 0.3px; }}
QLabel#panelTitle {{ font-size: 17px; font-weight: 600; color: {TEXT}; padding-bottom: 4px; }}
QLabel#hintLabel {{ color: {TEXT_DIM}; font-size: 11px; }}
QLabel#fieldLabel {{ color: {TEXT_MUTED}; font-size: 12px; }}
QLabel#transformValue {{
    color: {TEXT}; background-color: {INPUT}; border: 1px solid {BORDER_SOFT};
    border-radius: {RADIUS_SM}px; padding: 4px 8px;
    font-family: "Cascadia Mono", "Consolas", monospace;
}}
QLabel#videoPreview {{
    background-color: {INPUT}; border: 1px solid {BORDER_SOFT};
    border-radius: {RADIUS}px; color: {TEXT_DIM};
}}
QLabel a {{ color: {ACCENT}; }}

/* --- Группы-карточки -------------------------------------------------- */
QGroupBox {{
    background-color: {SURFACE}; border: 1px solid {BORDER_SOFT};
    border-radius: {RADIUS}px; margin-top: 22px; padding: 14px 12px 12px 12px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 6px; top: 0px; padding: 0 4px;
    color: {TEXT_MUTED}; font-size: 12px; font-weight: 600; letter-spacing: 0.4px;
}}

/* --- Кнопки ----------------------------------------------------------- */
QPushButton {{
    background-color: {SURFACE_ALT}; border: 1px solid {BORDER};
    border-radius: {RADIUS_SM}px; padding: 7px 16px; font-weight: 500; color: {TEXT};
}}
QPushButton:hover {{ background-color: #403F3A; border-color: #58564F; }}
QPushButton:pressed {{ background-color: {INPUT}; }}
QPushButton:focus {{ border-color: {ACCENT}; outline: none; }}
QPushButton:disabled {{ background-color: {SURFACE}; color: {TEXT_DIM}; border-color: {BORDER_SOFT}; }}

QPushButton#generateButton {{
    background-color: {ACCENT}; border: 1px solid {ACCENT_PRESSED};
    color: #FFFFFF; font-weight: 600; padding: 8px 22px;
}}
QPushButton#generateButton:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton#generateButton:pressed {{ background-color: {ACCENT_PRESSED}; }}
QPushButton#generateButton:disabled {{
    background-color: {SURFACE_ALT}; color: {TEXT_DIM}; border-color: {BORDER_SOFT};
}}

QPushButton#transformModeButton {{
    background-color: transparent; border: 1px solid transparent;
    color: {TEXT_MUTED}; padding: 6px 12px;
}}
QPushButton#transformModeButton:hover {{ background-color: {SURFACE_ALT}; color: {TEXT}; }}
QPushButton#transformModeButton:checked {{
    background-color: {ACCENT_SOFT}; border: 1px solid {ACCENT}; color: {ACCENT_HOVER};
}}
QPushButton#transformModeButton:disabled {{ color: {TEXT_DIM}; background: transparent; }}
QPushButton#nudgeButton {{ padding: 2px 8px; font-size: 15px; }}

/* --- Поля ввода ------------------------------------------------------- */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {INPUT}; border: 1px solid {BORDER_SOFT};
    border-radius: {RADIUS_SM}px; padding: 5px 9px; min-height: 18px;
    selection-background-color: {ACCENT}; selection-color: #FFFFFF;
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: {BORDER}; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {SURFACE}; border: 1px solid {BORDER}; outline: none;
    selection-background-color: {ACCENT_SOFT}; selection-color: {ACCENT_HOVER};
    padding: 4px;
}}

/* --- Слайдеры --------------------------------------------------------- */
QSlider::groove:horizontal {{ height: 4px; background: {INPUT}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {TEXT}; border: 2px solid {SURFACE}; width: 12px; height: 12px;
    margin: -7px 0; border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ background: #FFFFFF; border-color: {ACCENT}; }}

/* --- Чекбоксы --------------------------------------------------------- */
QCheckBox {{ color: {TEXT_MUTED}; spacing: 8px; background: transparent; }}
QCheckBox:hover {{ color: {TEXT}; }}
QCheckBox:disabled {{ color: {TEXT_DIM}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid {BORDER}; background-color: {INPUT};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{
    background-color: {ACCENT}; border-color: {ACCENT}; image: url({_CHECK_ICON});
}}

/* --- Скроллбары ------------------------------------------------------- */
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; min-height: 30px; border-radius: 3px; }}
QScrollBar::handle:vertical:hover {{ background: #5E5C54; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; min-width: 30px; border-radius: 3px; }}
QScrollBar::handle:horizontal:hover {{ background: #5E5C54; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* --- Лог и прогресс --------------------------------------------------- */
QPlainTextEdit#logPanel {{
    background-color: {INPUT}; border: 1px solid {BORDER_SOFT};
    border-radius: {RADIUS}px; padding: 8px; color: {TEXT_MUTED};
    font-family: "Cascadia Mono", "Consolas", "Menlo", monospace; font-size: 11px;
}}
QProgressBar {{
    background-color: {INPUT}; border: 1px solid {BORDER_SOFT};
    border-radius: 5px; height: 8px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 4px; }}
QStatusBar {{ background-color: {BG}; border-top: 1px solid {BORDER_SOFT}; color: {TEXT_MUTED}; }}
QStatusBar::item {{ border: none; }}

/* --- Панель масштаба над вьюпортом ----------------------------------- */
QFrame#scaleControls {{
    background-color: {SURFACE}; border: 1px solid {BORDER}; border-radius: {RADIUS}px;
}}
QWidget#transformToolbar {{ background: transparent; }}

/* --- Вкладки ---------------------------------------------------------- */
QTabWidget::pane {{
    border: none; background: transparent; top: 0px;
}}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: transparent; color: {TEXT_DIM}; font-weight: 500;
    padding: 9px 20px; margin-right: 2px; border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover:!selected {{ color: {TEXT_MUTED}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}

/* --- Разделитель ------------------------------------------------------ */
QSplitter::handle:horizontal {{
    background-color: #333333;
    width: 16px;
    border-left: 1px solid #1E1E1E;
    border-right: 1px solid #1E1E1E;
    image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23888888' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='15 18 9 12 15 6'></polyline></svg>");
}}
QSplitter::handle:horizontal:hover {{
    background-color: #1E90FF;
    image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='15 18 9 12 15 6'></polyline></svg>");
}}

QSplitter::handle:vertical {{
    background-color: #333333;
    height: 4px;
    margin: 0;
    border: none;
}}
QSplitter::handle:vertical:hover {{
    background-color: #1E90FF;
}}

QDoubleSpinBox {{
    background-color: #2D2D2D;
    color: #E0E0E0;
    border: 1px solid #444444;
    border-radius: 4px;
    padding: 2px 4px;
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: #3C3C3C;
    border-left: 1px solid #444444;
    width: 14px;
}}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: #4A4A4A;
}}
QDoubleSpinBox::up-arrow {{ width: 0px; height: 0px; image: none; }}
QDoubleSpinBox::down-arrow {{ width: 0px; height: 0px; image: none; }}
"""


def apply_theme(app: QApplication) -> None:
    """Применить тему ко всему приложению (стиль Fusion + палитра + QSS)."""

    app.setStyle("Fusion")
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: BG,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: INPUT,
        QPalette.ColorRole.AlternateBase: SURFACE,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: SURFACE_ALT,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.ToolTipBase: SURFACE_ALT,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: "#FFFFFF",
        QPalette.ColorRole.PlaceholderText: TEXT_DIM,
        QPalette.ColorRole.Link: ACCENT,
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(build_qss())
