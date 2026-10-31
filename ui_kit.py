"""
ui_kit.py
=========
Small custom widgets of the SpinSlicer interface.

  NavButton      — header tab; the active one gets a hand-drawn marker underline
                   drawn frame by frame.
  SunMoonToggle  — day/night switch: the sun rolls right and turns into a moon.
  Switch         — on/off toggle.
  Stepper        — "−  value  +" with typing; same API as the old LabeledSlider.
  Segmented      — a row of mutually exclusive options.
  SettingRow     — label + control + "what it changes" line + marker hint.
  Chip           — one value in the readout strip under the scene.

Hints (the marker notes for newcomers) are shown or hidden all at once with
``set_hints_visible``. Everything paints with ``theme.current()`` and only
animates transforms/colours, so it stays smooth on weak machines.
"""
from __future__ import annotations

import math
import weakref
from typing import Any, Callable, Optional, Sequence

from PyQt6.QtCore import QEasingCurve, QPointF, QRectF, QSize, Qt, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QButtonGroup,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import theme
from i18n import tr

# ---------------------------------------------------------------------------
# hints
# ---------------------------------------------------------------------------
_hint_labels: "weakref.WeakSet[HintLabel]" = weakref.WeakSet()
_hints_on = True


def hints_visible() -> bool:
    return _hints_on


def set_hints_visible(on: bool) -> None:
    global _hints_on
    _hints_on = bool(on)
    for label in list(_hint_labels):
        label.refresh_visibility()


class HintLabel(QLabel):
    """Marker note for newcomers (Caveat font, accent colour). Never animated."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(tr(text), parent)
        self.setObjectName("rowHint")
        self.setWordWrap(True)
        _hint_labels.add(self)
        self.refresh_visibility()

    def setText(self, text: str | None) -> None:  # noqa: N802 (Qt API name)
        super().setText(tr(text or ""))
        self.refresh_visibility()

    def refresh_visibility(self) -> None:
        self.setVisible(_hints_on and bool(self.text()))


def _mix(a: str, b: str, t: float) -> QColor:
    ca, cb = QColor(a), QColor(b)
    return QColor(
        round(ca.red() + (cb.red() - ca.red()) * t),
        round(ca.green() + (cb.green() - ca.green()) * t),
        round(ca.blue() + (cb.blue() - ca.blue()) * t),
    )


def _animation(owner: QWidget, duration: int, on_value: Callable[[float], None],
               curve: QEasingCurve.Type = QEasingCurve.Type.InOutCubic) -> QVariantAnimation:
    anim = QVariantAnimation(owner)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(curve)
    anim.valueChanged.connect(lambda v: on_value(float(v)))
    return anim


def marker_path(x0: float, x1: float, y: float, seed: int = 0) -> QPainterPath:
    """A slightly uneven marker stroke from x0 to x1."""
    w = x1 - x0
    wob = 1.6 + (seed % 3) * 0.4
    path = QPainterPath(QPointF(x0, y + 1.0))
    path.cubicTo(QPointF(x0 + w * 0.25, y - wob), QPointF(x0 + w * 0.6, y - wob * 0.6), QPointF(x0 + w * 0.8, y))
    path.cubicTo(QPointF(x0 + w * 0.9, y + wob * 0.5), QPointF(x0 + w * 0.97, y + wob * 0.6), QPointF(x1, y - 1.0))
    return path


def draw_partial(painter: QPainter, path: QPainterPath, fraction: float, steps: int = 40) -> None:
    n = max(2, int(steps * fraction))
    pts = [path.pointAtPercent(min(1.0, fraction * i / (n - 1))) for i in range(n)]
    for a, b in zip(pts, pts[1:]):
        painter.drawLine(a, b)


# ---------------------------------------------------------------------------
# header tab
# ---------------------------------------------------------------------------
class NavButton(QAbstractButton):
    """Header tab: title, optional mono subtitle or step badge, marker underline."""

    def __init__(self, title: str, sub: str = "", badge: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # Sources; translated when painted, so a language switch shows at once.
        self._title, self._sub, self._badge, self._done = title, sub, badge, False
        self._draw = 1.0
        self._hover = False
        self._anim = _animation(self, 420, self._on_draw, QEasingCurve.Type.Linear)
        self.toggled.connect(self._on_toggled)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_title(self, title: str) -> None:
        self._title = title
        self.updateGeometry()
        self.update()

    def set_sub(self, sub: str) -> None:
        self._sub = sub
        self.updateGeometry()
        self.update()

    def set_badge(self, badge: str, done: bool = False) -> None:
        self._badge, self._done = badge, done
        self.update()

    def _fonts(self) -> tuple[QFont, QFont]:
        title = QFont(theme.FONT_UI)
        title.setPixelSize(14)
        title.setWeight(QFont.Weight.DemiBold)
        return title, theme.mono_font(10)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt API name)
        title, sub = self._fonts()
        from PyQt6.QtGui import QFontMetrics
        w = QFontMetrics(title).horizontalAdvance(tr(self._title))
        if self._sub:
            w = max(w, QFontMetrics(sub).horizontalAdvance(self._sub))
        if self._badge:
            w += 28
        return QSize(w + 24, 46)

    def enterEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = True
        self.update()

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = False
        self.update()

    def _on_toggled(self, checked: bool) -> None:
        if checked:
            self._anim.stop()
            self._draw = 0.0
            self._anim.start()
        self.update()

    def _on_draw(self, value: float) -> None:
        self._draw = math.floor(value * 9) / 9.0   # frame-by-frame, like hand animation
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        p = theme.current()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        alpha = 1.0 if self.isChecked() else (0.85 if self._hover else 0.6)
        painter.setOpacity(alpha)
        title, sub = self._fonts()
        x = 12.0
        if self._badge:
            cy = self.height() / 2 - 2
            rect = QRectF(x, cy - 10, 20, 20)
            if self._done:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(p.text))
                painter.drawEllipse(rect)
                painter.setPen(QPen(QColor(p.bg), 1.8, cap=Qt.PenCapStyle.RoundCap))
                painter.drawLine(QPointF(x + 5.5, cy + 0.5), QPointF(x + 8.8, cy + 3.8))
                painter.drawLine(QPointF(x + 8.8, cy + 3.8), QPointF(x + 14.5, cy - 3.5))
            else:
                painter.setPen(QPen(QColor(p.muted), 1.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(rect)
                painter.setFont(theme.mono_font(11))
                painter.setPen(QColor(p.text))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._badge)
            x += 28
        painter.setPen(QColor(p.text))
        painter.setFont(title)
        if self._sub:
            painter.drawText(QRectF(x, 5, self.width(), 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, tr(self._title))
            painter.setFont(sub)
            painter.setPen(QColor(p.muted))
            painter.drawText(QRectF(x, 23, self.width(), 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, tr(self._sub))
        else:
            painter.drawText(QRectF(x, 0, self.width(), self.height() - 4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, tr(self._title))
        if self.isChecked() and self._draw > 0:
            painter.setOpacity(1.0)
            painter.setPen(QPen(QColor(p.accent), 2.6, cap=Qt.PenCapStyle.RoundCap))
            path = marker_path(8.0, self.width() - 8.0, self.height() - 4.5, len(tr(self._title)))
            draw_partial(painter, path, self._draw)
        painter.end()


# ---------------------------------------------------------------------------
# toggles
# ---------------------------------------------------------------------------
class SunMoonToggle(QAbstractButton):
    """Checked = night. The knob rolls (rotates) and the sun gets a bite → moon."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(56, 28)
        self.setToolTip(tr("День / ночь"))
        self._t = 0.0
        self._anim = _animation(self, 650, self._on_anim)
        self.toggled.connect(self._on_toggled)

    def set_dark_silent(self, dark: bool) -> None:
        self.blockSignals(True)
        self.setChecked(dark)
        self.blockSignals(False)
        self._t = 1.0 if dark else 0.0
        self.update()

    def _on_toggled(self, checked: bool) -> None:
        start = self._t
        self._anim.stop()
        self._anim.setStartValue(start)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def _on_anim(self, value: float) -> None:
        self._t = value
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        t = self._t
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_mix(theme.LIGHT.track, theme.DARK.track, t))
        painter.drawRoundedRect(QRectF(0, 0, 56, 28), 14, 14)
        painter.setOpacity(t)
        painter.setBrush(QColor("#FFFFFF"))
        for sx, sy, r in ((14, 9, 1.1), (22, 19, 0.9), (30, 11, 0.7)):
            painter.drawEllipse(QPointF(sx, sy), r, r)
        painter.setOpacity(1.0)
        cx, cy = 14.0 + 28.0 * t, 14.0
        painter.translate(cx, cy)
        painter.rotate(360.0 * t)
        disk = QPainterPath()
        disk.addEllipse(QPointF(0, 0), 7.5, 7.5)
        bx = 14.0 + (-8.0 - 14.0) * t
        by = -14.0 + (6.0 + 14.0) * t
        bite = QPainterPath()
        bite.addEllipse(QPointF((12.0 + bx) * 0.5, (-10.0 + by) * 0.5), 6.5, 6.5)
        painter.setBrush(_mix(theme.LIGHT.sun, theme.DARK.sun, t))
        painter.drawPath(disk.subtracted(bite))
        painter.end()


class Switch(QAbstractButton):
    def __init__(self, checked: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(40, 24)
        self._t = 1.0 if checked else 0.0
        self._anim = _animation(self, 220, self._on_anim, QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._t)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def _on_anim(self, value: float) -> None:
        self._t = value
        self.update()

    def setChecked(self, checked: bool) -> None:  # noqa: N802 (Qt API name)
        super().setChecked(checked)
        if not hasattr(self, "_anim") or not self.isVisible() or self.signalsBlocked():
            if hasattr(self, "_anim"):
                self._anim.stop()
            self._t = 1.0 if checked else 0.0
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        if self._anim.state() != QVariantAnimation.State.Running:
            # Always draw the real state, even after a change made with signals blocked.
            self._t = 1.0 if self.isChecked() else 0.0
        p = theme.current()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        off = p.surface_alt if p.dark else "#D9D4CC"
        if not self.isEnabled():
            painter.setOpacity(0.45)
            painter.setBrush(QColor(off))
        else:
            painter.setBrush(_mix(off, p.accent, self._t))
        painter.drawRoundedRect(QRectF(0, 0, 40, 24), 12, 12)
        painter.setBrush(_mix("#FFFFFF" if not p.dark else p.text, p.bg, self._t))
        painter.drawEllipse(QRectF(3 + 16 * self._t, 3, 18, 18))
        painter.end()


# ---------------------------------------------------------------------------
# value controls
# ---------------------------------------------------------------------------
class _ValueBox(QDoubleSpinBox):
    """The typed value between − and +: looks like text, wheel only when focused."""

    def wheelEvent(self, event: Any) -> None:  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

    def textFromValue(self, value: float) -> str:  # noqa: N802
        from i18n import language
        text = f"{value:.{self.decimals()}f}"
        return text.replace(".", ",") if language() == "ru" else text

    def valueFromText(self, text: str | None) -> float:  # noqa: N802
        cleaned = (text or "").replace(self.suffix(), "").replace(",", ".").strip()
        try:
            return float(cleaned)
        except ValueError:
            return self.value()

    def validate(self, text: str | None, pos: int) -> tuple[Any, str, int]:
        from PyQt6.QtGui import QValidator
        return QValidator.State.Acceptable, text or "", pos


class Stepper(QWidget):
    """− value + . API compatible with widgets.LabeledSlider (value/setValue/valueChanged)."""

    valueChanged = pyqtSignal(float)

    def __init__(self, minimum: float, maximum: float, value: float, step: float = 1.0,
                 decimals: int = 0, suffix: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._minus = QPushButton("−")
        self._plus = QPushButton("+")
        for b in (self._minus, self._plus):
            b.setObjectName("stepButton")
            b.setAutoRepeat(True)
            b.setAutoRepeatDelay(380)
            b.setAutoRepeatInterval(70)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._box = _ValueBox()
        self._box.setObjectName("rowValue")
        self._box.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._box.setFrame(False)
        self._box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._box.setDecimals(decimals)
        self._box.setRange(minimum, maximum)
        self._box.setSingleStep(step)
        self._box.setSuffix(f" {tr(suffix)}" if suffix else "")
        self._box.setKeyboardTracking(False)
        self._box.setMinimumWidth(88)
        self._box.setStyleSheet("background: transparent; border: none; padding: 0;")
        self._box.setValue(value)
        self._box.valueChanged.connect(self.valueChanged.emit)
        self._minus.clicked.connect(lambda: self._box.stepBy(-1))
        self._plus.clicked.connect(lambda: self._box.stepBy(1))
        layout.addWidget(self._minus)
        layout.addWidget(self._box)
        layout.addWidget(self._plus)

    def value(self) -> float:
        return float(self._box.value())

    def value_int(self) -> int:
        return int(round(self._box.value()))

    def setValue(self, v: float) -> None:  # noqa: N802
        self._box.setValue(float(v))

    def setRange(self, minimum: float, maximum: float) -> None:  # noqa: N802
        self._box.setRange(minimum, maximum)


class Segmented(QFrame):
    """A row of exclusive options. value()/value_int() return the option's data."""

    changed = pyqtSignal(int)
    valueChanged = pyqtSignal(float)

    def __init__(self, options: Sequence[tuple[str, Any]], current: int = 0,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("segFrame")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._data = [d for _, d in options]
        for i, (label, _data) in enumerate(options):
            b = QPushButton(tr(label))
            b.setObjectName("segButton")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self._group.addButton(b, i)
            layout.addWidget(b, 1)
        btn = self._group.button(current)
        if btn is not None:
            btn.setChecked(True)
        self._group.idClicked.connect(self._on_clicked)

    def _on_clicked(self, index: int) -> None:
        self.changed.emit(index)
        data = self._data[index]
        if isinstance(data, (int, float)):
            self.valueChanged.emit(float(data))

    def currentIndex(self) -> int:  # noqa: N802
        return max(0, self._group.checkedId())

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        b = self._group.button(int(index))
        if b is not None and not b.isChecked():
            b.setChecked(True)
            self._on_clicked(int(index))   # programmatic changes notify like clicks

    def currentData(self) -> Any:  # noqa: N802
        return self._data[self.currentIndex()]

    def value(self) -> float:
        return float(self.currentData())

    def value_int(self) -> int:
        return int(self.currentData())

    def setValue(self, v: float) -> None:  # noqa: N802
        numeric = [(i, d) for i, d in enumerate(self._data) if isinstance(d, (int, float))]
        if numeric:
            self.setCurrentIndex(min(numeric, key=lambda item: abs(float(item[1]) - float(v)))[0])


# ---------------------------------------------------------------------------
# layout pieces
# ---------------------------------------------------------------------------
class SettingRow(QWidget):
    """label · control, then a mono line saying what the setting changes, then a hint."""

    def __init__(self, label: str, control: Optional[QWidget] = None, effect: str = "",
                 hint: str = "", stacked: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.control = control
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 11, 0, 11)
        outer.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(12)
        self.label = QLabel(tr(label))
        self.label.setObjectName("rowLabel")
        top.addWidget(self.label, 1)
        if control is not None and not stacked:
            top.addWidget(control, 0, Qt.AlignmentFlag.AlignRight)
        outer.addLayout(top)
        if control is not None and stacked:
            outer.addWidget(control)
        self.effect = QLabel(tr(effect))
        self.effect.setObjectName("rowEffect")
        self.effect.setWordWrap(True)
        self.effect.setVisible(bool(effect))
        outer.addWidget(self.effect)
        self.hint = HintLabel(hint)
        outer.addWidget(self.hint)

    def set_effect(self, text: str, warn: bool = False) -> None:
        text = tr(text)
        self.effect.setText(text)
        self.effect.setVisible(bool(text))
        if self.effect.property("warn") != warn:
            self.effect.setProperty("warn", warn)
            style = self.effect.style()
            if style is not None:
                style.unpolish(self.effect)
                style.polish(self.effect)

    def set_hint(self, text: str) -> None:
        self.hint.setText(text)


def section_head(text: str) -> QLabel:
    label = QLabel(tr(text).upper())
    label.setObjectName("sectionHead")
    label.setProperty("_i18n_upper", text)        # re-rendered in capitals on a language switch
    return label


def divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {theme.current().border_soft}; border: none;")
    line.setProperty("themeDivider", True)
    return line


class Chip(QLabel):
    """'key value' in the readout strip; warn turns it amber."""

    def __init__(self, key: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("chip")
        self._key, self._value, self._warn = key, "—", False      # sources; translated when drawn
        self.refresh()

    def set_value(self, value: str, warn: bool = False) -> None:
        self._value, self._warn = value, warn
        self.refresh()

    def refresh(self) -> None:
        p = theme.current()
        self.setText(f"<span style='color:{p.muted}'>{tr(self._key)}</span>&nbsp;&nbsp;{tr(self._value)}")
        if self.property("warn") != self._warn:
            self.setProperty("warn", self._warn)
            style = self.style()
            if style is not None:
                style.unpolish(self)
                style.polish(self)


class Metric(QWidget):
    def __init__(self, key: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        self.key = QLabel(tr(key))
        self.key.setObjectName("metricKey")
        self.value = QLabel("—")
        self.value.setObjectName("metricValue")
        self.note = HintLabel("")
        layout.addWidget(self.key)
        layout.addWidget(self.value)
        layout.addWidget(self.note)
        layout.addStretch(1)

    def set(self, value: str, note: str = "") -> None:
        self.value.setText(tr(value))
        self.note.setText(note)


class EmptyScene(QWidget):
    """What the scene shows before a model is loaded: a drawn vial and a drop target."""

    openRequested = pyqtSignal()
    sampleRequested = pyqtSignal()
    fileDropped = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(200, 200)
        self._hover = False
        self._frame = 0
        from PyQt6.QtCore import QTimer
        self._timer = QTimer(self)
        self._timer.setInterval(160)
        self._timer.timeout.connect(self._tick)
        layout = QVBoxLayout(self)
        layout.addStretch(5)
        self.title = QLabel(tr("Перетащите STL сюда"))
        self.title.setObjectName("emptyTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub = QLabel(tr("файл детали в формате STL, в миллиметрах"))
        self.sub.setObjectName("rowEffect")
        self.sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.open_btn = QPushButton(tr("Выбрать файл"))
        self.open_btn.setObjectName("primaryAction")
        self.open_btn.setFixedWidth(220)
        self.open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_btn.clicked.connect(self.openRequested.emit)
        buttons.addWidget(self.open_btn)
        buttons.addStretch(1)
        layout.addWidget(self.title)
        layout.addWidget(self.sub)
        layout.addSpacing(14)
        layout.addLayout(buttons)
        layout.addSpacing(6)
        layout.addStretch(2)

    def showEvent(self, event: Any) -> None:  # noqa: N802
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event: Any) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % 3
        self.update()

    @staticmethod
    def _stl_path(event: Any) -> str:
        data = event.mimeData()
        if data is not None and data.hasUrls():
            for url in data.urls():
                path = url.toLocalFile()
                if path.lower().endswith(".stl"):
                    return path
        return ""

    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802
        if self._stl_path(event):
            self._hover = True
            event.acceptProposedAction()
            self.update()

    def dragLeaveEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = False
        self.update()

    def dropEvent(self, event: Any) -> None:  # noqa: N802
        path = self._stl_path(event)
        self._hover = False
        self.update()
        if path:
            event.acceptProposedAction()
            self.fileDropped.emit(path)

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        import random
        p = theme.current()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(p.view))
        cx = self.width() / 2
        # The drawing lives in the space above the title and shrinks (or hides) to fit.
        avail = self.title.geometry().top() - 24.0
        h = min(avail * 0.62, 230.0)
        if h < 56:
            painter.end()
            return
        top = max(12.0, (avail - h * 1.2) / 2)
        r = h * 0.36
        rng = random.Random(7 + self._frame * 31)

        def j() -> float:
            return rng.uniform(-1.1, 1.1)
        pen = QPen(QColor(p.sketch), 1.8, cap=Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # Vial: rim ellipse, walls, bottom arc — slightly different every frame.
        painter.drawEllipse(QPointF(cx + j(), top + j()), r + j(), r * 0.22 + j() * 0.3)
        painter.drawLine(QPointF(cx - r + j(), top + j()), QPointF(cx - r + j(), top + h + j()))
        painter.drawLine(QPointF(cx + r + j(), top + j()), QPointF(cx + r + j(), top + h + j()))
        arc = QPainterPath()
        arc.moveTo(cx - r + j(), top + h)
        arc.cubicTo(QPointF(cx - r * 0.6, top + h + r * 0.34 + j()), QPointF(cx + r * 0.6, top + h + r * 0.34 + j()),
                    QPointF(cx + r + j(), top + h))
        painter.drawPath(arc)
        # A dashed "where your part goes" box inside.
        dash = QPen(QColor(p.accent if self._hover else p.dim), 1.6, Qt.PenStyle.DashLine)
        painter.setPen(dash)
        painter.drawRoundedRect(QRectF(cx - r * 0.55, top + h * 0.38, r * 1.1, h * 0.38), 6, 6)
        # Arrow pointing into the vial.
        painter.setPen(QPen(QColor(p.accent), 2.2, cap=Qt.PenCapStyle.RoundCap))
        ax = cx + r * 1.9
        ay = top + h * 0.2
        path = QPainterPath(QPointF(ax + 30 + j(), ay - 26))
        path.cubicTo(QPointF(ax + 10, ay - 30 + j()), QPointF(ax - 8, ay - 10), QPointF(cx + r * 0.7, top + h * 0.48))
        painter.drawPath(path)
        tip = QPointF(cx + r * 0.7, top + h * 0.48)
        painter.drawLine(tip, tip + QPointF(12, -3))
        painter.drawLine(tip, tip + QPointF(6, -11))
        if self._hover:
            painter.setPen(QPen(QColor(p.accent), 2.0, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(14, 14, -14, -14), 16, 16)
        painter.end()
