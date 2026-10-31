"""
video_tab.py
============
The "Print" screen: the projector's whole screen as it will really look.

Frames are shown placed 1:1 on the projector canvas (projector_canvas.py), at
the real frame rate of the print. From here they go to a second screen (the
projector) full screen, to an H.264 video of the projector's native size, or
stay as PNG files. A vial-alignment pattern and a gray ramp help set the
projector up.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QKeyEvent, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import theme
from i18n import tr
from job_controller import JobController
from projector_canvas import CanvasGeometry, compose
from workers import VideoAssembleWorker, VideoExportWorker, calibration_ramp

BASE_FPS = 24.0


def _qimage(frame: np.ndarray) -> QImage:
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    h, w = frame.shape[:2]
    return QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8).copy()


def alignment_pattern(geometry: CanvasGeometry, vial_px: int, height_px: int) -> np.ndarray:
    """White marks where the vial must be: its two walls, the axis and the frame height."""
    canvas = np.zeros((geometry.height, geometry.width), dtype=np.uint8)
    x, y = geometry.placement(vial_px, height_px)
    cx = x + vial_px // 2
    for col, value in ((x, 255), (x + vial_px - 1, 255), (cx, 140)):
        if 0 <= col < geometry.width:
            canvas[max(y, 0):min(y + height_px, geometry.height), col] = value
    for row in (y, y + height_px - 1):
        if 0 <= row < geometry.height:
            canvas[row, max(x, 0):min(x + vial_px, geometry.width)] = 255
    return canvas


class ScreenPreview(QWidget):
    """The projector screen in miniature: black canvas, the frame where it really lands."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(240, 140)
        self._image: Optional[QImage] = None
        self._geometry = CanvasGeometry()
        self._true_size: Optional[tuple[int, int]] = None
        self._message = tr("Кадры появятся после расчёта света")

    def set_true_size(self, size: Optional[tuple[int, int]]) -> None:
        """Real frame size in projector pixels (the preview copy may be smaller)."""
        self._true_size = size
        self.update()

    def set_geometry(self, geometry: CanvasGeometry) -> None:
        self._geometry = geometry
        self.update()

    def set_frame(self, frame: Optional[np.ndarray]) -> None:
        self._image = None if frame is None else _qimage(frame)
        self.update()

    def set_message(self, text: str) -> None:
        self._message = text
        self.update()

    def screen_rect(self) -> QRectF:
        g = self._geometry
        avail_w, avail_h = self.width() - 2.0, self.height() - 2.0
        k = min(avail_w / g.width, avail_h / g.height)
        w, h = g.width * k, g.height * k
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        p = theme.current()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.screen_rect()
        painter.setPen(QPen(QColor(p.border), 1.0))
        painter.setBrush(QColor("#000000"))
        painter.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), 6, 6)
        k = rect.width() / self._geometry.width
        if self._image is not None:
            # Where the vial walls are on the screen (the frame is exactly as wide as the vial).
            fw, fh = self._true_size or (self._image.width(), self._image.height())
            vx, _vy = self._geometry.placement(fw, fh)
            painter.setPen(QPen(QColor(p.dim), 1.0, Qt.PenStyle.DashLine))
            for wall in (vx, vx + fw):
                xw = rect.x() + wall * k
                painter.drawLine(QPointF(xw, rect.top() + 6), QPointF(xw, rect.bottom() - 6))
            x, y = self._geometry.placement(fw, fh)
            target = QRectF(rect.x() + x * k, rect.y() + y * k, fw * k, fh * k)
            painter.save()
            painter.setClipRect(rect)
            # Enlarging: keep the pixels crisp; shrinking: smooth (no aliasing).
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, k < 1.0)
            painter.drawImage(target, self._image)
            painter.restore()
        else:
            painter.setPen(QColor(p.dim))
            painter.setFont(theme.mono_font(12))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, tr(self._message))
        # Vial axis: a faint tick above and below the screen.
        cx = rect.center().x() + self._geometry.offset_x * k
        painter.setPen(QPen(QColor(p.accent), 1.4))
        painter.drawLine(QPointF(cx, rect.top() - 7), QPointF(cx, rect.top() - 2))
        painter.drawLine(QPointF(cx, rect.bottom() + 2), QPointF(cx, rect.bottom() + 7))
        painter.end()


class _DoseMap(QWidget):
    """The round map of the middle layer (painted by DoseCard's data)."""

    SIDE = 156

    def __init__(self, card: "DoseCard") -> None:
        super().__init__(card)
        self._card = card
        self.setFixedSize(self.SIDE, self.SIDE)

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        p = theme.current()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        painter.setPen(QPen(QColor(p.border), 1.0))
        painter.setBrush(QColor(p.view))
        painter.drawRoundedRect(rect, 10, 10)
        image = self._card._image
        if image is not None:
            side = min(rect.width(), rect.height()) - 14
            target = QRectF(rect.center().x() - side / 2, rect.center().y() - side / 2, side, side)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            clip = QPainterPath()
            clip.addEllipse(target)
            painter.setClipPath(clip)          # the vial is round
            painter.drawImage(target, image)
            painter.setClipping(False)
            painter.setPen(QPen(QColor(p.sketch), 1.4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(target)
        else:
            painter.setPen(QColor(p.dim))
            painter.setFont(theme.mono_font(10))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, tr("нет данных"))
        painter.end()


class DoseCard(QWidget):
    """Light dose in the middle layer: what cures, what stays liquid — and the exposure.

    The slider is the print's exposure (100 % = the plan): the map shows what cures at
    that exposure, and the print time follows it (exposureChanged).
    """

    exposureChanged = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dose: Optional[np.ndarray] = None
        self._target: Optional[np.ndarray] = None
        self._threshold = 1.0
        self._exposure = 1.0
        self._rmse: Optional[float] = None
        self._image: Optional[QImage] = None
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 4, 0, 4)
        row.setSpacing(14)
        self.map = _DoseMap(self)
        row.addWidget(self.map, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(5)
        self.title = QLabel(tr("Доза в смоле"))
        self.title.setObjectName("rowLabel")
        self.sub = QLabel(tr("средний слой детали"))
        self.sub.setObjectName("rowEffect")
        self.exposure_label = QLabel()
        self.exposure_label.setObjectName("rowLabel")
        self.exposure_label.setProperty("_spinslicer_skip_text_translation", True)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(70, 130)
        self.slider.setValue(100)
        self.slider.setPageStep(5)
        self.slider.valueChanged.connect(self._on_exposure)
        self.stats = QLabel()
        self.stats.setObjectName("rowEffect")
        self.stats.setWordWrap(True)
        self.stats.setProperty("_spinslicer_skip_text_translation", True)
        self.legend = QLabel(tr("розовое — затвердеет · пунктир — модель"))
        self.legend.setObjectName("rowEffect")
        self.legend.setWordWrap(True)
        for widget in (self.title, self.sub):
            col.addWidget(widget)
        col.addSpacing(4)
        col.addWidget(self.exposure_label)
        col.addWidget(self.slider)
        col.addWidget(self.stats)
        col.addWidget(self.legend)
        col.addStretch(1)
        row.addLayout(col, 1)
        self._refresh_text()

    # ------------------------------------------------------------------ exposure
    def exposure(self) -> float:
        return self._exposure

    def set_exposure(self, value: float) -> None:
        self.slider.setValue(int(round(max(0.7, min(1.3, float(value))) * 100)))

    def load(self, out_dir: Optional[str]) -> None:
        self._dose = self._target = None
        self._rmse = None
        if out_dir:
            try:
                self._dose = np.load(os.path.join(out_dir, "reconstruction_dose.npy"))
                self._target = np.load(os.path.join(out_dir, "reconstruction_target.npy"))
                import json
                plan_path = os.path.join(out_dir, "print_plan.json")
                self._threshold = 1.0
                if os.path.isfile(plan_path):
                    with open(plan_path, encoding="utf-8") as handle:
                        self._threshold = float(json.load(handle).get("relative_threshold_dose", 0)) or 0.0
                if self._threshold <= 0 and self._target is not None and (self._target > 0.5).any():
                    inside = self._dose[self._target > 0.5]
                    self._threshold = float(np.percentile(inside, 5))
                with open(os.path.join(out_dir, "dose_stats.txt"), encoding="utf-8") as handle:
                    stats = dict(line.split("=", 1) for line in handle.read().splitlines() if "=" in line)
                self._rmse = float(stats["rmse"]) if "rmse" in stats else None
            except (OSError, ValueError, KeyError):
                self._dose = self._target = None
        self._render()

    def _on_exposure(self, value: int) -> None:
        self._exposure = value / 100.0
        self._render()
        self.exposureChanged.emit(self._exposure)

    def _refresh_text(self) -> None:
        pct = round((self._exposure - 1.0) * 100)
        self.exposure_label.setText(tr(f"экспозиция {'+' if pct > 0 else ''}{pct} %") if pct else tr("экспозиция по плану"))
        if self._dose is None or self._target is None:
            self.stats.setText(tr("появится после расчёта света"))
            return
        part = self._target > 0.5
        cured = self._dose * self._exposure >= self._threshold
        missing = float((part & ~cured).sum()) / max(int(part.sum()), 1)
        extra = float((~part & cured).sum()) / max(int(part.sum()), 1)
        line = tr(f"не досвечено {missing * 100:.1f} % · лишнее {extra * 100:.1f} %".replace(".", ","))
        if self._rmse is not None:
            line += "\n" + tr(f"ошибка формы дозы {self._rmse:.3f}".replace(".", ","))
        self.stats.setText(line)

    def _render(self) -> None:
        self._image = None
        if self._dose is not None and self._target is not None:
            p = theme.current()
            bg, acc, hi = (np.array(QColor(c).getRgb()[:3], dtype=np.float32) for c in (p.view, p.accent, p.text))
            ratio = np.clip(self._dose * self._exposure / max(self._threshold, 1e-9), 0.0, 1.6)
            low = np.clip(ratio, 0, 1)[..., None]
            rgb = bg * (1 - low * 0.55) + acc * (low * 0.55)
            over = (ratio >= 1.0)[..., None]
            glow = np.clip((ratio - 1.0) / 0.6, 0, 1)[..., None]
            rgb = np.where(over, acc * (1 - glow * 0.5) + hi * (glow * 0.5), rgb)
            part = self._target > 0.5
            edge = part & ~(np.roll(part, 1, 0) & np.roll(part, -1, 0) & np.roll(part, 1, 1) & np.roll(part, -1, 1))
            yy, xx = np.nonzero(edge)
            dashed = ((yy + xx) // 2) % 2 == 0
            rgb[yy[dashed], xx[dashed]] = np.array(QColor(p.sketch).getRgb()[:3], dtype=np.float32)
            img8 = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
            h, w = img8.shape[:2]
            self._image = QImage(img8.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._refresh_text()
        self.map.update()

    def refresh_theme(self) -> None:
        self._render()

    def refresh_language(self) -> None:
        self._refresh_text()


def brightness_lut(brightness: float, gamma: float) -> np.ndarray:
    """Gray → gray that shows `brightness` × the light on a projector with this gamma.

    Light ∝ (g/255)^γ, so scaling the light by b scales the gray code by b^(1/γ).
    """
    b = max(0.0, min(1.0, float(brightness)))
    g = np.arange(256, dtype=np.float64) * b ** (1.0 / max(float(gamma), 0.1))
    return np.clip(np.rint(g), 0, 255).astype(np.uint8)


class _FrameCache:
    """The print frames in memory, decoded in the background before they are needed.

    Reading a PNG from disk in the middle of a print can be late by a frame when the
    disk is busy; a late frame is a wrong angle. Frames are decoded once (with the
    brightness table applied) on a background thread; if they would not fit in
    memory they are read from disk as they come.
    """

    MAX_BYTES = 1_500_000_000

    def __init__(self, frames: list, lut: Optional[np.ndarray]) -> None:
        import threading
        self._frames = frames
        self._lut = lut
        self._images: list[Optional[QImage]] = [None] * len(frames)
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        first = self._decode(0) if frames else None
        if first is not None:
            self._images[0] = first
            if first.sizeInBytes() * len(frames) <= self.MAX_BYTES:
                self._thread = threading.Thread(target=self._fill, daemon=True)
                self._thread.start()

    def _decode(self, index: int) -> Optional[QImage]:
        item = self._frames[index]
        if isinstance(item, str):
            import cv2
            item = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
            if item is None:
                return None
        array = np.asarray(item, dtype=np.uint8)
        if self._lut is not None:
            array = self._lut[array]
        return _qimage(array)

    def _fill(self) -> None:
        for i in range(1, len(self._frames)):
            if self._stop:
                return
            if self._images[i] is None:
                self._images[i] = self._decode(i)

    def get(self, index: int) -> Optional[QImage]:
        image = self._images[index]
        if image is None:
            image = self._decode(index)
            if self._thread is not None:
                self._images[index] = image
        return image

    def ready(self) -> bool:
        return self._thread is None or not self._thread.is_alive()

    def close(self) -> None:
        self._stop = True


class ProjectorOutputWindow(QWidget):
    """Full-screen window on the projector: frames 1:1 at the print frame rate. Esc stops.

    The frame shown is computed from the elapsed time (frame = ⌊t · fps⌋), never by
    counting timer ticks: a 24 Hz timer ticks every 42 ms instead of 41.67, which would
    let the picture fall behind the turning vial by 0.8 % — over a print of 200 turns
    more than a whole turn. With `duration_s` the window stops by itself when the
    planned dose has been delivered.
    """

    tick = pyqtSignal(float, float)          # elapsed s, total s (0 = no limit)
    finished = pyqtSignal(bool)              # True = the planned exposure was completed

    def __init__(self, frames: list, geometry: CanvasGeometry, fps: float, screen: object,
                 static: Optional[np.ndarray] = None, duration_s: float = 0.0,
                 lut: Optional[np.ndarray] = None) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        from PyQt6.QtCore import QElapsedTimer
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setCursor(Qt.CursorShape.BlankCursor)
        self._geometry = geometry
        self._fps = max(0.1, float(fps))
        self._duration = max(0.0, float(duration_s))
        self._static = None if static is None else _qimage(static if lut is None else lut[static])
        self._cache = _FrameCache(frames, lut) if static is None and frames else None
        self._count = len(frames)
        self._shown = -1
        self._image: Optional[QImage] = None
        self._completed = False
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._next)
        self._last_tick = -1.0
        self.setScreen(screen)  # type: ignore[arg-type]
        self.setGeometry(screen.geometry())  # type: ignore[attr-defined]
        self.showFullScreen()
        if self._cache is not None:
            self._clock.start()
            self._next()
            # Poll faster than the frame rate; only a new frame index repaints.
            self._timer.start(max(2, min(10, int(500.0 / self._fps))))

    def elapsed_s(self) -> float:
        return self._clock.nsecsElapsed() / 1e9 if self._clock.isValid() else 0.0

    def _next(self) -> None:
        t = self.elapsed_s()
        if self._duration and t >= self._duration:
            self._completed = True
            self.tick.emit(self._duration, self._duration)
            self.close()
            return
        index = int(t * self._fps) % self._count
        if index != self._shown and self._cache is not None:
            image = self._cache.get(index)
            if image is not None:
                self._image = image
            self._shown = index
            self.update()
        if t - self._last_tick >= 0.5:
            self._last_tick = t
            self.tick.emit(t, self._duration)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:  # noqa: N802
        if event is not None and event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Space):
            self.close()

    def closeEvent(self, event: object) -> None:  # noqa: N802
        self._timer.stop()
        if self._cache is not None:
            self._cache.close()
        self.finished.emit(self._completed)
        super().closeEvent(event)  # type: ignore[arg-type]

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        dpr = self.devicePixelRatioF() or 1.0
        image = self._static or self._image
        if image is not None:
            if self._static is not None:
                painter.drawImage(QRectF(0, 0, image.width() / dpr, image.height() / dpr), image)
            else:
                x, y = self._geometry.placement(image.width(), image.height())
                painter.drawImage(QRectF(x / dpr, y / dpr, image.width() / dpr, image.height() / dpr), image)
        painter.end()


class ProjectorTab(QWidget):
    progress = pyqtSignal(float, str)
    logMessage = pyqtSignal(str)
    framesReady = pyqtSignal(int)
    projectionTick = pyqtSignal(float, float)      # elapsed s, planned s (0 = endless)
    projectionFinished = pyqtSignal(bool)          # True = planned exposure completed

    def __init__(self, parent: Optional[QWidget] = None, job_controller: Optional[JobController] = None):
        super().__init__(parent)
        self._job_controller = job_controller or JobController()
        self._output_dir: Optional[str] = None
        self._frames: List[np.ndarray] = []
        self._frame_index = 0
        self._assemble_worker: Optional[VideoAssembleWorker] = None
        self._export_worker: Optional[VideoExportWorker] = None
        self._output_window: Optional[ProjectorOutputWindow] = None
        self._geometry = CanvasGeometry()
        self._true_size: Optional[tuple[int, int]] = None
        self._fps = BASE_FPS
        self._brightness = 1.0
        self._gamma = 2.2
        self._one_turn = False
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._advance_frame)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        top = QHBoxLayout()
        title = QLabel(tr("Экран проектора"))
        title.setObjectName("rowLabel")
        self.info_label = QLabel()
        self.info_label.setObjectName("rowEffect")
        self.info_label.setProperty("_spinslicer_skip_text_translation", True)
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.info_label)
        root.addLayout(top)

        self.preview = ScreenPreview()
        # The dose card lives in the print panel (SpinSlicer puts it there); the
        # screen keeps the whole width for the projector picture.
        self.dose = DoseCard()
        root.addWidget(self.preview, 1)

        transport = QHBoxLayout()
        transport.setSpacing(10)
        self.play_pause_btn = QPushButton("▶")
        self.play_pause_btn.setObjectName("stepButton")
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play_pause_btn.clicked.connect(self._on_play_pause_clicked)
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setEnabled(False)
        self.scrubber.valueChanged.connect(self._on_scrub)
        self.frame_label = QLabel("— / —")
        self.frame_label.setObjectName("rowEffect")
        self.frame_label.setProperty("_spinslicer_skip_text_translation", True)
        self.frame_label.setMinimumWidth(96)
        transport.addWidget(self.play_pause_btn)
        transport.addWidget(self.scrubber, 1)
        transport.addWidget(self.frame_label)
        root.addLayout(transport)

        # Programmatic surface kept from the old screen (release check, panel actions).
        self.dir_label = QLabel()
        self.dir_label.setVisible(False)
        self.assemble_btn = QPushButton()
        self.assemble_btn.clicked.connect(self._on_assemble_clicked)
        self.save_btn = QPushButton()
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.ramp_btn = QPushButton()
        self.ramp_btn.clicked.connect(self._on_ramp_clicked)
        for hidden in (self.assemble_btn, self.save_btn, self.ramp_btn):
            hidden.setVisible(False)
        self._refresh_info()

    # ------------------------------------------------------------------ setup from the panel
    def set_canvas(self, geometry: CanvasGeometry) -> None:
        self._geometry = geometry
        self.preview.set_geometry(geometry)
        self._refresh_info()

    def set_fps(self, fps: float) -> None:
        self._fps = max(0.1, float(fps))
        if self._timer.isActive():
            self._timer.start(self._interval())
        self._refresh_info()

    def set_output_light(self, brightness: float, gamma: float) -> None:
        """Projector brightness (0.1…1 of full light) used for the projector and the video."""
        self._brightness = max(0.05, min(1.0, float(brightness)))
        self._gamma = float(gamma)

    def _lut(self) -> Optional[np.ndarray]:
        return None if self._brightness >= 0.999 else brightness_lut(self._brightness, self._gamma)

    def is_projecting(self) -> bool:
        return self._output_window is not None

    def _interval(self) -> int:
        return max(5, int(round(1000.0 / self._fps)))

    def _refresh_info(self) -> None:
        g = self._geometry
        size = ""
        if self._frames:
            w, h = self._true_size or (self._frames[0].shape[1], self._frames[0].shape[0])
            size = f" · кадр {w}×{h}"
            if not g.fits(w, h):
                size += " — не помещается!"
        self.info_label.setText(tr(f"{g.width}×{g.height}{size} · {self._fps:.1f} Гц".replace(".", ",")))

    def has_frames(self) -> bool:
        return bool(self._frames)

    # ------------------------------------------------------------------ frames
    def set_output_dir(self, path: str) -> None:
        if path != self._output_dir:
            self._timer.stop()
            self._frames = []
            self._frame_index = 0
            self.play_pause_btn.setEnabled(False)
            self.scrubber.setEnabled(False)
            self.preview.set_frame(None)
        self._output_dir = path
        self.dir_label.setText(f"{tr('Папка кадров: ')}{path}")
        self.dose.load(path)

    def _on_assemble_clicked(self) -> None:
        if not self._output_dir or not os.path.isdir(self._output_dir):
            QMessageBox.warning(self, tr("Внимание"), tr("Сначала рассчитайте свет."))
            return
        self._timer.stop()
        self.preview.set_message(tr("Загружаю кадры…"))
        self.progress.emit(0.0, "Загрузка кадров...")
        self._assemble_worker = VideoAssembleWorker(self._output_dir, self)
        self._job_controller.register(self._assemble_worker)
        self._assemble_worker.progress.connect(lambda f, m: self.progress.emit(f, m))
        self._assemble_worker.finished_ok.connect(self._on_assembled)
        self._assemble_worker.failed.connect(self._on_assemble_failed)
        self._assemble_worker.start()

    def _on_assembled(self, frames: List[np.ndarray]) -> None:
        self._frames = frames
        self._true_size = None
        try:
            from frame_io import FrameRepository
            frame_set = FrameRepository.validate(self._output_dir or "")
            self._true_size = (int(frame_set.width), int(frame_set.height))
        except Exception:
            pass
        self.preview.set_true_size(self._true_size)
        self._frame_index = 0
        self.play_pause_btn.setEnabled(True)
        self.scrubber.setEnabled(True)
        self.scrubber.setRange(0, max(0, len(frames) - 1))
        self.progress.emit(1.0, tr("Видео собрано."))
        self.logMessage.emit(f"{tr('Собрано')} {len(frames)} {tr('кадров для воспроизведения.')}")
        self._show(0)
        self._refresh_info()
        self.framesReady.emit(len(frames))
        # Show one full turn, then hold still: an idle print screen costs nothing.
        # ▶ plays in a loop.
        self._start_playback(one_turn=True)

    def _on_assemble_failed(self, msg: str) -> None:
        self.progress.emit(0.0, tr("Сборка видео не удалась."))
        self.preview.set_message(tr("Не удалось прочитать кадры"))
        QMessageBox.critical(self, tr("Ошибка"), msg)

    def _show(self, index: int) -> None:
        if not self._frames:
            return
        self._frame_index = index % len(self._frames)
        self.preview.set_frame(self._frames[self._frame_index])
        self.frame_label.setText(f"{self._frame_index + 1} / {len(self._frames)}")
        if self.scrubber.value() != self._frame_index:
            self.scrubber.blockSignals(True)
            self.scrubber.setValue(self._frame_index)
            self.scrubber.blockSignals(False)

    def _start_playback(self, one_turn: bool = False) -> None:
        self._one_turn = one_turn
        if self._frames:
            self._timer.start(self._interval())
            self.play_pause_btn.setText("❚❚")

    def _on_play_pause_clicked(self) -> None:
        if not self._frames:
            return
        if self._timer.isActive():
            self._timer.stop()
            self.play_pause_btn.setText("▶")
        else:
            self._start_playback()

    def _on_scrub(self, value: int) -> None:
        self._show(value)

    def _advance_frame(self) -> None:
        if not self.isVisible():
            return
        if self._one_turn and self._frame_index + 1 >= len(self._frames):
            self._timer.stop()
            self.play_pause_btn.setText("▶")
            self._show(0)
            return
        self._show(self._frame_index + 1)

    def hideEvent(self, event: object) -> None:  # noqa: N802
        self._timer.stop()
        self.play_pause_btn.setText("▶")
        super().hideEvent(event)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ output
    def _printer_frames(self) -> list:
        """The saved PNG frames at full resolution when available (the preview may be downscaled)."""
        if self._output_dir:
            try:
                from frame_io import FrameRepository
                return list(FrameRepository.validate(self._output_dir).paths)
            except Exception:
                pass
        return list(self._frames)

    @staticmethod
    def projector_screens() -> list:
        """Screens other than the one the window is on (candidates for the projector)."""
        screens = QGuiApplication.screens()
        primary = QGuiApplication.primaryScreen()
        return [s for s in screens if s is not primary]

    def show_on_projector(self, pattern: bool = False, duration_s: float = 0.0) -> bool:
        """Frames full screen on the second display; stops by itself after duration_s (0 = never)."""
        screens = self.projector_screens()
        if not screens:
            QMessageBox.information(self, tr("Проектор"), tr(
                "Второй экран не найден. Подключите проектор как второй монитор "
                "(расширенный рабочий стол) или сохраните видео."))
            return False
        self.stop_projector()
        if pattern:
            w, h = self._true_size or ((self._frames[0].shape[1], self._frames[0].shape[0]) if self._frames
                                       else (self._geometry.height // 2, self._geometry.height // 3))
            static = alignment_pattern(self._geometry, w, h)
            self._output_window = ProjectorOutputWindow([], self._geometry, self._fps, screens[0], static)
        else:
            frames = self._printer_frames()
            if not frames:
                return False
            self._output_window = ProjectorOutputWindow(frames, self._geometry, self._fps, screens[0],
                                                        duration_s=duration_s, lut=self._lut())
        window = self._output_window
        window.tick.connect(self.projectionTick.emit)
        window.finished.connect(lambda done, w=window: self._on_projection_finished(w, done))
        self.logMessage.emit(tr("Кадры выводятся на второй экран. Esc — остановить."))
        return True

    def _on_projection_finished(self, window: object, completed: bool) -> None:
        if self._output_window is window:
            self._output_window = None
        self.projectionFinished.emit(completed)
        if completed:
            self.logMessage.emit(tr("Засветка завершена: доза набрана."))

    def stop_projector(self) -> None:
        if self._output_window is not None:
            try:
                self._output_window.close()
            except RuntimeError:
                pass
            self._output_window = None

    def _on_save_clicked(self) -> None:
        frames = self._printer_frames()
        if not frames:
            QMessageBox.warning(self, tr("Внимание"), tr("Сначала рассчитайте свет."))
            return
        g = self._geometry
        path, _ = QFileDialog.getSaveFileName(self, tr("Сохранить видео"), f"print_{g.width}x{g.height}.mp4", "MP4 (*.mp4)")
        if not path:
            return
        self.progress.emit(0.0, "Экспорт в MP4...")
        self._start_export(frames, self._fps, path, g, self._lut())

    def _on_ramp_clicked(self) -> None:
        g = self._geometry
        path, _ = QFileDialog.getSaveFileName(self, tr("Сохранить видео"), "gray_ramp.mp4", "MP4 (*.mp4)")
        if not path:
            return
        ramp = calibration_ramp(g.width, g.height)
        self.progress.emit(0.0, "Экспорт тестовой шкалы...")
        self._start_export([ramp] * int(BASE_FPS * 10), BASE_FPS, path, None)

    def _start_export(self, frames: list, fps: float, path: str, canvas: Optional[CanvasGeometry],
                      lut: Optional[np.ndarray] = None) -> None:
        self._export_worker = VideoExportWorker(frames, fps, path, self, canvas=canvas, lut=lut)
        self._job_controller.register(self._export_worker)
        self._export_worker.progress.connect(lambda f, m: self.progress.emit(f, m))
        self._export_worker.finished_ok.connect(self._on_export_done)
        self._export_worker.failed.connect(self._on_export_failed)
        self._export_worker.start()

    def _on_export_done(self, path: str) -> None:
        self.progress.emit(1.0, "Видео сохранено.")
        self.logMessage.emit(f"MP4: {path}")
        QMessageBox.information(self, tr("Готово"), f"{tr('Видео сохранено')}:\n{path}\n\n"
                                + tr("Показывайте его на проекторе во весь экран, без масштабирования."))

    def _on_export_failed(self, msg: str) -> None:
        self.progress.emit(0.0, "Экспорт не удался.")
        QMessageBox.critical(self, tr("Экспорт не удался"), msg)

    def composed_frame(self, index: int = 0) -> Optional[np.ndarray]:
        if not self._frames:
            return None
        return compose(self._frames[index % len(self._frames)], self._geometry)

    def update_language(self) -> None:
        self._refresh_info()

    def closeEvent(self, event) -> None:  # noqa: N802 (имя метода задано Qt)
        self.stop_projector()
        if self._job_controller.shutdown():
            event.accept()
        else:
            event.ignore()
