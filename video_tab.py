"""
video_tab.py
============
Вкладка "Проектор (Видео)": собирает кадры-проекции из output_frames в
последовательность numpy-массивов в памяти (через OpenCV) и проигрывает
её прямо во вкладке через QTimer, с регулировкой скорости и экспортом
в MP4 (тот же набор кадров, без повторного чтения с диска).
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from PyQt6.QtCore import QSignalBlocker, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from constants import DEFAULT_FRAME_RATE_HZ, DEFAULT_PROJECTOR_GAMMA
from frame_io import FrameRepository
from i18n import language, tr
from job_controller import JobController
from widgets import LabeledSlider
from workers import VideoAssembleWorker, VideoExportWorker

BASE_FPS = DEFAULT_FRAME_RATE_HZ
# Sub-folder written by the slicer with the projection without vial compensation.
BARE_PROJECTION_DIR = "bare_projection"


class VideoPreviewWidget(QLabel):
    """Отображает кадры видео без изменения геометрии родительского окна."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self._pixmap: Optional[QPixmap] = None
        self._intensity: float = 1.0
        self._current_raw_frame: Optional[np.ndarray] = None

    def set_intensity(self, val: float) -> None:
        self._intensity = max(0.1, float(val))
        if self._current_raw_frame is not None:
            self._render_frame(self._current_raw_frame)

    def set_frame(self, frame: np.ndarray) -> None:
        self._current_raw_frame = frame
        self._render_frame(frame)

    def _render_frame(self, frame: np.ndarray) -> None:
        if self._intensity != 1.0:
            frame_adj = np.clip(frame.astype(np.float32) * self._intensity, 0, 255).astype(np.uint8)
        else:
            frame_adj = frame
        h, w = frame_adj.shape[:2]
        qimg = QImage(frame_adj.tobytes(), w, h, w, QImage.Format.Format_Grayscale8).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def clear_frame(self) -> None:
        self._pixmap = None
        self._current_raw_frame = None
        self.update()

    def paintEvent(self, event) -> None:
        if self._pixmap is not None and not self._pixmap.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            super().paintEvent(event)


class ProjectorTab(QWidget):
    progress = pyqtSignal(float, str)
    logMessage = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        job_controller: Optional[JobController] = None,
    ):
        super().__init__(parent)

        self._job_controller = job_controller or JobController()
        self._output_dir: Optional[str] = None
        self._frames: List[np.ndarray] = []
        self._frame_index = 0
        self._assemble_worker: Optional[VideoAssembleWorker] = None
        self._export_worker: Optional[VideoExportWorker] = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

        self._build_ui()

    # =======================================================================
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # Which frames to play: compensated for the vial, or the bare projection.
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self.mode_label = QLabel("Тип проекции:")
        mode_row.addWidget(self.mode_label)
        self.mode_corrected_btn = QPushButton("Скорректированная")
        self.mode_corrected_btn.setObjectName("secondaryButton")
        self.mode_corrected_btn.setCheckable(True)
        self.mode_corrected_btn.setChecked(True)
        self.mode_corrected_btn.setToolTip(
            "Проекция для печати: компенсированы преломление и поглощение в колбе"
        )
        self.mode_bare_btn = QPushButton("Нескорректированная")
        self.mode_bare_btn.setObjectName("secondaryButton")
        self.mode_bare_btn.setCheckable(True)
        self.mode_bare_btn.setEnabled(False)
        self.mode_bare_btn.setToolTip(
            "Прямая проекция детали без компенсации оптических искажений колбы"
        )
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self.mode_corrected_btn)
        self._mode_group.addButton(self.mode_bare_btn)
        self.mode_bare_btn.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_corrected_btn)
        mode_row.addWidget(self.mode_bare_btn)
        mode_row.addStretch(1)
        root.addLayout(mode_row)

        # Content row: Video Preview on left, Tomo Dose Reconstruction card on right
        content_row = QHBoxLayout()
        content_row.setSpacing(14)

        self.preview = VideoPreviewWidget("Нажмите «Собрать и воспроизвести»,\nчтобы увидеть анимацию проекций.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setObjectName("videoPreview")
        self.preview.setMinimumHeight(120)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_row.addWidget(self.preview, 3)

        # Side column on right: Dose Reconstruction + CyberPepe Meme
        # The column scrolls instead of imposing a tall minimum on the window,
        # so the log pane below can be dragged as tall as the user likes.
        side_widget = QWidget()
        side_col = QVBoxLayout(side_widget)
        side_col.setContentsMargins(0, 0, 0, 0)
        side_col.setSpacing(8)

        # Dose Reconstruction Card
        recon_box = QGroupBox("Реконструкция дозы")
        recon_box.setObjectName("reconBox")
        recon_box.setFixedWidth(265)
        recon_layout = QVBoxLayout(recon_box)
        recon_layout.setContentsMargins(10, 8, 10, 8)
        recon_layout.setSpacing(4)

        recon_title = QLabel("Средний срез (Mid-Slice)")
        recon_title.setStyleSheet("font-weight: bold; color: #00A8FF;")
        recon_layout.addWidget(recon_title)

        self.recon_label = QLabel("Нет данных\n(Сгенерируйте срез)")
        self.recon_label.setProperty("_spinslicer_skip_text_translation", True)
        self.recon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recon_label.setStyleSheet("background-color: #1A1A1A; border: 1px solid #333333; border-radius: 6px; color: #888888;")
        self.recon_label.setFixedSize(160, 160)
        recon_layout.addWidget(self.recon_label, 0, Qt.AlignmentFlag.AlignCenter)

        self.recon_stats = QLabel("RMSE относительной дозы: —")
        self.recon_stats.setToolTip("Ошибка формы дозы после подбора общего масштаба. Не прогноз отверждения смолы.")
        self.recon_stats.setWordWrap(True)
        self.recon_stats.setMinimumHeight(32)
        self.recon_stats.setStyleSheet("color: #CCCCCC; font-size: 11px; margin-top: 2px;")
        recon_layout.addWidget(self.recon_stats)

        side_col.addWidget(recon_box)

        # CyberPepe CAL Lab Easter Egg Card
        pepe_box = QGroupBox("Лаборатория Пепе 🐸")
        pepe_box.setObjectName("pepeBox")
        pepe_box.setFixedWidth(265)
        pepe_layout = QVBoxLayout(pepe_box)
        pepe_layout.setContentsMargins(10, 8, 10, 8)
        pepe_layout.setSpacing(4)

        pepe_img_label = QLabel()
        pepe_img_label.setProperty("_spinslicer_skip_text_translation", True)
        pepe_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pepe_img_label.setFixedSize(160, 160)
        pepe_img_label.setStyleSheet("border-radius: 8px; border: none; background-color: #0d0d0d;")
        pepe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "pepe_meme.jpg")
        if os.path.exists(pepe_path):
            pix = QPixmap(pepe_path)
            if not pix.isNull():
                pepe_img_label.setPixmap(pix.scaled(160, 160, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation))
        pepe_layout.addWidget(pepe_img_label, 0, Qt.AlignmentFlag.AlignCenter)

        self._pepe_quotes_ru = [
            "«Кайф, когда доза сошлась» 🐸⚡",
            "«Нельзя просто так напечатать без OpenCAL» 🐸🧪",
            "«Когда Бугер-Ламберт лёг как надо» 🐸🔬",
            "«0% поддержек, 100% объёмная скорость» 🐸🚀",
            "«Фотонное превосходство в действии» 🐸✨",
            "«Томография за 30 секунд — это база» 🐸⏱",
            "«Свет пронзает смолу, деталь растёт» 🐸💡",
        ]
        self._pepe_quotes_en = [
            "“Feels good when dose converges” 🐸⚡",
            "“One does not simply print without OpenCAL” 🐸🧪",
            "“When Beer-Lambert hits just right” 🐸🔬",
            "“0% supports, 100% volumetric speed” 🐸🚀",
            "“Photon supremacy in full effect” 🐸✨",
            "“30-second tomography is the standard” 🐸⏱",
            "“Light penetrates resin, structure rises” 🐸💡",
        ]
        self._pepe_quote_idx = 0
        self.pepe_caption = QLabel(self._current_quotes()[0])
        self.pepe_caption.setProperty("_spinslicer_skip_text_translation", True)
        self.pepe_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pepe_caption.setWordWrap(True)
        self.pepe_caption.setMinimumHeight(34)
        self.pepe_caption.setStyleSheet("color: #00FF88; font-weight: bold; font-size: 10px; margin-top: 2px; margin-bottom: 2px;")
        pepe_layout.addWidget(self.pepe_caption)

        quote_btn = QPushButton("🎲 Другая цитата")
        quote_btn.setObjectName("secondaryButton")
        quote_btn.setMinimumHeight(28)
        quote_btn.clicked.connect(self._cycle_pepe_quote)
        pepe_layout.addWidget(quote_btn)

        side_col.addWidget(pepe_box)
        side_col.addStretch(1)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setFixedWidth(285)
        side_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        side_scroll.setWidget(side_widget)
        content_row.addWidget(side_scroll)
        root.addLayout(content_row, 1)

        buttons = QHBoxLayout()
        self.assemble_btn = QPushButton("▶ Собрать и воспроизвести")
        self.assemble_btn.setObjectName("generateButton")
        self.assemble_btn.setToolTip("Прочитать все кадры из папки и запустить проигрывание")
        self.assemble_btn.clicked.connect(self._on_assemble_clicked)
        buttons.addWidget(self.assemble_btn)

        self.play_pause_btn = QPushButton("⏸ Пауза")
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.clicked.connect(self._on_play_pause_clicked)
        buttons.addWidget(self.play_pause_btn)

        self.save_btn = QPushButton("💾 Сохранить в MP4")
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip("Экспортировать уже собранные кадры в видеофайл")
        self.save_btn.clicked.connect(self._on_save_clicked)
        buttons.addWidget(self.save_btn)
        root.addLayout(buttons)

        sliders_row = QHBoxLayout()
        self.speed_slider = LabeledSlider(
            "Скорость воспроизведения", 0.1, 4.0, 1.0, decimals=2, suffix="×",
        )
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        sliders_row.addWidget(self.speed_slider)

        self.live_intensity_slider = LabeledSlider(
            "Яркость проекции (Live Intensity)", 0.2, 5.0, 1.0, decimals=2, suffix="×",
        )
        self.live_intensity_slider.valueChanged.connect(self._on_live_intensity_changed)
        sliders_row.addWidget(self.live_intensity_slider)

        root.addLayout(sliders_row)

    def _on_live_intensity_changed(self, val: float) -> None:
        self.preview.set_intensity(val)

    def _current_quotes(self) -> list[str]:
        if language() == "ru":
            return self._pepe_quotes_ru
        return self._pepe_quotes_en

    def _cycle_pepe_quote(self) -> None:
        quotes = self._current_quotes()
        self._pepe_quote_idx = (self._pepe_quote_idx + 1) % len(quotes)
        self.pepe_caption.setText(quotes[self._pepe_quote_idx])

    def update_language(self) -> None:
        quotes = self._current_quotes()
        self._pepe_quote_idx = self._pepe_quote_idx % len(quotes)
        self.pepe_caption.setText(quotes[self._pepe_quote_idx])
        if self._output_dir and os.path.exists(os.path.join(self._output_dir, "reconstruction_preview.png")):
            self._load_dose_preview(self._output_dir)
        else:
            self.recon_label.setText(tr("Нет данных\n(Сгенерируйте срез)"))
            self.recon_stats.setText(tr("RMSE относительной дозы: —"))

    def _load_dose_preview(self, path: str) -> None:
        if not path or not os.path.isdir(path):
            return
        preview_path = os.path.join(path, "reconstruction_preview.png")
        if os.path.exists(preview_path):
            pix = QPixmap(preview_path)
            if not pix.isNull():
                scaled_pix = pix.scaled(160, 160, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.recon_label.setPixmap(scaled_pix)

        stats_path = os.path.join(path, "dose_stats.txt")
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    content = f.read()
                statistics = dict(line.split("=", 1) for line in content.splitlines() if "=" in line)
                if "rmse" in statistics:
                    value = float(statistics["rmse"])
                    self.recon_stats.setText(f"{tr('RMSE относительной дозы:')} {value:.4f}")
                else:
                    value = float(statistics["loss"])
                    self.recon_stats.setText(f"{tr('Потери дозы (L2 Loss):')} {value:.4f}")
            except (OSError, ValueError, KeyError):
                self.recon_stats.setText(tr("Не удалось прочитать статистику реконструкции."))

    # --- общая папка вывода (устанавливается из вкладки "Слайсер") -------------
    def set_output_dir(self, path: str) -> None:
        if path != self._output_dir:
            self._timer.stop()
            self._frames = []
            self._frame_index = 0
            self.play_pause_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.preview.clear_frame()
        self._output_dir = path
        self._load_dose_preview(path)
        self._refresh_modes()

    def _run_dir(self) -> Optional[str]:
        if not self._output_dir:
            return None
        try:
            return FrameRepository.validate(self._output_dir).resolved_dir
        except Exception:
            return None

    def _bare_available(self) -> bool:
        run_dir = self._run_dir()
        return bool(run_dir) and os.path.isdir(os.path.join(run_dir or "", BARE_PROJECTION_DIR))

    def _bare_selected(self) -> bool:
        return self.mode_bare_btn.isChecked()

    def _refresh_modes(self) -> None:
        """Enable the bare view only for runs that actually contain it."""
        available = self._bare_available()
        self.mode_bare_btn.setEnabled(available)
        if not available and self.mode_bare_btn.isChecked():
            blocker = QSignalBlocker(self.mode_bare_btn)
            self.mode_corrected_btn.setChecked(True)
            del blocker
        if not available:
            self.mode_bare_btn.setToolTip(
                "Для этого прогона нескорректированная проекция недоступна (не создаётся в экономном режиме)"
            )

    def _on_mode_changed(self, _checked: bool) -> None:
        # Frames of the other mode have to be loaded; replay from the start.
        if self._frames or self._timer.isActive():
            self._on_assemble_clicked()

    # --- сборка и проигрывание ---------------------------------------------------
    def _on_assemble_clicked(self) -> None:
        if not self._output_dir or not os.path.isdir(self._output_dir):
            QMessageBox.warning(
                self, tr("Внимание"), "Select a frame folder first (output_frames)."
            )
            return

        self._timer.stop()
        self.assemble_btn.setEnabled(False)
        self.progress.emit(0.0, "Сборка видео из кадров...")

        self._assemble_worker = VideoAssembleWorker(
            self._output_dir, self,
            subdir=BARE_PROJECTION_DIR if self._bare_selected() else "",
        )
        self._job_controller.register(self._assemble_worker)
        self._assemble_worker.progress.connect(lambda f, m: self.progress.emit(f, m))
        self._assemble_worker.finished_ok.connect(self._on_assembled)
        self._assemble_worker.failed.connect(self._on_assemble_failed)
        self._assemble_worker.cancelled.connect(
            lambda: self.progress.emit(0.0, "Video assembly cancelled.")
        )
        self._assemble_worker.finished.connect(lambda: self.assemble_btn.setEnabled(True))
        self._assemble_worker.start()

    def _on_assembled(self, frames: List[np.ndarray]) -> None:
        self._frames = frames
        self._frame_index = 0
        self.play_pause_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.play_pause_btn.setText(tr("⏸ Пауза"))
        self.progress.emit(1.0, tr("Видео собрано."))
        self.logMessage.emit(f"{tr('Собрано')} {len(frames)} {tr('кадров для воспроизведения.')}")
        if frames:
            self.preview.set_frame(frames[0])
        self._start_playback()

    def _on_assemble_failed(self, msg: str) -> None:
        self.progress.emit(0.0, tr("Сборка видео не удалась."))
        QMessageBox.critical(self, tr("Ошибка"), msg)

    def _start_playback(self) -> None:
        if not self._frames:
            return
        interval_ms = max(int(1000 / (BASE_FPS * self.speed_slider.value())), 5)
        self._timer.start(interval_ms)

    def _on_play_pause_clicked(self) -> None:
        if not self._frames:
            return
        if self._timer.isActive():
            self._timer.stop()
            self.play_pause_btn.setText(tr("▶ Воспроизведение"))
        else:
            self._start_playback()
            self.play_pause_btn.setText(tr("⏸ Пауза"))

    def _on_speed_changed(self, _v: float) -> None:
        if self._timer.isActive():
            self._start_playback()

    def _advance_frame(self) -> None:
        if not self._frames:
            return
        frame = self._frames[self._frame_index]
        self._frame_index = (self._frame_index + 1) % len(self._frames)
        self.preview.set_frame(frame)

    # --- экспорт в MP4 -----------------------------------------------------------
    def _export_settings(self) -> tuple[float, float]:
        """FPS and projector gamma of the printed file, taken from the run manifest.

        FPS follows the generation schedule, never the preview speed; gamma is
        the machine profile's projector gamma (1.0 keeps the frames linear).
        """
        from frame_io import FrameRepository, load_manifest

        rate, gamma = 0.0, DEFAULT_PROJECTOR_GAMMA
        try:
            directory = FrameRepository.validate(self._output_dir or "").resolved_dir
            manifest = load_manifest(directory)
            if manifest:
                rate = float((manifest.frame_schedule or {}).get("frame_rate_hz", 0.0))
                projector = (manifest.machine_profile or {}).get("projector") or {}
                gamma = float(projector.get("gamma", DEFAULT_PROJECTOR_GAMMA))
        except Exception:
            pass
        return (rate if rate > 0 else DEFAULT_FRAME_RATE_HZ), gamma

    def _on_save_clicked(self) -> None:
        if not self._frames:
            QMessageBox.warning(self, "Warning", "Assemble the video first.")
            return

        if self._bare_selected():
            answer = QMessageBox.question(
                self,
                tr("Внимание"),
                tr("Нескорректированная проекция не компенсирует преломление и поглощение в колбе. "
                   "Для печати используйте скорректированную проекцию. Сохранить файл?"),
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        fps, gamma = self._export_settings()
        rpm = 60.0 * fps / len(self._frames)
        name = "projection_uncorrected" if self._bare_selected() else "projection"
        # OpenCAL reads the rotation speed from the file name: <name>_<rpm>rpm.mp4
        path, _ = QFileDialog.getSaveFileName(
            self, "Save video", f"{name}_{rpm:g}rpm.mp4", "MP4 (*.mp4)"
        )
        if not path:
            return

        self.save_btn.setEnabled(False)
        self.progress.emit(0.0, "Экспорт в MP4...")

        self._export_worker = VideoExportWorker(self._frames, fps, path, self, projector_gamma=gamma)
        self._job_controller.register(self._export_worker)
        self._export_worker.progress.connect(lambda f, m: self.progress.emit(f, m))
        self._export_worker.finished_ok.connect(self._on_export_done)
        self._export_worker.failed.connect(self._on_export_failed)
        self._export_worker.cancelled.connect(
            lambda: self.progress.emit(0.0, "Video export cancelled.")
        )
        self._export_worker.finished.connect(lambda: self.save_btn.setEnabled(True))
        self._export_worker.start()

    def _on_export_done(self, path: str) -> None:
        self.progress.emit(1.0, "Video saved.")
        self.logMessage.emit(f"MP4 saved: {path}")
        QMessageBox.information(self, "Success", f"Video saved:\n{path}")

    def _on_export_failed(self, msg: str) -> None:
        self.progress.emit(0.0, "Video export failed.")
        QMessageBox.critical(self, "Video export failed", msg)

    def closeEvent(self, event) -> None:  # noqa: N802 (имя метода задано Qt)
        if self._job_controller.shutdown():
            event.accept()
        else:
            event.ignore()
