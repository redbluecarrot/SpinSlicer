"""
workers.py
==========
UI-поток никогда не должен блокироваться загрузкой STL или расчётом
проекций. Вся тяжёлая работа выполняется в QThread, наружу летят только
Qt-сигналы — без ручных queue.Queue и after()-поллинга, как в прототипе
на tkinter.
"""
from __future__ import annotations

import math
import os
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import trimesh
from PyQt6.QtCore import QThread, pyqtSignal

from constants import (
    MAX_FRAME_DIMENSION,
    MAX_FRAME_TOTAL_PIXELS,
    MAX_NUM_FRAMES,
    MAX_VIDEO_PREVIEW_BYTES,
)
from frame_io import FrameRepository
from slicing_engine import SliceCancelled, SliceParams, SlicingEngine
from validation import ValidationError, validate_mesh_geometry, validate_stl_path

# Верхняя граница стороны кадра при сборке видео в памяти — защита от
# многогигабайтного потребления RAM на длинных сериях в высоком разрешении.
# Влияет и на предпросмотр, и на экспортируемый MP4 (кадры переиспользуются).
VIDEO_PREVIEW_MAX_DIM = 900


class LoadMeshWorker(QThread):
    """Загружает STL с диска и центрирует его по bounding box."""

    loaded = pyqtSignal(str, object)   # (путь, trimesh.Trimesh)
    failed = pyqtSignal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            validate_stl_path(self.path)
            if self._cancel_event.is_set():
                return
            loaded = trimesh.load(self.path, force="mesh")
            mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError("Не удалось загрузить корректную триангуляцию.")
            validate_mesh_geometry(mesh)
            if self._cancel_event.is_set():
                return

            # Строгое геометрическое центрирование по bounding box —
            # ModelNode дальше работает только со сдвигом/масштабом поверх этого.
            mesh.apply_translation(-mesh.bounding_box.centroid)

            self.loaded.emit(self.path, mesh)
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=3)}")


class GenerationWorker(QThread):
    """Запускает SlicingEngine в отдельном потоке и транслирует прогресс в UI."""

    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(int, str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        params: SliceParams,
        parent=None,
        manifest_context: Optional[dict[str, Any]] = None,
    ):
        super().__init__(parent)
        self.mesh = mesh
        self.params = params
        self.manifest_context = manifest_context or {}
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            num_frames, out_dir = SlicingEngine.run(
                self.mesh,
                self.params,
                progress_cb=lambda frac, msg: self.progress.emit(frac, msg),
                is_cancelled=self._cancel_event.is_set,
                manifest_context=self.manifest_context,
            )
            self.finished_ok.emit(num_frames, out_dir)
        except SliceCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=6)}")


class VideoAssembleWorker(QThread):
    """Читает кадры из output_frames через OpenCV и держит их в памяти
    (список numpy-массивов) — без промежуточного видеофайла на диске."""

    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(list)   # List[np.ndarray], grayscale
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, frames_dir: str, parent=None):
        super().__init__(parent)
        self.frames_dir = frames_dir
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            import cv2
        except ImportError:
            self.failed.emit(
                "Не установлен OpenCV. Выполните: pip install opencv-python-headless"
            )
            return

        try:
            frame_set = FrameRepository.validate(self.frames_dir)
            paths = frame_set.paths

            frames: List[np.ndarray] = []
            target_shape: Optional[Tuple[int, int]] = None  # (h, w) по первому кадру
            report_step = max(1, len(paths) // 30)
            preview_bytes = 0

            for i, path in enumerate(paths):
                if self._cancel_event.is_set():
                    self.cancelled.emit()
                    return
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValidationError(f"Не удалось декодировать кадр {Path(path).name}.")
                if img.ndim != 2 or img.shape[:2] != (frame_set.height, frame_set.width):
                    raise ValidationError(
                        f"Размер кадра изменился во время чтения: {Path(path).name}."
                    )
                h, w = img.shape[:2]
                max_dim = max(h, w)
                if max_dim > VIDEO_PREVIEW_MAX_DIM:
                    scale = VIDEO_PREVIEW_MAX_DIM / max_dim
                    img = cv2.resize(img, (max(int(w * scale), 1), max(int(h * scale), 1)),
                                      interpolation=cv2.INTER_AREA)

                if target_shape is None:
                    target_shape = img.shape[:2]
                elif img.shape[:2] != target_shape:
                    # Кадр другого размера — обычно "хвост" от генерации с
                    # другими параметрами (см. проверку метаданных выше).
                    # Приводим к размеру первого кадра: иначе видео дёргается,
                    # а cv2.VideoWriter при экспорте требует одинаковый размер
                    # каждого кадра и на несовпадении либо тихо портит файл,
                    # либо падает.
                    img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

                contiguous_img = np.ascontiguousarray(img)
                preview_bytes += contiguous_img.nbytes
                if preview_bytes > MAX_VIDEO_PREVIEW_BYTES:
                    raise ValidationError(
                        "Память для предпросмотра видео превышает установленный лимит."
                    )

                frames.append(contiguous_img)

                if i % report_step == 0:
                    self.progress.emit(i / len(paths), f"Загрузка кадра {i}/{len(paths)}...")

            if not frames:
                raise ValueError("Не удалось прочитать ни одного кадра.")

            self.finished_ok.emit(frames)
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=4)}")


class VideoExportWorker(QThread):
    """Кодирует уже собранные в памяти кадры в MP4 через cv2.VideoWriter."""

    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, frames: List[np.ndarray], fps: float, out_path: str, parent=None):
        super().__init__(parent)
        self.frames = frames
        self.fps = fps
        self.out_path = out_path
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            import cv2
        except ImportError:
            self.failed.emit(
                "Не установлен OpenCV. Выполните: pip install opencv-python-headless"
            )
            return

        writer = None
        temporary_path: Optional[str] = None
        published = False
        try:
            if not self.frames:
                raise ValueError("Нет кадров для сохранения.")
            if not math.isfinite(float(self.fps)) or not 0.1 <= self.fps <= 240.0:
                raise ValidationError("FPS должен быть конечным числом от 0.1 до 240.")
            if len(self.frames) > MAX_NUM_FRAMES:
                raise ValidationError("Слишком много кадров для экспорта.")

            total_pixels = 0
            total_memory = 0
            for frame in self.frames:
                if frame.ndim != 2:
                    raise ValueError("Кадры для экспорта должны быть grayscale-массивами.")
                if (
                    not frame.shape[0]
                    or not frame.shape[1]
                    or frame.shape[0] > MAX_FRAME_DIMENSION
                    or frame.shape[1] > MAX_FRAME_DIMENSION
                ):
                    raise ValidationError("Размер кадра для экспорта выходит за лимит.")
                total_pixels += int(frame.shape[0]) * int(frame.shape[1])
                total_memory += int(frame.nbytes)
            if total_pixels > MAX_FRAME_TOTAL_PIXELS:
                raise ValidationError("Суммарный размер кадров превышает лимит экспорта.")
            if total_memory > MAX_VIDEO_PREVIEW_BYTES:
                raise ValidationError("Память для экспорта видео превышает установленный лимит.")

            h, w = self.frames[0].shape[:2]
            fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
            output_path = Path(self.out_path)
            if not output_path.parent.is_dir():
                raise ValueError("Папка для сохранения видео не существует.")
            if output_path.parent.is_symlink() or (
                output_path.exists() and output_path.is_symlink()
            ):
                raise ValueError("Симлинки для выходного видео не поддерживаются.")
            fd, temporary_path = tempfile.mkstemp(
                prefix=".spinslicer-", suffix=".mp4", dir=str(output_path.parent)
            )
            os.close(fd)
            writer = cv2.VideoWriter(temporary_path, fourcc, self.fps, (w, h), isColor=True)
            if not writer.isOpened():
                raise RuntimeError("Не удалось открыть VideoWriter — проверьте кодек/путь сохранения.")

            total = len(self.frames)
            report_step = max(1, total // 30)

            for i, frame in enumerate(self.frames):
                if self._cancel_event.is_set():
                    self.cancelled.emit()
                    return
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
                if i % report_step == 0:
                    self.progress.emit(i / total, f"Кодирование кадра {i}/{total}...")

            writer.release()
            writer = None
            os.replace(temporary_path, output_path)
            published = True
            self.finished_ok.emit(str(output_path))
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=4)}")
        finally:
            if writer is not None:
                writer.release()
            if temporary_path is not None and not published:
                try:
                    Path(temporary_path).unlink(missing_ok=True)
                except OSError:
                    pass
