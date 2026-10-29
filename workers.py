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
import subprocess
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import trimesh
from PyQt6.QtCore import QThread, pyqtSignal

from constants import (
    DEFAULT_PROJECTOR_GAMMA,
    MAX_FRAME_DIMENSION,
    MAX_FRAME_TOTAL_PIXELS,
    MAX_NUM_FRAMES,
    MAX_PROJECTOR_GAMMA,
    MAX_VIDEO_PREVIEW_BYTES,
    MIN_PROJECTOR_GAMMA,
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

    def __init__(self, frames_dir: str, parent=None, subdir: str = ""):
        super().__init__(parent)
        self.frames_dir = frames_dir
        # Optional sub-folder of the run (e.g. the bare projection frames).
        self.subdir = subdir
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
            if self.subdir:
                folder = Path(frame_set.resolved_dir) / self.subdir
                if folder.is_symlink() or not folder.is_dir():
                    raise ValidationError(f"Папка кадров {self.subdir} не найдена.")
                paths = sorted(str(item) for item in folder.glob("frame_*.png") if not item.is_symlink())
                if len(paths) != frame_set.frame_count:
                    raise ValidationError(
                        f"В папке {self.subdir} {len(paths)} кадров, ожидалось {frame_set.frame_count}."
                    )

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


# H.264 quality for the exported MP4: near-lossless for smooth gray projections
# while still playing on ordinary yuv420p decoders.
H264_CRF = 10


# Gray bars that cover both ends of the range (0-16 and 235-255 are where a player that
# ignores the range flag clips), for checking a whole playback chain with a camera or a
# light meter.
CALIBRATION_LEVELS = (
    0, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224,
    232, 236, 240, 244, 248, 250, 252, 254, 255,
)

VIDEO_RANGES = ("full", "limited")


def calibration_ramp(width: int, height: int) -> np.ndarray:
    """One frame of vertical gray bars, one per CALIBRATION_LEVELS entry."""
    bars = len(CALIBRATION_LEVELS)
    edges = np.linspace(0, width, bars + 1).astype(int)
    image = np.zeros((height, width), dtype=np.uint8)
    for level, left, right in zip(CALIBRATION_LEVELS, edges[:-1], edges[1:]):
        image[:, left:right] = level
    return image


def _ffmpeg_executable() -> Optional[str]:
    """ffmpeg with libx264 shipped by imageio-ffmpeg, or None when it is missing.

    The pip builds of OpenCV cannot encode H.264, and their mp4v fallback is
    lossy enough to wreck gray levels, so MP4 export goes through ffmpeg.
    """
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # package or its bundled binary missing
        return None


class VideoExportWorker(QThread):
    """Кодирует уже собранные в памяти кадры в MP4 (H.264 через ffmpeg)."""

    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        frames: Sequence[Union[np.ndarray, str]],
        fps: float,
        out_path: str,
        parent=None,
        projector_gamma: float = DEFAULT_PROJECTOR_GAMMA,
        video_range: str = "full",
    ):
        """``frames`` are grayscale arrays or paths of PNG files.

        Paths are read one at a time at full resolution, which is how the file for
        the printer is made; arrays (like the downscaled preview) are used as given.
        """
        super().__init__(parent)
        self.frames = frames
        self.projector_gamma = projector_gamma
        self.video_range = video_range
        self.fps = fps
        self.out_path = out_path
        self._cancel_event = threading.Event()

    @staticmethod
    def _load(item: Union[np.ndarray, str]) -> np.ndarray:
        """One grayscale frame from an array or a PNG path."""
        if isinstance(item, np.ndarray):
            return item
        import cv2

        image = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValidationError(f"Не удалось прочитать кадр {Path(str(item)).name}.")
        return image

    def _encode_frames(self, write, prepared) -> bool:
        """Feed every prepared frame to ``write``; False when cancelled."""
        total = len(self.frames)
        report_step = max(1, total // 30)
        for i, item in enumerate(self.frames):
            if self._cancel_event.is_set():
                return False
            write(prepared(self._load(item)))
            if i % report_step == 0:
                self.progress.emit(i / total, f"Кодирование кадра {i}/{total}...")
        return True

    def _encode_h264(self, ffmpeg: str, path: str, w: int, h: int, prepared) -> bool:
        """Pipe raw gray frames into ffmpeg/libx264; False when cancelled."""
        command = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", repr(float(self.fps)),
            "-i", "-",
            # yuv420p needs even sides; pad with black (no light) instead of resampling.
            # Full range (the default): black stays 0 and white 255. In the default limited range the
            # background would be 16/255 (about 6 % light) all the way round the vial and
            # 36 gray levels would be lost; the pc flag tells decoders that the range is full.
            "-vf", f"pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=in_range=full:out_range={self.video_range}",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(H264_CRF), "-pix_fmt", "yuv420p",
            "-color_range", "pc" if self.video_range == "full" else "tv",
            "-movflags", "+faststart", "-f", "mp4", path,
        ]
        flags = 0x08000000 if sys.platform == "win32" else 0  # no console flash in the GUI
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors,
                creationflags=flags,
            )
            stdin = process.stdin
            assert stdin is not None
            try:
                done = self._encode_frames(lambda frame: stdin.write(frame.tobytes()), prepared)
                if not done:
                    process.kill()
                stdin.close()
                process.wait()
            except BrokenPipeError:
                process.wait()  # ffmpeg quit early; its exit code carries the error
                done = True
            except BaseException:
                process.kill()
                process.wait()
                raise
            if done and process.returncode != 0:
                errors.seek(0)
                detail = errors.read().decode("utf-8", "replace").strip()
                raise RuntimeError(f"ffmpeg не смог закодировать видео: {detail}")
        return done

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
            if self.video_range not in VIDEO_RANGES:
                raise ValidationError("video_range must be 'full' or 'limited'.")
            if not math.isfinite(float(self.fps)) or not 0.1 <= self.fps <= 240.0:
                raise ValidationError("FPS должен быть конечным числом от 0.1 до 240.")
            if len(self.frames) > MAX_NUM_FRAMES:
                raise ValidationError("Слишком много кадров для экспорта.")

            first = self._load(self.frames[0])
            first_h, first_w = first.shape[:2]
            total_pixels = 0
            total_memory = 0
            held: set[int] = set()
            for item in self.frames:
                if isinstance(item, np.ndarray):
                    frame = item
                    if frame.ndim != 2:
                        raise ValueError("Кадры для экспорта должны быть grayscale-массивами.")
                    frame_h, frame_w = frame.shape[:2]
                    if id(frame) not in held:  # a repeated array occupies memory once
                        held.add(id(frame))
                        total_memory += int(frame.nbytes)
                else:
                    frame_h, frame_w = first_h, first_w  # PNG files are streamed, not held
                if (
                    not frame_h
                    or not frame_w
                    or frame_h > MAX_FRAME_DIMENSION
                    or frame_w > MAX_FRAME_DIMENSION
                ):
                    raise ValidationError("Размер кадра для экспорта выходит за лимит.")
                total_pixels += int(frame_h) * int(frame_w)
            if total_pixels > MAX_FRAME_TOTAL_PIXELS:
                raise ValidationError("Суммарный размер кадров превышает лимит экспорта.")
            if total_memory > MAX_VIDEO_PREVIEW_BYTES:
                raise ValidationError("Память для экспорта видео превышает установленный лимит.")

            h, w = first_h, first_w
            # Gamma pre-compensation: a projector shows value v as (v/255)**gamma
            # of its peak light, so encode (v/255)**(1/gamma) to get linear dose.
            gamma = float(self.projector_gamma)
            if not math.isfinite(gamma) or not MIN_PROJECTOR_GAMMA <= gamma <= MAX_PROJECTOR_GAMMA:
                raise ValidationError(
                    f"Projector gamma must be between {MIN_PROJECTOR_GAMMA:g} and {MAX_PROJECTOR_GAMMA:g}."
                )
            lut = np.rint(255.0 * (np.arange(256) / 255.0) ** (1.0 / gamma)).astype(np.uint8)
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

            def prepared(frame: np.ndarray) -> np.ndarray:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                return np.ascontiguousarray(cv2.LUT(frame, lut))

            ffmpeg = _ffmpeg_executable()
            if ffmpeg is not None:
                done = self._encode_h264(ffmpeg, temporary_path, w, h, prepared)
            else:
                self.progress.emit(0.0, "ffmpeg недоступен: MP4 пишется кодеком mp4v с потерей качества.")
                writer = cv2.VideoWriter(
                    temporary_path, getattr(cv2, "VideoWriter_fourcc")(*"mp4v"), self.fps, (w, h),
                    isColor=True,
                )
                if not writer.isOpened():
                    raise RuntimeError("Не удалось открыть VideoWriter — проверьте кодек/путь сохранения.")
                mp4v = writer
                done = self._encode_frames(
                    lambda frame: mp4v.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)), prepared
                )
                writer.release()
                writer = None
            if not done:
                self.cancelled.emit()
                return

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
