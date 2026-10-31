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
VIDEO_PREVIEW_MAX_DIM = 960  # preview copies only; the print screen knows the true frame size


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
            try:
                # Build the viewport copy here, off the UI thread.
                from viewport import prepare_display_mesh
                mesh.metadata["spinslicer_display"] = prepare_display_mesh(mesh)
            except Exception:
                pass

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
        calibration: Any = None,
    ):
        super().__init__(parent)
        self.mesh = mesh
        self.params = params
        self.manifest_context = manifest_context or {}
        self.calibration = calibration
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()
        flag = getattr(self, "_cancel_flag", None)
        if flag:
            try:
                open(flag, "w").close()
            except OSError:
                pass

    def run(self) -> None:
        if os.environ.get("SPINSLICER_INPROCESS") == "1":
            self._run_in_process()
        else:
            self._run_child()

    def _run_in_process(self) -> None:
        try:
            num_frames, out_dir = SlicingEngine.run(
                self.mesh,
                self.params,
                progress_cb=lambda frac, msg: self.progress.emit(frac, msg),
                is_cancelled=self._cancel_event.is_set,
                manifest_context=self.manifest_context,
                calibration=self.calibration,
            )
            self.finished_ok.emit(num_frames, out_dir)
        except SliceCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=6)}")

    @staticmethod
    def _child_command(job: str, flag: str) -> list[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, "--engine-worker", job, flag]
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_process.py")
        return [sys.executable, script, job, flag]

    def _run_child(self) -> None:
        """The engine in its own process: the UI keeps the GIL to itself and stays fluid."""
        import json
        import pickle
        import subprocess

        workdir = tempfile.mkdtemp(prefix="spinslicer-job-")
        job = os.path.join(workdir, "job.pkl")
        self._cancel_flag = os.path.join(workdir, "cancel")
        if self._cancel_event.is_set():
            open(self._cancel_flag, "w").close()
        process = None
        try:
            with open(job, "wb") as handle:
                pickle.dump((self.mesh, self.params, self.manifest_context, self.calibration), handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            flags = 0
            if sys.platform == "win32":
                flags = 0x08000000 | 0x00004000   # no console window, below-normal priority
            process = subprocess.Popen(self._child_command(job, self._cancel_flag), stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                                       creationflags=flags, bufsize=1)
            assert process.stdout is not None
            outcome: dict[str, Any] = {}
            for line in process.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if "p" in event:
                    self.progress.emit(float(event["p"]), str(event.get("m", "")))
                else:
                    outcome = event
            process.wait()
            if "done" in outcome:
                n, out_dir = outcome["done"]
                self.finished_ok.emit(int(n), str(out_dir))
            elif "cancelled" in outcome or self._cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(outcome.get("error") or f"Процесс расчёта завершился с кодом {process.returncode}."))
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=6)}")
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            for name in (job, self._cancel_flag):
                try:
                    os.remove(name)
                except OSError:
                    pass
            try:
                os.rmdir(workdir)
            except OSError:
                pass


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


# H.264 quality of the exported MP4: near-lossless for smooth gray projections while
# still playing on ordinary yuv420p decoders.
H264_CRF = 10

# Gray bars that cover both ends of the range (0-16 and 235-255 are where a player that
# ignores the range flag clips), to check the whole playback chain with a camera or a
# light meter and to measure the projector gamma.
CALIBRATION_LEVELS = (
    0, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224,
    232, 236, 240, 244, 248, 250, 252, 254, 255,
)


def calibration_ramp(width: int, height: int) -> np.ndarray:
    """One frame of vertical gray bars, one per CALIBRATION_LEVELS entry."""
    edges = np.linspace(0, width, len(CALIBRATION_LEVELS) + 1).astype(int)
    image = np.zeros((height, width), dtype=np.uint8)
    for level, left, right in zip(CALIBRATION_LEVELS, edges[:-1], edges[1:]):
        image[:, left:right] = level
    return image


def _ffmpeg_executable() -> Optional[str]:
    """ffmpeg with libx264: the one bundled by imageio-ffmpeg, else the system one.

    The pip builds of OpenCV cannot encode H.264, and their mp4v fallback is lossy
    enough to wreck gray levels, so MP4 export goes through ffmpeg when it can.
    """
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # package or its bundled binary missing
        import shutil

        return shutil.which("ffmpeg")


class VideoExportWorker(QThread):
    """Encodes frames into an MP4 for the projector: H.264 via ffmpeg, full gray range.

    ``frames`` are grayscale arrays or paths of PNG files. Paths are read one at a time
    at full resolution (the file for the printer); arrays are used as given.
    """

    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, frames: Sequence[Union[np.ndarray, str]], fps: float, out_path: str,
                 parent=None, video_range: str = "full", canvas: Optional[Any] = None,
                 lut: Optional[np.ndarray] = None):
        super().__init__(parent)
        self.frames = frames
        # Gray → gray table (projector brightness, video_tab.brightness_lut) or None.
        self.lut = lut
        # projector_canvas.CanvasGeometry: place every frame 1:1 on the projector's
        # full screen, so a player shows it pixel for pixel instead of stretching a strip.
        self.canvas = canvas
        self.fps = fps
        self.out_path = out_path
        self.video_range = video_range
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    @staticmethod
    def _load(item: Union[np.ndarray, str]) -> np.ndarray:
        if isinstance(item, np.ndarray):
            return item
        import cv2

        image = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValidationError(f"Не удалось прочитать кадр {Path(str(item)).name}.")
        return image

    def _feed(self, write: Any, size: Tuple[int, int]) -> bool:
        """Every frame, resized to `size` (w, h) when needed, into `write`; False if cancelled."""
        import cv2

        total = len(self.frames)
        report_step = max(1, total // 30)
        for i, item in enumerate(self.frames):
            if self._cancel_event.is_set():
                return False
            frame = self._load(item)
            if frame.ndim != 2:
                raise ValueError("Кадры для экспорта должны быть grayscale-массивами.")
            if self.lut is not None:
                frame = self.lut[np.asarray(frame, dtype=np.uint8)]
            if self.canvas is not None:
                from projector_canvas import compose
                frame = compose(frame, self.canvas)
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            write(np.ascontiguousarray(frame, dtype=np.uint8))
            if i % report_step == 0:
                self.progress.emit(i / total, f"Кодирование кадра {i}/{total}...")
        return True

    def _encode_h264(self, ffmpeg: str, path: str, w: int, h: int) -> bool:
        import subprocess
        import sys

        command = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", repr(float(self.fps)),
            "-i", "-",
            # yuv420p needs even sides: pad with black (no light) instead of resampling.
            # Full range keeps black at 0 and white at 255; limited range would light the
            # whole vial at 16/255 and lose 36 gray levels. The pc flag tells players.
            "-vf", f"pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=in_range=full:out_range={self.video_range}",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(H264_CRF), "-pix_fmt", "yuv420p",
            "-color_range", "pc" if self.video_range == "full" else "tv",
            "-movflags", "+faststart", "-f", "mp4", path,
        ]
        flags = 0x08000000 if sys.platform == "win32" else 0  # no console flash in the GUI
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=errors, creationflags=flags)
            stdin = process.stdin
            assert stdin is not None
            try:
                done = self._feed(lambda frame: stdin.write(frame.tobytes()), (w, h))
                if not done:
                    process.kill()
                stdin.close()
                process.wait()
            except BrokenPipeError:
                process.wait()            # ffmpeg quit early; its exit code carries the error
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

    def run(self) -> None:
        try:
            import cv2
        except ImportError:
            self.failed.emit("Не установлен OpenCV. Выполните: pip install opencv-python-headless")
            return

        temporary_path: Optional[str] = None
        published = False
        writer = None
        try:
            if not self.frames:
                raise ValueError("Нет кадров для сохранения.")
            if not math.isfinite(float(self.fps)) or not 0.1 <= self.fps <= 240.0:
                raise ValidationError("FPS должен быть конечным числом от 0.1 до 240.")
            if len(self.frames) > MAX_NUM_FRAMES:
                raise ValidationError("Слишком много кадров для экспорта.")
            if self.video_range not in ("full", "limited"):
                raise ValidationError("video_range must be 'full' or 'limited'.")
            first = self._load(self.frames[0])
            if first.ndim != 2:
                raise ValueError("Кадры для экспорта должны быть grayscale-массивами.")
            h, w = first.shape[:2]
            if not h or not w or h > MAX_FRAME_DIMENSION or w > MAX_FRAME_DIMENSION:
                raise ValidationError("Размер кадра для экспорта выходит за лимит.")
            if h * w * len(self.frames) > MAX_FRAME_TOTAL_PIXELS:
                raise ValidationError("Суммарный размер кадров превышает лимит экспорта.")
            if self.canvas is not None:
                # Frames are streamed one by one, so only the screen size is limited here.
                w, h = int(self.canvas.width), int(self.canvas.height)
                if not 16 <= w <= MAX_FRAME_DIMENSION or not 16 <= h <= MAX_FRAME_DIMENSION:
                    raise ValidationError("Разрешение проектора вне допустимого диапазона.")

            output_path = Path(self.out_path)
            if not output_path.parent.is_dir():
                raise ValueError("Папка для сохранения видео не существует.")
            if output_path.parent.is_symlink() or (output_path.exists() and output_path.is_symlink()):
                raise ValueError("Симлинки для выходного видео не поддерживаются.")
            fd, temporary_path = tempfile.mkstemp(prefix=".spinslicer-", suffix=".mp4",
                                                  dir=str(output_path.parent))
            os.close(fd)

            ffmpeg = _ffmpeg_executable()
            if ffmpeg is not None:
                done = self._encode_h264(ffmpeg, temporary_path, w, h)
            else:
                self.progress.emit(0.0, "ffmpeg недоступен: MP4 пишется кодеком mp4v с потерей качества.")
                fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
                writer = cv2.VideoWriter(temporary_path, fourcc, self.fps, (w, h), isColor=True)
                if not writer.isOpened():
                    raise RuntimeError("Не удалось открыть VideoWriter — проверьте кодек/путь сохранения.")
                done = self._feed(lambda frame: writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)), (w, h))
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
