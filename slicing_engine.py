"""
slicing_engine.py
==================
Математическое ядро Spin Slicer.

Этот модуль НИЧЕГО не знает о PyQt6 / PyVista / customtkinter — это чистая
математика и обработка изображений. Он принимает уже готовый (трансформированный)
trimesh.Trimesh и параметры, а сообщает о прогрессе через простой callback.

Сечения растеризуются в порядке [x, y, z]. Физический backend подбирает
неотрицательные проекции для относительной линейной дозы. Проверка дозы
использует экспортированные PNG; модель отверждения без калибровки не задаётся.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Tuple, cast

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes
from skimage.transform import radon
from skimage.transform import resize as sk_resize

from compute import MODES, resolve_profile, shutdown_taichi
from constants import (
    ANTI_ALIAS_FACTOR,
    BARE_PROJECTION_DIR,
    DEFAULT_ANGLE_OFFSET_DEG,
    DEFAULT_FRAME_RATE_HZ,
    PROJECTOR_FRAMES_DIR,
    VAT_HEIGHT_RATIO,
)
from frame_io import (
    FrameRepository,
    GenerationManifest,
    SliceMeta,
    save_manifest,
    save_meta,
    sha256_file,
)
from i18n import tr
from optics.geometry import detector_span_mm
from optics.sinogram_store import SinogramStore
from profiles import (
    MachineProfile,
    ProfileValidationError,
    ProjectorProfile,
    ResinProfile,
    VatProfile,
)
from validation import (
    ValidationError,
    ensure_directory,
    preflight_mesh,
    validate_slice_parameters,
    validate_stl_path,
)
from vam_backend import (
    DEFAULT_VAM_ENV_NAME,
    VAMToolboxUnavailable,
    optimize_sinograms,
)

ProgressCallback = Callable[[float, str], None]
# Sub-folder of a run holding the projection without vial compensation.
# PNG is lossless at every level; a low one writes the same pixels much faster.
PNG_COMPRESS_LEVEL = 1
CancelCheck = Callable[[], bool]


def _ordered_parallel(function: Callable[..., Any], arguments, workers: int):
    """Yield ``function(*args)`` in order, computed on a thread pool.

    When one call fails (or the consumer stops early) the calls that have not
    started are cancelled instead of being run to completion.
    """
    pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
    try:
        futures = [pool.submit(function, *args) for args in arguments]
        for future in futures:
            yield future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _radon_sinograms(
    slices: np.ndarray,
    angles: np.ndarray,
    grid_res: int,
    num_frames: int,
    report: ProgressCallback,
    cancelled: CancelCheck,
) -> np.ndarray:
    """Calculate the deterministic internal projection sequence with multi-threaded speed."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    num_layers = int(slices.shape[2])
    report_step = max(1, num_layers // 25)
    sinograms = np.zeros((num_layers, grid_res, num_frames), dtype=np.float32)

    max_workers = min(os.cpu_count() or 4, 16)

    def _process_slice(i: int):
        if cancelled():
            raise SliceCancelled()
        sino = radon(slices[:, :, i], theta=angles, circle=True)
        return i, sino

    done_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, sino in executor.map(_process_slice, range(num_layers)):
            if cancelled():
                raise SliceCancelled()
            if sino.shape != (grid_res, num_frames):
                raise ValueError(
                    f"Unexpected Radon shape {sino.shape}; expected {(grid_res, num_frames)}."
                )
            sinograms[i] = sino
            done_count += 1
            if done_count % report_step == 0 or done_count == num_layers:
                report(0.30 + 0.50 * (done_count / num_layers), f"Radon {done_count}/{num_layers}...")
    return sinograms


def _projection_gray(frame: np.ndarray, maximum: float, resin: ResinSettings) -> np.ndarray:
    """Common global gain and explicit gray cutoff, without percentile clipping."""
    relative = frame / maximum if maximum > 0 else np.zeros_like(frame)
    cutoff = resin.threshold / 100.0
    if cutoff >= 1:
        return np.zeros_like(frame)
    relative = np.maximum(relative - cutoff, 0) / (1 - cutoff)
    return np.clip(relative * (resin.intensity / 100.0) * resin.base_exposure, 0, 1)


def _exported_detector_row(frame: np.ndarray, row_index: int, grid_res: int) -> np.ndarray:
    """Average actual PNG pixels into detector bins, without a second uint8 rounding."""
    row = Image.fromarray(frame[row_index:row_index + 1].astype(np.float32))
    return np.asarray(row.resize((grid_res, 1), Image.Resampling.BOX)).ravel() / 255.0


def _relative_dose_fit(dose: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares shape fit in relative units, not a resin cure prediction."""
    values = np.asarray(dose, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if values.shape != target.shape or not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("Dose and target must have matching shapes and finite values.")
    square_sum = float(np.sum(values * values))
    scale = max(0.0, float(np.sum(values * target)) / square_sum) if square_sum > 0 else 0.0
    relative = values * scale
    rmse = float(np.sqrt(np.mean((relative - target)**2)))
    return relative.astype(np.float32), rmse, scale


@dataclass
class ResinSettings:
    """Параметры фотополимера."""
    base_exposure: float = 1.0      # базовое время засветки, с
    intensity: float = 100.0        # интенсивность источника света, %
    threshold: float = 0.0          # отсечение яркости кадров, %; не критическая доза смолы


@dataclass
class SliceParams:
    """Все параметры одного запуска генерации проекций."""
    diameter_mm: float
    grid_res: int
    output_res: int
    num_frames: int
    fill_holes: bool
    resin: ResinSettings
    output_dir: str
    projection_backend: str = "internal"
    optimizer_iterations: int = 8
    # Optional isolated Conda runtime for VAMToolbox.  The internal Radon
    # backend never needs these values.
    vam_conda_executable: str = ""
    vam_environment_name: str = DEFAULT_VAM_ENV_NAME
    # A valid nut is defined by its cavity. Keep internal voids by default so
    # every imported threaded STL follows the same path as any other model.
    preserve_internal_voids: bool = True
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ
    angle_offset_deg: float = DEFAULT_ANGLE_OFFSET_DEG
    angle_direction: int = 1
    container_mode: int = 0
    cuvette_outer_mm: float = 0.0
    cuvette_inner_mm: float = 0.0
    n_air: float = 1.0
    n_glass: float = 1.51
    n_liquid: float = 1.45
    n_resin: float = 1.50
    alpha: float = 0.0
    # Threshold-aware refinement after the least-squares fit: gives the part a
    # dose window above its surroundings instead of a blurred relative dose.
    dose_window: bool = False
    # Heuristic threshold of the window stage, in units where 1 is the target dose.
    window_level: float = 0.8
    window_steepness: float = 10.0
    # Also write the plain projection without vial compensation, for the
    # projector tab's "bare projection" view.
    save_bare_projection: bool = True
    # auto / economy / maximum: how the work is spread over this machine. It
    # changes speed and memory, never the computed result.
    compute_mode: str = "auto"

    def validate(self) -> None:
        """Validate values before they can determine array dimensions."""

        validate_slice_parameters(
            self.diameter_mm,
            self.grid_res,
            self.output_res,
            self.num_frames,
            self.resin,
            self.fill_holes,
            self.projection_backend,
            self.optimizer_iterations,
            self.preserve_internal_voids,
            self.frame_rate_hz,
            self.angle_offset_deg,
            self.angle_direction,
        )
        if not 0.1 <= float(self.window_level) <= 1.5:
            raise ValidationError("window_level must be between 0.1 and 1.5.")
        if not 1.0 <= float(self.window_steepness) <= 50.0:
            raise ValidationError("window_steepness must be between 1 and 50.")
        if self.compute_mode not in MODES:
            raise ValidationError(f"compute_mode must be one of: {', '.join(MODES)}.")
        if self.projection_backend == "physical":
            from optics.optimization import _validate_geometry
            try:
                _validate_geometry(
                    np.array([0.0]), self.grid_res, self.grid_res, self.diameter_mm,
                    self.cuvette_outer_mm, self.cuvette_inner_mm,
                    (self.n_air, self.n_glass, self.n_liquid, self.n_resin),
                    self.alpha, self.container_mode,
                )
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc


class SliceCancelled(Exception):
    """Поднимается, когда пользователь отменил генерацию из UI."""


def _machine_profile_payload(
    value: Any,
    diameter_mm: float,
    output_res: int,
    target_h: int,
    frame_rate_hz: float,
    angle_offset_deg: float,
    angle_direction: int,
) -> dict[str, Any]:
    """Return a validated machine profile for the generated job manifest."""

    try:
        if value is None:
            profile = MachineProfile(
                name="offline-unbound",
                projector=ProjectorProfile(
                    name="offline-frame-target",
                    width_px=output_res,
                    height_px=max(target_h, 16),
                    fps=frame_rate_hz,
                ),
                vat=VatProfile(
                    name="virtual-vat",
                    diameter_mm=diameter_mm,
                    height_mm=diameter_mm * VAT_HEIGHT_RATIO,
                ),
                angle_offset_deg=angle_offset_deg,
                angle_direction=angle_direction,
            )
        elif isinstance(value, MachineProfile):
            profile = value
        elif isinstance(value, dict):
            profile = MachineProfile.from_dict(value)
        else:
            raise ProfileValidationError("machine_profile must be an object or MachineProfile.")
        return profile.to_dict()
    except ProfileValidationError as exc:
        raise ValidationError(f"Invalid machine profile: {exc}") from exc


def _resin_profile_payload(value: Any, resin: ResinSettings) -> dict[str, Any]:
    """Return a validated material profile for the generated job manifest."""

    try:
        if value is None:
            profile = ResinProfile(
                base_exposure_s=resin.base_exposure,
                intensity_pct=resin.intensity,
                threshold_pct=resin.threshold,
            )
        elif isinstance(value, ResinProfile):
            profile = value
        elif isinstance(value, dict):
            profile = ResinProfile.from_dict(value)
        else:
            raise ProfileValidationError("resin_profile must be an object or ResinProfile.")
        return profile.to_dict()
    except ProfileValidationError as exc:
        raise ValidationError(f"Invalid resin profile: {exc}") from exc


class SlicingEngine:
    """Mesh sections, projection planning and verified frame export."""

    @staticmethod
    def run(
        mesh: trimesh.Trimesh,
        params: SliceParams,
        progress_cb: Optional[ProgressCallback] = None,
        is_cancelled: Optional[CancelCheck] = None,
        manifest_context: Optional[dict[str, Any]] = None,
    ) -> Tuple[int, str]:
        """Slice one mesh; scratch files, the GPU and caches are released here, in
        the thread that did the work, whether the job finishes, fails or is cancelled."""
        state: dict[str, Any] = {"profile": None, "cleanups": []}
        try:
            return SlicingEngine._run(mesh, params, progress_cb, is_cancelled, manifest_context, state)
        finally:
            for cleanup in state["cleanups"]:
                try:
                    cleanup()
                except Exception:  # cleaning up must not hide the real error
                    pass
            shutdown_taichi()
            profile = state["profile"]
            if profile is None or profile.mode != "maximum":
                from optics.optimization import _matrix_cache

                # Hundreds of MB; economy gives them back, maximum keeps them so that
                # repeating a job with the same geometry skips the ray tracing.
                _matrix_cache.clear()

    @staticmethod
    def _run(
        mesh: trimesh.Trimesh,
        params: SliceParams,
        progress_cb: Optional[ProgressCallback] = None,
        is_cancelled: Optional[CancelCheck] = None,
        manifest_context: Optional[dict[str, Any]] = None,
        state: Optional[dict[str, Any]] = None,
    ) -> Tuple[int, str]:
        """Считает проекции для одного (уже финально трансформированного) меша.

        Возвращает (num_frames, out_dir). Бросает SliceCancelled при отмене
        и обычные исключения при ошибках геометрии/параметров.
        """

        def report(frac: float, msg: str) -> None:
            if progress_cb is not None:
                progress_cb(frac, msg)

        def cancelled() -> bool:
            return is_cancelled() if is_cancelled is not None else False

        report(0.02, "Подготовка модели...")

        params.validate()
        preflight = preflight_mesh(
            mesh,
            params,
            vat_height_mm=params.diameter_mm * VAT_HEIGHT_RATIO,
        )
        if not preflight.ok:
            raise ValidationError("\n".join(preflight.errors))
        ensure_directory(params.output_dir)

        state = state if state is not None else {"profile": None, "cleanups": []}
        profile = resolve_profile(params.compute_mode, use_gpu=params.projection_backend == "physical")
        state["profile"] = profile
        report(0.03, profile.describe())
        if params.projection_backend == "physical" and params.save_bare_projection and not profile.save_bare:
            report(0.03, tr("Экономный режим: прямая проекция не создаётся."))

        diameter_mm = params.diameter_mm
        grid_res = params.grid_res
        output_res = params.output_res
        num_frames = params.num_frames
        fill_holes = params.fill_holes
        preserve_internal_voids = params.preserve_internal_voids
        resin = params.resin
        projection_span = (
            detector_span_mm(params.cuvette_outer_mm, params.container_mode)
            if params.projection_backend == "physical" else diameter_mm
        )

        z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
        # Decouple Z-resolution from X/Y grid_res to eliminate Z stair-stepping.
        # Slice at the projector's native vertical pixel pitch for maximum smoothness!
        pitch = min(diameter_mm / grid_res, projection_span / output_res)
        z_levels = np.arange(z_min + pitch / 2.0, z_max, pitch)
        nz = len(z_levels)
        if nz == 0:
            raise ValueError("Модель слишком плоская для заданного разрешения.")

        aa_size = grid_res * ANTI_ALIAS_FACTOR
        scale_2d = aa_size / diameter_mm
        offset = aa_size / 2.0

        slices = np.zeros((grid_res, grid_res, nz), dtype=np.uint8)  # value / 255

        report(0.05, "Оптическая нарезка сечений...")
        report_step = max(1, nz // 25)

        # --- Бронебойный послойный метод: не крашится на грязных сетках ---
        def render_layer(i: int, z: float) -> Optional[np.ndarray]:
            if cancelled():
                raise SliceCancelled()

            slice_3d = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])

            if slice_3d is not None:
                matrix_2d = np.array([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -z],
                    [0.0, 0.0, 0.0, 1.0],
                ])

                try:
                    to_2d = getattr(slice_3d, "to_2D", None)
                    if callable(to_2d):
                        slice_2d, _ = to_2d(to_2D=matrix_2d)
                    else:  # trimesh < 5 compatibility
                        slice_2d, _ = slice_3d.to_planar(to_2D=matrix_2d)
                except Exception as exc:
                    raise ValidationError(
                        f"Could not rasterize layer {i} at z={z:.6g} mm: {exc}"
                    ) from exc

                img = Image.new("L", (aa_size, aa_size), color=0)
                draw = ImageDraw.Draw(img)

                # 1. Заливаем честные замкнутые полигоны
                polys = getattr(slice_2d, "polygons_full", None)
                if polys is None:
                    polys = getattr(slice_2d, "polygons_closed", [])

                void_image = Image.new("1", (aa_size, aa_size), color=0)
                void_draw = ImageDraw.Draw(void_image)
                for poly in polys:
                    ext = np.asarray(poly.exterior.coords)
                    draw.polygon(
                        list(zip(ext[:, 0] * scale_2d + offset, ext[:, 1] * scale_2d + offset)),
                        fill=255,
                    )
                    for interior in poly.interiors:
                        ints = np.asarray(interior.coords)
                        coordinates = list(zip(ints[:, 0] * scale_2d + offset, ints[:, 1] * scale_2d + offset))
                        draw.polygon(
                            coordinates,
                            fill=0,
                        )
                        void_draw.polygon(coordinates, fill=1)

                # Open paths are a fallback only. Outlining valid polygons by
                # a fixed 0.5 mm used to enlarge solids and shrink their holes.
                for entity in ([] if len(polys) else slice_2d.entities):
                    discrete_pts = entity.discrete(slice_2d.vertices)
                    if len(discrete_pts) > 1:
                        px = discrete_pts[:, 0] * scale_2d + offset
                        py = discrete_pts[:, 1] * scale_2d + offset
                        draw.line(list(zip(px, py)), fill=255, width=1)

                arr = np.array(img) > 127
                preserved_voids = np.asarray(void_image, dtype=bool) & ~arr

                # 3. Опциональный ремонт сетки (для монолитных / сломанных STL)
                if fill_holes:
                    # Используем OpenCV для молниеносной скорости (в 65 раз быстрее scipy)
                    arr_u8 = (arr * 255).astype(np.uint8)
                    kernel = np.ones((3, 3), np.uint8)
                    import cv2
                    arr_u8 = np.asarray(cv2.morphologyEx(arr_u8, cv2.MORPH_CLOSE, kernel), dtype=np.uint8)

                    if not preserve_internal_voids:
                        arr = binary_fill_holes(arr_u8 > 127)
                    else:
                        arr = arr_u8 > 127
                        arr[preserved_voids] = False

                # Уменьшаем с антиалиасингом для мягких краёв
                img_filled = Image.fromarray((arr * 255).astype(np.uint8))
                img_resized = img_filled.resize((grid_res, grid_res), Image.Resampling.LANCZOS)
                return np.array(img_resized, dtype=np.uint8).T
            return None

        # Build trimesh's lazy caches once before the layers are cut in parallel.
        mesh.section(plane_origin=[0, 0, float(z_levels[0])], plane_normal=[0, 0, 1])
        layer_arguments = ((i, float(z)) for i, z in enumerate(z_levels))
        for i, layer in enumerate(_ordered_parallel(render_layer, layer_arguments, profile.slicing_threads)):
            if layer is not None:
                slices[:, :, i] = layer
            if i % report_step == 0:
                report(0.05 + 0.25 * (i / nz), f"Нарезка слоя {i}/{nz}...")

        # --- Проекционный backend ------------------------------------------
        base_angles = np.linspace(0.0, 360.0, num_frames, endpoint=False)
        angles = (
            params.angle_offset_deg + params.angle_direction * base_angles
        ) % 360.0
        backend = params.projection_backend
        # What actually produced the frames; "auto" may fall back below.
        backend_used = "internal" if backend == "auto" else backend

        def float_slices() -> np.ndarray:
            return slices.astype(np.float32) / np.float32(255.0)

        bare_sinograms: Optional[np.ndarray] = None
        sinograms: Optional[np.ndarray] = None
        device_status: dict[str, Any] = {}
        store: Optional[SinogramStore] = None
        if backend == "physical":
            report(0.30, "Физическая оптика: трассировка лучей и оптимизация...")
            from optics.optimization import optimize_projections_sirt_volume, project_volume
            from optics.sinogram_store import SinogramStore
            angles_rad = np.deg2rad(angles)
            L_out = params.cuvette_outer_mm
            L_in = params.cuvette_inner_mm
            if params.save_bare_projection and profile.save_bare:
                report(0.30, "Расчёт прямой проекции...")
                bare_sinograms = project_volume(
                    slices, angles_rad, L_out=L_out, L_in=L_in, grid_res=grid_res,
                    grid_span_mm=params.diameter_mm, container_mode=params.container_mode,
                    cancelled_cb=cancelled, profile=profile,
                )

            store = SinogramStore(
                nz, grid_res, num_frames,
                directory=str(params.output_dir) if profile.disk_sinograms else None,
            )
            state["cleanups"].append(store.close)
            optimize_projections_sirt_volume(
                slices,
                angles_rad,
                L_out=L_out,
                L_in=L_in,
                n_air=params.n_air,
                n_glass=params.n_glass,
                n_liquid=params.n_liquid,
                n_resin=params.n_resin,
                alpha=params.alpha,
                grid_res=grid_res,
                grid_span_mm=params.diameter_mm,
                iterations=params.optimizer_iterations,
                container_mode=params.container_mode,
                report_cb=report,
                cancelled_cb=cancelled,
                window_iterations=params.optimizer_iterations if params.dose_window else 0,
                profile=profile, sink=store.add_block, status=device_status,
                window_level=params.window_level, window_steepness=params.window_steepness,
            )
        elif backend in {"auto", "vamtoolbox"}:
            report(0.30, "Optimizing projections with VAMToolbox CAL...")
            try:
                # VAMToolbox consumes the complete voxel target and returns a
                # projection sequence.  The adapter normalizes its layout to
                # the same (z, detector, angle) convention used by the
                # internal Radon path.
                sinograms = optimize_sinograms(
                    float_slices(),
                    angles,
                    iterations=params.optimizer_iterations,
                    conda_executable=params.vam_conda_executable or None,
                    environment_name=params.vam_environment_name or DEFAULT_VAM_ENV_NAME,
                )
                backend_used = "vamtoolbox"
                report(0.78, "VAMToolbox projection optimization complete.")
            except VAMToolboxUnavailable as exc:
                if backend == "vamtoolbox":
                    raise ValidationError(str(exc)) from exc
                report(0.30, "VAMToolbox is unavailable; using internal Radon backend.")
                sinograms = _radon_sinograms(
                    float_slices(), angles, grid_res, num_frames, report, cancelled
                )
            except Exception:
                if backend == "vamtoolbox":
                    raise
                report(0.30, "VAMToolbox failed; using internal Radon backend.")
                sinograms = _radon_sinograms(
                    float_slices(), angles, grid_res, num_frames, report, cancelled
                )
        else:
            report(0.30, "Calculating Radon projections...")
            sinograms = _radon_sinograms(
                float_slices(), angles, grid_res, num_frames, report, cancelled
            )

        if store is not None:
            global_max = store.maximum
        else:
            assert sinograms is not None
            global_max = float(sinograms.max()) if sinograms.size else 1.0
        if global_max <= 0:
            global_max = 1.0

        out_dir = FrameRepository.create_run_dir(params.output_dir)
        model_height_mm = z_max - z_min
        target_h = max(int(round(output_res * model_height_mm / projection_span)), 1)
        context = manifest_context or {}
        machine_profile = _machine_profile_payload(
            context.get("machine_profile"),
            diameter_mm=diameter_mm,
            output_res=output_res,
            target_h=target_h,
            frame_rate_hz=params.frame_rate_hz,
            angle_offset_deg=params.angle_offset_deg,
            angle_direction=params.angle_direction,
        )
        projector_gamma = float((machine_profile.get("projector") or {}).get("gamma", 1.0))
        # With a non-linear projector the gamma is applied to the float frame, before the
        # single 8-bit rounding, and the result is kept next to the linear PNGs.
        encode_gamma = abs(projector_gamma - 1.0) > 1e-9
        projector_dir = os.path.join(out_dir, PROJECTOR_FRAMES_DIR)
        if encode_gamma:
            os.makedirs(projector_dir, exist_ok=True)
        mid_z = nz // 2
        mid_frame_row = int(np.clip(round((nz - mid_z - 0.5) * target_h / nz - 0.5), 0, target_h - 1))
        exported_mid_sinograms = np.empty((grid_res, num_frames), dtype=np.float32)
        bare_dir = os.path.join(out_dir, BARE_PROJECTION_DIR)
        bare_max = 1.0
        if bare_sinograms is not None:
            os.makedirs(bare_dir, exist_ok=True)
            bare_max = float(bare_sinograms.max()) or 1.0

        report(0.80, "Сохранение кадров...")
        save_step = max(1, num_frames // 20)
        def export_frame(i: int) -> None:
            if cancelled():
                raise SliceCancelled()
            if store is not None:
                frame = np.flipud(store.frame(i))
            else:
                assert sinograms is not None
                frame = np.flipud(sinograms[:, :, i])
            normalized = _projection_gray(frame, global_max, resin)
            # Resize floating gray values before the single 8-bit quantization.
            resized = sk_resize(
                normalized, (target_h, output_res), order=1, mode="edge",
                anti_aliasing=True, preserve_range=True,
            )
            output = np.rint(np.clip(resized, 0, 1) * 255).astype(np.uint8)
            exported_mid_sinograms[:, i] = _exported_detector_row(output, mid_frame_row, grid_res)
            Image.fromarray(output).save(
                os.path.join(out_dir, f"frame_{i:04d}.png"), compress_level=PNG_COMPRESS_LEVEL
            )
            if encode_gamma:
                encoded = np.rint(np.clip(resized, 0, 1) ** (1.0 / projector_gamma) * 255).astype(np.uint8)
                Image.fromarray(encoded).save(
                    os.path.join(projector_dir, f"frame_{i:04d}.png"), compress_level=PNG_COMPRESS_LEVEL
                )
            if bare_sinograms is not None:
                bare_gray = _projection_gray(np.flipud(bare_sinograms[:, :, i]), bare_max, resin)
                bare_resized = sk_resize(
                    bare_gray, (target_h, output_res), order=1, mode="edge",
                    anti_aliasing=True, preserve_range=True,
                )
                Image.fromarray(np.rint(np.clip(bare_resized, 0, 1) * 255).astype(np.uint8)).save(
                    os.path.join(bare_dir, f"frame_{i:04d}.png"), compress_level=PNG_COMPRESS_LEVEL
                )
        for done, _ in enumerate(
            _ordered_parallel(export_frame, ((i,) for i in range(num_frames)), profile.io_threads)
        ):
            if done % save_step == 0:
                report(0.80 + 0.19 * (done / num_frames), f"Сохранение кадра {done}/{num_frames}...")

        if store is not None:
            store.close()
        if cancelled():
            raise SliceCancelled()
        # Reconstruct actual saved drive values, including user gain, cutoff,
        # resampling, saturation and PNG quantization.
        if backend == "physical":
            from optics.optimization import build_system_matrix
            _, adjoint = build_system_matrix(
                angles_rad, num_s=grid_res, grid_res=grid_res, grid_span_mm=diameter_mm,
                L_out=params.cuvette_outer_mm, L_in=params.cuvette_inner_mm,
                n_air=params.n_air, n_glass=params.n_glass, n_liquid=params.n_liquid,
                n_resin=params.n_resin, alpha=params.alpha, container_mode=params.container_mode,
                cancelled_cb=cancelled, lean_transpose=profile.lean_memory,
            )
            dose_mid = (adjoint @ exported_mid_sinograms.T.ravel()).reshape(grid_res, grid_res)
        else:
            from skimage.transform import iradon
            # Optical exposure accumulates positive frames; a ramp-filtered CT
            # inversion would hide backprojection blur and is not delivered dose.
            dose_mid = iradon(exported_mid_sinograms, theta=angles, circle=True, filter_name=None)
        target_mid = slices[:, :, mid_z].astype(np.float32) / np.float32(255.0)
        relative_dose, dose_rmse, dose_scale = _relative_dose_fit(dose_mid, target_mid)
        # Numeric artifacts allow the score to be reproduced independently.
        np.save(os.path.join(out_dir, "reconstruction_dose.npy"), dose_mid)
        np.save(os.path.join(out_dir, "reconstruction_target.npy"), target_mid)
        from matplotlib import colormaps
        from scipy.ndimage import binary_dilation, binary_erosion

        # Show the dose against its own peak and outline the target on top, so
        # it is obvious which pixels were supposed to be solid.
        peak = float(dose_mid.max())
        shown = np.clip(dose_mid / peak, 0, 1) if peak > 0 else np.zeros_like(dose_mid)
        rgb = np.rint(colormaps["viridis"](np.flipud(shown.T))[:, :, :3] * 255).astype(np.uint8)
        part = target_mid > 0.5
        outline = part & ~binary_erosion(part)
        rgb[np.flipud(outline.T)] = (255, 70, 70)
        Image.fromarray(rgb).save(os.path.join(out_dir, "reconstruction_preview.png"))
        # Window: weakest dose inside the part vs strongest dose well outside it.
        core = binary_erosion(part, iterations=2)
        far = ~binary_dilation(part, iterations=2)
        part_min = float(dose_mid[core].min()) if core.any() else float("nan")
        background_max = float(dose_mid[far].max()) if far.any() else float("nan")
        contrast = part_min / background_max if background_max > 0 else float("nan")
        with open(os.path.join(out_dir, "dose_stats.txt"), "w", encoding="utf-8") as handle:
            handle.write(
                f"rmse={dose_rmse:.8f}\nloss={np.linalg.norm(relative_dose-target_mid):.8f}\n"
                f"relative_scale={dose_scale:.10g}\nsource=exported_png\nmodel=relative_linear_dose\n"
                f"part_core_min={part_min:.8g}\nbackground_max={background_max:.8g}\n"
                f"window_contrast={contrast:.6g}\n"
            )

        # Metadata становится видимым только после того, как все PNG уже
        # записаны. Manifest публикуется после проверки полного frame-set.
        save_meta(out_dir, SliceMeta(
            diameter_mm=diameter_mm, grid_res=grid_res, output_res=output_res,
            num_frames=num_frames, fill_holes=fill_holes,
            resin_base_exposure=resin.base_exposure, resin_intensity=resin.intensity,
            resin_threshold=resin.threshold,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ))

        # Publishing the complete manifest is the commit point for a run. A
        # direct caller of SlicingEngine therefore gets the same safe output
        # contract as the GUI worker, while optional context records the STL
        # hash and the ModelNode transform when those are available.
        if cancelled():
            raise SliceCancelled()
        source_path = str(context.get("source_path", ""))
        source_hash = ""
        if source_path:
            validate_stl_path(source_path)
            source_hash = sha256_file(source_path)
        if cancelled():
            raise SliceCancelled()
        transform = context.get("transform_matrix")
        if not isinstance(transform, list):
            transform = np.eye(4, dtype=np.float64).tolist()
        frame_set = FrameRepository.validate(out_dir, require_complete=False)
        source_name = str(
            context.get(
                "source_name",
                os.path.basename(source_path) if source_path else "",
            )
        )
        resin_profile = _resin_profile_payload(context.get("resin_profile"), resin)
        save_manifest(
            out_dir,
            GenerationManifest(
                source_sha256=source_hash,
                source_name=source_name,
                units=str(context.get("units", "mm")),
                transform_matrix=cast(list[list[float]], transform),
                slice_parameters={
                    "diameter_mm": params.diameter_mm,
                    "grid_res": params.grid_res,
                    "output_res": params.output_res,
                    "num_frames": params.num_frames,
                    "fill_holes": params.fill_holes,
                    "projection_backend": params.projection_backend,
                    "projection_backend_used": backend_used,
                    "dose_window": params.dose_window and backend == "physical",
                    "compute": {**profile.as_manifest(), **device_status},
                    "bare_projection_dir": BARE_PROJECTION_DIR if bare_sinograms is not None else "",
                    "projector_frames_dir": PROJECTOR_FRAMES_DIR if encode_gamma else "",
                    "projector_gamma": projector_gamma,
                    "window_level": params.window_level,
                    "window_steepness": params.window_steepness,
                    "optimizer_iterations": params.optimizer_iterations,
                    "raster_layers": nz,
                    "voxel_pitch_z_mm": pitch,
                    "optics": {
                        "container_mode": params.container_mode,
                        "outer_mm": params.cuvette_outer_mm,
                        "inner_mm": params.cuvette_inner_mm,
                        "n_air": params.n_air,
                        "n_glass": params.n_glass,
                        "n_liquid": params.n_liquid,
                        "n_resin": params.n_resin,
                        "alpha_per_mm": params.alpha,
                        "voxel_grid_span_mm": diameter_mm,
                        "detector_span_mm": projection_span,
                        "detector_spacing_mm": projection_span / grid_res,
                        "output_pixel_pitch_mm": projection_span / output_res,
                        "dose_model": "relative_linear",
                    },
                    "preserve_internal_voids": params.preserve_internal_voids,
                    "frame_rate_hz": params.frame_rate_hz,
                    "angle_offset_deg": params.angle_offset_deg,
                    "angle_direction": params.angle_direction,
                    "model_type": str(context.get("model_type", "stl")),
                    "model_parameters": context.get("model_parameters", {}),
                    "resin": {
                        "base_exposure": params.resin.base_exposure,
                        "intensity": params.resin.intensity,
                        "threshold": params.resin.threshold,
                    },
                },
                frame_count=frame_set.frame_count,
                frame_size=(frame_set.width, frame_set.height),
                machine_profile=machine_profile,
                resin_profile=resin_profile,
                frame_schedule={
                    "angles_deg": [float(angle) for angle in angles],
                    "frame_duration_s": 1.0 / params.frame_rate_hz,
                    "frame_rate_hz": params.frame_rate_hz,
                },
                complete=True,
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )

        report(1.0, "Готово.")
        return num_frames, out_dir
