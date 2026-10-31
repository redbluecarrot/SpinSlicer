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

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Tuple, cast

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes
from skimage.transform import radon
from skimage.transform import resize as sk_resize

from constants import (
    ANTI_ALIAS_FACTOR,
    DEFAULT_ANGLE_OFFSET_DEG,
    DEFAULT_FRAME_RATE_HZ,
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
from optics.geometry import detector_span_mm
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
CancelCheck = Callable[[], bool]


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


def _exported_detector_row(frame: np.ndarray, row_index: int, grid_res: int,
                           gamma: float = 1.0) -> np.ndarray:
    """Average the light of actual PNG pixels into detector bins.

    Gray levels are converted to displayed relative irradiance (g/255)^gamma
    first, because the projector's light, not its gray code, adds up.
    """
    light = (frame[row_index:row_index + 1].astype(np.float32) / 255.0) ** np.float32(gamma)
    row = Image.fromarray(light)
    return np.asarray(row.resize((grid_res, 1), Image.Resampling.BOX)).ravel()


def _encode_gray(relative: np.ndarray, gamma: float) -> np.ndarray:
    """Relative irradiance -> 8-bit gray code for a projector with this gamma."""
    return np.rint(np.clip(relative, 0, 1) ** (1.0 / gamma) * 255).astype(np.uint8)


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
    optimizer_iterations: int = 30
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
    # Display and material physics for a ready-to-print exposure plan.
    # Zero means "unknown"; then no absolute print time is computed.
    projector_gamma: float = 1.0            # displayed light = (gray/255) ** gamma
    projector_irradiance_mw_cm2: float = 0.0  # full white, at the vial axis
    projector_contrast: float = 0.0         # white/black ratio; 0 = ideal black
    projector_throw_mm: float = 0.0         # pupil -> axis distance; 0 = telecentric
    projector_blur_mm: float = 0.0          # Gaussian blur sigma between frame rows; 0 = sharp
    resin_gel_dose_mj_cm3: float = 0.0      # absorbed energy at the gel point

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
        physics = {
            "projector_gamma": self.projector_gamma,
            "projector_irradiance_mw_cm2": self.projector_irradiance_mw_cm2,
            "projector_contrast": self.projector_contrast,
            "projector_throw_mm": self.projector_throw_mm,
            "projector_blur_mm": self.projector_blur_mm,
            "resin_gel_dose_mj_cm3": self.resin_gel_dose_mj_cm3,
        }
        for name, value in physics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValidationError(f"{name} must be a finite nonnegative number.")
        if not 0.2 <= self.projector_gamma <= 5.0:
            raise ValidationError("projector_gamma must be between 0.2 and 5.")
        if 0 < self.projector_contrast < 10:
            raise ValidationError("projector_contrast must be 0 (ideal) or at least 10.")
        if self.projector_irradiance_mw_cm2 > 10_000 or self.resin_gel_dose_mj_cm3 > 1e7:
            raise ValidationError("Irradiance or gel dose is outside the supported range.")
        if self.projector_blur_mm > 2.0:
            raise ValidationError("projector_blur_mm must be between 0 and 2 mm.")
        if 0 < self.projector_throw_mm <= max(self.cuvette_outer_mm, self.diameter_mm):
            raise ValidationError("projector_throw_mm must be larger than the vial (or 0).")
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


def _coverage(values: np.ndarray) -> np.ndarray:
    """Coverage as float32 in [0, 1] from either float or uint8 (0..255) storage."""
    if values.dtype == np.uint8:
        return values.astype(np.float32) * np.float32(1 / 255)
    return np.asarray(values, dtype=np.float32)


def _build_print_plan(engine: Any, slices: np.ndarray, sinograms: np.ndarray, global_max: float,
                      resin: "ResinSettings", params: "SliceParams",
                      z_levels: Optional[np.ndarray] = None, out_dir: Optional[str] = None,
                      cancelled: Optional[Callable[[], bool]] = None,
                      calibration: Optional[Any] = None) -> dict[str, Any]:
    """Whole-volume dose of the frames as displayed -> exposure plan.

    With z_levels and out_dir it also runs the virtual print (virtual_print.py). For an
    exposure ladder (calibration.py) the plan instead times the reference pillar to the
    gel dose and records every pillar's relative dose.
    """
    from optics.fastcal import dose_statistics
    from print_plan import compute_print_plan

    gamma = float(params.projector_gamma)
    contrast = float(params.projector_contrast)

    def light_of(layers: list[int]) -> np.ndarray:
        light = _projection_gray(np.asarray(sinograms[layers], np.float32), global_max, resin)
        light = (_encode_gray(light, gamma).astype(np.float32) / 255.0) ** np.float32(gamma)
        if contrast > 0:
            light = light + (1.0 - light) / np.float32(contrast)
        return light

    # What the projector shows: its row blur mixes neighbouring rows (layers).
    from optics.presharpen import blurred_rows_getter, row_sigma
    layer_pitch = float(np.median(np.diff(z_levels))) if z_levels is not None and len(z_levels) > 1 else engine.pitch
    displayed = blurred_rows_getter(light_of, sinograms.shape[0],
                                    row_sigma(float(params.projector_blur_mm), layer_pitch))

    stats = dose_statistics(engine, slices, displayed)
    ladder = None
    if calibration is not None:
        ladder = calibration.measure(engine, slices, displayed)
        stats["threshold"] = ladder["reference_dose"]
        stats["window"] = 0.0
    virtual = None
    if z_levels is not None and out_dir is not None and calibration is None:
        # Virtual print of the frames as displayed: it also finds the exposure that
        # reproduces the model best, which sets the recommended print time.
        import virtual_print
        virtual = virtual_print.simulate(engine, slices, displayed, stats["threshold"], z_levels,
                                         cancelled=cancelled)
        virtual_print.save(virtual, slices, out_dir)
    span = engine.span
    c = (np.arange(engine.N) + 0.5) * span / engine.N - span / 2
    radius = np.hypot(*np.meshgrid(c, c, indexing="ij"))
    solid = slices >= (128 if slices.dtype == np.uint8 else 0.5)
    per_pixel = solid.sum(axis=2)
    total = float(per_pixel.sum())
    part_r = float(radius[per_pixel > 0].max()) if total else 0.0
    outside = float(per_pixel[radius > engine.op.coverage_radius].sum()) / total if total else 0.0
    plan = compute_print_plan(
        stats, alpha_per_mm=params.alpha, irradiance_mw_cm2=params.projector_irradiance_mw_cm2,
        gel_dose_mj_cm3=params.resin_gel_dose_mj_cm3,
        detector_bin_mm=engine.op.detector_span / engine.op.num_s, voxel_pitch_mm=engine.pitch,
        frame_rate_hz=params.frame_rate_hz, num_frames=params.num_frames,
        coverage_radius_mm=engine.op.coverage_radius, part_radius_mm=part_r,
        part_fraction_outside=outside, projector_contrast=contrast,
        exposure=None if virtual is None else virtual.metrics,
    )
    plan["dose_statistics"] = {k: float(v) for k, v in stats.items()}
    if ladder is not None:
        plan["calibration"] = ladder
        plan["warnings"] = [w for w in plan["warnings"] if "окно процесса" not in w]
    if virtual is not None:
        plan["virtual_print"] = virtual.metrics
    return plan


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


def _rasterize_layer_legacy(
    mesh: trimesh.Trimesh, z: float, i: int, grid_res: int, diameter_mm: float,
    fill_holes: bool, preserve_internal_voids: bool,
) -> np.ndarray:
    """Polygon + bitmap rasterisation of one layer, with morphological repair.

    Used for layers where the mesh is open (unbalanced winding), because the
    polygon assembly in trimesh tolerates such defects. Returns [x, y].
    """
    aa_size = grid_res * ANTI_ALIAS_FACTOR
    scale_2d = aa_size / diameter_mm
    offset = aa_size / 2.0
    result = np.zeros((grid_res, grid_res), dtype=np.float32)
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
        result[:, :] = (np.array(img_resized, dtype=np.float32) / 255.0).T
    return result


class SlicingEngine:
    """Mesh sections, projection planning and verified frame export."""

    @staticmethod
    def run(
        mesh: trimesh.Trimesh,
        params: SliceParams,
        progress_cb: Optional[ProgressCallback] = None,
        is_cancelled: Optional[CancelCheck] = None,
        manifest_context: Optional[dict[str, Any]] = None,
        calibration: Optional[Any] = None,
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

        base_angles = np.linspace(0.0, 360.0, num_frames, endpoint=False)
        angles = (
            params.angle_offset_deg + params.angle_direction * base_angles
        ) % 360.0
        from optics import fastcal as _fastcal
        # FastCAL reads coverage as uint8 (0..255): a quarter of the float32 volume.
        compact = params.projection_backend == "physical" and _fastcal.supports(
            params.container_mode, np.deg2rad(angles))

        report(0.05, "Нарезка сечений...")
        # Exact-coverage vectorised voxelizer (non-zero winding). Layers where
        # the mesh is open fall back to the tolerant polygon/bitmap path.
        from voxelizer import voxelize
        voxels = voxelize(
            mesh, z_levels, grid_res, diameter_mm, cancelled=cancelled,
            progress=lambda f: report(0.05 + 0.2 * f, "Нарезка сечений..."),
            out_dtype=np.uint8 if compact else np.float32,
        )
        slices = voxels.volume
        for n_fixed, i in enumerate(voxels.open_layers):
            if cancelled():
                raise SliceCancelled()
            layer = _rasterize_layer_legacy(
                mesh, float(z_levels[i]), int(i), grid_res, diameter_mm,
                fill_holes, preserve_internal_voids,
            )
            slices[:, :, i] = np.rint(layer * 255) if compact else layer
            report(0.25 + 0.05 * (n_fixed + 1) / len(voxels.open_layers),
                   f"Ремонт открытых сечений {n_fixed + 1}/{len(voxels.open_layers)}...")
        if fill_holes and not preserve_internal_voids:
            closed = np.setdiff1d(np.arange(nz), voxels.open_layers)
            full = 255 if compact else 1.0
            for i in closed:
                solid = binary_fill_holes(slices[:, :, i] >= 0.5 * full)
                slices[:, :, i] = np.maximum(slices[:, :, i], solid.astype(slices.dtype) * full)
        if len(voxels.open_layers):
            report(0.30, f"Открытая сетка: {len(voxels.open_layers)} сечений восстановлено.")
        if calibration is not None:
            # Exposure ladder (calibration.py): every pillar gets its own planned dose.
            calibration.apply_levels(slices, diameter_mm)

        # --- Проекционный backend ------------------------------------------
        backend = params.projection_backend
        use_fastcal = False
        detector = grid_res                 # detector bins of the planned frames
        optimizer_name = backend

        if backend == "physical":
            report(0.30, "Физическая оптика: трассировка лучей и оптимизация...")
            from optics.optimization import optimize_projections_sirt_volume
            angles_rad = np.deg2rad(angles)
            L_out = params.cuvette_outer_mm
            L_in = params.cuvette_inner_mm

            from optics import fastcal
            use_fastcal = fastcal.supports(params.container_mode, angles_rad)
            optimizer_name = "fastcal-band" if use_fastcal else "sirt"
            optimize = fastcal.optimize_projections_fast if use_fastcal else optimize_projections_sirt_volume
            fast_engines: list[Any] = []
            extra: dict[str, Any] = {}
            if use_fastcal:
                detector = fastcal.detector_bins(grid_res, output_res)
                extra = {"engine_out": fast_engines, "num_s": detector,
                         "optics": fastcal.OpticsOptions(throw_mm=params.projector_throw_mm)}
            sinograms = optimize(
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
                **extra,
            )
        elif backend in {"auto", "vamtoolbox"}:
            report(0.30, "Optimizing projections with VAMToolbox CAL...")
            try:
                # VAMToolbox consumes the complete voxel target and returns a
                # projection sequence.  The adapter normalizes its layout to
                # the same (z, detector, angle) convention used by the
                # internal Radon path.
                sinograms = optimize_sinograms(
                    slices,
                    angles,
                    iterations=params.optimizer_iterations,
                    conda_executable=params.vam_conda_executable or None,
                    environment_name=params.vam_environment_name or DEFAULT_VAM_ENV_NAME,
                )
                report(0.78, "VAMToolbox projection optimization complete.")
            except VAMToolboxUnavailable as exc:
                if backend == "vamtoolbox":
                    raise ValidationError(str(exc)) from exc
                report(0.30, "VAMToolbox is unavailable; using internal Radon backend.")
                sinograms = _radon_sinograms(
                    slices, angles, grid_res, num_frames, report, cancelled
                )
            except Exception:
                if backend == "vamtoolbox":
                    raise
                report(0.30, "VAMToolbox failed; using internal Radon backend.")
                sinograms = _radon_sinograms(
                    slices, angles, grid_res, num_frames, report, cancelled
                )
        else:
            report(0.30, "Calculating Radon projections...")
            sinograms = _radon_sinograms(
                slices, angles, grid_res, num_frames, report, cancelled
            )

        if params.projector_blur_mm > 0 and nz > 1:
            # The projector blurs rows into each other: show it frames that it blurs back
            # into the planned ones (optics/presharpen.py).
            from optics.presharpen import presharpen_rows, row_sigma
            report(0.79, "Компенсация размытия проектора...")

            def stop() -> None:
                if cancelled():
                    raise SliceCancelled()

            if not sinograms.flags.writeable or sinograms.dtype not in (np.float16, np.float32):
                sinograms = np.array(sinograms, np.float32)
            presharpen_rows(sinograms, row_sigma(params.projector_blur_mm, pitch), out=sinograms,
                            cancelled=stop)
        global_max = float(sinograms.max()) if sinograms.size else 1.0
        if global_max <= 0:
            global_max = 1.0

        out_dir = FrameRepository.create_run_dir(params.output_dir)
        model_height_mm = z_max - z_min
        row_pitch = projection_span / output_res
        if abs(pitch - row_pitch) <= 1e-9 * row_pitch:
            # Layers were sliced at the projector's row pitch: one layer = one row,
            # exactly. Rounding the part height instead gave a frame one row taller
            # or shorter for ~8 % of part heights, and every row got resampled (half a
            # pixel of vertical blur and a 1-row stretch).
            target_h = nz
        else:
            target_h = max(int(round(output_res * model_height_mm / projection_span)), 1)
        mid_z = nz // 2
        mid_frame_row = int(np.clip(round((nz - mid_z - 0.5) * target_h / nz - 0.5), 0, target_h - 1))
        exported_mid_sinograms = np.empty((detector, num_frames), dtype=np.float32)

        report(0.80, "Сохранение кадров...")
        save_step = max(1, num_frames // 20)
        import cv2
        gamma = float(params.projector_gamma)
        upsample = target_h >= nz and output_res >= detector
        def export_frame(i: int) -> tuple[int, np.ndarray]:
            frame = np.flipud(np.asarray(sinograms[:, :, i], dtype=np.float32))
            normalized = _projection_gray(frame, global_max, resin)
            # Resize floating gray values before the single 8-bit quantization.
            if upsample:
                # Same half-pixel linear interpolation as skimage, ~20x faster.
                resized = cv2.resize(normalized.astype(np.float32), (output_res, target_h),
                                     interpolation=cv2.INTER_LINEAR)
            else:
                resized = sk_resize(
                    normalized, (target_h, output_res), order=1, mode="edge",
                    anti_aliasing=True, preserve_range=True,
                )
            # Light is resized linearly; gamma is applied once, at quantisation.
            output = _encode_gray(resized, gamma)
            Image.fromarray(output).save(os.path.join(out_dir, f"frame_{i:04d}.png"), compress_level=1)
            return i, _exported_detector_row(output, mid_frame_row, detector, gamma)

        # Resizing and PNG encoding release the GIL: frames are written in parallel.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as pool:
            for done_frames, (i, row) in enumerate(pool.map(export_frame, range(num_frames))):
                if cancelled():
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise SliceCancelled()
                exported_mid_sinograms[:, i] = row
                if done_frames % save_step == 0:
                    report(0.80 + 0.19 * (done_frames / num_frames),
                           f"Сохранение кадра {done_frames}/{num_frames}...")

        if cancelled():
            raise SliceCancelled()
        # Reconstruct actual saved drive values, including user gain, cutoff,
        # resampling, saturation and PNG quantization.
        if backend == "physical" and use_fastcal:
            # Same rotation-symmetric operator that planned the frames; no
            # full sparse system matrix is built just for the preview.
            dose_mid = fast_engines[0].layer_dose(exported_mid_sinograms)
        elif backend == "physical":
            from optics.optimization import build_system_matrix
            _, adjoint = build_system_matrix(
                angles_rad, num_s=grid_res, grid_res=grid_res, grid_span_mm=diameter_mm,
                L_out=params.cuvette_outer_mm, L_in=params.cuvette_inner_mm,
                n_air=params.n_air, n_glass=params.n_glass, n_liquid=params.n_liquid,
                n_resin=params.n_resin, alpha=params.alpha, container_mode=params.container_mode,
                cancelled_cb=cancelled,
            )
            dose_mid = (adjoint @ exported_mid_sinograms.T.ravel()).reshape(grid_res, grid_res)
        else:
            from skimage.transform import iradon
            # Optical exposure accumulates positive frames; a ramp-filtered CT
            # inversion would hide backprojection blur and is not delivered dose.
            dose_mid = iradon(exported_mid_sinograms, theta=angles, circle=True, filter_name=None)
        target_mid = _coverage(slices[:, :, mid_z])
        relative_dose, dose_rmse, dose_scale = _relative_dose_fit(dose_mid, target_mid)
        # Numeric artifacts allow the score to be reproduced independently.
        np.save(os.path.join(out_dir, "reconstruction_dose.npy"), dose_mid)
        np.save(os.path.join(out_dir, "reconstruction_target.npy"), target_mid)
        from matplotlib import colormaps
        rgba = colormaps["viridis"](np.flipud(np.clip(relative_dose, 0, 1).T))
        Image.fromarray(np.rint(rgba[:, :, :3] * 255).astype(np.uint8)).save(
            os.path.join(out_dir, "reconstruction_preview.png")
        )
        with open(os.path.join(out_dir, "dose_stats.txt"), "w", encoding="utf-8") as handle:
            handle.write(
                f"rmse={dose_rmse:.8f}\nloss={np.linalg.norm(relative_dose-target_mid):.8f}\n"
                f"relative_scale={dose_scale:.10g}\nsource=exported_png\nmodel=relative_linear_dose\n"
                f"optimizer={optimizer_name}\n"
            )

        print_plan_data: dict[str, Any] = {}
        if backend == "physical" and use_fastcal:
            report(0.995, "План печати...")
            print_plan_data = _build_print_plan(
                fast_engines[0], slices, sinograms, global_max, resin, params,
                z_levels=z_levels, out_dir=out_dir, cancelled=cancelled, calibration=calibration,
            )
            with open(os.path.join(out_dir, "print_plan.json"), "w", encoding="utf-8") as handle:
                json.dump(print_plan_data, handle, ensure_ascii=False, indent=2)

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
        context = manifest_context or {}
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
        machine_profile = _machine_profile_payload(
            context.get("machine_profile"),
            diameter_mm=diameter_mm,
            output_res=output_res,
            target_h=target_h,
            frame_rate_hz=params.frame_rate_hz,
            angle_offset_deg=params.angle_offset_deg,
            angle_direction=params.angle_direction,
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
                        "detector_spacing_mm": projection_span / detector,
                        "detector_bins": int(detector),
                        "output_pixel_pitch_mm": projection_span / output_res,
                        "dose_model": "relative_linear",
                        "optimizer": optimizer_name,
                        "projector_gamma": params.projector_gamma,
                        "projector_throw_mm": params.projector_throw_mm,
                        "projector_blur_mm": params.projector_blur_mm,
                    },
                    "print_plan": print_plan_data,
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
