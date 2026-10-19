"""Validation and workload budgeting for untrusted local inputs.

The GUI sliders are useful presentation constraints, but the numerical core is
also a public Python API.  This module therefore keeps the important limits in
one place and applies them before expensive parsing, decoding, or allocation.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from constants import (
    ANTI_ALIAS_FACTOR,
    MAX_DIAMETER_MM,
    MAX_ESTIMATED_MEMORY_BYTES,
    MAX_FRAME_DIMENSION,
    MAX_FRAME_TOTAL_PIXELS,
    MAX_GRID_RESOLUTION,
    MAX_LAYERS,
    MAX_MESH_TRIANGLES,
    MAX_NUM_FRAMES,
    MAX_OUTPUT_RESOLUTION,
    MAX_STL_FILE_BYTES,
    MIN_DIAMETER_MM,
    MIN_GRID_RESOLUTION,
    MIN_NUM_FRAMES,
    MIN_OUTPUT_RESOLUTION,
)


class ValidationError(ValueError):
    """Raised when an input would violate a safety or consistency invariant."""


@dataclass(frozen=True)
class PreflightReport:
    """Human-readable model and workload validation result."""

    ok: bool
    errors: list[str]
    warnings: list[str]
    triangle_count: int
    bounds_mm: tuple[float, float, float]
    estimated_layers: int
    estimated_memory_bytes: int


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a number.") from exc
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be finite.")
    return number


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be an integer.") from exc
    if number != value:
        raise ValidationError(f"{name} must be an integer.")
    return number


def validate_slice_parameters(
    diameter_mm: Any,
    grid_res: Any,
    output_res: Any,
    num_frames: Any,
    resin: Any,
    fill_holes: Any = True,
    projection_backend: Any = "internal",
    optimizer_iterations: Any = 8,
    preserve_internal_voids: Any = True,
    frame_rate_hz: Any = 24.0,
    angle_offset_deg: Any = 90.0,
    angle_direction: Any = 1,
) -> None:
    """Validate parameters before they influence array dimensions or math."""

    diameter = _finite_number(diameter_mm, "diameter_mm")
    if not MIN_DIAMETER_MM <= diameter <= MAX_DIAMETER_MM:
        raise ValidationError(
            f"diameter_mm must be between {MIN_DIAMETER_MM:g} and "
            f"{MAX_DIAMETER_MM:g}."
        )

    grid = _integer(grid_res, "grid_res")
    if not MIN_GRID_RESOLUTION <= grid <= MAX_GRID_RESOLUTION:
        raise ValidationError(
            f"grid_res must be between {MIN_GRID_RESOLUTION} and {MAX_GRID_RESOLUTION}."
        )

    output = _integer(output_res, "output_res")
    if not MIN_OUTPUT_RESOLUTION <= output <= MAX_OUTPUT_RESOLUTION:
        raise ValidationError(
            f"output_res must be between {MIN_OUTPUT_RESOLUTION} and {MAX_OUTPUT_RESOLUTION}."
        )

    frames = _integer(num_frames, "num_frames")
    if not MIN_NUM_FRAMES <= frames <= MAX_NUM_FRAMES:
        raise ValidationError(
            f"num_frames must be between {MIN_NUM_FRAMES} and {MAX_NUM_FRAMES}."
        )

    exposure = _finite_number(getattr(resin, "base_exposure", None), "base_exposure")
    intensity = _finite_number(getattr(resin, "intensity", None), "intensity")
    threshold = _finite_number(getattr(resin, "threshold", None), "threshold")
    if not isinstance(fill_holes, bool):
        raise ValidationError("fill_holes must be boolean.")
    if not isinstance(preserve_internal_voids, bool):
        raise ValidationError("preserve_internal_voids must be boolean.")
    if exposure <= 0.0:
        raise ValidationError("base_exposure must be greater than zero.")
    if intensity <= 0.0:
        raise ValidationError("intensity must be greater than zero.")
    if not 0.0 <= threshold <= 100.0:
        raise ValidationError("threshold must be between 0 and 100.")

    if projection_backend not in {"internal", "auto", "vamtoolbox", "physical"}:
        raise ValidationError(
            "projection_backend must be one of: internal, auto, vamtoolbox, physical."
        )
    iterations = _integer(optimizer_iterations, "optimizer_iterations")
    if not 1 <= iterations <= 100:
        raise ValidationError("optimizer_iterations must be between 1 and 100.")
    frame_rate = _finite_number(frame_rate_hz, "frame_rate_hz")
    if not 0.1 <= frame_rate <= 240.0:
        raise ValidationError("frame_rate_hz must be between 0.1 and 240.")
    _finite_number(angle_offset_deg, "angle_offset_deg")
    direction = _integer(angle_direction, "angle_direction")
    if direction not in {-1, 1}:
        raise ValidationError("angle_direction must be -1 or 1.")


def validate_stl_path(path: str) -> int:
    """Check a selected STL before handing it to a third-party parser."""

    if not path:
        raise ValidationError("STL path is empty.")
    candidate = Path(path)
    if not candidate.is_file():
        raise ValidationError(f"STL file does not exist: {path}")
    if candidate.is_symlink():
        raise ValidationError("STL symlinks are not supported for safety.")
    size = candidate.stat().st_size
    if size <= 0:
        raise ValidationError("STL file is empty.")
    if size > MAX_STL_FILE_BYTES:
        raise ValidationError(
            f"STL file is too large ({size / 1024**2:.1f} MiB); "
            f"limit is {MAX_STL_FILE_BYTES / 1024**2:.0f} MiB."
        )
    # A binary STL stores its triangle count in bytes 80..83.  When the
    # declared record length matches the file size, reject an over-budget
    # model before trimesh allocates its mesh arrays. ASCII STL files simply
    # continue to the parser and are checked again after loading.
    if size >= 84:
        try:
            with candidate.open("rb") as handle:
                header = handle.read(84)
            declared_triangles = int.from_bytes(header[80:84], "little")
            if 84 + declared_triangles * 50 == size and declared_triangles > MAX_MESH_TRIANGLES:
                raise ValidationError(
                    f"STL declares {declared_triangles:,} triangles; "
                    f"limit is {MAX_MESH_TRIANGLES:,}."
                )
        except OSError as exc:
            raise ValidationError(f"Could not inspect STL header: {exc}") from exc
    return size


def validate_mesh_geometry(mesh: Any) -> None:
    """Reject malformed geometry before centering or exposing it to the UI."""

    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
    faces = np.asarray(getattr(mesh, "faces", []))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValidationError("Mesh vertices must be a non-empty N×3 array.")
    if not np.isfinite(vertices).all():
        raise ValidationError("Mesh contains NaN or infinite coordinates.")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValidationError("Mesh faces must be a non-empty triangle array.")
    if len(faces) > MAX_MESH_TRIANGLES:
        raise ValidationError(
            f"Mesh has {len(faces):,} triangles; limit is {MAX_MESH_TRIANGLES:,}."
        )
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValidationError("Mesh face indices must be integers.")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValidationError("Mesh contains an out-of-range face index.")
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    if not np.isfinite(extents).all() or np.any(extents <= 1e-9):
        raise ValidationError("Mesh has a zero-sized or non-finite axis.")


def estimate_layer_count(z_min: float, z_max: float, pitch_mm: float) -> int:
    """Return a safe upper bound close to the engine's ``np.arange`` count."""

    span = max(0.0, z_max - z_min)
    if span <= 0.0:
        return 0
    return max(1, int(math.ceil(span / pitch_mm)))


def estimate_slicing_memory(
    grid_res: int,
    layers: int,
    num_frames: int,
    output_res: int | None = None,
    projection_backend: str = "internal",
) -> int:
    """Estimate peak buffers used by forward slicing and PNG rendering."""

    grid = max(1, int(grid_res))
    nz = max(1, int(layers))
    frames = max(1, int(num_frames))
    float_bytes = 4
    slices = grid * grid * nz * float_bytes
    sinograms = nz * grid * frames * float_bytes
    anti_alias = (grid * ANTI_ALIAS_FACTOR) ** 2
    output_buffers = 0
    if output_res is not None:
        target_height = max(int(round(int(output_res) * nz / max(grid, int(output_res)))), 1)
        output_pixels = target_height * int(output_res)
        # skimage.resize and the uint8 output coexist briefly during a frame
        # export, so count two output-sized buffers before adding headroom.
        output_buffers = output_pixels * (4 + 1)
    matrices = 0
    if projection_backend == "physical":
        # Exact grid traversal crosses at most 2*N+1 segments per ray.
        # Three CSR payloads conservatively cover assembly and transposition.
        matrices = frames * grid * (2 * grid + 1) * 24
    workspace = 32 * 1024**2
    return int((slices + sinograms + anti_alias + output_buffers + workspace) * 1.5 + matrices)


def preflight_mesh(
    mesh: Any,
    params: Any,
    vat_height_mm: float | None = None,
) -> PreflightReport:
    """Validate a transformed mesh and calculate its expected workload."""

    errors: list[str] = []
    warnings: list[str] = []
    try:
        validate_slice_parameters(
            params.diameter_mm,
            params.grid_res,
            params.output_res,
            params.num_frames,
            params.resin,
            params.fill_holes,
            getattr(params, "projection_backend", "internal"),
            getattr(params, "optimizer_iterations", 8),
            getattr(params, "preserve_internal_voids", True),
            getattr(params, "frame_rate_hz", 24.0),
            getattr(params, "angle_offset_deg", 90.0),
            getattr(params, "angle_direction", 1),
        )
    except ValidationError as exc:
        return PreflightReport(False, [str(exc)], [], 0, (0.0, 0.0, 0.0), 0, 0)

    try:
        validate_mesh_geometry(mesh)
    except ValidationError as exc:
        errors.append(str(exc))

    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
    faces = np.asarray(getattr(mesh, "faces", []))
    triangle_count = int(len(faces))
    if triangle_count <= 0:
        errors.append("Mesh contains no triangles.")
    if triangle_count > MAX_MESH_TRIANGLES:
        errors.append(
            f"Mesh has {triangle_count:,} triangles; limit is {MAX_MESH_TRIANGLES:,}."
        )
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        errors.append("Mesh vertices must be a non-empty N×3 array.")
        return PreflightReport(False, errors, warnings, triangle_count, (0.0, 0.0, 0.0), 0, 0)
    if not np.isfinite(vertices).all():
        errors.append("Mesh contains NaN or infinite coordinates.")

    bounds = np.asarray(getattr(mesh, "bounds", []), dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        errors.append("Mesh bounds are missing or non-finite.")
        return PreflightReport(False, errors, warnings, triangle_count, (0.0, 0.0, 0.0), 0, 0)

    extents = bounds[1] - bounds[0]
    if not np.isfinite(extents).all():
        errors.append("Mesh dimensions overflow the supported numeric range.")
        return PreflightReport(False, errors, warnings, triangle_count, (0.0, 0.0, 0.0), 0, 0)
    bounds_mm = (float(extents[0]), float(extents[1]), float(extents[2]))
    if np.any(extents <= 1e-9):
        errors.append("Mesh has a zero-sized or degenerate axis.")

    diameter = float(params.diameter_mm)
    grid = int(params.grid_res)
    frames = int(params.num_frames)
    pitch = diameter / max(grid, int(params.output_res))
    estimated_layers = estimate_layer_count(float(bounds[0, 2]), float(bounds[1, 2]), pitch)
    if estimated_layers > MAX_LAYERS:
        errors.append(
            f"Estimated layer count is {estimated_layers:,}; limit is {MAX_LAYERS:,}."
        )

    target_height = max(int(round(params.output_res * float(extents[2]) / diameter)), 1)
    if target_height > MAX_FRAME_DIMENSION or params.output_res > MAX_FRAME_DIMENSION:
        errors.append(
            f"Output frame dimensions would be {target_height}×{params.output_res} px; "
            f"limit is {MAX_FRAME_DIMENSION} px per side."
        )
    total_output_pixels = target_height * int(params.output_res) * frames
    if total_output_pixels > MAX_FRAME_TOTAL_PIXELS:
        errors.append(
            f"Estimated output contains {total_output_pixels:,} pixels; "
            f"limit is {MAX_FRAME_TOTAL_PIXELS:,}."
        )

    estimated_memory = estimate_slicing_memory(
        grid, estimated_layers, frames, output_res=int(params.output_res),
        projection_backend=getattr(params, "projection_backend", "internal"),
    )
    if estimated_memory > MAX_ESTIMATED_MEMORY_BYTES:
        errors.append(
            f"Estimated slicing memory is {estimated_memory / 1024**3:.2f} GiB; "
            f"limit is {MAX_ESTIMATED_MEMORY_BYTES / 1024**3:.2f} GiB."
        )

    if vat_height_mm is not None:
        vat_height = _finite_number(vat_height_mm, "vat_height_mm")
        physical = getattr(params, "projection_backend", "internal") == "physical"
        resin_diameter = float(getattr(params, "cuvette_inner_mm", diameter)) if physical else diameter
        if physical and getattr(params, "container_mode", 0) == 1:
            radial = np.max(np.abs(vertices[:, :2]), axis=1)
        else:
            radial = np.hypot(vertices[:, 0], vertices[:, 1])
        max_radius = float(np.max(radial)) if len(radial) else 0.0
        if max_radius > resin_diameter / 2.0 + 1e-6:
            errors.append(
                f"Model exceeds the resin radius/half-width ({max_radius:.2f} mm > {resin_diameter / 2.0:.2f} mm)."
            )
        z_abs = max(abs(float(bounds[0, 2])), abs(float(bounds[1, 2])))
        if z_abs > vat_height / 2.0 + 1e-6:
            errors.append(
                f"Model exceeds the vat height ({z_abs * 2.0:.2f} mm > {vat_height:.2f} mm)."
            )
        if max_radius > resin_diameter / 2.0 * 0.95:
            warnings.append("Model is very close to the vat wall.")
        if z_abs > vat_height / 2.0 * 0.95:
            warnings.append("Model is very close to the top or bottom of the vat.")

    return PreflightReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        triangle_count=triangle_count,
        bounds_mm=bounds_mm,
        estimated_layers=estimated_layers,
        estimated_memory_bytes=estimated_memory,
    )


def format_preflight_report(report: PreflightReport) -> str:
    """Format a report for a QMessageBox or the shared application log."""

    size = " × ".join(f"{value:.2f}" for value in report.bounds_mm)
    memory_gib = report.estimated_memory_bytes / 1024**3
    lines = [
        f"Model size: {size} mm",
        f"Triangles: {report.triangle_count:,}",
        f"Estimated layers: {report.estimated_layers:,}",
        f"Estimated memory: {memory_gib:.2f} GiB",
    ]
    if report.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in report.warnings)
    if report.errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in report.errors)
    return "\n".join(lines)


def ensure_directory(path: str) -> Path:
    """Resolve and create a local directory while rejecting linked parents."""

    if not path:
        raise ValidationError("Output directory is empty.")
    candidate = Path(path)
    if _has_link_component(candidate):
        raise ValidationError("Linked output directories are not supported for safety.")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate.resolve()


def validate_directory(path: str) -> Path:
    """Resolve an existing directory while rejecting linked path components."""

    if not path:
        raise ValidationError("Frame directory is empty.")
    candidate = Path(path)
    if not candidate.is_dir():
        raise ValidationError(f"Directory does not exist: {path}")
    if _has_link_component(candidate):
        raise ValidationError("Linked input directories are not supported for safety.")
    return candidate.resolve()


def _has_link_component(path: Path) -> bool:
    """Check the path itself and existing parents for symlinks/junctions."""

    is_junction = getattr(os.path, "isjunction", lambda value: False)
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        try:
            if os.path.lexists(str(component)) and (
                component.is_symlink()
                or os.path.islink(str(component))
                or bool(is_junction(str(component)))
            ):
                return True
        except OSError:
            return True
    return False
