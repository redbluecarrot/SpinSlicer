"""Shared, validated storage for generated projection frame sets.

The original application treated every directory containing ``frame_*.png``
as a complete job.  This module keeps that legacy format readable while
adding run-scoped directories, atomic JSON writes, manifests, and bounded
frame-set inspection before any expensive decoding.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from constants import (
    MAX_ESTIMATED_MEMORY_BYTES,
    MAX_FRAME_DIMENSION,
    MAX_FRAME_FILE_BYTES,
    MAX_FRAME_TOTAL_BYTES,
    MAX_FRAME_TOTAL_PIXELS,
    MAX_GRID_RESOLUTION,
    MAX_LAYERS,
    MAX_METADATA_FILE_BYTES,
    MAX_NUM_FRAMES,
    MAX_OUTPUT_RESOLUTION,
    MIN_GRID_RESOLUTION,
    MIN_NUM_FRAMES,
    MIN_OUTPUT_RESOLUTION,
)
from validation import (
    ValidationError,
    ensure_directory,
    validate_directory,
)

META_FILENAME = "slice_meta.json"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
LEGACY_GRID_RES = 112
_FRAME_NAME = re.compile(r"^frame_\d+\.png$")


@dataclass
class SliceMeta:
    diameter_mm: float
    grid_res: int
    output_res: int
    num_frames: int
    fill_holes: bool
    resin_base_exposure: float
    resin_intensity: float
    resin_threshold: float
    generated_at: str = ""
    schema_version: int = MANIFEST_SCHEMA_VERSION
    units: str = "mm"
    complete: bool = True


@dataclass
class GenerationManifest:
    schema_version: int = MANIFEST_SCHEMA_VERSION
    source_sha256: str = ""
    source_name: str = ""
    units: str = "mm"
    transform_matrix: list[list[float]] = field(default_factory=list)
    slice_parameters: dict[str, Any] = field(default_factory=dict)
    machine_profile: dict[str, Any] = field(default_factory=dict)
    resin_profile: dict[str, Any] = field(default_factory=dict)
    frame_schedule: dict[str, Any] = field(default_factory=dict)
    frame_count: int = 0
    frame_size: tuple[int, int] = (0, 0)  # width, height
    complete: bool = False
    generated_at: str = ""


@dataclass(frozen=True)
class FrameSetInfo:
    paths: list[str]
    width: int
    height: int
    frame_count: int
    estimated_bytes: int
    resolved_dir: str
    grid_res: int
    nz: int
    meta: Optional[SliceMeta] = None


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(str(path.parent))
    if _is_link(path):
        raise ValidationError(f"Refusing to replace linked metadata file: {path.name}.")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def _validate_meta(meta: SliceMeta) -> None:
    numeric = {
        "diameter_mm": meta.diameter_mm,
        "resin_base_exposure": meta.resin_base_exposure,
        "resin_intensity": meta.resin_intensity,
        "resin_threshold": meta.resin_threshold,
    }
    for name, value in numeric.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValidationError(f"Invalid metadata value: {name}.")
    integer_fields = {
        "schema_version": meta.schema_version,
        "grid_res": meta.grid_res,
        "output_res": meta.output_res,
        "num_frames": meta.num_frames,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_fields.values()
    ):
        raise ValidationError("Metadata integer fields are invalid.")
    if meta.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValidationError("Unsupported metadata schema version.")
    if meta.diameter_mm <= 0.0:
        raise ValidationError("Metadata diameter_mm must be greater than zero.")
    if not MIN_GRID_RESOLUTION <= meta.grid_res <= MAX_GRID_RESOLUTION:
        raise ValidationError("Metadata grid_res is outside the supported range.")
    if not MIN_OUTPUT_RESOLUTION <= meta.output_res <= MAX_OUTPUT_RESOLUTION:
        raise ValidationError("Metadata output_res is outside the supported range.")
    if not MIN_NUM_FRAMES <= meta.num_frames <= MAX_NUM_FRAMES:
        raise ValidationError("Metadata num_frames is outside the supported range.")
    if meta.resin_base_exposure <= 0.0 or meta.resin_intensity <= 0.0:
        raise ValidationError("Metadata resin exposure and intensity must be positive.")
    if not 0.0 <= meta.resin_threshold <= 100.0:
        raise ValidationError("Metadata resin_threshold must be between 0 and 100.")
    if not isinstance(meta.fill_holes, bool) or not isinstance(meta.complete, bool):
        raise ValidationError("Metadata boolean fields are invalid.")
    if not isinstance(meta.generated_at, str) or not isinstance(meta.units, str):
        raise ValidationError("Metadata text fields are invalid.")
    if meta.units != "mm":
        raise ValidationError("Only millimetre metadata is supported by this version.")


def save_meta(out_dir: str, meta: SliceMeta) -> None:
    """Atomically save validated metadata after a generation is complete."""

    _validate_meta(meta)
    _atomic_json_write(Path(out_dir) / META_FILENAME, asdict(meta))


def load_meta(out_dir: str) -> Optional[SliceMeta]:
    """Load metadata; missing metadata is supported only for legacy folders."""

    path = Path(out_dir) / META_FILENAME
    if not path.is_file():
        return None
    try:
        if _is_link(path) or path.stat().st_size > MAX_METADATA_FILE_BYTES:
            raise ValidationError(f"Invalid {META_FILENAME}: file is too large or linked.")
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise TypeError("metadata root must be an object")
        meta = SliceMeta(**raw)
        _validate_meta(meta)
        return meta
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid {META_FILENAME}: {exc}") from exc


def _validate_manifest(manifest: GenerationManifest) -> None:
    if (
        isinstance(manifest.schema_version, bool)
        or not isinstance(manifest.schema_version, int)
        or manifest.schema_version != MANIFEST_SCHEMA_VERSION
    ):
        raise ValidationError("Unsupported manifest schema version.")
    if not isinstance(manifest.complete, bool):
        raise ValidationError("Manifest complete must be boolean.")
    if (
        isinstance(manifest.frame_count, bool)
        or not isinstance(manifest.frame_count, int)
        or manifest.frame_count < 0
        or manifest.frame_count > MAX_NUM_FRAMES
    ):
        raise ValidationError("Manifest frame_count is outside the supported range.")
    if not isinstance(manifest.frame_size, (tuple, list)) or len(manifest.frame_size) != 2:
        raise ValidationError("Manifest frame_size is invalid.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in manifest.frame_size
    ):
        raise ValidationError("Manifest frame_size is invalid.")
    if manifest.complete and manifest.frame_count < MIN_NUM_FRAMES:
        raise ValidationError("A complete manifest must contain at least two frames.")
    if manifest.complete and any(
        value <= 0 or value > MAX_FRAME_DIMENSION for value in manifest.frame_size
    ):
        raise ValidationError("Complete manifest frame_size is outside the supported range.")
    if not isinstance(manifest.source_sha256, str) or not isinstance(manifest.source_name, str):
        raise ValidationError("Manifest source fields are invalid.")
    if not isinstance(manifest.units, str) or manifest.units != "mm":
        raise ValidationError("Only millimetre manifests are supported by this version.")
    if not isinstance(manifest.slice_parameters, dict) or not isinstance(
        manifest.generated_at, str
    ):
        raise ValidationError("Manifest payload fields are invalid.")
    if not isinstance(manifest.machine_profile, dict):
        raise ValidationError("Manifest machine_profile must be an object.")
    if not isinstance(manifest.resin_profile, dict):
        raise ValidationError("Manifest resin_profile must be an object.")
    if not isinstance(manifest.frame_schedule, dict):
        raise ValidationError("Manifest frame_schedule must be an object.")
    schedule = manifest.frame_schedule
    raw_angles = schedule.get("angles_deg")
    if raw_angles is not None:
        if not isinstance(raw_angles, list) or len(raw_angles) != manifest.frame_count:
            raise ValidationError("Manifest frame schedule angle count is invalid.")
        try:
            if any(not math.isfinite(float(value)) for value in raw_angles):
                raise ValidationError("Manifest frame schedule angles must be finite.")
        except (TypeError, ValueError) as exc:
            raise ValidationError("Manifest frame schedule angles are invalid.") from exc
    if "frame_duration_s" in schedule:
        try:
            duration = float(schedule["frame_duration_s"])
        except (TypeError, ValueError) as exc:
            raise ValidationError("Manifest frame duration is invalid.") from exc
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValidationError("Manifest frame duration must be positive and finite.")
    if "frame_rate_hz" in schedule:
        try:
            frame_rate = float(schedule["frame_rate_hz"])
        except (TypeError, ValueError) as exc:
            raise ValidationError("Manifest frame rate is invalid.") from exc
        if not math.isfinite(frame_rate) or frame_rate <= 0.0:
            raise ValidationError("Manifest frame rate must be positive and finite.")
    matrix = manifest.transform_matrix
    if not isinstance(matrix, list):
        raise ValidationError("Manifest transform_matrix must be a finite 4×4 matrix.")
    try:
        matrix_invalid = bool(matrix) and (
            len(matrix) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
            or any(not math.isfinite(float(value)) for row in matrix for value in row)
        )
    except (TypeError, ValueError):
        matrix_invalid = True
    if matrix_invalid:
        raise ValidationError("Manifest transform_matrix must be a finite 4×4 matrix.")


def save_manifest(out_dir: str, manifest: GenerationManifest) -> None:
    """Atomically publish a manifest; consumers only accept ``complete=True``."""

    _validate_manifest(manifest)
    _atomic_json_write(Path(out_dir) / MANIFEST_FILENAME, asdict(manifest))


def load_manifest(out_dir: str) -> Optional[GenerationManifest]:
    path = Path(out_dir) / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        if _is_link(path) or path.stat().st_size > MAX_METADATA_FILE_BYTES:
            raise ValidationError(f"Invalid {MANIFEST_FILENAME}: file is too large or linked.")
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise TypeError("manifest root must be an object")
        if isinstance(raw.get("frame_size"), list):
            raw["frame_size"] = tuple(raw["frame_size"])
        manifest = GenerationManifest(**raw)
        _validate_manifest(manifest)
        return manifest
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid {MANIFEST_FILENAME}: {exc}") from exc


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_link(path: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", lambda value: False)
    return path.is_symlink() or os.path.islink(str(path)) or bool(is_junction(str(path)))


def _matching_entries(directory: Path, limit: int | None = None) -> list[Path]:
    """Enumerate numeric frame names without retaining an unbounded list."""

    entries: list[Path] = []
    for entry in directory.iterdir():
        if not _FRAME_NAME.fullmatch(entry.name):
            continue
        entries.append(entry)
        if limit is not None and len(entries) >= limit:
            break
    return sorted(entries)


class FrameRepository:
    """Create, resolve, and validate both current and legacy frame folders."""

    @staticmethod
    def create_run_dir(output_root: str) -> str:
        root = ensure_directory(output_root)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for _ in range(5):
            candidate = root / f"run-{timestamp}-{uuid.uuid4().hex[:8]}"
            try:
                candidate.mkdir()
                return str(candidate)
            except FileExistsError:
                continue
        raise ValidationError("Could not allocate a unique output run directory.")

    @staticmethod
    def resolve_directory(frames_dir: str) -> Path:
        candidate = validate_directory(frames_dir)

        latest_complete_run: Optional[Path] = None
        latest_mtime_ns = -1
        for child in candidate.iterdir():
            if not child.is_dir() or not child.name.startswith("run-") or _is_link(child):
                continue
            try:
                manifest = load_manifest(str(child))
            except ValidationError:
                continue
            if manifest is None or not manifest.complete:
                continue
            try:
                mtime_ns = child.stat().st_mtime_ns
            except OSError:
                continue
            if mtime_ns > latest_mtime_ns:
                latest_complete_run = child
                latest_mtime_ns = mtime_ns
        if latest_complete_run is not None:
            return latest_complete_run

        # Legacy compatibility: a flat output_frames/ directory is accepted
        # only when no completed run is available to supersede it.
        direct = _matching_entries(candidate, limit=1)
        if direct:
            return candidate
        return candidate

    @staticmethod
    def validate(frames_dir: str, require_complete: bool = True) -> FrameSetInfo:
        directory = FrameRepository.resolve_directory(frames_dir)
        manifest = load_manifest(str(directory))
        is_run_directory = directory.name.startswith("run-")
        if require_complete and (is_run_directory or manifest is not None):
            if manifest is None or not manifest.complete:
                raise ValidationError("The selected generation is not complete.")

        entries = _matching_entries(directory, limit=MAX_NUM_FRAMES + 1)
        linked = [entry for entry in entries if _is_link(entry)]
        if linked:
            raise ValidationError("Linked frame files are not supported for safety.")
        non_files = [entry for entry in entries if not entry.is_file()]
        if non_files:
            raise ValidationError(f"Frame entry is not a regular file: {non_files[0].name}.")
        paths = [str(entry) for entry in entries]
        if not paths:
            raise ValidationError(f"No frame_*.png files found in {directory}.")
        if len(paths) > MAX_NUM_FRAMES:
            raise ValidationError(
                f"Frame set contains {len(paths):,} files; limit is {MAX_NUM_FRAMES:,}."
            )

        meta = load_meta(str(directory))
        if is_run_directory and meta is None:
            raise ValidationError("A generated run is missing slice_meta.json.")
        if meta is not None and not meta.complete:
            raise ValidationError("The selected metadata is incomplete.")
        if meta is not None and meta.num_frames != len(paths):
            raise ValidationError(
                f"Metadata expects {meta.num_frames} frames, but {len(paths)} were found."
            )
        if manifest is not None:
            if manifest.frame_count != len(paths):
                raise ValidationError("Manifest frame_count does not match the frame set.")
            if not manifest.complete:
                raise ValidationError("The selected manifest is incomplete.")
            if manifest.complete and meta is None:
                raise ValidationError("A complete generation is missing slice_meta.json.")
        if manifest is not None and manifest.complete:
            expected_names = {f"frame_{index:04d}.png" for index in range(len(paths))}
            if {Path(path).name for path in paths} != expected_names:
                raise ValidationError("Complete runs must contain contiguous frame names.")

        width = height = 0
        total_pixels = 0
        total_bytes = 0
        for path_string in paths:
            path = Path(path_string)
            file_size = path.stat().st_size
            if file_size > MAX_FRAME_FILE_BYTES:
                raise ValidationError(f"Frame is too large: {path.name}.")
            total_bytes += file_size
            if total_bytes > MAX_FRAME_TOTAL_BYTES:
                raise ValidationError("Total encoded frame-set size exceeds the limit.")
            try:
                with Image.open(path) as image:
                    image.verify()
                    current_width, current_height = image.size
            except Exception as exc:
                raise ValidationError(f"Invalid PNG frame {path.name}: {exc}") from exc
            if current_width <= 0 or current_height <= 0:
                raise ValidationError(f"Frame {path.name} has invalid dimensions.")
            if current_width > MAX_FRAME_DIMENSION or current_height > MAX_FRAME_DIMENSION:
                raise ValidationError(f"Frame {path.name} exceeds the dimension limit.")
            if width == 0:
                width, height = current_width, current_height
            elif (current_width, current_height) != (width, height):
                raise ValidationError("All frames must have identical dimensions.")
            total_pixels += current_width * current_height
            if total_pixels > MAX_FRAME_TOTAL_PIXELS:
                raise ValidationError("Total decoded frame pixels exceed the limit.")

        grid_res = meta.grid_res if meta is not None else LEGACY_GRID_RES
        output_res = meta.output_res if meta is not None else height
        nz = max(int(round(height * grid_res / output_res)), 1)
        if nz > MAX_LAYERS:
            raise ValidationError(
                f"Frame-set depth is {nz:,} layers; limit is {MAX_LAYERS:,}."
            )
        estimated_bytes = total_pixels * 4
        if estimated_bytes > MAX_ESTIMATED_MEMORY_BYTES:
            raise ValidationError("Estimated frame-preview memory exceeds the limit.")
        if manifest is not None and manifest.frame_size != (width, height):
            raise ValidationError("Manifest frame_size does not match the frame set.")

        return FrameSetInfo(
            paths=paths,
            width=width,
            height=height,
            frame_count=len(paths),
            estimated_bytes=estimated_bytes,
            resolved_dir=str(directory),
            grid_res=grid_res,
            nz=nz,
            meta=meta,
        )


def list_frame_paths(out_dir: str) -> list[str]:
    """Return sorted frame paths, resolving a root to its latest complete run."""

    return FrameRepository.validate(out_dir).paths
