"""Validated machine and material profiles for offline and real VAM jobs.

The current application can generate projection frames without knowing the
eventual printer.  These profiles provide the boundary between that numerical
pipeline and a future projector/rotation-stage driver.  They are deliberately
plain dataclasses so the same objects can be stored in a manifest, edited as
JSON, and consumed by a virtual device during development.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from constants import MAX_FRAME_DIMENSION

PROFILE_SCHEMA_VERSION = 1
MIN_PROFILE_FPS = 0.1
MAX_PROFILE_FPS = 240.0


class ProfileValidationError(ValueError):
    """Raised when a machine or resin profile is unsafe or inconsistent."""


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProfileValidationError(f"{name} must be a number.") from exc
    if not math.isfinite(number):
        raise ProfileValidationError(f"{name} must be finite.")
    return number


def _positive(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ProfileValidationError(f"{name} must be greater than zero.")
    return number


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ProfileValidationError(f"{name} must be an integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ProfileValidationError(f"{name} must be an integer.") from exc
    if integer != value:
        raise ProfileValidationError(f"{name} must be an integer.")
    return integer


@dataclass(frozen=True)
class ProjectorProfile:
    """Native projector properties needed to render a frame sequence."""

    name: str = "virtual-projector"
    width_px: int = 720
    height_px: int = 720
    bit_depth: int = 8
    fps: float = 24.0
    gamma: float = 1.0
    rotate_deg: int = 0
    mirror_x: bool = False
    mirror_y: bool = False

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ProfileValidationError("projector name must be a non-empty string.")
        width = _integer(self.width_px, "projector width_px")
        height = _integer(self.height_px, "projector height_px")
        if not 16 <= width <= MAX_FRAME_DIMENSION:
            raise ProfileValidationError("projector width_px is outside the supported range.")
        if not 16 <= height <= MAX_FRAME_DIMENSION:
            raise ProfileValidationError("projector height_px is outside the supported range.")
        bit_depth = _integer(self.bit_depth, "projector bit_depth")
        if bit_depth not in {8, 10, 12, 16}:
            raise ProfileValidationError("projector bit_depth must be 8, 10, 12, or 16.")
        fps = _finite(self.fps, "projector fps")
        if not MIN_PROFILE_FPS <= fps <= MAX_PROFILE_FPS:
            raise ProfileValidationError("projector fps is outside the supported range.")
        gamma = _positive(self.gamma, "projector gamma")
        if gamma > 10.0:
            raise ProfileValidationError("projector gamma is outside the supported range.")
        rotation = _integer(self.rotate_deg, "projector rotate_deg")
        if rotation not in {0, 90, 180, 270}:
            raise ProfileValidationError("projector rotate_deg must be 0, 90, 180, or 270.")
        if not isinstance(self.mirror_x, bool) or not isinstance(self.mirror_y, bool):
            raise ProfileValidationError("projector mirror flags must be boolean.")


@dataclass(frozen=True)
class VatProfile:
    """Physical vat and rotation-axis geometry, independent of a driver."""

    name: str = "virtual-vat"
    diameter_mm: float = 60.0
    height_mm: float = 96.0
    axis_offset_x_mm: float = 0.0
    axis_offset_y_mm: float = 0.0
    angle_zero_deg: float = 0.0

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ProfileValidationError("vat name must be a non-empty string.")
        diameter = _positive(self.diameter_mm, "vat diameter_mm")
        height = _positive(self.height_mm, "vat height_mm")
        if height < diameter * 0.1:
            raise ProfileValidationError("vat height_mm is implausibly small for its diameter.")
        if abs(_finite(self.axis_offset_x_mm, "vat axis_offset_x_mm")) > diameter:
            raise ProfileValidationError("vat axis_offset_x_mm is outside the vat.")
        if abs(_finite(self.axis_offset_y_mm, "vat axis_offset_y_mm")) > diameter:
            raise ProfileValidationError("vat axis_offset_y_mm is outside the vat.")
        _finite(self.angle_zero_deg, "vat angle_zero_deg")


@dataclass(frozen=True)
class MachineProfile:
    """Complete machine-side contract for a projection job."""

    name: str = "offline-vam"
    projector: ProjectorProfile = ProjectorProfile()
    vat: VatProfile = VatProfile()
    angle_offset_deg: float = 90.0
    angle_direction: int = 1

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ProfileValidationError("machine name must be a non-empty string.")
        self.projector.validate()
        self.vat.validate()
        _finite(self.angle_offset_deg, "machine angle_offset_deg")
        direction = _integer(self.angle_direction, "machine angle_direction")
        if direction not in {-1, 1}:
            raise ProfileValidationError("machine angle_direction must be -1 or 1.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MachineProfile":
        if not isinstance(payload, dict):
            raise ProfileValidationError("machine profile must be an object.")
        projector_payload = payload.get("projector", {})
        vat_payload = payload.get("vat", {})
        if not isinstance(projector_payload, dict) or not isinstance(vat_payload, dict):
            raise ProfileValidationError("machine projector and vat must be objects.")
        profile = cls(
            name=payload.get("name", cls.name),
            projector=ProjectorProfile(**projector_payload),
            vat=VatProfile(**vat_payload),
            angle_offset_deg=payload.get("angle_offset_deg", cls.angle_offset_deg),
            angle_direction=payload.get("angle_direction", cls.angle_direction),
        )
        profile.validate()
        return profile


@dataclass(frozen=True)
class ResinProfile:
    """Material-side settings; values remain provisional until calibration."""

    name: str = "unassigned-resin"
    base_exposure_s: float = 1.0
    intensity_pct: float = 100.0
    threshold_pct: float = 0.0
    notes: str = "Uncalibrated offline profile"

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ProfileValidationError("resin name must be a non-empty string.")
        if _positive(self.base_exposure_s, "resin base_exposure_s") > 3600.0:
            raise ProfileValidationError("resin base_exposure_s is outside the supported range.")
        intensity = _positive(self.intensity_pct, "resin intensity_pct")
        if intensity > 1000.0:
            raise ProfileValidationError("resin intensity_pct is outside the supported range.")
        threshold = _finite(self.threshold_pct, "resin threshold_pct")
        if not 0.0 <= threshold <= 100.0:
            raise ProfileValidationError("resin threshold_pct must be between 0 and 100.")
        if not isinstance(self.notes, str):
            raise ProfileValidationError("resin notes must be a string.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResinProfile":
        if not isinstance(payload, dict):
            raise ProfileValidationError("resin profile must be an object.")
        profile = cls(**payload)
        profile.validate()
        return profile


ProfileType = TypeVar("ProfileType", MachineProfile, ResinProfile)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_profile(path: str, profile: MachineProfile | ResinProfile) -> None:
    """Save a validated profile with an explicit schema marker."""

    if isinstance(profile, MachineProfile):
        profile_type = "machine"
        payload = profile.to_dict()
    elif isinstance(profile, ResinProfile):
        profile_type = "resin"
        profile.validate()
        payload = profile.to_dict()
    else:
        raise TypeError("profile must be a MachineProfile or ResinProfile.")
    _atomic_json_write(
        Path(path),
        {"schema_version": PROFILE_SCHEMA_VERSION, "profile_type": profile_type, "profile": payload},
    )


def load_profile(path: str) -> MachineProfile | ResinProfile:
    """Load and validate a machine or resin profile from JSON."""

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ProfileValidationError("profile root must be an object.")
        if raw.get("schema_version") != PROFILE_SCHEMA_VERSION:
            raise ProfileValidationError("unsupported profile schema version.")
        payload = raw.get("profile")
        profile_type = raw.get("profile_type")
        if not isinstance(payload, dict):
            raise ProfileValidationError("profile payload must be an object.")
        if profile_type == "machine":
            return MachineProfile.from_dict(payload)
        if profile_type == "resin":
            return ResinProfile.from_dict(payload)
        raise ProfileValidationError("profile_type must be machine or resin.")
    except ProfileValidationError:
        raise
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProfileValidationError(f"invalid profile: {exc}") from exc
