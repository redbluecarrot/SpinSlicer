"""The ``.spin`` print file (version 1) shared by SpinSlicer and SpinPrinter.

One file per print: an uncompressed (stored) ZIP, like ``.3mf`` in FDM printing, so
any archiver opens it and every language reads it with its standard library::

    part.spin
    ├── spin.json      metadata (required, always the first entry)
    ├── video.mp4      one revolution of the vial, H.264 (required)
    ├── preview.png    320×320 picture of the part (optional)
    └── slicer.json    the slicing settings, to redo the slice (optional)

The specification lives in SpinPrinter ``docs/ecosystem.ru.md`` (section 3).
Readers ignore fields they do not know, so new fields can be added at any time;
``version`` grows only on incompatible changes.

Frames and vial angle
---------------------
The projection code (``optics.ray_tracing._co_rotating``, ``optics.fastcal``) turns
the lab rays by ``-θ`` into the vial frame, i.e. the vial turns by ``+θ`` about +z
(z up, the top row of a frame is the top of the vial). Frame ``m`` is made for
``θ_m = angle_offset + direction · m · 360° / N``. With ``direction = +1`` the vial
turns counter-clockwise seen from above (``ccw_from_top``), with ``-1`` clockwise.
``angle_offset`` only places the model in the vial's own frame; the printer starts
every print at its zero angle, so frame 0 belongs to vial angle 0
(``frame0_angle_deg = 0``). A mirror in the light path (rear projection, a fold
mirror) flips the image and with it the apparent direction: the printer's motor
direction setting covers that.

The same plan, flattened, also goes into the MP4 ``comment`` tag of a plain video
export (``"format": "spinslicer-print"``), which SpinPi reads in ``probe()``.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SPIN_FORMAT = "spin"
SPIN_VERSION = 1
SPIN_EXTENSION = ".spin"
MP4_TAG_FORMAT = "spinslicer-print"

MANIFEST_NAME = "spin.json"
VIDEO_NAME = "video.mp4"
PREVIEW_NAME = "preview.png"
SLICER_NAME = "slicer.json"
PREVIEW_SIZE = 320

ROTATIONS = ("ccw_from_top", "cw_from_top")


class SpinFormatError(ValueError):
    """A ``.spin`` file that cannot be written or read."""


def rotation_name(angle_direction: int) -> str:
    """``video.rotation`` for the engine's ``angle_direction`` (+1 / -1), see the module doc."""
    if int(angle_direction) not in (1, -1):
        raise SpinFormatError("angle_direction must be 1 or -1.")
    return "ccw_from_top" if int(angle_direction) == 1 else "cw_from_top"


def generator_name() -> str:
    try:
        from importlib.metadata import version

        return f"SpinSlicer {version('spinslicer')}"
    except Exception:  # noqa: BLE001 - running from a source checkout
        pass
    try:
        import tomllib

        root = Path(__file__).resolve().parent.parent
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        return f"SpinSlicer {data['project']['version']}"
    except Exception:  # noqa: BLE001
        return "SpinSlicer"


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class PrintJob:
    """Everything the printer needs to know about one print, gathered from the slicer."""

    name: str
    frames: int                     # frames in the video
    fps: float
    width: int                      # video size (the projector screen)
    height: int
    frames_per_rev: int = 0         # 0 = frames (one video = one revolution)
    angle_direction: int = 1
    frame0_angle_deg: float = 0.0
    plan: dict[str, Any] = field(default_factory=dict)   # print_plan.json (compute_print_plan)
    exposure: float = 1.0           # the print page's exposure slider (1 = the plan)
    brightness: float = 1.0         # projector brightness baked into the video (0.05…1)
    resin: dict[str, Any] = field(default_factory=dict)
    printer: dict[str, Any] = field(default_factory=dict)
    # the centering shift already in the frames (CanvasGeometry offset_x / offset_y). SpinPi applies
    # its own calibration minus this, so the picture is not shifted twice.
    offset_px: tuple[int, int] = (0, 0)

    @property
    def revolution_s(self) -> float:
        per_rev = self.frames_per_rev or self.frames
        return per_rev / self.fps if self.fps > 0 else 0.0

    def plan_section(self) -> dict[str, Any]:
        """``plan`` of spin.json: what the printer runs, for the video as exported.

        compute_print_plan times the dose at full brightness; the exported video is
        dimmed by ``brightness`` and the user may push the exposure, so the number of
        revolutions is scaled the same way as the print page's duration.
        """
        revolution_s = self.revolution_s
        section: dict[str, Any] = {
            "revolution_s": revolution_s,
            "rpm": 60.0 / revolution_s if revolution_s > 0 else None,
            "tolerance_pct": _finite(self.plan.get("exposure_tolerance_pct")) or 0.0,
            "warnings": [str(w) for w in self.plan.get("warnings", [])],
        }
        total = _finite(self.plan.get("frame_exposure_total_s"))
        if total and total > 0 and self.fps > 0:
            scale = max(float(self.exposure), 0.0) / max(float(self.brightness), 0.05)
            # frame_exposure_total_s is the light time of each frame over the whole print;
            # every revolution shows each frame for 1/fps.
            revolutions = total * self.fps * scale
            section["revolutions"] = revolutions
            section["print_time_s"] = revolutions * revolution_s
            section["exposure_factor"] = float(self.exposure)
            section["brightness"] = float(self.brightness)
        else:
            section["revolutions"] = None
            section["print_time_s"] = None
        return section

    def video_section(self, sha256: str = "", file: str = VIDEO_NAME) -> dict[str, Any]:
        return {
            "file": file,
            "sha256": sha256,
            "frames": int(self.frames),
            "fps": float(self.fps),
            "width": int(self.width),
            "height": int(self.height),
            "frames_per_rev": int(self.frames_per_rev or self.frames),
            "rotation": rotation_name(self.angle_direction),
            "frame0_angle_deg": float(self.frame0_angle_deg),
            "offset_px": [int(self.offset_px[0]), int(self.offset_px[1])],
        }

    def manifest(self, video_sha256: str, created: Optional[str] = None) -> dict[str, Any]:
        """spin.json."""
        return {
            "format": SPIN_FORMAT,
            "version": SPIN_VERSION,
            "created": created or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator": generator_name(),
            "name": self.name,
            "video": self.video_section(video_sha256),
            "plan": self.plan_section(),
            "resin": dict(self.resin),
            "printer": dict(self.printer),
        }

    def mp4_tag(self) -> dict[str, Any]:
        """The flat plan for the MP4 ``comment`` tag / ``<name>.spin.json`` (SpinPi probe())."""
        plan = self.plan_section()
        video = self.video_section()
        video.pop("file")
        video.pop("sha256")
        tag: dict[str, Any] = {"format": MP4_TAG_FORMAT, "version": SPIN_VERSION,
                               "generator": generator_name(), "name": self.name}
        tag.update(plan)
        tag.update(video)
        if self.resin:
            tag["resin"] = dict(self.resin)
        if self.printer.get("profile_id"):
            tag["printer_profile_id"] = self.printer["profile_id"]
        return tag


def mp4_comment(job: PrintJob) -> str:
    """JSON for ffmpeg ``-metadata comment=…`` (ASCII so every tool keeps it intact)."""
    return json.dumps(job.mp4_tag(), ensure_ascii=True, separators=(",", ":"))


def write_spin(path: str | os.PathLike[str], job: PrintJob, video_path: str | os.PathLike[str], *,
               preview_png: Optional[bytes] = None, slicer_settings: Optional[dict[str, Any]] = None,
               created: Optional[str] = None) -> dict[str, Any]:
    """Write a ``.spin`` file atomically and return its spin.json."""
    video_path = Path(video_path)
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise SpinFormatError(f"No video to pack: {video_path}")
    manifest = job.manifest(sha256_file(video_path), created=created)
    target = Path(path)
    if not target.parent.is_dir():
        raise SpinFormatError(f"Folder does not exist: {target.parent}")
    fd, temporary = tempfile.mkstemp(prefix=".spinslicer-", suffix=SPIN_EXTENSION, dir=str(target.parent))
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.write(video_path, VIDEO_NAME)
            if preview_png:
                archive.writestr(PREVIEW_NAME, preview_png)
            if slicer_settings is not None:
                archive.writestr(SLICER_NAME, json.dumps(slicer_settings, ensure_ascii=False, indent=2,
                                                         default=str))
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return manifest


@dataclass
class SpinFile:
    manifest: dict[str, Any]
    names: list[str]
    path: str

    def read(self, name: str) -> bytes:
        with zipfile.ZipFile(self.path) as archive:
            return archive.read(name)


def read_spin(path: str | os.PathLike[str], verify: bool = True) -> SpinFile:
    """Open a ``.spin`` file and check what the format promises (sha256 when ``verify``)."""
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SpinFormatError(f"Not a .spin file: {exc}") from exc
    with archive:
        names = archive.namelist()
        if not names or names[0] != MANIFEST_NAME:
            raise SpinFormatError("spin.json must be the first entry.")
        try:
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        except ValueError as exc:
            raise SpinFormatError(f"spin.json is not valid JSON: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("format") != SPIN_FORMAT:
            raise SpinFormatError("spin.json: format is not \"spin\".")
        if manifest.get("version") != SPIN_VERSION:
            raise SpinFormatError(f"Unsupported .spin version {manifest.get('version')!r}.")
        video = manifest.get("video") or {}
        video_name = str(video.get("file", VIDEO_NAME))
        if video_name not in names:
            raise SpinFormatError(f"{video_name} is missing.")
        if verify and video.get("sha256"):
            digest = hashlib.sha256()
            try:
                with archive.open(video_name) as handle:
                    for block in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(block)
            except (OSError, zipfile.BadZipFile) as exc:      # the ZIP's own CRC caught it first
                raise SpinFormatError(f"The video is damaged: {exc}") from exc
            if digest.hexdigest() != video["sha256"]:
                raise SpinFormatError("The video's sha256 does not match spin.json (damaged copy).")
    return SpinFile(manifest, names, str(path))


# ------------------------------------------------------------------ preview
def render_preview(vertices: Any, faces: Any, size: int = PREVIEW_SIZE) -> bytes:
    """Shaded 3/4 view of a mesh as PNG bytes (no GPU: numpy + Pillow)."""
    import numpy as np
    from PIL import Image, ImageDraw

    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    if not len(v) or not len(f):
        return _png(image)
    # Camera: 35° around z, 25° down; z stays up in the picture.
    a, b = math.radians(35.0), math.radians(25.0)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    p = (v - v.mean(axis=0)) @ rz.T @ rx.T          # x right, y towards the camera… z up
    span = max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 2])), 1e-9)
    scale = 0.86 * size / span
    centre = np.array([(p[:, 0].max() + p[:, 0].min()) / 2, (p[:, 2].max() + p[:, 2].min()) / 2])
    sx = (p[:, 0] - centre[0]) * scale + size / 2
    sy = size / 2 - (p[:, 2] - centre[1]) * scale
    tri = p[f]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 0
    normals[keep] /= lengths[keep, None]
    light = np.array([-0.35, -0.75, 0.55])
    light /= np.linalg.norm(light)
    shade = 0.28 + 0.72 * np.abs(normals @ light)
    depth = tri[:, :, 1].mean(axis=1)               # larger y = farther from the camera
    base = np.array([96, 170, 255], dtype=np.float64)
    colours = np.clip(base[None, :] * shade[:, None], 0, 255).astype(np.uint8)
    if len(f) <= 150_000:
        draw = ImageDraw.Draw(image)
        for i in np.argsort(-depth):                # painter's algorithm: far faces first
            if not keep[i]:
                continue
            a0, a1, a2 = f[i]
            r, g, bl = (int(c) for c in colours[i])
            draw.polygon([(sx[a0], sy[a0]), (sx[a1], sy[a1]), (sx[a2], sy[a2])], fill=(r, g, bl, 255))
    else:
        # Dense meshes: faces are smaller than a pixel, a z-buffered splat is enough.
        cx = np.clip(sx[f].mean(axis=1).astype(np.int64), 0, size - 1)
        cy = np.clip(sy[f].mean(axis=1).astype(np.int64), 0, size - 1)
        order = np.argsort(-depth)                  # nearest last, so it wins
        buffer = np.zeros((size, size, 4), dtype=np.uint8)
        buffer[cy[order], cx[order], :3] = colours[order]
        buffer[cy[order], cx[order], 3] = 255
        image = Image.fromarray(buffer, "RGBA")
    return _png(image)


def preview_from_frame(frame: Any, size: int = PREVIEW_SIZE) -> bytes:
    """Fallback preview: a projector frame, letterboxed into a square."""
    import numpy as np
    from PIL import Image

    picture = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    picture.thumbnail((size, size))
    square = Image.new("L", (size, size), 0)
    square.paste(picture, ((size - picture.width) // 2, (size - picture.height) // 2))
    return _png(square)


def _png(image: Any) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def spin_name(name: str) -> str:
    """A file name for the printer: no folders, safe characters, ``.spin`` at the end."""
    base = os.path.basename(str(name).replace("\\", "/")).strip()
    if base.lower().endswith(SPIN_EXTENSION):
        base = base[: -len(SPIN_EXTENSION)]
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in base).strip("._") or "print"
    return safe[:100] + SPIN_EXTENSION
