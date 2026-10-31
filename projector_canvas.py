"""
projector_canvas.py
===================
Where a frame lands on the projector's screen.

The engine produces frames as wide as the vial (``output_res`` pixels) and as
tall as the part. A projector shows a whole screen (for example 1920×1080).
If such a strip is played full screen the player stretches it: the picture goes
soft and the part prints at the wrong size. Here every frame is placed 1:1
on a black canvas of the projector's native resolution, centred on the vial
axis plus an optional alignment offset.

Also: the projector's throw ratio → distance to the vial axis (for the cone of
rays the engine traces), and how much the vertical part of that cone smears
the top and bottom of the part.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RESOLUTIONS: tuple[tuple[int, int], ...] = ((1280, 720), (1920, 1080), (2560, 1440), (3840, 2160))


@dataclass(frozen=True)
class CanvasGeometry:
    width: int = 1920
    height: int = 1080
    offset_x: int = 0       # px, + moves the picture right
    offset_y: int = 0       # px, + moves the picture down

    def placement(self, frame_w: int, frame_h: int) -> tuple[int, int]:
        """Top-left corner of a frame on the canvas (may be partly outside)."""
        x = (self.width - frame_w) // 2 + int(self.offset_x)
        y = (self.height - frame_h) // 2 + int(self.offset_y)
        return x, y

    def fits(self, frame_w: int, frame_h: int) -> bool:
        x, y = self.placement(frame_w, frame_h)
        return x >= 0 and y >= 0 and x + frame_w <= self.width and y + frame_h <= self.height


def compose(frame: np.ndarray, geometry: CanvasGeometry) -> np.ndarray:
    """The frame pasted 1:1 onto a black canvas; parts outside the screen are cut."""
    canvas = np.zeros((geometry.height, geometry.width), dtype=np.uint8)
    h, w = frame.shape[:2]
    x, y = geometry.placement(w, h)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, geometry.width), min(y + h, geometry.height)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = frame[y0 - y:y1 - y, x0 - x:x1 - x]
    return canvas


def image_width_mm(vial_diameter_mm: float, vial_px: int, screen_w: int) -> float:
    """Width of the whole projected picture at the vial axis."""
    return float(vial_diameter_mm) * float(screen_w) / max(int(vial_px), 1)


def throw_distance_mm(throw_ratio: float, vial_diameter_mm: float, vial_px: int, screen_w: int) -> float:
    """Throw ratio (distance / picture width, from the projector datasheet) → distance to the axis."""
    return float(throw_ratio) * image_width_mm(vial_diameter_mm, vial_px, screen_w)


def cone_smear_mm(throw_mm: float, half_height_mm: float, inner_radius_mm: float) -> float:
    """Worst vertical smear from the cone of rays.

    A row aimed at height h on the axis plane is at h·(d ± r)/d at depth ±r. As the
    vial turns every point sees both, so the top and bottom of the part blur by about
    this much. Zero for parallel rays; smaller when the projector is farther away and
    when the beam centre is at mid-height of the part.
    """
    if throw_mm <= 0:
        return 0.0
    return float(half_height_mm) * float(inner_radius_mm) / float(throw_mm)
