"""Physical projection field shared by the planner and frame export."""
from __future__ import annotations

import math


def detector_span_mm(outer_mm: float, container_mode: int) -> float:
    """Fixed field covering the round vial: its outer diameter at every angle.

    One field for every frame gives a single calibrated detector pitch.
    """
    if not math.isfinite(outer_mm) or outer_mm <= 0:
        raise ValueError("The outer container size must be finite and positive.")
    if container_mode not in (0, 2):
        raise ValueError("Unknown optical container mode.")
    return float(outer_mm)
