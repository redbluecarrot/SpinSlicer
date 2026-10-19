"""Physical projection field shared by the matrix builder and frame export."""
from __future__ import annotations

import math


def detector_span_mm(outer_mm: float, container_mode: int) -> float:
    """Fixed field covering the container at every rotation angle.

    A square of side L has projected width L*(abs(cos theta)+abs(sin theta)),
    whose maximum is sqrt(2)*L. Keeping this maximum for every frame gives
    a single calibrated detector pitch instead of changing scale by angle.
    """
    if not math.isfinite(outer_mm) or outer_mm <= 0:
        raise ValueError("The outer container size must be finite and positive.")
    if container_mode not in (0, 1, 2):
        raise ValueError("Unknown optical container mode.")
    return outer_mm * math.sqrt(2.0) if container_mode == 1 else outer_mm
