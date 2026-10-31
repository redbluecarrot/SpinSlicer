"""Exposure calibration in one print: a ladder of pillars planned at different doses.

Nine pillars stand on a circle around the axis. FastCAL plans each one at its own dose
level, from 12 % below to 12 % above the reference pillar (number 5) in 3 % steps, and the
print plan sets the print time so that the reference pillar reaches the gel dose the
settings assume. After printing, the lowest-dose pillar that came out solid tells how far
off that assumption is:

    real gel dose = assumed gel dose * (dose of the threshold) / (dose of pillar 5)

with the threshold halfway between the last pillar that stayed liquid and the first one
that cured. Pillar i is (4 + i) mm tall, so pillars can be told apart even when the
shorter, dimmer ones did not cure.

How the levels are planned: FastCAL pins voxels with partial coverage t (0.05 < t < 0.95)
to the dose d_low + (d_high - d_low) * t on a ramp (optics/fastcal.band_pqn). Filling a
whole pillar with coverage t therefore plans that pillar at that dose, uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

PILLARS = 9
REFERENCE = 4                       # zero-based index of pillar 5
LEVELS = np.linspace(0.15, 0.85, PILLARS)   # planned coverage -> dose 0.745 ... 0.955
RING_RADIUS_MM = 11.0
PILLAR_DIAMETER_MM = 3.5
BASE_HEIGHT_MM = 4.0                # pillar i (1-based) is BASE + i mm tall


@dataclass
class Ladder:
    """Calibration job: geometry, target levels and, after slicing, measured doses."""
    levels: np.ndarray = field(default_factory=lambda: LEVELS.copy())
    radius_mm: float = RING_RADIUS_MM
    diameter_mm: float = PILLAR_DIAMETER_MM

    def centres(self) -> np.ndarray:
        a = 2 * np.pi * np.arange(PILLARS) / PILLARS
        return np.stack([self.radius_mm * np.cos(a), self.radius_mm * np.sin(a)], axis=1)

    def heights(self) -> np.ndarray:
        return BASE_HEIGHT_MM + np.arange(1, PILLARS + 1, dtype=float)

    def mesh(self) -> Any:
        import trimesh
        parts = []
        top = self.heights().max()
        for (x, y), h in zip(self.centres(), self.heights()):
            c = trimesh.creation.cylinder(radius=self.diameter_mm / 2, height=h, sections=48)
            c.apply_translation((x, y, h / 2 - top / 2))
            parts.append(c)
        return trimesh.util.concatenate(parts)

    def pillar_masks(self, grid_res: int, span_mm: float) -> np.ndarray:
        """[pillar, x, y] masks of pixel centres inside each pillar (slightly widened)."""
        c = (np.arange(grid_res) + 0.5) * span_mm / grid_res - span_mm / 2
        X, Y = np.meshgrid(c, c, indexing="ij")
        r = self.diameter_mm / 2 + span_mm / grid_res
        return np.stack([(X - x) ** 2 + (Y - y) ** 2 <= r * r for x, y in self.centres()])

    def apply_levels(self, slices: np.ndarray, span_mm: float) -> None:
        """Scale the voxel coverage of pillar i by its level (in place)."""
        masks = self.pillar_masks(slices.shape[0], span_mm)
        for mask, level in zip(masks, self.levels):
            block = slices[mask]                                     # [pixels, z]
            scaled = block.astype(np.float32) * np.float32(level)
            slices[mask] = np.rint(scaled).astype(slices.dtype) if slices.dtype == np.uint8 else scaled

    def measure(self, engine: Any, slices: np.ndarray, displayed: Any) -> dict[str, Any]:
        """Median delivered dose of every pillar, from the frames as displayed."""
        from optics.fastcal import polar_to_cartesian_matrix
        N, NZ = engine.N, slices.shape[2]
        H = polar_to_cartesian_matrix(engine.op.grid, N, engine.span)
        masks = self.pillar_masks(N, engine.span).reshape(PILLARS, -1)
        full = slices.reshape(N * N, NZ) > 0
        samples: list[list[np.ndarray]] = [[] for _ in range(PILLARS)]
        layers = np.unique(np.linspace(0, NZ - 1, min(NZ, 48)).round().astype(int))
        for z0 in range(0, len(layers), 16):
            chunk = [int(z) for z in layers[z0:z0 + 16]]
            light = np.asarray(displayed(chunk), np.float32)
            D = engine.planner.dose(np.ascontiguousarray(light.transpose(2, 1, 0)))
            flat = np.asarray(H @ D.reshape(-1, len(chunk)))
            for i in range(PILLARS):
                core = masks[i][:, None] & full[:, chunk]
                if core.any():
                    samples[i].append(flat[core])
        doses = [float(np.median(np.concatenate(s))) if s else float("nan") for s in samples]
        ref = doses[REFERENCE]
        return {
            "pillars": PILLARS, "reference_pillar": REFERENCE + 1,
            "heights_mm": [float(h) for h in self.heights()],
            "relative_dose": [d / ref for d in doses],
            "reference_dose": ref,
        }


def gel_dose_factor(calibration: dict[str, Any], lowest_cured: int | None) -> float:
    """Factor for the gel dose setting (and the print time) from the print result.

    lowest_cured: 1-based number of the lowest-dose pillar that cured; None if none did.
    """
    rel = list(calibration["relative_dose"])
    step = (rel[-1] - rel[0]) / (len(rel) - 1)
    if lowest_cured is None:                   # even the brightest stayed liquid
        return rel[-1] + step / 2
    k = int(lowest_cured) - 1
    if not 0 <= k < len(rel):
        raise ValueError("pillar number out of range")
    if k == 0:                                 # even the dimmest cured
        return rel[0] - step / 2
    return 0.5 * (rel[k - 1] + rel[k])
