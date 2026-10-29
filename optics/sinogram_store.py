"""Optimised projections kept in memory or, for small machines, in block files.

The optimiser produces the projections one Z block at a time. In memory mode the
blocks simply stay in RAM. In disk mode each block is written to a temporary
file as soon as it is ready, and frames are read back later one angle at a time
(each block is stored angle-major, so one angle is a contiguous slice). The
numbers are identical in both modes.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np


class SinogramStore:
    """Blocks of shape (angles, detector, layers_in_block), float32."""

    def __init__(self, layers: int, detector: int, angles: int, directory: Optional[str] = None) -> None:
        self.layers = int(layers)
        self.detector = int(detector)
        self.angles = int(angles)
        self.maximum = 0.0
        self._blocks: list[tuple[int, int, object]] = []
        self._scratch: Optional[tempfile.TemporaryDirectory] = None
        if directory is not None:
            self._scratch = tempfile.TemporaryDirectory(prefix=".spinslicer-sino-", dir=directory)

    @property
    def on_disk(self) -> bool:
        return self._scratch is not None

    def add_block(self, start: int, stop: int, values: np.ndarray) -> None:
        """Store layers ``start:stop``; ``values`` is (angles * detector, layers)."""
        block = np.ascontiguousarray(values, dtype=np.float32).reshape(
            self.angles, self.detector, stop - start
        )
        self.maximum = max(self.maximum, float(block.max()))
        if self._scratch is None:
            self._blocks.append((start, stop, block))
            return
        path = os.path.join(self._scratch.name, f"block_{start:06d}.npy")
        np.save(path, block)
        self._blocks.append((start, stop, path))

    def frame(self, angle: int) -> np.ndarray:
        """Projection for one angle as (layers, detector)."""
        out = np.empty((self.layers, self.detector), dtype=np.float32)
        for start, stop, reference in self._blocks:
            if isinstance(reference, str):
                block = np.load(reference, mmap_mode="r")
                out[start:stop] = block[angle].T
                del block
            else:
                out[start:stop] = reference[angle].T  # type: ignore[index]
        return out

    def close(self) -> None:
        """Delete the scratch files (also done when the store is garbage collected)."""
        self._blocks.clear()
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None
