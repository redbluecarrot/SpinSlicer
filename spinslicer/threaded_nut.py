"""Parametric threaded-nut geometry for tomographic VAM experiments.

The generator deliberately builds the nut as a closed parametric shell instead
of relying on a boolean CAD backend.  That keeps the feature portable on the
same machines that run SpinSlicer and preserves the helical internal thread
when the mesh is voxelized for projection generation.

The resulting mesh is a single solid ring.  Its inner radius follows a
triangular helical profile, so every axial section contains the thread relief
and every turn is phase-shifted around the bore.  This is a useful printable
approximation for VAM; it is not intended to replace a standards-compliant CAD
thread kernel for tolerance-critical hardware.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import trimesh


@dataclass(frozen=True)
class ThreadedNutParameters:
    """Dimensions of a helical internal-thread nut, in millimetres.

    ``bore_diameter_mm`` is the maximum (thread-valley) diameter of the
    internal thread before radial clearance is added.  ``thread_depth_mm`` is
    the radial height of the triangular thread crest.  ``clearance_mm`` is
    radial clearance added to the bore so a printed mating screw has room.
    """

    outer_diameter_mm: float = 24.0
    bore_diameter_mm: float = 12.0
    height_mm: float = 10.0
    pitch_mm: float = 2.0
    thread_depth_mm: float = 0.75
    clearance_mm: float = 0.15
    segments_per_turn: int = 128
    axial_segments_per_pitch: int = 24
    phase: float = 0.0

    def validate(self) -> None:
        """Validate dimensions before allocating the parametric mesh."""

        numeric = {
            "outer_diameter_mm": self.outer_diameter_mm,
            "bore_diameter_mm": self.bore_diameter_mm,
            "height_mm": self.height_mm,
            "pitch_mm": self.pitch_mm,
            "thread_depth_mm": self.thread_depth_mm,
            "clearance_mm": self.clearance_mm,
            "phase": self.phase,
        }
        for name, value in numeric.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

        if self.outer_diameter_mm <= 0.0:
            raise ValueError("outer_diameter_mm must be greater than zero.")
        if self.bore_diameter_mm <= 0.0:
            raise ValueError("bore_diameter_mm must be greater than zero.")
        if self.height_mm <= 0.0:
            raise ValueError("height_mm must be greater than zero.")
        if self.pitch_mm <= 0.0:
            raise ValueError("pitch_mm must be greater than zero.")
        if self.thread_depth_mm <= 0.0:
            raise ValueError("thread_depth_mm must be greater than zero.")
        if self.clearance_mm < 0.0:
            raise ValueError("clearance_mm cannot be negative.")
        if self.segments_per_turn < 24 or self.segments_per_turn > 1024:
            raise ValueError("segments_per_turn must be between 24 and 1024.")
        if self.axial_segments_per_pitch < 6 or self.axial_segments_per_pitch > 256:
            raise ValueError("axial_segments_per_pitch must be between 6 and 256.")

        outer_radius = self.outer_diameter_mm / 2.0
        major_radius = self.bore_diameter_mm / 2.0 + self.clearance_mm
        root_radius = major_radius - self.thread_depth_mm
        if major_radius >= outer_radius:
            raise ValueError(
                "The outer diameter must be larger than the threaded bore "
                "diameter plus a wall."
            )
        if root_radius <= 0.0:
            raise ValueError("Thread depth and bore diameter leave no central opening.")

    def as_dict(self) -> dict[str, float | int]:
        """Return JSON-friendly generation parameters for a manifest."""

        return dict(asdict(self))


def create_threaded_nut(params: ThreadedNutParameters) -> trimesh.Trimesh:
    """Create a watertight mesh with a triangular helical internal thread.

    The mesh has two periodic cylindrical surfaces (outer wall and threaded
    bore) joined by planar end annuli.  The shared periodic topology avoids
    boolean-library dependencies and makes the geometry deterministic, which
    is valuable for comparing VAM projection algorithms.
    """

    params.validate()

    n_theta = int(params.segments_per_turn)
    n_z = max(
        2,
        int(math.ceil(params.height_mm / params.pitch_mm * params.axial_segments_per_pitch))
        + 1,
    )
    # Keep accidental very fine settings from allocating an unbounded mesh.
    n_z = min(n_z, 8193)

    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False, dtype=np.float64)
    z = np.linspace(
        -params.height_mm / 2.0,
        params.height_mm / 2.0,
        n_z,
        dtype=np.float64,
    )

    # A triangular wave has a sharp, well-defined crest and a wide valley.
    # The z - pitch * theta/(2π) term advances one turn per pitch.
    phase = (
        (z[:, None] - params.pitch_mm * theta[None, :] / (2.0 * math.pi))
        / params.pitch_mm
        + params.phase
    ) % 1.0
    triangular = 1.0 - np.abs(2.0 * phase - 1.0)

    outer_radius = params.outer_diameter_mm / 2.0
    major_radius = params.bore_diameter_mm / 2.0 + params.clearance_mm
    inner_radius = major_radius - params.thread_depth_mm * triangular

    cos_theta = np.cos(theta)[None, :]
    sin_theta = np.sin(theta)[None, :]
    outer = np.stack(
        (
            np.broadcast_to(outer_radius * cos_theta, (n_z, n_theta)),
            np.broadcast_to(outer_radius * sin_theta, (n_z, n_theta)),
            np.broadcast_to(z[:, None], (n_z, n_theta)),
        ),
        axis=-1,
    )
    inner = np.stack(
        (
            inner_radius * cos_theta,
            inner_radius * sin_theta,
            np.broadcast_to(z[:, None], (n_z, n_theta)),
        ),
        axis=-1,
    )

    outer_flat = outer.reshape(-1, 3)
    inner_flat = inner.reshape(-1, 3)
    vertices = np.vstack((outer_flat, inner_flat))
    outer_indices = np.arange(n_z * n_theta, dtype=np.int64).reshape(n_z, n_theta)
    inner_indices = outer_indices + n_z * n_theta

    i = np.arange(n_theta, dtype=np.int64)[None, :]
    j = np.arange(n_z - 1, dtype=np.int64)[:, None]
    i_next = (i + 1) % n_theta

    oa = outer_indices[j, i].ravel()
    ob = outer_indices[j, i_next].ravel()
    oc = outer_indices[j + 1, i_next].ravel()
    od = outer_indices[j + 1, i].ravel()
    ia = inner_indices[j, i].ravel()
    ib = inner_indices[j, i_next].ravel()
    ic = inner_indices[j + 1, i_next].ravel()
    id_ = inner_indices[j + 1, i].ravel()

    # Outward normals: outer cylinder points out, the bore points into the
    # hole.  The end annuli close the volume and remove open-boundary artifacts
    # during voxelization.
    faces = [
        np.column_stack((oa, ob, oc)),
        np.column_stack((oa, oc, od)),
        np.column_stack((ia, id_, ic)),
        np.column_stack((ia, ic, ib)),
    ]

    outer_bottom = outer_indices[0]
    inner_bottom = inner_indices[0]
    outer_top = outer_indices[-1]
    inner_top = inner_indices[-1]
    ob0 = outer_bottom
    ob1 = np.roll(outer_bottom, -1)
    ib0 = inner_bottom
    ib1 = np.roll(inner_bottom, -1)
    ot0 = outer_top
    ot1 = np.roll(outer_top, -1)
    it0 = inner_top
    it1 = np.roll(inner_top, -1)

    # Top normal is +Z; bottom normal is -Z.
    faces.extend(
        [
            np.column_stack((ot0, ot1, it1)),
            np.column_stack((ot0, it1, it0)),
            np.column_stack((ob0, ib0, ib1)),
            np.column_stack((ob0, ib1, ob1)),
        ]
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.vstack(faces).astype(np.int64, copy=False),
        process=False,
    )
    # The parametric construction is already centered; keep this explicit so
    # callers can rely on the same contract as synthetic_shapes.py.
    mesh.apply_translation(-mesh.bounding_box.centroid)
    return mesh


def create_default_threaded_nut() -> trimesh.Trimesh:
    """Convenience factory for the UI preview and smoke tests."""

    return create_threaded_nut(ThreadedNutParameters())
