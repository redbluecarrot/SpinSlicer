"""
sketch_scene.py
===============
The printer drawn "by hand" inside the 3D viewport.

Everything here is driven by real settings: the vial grows with its diameter,
the wall with its thickness, the beam with the projector field height, lens
throw, resin absorption and refraction at the vial wall. Lines are jittered
like a marker sketch; three jitter variants are swapped a few times a second
("line boil"), which only toggles actor visibility, so it is cheap even on weak
GPUs. The model itself is drawn with an ordered-dither shader (1-bit print look).

The module knows nothing about Qt: Viewport3D owns the timer and calls
``tick()``; ``on_camera()`` refreshes the camera-dependent cylinder outline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyvista as pv

BOIL_VARIANTS = 3


@dataclass(frozen=True)
class ScenePalette:
    background: str
    sketch: str
    accent: str
    muted: str
    resin: str
    liquid: str
    dither_ink: str
    dither_paper: str
    warn: str
    hatch: bool = False           # day theme: beam drawn as pen hatching, not a wash


@dataclass(frozen=True)
class SceneParams:
    """Physical inputs of the drawing (all in mm unless noted)."""

    diameter_mm: float = 50.0
    wall_mm: float = 2.0
    bath: bool = True
    field_mm: float = 30.0
    throw_mm: float = 0.0          # 0 = telecentric (parallel beam)
    alpha_per_mm: float = 0.0
    n_resin: float = 1.47
    n_liquid: float = 1.45
    height_ratio: float = 1.6

    @property
    def radius(self) -> float:
        return self.diameter_mm / 2.0

    @property
    def inner_radius(self) -> float:
        return max(self.radius - self.wall_mm, 0.5)

    @property
    def height(self) -> float:
        return self.diameter_mm * self.height_ratio

    @property
    def bend(self) -> float:
        """Relative narrowing of the light sheet across the vial (refraction proxy)."""
        outside = self.n_liquid if self.bath else 1.0
        return min(0.6, abs(self.n_resin - outside) * 0.95)

    @property
    def far_wall_light(self) -> float:
        return float(math.exp(-self.alpha_per_mm * 2.0 * self.inner_radius))

    @property
    def bend_degrees(self) -> float:
        outside = self.n_liquid if self.bath else 1.0
        return abs(self.n_resin - outside) * 20.0


def projector_x(p: SceneParams) -> float:
    return -(p.radius + max(0.6 * p.diameter_mm, 24.0))


def scene_bounds(p: SceneParams) -> tuple[np.ndarray, np.ndarray]:
    half = p.height / 2.0
    lo = np.array([projector_x(p) - 18.0, -p.radius * 1.5, -half - 6.0])
    hi = np.array([p.radius * 1.5, p.radius * 1.5, half + 4.0])
    return lo, hi


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def _circle(cx: float, cy: float, z: float, r: float, n: int = 72,
            a0: float = 0.0, a1: float = 2 * math.pi) -> np.ndarray:
    t = np.linspace(a0, a1, n)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t), np.full(n, z)])


def _segment(a: tuple[float, float, float], b: tuple[float, float, float], n: int = 8) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.asarray(a) * (1 - t) + np.asarray(b) * t


def _dashed(points: np.ndarray, on: int = 3, off: int = 3) -> list[np.ndarray]:
    out = []
    i = 0
    while i < len(points) - 1:
        out.append(points[i:i + on + 1])
        i += on + off
    return out


def _jitter(lines: list[np.ndarray], amplitude: float, seed: int) -> list[np.ndarray]:
    """Smooth, low-frequency wobble (a marker line, not noise)."""
    rng = np.random.default_rng(seed)
    out = []
    for pts in lines:
        n = len(pts)
        if n < 2:
            out.append(pts)
            continue
        s = np.linspace(0.0, 1.0, n)[:, None]
        phases = rng.uniform(0, 2 * math.pi, size=(2, 3))
        freqs = rng.uniform(1.0, 3.0, size=(2, 1))
        wobble = (np.sin(2 * math.pi * freqs[0] * s + phases[0]) * 0.65
                  + np.sin(2 * math.pi * freqs[1] * s + phases[1]) * 0.35)
        out.append(pts + amplitude * wobble + rng.normal(0, amplitude * 0.15, size=pts.shape))
    return out


def _polylines(lines: list[np.ndarray]) -> pv.PolyData:
    if not lines:
        return pv.PolyData()
    points = np.vstack(lines)
    cells = []
    offset = 0
    for pts in lines:
        cells.append(np.concatenate([[len(pts)], np.arange(offset, offset + len(pts))]))
        offset += len(pts)
    return pv.PolyData(points, lines=np.concatenate(cells).astype(np.int64))


def _hex(color: str) -> tuple[float, float, float]:
    c = color.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# drawing pieces (pure functions, testable without a renderer)
# ---------------------------------------------------------------------------
def static_lines(p: SceneParams) -> list[np.ndarray]:
    """Lines that do not depend on the camera."""
    r, ri, half = p.radius, p.inner_radius, p.height / 2.0
    lines = [
        _circle(0, 0, half, r), _circle(0, 0, -half, r),
        _circle(0, 0, half - 0.6, ri, 64),
    ]
    px = projector_x(p)
    box = (14.0, 22.0, 18.0)          # depth (x), width (y), height (z) of the projector body
    x0, x1 = px - box[0], px
    y0, y1, z0, z1 = -box[1] / 2, box[1] / 2, -box[2] / 2, box[2] / 2
    corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
               (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
        lines.append(_segment(corners[a], corners[b], 6))
    lens = np.column_stack([np.full(24, px + 1.2), 3.5 * np.cos(np.linspace(0, 2 * math.pi, 24)),
                            3.5 * np.sin(np.linspace(0, 2 * math.pi, 24))])
    lines.append(lens)
    lines.extend(beam_edges(p))
    return lines


def bath_lines(p: SceneParams) -> list[np.ndarray]:
    """Square immersion tank around the vial (only with a bath)."""
    if not p.bath:
        return []
    half = p.height / 2.0
    s, zb, zt = p.radius * 1.45, -half - 2.0, half * 0.72
    sq = [(-s, -s), (s, -s), (s, s), (-s, s)]
    lines = []
    for i in range(4):
        (ax, ay), (bx, by) = sq[i], sq[(i + 1) % 4]
        lines.append(_segment((ax, ay, zb), (bx, by, zb), 10))
        lines.append(_segment((ax, ay, zt), (bx, by, zt), 10))
        lines.append(_segment((ax, ay, zb), (ax, ay, zt), 8))
    return lines


def zone_lines(p: SceneParams) -> list[np.ndarray]:
    """Dashed ring on the vial floor: the printable zone."""
    return _dashed(_circle(0, 0, -p.height / 2.0 + 0.4, p.inner_radius * 0.92, 96), 2, 2)


def silhouette_lines(p: SceneParams, camera_xy: np.ndarray) -> list[np.ndarray]:
    """The two vertical outline lines a cylinder shows from this camera."""
    half = p.height / 2.0
    d = float(np.hypot(*camera_xy))
    out = []
    for radius in (p.radius, p.inner_radius):
        if d <= radius * 1.001:
            angles = [math.atan2(camera_xy[1], camera_xy[0]) + s * math.pi / 2 for s in (-1, 1)]
        else:
            base = math.atan2(camera_xy[1], camera_xy[0])
            spread = math.acos(radius / d)
            angles = [base - spread, base + spread]
        for a in angles:
            x, y = radius * math.cos(a), radius * math.sin(a)
            out.append(_segment((x, y, -half), (x, y, half), 10))
    return out


def _beam_profile(p: SceneParams, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-width (y) and half-height (z) of the light sheet along x."""
    r = p.radius
    half_w0, half_h0 = r * 1.04, p.field_mm / 2.0
    if p.throw_mm > 0:
        src = -p.throw_mm
        scale = np.clip((x - src) / max(-src, 1e-6), 0.05, None)
    else:
        scale = np.ones_like(x)
    w = half_w0 * scale
    h = half_h0 * scale
    inside = np.clip((x + r) / (2 * r), 0.0, 1.0)
    h = h * (1.0 - p.bend * 0.5 * inside)
    w = w * (1.0 - p.bend * 0.25 * inside)
    return w, h


def beam_mesh(p: SceneParams, color: str, base_alpha: float) -> pv.PolyData:
    """Translucent light sheet with absorption fading inside the vial."""
    px = projector_x(p) + 1.5
    x = np.linspace(px, p.radius * 1.25, 40)
    w, h = _beam_profile(p, x)
    rgb = np.array(_hex(color)) * 255
    r = p.radius
    fade = np.where(x < -r, 1.0, np.exp(-p.alpha_per_mm * np.clip(x + r, 0, None)))
    fade *= np.where(x > r, np.clip(1 - (x - r) / (0.25 * r), 0, 1), 1.0)
    pts, colors, faces = [], [], []
    for sy, sz in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        for xi, wi, hi, fi in zip(x, w, h, fade, strict=True):
            pts.append((xi, sy * wi, sz * hi))
            colors.append((*rgb, 255 * base_alpha * fi))
    n = len(x)
    quads = ((0, 1), (1, 2), (2, 3), (3, 0))     # top, far side, bottom, near side
    for a, b in quads:
        for i in range(n - 1):
            faces.append([4, a * n + i, a * n + i + 1, b * n + i + 1, b * n + i])
    mesh = pv.PolyData(np.asarray(pts, dtype=float), np.asarray(faces).ravel())
    mesh.point_data["rgba"] = np.asarray(colors, dtype=np.uint8)
    return mesh


def beam_edges(p: SceneParams) -> list[np.ndarray]:
    px = projector_x(p) + 1.5
    x = np.linspace(px, p.radius * 1.1, 24)
    w, h = _beam_profile(p, x)
    return [np.column_stack([x, sy * w, sz * h]) for sy, sz in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def beam_hatch(p: SceneParams, spacing: float) -> list[np.ndarray]:
    """Diagonal pen strokes across the side and top faces of the light sheet."""
    px = projector_x(p) + 1.5
    x_end = p.radius * 1.05
    lines: list[np.ndarray] = []
    for face in ("near", "top"):
        x0 = px
        while x0 < x_end:
            xs = np.linspace(x0, min(x0 + spacing * 2.5, x_end), 6)
            w, h = _beam_profile(p, xs)
            t = np.linspace(0.0, 1.0, len(xs))
            if face == "near":
                pts = np.column_stack([xs, -w, h * (1 - 2 * t)])
            else:
                pts = np.column_stack([xs, w * (1 - 2 * t), h])
            lines.append(pts)
            x0 += spacing
    return lines


def rotation_arc(p: SceneParams) -> list[np.ndarray]:
    """Dashed ring with an arrow head under the vial: 'the vial turns'."""
    r, z = p.radius * 1.22, -p.height / 2.0 - 3.0
    ring = _circle(0, 0, z, r, 120, 0.15, 2 * math.pi - 0.15)
    head_a = ring[-1]
    tangent = ring[-1] - ring[-3]
    tangent /= max(np.linalg.norm(tangent), 1e-9)
    normal = np.array([tangent[1], -tangent[0], 0.0])
    size = max(2.5, r * 0.08)
    head = [np.vstack([head_a - tangent * size + normal * size * 0.6, head_a]),
            np.vstack([head_a - tangent * size - normal * size * 0.6, head_a])]
    return _dashed(ring, 3, 2) + head


def angle_ticks(p: SceneParams, frames: int, lit_fraction: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """One tick per 10 frames around the vial rim; lit ticks show compute progress."""
    n = max(4, frames // 10)
    r, z = p.radius + 1.6, p.height / 2.0
    lit: list[np.ndarray] = []
    dim: list[np.ndarray] = []
    for i in range(n):
        a = 2 * math.pi * i / n
        seg = np.array([[r * math.cos(a), r * math.sin(a), z], [r * math.cos(a), r * math.sin(a), z + 2.4]])
        (lit if i / n < lit_fraction else dim).append(seg)
    return lit, dim


DITHER_FS = """
//VTK::Light::Impl
{{
  float lum = dot(gl_FragData[0].rgb, vec3(0.299, 0.587, 0.114));
  lum = clamp((lum - 0.18) * 1.55, 0.0, 1.0);
  int bx = int(mod(floor(gl_FragCoord.x / {pix:.1f}), 4.0));
  int by = int(mod(floor(gl_FragCoord.y / {pix:.1f}), 4.0));
  int idx = by * 4 + bx;
  float m = 0.0;
  if (idx == 0) m = 0.0; else if (idx == 1) m = 8.0; else if (idx == 2) m = 2.0; else if (idx == 3) m = 10.0;
  else if (idx == 4) m = 12.0; else if (idx == 5) m = 4.0; else if (idx == 6) m = 14.0; else if (idx == 7) m = 6.0;
  else if (idx == 8) m = 3.0; else if (idx == 9) m = 11.0; else if (idx == 10) m = 1.0; else if (idx == 11) m = 9.0;
  else if (idx == 12) m = 15.0; else if (idx == 13) m = 7.0; else if (idx == 14) m = 13.0; else m = 5.0;
  float on = step((m + 0.5) / 16.0, lum);
  gl_FragData[0] = vec4(mix(vec3({pr:.4f}, {pg:.4f}, {pb:.4f}), vec3({ir:.4f}, {ig:.4f}, {ib:.4f}), on), 1.0);
}}
"""


def apply_dither(actor: Any, ink: str, paper: str, pixel: float = 2.0) -> bool:
    """Draw an actor as ordered-dither dots of ``ink`` on ``paper``. False if unsupported."""
    try:
        prop = actor.GetShaderProperty()
        prop.ClearAllFragmentShaderReplacements()
        ir, ig, ib = _hex(ink)
        pr, pg, pb = _hex(paper)
        code = DITHER_FS.format(pix=max(1.0, pixel), ir=ir, ig=ig, ib=ib, pr=pr, pg=pg, pb=pb)
        prop.AddFragmentShaderReplacement("//VTK::Light::Impl", True, code, False)
        return True
    except Exception:
        return False


def clear_dither(actor: Any) -> None:
    try:
        actor.GetShaderProperty().ClearAllFragmentShaderReplacements()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# scene object
# ---------------------------------------------------------------------------
class SketchScene:
    """Owns the sketch actors in a pyvista plotter."""

    def __init__(self, plotter: Any, palette: ScenePalette) -> None:
        self._plotter = plotter
        self._palette = palette
        self._params = SceneParams()
        self._frame = 0
        self._frames_per_rev = 360
        self._progress = 0.0
        self._activity = "idle"           # idle | computing | printing
        self._show_rotation = False
        self._camera_xy = np.array([1.0, -1.0]) * 200.0
        self._actors: dict[str, Any] = {}
        self._boil: list[Any] = []
        self._sil: list[Any] = []
        self._rot_actor: Any = None
        self._rot_angle = 0.0
        self._scan_actor: Any = None
        self._scan_x = 0.0
        self._line_width = 1.8

    # -- public API --------------------------------------------------------
    @property
    def params(self) -> SceneParams:
        return self._params

    def set_palette(self, palette: ScenePalette) -> None:
        self._palette = palette
        self.rebuild()

    def update(self, params: SceneParams) -> None:
        self._params = params
        self.rebuild()

    def set_frames(self, frames: int) -> None:
        if int(frames) != self._frames_per_rev or "ticks" not in self._actors:
            self._frames_per_rev = int(frames)
            self._rebuild_ticks()

    def set_activity(self, activity: str, progress: float = 0.0) -> bool:
        """Returns True if the picture changed (caller decides whether to redraw)."""
        changed = activity != self._activity
        self._activity = activity
        self._progress = float(np.clip(progress, 0.0, 1.0))
        if changed:
            self._rebuild_dynamic()
        return self._recolor_ticks() or changed

    def set_rotation_visible(self, visible: bool) -> None:
        self._show_rotation = visible
        self._rebuild_dynamic()

    @property
    def animating(self) -> bool:
        return True

    def tick(self) -> None:
        """Advance one animation frame (boil, rotation arrow, light scan)."""
        self._frame = (self._frame + 1) % BOIL_VARIANTS
        for group in (self._boil, self._sil):
            for i, actor in enumerate(group):
                actor.SetVisibility(i == self._frame)
        if self._rot_actor is not None:
            self._rot_angle = (self._rot_angle + 6.0) % 360.0
            self._rot_actor.SetOrientation(0.0, 0.0, self._rot_angle)
        if self._scan_actor is not None:
            p = self._params
            start, end = projector_x(p), p.radius
            self._scan_x = self._scan_x + (end - start) * 0.06
            if self._scan_x > end - start:
                self._scan_x = 0.0
            self._scan_actor.SetPosition(self._scan_x, 0.0, 0.0)

    def on_camera(self, position: np.ndarray) -> None:
        xy = np.asarray(position[:2], dtype=float)
        if np.allclose(xy, self._camera_xy, atol=1e-3):
            return
        self._camera_xy = xy
        self._rebuild_silhouettes()

    def remove(self) -> None:
        for actor in list(self._actors.values()) + self._boil + self._sil:
            self._plotter.remove_actor(actor, render=False)
        for actor in (self._rot_actor, self._scan_actor):
            if actor is not None:
                self._plotter.remove_actor(actor, render=False)
        self._actors.clear()
        self._boil, self._sil = [], []
        self._rot_actor = self._scan_actor = None

    # -- building ----------------------------------------------------------
    def _amp(self) -> float:
        return max(0.12, self._params.diameter_mm * 0.0035)

    def _add_lines(self, lines: list[np.ndarray], color: str, width: float, name: str,
                   opacity: float = 1.0) -> Any:
        mesh = _polylines(lines)
        if mesh.n_points == 0:
            return None
        return self._plotter.add_mesh(
            mesh, color=color, line_width=width, opacity=opacity, lighting=False,
            name=name, pickable=False, render=False, reset_camera=False)

    def rebuild(self) -> None:
        self.remove()
        p, pal = self._params, self._palette
        half = p.height / 2.0
        resin = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=p.inner_radius,
                            height=p.height - 1.0, resolution=64, capping=True)
        self._actors["resin"] = self._plotter.add_mesh(
            resin, color=pal.resin, opacity=0.22, lighting=False, name="sketch-resin",
            pickable=False, render=False, reset_camera=False)
        if p.bath:
            s = p.radius * 1.45
            liquid = pv.Box(bounds=(-s, s, -s, s, -half - 2.0, half * 0.72))
            self._actors["liquid"] = self._plotter.add_mesh(
                liquid, color=pal.liquid, opacity=0.035, lighting=False, name="sketch-liquid",
                pickable=False, render=False, reset_camera=False)
        alpha = 0.15 if self._activity != "computing" else 0.26
        beam = beam_mesh(p, pal.accent, alpha * (0.35 if pal.hatch else 1.0))
        self._actors["beam"] = self._plotter.add_mesh(
            beam, scalars="rgba", rgba=True, lighting=False, name="sketch-beam",
            pickable=False, render=False, reset_camera=False, show_scalar_bar=False)
        bath = self._add_lines(_jitter(bath_lines(p), self._amp(), 3), pal.muted, 1.0, "sketch-bath", opacity=0.55)
        if bath is not None:
            self._actors["bath"] = bath
        zone = self._add_lines(zone_lines(p), pal.muted, 1.0, "sketch-zone")
        if zone is not None:
            self._actors["zone"] = zone
        if pal.hatch:
            hatch = self._add_lines(_jitter(beam_hatch(p, max(3.0, p.diameter_mm * 0.085)), self._amp(), 7),
                                    pal.accent, 1.0, "sketch-hatch", opacity=0.55)
            if hatch is not None:
                self._actors["hatch"] = hatch
        base = static_lines(p)
        for k in range(BOIL_VARIANTS):
            actor = self._add_lines(_jitter(base, self._amp(), 11 + 17 * k), pal.sketch,
                                    self._line_width, f"sketch-boil-{k}")
            actor.SetVisibility(k == self._frame)
            self._boil.append(actor)
        self._rebuild_silhouettes()
        self._rebuild_ticks()
        self._rebuild_dynamic()

    def _rebuild_silhouettes(self) -> None:
        lines = silhouette_lines(self._params, self._camera_xy)
        meshes = [_polylines(_jitter(lines, self._amp(), 5 + 13 * k)) for k in range(BOIL_VARIANTS)]
        if len(self._sil) == BOIL_VARIANTS:
            # Update in place: called before renders while the camera moves.
            for actor, mesh in zip(self._sil, meshes, strict=True):
                actor.GetMapper().GetInputAsDataSet().DeepCopy(mesh)
            return
        for actor in self._sil:
            self._plotter.remove_actor(actor, render=False)
        self._sil = []
        for k, mesh in enumerate(meshes):
            actor = self._plotter.add_mesh(
                mesh, color=self._palette.sketch, line_width=self._line_width, lighting=False,
                name=f"sketch-sil-{k}", pickable=False, render=False, reset_camera=False)
            actor.SetVisibility(k == self._frame)
            self._sil.append(actor)

    def _lit_fraction(self) -> float:
        if self._activity == "computing":
            return self._progress
        return 1.0 if self._activity == "done" else 0.0

    def _rebuild_ticks(self) -> None:
        """One actor for all rim ticks; progress only recolours it (no new geometry)."""
        actor = self._actors.pop("ticks", None)
        if actor is not None:
            self._plotter.remove_actor(actor, render=False)
        lit, dim = angle_ticks(self._params, self._frames_per_rev, 0.0)
        segments = lit + dim
        if not segments:
            return
        mesh = _polylines(segments)
        self._tick_count = len(segments)
        mesh.cell_data["rgb"] = np.zeros((len(segments), 3), dtype=np.uint8)
        self._tick_mesh = mesh
        self._lit_ticks = -1
        self._actors["ticks"] = self._plotter.add_mesh(
            mesh, scalars="rgb", rgb=True, preference="cell", line_width=1.6, lighting=False,
            name="sketch-ticks", pickable=False, render=False, reset_camera=False, show_scalar_bar=False)
        self._recolor_ticks()

    def _recolor_ticks(self) -> bool:
        """True if any tick changed colour (then a redraw is worth it)."""
        mesh = getattr(self, "_tick_mesh", None)
        if mesh is None:
            return False
        lit = int(round(self._lit_fraction() * self._tick_count))
        if lit == self._lit_ticks:
            return False
        self._lit_ticks = lit
        colors = np.empty((self._tick_count, 3), dtype=np.uint8)
        colors[:] = np.array(_hex(self._palette.muted)) * 255
        colors[:lit] = np.array(_hex(self._palette.accent)) * 255
        mesh.cell_data["rgb"][:] = colors
        mesh.Modified()
        return True

    def _rebuild_dynamic(self) -> None:
        for actor in (self._rot_actor, self._scan_actor):
            if actor is not None:
                self._plotter.remove_actor(actor, render=False)
        self._rot_actor = self._scan_actor = None
        p = self._params
        if self._show_rotation or self._activity == "printing":
            self._rot_actor = self._add_lines(rotation_arc(p), self._palette.accent, 2.0, "sketch-rotation")
            if self._rot_actor is not None:
                self._rot_actor.SetOrientation(0.0, 0.0, self._rot_angle)
        if self._activity == "computing":
            w, h = _beam_profile(p, np.array([projector_x(p)]))
            slab = pv.Box(bounds=(projector_x(p), projector_x(p) + p.diameter_mm * 0.08,
                                  -float(w[0]), float(w[0]), -float(h[0]), float(h[0])))
            self._scan_actor = self._plotter.add_mesh(
                slab, color=self._palette.accent, opacity=0.18, lighting=False, name="sketch-scan",
                pickable=False, render=False, reset_camera=False)
