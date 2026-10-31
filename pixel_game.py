"""
pixel_game.py
=============
Three tiny pixel games for the minutes while the light is computed.

  «Засветка» (Cure)   — breakout where the bricks are *your* part's silhouette;
                         the photon cures every voxel it touches. Cure it all.
  «Слои» (Layers)     — stack falling voxel blocks in the vial; a full row is a
                         cured layer and disappears.
  «Фотон» (Photon)    — one button: guide a photon through gaps in glass walls.

Everything is drawn into a 160×120 image and scaled up with nearest-neighbour,
so it is real pixel art and costs almost nothing. Runs only while visible.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QEvent, QRect, QSettings, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter
from PyQt6.QtWidgets import QSizePolicy, QWidget

import theme
from constants import APP_ORG, APP_TITLE

W, H = 160, 120
COLS, ROWS = 34, 19
CELL_W, CELL_H = 4, 3
FIELD_X = (W - COLS * CELL_W) // 2
FIELD_Y = 14
PADDLE_Y = H - 9

# 3×5 pixel font for the HUD.
_FONT = {
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
    "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001010010010",
    "8": "111101111101111", "9": "111101111001111", "%": "101001010100101", "/": "001001010100100",
    " ": "000000000000000", "x": "000101010101000", "P": "111101111100100", "A": "010101111101101",
    "U": "101101101101111", "S": "111100111001111", "E": "111100110100111", "R": "110101110101101",
    "D": "110101101101110", "Y": "101101010010010", "O": "111101101101111", "K": "101101110101101",
    "C": "111100100100111", "L": "100100100100111", "I": "111010010010111", "N": "110101101101101",
    "T": "111010010010010", "G": "111100101101111", "M": "101111101101101", "!": "010010010000010",
    ".": "000000000000010", ":": "000010000010000", "H": "101101111101101", "B": "110101110101110",
    "-": "000000111000000", "V": "101101101101010", "F": "111100110100100", "W": "101101101111101",
}


def silhouette_from_mesh(mesh: Any, cols: int = COLS, rows: int = ROWS) -> np.ndarray:
    """Side view (x–z) of a mesh as a cols×rows boolean grid, keeping its proportions."""
    try:
        import cv2

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces)
        x, z = verts[:, 0], verts[:, 2]
        span_x, span_z = np.ptp(x) or 1.0, np.ptp(z) or 1.0
        scale = min(cols * CELL_W / span_x, rows * CELL_H / span_z)
        px = (x - x.min()) * scale / CELL_W + (cols - span_x * scale / CELL_W) / 2
        pz = (z.max() - z) * scale / CELL_H + (rows - span_z * scale / CELL_H) / 2
        mask = np.zeros((rows * 8, cols * 8), np.uint8)
        tris = np.stack([px[faces] * 8, pz[faces] * 8], axis=-1).round().astype(np.int32)
        if len(tris) > 60000:
            tris = tris[:: len(tris) // 60000 + 1]
        cv2.fillPoly(mask, list(tris), 255)
        small = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_AREA)
        grid = small > 70
        if grid.sum() >= 6:
            return grid
    except Exception:
        pass
    return default_shape(cols, rows)


def default_shape(cols: int = COLS, rows: int = ROWS) -> np.ndarray:
    """A hex nut with a hole, when there is no model yet."""
    yy, xx = np.mgrid[0:rows, 0:cols]
    cx, cy = (cols - 1) / 2, (rows - 1) / 2
    dx, dy = np.abs(xx - cx) / (cols * 0.42), np.abs(yy - cy) / (rows * 0.46)
    hexagon = (dy <= 1) & (dx + dy * 0.5 <= 1)
    hole = (xx - cx) ** 2 / (cols * 0.14) ** 2 + (yy - cy) ** 2 / (rows * 0.2) ** 2 < 1
    return hexagon & ~hole


class Colors:
    def __init__(self) -> None:
        p = theme.current()
        self.bg, self.ink, self.acc, self.dim = QColor(p.view), QColor(p.sketch), QColor(p.accent), QColor(p.dim)


def text(g: QPainter, x: int, y: int, s: str, color: QColor) -> None:
    g.setPen(Qt.PenStyle.NoPen)
    g.setBrush(color)
    for ch in s.upper():
        bits = _FONT.get(ch, _FONT[" "])
        for i, b in enumerate(bits):
            if b == "1":
                g.drawRect(x + i % 3, y + i // 3, 1, 1)
        x += 4


def text_center(g: QPainter, y: int, s: str, color: QColor) -> None:
    text(g, (W - len(s) * 4) // 2, y, s, color)


class Game:
    """Base: one game's state, logic and drawing in 160×120 pixels."""

    key = "game"
    title = ""
    hint = ""

    def __init__(self) -> None:
        self.score = 0
        self.state = "ready"            # ready | play | over
        self.paddle_x = W / 2.0

    def reset(self) -> None:
        self.score = 0
        self.state = "ready"

    def step(self, keys: set[int]) -> None:
        pass

    def click(self) -> None:
        pass

    def key_press(self, key: int) -> None:
        pass

    def mouse_x(self, x: float) -> None:
        pass

    def draw(self, g: QPainter, c: Colors, best: int) -> None:
        pass

    @property
    def over(self) -> bool:
        return self.state == "over"

    def moving(self) -> bool:
        """Anything on screen changes by itself this frame (else no repaint is needed)."""
        return self.state == "play"


# ---------------------------------------------------------------------------
class Cure(Game):
    key = "cure"
    title = "Засветка"
    hint = "кирпичи — силуэт вашей детали; фотон засвечивает воксели"
    PADDLE = 15            # half width: generous, the game is for relaxing
    LIVES = 5

    def __init__(self) -> None:
        super().__init__()
        self.shape = default_shape()
        self._field_key: Optional[tuple] = None
        self._field_img = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
        self.reset()

    def set_shape(self, grid: np.ndarray) -> None:
        self.shape = grid.astype(bool)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.cured = np.zeros_like(self.shape)
        self.lives = self.LIVES
        self.combo = 0
        self.stuck = True
        self.ball = [W / 2, PADDLE_Y - 3.0]
        self.vel = [0.0, 0.0]
        self.sparks: list[list[float]] = []
        self.flash = 0
        self.won = False

    def click(self) -> None:
        if self.over:
            self.reset()
        elif self.stuck:
            angle = random.uniform(-0.6, 0.6)
            speed = 1.05
            self.vel = [speed * math.sin(angle), -speed * math.cos(angle)]
            self.stuck = False
            self.state = "play"

    def key_press(self, key: int) -> None:
        if key in (Qt.Key.Key_Space, Qt.Key.Key_Up):
            self.click()

    def mouse_x(self, x: float) -> None:
        self.paddle_x = min(W - self.PADDLE, max(self.PADDLE, x))

    def moving(self) -> bool:
        return self.state == "play" or bool(self.sparks) or bool(self.flash)

    def step(self, keys: set[int]) -> None:
        move = (Qt.Key.Key_Right in keys) - (Qt.Key.Key_Left in keys)
        if move:
            self.mouse_x(self.paddle_x + 2.8 * move)
        if self.stuck:
            self.ball = [self.paddle_x, PADDLE_Y - 3.0]
        elif self.state == "play":
            for _ in range(3):
                self._advance(1.0 / 3.0)
        self.sparks = [[x + vx, y + vy, vx * 0.92, vy * 0.92 + 0.03, life - 1]
                       for x, y, vx, vy, life in self.sparks if life > 0]
        if self.flash:
            self.flash -= 1

    def _advance(self, dt: float) -> None:
        bx, by = self.ball
        vx, vy = self.vel
        nx, ny = bx + vx * dt * 2, by + vy * dt * 2
        if nx < 2 or nx > W - 2:
            vx = -vx
            nx = min(W - 2, max(2, nx))
        if ny < 10:
            vy = abs(vy)
            ny = 10
        if vy > 0 and PADDLE_Y - 3 <= ny <= PADDLE_Y + 2 and abs(nx - self.paddle_x) <= self.PADDLE + 1:
            offset = (nx - self.paddle_x) / self.PADDLE
            speed = min(1.7, math.hypot(vx, vy) * 1.01)
            angle = offset * 0.95
            vx, vy = speed * math.sin(angle), -speed * math.cos(angle)
            ny = PADDLE_Y - 3
            self.combo = 0
        col = int((nx - FIELD_X) // CELL_W)
        row = int((ny - FIELD_Y) // CELL_H)
        if 0 <= col < COLS and 0 <= row < ROWS and self.shape[row, col] and not self.cured[row, col]:
            self.cured[row, col] = True
            self.combo += 1
            self.score += 10 * self.combo
            cx, cy = FIELD_X + col * CELL_W + CELL_W / 2, FIELD_Y + row * CELL_H + CELL_H / 2
            for _ in range(6):
                a = random.uniform(0, 2 * math.pi)
                self.sparks.append([cx, cy, math.cos(a) * 0.9, math.sin(a) * 0.9, random.randint(8, 16)])
            if int((bx - FIELD_X) // CELL_W) != col:
                vx = -vx
            else:
                vy = -vy
            nx, ny = bx, by
            if self.cured[self.shape].all():
                self.won = True
                self.state = "over"
                self.flash = 30
                self.score += 100 * self.lives
        if ny > H + 2:
            self.lives -= 1
            self.combo = 0
            if self.lives <= 0:
                self.state = "over"
            else:
                self.stuck = True
                self.state = "ready"
            nx, ny, vx, vy = self.paddle_x, PADDLE_Y - 3.0, 0.0, 0.0
        self.ball = [nx, ny]
        self.vel = [vx, vy]

    def cured_fraction(self) -> float:
        total = int(self.shape.sum())
        return float(self.cured[self.shape].sum()) / max(total, 1)

    def _field(self, c: Colors) -> QImage:
        """The vial and voxels, redrawn only when a voxel cures or the theme changes."""
        key = (int(self.cured.sum()), id(self.shape), c.acc.rgb(), c.ink.rgb(), c.bg.rgb())
        if self._field_key == key:
            return self._field_img
        img = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.transparent)
        g = QPainter(img)
        g.setPen(Qt.PenStyle.NoPen)
        g.setBrush(c.dim)
        for y in range(FIELD_Y - 4, PADDLE_Y - 4, 3):
            g.drawRect(FIELD_X - 4, y, 1, 1)
            g.drawRect(FIELD_X + COLS * CELL_W + 3, y, 1, 1)
        rows, cols = np.nonzero(self.shape)
        for r, col in zip(rows.tolist(), cols.tolist()):
            x, y = FIELD_X + col * CELL_W, FIELD_Y + r * CELL_H
            if self.cured[r, col]:
                g.setBrush(c.acc)
                g.drawRect(x, y, CELL_W - 1, CELL_H - 1)
            else:
                g.setBrush(c.ink)
                g.drawRect(x + (r % 2), y, 1, 1)
                g.drawRect(x + 2 - (r % 2), y + 1, 1, 1)
        g.end()
        self._field_key, self._field_img = key, img
        return img

    def draw(self, g: QPainter, c: Colors, best: int) -> None:
        g.drawImage(0, 0, self._field(c))
        g.setBrush(c.acc)
        for sx, sy, _vx, _vy, life in self.sparks:
            if life > 0:
                g.drawRect(int(sx), int(sy), 1, 1)
        bx, by = int(self.ball[0]), int(self.ball[1])
        g.drawRect(bx - 1, by - 1, 3, 3)
        px = int(self.paddle_x)
        g.setBrush(c.ink)
        g.drawRect(px - self.PADDLE, PADDLE_Y, 2 * self.PADDLE, 3)
        g.setBrush(c.acc)
        g.drawRect(px - 3, PADDLE_Y - 1, 6, 1)
        pct = round(100 * self.cured_fraction())
        text(g, 3, 3, f"{pct}%", c.acc)
        text(g, W - 3 - 4 * len(str(self.score)), 3, str(self.score), c.ink)
        g.setBrush(c.acc)
        for i in range(self.lives):
            g.drawRect(W // 2 - 10 + i * 5, 4, 2, 2)
        if self.state == "ready" and self.lives == self.LIVES and not self.cured.any():
            text_center(g, H - 24, "CLICK - START", c.dim)
        elif self.over:
            if self.won:
                text_center(g, H // 2 - 8, "PRINTED!", c.acc if (self.flash // 4) % 2 == 0 else c.ink)
            else:
                text_center(g, H // 2 - 8, "UNDER-CURED", c.acc)
            text_center(g, H // 2, f"BEST {best}", c.ink)
            text_center(g, H // 2 + 8, "CLICK - AGAIN", c.dim)

    def simulate_hit(self, row: int, col: int) -> None:
        self.ball = [FIELD_X + col * CELL_W + 2.0, FIELD_Y + row * CELL_H + CELL_H + 1.0]
        self.vel = [0.0, -1.0]
        self.stuck = False
        self.state = "play"
        self._advance(1.0)


# ---------------------------------------------------------------------------
_PIECES = [
    [(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (1, 0), (0, 1), (1, 1)], [(1, 0), (0, 1), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (2, 1)], [(2, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (2, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (1, 1), (2, 1)],
]


class Layers(Game):
    key = "layers"
    title = "Слои"
    hint = "укладывайте воксели в колбу без щелей"
    BW, BH, CELL = 10, 19, 5
    X0 = (W - 10 * 5) // 2
    Y0 = 12

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.board = np.zeros((self.BH, self.BW), dtype=np.int8)
        self.lines = 0
        self.tick = 0
        self.flash_rows: list[int] = []
        self.flash = 0
        self._next = random.randrange(len(_PIECES))
        self._spawn()

    def _spawn(self) -> None:
        self.cells = list(_PIECES[self._next])
        self._next = random.randrange(len(_PIECES))
        self.px, self.py = self.BW // 2 - 2, 0
        if not self._fits(self.cells, self.px, self.py):
            self.state = "over"

    def _fits(self, cells: list[tuple[int, int]], px: int, py: int) -> bool:
        for x, y in cells:
            bx, by = px + x, py + y
            if bx < 0 or bx >= self.BW or by >= self.BH:
                return False
            if by >= 0 and self.board[by, bx]:
                return False
        return True

    def _rotate(self) -> None:
        rotated = [(-y, x) for x, y in self.cells]
        minx = min(x for x, _ in rotated)
        miny = min(y for _, y in rotated)
        rotated = [(x - minx, y - miny) for x, y in rotated]
        for kick in (0, -1, 1, -2, 2):
            if self._fits(rotated, self.px + kick, self.py):
                self.cells, self.px = rotated, self.px + kick
                return

    def _lock(self) -> None:
        for x, y in self.cells:
            if 0 <= self.py + y < self.BH:
                self.board[self.py + y, self.px + x] = 1
        full = [r for r in range(self.BH) if self.board[r].all()]
        if full:
            self.flash_rows, self.flash = full, 8
            self.score += [0, 40, 100, 300, 1200][len(full)]
            self.lines += len(full)
        else:
            self._spawn()

    def click(self) -> None:
        if self.over:
            self.reset()
        else:
            self.state = "play"
            self._rotate()

    def key_press(self, key: int) -> None:
        if self.over:
            if key == Qt.Key.Key_Space:
                self.reset()
            return
        self.state = "play"
        if key == Qt.Key.Key_Left and self._fits(self.cells, self.px - 1, self.py):
            self.px -= 1
        elif key == Qt.Key.Key_Right and self._fits(self.cells, self.px + 1, self.py):
            self.px += 1
        elif key == Qt.Key.Key_Up:
            self._rotate()
        elif key == Qt.Key.Key_Space:
            while self._fits(self.cells, self.px, self.py + 1):
                self.py += 1
            self._lock()

    def mouse_x(self, x: float) -> None:
        # Arrows move the piece. Following the hovering cursor made every new piece
        # slide towards wherever the mouse happened to rest (usually off to the left).
        return

    def step(self, keys: set[int]) -> None:
        if self.state != "play":
            return
        if self.flash:
            self.flash -= 1
            if self.flash == 0:
                keep = [r for r in range(self.BH) if r not in self.flash_rows]
                rest = self.board[keep]
                self.board = np.vstack([np.zeros((self.BH - len(rest), self.BW), np.int8), rest])
                self.flash_rows = []
                self._spawn()
            return
        self.tick += 1
        speed = max(6, 30 - self.lines * 2)
        if Qt.Key.Key_Down in keys:
            speed = 2
        if self.tick % speed == 0:
            if self._fits(self.cells, self.px, self.py + 1):
                self.py += 1
            else:
                self._lock()

    def draw(self, g: QPainter, c: Colors, best: int) -> None:
        x0, y0, cs = self.X0, self.Y0, self.CELL
        g.setBrush(c.dim)
        for y in range(y0, y0 + self.BH * cs + 1, 2):
            g.drawRect(x0 - 3, y, 1, 1)
            g.drawRect(x0 + self.BW * cs + 2, y, 1, 1)
        for x in range(x0 - 3, x0 + self.BW * cs + 3, 2):
            g.drawRect(x, y0 + self.BH * cs + 2, 1, 1)
        for r in range(self.BH):
            for col in range(self.BW):
                if self.board[r, col]:
                    flashing = r in self.flash_rows and (self.flash // 2) % 2 == 0
                    g.setBrush(c.acc if flashing else c.ink)
                    g.drawRect(x0 + col * cs, y0 + r * cs, cs - 1, cs - 1)
        if not self.flash and not self.over:
            g.setBrush(c.acc)
            for x, y in self.cells:
                if self.py + y >= 0:
                    g.drawRect(x0 + (self.px + x) * cs, y0 + (self.py + y) * cs, cs - 1, cs - 1)
        text(g, 3, 3, f"{self.lines}", c.acc)
        text(g, W - 3 - 4 * len(str(self.score)), 3, str(self.score), c.ink)
        g.setBrush(c.dim)
        for x, y in _PIECES[self._next]:
            g.drawRect(W - 22 + x * 3, 14 + y * 3, 2, 2)
        if self.state == "ready":
            text_center(g, H // 2, "CLICK - START", c.dim)
        elif self.over:
            text_center(g, H // 2 - 8, "FULL VIAL", c.acc)
            text_center(g, H // 2, f"BEST {best}", c.ink)


# ---------------------------------------------------------------------------
class Photon(Game):
    key = "photon"
    title = "Фотон"
    hint = "проведите фотон через стенки, не задев стекло"
    GAP = 38

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    TOP, BOTTOM = 10, H - 3
    level: list[np.ndarray] = []

    def set_level_from_frames(self, frames: list) -> None:
        """Walls cut from the real print: each wall is one projector frame's column of light.

        Bright pixels are where light goes through the resin — the photon may pass
        there. The column sweeps across the frames, so the gaps follow the part's
        real shape as the vial turns.
        """
        rows = self.BOTTOM - self.TOP
        level: list[np.ndarray] = []
        if frames:
            step = max(1, len(frames) // 90)
            for i, frame in enumerate(frames[::step]):
                f = np.asarray(frame)
                if f.ndim != 2 or f.size == 0:
                    continue
                h, w = f.shape
                x = int(np.clip(w * (0.5 + 0.34 * math.sin(i * 0.33)), 0, w - 1))
                col = f[:, max(0, x - 2):x + 3].astype(np.float32).mean(axis=1)
                col = np.interp(np.linspace(0, h - 1, rows), np.arange(h), col)
                peak = float(col.max())
                if peak <= 0:
                    continue
                mask = col > 0.35 * peak
                # Close pin-holes and keep a gap you can actually fly through.
                for _ in range(2):
                    mask = mask | (np.roll(mask, 1) & np.roll(mask, -1))
                if mask.sum() < 20:
                    centre = int(np.argmax(col))
                    mask[max(0, centre - 10):centre + 10] = True
                level.append(mask)
        self.level = level
        self.hint = "летите сквозь свет вашей печати" if level else "проведите фотон через стенки, не задев стекло"
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.y, self.vy = H / 2, 0.0
        self.walls: list[list[Any]] = []
        self.t = 0
        self.trail: list[tuple[int, int]] = []
        self._level_i = 0

    def _new_wall(self) -> list[Any]:
        if self.level:
            mask = self.level[self._level_i % len(self.level)]
            self._level_i += 1
            return [float(W + 4), mask, 0.0]
        rows = self.BOTTOM - self.TOP
        mask = np.zeros(rows, dtype=bool)
        start = int(random.uniform(12, rows - 12 - self.GAP))
        mask[start:start + self.GAP] = True
        return [float(W + 4), mask, 0.0]

    def _open(self, mask: np.ndarray, y: float) -> bool:
        r = int(y) - self.TOP
        return 0 <= r < len(mask) and bool(mask[max(0, r - 1):r + 2].all())

    def click(self) -> None:
        if self.over:
            self.reset()
            return
        self.state = "play"
        self.vy = -1.75

    def key_press(self, key: int) -> None:
        if key in (Qt.Key.Key_Space, Qt.Key.Key_Up):
            self.click()

    def step(self, keys: set[int]) -> None:
        if self.state != "play":
            return
        self.t += 1
        self.vy = min(2.2, self.vy + 0.11)
        self.y += self.vy
        speed = 1.0 + min(0.8, self.score * 0.03)
        if not self.walls or self.walls[-1][0] < W - 58:
            self.walls.append(self._new_wall())
        for wall in self.walls:
            wall[0] -= speed
            if wall[2] == 0.0 and wall[0] + 6 < 38:
                wall[2] = 1.0
                self.score += 1
        self.walls = [w for w in self.walls if w[0] > -10]
        self.trail = ([(38, int(self.y))] + [(x - int(speed), y) for x, y in self.trail])[:18]
        if self.y < self.TOP or self.y > self.BOTTOM:
            self.state = "over"
        for x, mask, _ in self.walls:
            if x - 1 <= 40 and x + 6 >= 36 and not self._open(mask, self.y):
                self.state = "over"

    def draw(self, g: QPainter, c: Colors, best: int) -> None:
        for x, mask, _ in self.walls:
            xi = int(x)
            for r in range(0, len(mask), 2):
                yy = self.TOP + r
                if mask[r]:
                    if self.level:
                        g.setBrush(c.dim)
                        g.drawRect(xi + 2 + (r // 2) % 2, yy, 1, 1)
                else:
                    g.setBrush(c.ink)
                    g.drawRect(xi + (r // 2) % 2, yy, 1, 1)
                    g.drawRect(xi + 5 - (r // 2) % 2, yy, 1, 1)
        g.setBrush(c.acc)
        for i, (x, y) in enumerate(self.trail):
            if i % 2 == 0:
                g.drawRect(x, y, 1, 1)
        g.drawRect(37, int(self.y) - 1, 3, 3)
        text(g, W - 3 - 4 * len(str(self.score)), 3, str(self.score), c.ink)
        if self.state == "ready":
            text_center(g, H // 2 + 14, "CLICK - FLY", c.dim)
        elif self.over:
            text_center(g, H // 2 - 8, "ABSORBED", c.acc)
            text_center(g, H // 2, f"BEST {best}", c.ink)


# ---------------------------------------------------------------------------
class PixelGame(QWidget):
    """Hosts the games; ``set_game(i)`` switches. Emits ``finished(score, won)``."""

    finished = pyqtSignal(int, bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumSize(W * 2, H * 2)
        self._settings = QSettings(APP_ORG, APP_TITLE)
        self.cure, self.layers, self.photon = Cure(), Layers(), Photon()
        self.games: list[Game] = [self.cure, self.layers, self.photon]
        self._game: Game = self.cure
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._step)
        self._image = QImage(W, H, QImage.Format.Format_RGB32)
        self._keys: set[int] = set()
        self._paused = False
        self._was_over = False
        # Adaptive frame rate: 60 fps, or 30 fps (two logic steps per picture, same
        # game speed) when drawing the window costs the machine too much CPU.
        self._steps_per_frame = 1
        self._load_frames = 0
        self._load_wall = time.perf_counter()
        self._load_cpu = time.process_time()

    # ------------------------------------------------------------------ public
    def set_shape(self, grid: np.ndarray) -> None:
        self.cure.set_shape(grid)
        self.update()

    def set_game(self, index: int) -> None:
        self._game = self.games[index % len(self.games)]
        self._was_over = self._game.over
        self.update()

    def game(self) -> Game:
        return self._game

    def reset(self) -> None:
        self._game.reset()
        self.update()

    def cured_fraction(self) -> float:
        return self.cure.cured_fraction()

    def simulate_hit(self, row: int, col: int) -> None:
        self.cure.simulate_hit(row, col)

    def best(self, game: Optional[Game] = None) -> int:
        g = game or self._game
        return int(self._settings.value(f"game/best_{g.key}", 0) or 0)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(W * 2, H * 2)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return int(width * H / W)

    # ------------------------------------------------------------------ loop
    def showEvent(self, event: Any) -> None:  # noqa: N802
        self._timer.start(16 if self._steps_per_frame == 1 else 33)
        super().showEvent(event)

    def hideEvent(self, event: Any) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    def _step(self) -> None:
        if self._paused or not self.isVisible():
            return
        if not self._game.moving() and not self._keys:
            return      # nothing moves: keep the last picture, no repaint, no CPU
        for _ in range(self._steps_per_frame):
            self._game.step(self._keys)
        self._adapt_frame_rate()
        if self._game.over and not self._was_over:
            score = self._game.score
            if score > self.best():
                self._settings.setValue(f"game/best_{self._game.key}", score)
            self.finished.emit(score, bool(getattr(self._game, "won", False)))
        self._was_over = self._game.over
        self.update()

    def _adapt_frame_rate(self) -> None:
        """Every ~1 s of play: if the whole app used most of a core, halve the pictures.

        On a real GPU a frame costs well under a millisecond and the game stays at
        60 fps. With software OpenGL each repaint re-composes the window on the CPU;
        30 fps then keeps the compute and the rest of the interface responsive.
        """
        self._load_frames += 1
        if self._load_frames < 60 // self._steps_per_frame:
            return
        wall = time.perf_counter() - self._load_wall
        load = (time.process_time() - self._load_cpu) / max(wall, 1e-6)
        if self._steps_per_frame == 1 and load > 0.7:
            self._steps_per_frame = 2
            self._timer.setInterval(33)
        elif self._steps_per_frame == 2 and load < 0.3:
            self._steps_per_frame = 1
            self._timer.setInterval(16)
        self._load_frames = 0
        self._load_wall = time.perf_counter()
        self._load_cpu = time.process_time()

    # ------------------------------------------------------------------ input
    def keyPressEvent(self, event: QKeyEvent | None) -> None:  # noqa: N802
        if event is None:
            return
        key = event.key()
        if key == Qt.Key.Key_P:
            self._paused = not self._paused
        else:
            self._keys.add(key)
            self._game.key_press(key)
        self.update()

    def keyReleaseEvent(self, event: QKeyEvent | None) -> None:  # noqa: N802
        if event is not None:
            self._keys.discard(event.key())

    def _game_x(self, x: float) -> float:
        rect = self._target()
        return (x - rect.x()) * W / max(rect.width(), 1)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if event is not None:
            self._game.mouse_x(self._game_x(event.position().x()))
            if not self._game.moving():
                self._game.step(self._keys)      # the waiting ball follows the paddle
                self.update()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        self.setFocus()
        self._paused = False
        self._game.click()
        self.update()

    def focusOutEvent(self, event: Any) -> None:  # noqa: N802
        # A key released while another widget had focus never reaches us: forget it,
        # or the game would keep "holding" that arrow.
        self._keys.clear()
        super().focusOutEvent(event)

    def changeEvent(self, event: Any) -> None:  # noqa: N802
        # Like any game on a Mac: switching to another window pauses a running round.
        if event is not None and event.type() == QEvent.Type.ActivationChange:
            if not self.isActiveWindow() and self._game.state == "play":
                self._paused = True
                self._keys.clear()
                self.update()
        super().changeEvent(event)

    # ------------------------------------------------------------------ drawing
    def _target(self) -> QRect:
        k = max(1, min(self.width() // W, self.height() // H))
        w, h = W * k, H * k
        return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        c = Colors()
        img = self._image
        img.fill(c.bg)
        g = QPainter(img)
        g.setPen(Qt.PenStyle.NoPen)
        self._game.draw(g, c, max(self.best(), self._game.score))
        if self._paused:
            text_center(g, H // 2 + 16, "PAUSE", c.ink)
        g.end()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(self._target(), img)
        painter.end()
