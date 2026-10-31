"""
model_node.py
=============
ModelNode хранит:
  • original_mesh — оригинальный trimesh.Trimesh, ЦЕНТРИРОВАННЫЙ по bounding
    box при загрузке и БОЛЬШЕ НИКОГДА не изменяемый напрямую;
  • transform — отдельную матрицу трансформации (Scale, Rotation, Translation).

Трансформация применяется только в двух местах:
  1. Во вьюпорте — как actor.user_matrix (GPU-трансформация, без пересчёта
     вершин => никаких лагов при вращении/масштабировании);
  2. Перед генерацией — на КОПИИ меша, непосредственно перед передачей
     в SlicingEngine.

Масштаб всегда выражается в физических размерах (мм), а не абстрактным
множителем 0.1x–3.0x: пользователь задаёт целевой размер по оси, ModelNode
сам считает нужный коэффициент масштабирования от исходного bounding box.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
import trimesh.transformations as tf


@dataclass
class Transform:
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    rotation_deg: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def reset(self) -> None:
        self.scale = np.ones(3, dtype=np.float64)
        self.rotation_deg = np.zeros(3, dtype=np.float64)
        self.translation = np.zeros(3, dtype=np.float64)

    def matrix(self) -> np.ndarray:
        """4x4 матрица: T * R * S (масштаб -> поворот -> сдвиг)."""
        s = np.eye(4)
        s[0, 0], s[1, 1], s[2, 2] = self.scale

        rx, ry, rz = np.radians(self.rotation_deg)
        r = tf.euler_matrix(rx, ry, rz, axes="sxyz")

        t = tf.translation_matrix(self.translation)
        return t @ r @ s


class ModelNode:
    """Обёртка над одной загруженной моделью в сцене."""

    def __init__(self, mesh: trimesh.Trimesh, source_path: str = ""):
        self.original_mesh = mesh
        self.source_path = source_path
        self.transform = Transform()

    # --- геометрия -------------------------------------------------------
    @property
    def base_extents(self) -> np.ndarray:
        """Размеры исходного (немасштабированного) bounding box, мм."""
        return self.original_mesh.extents.astype(np.float64)

    @property
    def vertex_count(self) -> int:
        return int(len(self.original_mesh.vertices))

    @property
    def face_count(self) -> int:
        return int(len(self.original_mesh.faces))

    # --- трансформация -----------------------------------------------------
    def matrix(self) -> np.ndarray:
        return self.transform.matrix()

    def current_size_mm(self) -> np.ndarray:
        return self.base_extents * self.transform.scale

    def set_size_mm(self, x: float | None = None, y: float | None = None,
                     z: float | None = None, uniform: bool = False) -> None:
        """Задаёт целевой физический размер по одной или нескольким осям.

        Если uniform=True — коэффициент масштаба, посчитанный для изменённой
        оси, применяется одинаково ко всем трём осям (пропорциональный
        масштаб), а не подгоняет остальные оси под точный мм-размер.
        """
        base = self.base_extents
        safe_base = np.where(base > 1e-9, base, 1.0)
        target = self.current_size_mm().copy()

        changed_axis = None
        if x is not None:
            target[0] = x
            changed_axis = 0
        if y is not None:
            target[1] = y
            changed_axis = 1
        if z is not None:
            target[2] = z
            changed_axis = 2

        new_scale = target / safe_base

        if uniform and changed_axis is not None:
            new_scale[:] = new_scale[changed_axis]

        self.transform.scale = new_scale

    def set_rotation_deg(self, x: float | None = None, y: float | None = None,
                          z: float | None = None) -> None:
        r = self.transform.rotation_deg
        if x is not None:
            r[0] = x
        if y is not None:
            r[1] = y
        if z is not None:
            r[2] = z

    def set_translation_mm(self, x: float | None = None, y: float | None = None,
                            z: float | None = None) -> None:
        t = self.transform.translation
        if x is not None:
            t[0] = x
        if y is not None:
            t[1] = y
        if z is not None:
            t[2] = z

    def set_matrix(self, matrix: np.ndarray, uniform: bool = False) -> None:
        """Store a matrix produced by an interactive 3D transform widget.

        VTK/PyVista widgets operate on one affine matrix, while the rest of
        the application stores scale, Euler rotation, and translation as
        separate fields.  Decomposing at this boundary keeps viewport edits
        and generated projections on exactly the same transform.
        """
        candidate = np.asarray(matrix, dtype=np.float64)
        if candidate.shape != (4, 4) or not np.all(np.isfinite(candidate)):
            raise ValueError("Transform matrix must be a finite 4x4 array.")

        scale, _shear, angles, translation, _perspective = tf.decompose_matrix(candidate)
        scale = np.asarray(scale, dtype=np.float64)
        if np.any(scale <= 1e-9) or not np.all(np.isfinite(scale)):
            raise ValueError("Transform matrix contains an invalid scale.")

        if uniform:
            previous = np.where(self.transform.scale > 1e-9, self.transform.scale, 1.0)
            ratios = scale / previous
            changed_axis = int(np.argmax(np.abs(np.log(np.maximum(ratios, 1e-12)))))
            # A proportional resize must preserve the existing axis ratios.
            # Replacing all axes with the changed axis value flattened any
            # intentional non-uniform scale after the first resize.
            scale = previous * ratios[changed_axis]

        self.transform.scale = scale
        self.transform.rotation_deg = np.degrees(np.asarray(angles, dtype=np.float64))
        self.transform.translation = np.asarray(translation, dtype=np.float64)

    def fit_to_vat(
        self,
        diameter_mm: float,
        vat_height_mm: float,
        fill_fraction: float = 0.92,
    ) -> None:
        """Авто-масштабирование с вписыванием в цилиндрическую ванну.

        Учитывает текущий поворот модели (rotation_deg): вершины
        прогоняются через матрицу вращения *без* масштаба и сдвига,
        после чего вычисляется:
        - максимальный радиальный размах в плоскости XY (цилиндр ванны);
        - максимальный полуразмах по Z (высота ванны).

        Масштаб подбирается пропорциональный (uniform) так, чтобы модель
        вписывалась и по диаметру, и по высоте. Сдвиг обнуляется.
        """
        verts = self.original_mesh.vertices  # (N, 3)

        # Матрица вращения без масштаба и сдвига
        rx, ry, rz = np.radians(self.transform.rotation_deg)
        r = tf.euler_matrix(rx, ry, rz, axes="sxyz")[:3, :3]

        rotated = (r @ verts.T).T  # (N, 3)

        # Радиальное расстояние в XY и полуразмах по Z
        r_max = float(np.max(np.sqrt(rotated[:, 0] ** 2 + rotated[:, 1] ** 2)))
        z_half = float(np.max(np.abs(rotated[:, 2])))

        if r_max < 1e-9 and z_half < 1e-9:
            return

        # Масштаб: вписать и по радиусу, и по высоте
        s_xy = (diameter_mm / 2.0) / r_max if r_max > 1e-9 else 1e9
        s_z = (vat_height_mm / 2.0) / z_half if z_half > 1e-9 else 1e9
        s = fill_fraction * min(s_xy, s_z)

        self.transform.scale = np.array([s, s, s], dtype=np.float64)
        self.transform.translation[:] = 0.0  # центрировать

    def fit_to_diameter(self, diameter_mm: float, fill_fraction: float) -> None:
        """Обратная совместимость: вписывает в ванну с высотой D × 1.6."""
        self.fit_to_vat(diameter_mm, diameter_mm * 1.6, fill_fraction)

    def center_xy(self) -> None:
        self.transform.translation[0] = 0.0
        self.transform.translation[1] = 0.0

    def reset(self) -> None:
        self.transform.reset()

    def get_transformed_mesh(self) -> trimesh.Trimesh:
        """Копия меша с окончательно применённой трансформацией.

        Используется ИСКЛЮЧИТЕЛЬНО перед генерацией проекций — вьюпорт
        трансформирует ту же геометрию на GPU через user_matrix, не трогая
        original_mesh.
        """
        mesh = self.original_mesh.copy()
        mesh.apply_transform(self.matrix())
        return mesh
