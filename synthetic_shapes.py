"""
synthetic_shapes.py
===================
Генерация канонических синтетических тестовых моделей для численной
валидации pipeline slice → project → reconstruct.

Каждая функция возвращает trimesh.Trimesh с центром масс в начале
координат — готовый к подаче в ModelNode / SlicingEngine.
"""
from __future__ import annotations

import trimesh


def create_sphere(radius_mm: float = 10.0, subdivisions: int = 3) -> trimesh.Trimesh:
    """Создать сферу заданного радиуса.

    Проверяет изотропность реконструкции — сфера инвариантна
    к любому направлению проекции.
    """
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius_mm)
    _center(mesh)
    return mesh


def create_cube(side_mm: float = 15.0) -> trimesh.Trimesh:
    """Создать куб заданного размера ребра.

    Проверяет точность передачи плоских граней и острых кромок.
    """
    mesh = trimesh.creation.box(extents=[side_mm, side_mm, side_mm])
    _center(mesh)
    return mesh


def create_cylinder(
    radius_mm: float = 8.0,
    height_mm: float = 20.0,
    sections: int = 64,
) -> trimesh.Trimesh:
    """Создать сплошной цилиндр.

    Проверяет вращательную инвариантность вокруг оси Z —
    все проекции должны быть идентичными.
    """
    mesh = trimesh.creation.cylinder(
        radius=radius_mm, height=height_mm, sections=sections,
    )
    _center(mesh)
    return mesh


def create_hollow_tube(
    outer_radius_mm: float = 10.0,
    inner_radius_mm: float = 6.0,
    height_mm: float = 20.0,
    sections: int = 64,
) -> trimesh.Trimesh:
    """Создать полый цилиндр (трубку) с пустотой внутри.

    Ключевой тест для CAL/томографической печати: FBP должен
    корректно восстанавливать пустую внутреннюю область без
    паразитной засветки (over-exposure артефакты).
    """
    outer = trimesh.creation.cylinder(
        radius=outer_radius_mm, height=height_mm, sections=sections,
    )
    inner = trimesh.creation.cylinder(
        radius=inner_radius_mm, height=height_mm + 0.02, sections=sections,
    )
    tube = outer.difference(inner, engine="blender")
    # Fallback: если blender/cork недоступен, пробуем manifold
    if tube is None or not hasattr(tube, "vertices") or len(tube.vertices) == 0:
        try:
            tube = outer.difference(inner, engine="manifold")
        except Exception:
            tube = outer.difference(inner)
    _center(tube)
    return tube


def _center(mesh: trimesh.Trimesh) -> None:
    """Переместить центр bounding box в начало координат."""
    centroid = mesh.bounding_box.centroid
    mesh.apply_translation(-centroid)
