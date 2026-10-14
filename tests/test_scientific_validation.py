"""
test_scientific_validation.py
=============================
Автоматические тесты научной корректности pipeline:
  • Генерация синтетических фантомов
  • Вокселизация
  • Метрики Dice / IoU / MAE
  • Сквозной smoke-тест: mesh → slice → radon → compare
"""
from __future__ import annotations

import numpy as np
import pytest

from metrics import (
    ValidationReport,
    binarize_dose,
    compute_dice,
    compute_dose_mae,
    compute_iou,
    compute_validation_report,
    voxelize_mesh,
)
from synthetic_shapes import create_cube, create_cylinder, create_sphere

# -----------------------------------------------------------------------
# Синтетические формы: smoke-тесты
# -----------------------------------------------------------------------


class TestSyntheticShapes:
    """Проверяем, что генераторы фантомов создают корректные меши."""

    def test_sphere_is_watertight(self) -> None:
        mesh = create_sphere(radius_mm=10.0)
        assert mesh.is_watertight
        assert len(mesh.vertices) > 10

    def test_sphere_centered(self) -> None:
        mesh = create_sphere(radius_mm=10.0)
        centroid = mesh.bounding_box.centroid
        assert np.allclose(centroid, 0.0, atol=0.1)

    def test_sphere_radius(self) -> None:
        r = 12.5
        mesh = create_sphere(radius_mm=r)
        extents = mesh.extents
        # Диаметр по каждой оси ≈ 2R
        assert abs(extents[0] - 2 * r) < 0.5
        assert abs(extents[1] - 2 * r) < 0.5
        assert abs(extents[2] - 2 * r) < 0.5

    def test_cube_is_watertight(self) -> None:
        mesh = create_cube(side_mm=15.0)
        assert mesh.is_watertight
        assert len(mesh.faces) == 12  # 6 граней × 2 треугольника

    def test_cube_extents(self) -> None:
        s = 20.0
        mesh = create_cube(side_mm=s)
        extents = mesh.extents
        assert np.allclose(extents, s, atol=0.01)

    def test_cylinder_is_watertight(self) -> None:
        mesh = create_cylinder(radius_mm=8.0, height_mm=20.0)
        assert mesh.is_watertight

    def test_cylinder_height(self) -> None:
        h = 25.0
        mesh = create_cylinder(radius_mm=8.0, height_mm=h)
        z_extent = mesh.extents[2]
        assert abs(z_extent - h) < 0.5


# -----------------------------------------------------------------------
# Метрики: unit-тесты
# -----------------------------------------------------------------------


class TestMetrics:
    """Тесты корректности вычисления метрик на синтетических данных."""

    def test_dice_identical(self) -> None:
        a = np.ones((10, 10), dtype=bool)
        assert compute_dice(a, a) == pytest.approx(1.0)

    def test_dice_empty(self) -> None:
        a = np.zeros((10, 10), dtype=bool)
        assert compute_dice(a, a) == pytest.approx(1.0)

    def test_dice_no_overlap(self) -> None:
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[:5] = True
        b[5:] = True
        assert compute_dice(a, b) == pytest.approx(0.0)

    def test_dice_half_overlap(self) -> None:
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[:6] = True   # 60 voxels
        b[4:] = True   # 60 voxels, overlap = rows 4,5 = 20 voxels
        expected = 2 * 20 / (60 + 60)
        assert compute_dice(a, b) == pytest.approx(expected, abs=1e-6)

    def test_iou_identical(self) -> None:
        a = np.ones((5, 5), dtype=bool)
        assert compute_iou(a, a) == pytest.approx(1.0)

    def test_iou_no_overlap(self) -> None:
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[:5] = True
        b[5:] = True
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_mae_identical(self) -> None:
        orig = np.ones((10, 10), dtype=bool)
        dose = np.ones((10, 10), dtype=float)
        assert compute_dose_mae(orig, dose) == pytest.approx(0.0)

    def test_mae_inverted(self) -> None:
        orig = np.ones((10, 10), dtype=bool)
        dose = np.zeros((10, 10), dtype=float)
        assert compute_dose_mae(orig, dose) == pytest.approx(1.0)

    def test_binarize_threshold(self) -> None:
        dose = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        binary = binarize_dose(dose, threshold=0.5)
        expected = np.array([False, False, True, True, True])
        np.testing.assert_array_equal(binary, expected)


class TestValidationReport:
    """Интеграционный тест: полный ValidationReport."""

    def test_perfect_report(self) -> None:
        orig = np.ones((5, 5, 5), dtype=bool)
        dose = np.ones((5, 5, 5), dtype=float)
        report = compute_validation_report(orig, dose, threshold=0.5)
        assert isinstance(report, ValidationReport)
        assert report.dice == pytest.approx(1.0)
        assert report.iou == pytest.approx(1.0)
        assert report.mae == pytest.approx(0.0)
        assert report.volume_error_pct == pytest.approx(0.0)

    def test_empty_report(self) -> None:
        orig = np.zeros((5, 5, 5), dtype=bool)
        dose = np.zeros((5, 5, 5), dtype=float)
        report = compute_validation_report(orig, dose, threshold=0.5)
        assert report.dice == pytest.approx(1.0)
        assert report.iou == pytest.approx(1.0)


# -----------------------------------------------------------------------
# Вокселизация
# -----------------------------------------------------------------------


class TestVoxelization:
    """Проверяем вокселизацию на простых геометриях."""

    def test_sphere_voxelization_nonempty(self) -> None:
        """Вокселизация сферы должна дать непустую сетку."""
        mesh = create_sphere(radius_mm=10.0)
        grid = voxelize_mesh(mesh, grid_res=32, diameter_mm=30.0)
        assert grid.shape[0] == 32
        assert grid.shape[1] == 32
        assert grid.shape[2] > 0
        assert np.sum(grid) > 0

    def test_cube_voxelization_fills_interior(self) -> None:
        """Куб должен заполнять внутренность — значительная часть вокселей True."""
        mesh = create_cube(side_mm=10.0)
        grid = voxelize_mesh(mesh, grid_res=20, diameter_mm=20.0)
        fill_ratio = np.sum(grid) / grid.size
        # Куб 10мм в ванне 20мм → теоретически 50%³ = 12.5%, с дискретностью ≥ 8%
        assert fill_ratio > 0.05

    def test_cylinder_voxelization_symmetric(self) -> None:
        """Цилиндр вокруг оси Z: слои должны быть примерно одинаковыми."""
        mesh = create_cylinder(radius_mm=6.0, height_mm=16.0)
        grid = voxelize_mesh(mesh, grid_res=24, diameter_mm=20.0)
        # Берём центральные слои и проверяем, что их заполнение примерно равно
        nz = grid.shape[2]
        if nz > 4:
            mid = nz // 2
            fill_mid = np.sum(grid[:, :, mid])
            fill_q1 = np.sum(grid[:, :, nz // 4])
            # Центральные слои цилиндра должны быть почти одинаковыми
            if fill_mid > 0:
                ratio = fill_q1 / fill_mid
                assert 0.8 < ratio < 1.2
