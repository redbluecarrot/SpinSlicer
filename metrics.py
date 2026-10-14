"""
metrics.py
==========
Численные метрики качества реконструкции для валидации pipeline
slice → Radon → FBP.

Основные метрики:
  • Dice Similarity Coefficient (DSC) — объёмное пересечение
  • Jaccard Index (IoU)
  • Voxel-wise Mean Absolute Error (MAE) — ошибка дозы
  • Volume Relative Error — отклонение восстановленного объёма от
    аналитического
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class ValidationReport:
    """Результат численного сравнения оригинала и реконструкции."""

    dice: float
    iou: float
    mae: float
    volume_error_pct: float
    orig_voxels: int
    rec_voxels: int


# ---------------------------------------------------------------------------
# Вокселизация
# ---------------------------------------------------------------------------

def voxelize_mesh(
    mesh: trimesh.Trimesh,
    grid_res: int,
    diameter_mm: float,
) -> np.ndarray:
    """Вокселизировать trimesh.Trimesh в бинарную 3D-сетку.

    Воксельное пространство определяется:
      — по XY: [-diameter/2, +diameter/2] с разрешением grid_res;
      — по Z: [z_min, z_max] из bounding box меша с шагом pitch.

    Возвращает bool-массив shape (grid_res, grid_res, nz).
    """
    pitch = diameter_mm / grid_res
    z_min, z_max = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
    z_levels = np.arange(z_min + pitch / 2.0, z_max, pitch)
    nz = len(z_levels)

    if nz == 0:
        return np.zeros((grid_res, grid_res, 0), dtype=bool)

    half = diameter_mm / 2.0
    # Центры вокселей в физических координатах XY
    coords_1d = np.linspace(-half + pitch / 2.0, half - pitch / 2.0, grid_res)
    xx, yy = np.meshgrid(coords_1d, coords_1d, indexing="ij")

    grid = np.zeros((grid_res, grid_res, nz), dtype=bool)

    for k, z in enumerate(z_levels):
        # Точки в плоскости z
        points = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            np.full(grid_res * grid_res, z),
        ])
        inside = mesh.contains(points)
        grid[:, :, k] = inside.reshape(grid_res, grid_res)

    return grid


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def compute_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice Similarity Coefficient: 2|A∩B| / (|A|+|B|)."""
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    intersection = int(np.sum(a_bool & b_bool))
    total = int(np.sum(a_bool)) + int(np.sum(b_bool))
    if total == 0:
        return 1.0  # оба пустые — идеальное совпадение
    return 2.0 * intersection / total


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over Union (Jaccard): |A∩B| / |A∪B|."""
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    intersection = int(np.sum(a_bool & b_bool))
    union = int(np.sum(a_bool | b_bool))
    if union == 0:
        return 1.0
    return intersection / union


def compute_dose_mae(
    original_binary: np.ndarray,
    reconstructed_continuous: np.ndarray,
) -> float:
    """Mean Absolute Error между нормализованной дозой и бинарной маской.

    original_binary: bool/0-1 маска.
    reconstructed_continuous: непрерывное поле дозы [0, 1].
    """
    ref = original_binary.astype(np.float64)
    rec = np.clip(reconstructed_continuous.astype(np.float64), 0.0, 1.0)
    return float(np.mean(np.abs(ref - rec)))


def binarize_dose(dose: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Бинаризовать непрерывное поле дозы по порогу."""
    return dose >= threshold


# ---------------------------------------------------------------------------
# Сквозная валидация
# ---------------------------------------------------------------------------

def compute_validation_report(
    original_grid: np.ndarray,
    reconstructed_dose: np.ndarray,
    threshold: float = 0.5,
) -> ValidationReport:
    """Полный отчёт по сравнению оригинального меша и реконструкции.

    original_grid: бинарная воксельная сетка оригинала.
    reconstructed_dose: нормализованное [0,1] поле дозы FBP.
    threshold: порог бинаризации реконструкции.
    """
    rec_binary = binarize_dose(reconstructed_dose, threshold)
    orig = original_grid.astype(bool)

    dice = compute_dice(orig, rec_binary)
    iou = compute_iou(orig, rec_binary)
    mae = compute_dose_mae(orig, reconstructed_dose)

    orig_count = int(np.sum(orig))
    rec_count = int(np.sum(rec_binary))
    if orig_count > 0:
        volume_err = 100.0 * (rec_count - orig_count) / orig_count
    else:
        volume_err = 0.0 if rec_count == 0 else float("inf")

    return ValidationReport(
        dice=dice,
        iou=iou,
        mae=mae,
        volume_error_pct=volume_err,
        orig_voxels=orig_count,
        rec_voxels=rec_count,
    )
