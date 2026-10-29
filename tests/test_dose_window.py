import numpy as np
import pytest
from scipy.ndimage import binary_dilation, binary_erosion

from optics.optimization import build_system_matrix, optimize_projections_sirt_volume

GRID = 64
ANGLES = np.deg2rad(np.linspace(0.0, 360.0, 120, endpoint=False))
IDEAL = dict(
    L_out=40.0, L_in=40.0, n_air=1.0, n_glass=1.0, n_liquid=1.0, n_resin=1.0,
    alpha=0.0, grid_res=GRID, grid_span_mm=40.0, container_mode=0,
)


def _target() -> np.ndarray:
    target = np.zeros((GRID, GRID, 1), dtype=np.float32)
    target[20:44, 16:48, 0] = 1.0
    target[28:36, 26:38, 0] = 0.0
    return target


def _contrast(sinogram: np.ndarray, target: np.ndarray) -> float:
    _, adjoint = build_system_matrix(
        ANGLES, GRID, GRID, 40.0, 40.0, 40.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0, None, lambda: False
    )
    dose = (adjoint @ sinogram.T.ravel()).reshape(GRID, GRID)
    part = target[:, :, 0] > 0.5
    core = binary_erosion(part, iterations=2)
    far = ~binary_dilation(part, iterations=2)
    return float(dose[core].min() / dose[far].max())


def test_dose_window_stage_increases_part_to_background_contrast() -> None:
    target = _target()
    plain = optimize_projections_sirt_volume(target, ANGLES, iterations=8, **IDEAL)
    windowed = optimize_projections_sirt_volume(
        target, ANGLES, iterations=8, window_iterations=16, **IDEAL
    )
    assert windowed.shape == plain.shape
    assert (windowed >= 0).all() and windowed.max() == pytest.approx(1.0)
    assert _contrast(windowed[0], target) > _contrast(plain[0], target) + 0.1


def test_window_iterations_are_validated() -> None:
    with pytest.raises(ValueError):
        optimize_projections_sirt_volume(_target(), ANGLES, iterations=2, window_iterations=-1, **IDEAL)


def test_engine_records_dose_window_and_backend_used(tmp_path) -> None:
    import trimesh

    from frame_io import load_manifest
    from slicing_engine import ResinSettings, SliceParams, SlicingEngine

    mesh = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    params = SliceParams(
        diameter_mm=30.0, grid_res=24, output_res=48, num_frames=16, fill_holes=True,
        resin=ResinSettings(), output_dir=str(tmp_path), projection_backend="physical",
        optimizer_iterations=3, cuvette_outer_mm=30.0, cuvette_inner_mm=28.0,
        n_air=1.0, n_glass=1.0, n_liquid=1.0, n_resin=1.0,
    )
    _, run_dir = SlicingEngine.run(mesh, params)
    manifest = load_manifest(run_dir)
    assert manifest is not None
    assert manifest.slice_parameters["projection_backend_used"] == "physical"
    assert manifest.slice_parameters["dose_window"] is True
    assert manifest.frame_schedule["frame_rate_hz"] == 30.0
