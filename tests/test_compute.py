import glob

import numpy as np
import pytest
import scipy.sparse as sp
import trimesh
from PIL import Image

import compute
from compute import Hardware, resolve_profile
from slicing_engine import ResinSettings, SliceParams, SlicingEngine
from validation import ValidationError

GIB = 1024**3


def hardware(cores=8, free_ram=16 * GIB, gpu=False, taichi=True) -> Hardware:
    return Hardware(
        cores=cores, free_ram=free_ram, gpu_name="Test GPU" if gpu else "",
        gpu_memory=8 * GIB if gpu else None, taichi_installed=taichi,
    )


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(compute, "init_taichi_gpu", lambda: None)


@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setattr(compute, "init_taichi_gpu", lambda: "cuda")


def test_economy_is_light_and_never_uses_the_gpu(fake_gpu) -> None:
    profile = resolve_profile("economy", hardware(gpu=True))
    assert (profile.mode, profile.device) == ("economy", "cpu")
    assert profile.lean_memory and profile.threads <= 2 and profile.slicing_threads == 1
    assert profile.workspace_cap < resolve_profile("maximum", hardware()).workspace_cap


def test_maximum_uses_the_gpu_when_there_is_one(fake_gpu) -> None:
    profile = resolve_profile("maximum", hardware(gpu=True))
    assert (profile.mode, profile.device) == ("maximum", "gpu")
    assert not profile.lean_memory and "Test GPU" in profile.gpu_name


def test_maximum_without_a_gpu_uses_every_core(no_gpu) -> None:
    profile = resolve_profile("maximum", hardware(cores=12))
    assert (profile.device, profile.threads) == ("cpu", 12)
    assert profile.io_threads > 2 and profile.slicing_threads > 2


def test_maximum_skips_a_gpu_that_is_too_small(fake_gpu) -> None:
    small = Hardware(8, 16 * GIB, "Tiny GPU", 1 * GIB, True)
    assert resolve_profile("maximum", small).device == "cpu"


def test_maximum_without_taichi_stays_on_the_cpu(fake_gpu) -> None:
    assert resolve_profile("maximum", hardware(taichi=False)).device == "cpu"


@pytest.mark.parametrize(
    "hw,expected",
    [
        (hardware(cores=2, gpu=True), "economy"),            # too few cores, even with a GPU
        (hardware(free_ram=2 * GIB, gpu=True), "economy"),   # too little memory
        (hardware(cores=4, free_ram=8 * GIB), "economy"),    # modest machine, no GPU
        (hardware(cores=8, free_ram=16 * GIB), "maximum"),   # strong CPU, no GPU
        (hardware(cores=4, free_ram=8 * GIB, gpu=True), "maximum"),  # GPU found
    ],
)
def test_auto_picks_a_mode_for_the_machine(hw, expected, monkeypatch) -> None:
    monkeypatch.setattr(compute, "init_taichi_gpu", lambda: "cuda" if hw.gpu_name else None)
    profile = resolve_profile("auto", hw)
    assert profile.mode == expected and profile.requested == "auto"


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_profile("turbo", hardware())
    params = SliceParams(
        diameter_mm=40.0, grid_res=64, output_res=128, num_frames=30, fill_holes=True,
        resin=ResinSettings(), output_dir=".", compute_mode="turbo",
    )
    with pytest.raises(ValidationError):
        params.validate()


def _frames(tmp_path, mode: str) -> np.ndarray:
    mesh = trimesh.creation.cylinder(radius=7.0, height=10.0, sections=40)
    params = SliceParams(
        diameter_mm=30.0, grid_res=32, output_res=64, num_frames=24, fill_holes=True,
        resin=ResinSettings(), output_dir=str(tmp_path / mode), projection_backend="physical",
        optimizer_iterations=5, cuvette_outer_mm=30.0, cuvette_inner_mm=28.0,
        n_air=1.0, n_glass=1.51, n_liquid=1.45, n_resin=1.5, compute_mode=mode,
        dose_window=True,  # exercise the window stage on every path, too
    )
    _, run_dir = SlicingEngine.run(mesh, params)
    return np.stack(
        [np.asarray(Image.open(path)) for path in sorted(glob.glob(f"{run_dir}/frame_*.png"))]
    ).astype(int)


def test_economy_and_maximum_cpu_give_the_same_frames(tmp_path, no_gpu) -> None:
    economy = _frames(tmp_path, "economy")
    maximum = _frames(tmp_path, "maximum")
    # The same float32 maths in a different summation order: at most one gray level.
    assert np.abs(economy - maximum).max() <= 1


def _gpu_available() -> bool:
    arch = compute.init_taichi_gpu()
    compute.shutdown_taichi()
    return arch is not None


@pytest.mark.skipif(not compute.detect_hardware().taichi_installed, reason="Taichi is not installed")
def test_gpu_matrix_matches_scipy_and_falls_back_on_failure(monkeypatch) -> None:
    if not _gpu_available():
        pytest.skip("No GPU backend available for Taichi")
    from optics.gpu_operator import GpuMatrix

    rng = np.random.default_rng(1)
    matrix = sp.random(400, 300, density=0.05, format="csr", dtype=np.float32, random_state=2)
    dense = rng.random((300, 17), dtype=np.float32)
    gpu = None
    try:
        compute.init_taichi_gpu()
        gpu = GpuMatrix(matrix)
        expected = matrix @ dense
        np.testing.assert_allclose(gpu @ dense, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(gpu @ dense[:, 0], expected[:, 0], rtol=1e-5, atol=1e-6)

        def broken(_values):
            raise RuntimeError("device lost")

        monkeypatch.setattr(gpu, "_device_matmul", broken)
        np.testing.assert_allclose(gpu @ dense, expected, rtol=1e-5, atol=1e-6)
        assert gpu._broken
    finally:
        if gpu is not None:
            gpu.release()
        gpu = None
        compute.shutdown_taichi()


@pytest.mark.skipif(not compute.detect_hardware().taichi_installed, reason="Taichi is not installed")
def test_gpu_and_cpu_give_the_same_frames(tmp_path, monkeypatch) -> None:
    if not _gpu_available():
        pytest.skip("No GPU backend available for Taichi")
    gpu = _frames(tmp_path, "maximum")
    monkeypatch.setattr(compute, "init_taichi_gpu", lambda: None)
    cpu = _frames(tmp_path, "economy")
    assert np.abs(gpu - cpu).max() <= 1


def test_sinogram_store_in_memory_and_on_disk_hold_the_same_numbers(tmp_path) -> None:
    from optics.sinogram_store import SinogramStore

    layers, detector, angles = 7, 5, 9
    rng = np.random.default_rng(3)
    blocks = [(0, 3), (3, 6), (6, 7)]
    memory = SinogramStore(layers, detector, angles)
    disk = SinogramStore(layers, detector, angles, directory=str(tmp_path))
    for start, stop in blocks:
        values = rng.random((angles * detector, stop - start), dtype=np.float32)
        memory.add_block(start, stop, values.copy())
        disk.add_block(start, stop, values.copy())
    assert disk.on_disk and not memory.on_disk
    assert memory.maximum == disk.maximum > 0
    for angle in range(angles):
        np.testing.assert_array_equal(memory.frame(angle), disk.frame(angle))
        assert memory.frame(angle).shape == (layers, detector)
    assert any(tmp_path.glob(".spinslicer-sino-*"))
    disk.close()
    assert not any(tmp_path.glob(".spinslicer-sino-*"))


def test_uint8_layers_project_exactly_like_float_layers() -> None:
    from optics.optimization import project_volume

    grid = 24
    rng = np.random.default_rng(5)
    levels = rng.integers(0, 256, size=(grid, grid, 3), dtype=np.uint8)
    angles = np.deg2rad(np.linspace(0.0, 360.0, 12, endpoint=False))
    kwargs = dict(L_out=30.0, L_in=30.0, grid_res=grid, grid_span_mm=30.0)
    from_uint8 = project_volume(levels, angles, **kwargs)
    from_float = project_volume(levels.astype(np.float32) / np.float32(255.0), angles, **kwargs)
    np.testing.assert_array_equal(from_uint8, from_float)


def test_economy_job_leaves_no_scratch_files_and_skips_the_bare_projection(tmp_path, no_gpu) -> None:
    import os

    mesh = trimesh.creation.cylinder(radius=6.0, height=8.0, sections=32)
    params = SliceParams(
        diameter_mm=30.0, grid_res=24, output_res=48, num_frames=16, fill_holes=True,
        resin=ResinSettings(), output_dir=str(tmp_path), projection_backend="physical",
        optimizer_iterations=3, cuvette_outer_mm=30.0, cuvette_inner_mm=28.0,
        n_air=1.0, n_glass=1.51, n_liquid=1.45, n_resin=1.5, compute_mode="economy",
    )
    _, run_dir = SlicingEngine.run(mesh, params)
    assert not os.path.exists(os.path.join(run_dir, "bare_projection"))
    assert not any(tmp_path.glob(".spinslicer-sino-*"))
    assert len(list(tmp_path.glob("run-*/frame_*.png"))) == 16
