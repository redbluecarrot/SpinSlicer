"""Regression tests for the issues found in the review of the export and compute code."""
import glob
import os
import subprocess
import time

import numpy as np
import pytest
import scipy.sparse as sp
import trimesh
from PIL import Image

import compute
import slicing_engine
import workers
from constants import BARE_PROJECTION_DIR, PROJECTOR_FRAMES_DIR
from optics import optimization
from profiles import MachineProfile, ProjectorProfile, VatProfile
from slicing_engine import ResinSettings, SliceParams, SlicingEngine
from validation import ValidationError, validate_slice_parameters
from workers import VideoExportWorker


def _params(tmp_path, **overrides) -> SliceParams:
    values = dict(
        diameter_mm=30.0, grid_res=24, output_res=48, num_frames=16, fill_holes=True,
        resin=ResinSettings(), output_dir=str(tmp_path), projection_backend="physical",
        optimizer_iterations=3, cuvette_outer_mm=30.0, cuvette_inner_mm=28.0,
        n_air=1.0, n_glass=1.51, n_liquid=1.45, n_resin=1.5,
    )
    values.update(overrides)
    return SliceParams(**values)


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(compute, "init_taichi_gpu", lambda: None)


# ------------------------------------------------------------------ export
def _decode_gray(path, frame: int = 0) -> np.ndarray:
    ffmpeg = workers._ffmpeg_executable()
    if ffmpeg is None:
        pytest.skip("ffmpeg is not available")
    import cv2

    capture = cv2.VideoCapture(str(path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{frame})", "-vframes", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width)


def _stream_luma(path, width: int, height: int) -> np.ndarray:
    """Luma plane exactly as stored in the stream, with no range conversion."""
    ffmpeg = workers._ffmpeg_executable()
    if ffmpeg is None:
        pytest.skip("ffmpeg is not available")
    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8)[: width * height].reshape(height, width)


def _export(tmp_path, frames, **kwargs):
    path = tmp_path / "out.mp4"
    worker = VideoExportWorker(frames, 30.0, str(path), **kwargs)
    errors: list[str] = []
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors, errors
    return path


def test_mp4_keeps_black_at_zero_and_white_at_255(tmp_path) -> None:
    gradient = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
    decoded = _decode_gray(_export(tmp_path, [gradient] * 4)).astype(int)
    assert decoded[:, 0].max() <= 1          # black is 0, not 16
    assert decoded[:, 255].min() >= 254      # white is 255, not 235
    assert np.abs(decoded - gradient.astype(int)).max() <= 3
    assert len(np.unique(decoded)) > 230     # no loss of gray levels


def test_stream_black_background_is_zero_not_sixteen(tmp_path) -> None:
    picture = np.zeros((64, 128), dtype=np.uint8)
    picture[16:48, 64:96] = 255
    luma = _stream_luma(_export(tmp_path, [picture] * 4), 128, 64)
    assert luma[:, :48].max() == 0           # background: no light at all (limited range gave 16)
    assert luma[20:44, 68:92].min() >= 254   # white stays white (limited range gave 235)


def test_export_reads_png_files_at_their_full_resolution(tmp_path) -> None:
    paths = []
    for index in range(4):
        picture = np.full((600, 1000), 40 + 10 * index, dtype=np.uint8)
        path = tmp_path / f"frame_{index:04d}.png"
        Image.fromarray(picture).save(path)
        paths.append(str(path))
    decoded = _decode_gray(_export(tmp_path, paths))
    assert decoded.shape == (600, 1000)       # not the 900 px preview size
    assert abs(float(decoded.mean()) - 40) < 2


def test_projector_frames_hold_the_gamma_applied_before_rounding(tmp_path, no_gpu) -> None:
    machine = MachineProfile(
        name="gamma-test",
        projector=ProjectorProfile(name="p", width_px=48, height_px=64, fps=30.0, gamma=2.2),
        vat=VatProfile(name="v", diameter_mm=30.0, height_mm=48.0),
    )
    mesh = trimesh.creation.cylinder(radius=6.0, height=8.0, sections=32)
    _, run_dir = SlicingEngine.run(mesh, _params(tmp_path), manifest_context={"machine_profile": machine})
    linear = np.asarray(Image.open(f"{run_dir}/frame_0003.png"), dtype=float)
    encoded = np.asarray(Image.open(f"{run_dir}/{PROJECTOR_FRAMES_DIR}/frame_0003.png"), dtype=float)
    through_table = np.rint(255.0 * (linear / 255.0) ** (1 / 2.2))
    assert len(glob.glob(f"{run_dir}/{PROJECTOR_FRAMES_DIR}/frame_*.png")) == 16
    assert np.abs(encoded - through_table).mean() < 1.0
    # Gamma on the float frame is not the same as gamma on the rounded one.
    assert (encoded != through_table).any()
    # The PNG frames stay linear.
    assert linear.max() > 0 and encoded.mean() > linear.mean()


def test_linear_projector_writes_no_encoded_frames(tmp_path, no_gpu) -> None:
    mesh = trimesh.creation.cylinder(radius=6.0, height=8.0, sections=32)
    _, run_dir = SlicingEngine.run(mesh, _params(tmp_path))
    assert not os.path.isdir(f"{run_dir}/{PROJECTOR_FRAMES_DIR}")


def test_rpm_in_the_file_name_has_at_most_two_decimals() -> None:
    from video_tab import ProjectorTab

    assert ProjectorTab._format_rpm(5.0) == "5"
    assert ProjectorTab._format_rpm(4.5) == "4.5"
    assert ProjectorTab._format_rpm(60.0 * 30 / 432) == "4.17"


def test_output_res_must_be_even(tmp_path) -> None:
    kwargs = dict(
        diameter_mm=30.0, grid_res=64, num_frames=30, resin=ResinSettings(),
    )
    validate_slice_parameters(output_res=720, **kwargs)
    with pytest.raises(ValidationError, match="even"):
        validate_slice_parameters(output_res=721, **kwargs)


def test_constants_and_defaults_are_shared() -> None:
    import validation
    import video_tab

    assert video_tab.BARE_PROJECTION_DIR is BARE_PROJECTION_DIR is slicing_engine.BARE_PROJECTION_DIR
    assert validation.DEFAULT_FRAME_RATE_HZ == 30.0


# ------------------------------------------------------------------ optimiser
def _small_system(seed=0):
    rng = np.random.default_rng(seed)
    a = sp.random(60, 40, density=0.2, format="csr", dtype=np.float32, random_state=seed)
    at = a.T.tocsr()
    row = np.asarray(a.sum(axis=1)).ravel()
    col = np.asarray(at.sum(axis=1)).ravel()
    ray_norm = np.divide(1, row, out=np.zeros_like(row), where=row > 0).astype(np.float32)
    voxel_norm = np.divide(1, col, out=np.zeros_like(col), where=col > 0).astype(np.float32)
    values = rng.random((60, 5), dtype=np.float32)
    target = (rng.random((40, 5)) > 0.6).astype(np.float32)
    hits = values > 0.1
    return a, at, values, target, ray_norm, voxel_norm, hits


class _FakeNativeLibrary:
    """Stands in for the C++ library: forward_project over raw CSR pointers."""

    def forward_project(self, rows, columns, layers, values, indptr, indices, data, result):
        pointer = np.ctypeslib.as_array
        row_pointers = pointer(indptr, shape=(rows + 1,))
        count = int(row_pointers[-1])
        matrix = sp.csr_matrix(
            (pointer(data, shape=(count,)), pointer(indices, shape=(count,)), row_pointers),
            shape=(rows, columns),
        )
        pointer(result, shape=(rows, layers))[:] = matrix @ pointer(values, shape=(columns, layers))


def test_window_step_gives_the_same_values_through_the_native_path() -> None:
    a, at, values, target, ray_norm, voxel_norm, hits = _small_system()
    python_values = values.copy()
    native_values = values.copy()
    optimization._window_step(a, at, python_values, target, ray_norm, voxel_norm, hits, None, 0.8, 10.0)
    optimization._window_step(
        a, at, native_values, target, ray_norm, voxel_norm, hits, _FakeNativeLibrary(), 0.8, 10.0
    )
    np.testing.assert_allclose(native_values, python_values, rtol=1e-6, atol=1e-7)
    assert not np.array_equal(python_values, values)


def test_window_level_and_steepness_change_the_step() -> None:
    a, at, values, target, ray_norm, voxel_norm, hits = _small_system(1)
    low, high = values.copy(), values.copy()
    optimization._window_step(a, at, low, target, ray_norm, voxel_norm, hits, None, 0.3, 10.0)
    optimization._window_step(a, at, high, target, ray_norm, voxel_norm, hits, None, 1.2, 10.0)
    assert not np.allclose(low, high)


def test_window_parameters_are_validated() -> None:
    with pytest.raises(ValidationError):
        _params(".", window_level=0.0).validate()
    with pytest.raises(ValidationError):
        _params(".", window_steepness=100.0).validate()


def test_the_window_docstring_is_honest_about_the_objective() -> None:
    doc = optimization.optimize_projections_sirt_volume.__doc__ or ""
    assert "heuristic" in doc and "uncalibrated sigmoid changes" not in doc
    assert "not the gradient" in (optimization._window_step.__doc__ or "")


@pytest.mark.skipif(not compute.detect_hardware().taichi_installed, reason="Taichi is not installed")
def test_operators_release_the_first_gpu_matrix_when_the_second_fails(monkeypatch) -> None:
    import optics.gpu_operator as gpu_operator

    created = []

    class FlakyMatrix:
        def __init__(self, matrix):
            if created:
                raise RuntimeError("out of device memory")
            self.released = False
            created.append(self)

        def release(self):
            self.released = True

    monkeypatch.setattr(gpu_operator, "GpuMatrix", FlakyMatrix)
    a, at, *_ = _small_system()
    profile = compute.resolve_profile("economy")
    gpu_profile = type(profile)(**{**profile.__dict__, "device": "gpu", "lean_memory": False, "threads": 2})
    a_op, at_op, library, label = optimization._operators(a, at, gpu_profile, None)
    assert created[0].released                         # freed here, not by a random garbage collection
    assert isinstance(a_op, optimization.CpuMatrix)    # and the job continues on the CPU
    assert "CPU" in label


@pytest.mark.skipif(not compute.detect_hardware().taichi_installed, reason="Taichi is not installed")
def test_gpu_resident_steps_match_the_cpu_steps() -> None:
    if compute.init_taichi_gpu() is None:
        compute.shutdown_taichi()
        pytest.skip("No GPU backend available for Taichi")
    from optics.gpu_operator import GpuMatrix, GpuSolver

    a, at, values, target, ray_norm, voxel_norm, hits = _small_system(2)
    expected = values.copy()
    workspace = np.empty_like(target)
    for _ in range(3):
        optimization._python_sirt_step(a, at, expected, target, ray_norm, voxel_norm, hits, workspace)
    for _ in range(3):
        optimization._window_step(a, at, expected, target, ray_norm, voxel_norm, hits, None, 0.8, 10.0)
    gpu_a = gpu_at = solver = None
    try:
        gpu_a, gpu_at = GpuMatrix(a), GpuMatrix(at)
        solver = GpuSolver(gpu_a, gpu_at, ray_norm, voxel_norm)
        steps = []
        result = solver.run(values.copy(), target, hits, 3, 3, 0.8, 10.0, lambda kind, i: steps.append(kind))
    finally:
        optimization._release(solver, gpu_a, gpu_at)
        compute.shutdown_taichi()
    assert steps == ["ls"] * 3 + ["window"] * 3
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------ compute modes and cleanup
def test_economy_says_that_it_skips_the_uncorrected_projection(tmp_path, no_gpu) -> None:
    messages: list[str] = []
    mesh = trimesh.creation.cylinder(radius=6.0, height=8.0, sections=32)
    SlicingEngine.run(mesh, _params(tmp_path, compute_mode="economy"), progress_cb=lambda f, m: messages.append(m))
    assert any("нескорректированная проекция не создаётся" in m or "uncorrected projection is not created" in m
               for m in messages)
    assert not glob.glob(f"{tmp_path}/run-*/{BARE_PROJECTION_DIR}")


def test_ray_matrix_cache_follows_the_mode(tmp_path, no_gpu) -> None:
    mesh = trimesh.creation.cylinder(radius=6.0, height=8.0, sections=32)
    SlicingEngine.run(mesh, _params(tmp_path / "e", compute_mode="economy"))
    assert not optimization._matrix_cache            # economy gives the memory back
    SlicingEngine.run(mesh, _params(tmp_path / "m", compute_mode="maximum"))
    assert optimization._matrix_cache                # maximum keeps it for the next job
    optimization._matrix_cache.clear()


def test_ordered_parallel_cancels_what_has_not_started() -> None:
    calls = []

    def work(index: int) -> int:
        calls.append(index)
        time.sleep(0.02)
        if index == 1:
            raise RuntimeError("disk full")
        return index

    with pytest.raises(RuntimeError, match="disk full"):
        for _ in slicing_engine._ordered_parallel(work, ((i,) for i in range(40)), 1):
            pass
    assert len(calls) < 6   # the other ~35 frames were not written after the failure


def test_failed_export_removes_scratch_files_and_stops_early(tmp_path, no_gpu, monkeypatch) -> None:
    resized = []
    real = slicing_engine.sk_resize

    def failing(*args, **kwargs):
        resized.append(1)
        if len(resized) == 3:
            raise OSError("disk full")
        return real(*args, **kwargs)

    monkeypatch.setattr(slicing_engine, "sk_resize", failing)
    mesh = trimesh.creation.cylinder(radius=6.0, height=8.0, sections=32)
    with pytest.raises(OSError, match="disk full"):
        SlicingEngine.run(mesh, _params(tmp_path, compute_mode="economy", num_frames=40))
    assert not glob.glob(f"{tmp_path}/.spinslicer-sino-*")
    assert len(resized) < 10                    # not all 40 frames were processed after the error
