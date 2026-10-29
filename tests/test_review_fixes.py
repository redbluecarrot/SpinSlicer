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


def _stream_luma(path, width: int, height: int, stored: str = "pc") -> np.ndarray:
    """Luma plane exactly as stored in the stream, with no range conversion.

    ``stored`` is the range the stream was written in; naming it for both ends of the scaler
    keeps every ffmpeg build from rescaling a full-range stream while converting to yuv420p.
    """
    ffmpeg = workers._ffmpeg_executable()
    if ffmpeg is None:
        pytest.skip("ffmpeg is not available")
    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-vframes", "1",
         "-vf", f"scale=in_range={stored}:out_range={stored}", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
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
def test_economy_says_that_it_skips_the_direct_projection(tmp_path, no_gpu) -> None:
    messages: list[str] = []
    mesh = trimesh.creation.cylinder(radius=6.0, height=8.0, sections=32)
    SlicingEngine.run(mesh, _params(tmp_path, compute_mode="economy"), progress_cb=lambda f, m: messages.append(m))
    assert any("прямая проекция не создаётся" in m or "direct projection is not created" in m
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


# ------------------------------------------------------------------ second review
def test_second_job_reuses_the_main_ray_matrix(tmp_path, no_gpu, monkeypatch) -> None:
    built = []
    real = optimization._stack_csr
    monkeypatch.setattr(optimization, "_stack_csr", lambda batches: (built.append(len(batches)), real(batches))[1])
    optimization._matrix_cache.clear()
    mesh = trimesh.creation.cylinder(radius=6.0, height=8.0, sections=32)
    try:
        SlicingEngine.run(mesh, _params(tmp_path / "a", compute_mode="maximum"))
        first_job = len(built)
        SlicingEngine.run(mesh, _params(tmp_path / "b", compute_mode="maximum"))
    finally:
        optimization._matrix_cache.clear()
    assert first_job == 2                 # the direct projection's matrix and the main one
    assert len(built) - first_job == 1    # the main matrix came from the cache the second time


def test_an_uncached_matrix_does_not_evict_the_cached_one() -> None:
    angles = np.deg2rad(np.linspace(0.0, 360.0, 6, endpoint=False))
    args = (angles, 16, 16, 30.0, 30.0, 28.0, 1.0, 1.51, 1.45, 1.5, 0.0, 0)
    optimization._matrix_cache.clear()
    main = optimization.build_system_matrix(*args)
    optimization.build_system_matrix(
        angles, 16, 16, 30.0, 30.0, 30.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0, use_cache=False
    )
    assert optimization.build_system_matrix(*args) is main
    optimization._matrix_cache.clear()


def test_auto_leaves_a_gpu_of_unknown_memory_alone(monkeypatch) -> None:
    started = []
    monkeypatch.setattr(compute, "init_taichi_gpu", lambda: started.append(1) or "vulkan")
    integrated = compute.Hardware(
        cores=8, free_ram=16 * 1024**3, gpu_name="Intel UHD", gpu_memory=None, taichi_installed=True
    )
    auto = compute.resolve_profile("auto", integrated)
    assert auto.device == "cpu" and not started          # and Taichi was not even started
    explicit = compute.resolve_profile("maximum", integrated)
    assert explicit.device == "gpu" and started == [1]   # an explicit choice may use it


def test_limited_range_option_stores_16_to_235_and_flags_it(tmp_path) -> None:
    ramp = workers.calibration_ramp(116, 32)
    frames = [ramp] * 4
    limited = _export(tmp_path, frames, video_range="limited")
    luma = _stream_luma(limited, 116, 32, "tv")
    assert luma.min() == 16 and luma.max() == 235       # honest limited range in the stream
    decoded = _decode_gray(limited)                     # a player that reads the flag restores 0..255
    assert decoded.min() <= 1 and decoded.max() >= 254


def test_full_range_is_the_default_and_keeps_the_bars_apart(tmp_path) -> None:
    ramp = workers.calibration_ramp(116, 32)
    path = _export(tmp_path, [ramp] * 4)
    luma = _stream_luma(path, 116, 32)
    assert luma.min() == 0 and luma.max() >= 254
    levels = {int(round(float(np.median(luma[:, i * 4: i * 4 + 4])))) for i in range(29)}
    assert len(levels) >= 26                            # the bars stay distinguishable


def test_calibration_ramp_covers_both_ends_of_the_range() -> None:
    ramp = workers.calibration_ramp(580, 40)
    assert ramp.shape == (40, 580) and ramp.dtype == np.uint8
    present = set(np.unique(ramp).tolist())
    assert {0, 2, 4, 12, 16, 235 + 1, 254, 255} <= present
    assert present == set(workers.CALIBRATION_LEVELS)


def test_export_rejects_an_unknown_range(tmp_path) -> None:
    worker = VideoExportWorker([np.zeros((16, 16), np.uint8)], 30.0, str(tmp_path / "x.mp4"), video_range="wide")
    errors: list[str] = []
    worker.failed.connect(errors.append)
    worker.run()
    assert errors and "video_range" in errors[0]


# ------------------------------------------------------------------ third review: the GPU add-on must really work
needs_taichi = pytest.mark.skipif(
    not compute.detect_hardware().taichi_installed, reason="Taichi is not installed"
)


def _reset_taichi() -> None:
    import gc

    import taichi as ti

    gc.collect()
    ti.reset()
    compute._taichi_state["generation"] += 1


@needs_taichi
def test_selftest_runs_every_gpu_kernel_on_the_cpu_backend() -> None:
    result = compute.selftest_gpu_kernels("cpu")
    assert result["ok"] is True and result["error"] is None
    assert result["arch"] in {"x64", "arm64", "cpu"}   # Taichi names its CPU backend x64 or arm64


@needs_taichi
def test_selftest_catches_a_build_without_the_kernel_source(monkeypatch) -> None:
    """A packaged build has bytecode only; Taichi then cannot read the kernel source."""
    import inspect

    import optics.gpu_operator as gpu_operator

    def no_source(*args, **kwargs):
        raise OSError("Cannot find source code for Object")

    gpu_operator._kernels.clear()
    gpu_operator._step_kernels.clear()
    monkeypatch.setattr(inspect, "getsourcelines", no_source)
    monkeypatch.setattr(inspect, "getsource", no_source)
    result = compute.selftest_gpu_kernels("cpu")
    assert result["ok"] is False and "OSError" in result["error"]
    monkeypatch.undo()
    gpu_operator._kernels.clear()
    gpu_operator._step_kernels.clear()
    assert compute.selftest_gpu_kernels("cpu")["ok"] is True      # and it recovers


def _tiny_optimisation(profile, status, messages):
    target = np.zeros((16, 16, 3), dtype=np.uint8)
    target[5:11, 4:12, :] = 255
    angles = np.deg2rad(np.linspace(0.0, 360.0, 12, endpoint=False))
    sink = {}
    optimization.optimize_projections_sirt_volume(
        target, angles, L_out=30.0, L_in=28.0, grid_res=16, grid_span_mm=30.0, iterations=3,
        window_iterations=2, profile=profile, status=status,
        report_cb=lambda fraction, text: messages.append(text),
        sink=lambda start, stop, values: sink.__setitem__(start, values.copy()),
    )
    return np.concatenate([sink[key] for key in sorted(sink)], axis=1)


def _gpu_profile():
    base = compute.resolve_profile("economy")
    return type(base)(**{**base.__dict__, "mode": "maximum", "device": "gpu", "gpu_name": "test",
                         "lean_memory": False, "threads": 2, "save_bare": True, "disk_sinograms": False})


@needs_taichi
def test_device_path_gives_the_cpu_numbers_and_says_it_ran_on_the_device() -> None:
    import taichi as ti

    ti.init(arch=ti.cpu, log_level=ti.ERROR, default_fp=ti.f32, default_ip=ti.i32)
    try:
        status: dict = {}
        on_device = _tiny_optimisation(_gpu_profile(), status, [])
    finally:
        _reset_taichi()
    cpu_profile = compute.resolve_profile("maximum", use_gpu=False)
    on_cpu = _tiny_optimisation(cpu_profile, {}, [])
    assert status == {"device_used": "gpu", "gpu_error": ""}
    np.testing.assert_allclose(on_device, on_cpu, rtol=1e-4, atol=1e-5)


@needs_taichi
def test_a_failing_device_is_reported_and_the_job_finishes_on_the_cpu(monkeypatch) -> None:
    import taichi as ti

    import optics.gpu_operator as gpu_operator

    def broken(self, values):
        raise RuntimeError("Cannot find source code for Object")

    monkeypatch.setattr(gpu_operator.GpuMatrix, "_device_matmul", broken)
    ti.init(arch=ti.cpu, log_level=ti.ERROR, default_fp=ti.f32, default_ip=ti.i32)
    messages: list[str] = []
    status: dict = {}
    try:
        result = _tiny_optimisation(_gpu_profile(), status, messages)
    finally:
        _reset_taichi()
    assert status["device_used"] == "cpu" and "Cannot find source code" in status["gpu_error"]
    warning = [m for m in messages if "GPU" in m and "процессоре" in m or "The GPU is unavailable" in m]
    assert warning and "Cannot find source code" in warning[0]       # the user is told, not left guessing
    cpu = _tiny_optimisation(compute.resolve_profile("maximum", use_gpu=False), {}, [])
    np.testing.assert_allclose(result, cpu, rtol=1e-4, atol=1e-5)    # and the result is the same


def test_the_manifest_records_what_really_ran(tmp_path, no_gpu) -> None:
    import json

    mesh = trimesh.creation.cylinder(radius=6.0, height=8.0, sections=32)
    _, run_dir = SlicingEngine.run(mesh, _params(tmp_path, compute_mode="economy"))
    compute_info = json.loads(open(f"{run_dir}/manifest.json", encoding="utf-8").read())["slice_parameters"]["compute"]
    assert compute_info["device_used"] == "cpu" and compute_info["gpu_error"] == ""


def test_a_repeated_ramp_frame_is_counted_once_by_the_export(tmp_path, monkeypatch) -> None:
    ramp = workers.calibration_ramp(64, 64)
    monkeypatch.setattr(workers, "MAX_VIDEO_PREVIEW_BYTES", 3 * ramp.nbytes)
    path = _export(tmp_path, [ramp] * 40)         # 40 references to one array: it must not trip the limit
    assert path.stat().st_size > 0
    with pytest.raises(AssertionError):           # 40 different arrays do exceed it
        _export(tmp_path / "other", [ramp.copy() for _ in range(40)]) if (tmp_path / "other").mkdir() is None else None


def test_ramp_tooltips_and_defaults_are_honest() -> None:
    source = open(os.path.join(os.path.dirname(slicing_engine.__file__), "video_tab.py"), encoding="utf-8").read()
    assert "безопаснее для большинства плееров" in source
    assert "не теряет ничего ни в одном плеере" not in source
    assert "HDMI" in source                              # the case where limited range goes wrong


def test_the_ramp_is_linear_unless_the_profile_gamma_is_asked_for(qapp_for_tabs) -> None:
    from video_tab import ProjectorTab

    tab = ProjectorTab()
    assert not tab.ramp_gamma_box.isChecked()


@pytest.fixture(scope="module")
def qapp_for_tabs():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
