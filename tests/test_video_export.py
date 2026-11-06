"""MP4 export for the projector: H.264, full gray range, full-resolution frames."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from spinslicer import workers  # noqa: E402
from spinslicer.workers import VideoExportWorker, calibration_ramp  # noqa: E402


def _run(tmp_path, frames, **kwargs):
    path = tmp_path / "out.mp4"
    worker = VideoExportWorker(frames, 30.0, str(path), **kwargs)
    errors: list[str] = []
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors, errors
    return path


def _first_frame(path) -> tuple[np.ndarray, str]:
    capture = cv2.VideoCapture(str(path))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    ok, frame = capture.read()
    capture.release()
    assert ok
    return frame[:, :, 0].astype(float), "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))


def test_h264_keeps_gray_levels_and_pads_odd_sizes(tmp_path):
    if workers._ffmpeg_executable() is None:
        pytest.skip("no ffmpeg")
    ramp = np.tile(np.linspace(0, 255, 63).astype(np.uint8), (47, 1))
    frame, codec = _first_frame(_run(tmp_path, [ramp] * 5))
    assert codec.lower() in {"avc1", "h264"}
    assert frame.shape == (48, 64)                    # padded with black, not resampled
    assert np.abs(frame[:47, :63] - ramp).mean() < 2.0
    assert frame[:47, 0].mean() < 1.5                 # black stays black (full range)


def test_png_paths_are_exported_at_full_resolution(tmp_path):
    from PIL import Image
    paths = []
    for i in range(4):
        p = tmp_path / f"frame_{i:04d}.png"
        Image.fromarray(np.full((40, 60), 100 + i, np.uint8)).save(p)
        paths.append(str(p))
    frame, _ = _first_frame(_run(tmp_path, paths))
    assert frame.shape == (40, 60)


def test_falls_back_to_mp4v_without_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(workers, "_ffmpeg_executable", lambda: None)
    _frame, codec = _first_frame(_run(tmp_path, [np.full((32, 32), 100, np.uint8)] * 4))
    assert codec.lower() in {"mp4v", "fmp4"}


def test_calibration_ramp_covers_both_ends():
    ramp = calibration_ramp(290, 10)
    assert ramp.min() == 0 and ramp.max() == 255
    assert len(np.unique(ramp)) == len(workers.CALIBRATION_LEVELS)


def test_print_plan_travels_with_the_video(tmp_path):
    """The exposure plan goes into the MP4 comment tag and a .spin.json beside it."""
    import json
    import shutil
    import subprocess

    if workers._ffmpeg_executable() is None:
        pytest.skip("no ffmpeg")
    meta = {"format": "spinslicer-print/1", "fps": 30.0, "frames": 5, "revolutions": 7.5, "print_time_s": 1.25}
    frame = np.zeros((16, 16), np.uint8)
    path = _run(tmp_path, [frame] * 5, metadata=meta)
    sidecar = path.with_suffix(".spin.json")
    assert json.loads(sidecar.read_text(encoding="utf-8")) == meta
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format_tags=comment", "-of", "json",
                              str(path)], capture_output=True, text=True).stdout
        assert json.loads(json.loads(out)["format"]["tags"]["comment"]) == meta


def test_no_sidecar_without_a_plan(tmp_path):
    path = _run(tmp_path, [np.zeros((16, 16), np.uint8)] * 3)
    assert not path.with_suffix(".spin.json").exists()
