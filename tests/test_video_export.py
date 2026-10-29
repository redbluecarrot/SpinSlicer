import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import workers  # noqa: E402
from profiles import ProfileValidationError, ProjectorProfile  # noqa: E402
from workers import VideoExportWorker  # noqa: E402


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
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    return frame[:, :, 0].astype(float), codec


def _export(tmp_path, gamma: float) -> np.ndarray:
    frames = [np.full((64, 64), 128, dtype=np.uint8) for _ in range(6)]
    frame, _codec = _first_frame(_run(tmp_path, frames, projector_gamma=gamma))
    return frame


def test_export_applies_projector_gamma_precompensation(tmp_path) -> None:
    linear = _export(tmp_path, 1.0).mean()
    corrected = _export(tmp_path, 2.2).mean()
    # 128/255 encoded for gamma 2.2 is 255 * 0.502 ** (1 / 2.2) ~ 186.
    assert linear == pytest.approx(128, abs=3)
    assert corrected == pytest.approx(186, abs=3)


def test_export_keeps_frames_linear_by_default(tmp_path) -> None:
    frames = [np.full((32, 32), 128, dtype=np.uint8) for _ in range(4)]
    frame, _codec = _first_frame(_run(tmp_path, frames))
    assert frame.mean() == pytest.approx(128, abs=3)


def test_export_encodes_h264_with_odd_frame_size(tmp_path) -> None:
    ramp = np.tile(np.linspace(0, 255, 63).astype(np.uint8), (47, 1))
    frame, codec = _first_frame(_run(tmp_path, [ramp] * 5))
    assert codec.lower() in {"avc1", "h264"}
    # Padded to even sides with black, the picture itself is untouched.
    assert frame.shape == (48, 64)
    assert np.abs(frame[:47, :63] - ramp).mean() < 2.0


def test_export_falls_back_to_mp4v_without_ffmpeg(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(workers, "_ffmpeg_executable", lambda: None)
    frames = [np.full((32, 32), 100, dtype=np.uint8) for _ in range(4)]
    _frame, codec = _first_frame(_run(tmp_path, frames))
    assert codec.lower() in {"mp4v", "fmp4"}  # MPEG-4 Part 2, as OpenCV reports it


def test_export_rejects_nonsense_gamma(tmp_path) -> None:
    worker = VideoExportWorker([np.zeros((16, 16), np.uint8)], 30.0, str(tmp_path / "x.mp4"), projector_gamma=9.0)
    errors: list[str] = []
    worker.failed.connect(errors.append)
    worker.run()
    assert errors and "gamma" in errors[0].lower()


def test_profile_gamma_range_matches_the_export() -> None:
    ProjectorProfile(gamma=2.2).validate()
    with pytest.raises(ProfileValidationError):
        ProjectorProfile(gamma=9.0).validate()
