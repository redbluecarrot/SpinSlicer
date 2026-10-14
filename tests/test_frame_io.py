import json

import pytest
from PIL import Image

from frame_io import (
    FrameRepository,
    GenerationManifest,
    SliceMeta,
    load_manifest,
    load_meta,
    save_manifest,
    save_meta,
)
from validation import ValidationError


def write_frame(directory, index: int, size: tuple[int, int] = (8, 6)) -> None:
    Image.new("L", size, color=index).save(directory / f"frame_{index:04d}.png")


def write_meta(directory, count: int = 2) -> None:
    save_meta(
        str(directory),
        SliceMeta(
            diameter_mm=60.0,
            grid_res=64,
            output_res=128,
            num_frames=count,
            fill_holes=True,
            resin_base_exposure=1.0,
            resin_intensity=100.0,
            resin_threshold=0.0,
        ),
    )


def test_run_directory_requires_complete_manifest(tmp_path) -> None:
    run_dir = tmp_path / "run-incomplete"
    run_dir.mkdir()
    write_frame(run_dir, 0)
    write_frame(run_dir, 1)
    write_meta(run_dir)

    with pytest.raises(ValidationError, match="not complete"):
        FrameRepository.validate(str(run_dir))


def test_complete_run_is_resolved_from_output_root(tmp_path) -> None:
    run_dir = tmp_path / "run-complete"
    run_dir.mkdir()
    write_frame(run_dir, 0)
    write_frame(run_dir, 1)
    write_meta(run_dir)
    save_manifest(
        str(run_dir),
        GenerationManifest(
            frame_count=2,
            frame_size=(8, 6),
            complete=True,
        ),
    )

    info = FrameRepository.validate(str(tmp_path))

    assert info.resolved_dir == str(run_dir)
    assert info.frame_count == 2
    assert info.width == 8
    assert info.height == 6
    assert load_manifest(str(run_dir)).frame_size == (8, 6)


def test_completed_run_takes_precedence_over_legacy_flat_frames(tmp_path) -> None:
    write_frame(tmp_path, 0)
    write_frame(tmp_path, 1)

    run_dir = tmp_path / "run-new"
    run_dir.mkdir()
    write_frame(run_dir, 0)
    write_frame(run_dir, 1)
    write_meta(run_dir)
    save_manifest(
        str(run_dir),
        GenerationManifest(frame_count=2, frame_size=(8, 6), complete=True),
    )

    info = FrameRepository.validate(str(tmp_path))

    assert info.resolved_dir == str(run_dir)


def test_legacy_flat_directory_remains_supported(tmp_path) -> None:
    write_frame(tmp_path, 0)
    write_frame(tmp_path, 1)

    info = FrameRepository.validate(str(tmp_path))

    assert info.frame_count == 2
    assert info.meta is None
    assert load_meta(str(tmp_path)) is None


def test_mismatched_dimensions_are_rejected(tmp_path) -> None:
    write_frame(tmp_path, 0, (8, 6))
    write_frame(tmp_path, 1, (9, 6))

    with pytest.raises(ValidationError, match="identical dimensions"):
        FrameRepository.validate(str(tmp_path))


def test_frame_dimension_budget_is_checked_before_decode(monkeypatch, tmp_path) -> None:
    import frame_io

    write_frame(tmp_path, 0)
    write_frame(tmp_path, 1)
    monkeypatch.setattr(frame_io, "MAX_FRAME_DIMENSION", 4)

    with pytest.raises(ValidationError, match="dimension"):
        FrameRepository.validate(str(tmp_path))


def test_frame_pixel_budget_is_checked(monkeypatch, tmp_path) -> None:
    import frame_io

    write_frame(tmp_path, 0)
    write_frame(tmp_path, 1)
    monkeypatch.setattr(frame_io, "MAX_FRAME_TOTAL_PIXELS", 10)

    with pytest.raises(ValidationError, match="pixels"):
        FrameRepository.validate(str(tmp_path))


def test_mismatched_metadata_count_is_rejected(tmp_path) -> None:
    write_frame(tmp_path, 0)
    write_frame(tmp_path, 1)
    write_meta(tmp_path, count=3)

    with pytest.raises(ValidationError, match="expects 3"):
        FrameRepository.validate(str(tmp_path))


def test_malformed_metadata_is_not_silently_accepted(tmp_path) -> None:
    (tmp_path / "slice_meta.json").write_text("{bad", encoding="utf-8")

    with pytest.raises(ValidationError, match="Invalid slice_meta.json"):
        load_meta(str(tmp_path))


def test_manifest_json_round_trip_is_atomic_shape(tmp_path) -> None:
    run_dir = tmp_path / "run-shape"
    run_dir.mkdir()
    manifest = GenerationManifest(frame_count=2, frame_size=(8, 6), complete=True)

    save_manifest(str(run_dir), manifest)
    raw = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert raw["complete"] is True
    assert load_manifest(str(run_dir)).frame_size == (8, 6)


def test_complete_manifest_requires_metadata(tmp_path) -> None:
    run_dir = tmp_path / "run-no-meta"
    run_dir.mkdir()
    write_frame(run_dir, 0)
    write_frame(run_dir, 1)
    save_manifest(
        str(run_dir),
        GenerationManifest(frame_count=2, frame_size=(8, 6), complete=True),
    )

    with pytest.raises(ValidationError, match="slice_meta"):
        FrameRepository.validate(str(run_dir))
