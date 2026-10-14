from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_folder_opening_does_not_construct_shell_commands() -> None:
    source = (ROOT / "slicer_tab.py").read_text(encoding="utf-8")

    assert "os.system" not in source
    assert "QDesktopServices.openUrl" in source


@pytest.mark.parametrize(
    "suffix",
    [
        "folder with spaces",
        'folder "with quotes"',
        "folder;semicolon",
        "folder $()",
        "folder `backticks`",
    ],
)
def test_folder_opening_passes_special_paths_as_local_urls(monkeypatch, tmp_path, suffix) -> None:
    try:
        import slicer_tab
    except (ImportError, OSError, Exception) as exc:
        pytest.skip(f"slicer_tab cannot be imported in headless/non-GUI environment: {exc}")
    captured = {}

    class FakeDesktopServices:
        @staticmethod
        def openUrl(url):
            captured["path"] = url.toLocalFile()
            return True

    monkeypatch.setattr(slicer_tab, "QDesktopServices", FakeDesktopServices)
    target = str(tmp_path / suffix)

    assert slicer_tab.open_local_directory(target)
    assert Path(captured["path"]) == Path(target).absolute()


def test_generation_no_longer_deletes_previous_frame_set() -> None:
    source = (ROOT / "slicing_engine.py").read_text(encoding="utf-8")

    assert "glob.glob" not in source
    assert "os.remove(stale_path)" not in source
    assert "FrameRepository.create_run_dir" in source
