"""The in-app updater: versions, asset choice, verified download, binary swap."""
from __future__ import annotations

import hashlib
import io
import tarfile
import tomllib
from pathlib import Path

import pytest

from spinslicer import __version__, updater

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_pyproject() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert __version__ == project["version"]


@pytest.mark.parametrize(("text", "expected"), [
    ("v0.9.0", (0, 9, 0)), ("0.10", (0, 10, 0)), ("v1.2.3-rc1", (1, 2, 3)), ("nightly", None), ("", None),
])
def test_parse_version(text: str, expected) -> None:
    assert updater.parse_version(text) == expected


def test_is_newer_compares_numerically() -> None:
    assert updater.is_newer("v0.10.0", "0.9.9")
    assert not updater.is_newer("v0.8.1", "0.8.1")
    assert not updater.is_newer("v0.8.0", "0.8.1")
    assert not updater.is_newer("garbage", "0.8.1")


RELEASE_JSON = {
    "tag_name": "v0.9.0",
    "html_url": "https://github.com/redbluecarrot/SpinSlicer/releases/tag/v0.9.0",
    "body": "notes",
    "assets": [
        {"name": "SpinSlicer-Setup-0.9.0.exe", "browser_download_url": "https://x/setup", "size": 3,
         "digest": "sha256:" + "A" * 64},
        {"name": "SpinSlicer.exe", "browser_download_url": "https://x/exe", "size": 2, "digest": None},
        {"name": "SpinSlicer-linux-x64.tar.gz", "browser_download_url": "https://x/tgz", "size": 1},
        {"name": "SpinSlicer-0.9.0-amd64.deb", "browser_download_url": "https://x/deb", "size": 1},
    ],
}


def test_parse_release_reads_assets_and_digests() -> None:
    release = updater.parse_release(RELEASE_JSON)
    assert release.version == "0.9.0" and release.tag == "v0.9.0"
    assert release.assets[0].sha256 == "a" * 64
    assert release.assets[1].sha256 == ""


def test_parse_release_rejects_a_tag_without_version() -> None:
    with pytest.raises(updater.UpdateError):
        updater.parse_release({"tag_name": "latest"})


@pytest.mark.parametrize(("kind", "platform", "name"), [
    ("installer", "win32", "SpinSlicer-Setup-0.9.0.exe"),
    ("portable", "win32", "SpinSlicer.exe"),
    ("portable", "linux", "SpinSlicer-linux-x64.tar.gz"),
    ("manual", "linux", None),
    ("manual", "win32", None),
])
def test_pick_asset(kind: str, platform: str, name) -> None:
    asset = updater.pick_asset(updater.parse_release(RELEASE_JSON), kind, platform)
    assert (asset.name if asset else None) == name


def test_install_kind(tmp_path: Path) -> None:
    exe = tmp_path / "SpinSlicer.exe"
    exe.write_bytes(b"")
    assert updater.install_kind(str(exe), frozen=False, platform="win32") == "manual"
    assert updater.install_kind(str(exe), frozen=True, platform="win32") == "portable"
    (tmp_path / "unins000.exe").write_bytes(b"")
    assert updater.install_kind(str(exe), frozen=True, platform="win32") == "installer"
    assert updater.install_kind("/usr/bin/spinslicer", frozen=True, platform="linux") == "manual"


class _Response(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}


def _serve(monkeypatch, data: bytes) -> None:
    monkeypatch.setattr(updater, "_open", lambda url, accept="": _Response(data))


def test_download_checks_the_sha256(monkeypatch, tmp_path: Path) -> None:
    data = b"new build"
    _serve(monkeypatch, data)
    good = updater.Asset("SpinSlicer.exe", "https://x", len(data), hashlib.sha256(data).hexdigest())
    seen = []
    path = updater.download(good, tmp_path, lambda done, total: seen.append((done, total)))
    assert path.read_bytes() == data and seen[-1] == (len(data), len(data))

    bad = updater.Asset("Other.exe", "https://x", len(data), "0" * 64)
    with pytest.raises(updater.UpdateError):
        updater.download(bad, tmp_path)
    assert not list(tmp_path.glob("Other.exe*"))          # nothing half-trusted left behind


def test_download_rejects_a_short_file(monkeypatch, tmp_path: Path) -> None:
    _serve(monkeypatch, b"abc")
    with pytest.raises(updater.UpdateError):
        updater.download(updater.Asset("x.exe", "https://x", 10), tmp_path)


def test_download_can_be_cancelled(monkeypatch, tmp_path: Path) -> None:
    _serve(monkeypatch, b"abc")
    with pytest.raises(updater.UpdateError, match="отменена"):
        updater.download(updater.Asset("x.exe", "https://x", 3), tmp_path, cancelled=lambda: True)


def test_swap_binary_keeps_the_old_one_aside(tmp_path: Path) -> None:
    exe = tmp_path / "SpinSlicer.exe"
    exe.write_bytes(b"old")
    new = tmp_path / "download" / "SpinSlicer.exe"
    new.parent.mkdir()
    new.write_bytes(b"new")
    updater.swap_binary(new, str(exe))
    assert exe.read_bytes() == b"new"
    assert (tmp_path / "SpinSlicer.exe.old").read_bytes() == b"old"
    updater.cleanup_previous(str(exe))
    assert not (tmp_path / "SpinSlicer.exe.old").exists()


def test_binary_is_taken_out_of_the_linux_tarball(tmp_path: Path) -> None:
    archive = tmp_path / "SpinSlicer-linux-x64.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("SpinSlicer")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"elf"))
    binary = updater._binary_from_tarball(archive)
    assert binary.read_bytes() == b"elf"


def test_settings_show_the_update_state() -> None:
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from spinslicer import app as spinslicer_app

    app = QApplication.instance() or QApplication([])
    window = spinslicer_app.CALSlicerMainWindow()
    try:
        panel = window.settings_tab
        assert panel.r_update.label.text() == f"SpinSlicer {__version__}"
        actions = []
        panel.actionRequested.connect(actions.append)
        panel.set_update_state("found", "9.9.9")
        assert "9.9.9" in panel.update_link.text()
        panel.update_link.click()
        panel.set_update_state("latest")
        panel.update_link.click()
        assert actions == ["update_install", "update_check"]
    finally:
        window.close()
        app.processEvents()
