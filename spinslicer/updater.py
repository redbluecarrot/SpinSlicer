"""Updates from GitHub Releases: find a newer version, download it, put it in place.

How the new version gets installed depends on how this copy runs:

* ``installer`` — Windows, installed with SpinSlicer-Setup: the new Setup runs silently,
  closes this window and starts the new version (``/RELAUNCH``, see SpinSlicer.iss).
* ``portable`` — a single SpinSlicer.exe (Windows) or the SpinSlicer binary from the
  Linux tarball in a folder we can write to: the new binary takes the old one's place
  (a running executable can be renamed on both systems) and starts.
* ``manual`` — .deb/.rpm packages, a folder we cannot write to, a source checkout:
  the release page is opened, the user updates the usual way.

Every download is checked against the SHA-256 GitHub publishes for the asset.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from spinslicer import __version__

REPO = "redbluecarrot/SpinSlicer"
LATEST_API = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
TIMEOUT_S = 15
CHUNK = 1 << 20


class UpdateError(Exception):
    """Anything that stops an update.

    ``message`` is a fixed source string (translated with i18n.tr), ``detail`` the
    technical cause for the log.
    """

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(f"{message} {detail}".strip())
        self.message = message
        self.detail = detail


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int
    sha256: str = ""


@dataclass(frozen=True)
class Release:
    version: str                    # "0.9.0"
    tag: str                        # "v0.9.0"
    page: str
    notes: str = ""
    assets: tuple[Asset, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- versions
def parse_version(text: str) -> Optional[tuple[int, int, int]]:
    """``"v0.9.0"`` / ``"0.9"`` → ``(0, 9, 0)``; None if there is no version in it."""
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", str(text or ""))
    if not match:
        return None
    return int(match[1]), int(match[2]), int(match[3] or 0)


def is_newer(candidate: str, current: str = __version__) -> bool:
    new, old = parse_version(candidate), parse_version(current)
    return new is not None and old is not None and new > old


# --------------------------------------------------------------------------- network
def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:          # certifi missing or its bundle not packed: the system store
        return ssl.create_default_context()


def _open(url: str, accept: str = "application/octet-stream"):
    request = urllib.request.Request(url, headers={
        "User-Agent": f"SpinSlicer/{__version__} (update check)",
        "Accept": accept,
    })
    return urllib.request.urlopen(request, timeout=TIMEOUT_S, context=_ssl_context())


def parse_release(data: dict) -> Release:
    tag = str(data.get("tag_name") or "")
    version = parse_version(tag)
    if version is None:
        raise UpdateError("Не удалось прочитать версию релиза.", repr(tag))
    assets = []
    for item in data.get("assets") or []:
        digest = str(item.get("digest") or "")
        assets.append(Asset(name=str(item.get("name") or ""),
                            url=str(item.get("browser_download_url") or ""),
                            size=int(item.get("size") or 0),
                            sha256=digest.split(":", 1)[1].lower() if digest.startswith("sha256:") else ""))
    return Release(version=".".join(map(str, version)), tag=tag,
                   page=str(data.get("html_url") or RELEASES_PAGE),
                   notes=str(data.get("body") or ""), assets=tuple(assets))


def fetch_latest() -> Release:
    """The latest published (non-draft, non-prerelease) release."""
    try:
        with _open(LATEST_API, "application/vnd.github+json") as response:
            data = json.loads(response.read().decode("utf-8"))
    except UpdateError:
        raise
    except Exception as exc:
        raise UpdateError("Не удалось связаться с GitHub.", str(exc)) from exc
    return parse_release(data)


# --------------------------------------------------------------------------- this copy
def install_kind(executable: Optional[str] = None, frozen: Optional[bool] = None,
                 platform: str = sys.platform) -> str:
    """installer | portable | manual — see the module docstring."""
    frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    if not frozen:
        return "manual"
    exe = Path(executable or sys.executable).resolve()
    if platform.startswith("win"):
        if any(exe.parent.glob("unins*.exe")):
            return "installer"
        return "portable" if os.access(exe.parent, os.W_OK) else "manual"
    if platform.startswith("linux"):
        packaged = str(exe).startswith(("/usr/", "/opt/"))
        return "portable" if not packaged and os.access(exe.parent, os.W_OK) else "manual"
    return "manual"


def pick_asset(release: Release, kind: str, platform: str = sys.platform) -> Optional[Asset]:
    """The file this copy needs from the release, or None (→ the release page)."""
    if kind == "installer":
        wanted = re.compile(r"SpinSlicer-Setup-.*\.exe$", re.I)
    elif kind == "portable" and platform.startswith("win"):
        wanted = re.compile(r"SpinSlicer\.exe$", re.I)
    elif kind == "portable" and platform.startswith("linux"):
        wanted = re.compile(r"SpinSlicer-linux-x64\.tar\.gz$", re.I)
    else:
        return None
    return next((a for a in release.assets if wanted.fullmatch(a.name) and a.url), None)


# --------------------------------------------------------------------------- download
def download(asset: Asset, folder: Path,
             progress: Optional[Callable[[int, int], None]] = None,
             cancelled: Optional[Callable[[], bool]] = None) -> Path:
    """Fetch the asset into folder, check its size and SHA-256; returns the file."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / Path(asset.name).name
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    done = 0
    try:
        with _open(asset.url) as response, open(partial, "wb") as out:
            total = int(response.headers.get("Content-Length") or asset.size or 0)
            while True:
                if cancelled and cancelled():
                    raise UpdateError("Загрузка отменена.")
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    except UpdateError:
        partial.unlink(missing_ok=True)
        raise
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise UpdateError("Не удалось скачать обновление.", f"{asset.name}: {exc}") from exc
    if asset.size and done != asset.size:
        partial.unlink(missing_ok=True)
        raise UpdateError("Обновление скачалось не полностью.", f"{asset.name}: {done} / {asset.size}")
    if asset.sha256 and digest.hexdigest() != asset.sha256:
        partial.unlink(missing_ok=True)
        raise UpdateError("Контрольная сумма не совпала — обновление не будет установлено.", asset.name)
    os.replace(partial, target)
    return target


def _binary_from_tarball(archive: Path) -> Path:
    """The SpinSlicer binary out of the Linux tarball, next to it."""
    with tarfile.open(archive, "r:gz") as tar:
        member = next((m for m in tar.getmembers()
                       if m.isfile() and Path(m.name).name == "SpinSlicer"), None)
        if member is None:
            raise UpdateError("В архиве нет файла SpinSlicer.")
        source = tar.extractfile(member)
        if source is None:
            raise UpdateError("В архиве нет файла SpinSlicer.")
        target = archive.with_name("SpinSlicer.new")
        with source, open(target, "wb") as out:
            shutil.copyfileobj(source, out)
    target.chmod(0o755)
    return target


# --------------------------------------------------------------------------- install
def cleanup_previous(executable: Optional[str] = None) -> None:
    """After a portable update the old binary stays as <exe>.old until the next start."""
    if executable is None:
        if not getattr(sys, "frozen", False):
            return
        executable = sys.executable
    exe = Path(executable)
    try:
        exe.with_name(exe.name + ".old").unlink(missing_ok=True)
    except OSError:
        pass                    # still running (the first seconds after a swap): next time


def swap_binary(new_binary: Path, executable: Optional[str] = None) -> Path:
    """Put new_binary where the running executable is; the old one becomes <exe>.old."""
    exe = Path(executable or sys.executable).resolve()
    staged = exe.with_name(exe.name + ".new")
    old = exe.with_name(exe.name + ".old")
    shutil.move(str(new_binary), staged)       # same folder → the renames below are atomic
    if os.name != "nt":
        staged.chmod(0o755)
    old.unlink(missing_ok=True)
    os.replace(exe, old)
    try:
        os.replace(staged, exe)
    except OSError:
        os.replace(old, exe)                   # put the old one back: never leave no binary
        raise
    return exe


def launch(path: Path, args: tuple[str, ...] = ()) -> None:
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    env = dict(os.environ)
    # A PyInstaller onefile child must unpack itself, not reuse this process's folder.
    env.pop("_MEIPASS2", None)
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    subprocess.Popen([str(path), *args], close_fds=True, creationflags=flags, env=env,
                     cwd=str(path.parent))


def install(downloaded: Path, kind: str) -> None:
    """Start the new version; the caller then closes this window."""
    if kind == "installer":
        launch(downloaded, ("/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                            "/CLOSEAPPLICATIONS", "/RELAUNCH"))
        return
    if kind == "portable":
        binary = _binary_from_tarball(downloaded) if downloaded.name.endswith(".tar.gz") else downloaded
        exe = swap_binary(binary)
        launch(exe)
        return
    raise UpdateError("Эту копию SpinSlicer нужно обновить вручную.")


def download_folder() -> Path:
    return Path(tempfile.gettempdir()) / "SpinSlicer-update"
