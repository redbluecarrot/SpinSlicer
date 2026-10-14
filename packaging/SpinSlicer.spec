# -*- mode: python ; coding: utf-8 -*-
"""Portable PyInstaller build for SpinSlicer.

Windows API-set DLLs are provided by the operating system.  PyInstaller can
mistakenly collect copies from unrelated entries on PATH (for example a JDK
or image-processing runtime); shipping those copies can break Qt DLL loading
on another Windows installation.
"""

from pathlib import Path
import sys
from importlib.util import find_spec

from PyInstaller.utils.hooks import collect_all


PROJECT_ROOT = Path(SPECPATH).resolve().parent

datas = [(str(PROJECT_ROOT / "assets"), "assets")]
binaries: list[tuple[str, str]] = []
cpp_backend = PROJECT_ROOT / "cpp_backend"
if cpp_backend.is_dir():
    native_suffixes = {".dll", ".dylib", ".so"}
    for path in sorted(cpp_backend.rglob("*")):
        if not path.is_file():
            continue
        destination = str(path.parent.relative_to(cpp_backend))
        destination = "cpp_backend" if destination == "." else f"cpp_backend/{destination}"
        if path.suffix.casefold() in native_suffixes:
            binaries.append((str(path), destination))
        else:
            datas.append((str(path), destination))
qt_bin: Path | None = None
if sys.platform.startswith("win"):
    pyqt_spec = find_spec("PyQt6")
    if pyqt_spec is not None and pyqt_spec.origin:
        candidate = Path(pyqt_spec.origin).resolve().parent / "Qt6" / "bin"
        if candidate.is_dir():
            qt_bin = candidate
            binaries.extend(
                (str(path), ".")
                for path in sorted(candidate.glob("*.dll"))
                if path.name.casefold().startswith(("msvcp140", "vcruntime140"))
            )

hiddenimports = ["vtkmodules.qt.QVTKRenderWindowInteractor"]
for package in ("qdarktheme", "pyvista", "pyvistaqt", "vtkmodules"):
    collected_datas, collected_binaries, collected_hiddenimports = collect_all(package)
    datas += collected_datas
    binaries += collected_binaries
    hiddenimports += collected_hiddenimports


a = Analysis(
    [str(PROJECT_ROOT / "SpinSlicer.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(PROJECT_ROOT / "runtime_hook_qt_path.py")],
    # PyVista imports its lightweight runtime decorator from
    # ``pyvista.typing.mypy_plugin``.  When mypy is installed in the build
    # environment, that plugin conditionally imports mypy's compiled checker
    # modules, which are not application dependencies and can fail in a frozen
    # build due to an incomplete ``__mypyc`` module graph.
    excludes=["mypy", "mypy_extensions"],
    noarchive=False,
    optimize=0,
)

# API-set DLLs (api-ms-win-* and ext-ms-win-*) belong to Windows.  Do not
# let unrelated PATH entries override the target machine's system API sets.
# The build host can also expose UCRT and ICU copies from unrelated native
# runtimes.  Qt expects the system UCRT/ICU exports, so keep those out too.
a.binaries = [
    entry
    for entry in a.binaries
    if Path(entry[0]).name.casefold() != "ucrtbase.dll"
    and not Path(entry[0]).name.casefold().startswith(("api-ms-win-", "ext-ms-win-", "icu"))
    and not (
        Path(entry[0]).parent == Path(".")
        and Path(entry[0]).name.casefold().startswith(("msvcp140", "vcruntime140"))
    )
]
if qt_bin is not None:
    a.binaries.extend(
        (path.name, str(path), "BINARY")
        for path in sorted(qt_bin.glob("*.dll"))
        if path.name.casefold().startswith(("msvcp140", "vcruntime140"))
    )

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="SpinSlicer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
