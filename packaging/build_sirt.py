"""Build and verify the platform-native physical projection backend."""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "cpp_backend" / "sirt_engine.cpp"


def build() -> Path:
    if sys.platform == "win32":
        compiler = shutil.which("cl")
        if not compiler:
            raise SystemExit("MSVC cl.exe is required to build the Windows SIRT backend.")
        output = ROOT / "cpp_backend" / "sirtsolver.dll"
        command = [
            compiler,
            "/nologo",
            "/std:c++17",
            "/O2",
            "/EHsc",
            "/MT",
            "/LD",
            str(SOURCE),
            "/link",
            f"/OUT:{output}",
        ]
    elif sys.platform == "darwin":
        compiler = shutil.which("clang++")
        if not compiler:
            raise SystemExit("clang++ is required to build the macOS SIRT backend.")
        output = ROOT / "cpp_backend" / "sirtsolver.dylib"
        command = [compiler, "-std=c++17", "-O3", "-dynamiclib", str(SOURCE), "-o", str(output)]
    else:
        compiler = shutil.which("g++")
        if not compiler:
            raise SystemExit("g++ is required to build the Linux SIRT backend.")
        output = ROOT / "cpp_backend" / "sirtsolver.so"
        command = [
            compiler,
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-shared",
            "-fopenmp",
            str(SOURCE),
            "-o",
            str(output),
        ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spinslicer-sirt-") as temp_dir:
        subprocess.run(command, cwd=temp_dir, check=True)
    library = ctypes.CDLL(os.fspath(output))
    for symbol in ("forward_project", "sirt_loop"):
        getattr(library, symbol)
    print(f"Built and verified {output}")
    return output


if __name__ == "__main__":
    build()
