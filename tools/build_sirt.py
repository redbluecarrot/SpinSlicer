"""Build script for SIRT C++ backend library."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def build() -> bool:
    root = Path(__file__).resolve().parent.parent
    cpp_dir = root / "cpp_backend"
    src = cpp_dir / "sirt_engine.cpp"
    if not src.is_file():
        print(f"Source file not found: {src}", file=sys.stderr)
        return False

    if sys.platform.startswith("win"):
        dll_out = cpp_dir / "sirtsolver.dll"
        # Try MSVC first (cl.exe)
        cl_path = shutil.which("cl")
        if cl_path:
            cmd = [
                "cl",
                "/nologo",
                "/std:c++17",
                "/O2",
                "/openmp:experimental",
                "/EHsc",
                "/LD",
                str(src),
                "/link",
                f"/OUT:{dll_out}",
            ]
            print(f"Compiling with MSVC: {' '.join(cmd)}")
            res = subprocess.run(cmd, cwd=str(cpp_dir))
            if res.returncode == 0 and dll_out.is_file():
                print(f"Successfully built {dll_out}")
                return True
        # Try g++ (MinGW)
        gxx_path = shutil.which("g++")
        if gxx_path:
            cmd = [
                "g++",
                "-std=c++17",
                "-O3",
                "-shared",
                "-fopenmp",
                str(src),
                "-o",
                str(dll_out),
            ]
            print(f"Compiling with g++: {' '.join(cmd)}")
            res = subprocess.run(cmd, cwd=str(cpp_dir))
            if res.returncode == 0 and dll_out.is_file():
                print(f"Successfully built {dll_out}")
                return True
        print("Neither cl.exe nor g++ found for Windows build.", file=sys.stderr)
        return False
    else:
        so_out = cpp_dir / "sirtsolver.so"
        cxx = os.environ.get("CXX", "g++")
        cmd = [
            cxx,
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-shared",
            "-fopenmp",
            str(src),
            "-o",
            str(so_out),
        ]
        print(f"Compiling with {cxx}: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=str(cpp_dir))
        if res.returncode == 0 and so_out.is_file():
            lib_sym = cpp_dir / "libsirt.so"
            try:
                if lib_sym.exists() or lib_sym.is_symlink():
                    lib_sym.unlink()
                shutil.copy2(so_out, lib_sym)
            except Exception:
                pass
            print(f"Successfully built {so_out}")
            return True
        print(f"Failed to build with {cxx}", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = build()
    sys.exit(0 if success else 1)
