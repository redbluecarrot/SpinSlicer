"""Build the optional GPU add-on shipped by the Windows installer.

The add-on is Taichi plus its small pure-Python dependencies, installed into a plain
folder (``dist/gpu`` by default) that ends up next to the program. numpy is left out on
purpose: the program already contains it, and the add-on is appended to ``sys.path``
(see ``compute.enable_gpu_addon``), so the program's own copy is always the one in use.

    python packaging/build_gpu_addon.py [--out dist/gpu]

The pins come from ``requirements-gpu.txt``; every package is listed there, so nothing
is resolved and no numpy is pulled in.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Present in the wheels but useless at run time: numpy (the program has it), the
# stand-alone SPIR-V tools and headers.
PRUNE_PREFIXES = ("numpy", "bin", "include", "share", "SPIRV-Tools")


def prune(folder: Path) -> list[str]:
    """Delete what the add-on does not need; returns the removed names."""
    removed = []
    for entry in sorted(folder.iterdir()):
        name = entry.name
        if name.startswith(PRUNE_PREFIXES) or name == "__pycache__":
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
            removed.append(name)
    for cache in folder.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    return removed


def extension_tag() -> str:
    """Tag in the name of a compiled module of this interpreter, e.g. ``cp312``."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def check_abi(folder: Path) -> None:
    """The add-on holds a compiled Taichi core; it only loads in the Python it was built for.

    The program is built with one Python (3.12 in the release workflow), so the add-on has to
    be built with that same interpreter: a core for another version fails to import, and the
    program would then run without the GPU. This only compares the core with the interpreter
    running the script; ``--expect-tag`` states which Python the program uses, and the release
    workflow's check of the packaged program (it imports Taichi and runs a kernel) is the final
    guard.
    """
    cores = sorted(path.name for path in (folder / "taichi" / "_lib" / "core").glob("taichi_python*"))
    tag = extension_tag()
    if not any(f".{tag}-" in name for name in cores):
        raise SystemExit(
            f"The Taichi core in {folder} ({', '.join(cores) or 'none'}) is not built for this "
            f"Python ({tag}). Run this script with the Python that builds the program."
        )


def build(out: Path, requirements: Path = ROOT / "requirements-gpu.txt") -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(out), "-r", str(requirements)],
        check=True,
    )
    prune(out)
    if not (out / "taichi").is_dir():
        raise SystemExit(f"Taichi was not installed into {out}")
    check_abi(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "gpu")
    parser.add_argument(
        "--expect-tag", help="Python tag the program is built with (for example cp312); refuse any other"
    )
    arguments = parser.parse_args()
    if arguments.expect_tag and arguments.expect_tag != extension_tag():
        raise SystemExit(
            f"This script runs on Python {extension_tag()}, but the program uses {arguments.expect_tag}."
        )
    folder = build(arguments.out)
    size = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
    print(f"GPU add-on: {folder} ({size / 1024**2:.0f} MB, for Python {extension_tag()})")


if __name__ == "__main__":
    main()
