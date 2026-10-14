"""Optional VAMToolbox projection backend.

SpinSlicer keeps its small Radon implementation as a deterministic fallback,
but exposes the same target → projection → export separation used by Tomo and
VAMToolbox.  VAMToolbox is intentionally not a hard runtime dependency: its
Windows/conda stack and licensing terms make it unsuitable for the default
pip requirements of this research prototype.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_VAM_ENV_NAME = "spinslicer-vam"
VAM_INSTALL_CHANNELS = ("vamtoolbox", "conda-forge", "astra-toolbox")


class VAMToolboxUnavailable(RuntimeError):
    """Raised when the optional VAMToolbox backend is requested but missing."""


def is_vamtoolbox_available() -> bool:
    """Return whether the optional package can be imported in this environment."""

    try:
        import vamtoolbox  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def find_conda_executable() -> str | None:
    """Find a usable Conda executable without requiring it on ``PATH``.

    The official VAMToolbox distribution is published through Conda and its
    Windows installer is often not added to the system PATH.  Checking the
    standard per-user locations keeps the optional integration discoverable
    while leaving the normal SpinSlicer environment untouched.
    """

    candidates = [os.environ.get("CONDA_EXE", "")]
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    user_profile = os.environ.get("USERPROFILE", "")
    for root in (local_app_data, user_profile):
        if not root:
            continue
        candidates.extend(
            [
                str(Path(root) / "miniconda3" / "Scripts" / "conda.exe"),
                str(Path(root) / "anaconda3" / "Scripts" / "conda.exe"),
                str(Path(root) / "mambaforge" / "Scripts" / "conda.exe"),
            ]
        )

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate

    return shutil.which("conda.exe") or shutil.which("conda")


def build_vam_install_command(
    conda_executable: str,
    environment_name: str = DEFAULT_VAM_ENV_NAME,
) -> list[str]:
    """Return the official Windows Conda installation command as argv."""

    if not conda_executable:
        raise ValueError("A Conda executable is required to install VAMToolbox.")
    if not environment_name.strip():
        raise ValueError("The VAMToolbox environment name cannot be empty.")

    return [
        conda_executable,
        "create",
        "--yes",
        "--name",
        environment_name,
        "vamtoolbox",
        "-c",
        VAM_INSTALL_CHANNELS[0],
        "-c",
        VAM_INSTALL_CHANNELS[1],
        "-c",
        VAM_INSTALL_CHANNELS[2],
    ]


def build_vam_probe_command(
    conda_executable: str,
    environment_name: str = DEFAULT_VAM_ENV_NAME,
) -> list[str]:
    """Return a command that verifies the isolated VAMToolbox environment."""

    if not conda_executable:
        raise ValueError("A Conda executable is required to probe VAMToolbox.")
    if not environment_name.strip():
        raise ValueError("The VAMToolbox environment name cannot be empty.")

    return [
        conda_executable,
        "run",
        "--no-capture-output",
        "--name",
        environment_name,
        "python",
        "-c",
        "import vamtoolbox; print(vamtoolbox.__file__)",
    ]


def _as_array(value: Any) -> np.ndarray:
    array = getattr(value, "array", value)
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    result = np.asarray(array, dtype=np.float32)
    if result.ndim != 3:
        raise ValueError(
            f"VAMToolbox returned a {result.ndim}D sinogram; expected a 3D array."
        )
    return result


def _optimize_with_module(
    vam: Any,
    target: np.ndarray,
    angles: np.ndarray,
    iterations: int,
) -> np.ndarray:
    """Run CAL using an already imported VAMToolbox module."""

    target_geo = vam.geometry.TargetGeometry(
        target=target,
        clip_to_circle=False,
    )
    proj_geo = vam.geometry.ProjectionGeometry(
        angles=angles,
        ray_type="parallel",
        CUDA=False,
    )
    # VAMToolbox's CPU 3D projector uses the sparse operator.  The flag is
    # intentionally set explicitly: older releases defaulted differently and
    # the current high-level VAM pipeline does the same for CPU jobs.
    proj_geo.sparse = True
    options = vam.optimize.Options(
        method="CAL",
        n_iter=int(iterations),
        d_h=0.85,
        d_l=0.60,
        filter="hamming",
        units="normalized",
    )
    optimized = vam.optimize.optimize(
        target_geo=target_geo,
        proj_geo=proj_geo,
        options=options,
    )
    sinogram_obj = optimized[0] if isinstance(optimized, (tuple, list)) else optimized
    return _as_array(sinogram_obj)


_EXTERNAL_VAM_SCRIPT = r'''
import sys

import numpy as np
import vamtoolbox as vam


def as_array(value):
    array = getattr(value, "array", value)
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float32)


input_path, output_path, iterations_text = sys.argv[1:]
iterations = int(iterations_text)
with np.load(input_path, allow_pickle=False) as payload:
    target = np.asarray(payload["target"], dtype=np.float32)
    angles = np.asarray(payload["angles"], dtype=np.float64)

target_geo = vam.geometry.TargetGeometry(target=target, clip_to_circle=False)
proj_geo = vam.geometry.ProjectionGeometry(
    angles=angles,
    ray_type="parallel",
    CUDA=False,
)
proj_geo.sparse = True
options = vam.optimize.Options(
    method="CAL",
    n_iter=iterations,
    d_h=0.85,
    d_l=0.60,
    filter="hamming",
    units="normalized",
)
optimized = vam.optimize.optimize(
    target_geo=target_geo,
    proj_geo=proj_geo,
    options=options,
)
sinogram = optimized[0] if isinstance(optimized, (tuple, list)) else optimized
np.save(output_path, as_array(sinogram), allow_pickle=False)
'''


def _subprocess_command(command: list[str]) -> list[str]:
    """Make a Conda argv runnable when Windows resolved a ``.bat`` shim."""

    if command and command[0].lower().endswith((".bat", ".cmd")):
        return ["cmd.exe", "/d", "/s", "/c", subprocess.list2cmdline(command)]
    return command


def _optimize_with_conda(
    target: np.ndarray,
    angles: np.ndarray,
    iterations: int,
    conda_executable: str,
    environment_name: str,
) -> np.ndarray:
    """Run VAMToolbox in its isolated Conda environment.

    VAMToolbox is not imported into SpinSlicer's process.  This is important
    for the PyInstaller build: its native Astra dependencies remain isolated,
    and the internal Radon backend keeps working when Conda is unavailable.
    """

    resolved_conda = shutil.which(conda_executable) or conda_executable
    if not Path(resolved_conda).is_file() and not shutil.which(resolved_conda):
        raise VAMToolboxUnavailable(
            f"Conda executable was not found: {conda_executable}. "
            "Install Miniconda/Anaconda, then install VAMToolbox from Settings."
        )

    with tempfile.TemporaryDirectory(prefix="spinslicer-vam-") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "input.npz"
        output_path = temp_root / "sinograms.npy"
        np.savez(input_path, target=target, angles=angles)

        command = [
            resolved_conda,
            "run",
            "--no-capture-output",
            "--name",
            environment_name,
            "python",
            "-c",
            _EXTERNAL_VAM_SCRIPT,
            str(input_path),
            str(output_path),
            str(iterations),
        ]
        try:
            completed = subprocess.run(
                _subprocess_command(command),
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise VAMToolboxUnavailable(
                "VAMToolbox could not start Conda. "
                "Install Miniconda/Anaconda or repair the configured executable."
            ) from exc
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            if len(details) > 1200:
                details = details[-1200:]
            suffix = f"\n{details}" if details else ""
            raise VAMToolboxUnavailable(
                "VAMToolbox could not run in the configured Conda environment "
                f"'{environment_name}'. Install or repair it from Settings."
                f"{suffix}"
            )
        if not output_path.is_file():
            raise VAMToolboxUnavailable(
                "VAMToolbox finished without producing a sinogram. "
                "Check the configured Conda environment."
            )
        return np.asarray(np.load(output_path, allow_pickle=False), dtype=np.float32)


def normalize_sinogram_shape(
    sinogram: np.ndarray,
    grid_res: int,
    num_frames: int,
    num_layers: int,
) -> np.ndarray:
    """Convert common VAMToolbox layouts to ``(layers, pixels, angles)``."""

    expected = (grid_res, num_frames, num_layers)
    if sinogram.shape == expected:
        return np.transpose(sinogram, (2, 0, 1)).astype(np.float32, copy=False)

    expected = (num_frames, grid_res, num_layers)
    if sinogram.shape == expected:
        return np.transpose(sinogram, (2, 1, 0)).astype(np.float32, copy=False)

    expected = (num_layers, grid_res, num_frames)
    if sinogram.shape == expected:
        return sinogram.astype(np.float32, copy=False)

    expected = (num_layers, num_frames, grid_res)
    if sinogram.shape == expected:
        return np.transpose(sinogram, (0, 2, 1)).astype(np.float32, copy=False)

    raise ValueError(
        "Unsupported VAMToolbox sinogram shape "
        f"{sinogram.shape}; expected one of the common layouts for "
        f"grid={grid_res}, angles={num_frames}, layers={num_layers}."
    )


def optimize_sinograms(
    target_volume: np.ndarray,
    angles_deg: np.ndarray,
    iterations: int = 8,
    *,
    conda_executable: str | None = None,
    environment_name: str = DEFAULT_VAM_ENV_NAME,
) -> np.ndarray:
    """Run VAMToolbox CAL optimization and return normalized array layout.

    The target volume uses the same convention as SpinSlicer:
    ``(x, y, z)`` with values in ``[0, 1]``.  VAMToolbox currently documents
    the parallel-ray geometry and ``CAL`` optimizer as the portable baseline;
    CUDA is disabled here so the optional path can also run on CPU.
    """

    target = np.asarray(target_volume, dtype=np.float32)
    if target.ndim != 3 or target.shape[0] != target.shape[1]:
        raise ValueError("VAM target must be a square 3D voxel volume.")
    if not np.all(np.isfinite(target)):
        raise ValueError("VAM target must contain only finite voxel values.")
    if not 1 <= int(iterations) <= 100:
        raise ValueError("VAMToolbox optimizer iterations must be between 1 and 100.")

    angles = np.asarray(angles_deg, dtype=np.float64)
    if angles.ndim != 1 or angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("VAM projection angles must be a non-empty finite 1D array.")

    target = np.ascontiguousarray(np.clip(target, 0.0, 1.0))
    angles = np.ascontiguousarray(angles)

    try:
        import vamtoolbox as vam
    except (ImportError, OSError) as exc:
        if conda_executable:
            raw = _optimize_with_conda(
                target,
                angles,
                int(iterations),
                conda_executable,
                environment_name,
            )
        else:
            detail = str(exc).strip()
            suffix = f": {detail}" if detail else ""
            raise VAMToolboxUnavailable(
                "VAMToolbox could not be loaded"
                f"{suffix}. Install VAMToolbox from Settings using Conda, "
                "or select the internal Radon backend."
            ) from exc
    else:
        raw = _optimize_with_module(vam, target, angles, int(iterations))

    normalized = normalize_sinogram_shape(
        raw,
        grid_res=int(target.shape[0]),
        num_frames=int(angles.size),
        num_layers=int(target.shape[2]),
    )
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
