from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vam_backend import (
    DEFAULT_VAM_ENV_NAME,
    VAMToolboxUnavailable,
    _subprocess_command,
    build_vam_install_command,
    build_vam_probe_command,
    optimize_sinograms,
)


class _FakeTargetGeometry:
    def __init__(self, *, target: np.ndarray, clip_to_circle: bool):
        self.array = target
        self.clip_to_circle = clip_to_circle


class _FakeProjectionGeometry:
    instances: list[_FakeProjectionGeometry] = []

    def __init__(self, *, angles: np.ndarray, ray_type: str, CUDA: bool):
        self.angles = angles
        self.ray_type = ray_type
        self.CUDA = CUDA
        self.sparse = None
        self.instances.append(self)


class _FakeOptions:
    instances: list[_FakeOptions] = []

    def __init__(self, **kwargs: object):
        self.values = kwargs
        self.instances.append(self)


def test_vam_backend_configures_cpu_sparse_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = SimpleNamespace(
        TargetGeometry=_FakeTargetGeometry,
        ProjectionGeometry=_FakeProjectionGeometry,
    )
    optimize = SimpleNamespace(
        Options=_FakeOptions,
        optimize=lambda **kwargs: (
            SimpleNamespace(
                array=np.arange(8 * 12 * 5, dtype=np.float32).reshape(8, 12, 5),
            ),
            None,
            None,
        ),
    )
    fake_vam = SimpleNamespace(geometry=geometry, optimize=optimize)
    monkeypatch.setitem(sys.modules, "vamtoolbox", fake_vam)
    _FakeProjectionGeometry.instances.clear()
    _FakeOptions.instances.clear()

    target = np.ones((8, 8, 5), dtype=np.float32)
    angles = np.linspace(0.0, 360.0, 12, endpoint=False)
    result = optimize_sinograms(target, angles, iterations=3)

    assert result.shape == (5, 8, 12)
    assert _FakeProjectionGeometry.instances[0].sparse is True
    assert _FakeProjectionGeometry.instances[0].ray_type == "parallel"
    assert _FakeProjectionGeometry.instances[0].CUDA is False
    assert _FakeOptions.instances[0].values == {
        "method": "CAL",
        "n_iter": 3,
        "d_h": 0.85,
        "d_l": 0.60,
        "filter": "hamming",
        "units": "normalized",
    }


def test_vam_backend_rejects_invalid_angles_before_calling_toolbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "vamtoolbox", SimpleNamespace())

    with pytest.raises(ValueError, match="non-empty finite 1D"):
        optimize_sinograms(
            np.ones((4, 4, 2), dtype=np.float32),
            np.array([[0.0, 90.0]]),
        )


def test_vam_backend_wraps_native_import_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def failing_import(name: str, *args: object, **kwargs: object):
        if name == "vamtoolbox":
            raise OSError("missing native dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", failing_import)

    with pytest.raises(VAMToolboxUnavailable, match="could not be loaded"):
        optimize_sinograms(np.ones((4, 4, 2), dtype=np.float32), np.array([0.0]))


def test_vam_conda_commands_use_the_official_channels() -> None:
    install = build_vam_install_command("C:/Miniconda/Scripts/conda.bat")
    probe = build_vam_probe_command("C:/Miniconda/Scripts/conda.bat")

    assert install[:5] == [
        "C:/Miniconda/Scripts/conda.bat",
        "create",
        "--yes",
        "--name",
        DEFAULT_VAM_ENV_NAME,
    ]
    assert "python=3.12" not in install
    assert install[-6:] == [
        "vamtoolbox",
        "-c",
        "vamtoolbox",
        "-c",
        "conda-forge",
        "-c",
        "astra-toolbox",
    ][-6:]
    assert probe[1:5] == ["run", "--no-capture-output", "--name", DEFAULT_VAM_ENV_NAME]


def test_vam_windows_batch_command_is_wrapped_for_subprocess() -> None:
    command = _subprocess_command(["C:/Miniconda/Scripts/conda.bat", "--version"])

    assert command[:3] == ["cmd.exe", "/d", "/s"]
    assert command[-1].endswith("conda.bat --version")
