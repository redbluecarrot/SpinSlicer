import pytest

import compute


@pytest.fixture(autouse=True)
def _no_gpu_probe_without_a_card(monkeypatch):
    """Do not start Taichi on a GPU backend when the machine has no graphics card.

    On a headless machine (CI) the probe for Vulkan or CUDA can crash the whole process, and the
    GPU tests skip themselves anyway when Taichi reports no device.
    """
    if not compute.detect_hardware().gpu_name:
        monkeypatch.setattr(compute, "init_taichi_gpu", lambda: None)
