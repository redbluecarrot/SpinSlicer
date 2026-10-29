"""Hardware detection and the three compute modes: auto, economy, maximum.

The modes change how the same numbers are computed, never which numbers: grid
size, frame count and iteration count are left to the user. Economy keeps memory
and thread use small; maximum uses every core and, when a suitable graphics card
and Taichi are present, runs the heaviest products on the GPU.
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

MODES = ("auto", "economy", "maximum")

# Auto never picks maximum below these limits.
AUTO_MIN_CORES = 6
AUTO_MIN_FREE_RAM = 6 * 1024**3
ECONOMY_BELOW_FREE_RAM = 3 * 1024**3
# A discrete NVIDIA card needs room for both matrices plus working blocks.
MIN_GPU_MEMORY_BYTES = 3 * 1024**3

ECONOMY_WORKSPACE_BYTES = 64 * 1024**2
MAXIMUM_WORKSPACE_BYTES = 1024**3


@dataclass(frozen=True)
class Hardware:
    cores: int
    free_ram: Optional[int]
    gpu_name: str
    gpu_memory: Optional[int]
    taichi_installed: bool


@dataclass(frozen=True)
class ComputeProfile:
    requested: str          # what the user chose: auto / economy / maximum
    mode: str               # what runs: economy / maximum
    device: str             # "cpu" or "gpu"
    gpu_name: str
    threads: int            # CPU threads for sparse products
    workspace_cap: int      # bytes of dense working blocks
    lean_memory: bool       # skip the stored transpose of the ray matrix
    io_threads: int         # parallel frame resize and PNG writing
    slicing_threads: int    # parallel layer rasterisation
    save_bare: bool         # also compute the uncorrected projection
    disk_sinograms: bool    # keep optimised blocks in scratch files, not RAM
    reason: str

    def describe(self) -> str:
        device = f"GPU {self.gpu_name}" if self.device == "gpu" else f"CPU, потоков: {self.threads}"
        return f"Режим «{self.mode}» ({device}): {self.reason}"

    def as_manifest(self) -> dict:
        return {
            "requested": self.requested,
            "mode": self.mode,
            "device": self.device,
            "gpu": self.gpu_name,
            "threads": self.threads,
            "lean_memory": self.lean_memory,
            "disk_sinograms": self.disk_sinograms,
        }


def free_memory_bytes() -> Optional[int]:
    """Free physical memory, or None when the platform does not report it."""
    try:
        if sys.platform == "win32":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_ext", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return int(status.avail_phys)
            return None
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, AttributeError):
        return None


def _nvidia_gpu() -> tuple[str, Optional[int]]:
    """Name and memory of the first NVIDIA card, without touching any GPU API."""
    flags = 0x08000000 if sys.platform == "win32" else 0  # no console flash in the GUI
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, creationflags=flags, check=False,
        )
        line = result.stdout.strip().splitlines()[0]
        name, memory = (part.strip() for part in line.rsplit(",", 1))
        return name, int(float(memory)) * 1024**2
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return "", None


_hardware_cache: Optional[Hardware] = None


def detect_hardware(refresh: bool = False) -> Hardware:
    """Cheap hardware survey; it never initialises Taichi or a GPU API."""
    global _hardware_cache
    if _hardware_cache is None or refresh:
        name, memory = _nvidia_gpu()
        _hardware_cache = Hardware(
            cores=os.cpu_count() or 1,
            free_ram=free_memory_bytes(),
            gpu_name=name,
            gpu_memory=memory,
            taichi_installed=importlib.util.find_spec("taichi") is not None,
        )
    else:  # free RAM changes between runs
        _hardware_cache = Hardware(
            _hardware_cache.cores, free_memory_bytes(), _hardware_cache.gpu_name,
            _hardware_cache.gpu_memory, _hardware_cache.taichi_installed,
        )
    return _hardware_cache


_taichi_state: dict = {"tried": False, "arch": None, "generation": 0}


def init_taichi_gpu() -> Optional[str]:
    """Start Taichi once on a GPU backend; None when only the CPU is available.

    ``ti.init`` is global and resets all fields, so it runs at most once.
    """
    if _taichi_state["tried"]:
        return _taichi_state["arch"]
    _taichi_state["tried"] = True
    try:
        import taichi as ti

        ti.init(arch=ti.gpu, log_level=ti.ERROR, offline_cache=True,
                default_fp=ti.f32, default_ip=ti.i32)
        arch = ti.lang.impl.current_cfg().arch
        if arch != ti.cpu:
            _taichi_state["arch"] = str(arch).split(".")[-1]
    except Exception:  # missing wheel, no driver, unsupported device
        _taichi_state["arch"] = None
    return _taichi_state["arch"]


def taichi_generation() -> int:
    """Counts Taichi start-ups; kernels compiled for an older one are stale."""
    return int(_taichi_state["generation"])


def shutdown_taichi() -> None:
    """Release the GPU in the thread that started it.

    A CUDA context belongs to the thread that created it, so leaving Taichi
    running would crash the process at exit. A later job starts it again.
    """
    if not _taichi_state["tried"]:
        return
    if _taichi_state["arch"]:
        try:
            import gc

            import taichi as ti

            gc.collect()  # finalise leftover device arrays here, not in a random thread

            ti.reset()
        except Exception:
            pass
    _taichi_state["tried"] = False
    _taichi_state["arch"] = None
    _taichi_state["generation"] += 1


def resolve_profile(mode: str = "auto", hardware: Optional[Hardware] = None,
                    use_gpu: bool = True) -> ComputeProfile:
    """Turn the user's choice into concrete settings for this machine."""
    if mode not in MODES:
        raise ValueError(f"compute mode must be one of {', '.join(MODES)}.")
    hw = hardware or detect_hardware()
    free = hw.free_ram

    def economy(reason: str) -> ComputeProfile:
        return ComputeProfile(
            requested=mode, mode="economy", device="cpu", gpu_name="",
            threads=1, workspace_cap=ECONOMY_WORKSPACE_BYTES, lean_memory=True,
            io_threads=1, slicing_threads=1, save_bare=False, disk_sinograms=True,
            reason=reason,
        )

    def maximum(reason: str, gpu: str) -> ComputeProfile:
        return ComputeProfile(
            requested=mode, mode="maximum", device="gpu" if gpu else "cpu", gpu_name=gpu,
            threads=min(hw.cores, 16), workspace_cap=MAXIMUM_WORKSPACE_BYTES, lean_memory=False,
            io_threads=min(hw.cores, 8), slicing_threads=min(hw.cores, 8),
            save_bare=True, disk_sinograms=False, reason=reason,
        )

    def gpu_usable() -> str:
        if not (use_gpu and hw.taichi_installed):
            return ""
        if hw.gpu_memory is not None and hw.gpu_memory < MIN_GPU_MEMORY_BYTES:
            return ""
        arch = init_taichi_gpu()
        return (hw.gpu_name or "GPU") + f" ({arch})" if arch else ""

    if mode == "economy":
        return economy("минимум памяти и один поток, результат тот же")
    if mode == "maximum":
        gpu = gpu_usable()
        return maximum("вычисления на видеокарте" if gpu else "все ядра процессора", gpu)

    # auto
    if hw.cores <= 2 or (free is not None and free < ECONOMY_BELOW_FREE_RAM):
        return economy("мало ядер или свободной памяти")
    gpu = gpu_usable()
    if gpu:
        return maximum("найдена подходящая видеокарта", gpu)
    if hw.cores >= AUTO_MIN_CORES and (free is None or free >= AUTO_MIN_FREE_RAM):
        return maximum("достаточно ядер и памяти, видеокарта не используется", "")
    return economy("видеокарты нет, ядер или памяти немного")


def describe_hardware(hw: Optional[Hardware] = None) -> str:
    """One-line summary for the settings panel (does not initialise Taichi)."""
    hw = hw or detect_hardware()
    ram = f"{hw.free_ram / 1024**3:.1f} ГиБ свободно" if hw.free_ram else "память неизвестна"
    gpu = hw.gpu_name or "видеокарта не определена"
    taichi = "Taichi установлен" if hw.taichi_installed else "Taichi не установлен"
    return f"{hw.cores} ядер, {ram}; {gpu}; {taichi}"
