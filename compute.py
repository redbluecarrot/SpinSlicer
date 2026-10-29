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
from pathlib import Path
from typing import Optional

from i18n import tr

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
    save_bare: bool         # also compute the direct projection
    disk_sinograms: bool    # keep optimised blocks in scratch files, not RAM
    reason: str

    def describe(self) -> str:
        if self.device == "gpu":
            device = tr("GPU {name}").format(name=self.gpu_name)
        else:
            device = tr("CPU, потоков: {threads}").format(threads=self.threads)
        mode = tr("экономный") if self.mode == "economy" else tr("максимальный")
        return tr("Режим «{mode}» ({device}): {reason}").format(mode=mode, device=device, reason=self.reason)

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


GPU_ADDON_ENV = "SPINSLICER_GPU_DIR"


def gpu_addon_directory() -> Optional[Path]:
    """Folder of the optional GPU add-on (Taichi and its small dependencies).

    The Windows installer puts it in ``gpu`` next to the program when the "GPU
    acceleration" component is chosen; ``SPINSLICER_GPU_DIR`` points to any other place.
    """
    candidates = []
    override = os.environ.get(GPU_ADDON_ENV)
    if override:
        candidates.append(Path(override))
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "gpu")
    return next((folder for folder in candidates if folder.is_dir()), None)


def enable_gpu_addon() -> bool:
    """Make the add-on importable; False when there is none.

    It goes to the *end* of ``sys.path``, so the program's own numpy and other
    libraries always win over any copy inside the add-on.
    """
    folder = gpu_addon_directory()
    if folder is None:
        return False
    text = str(folder)
    if text not in sys.path:
        sys.path.append(text)
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(text)
        except OSError:
            pass
    importlib.invalidate_caches()
    return True


enable_gpu_addon()


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


_taichi_state: dict = {"tried": False, "arch": None, "generation": 0, "error": ""}


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
    except Exception as exc:  # missing wheel, no driver, unsupported device, wrong Python
        _taichi_state["arch"] = None
        _taichi_state["error"] = f"{type(exc).__name__}: {exc}".splitlines()[0][:160]
    return _taichi_state["arch"]


def selftest_gpu_kernels(arch: str = "cpu") -> dict:
    """Compile and run every GPU kernel on a tiny problem; nothing is swallowed.

    This is what catches a broken package: Taichi builds a kernel from the source text of
    its function, which a bytecode-only build does not have. ``arch="cpu"`` works on any
    machine (CI); ``"gpu"`` uses the graphics card. Returns ``{"ok", "arch", "error"}``.
    """
    result: dict = {"ok": False, "arch": arch, "error": None}
    started = False
    try:
        import numpy as np
        import scipy.sparse as sp
        import taichi as ti

        shutdown_taichi()
        ti.init(arch=ti.cpu if arch == "cpu" else ti.gpu, log_level=ti.ERROR, offline_cache=True,
                default_fp=ti.f32, default_ip=ti.i32)
        started = True
        result["arch"] = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
        from optics.gpu_operator import GpuMatrix, GpuSolver

        rng = np.random.default_rng(0)
        a = sp.random(30, 20, density=0.3, format="csr", dtype=np.float32, random_state=0)
        at = a.T.tocsr()
        gpu_a, gpu_at = GpuMatrix(a), GpuMatrix(at)
        values = rng.random((20, 3), dtype=np.float32)
        if not np.allclose(gpu_a._device_matmul(values), a @ values, rtol=1e-5, atol=1e-6):
            raise RuntimeError("the sparse product kernel returned wrong numbers")
        ones_rays = np.ones(30, dtype=np.float32)
        ones_pixels = np.ones(20, dtype=np.float32)
        solver = GpuSolver(gpu_a, gpu_at, ones_rays, ones_pixels)
        rays = rng.random((30, 3), dtype=np.float32)
        target = (rng.random((20, 3)) > 0.5).astype(np.float32)
        solver.run(rays, target, rays > 0.1, 1, 1, 0.8, 10.0, lambda kind, index: None)
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if started:
            try:
                import gc

                import taichi as ti

                gc.collect()
                ti.reset()
            except Exception:
                pass
            _taichi_state["generation"] += 1
    return result


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


def taichi_error() -> str:
    """Why Taichi could not be started in this process ('' when it did or was not tried)."""
    return str(_taichi_state["error"])


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

    def gpu_usable(asked_for_it: bool) -> str:
        if not (use_gpu and hw.taichi_installed):
            return ""
        # nvidia-smi is the only source of a card's memory. Without it the card may be an
        # integrated one (Intel, AMD APU) that is slower than the processor or has shaky
        # drivers: Auto leaves such cards alone, and does not even start Taichi for
        # them. Only an explicit "Maximum" tries whatever the machine offers.
        if hw.gpu_memory is None and not asked_for_it:
            return ""
        if hw.gpu_memory is not None and hw.gpu_memory < MIN_GPU_MEMORY_BYTES:
            return ""
        arch = init_taichi_gpu()
        return (hw.gpu_name or "GPU") + f" ({arch})" if arch else ""

    if mode == "economy":
        return economy(tr("минимум памяти и один поток, результат тот же"))
    if mode == "maximum":
        gpu = gpu_usable(True)
        if gpu:
            return maximum(tr("вычисления на видеокарте"), gpu)
        if use_gpu and hw.taichi_installed and taichi_error():
            # Installed but would not start: say so instead of pretending there is no GPU.
            return maximum(tr("Taichi не запустился ({error}); все ядра процессора").format(error=taichi_error()), "")
        return maximum(tr("все ядра процессора"), "")

    # auto
    if hw.cores <= 2 or (free is not None and free < ECONOMY_BELOW_FREE_RAM):
        return economy(tr("мало ядер или свободной памяти"))
    gpu = gpu_usable(False)
    if gpu:
        return maximum(tr("найдена подходящая видеокарта"), gpu)
    if use_gpu and hw.taichi_installed and taichi_error():
        return maximum(tr("Taichi не запустился ({error}); все ядра процессора").format(error=taichi_error()), "")
    if hw.cores >= AUTO_MIN_CORES and (free is None or free >= AUTO_MIN_FREE_RAM):
        return maximum(tr("достаточно ядер и памяти, видеокарта не используется"), "")
    return economy(tr("видеокарты нет, ядер или памяти немного"))


def describe_hardware(hw: Optional[Hardware] = None) -> str:
    """One-line summary for the settings panel (does not initialise Taichi)."""
    hw = hw or detect_hardware()
    ram = (
        tr("{value:.1f} ГиБ свободно").format(value=hw.free_ram / 1024**3)
        if hw.free_ram else tr("память неизвестна")
    )
    gpu = hw.gpu_name or tr("видеокарта не определена")
    if hw.taichi_installed:
        taichi = tr("Taichi установлен")
    elif hw.gpu_name:
        taichi = tr("Taichi не установлен (компонент «GPU-ускорение» установщика ускорит расчёт)")
    else:
        taichi = tr("Taichi не установлен")
    return tr("{cores} ядер, {ram}; {gpu}; {taichi}").format(cores=hw.cores, ram=ram, gpu=gpu, taichi=taichi)
