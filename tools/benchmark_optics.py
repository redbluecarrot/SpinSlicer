"""Reproducible full-resolution matrix benchmark (run in a fresh process)."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optics.optimization import build_system_matrix


def peak_rss_bytes() -> int:
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_working_set", "working_set", "peak_paged", "paged",
                    "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.peak_working_set
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=0, help="Also benchmark the full-grid solver with this many layers.")
    arguments = parser.parse_args()
    started = time.perf_counter()
    matrix, transpose = build_system_matrix(
        np.linspace(0, 2 * np.pi, 360, endpoint=False),
        num_s=256, grid_res=256, grid_span_mm=60.0, L_out=60.0, L_in=56.0,
        n_air=1.0, n_glass=1.0, n_liquid=1.0, n_resin=1.0,
    )
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
        "grid": 256, "angles": 360, "matrix_seconds": time.perf_counter() - started,
        "nnz": matrix.nnz, "matrix_pair_bytes": sum(
            data.nbytes for item in (matrix, transpose)
            for data in (item.data, item.indices, item.indptr)
        ), "matrix_process_peak_rss_bytes": peak_rss_bytes(),
    }
    if arguments.layers:
        from optics.optimization import _load_cpp_backend, optimize_projections_sirt_volume
        coordinates = (np.arange(256) + 0.5) * 60 / 256 - 30
        xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
        ring = ((xx-3)**2 + (yy+2)**2 < 14**2) & ((xx-3)**2 + (yy+2)**2 > 5**2)
        target = ring[:, :, None].astype(np.float32) * np.linspace(0.3, 1, arguments.layers, dtype=np.float32)[None, None, :]
        started = time.perf_counter()
        result = optimize_projections_sirt_volume(
            target, np.linspace(0, 2*np.pi, 360, endpoint=False),
            L_out=60., L_in=56., grid_res=256, grid_span_mm=60.,
            n_air=1., n_glass=1., n_liquid=1., n_resin=1., iterations=8,
        )
        report.update(layers=arguments.layers, iterations=8, native=_load_cpp_backend() is not None,
                      solver_seconds=time.perf_counter()-started, volume_process_peak_rss_bytes=peak_rss_bytes(),
                      output_shape=list(result.shape), output_finite=bool(np.isfinite(result).all()))
    print(json.dumps(report, indent=2))
