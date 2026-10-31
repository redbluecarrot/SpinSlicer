"""
engine_process.py
=================
Runs one slicing job in a separate process, so the interface never waits for it.

Inside one Python process a compute thread and the UI share the GIL: measured
in the app, the window got control only every ~0.7 s during a compute, with
stalls up to 6 s. A child process has its own interpreter, runs at a lower
priority, and the UI stays fluid.

Protocol: the parent pickles ``(mesh, params, manifest_context, calibration)``
into a job file and starts ``python engine_process.py JOB CANCEL_FLAG`` (or the
frozen app with ``--engine-worker``). The child prints one JSON line per event:
``{"p": frac, "m": msg}``, then ``{"done": [n, out_dir]}``, ``{"cancelled": 1}``
or ``{"error": text}``. Creating the CANCEL_FLAG file asks it to stop.

Imports nothing from Qt or the 3D stack: the child starts quickly and light.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import traceback


def _lower_priority() -> None:
    try:
        if hasattr(os, "nice"):
            os.nice(8)
    except OSError:
        pass


def _single_threaded_blas() -> None:
    """The engine splits its work across cores itself (optics.fastcal.parallel_map).

    A multi-threaded BLAS inside each of those threads would oversubscribe the CPU
    (8 threads × 8 BLAS threads on an 8-core laptop) and run slower, so BLAS is
    pinned to one thread. Must happen before numpy is imported.
    """
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def run_job(job_path: str, cancel_path: str) -> int:
    _single_threaded_blas()
    _lower_priority()
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    out = sys.stdout if sys.stdout is not None else os.fdopen(1, "w", encoding="utf-8")
    sys.stdout = sys.stderr if sys.stderr is not None else open(os.devnull, "w")      # stray prints from libraries must not mix with the protocol

    def emit(payload: dict) -> None:
        out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        out.flush()

    last = [0.0, -1.0]

    def progress(frac: float, msg: str) -> None:
        now = time.monotonic()
        # The UI needs at most ~15 updates a second.
        if now - last[0] >= 0.066 or frac >= 1.0 or abs(frac - last[1]) >= 0.05:
            last[0], last[1] = now, frac
            emit({"p": float(frac), "m": str(msg)})

    checked = [0.0, False]

    def cancelled() -> bool:
        now = time.monotonic()
        if now - checked[0] > 0.1:
            checked[0] = now
            checked[1] = os.path.exists(cancel_path)
        return bool(checked[1])

    try:
        from slicing_engine import SliceCancelled, SlicingEngine

        with open(job_path, "rb") as handle:
            mesh, params, context, calibration = pickle.load(handle)
        try:
            num_frames, out_dir = SlicingEngine.run(
                mesh, params, progress_cb=progress, is_cancelled=cancelled,
                manifest_context=context, calibration=calibration)
        except SliceCancelled:
            emit({"cancelled": 1})
            return 0
        emit({"done": [int(num_frames), str(out_dir)]})
        return 0
    except Exception as exc:  # reported to the UI, never a silent crash
        emit({"error": f"{exc}\n{traceback.format_exc(limit=6)}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(run_job(sys.argv[1], sys.argv[2]))
