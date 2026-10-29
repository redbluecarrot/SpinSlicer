"""Headless smoke check of the source app and the actual release executable."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import cv2
import trimesh
from PyQt6.QtCore import QTimer

import compute
from compute import selftest_gpu_kernels
from frame_io import FrameRepository, load_manifest
from optics.optimization import _load_cpp_backend
from slicing_engine import ResinSettings, SliceParams
from workers import GenerationWorker, VideoExportWorker


def check_release(app, window, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    report = {"ok": False, "tabs": [type(window.tabs.widget(i)).__name__ for i in range(window.tabs.count())],
              "simulation_loaded": any(name in sys.modules for name in
                                       ("simulator_tab", "simulation_workbench_tab", "guided_simulation_tab", "printer_search"))}
    report["native_backend"] = _load_cpp_backend() is not None
    # The optional GPU add-on (the installer's component): is it importable from here?
    try:
        import taichi

        report["taichi_version"] = ".".join(str(part) for part in taichi.__version__)
    except Exception as exc:
        report["taichi_version"] = None
        # Present but unusable (the add-on was built for another Python, a DLL is missing).
        report["taichi_error"] = f"{type(exc).__name__}: {exc}"[:300] if compute.gpu_addon_directory() else None
    # Importing is not enough: run a kernel (on the CPU, so it works without a graphics card).
    # SPINSLICER_SELFTEST_ARCH=gpu runs it on the graphics card instead (a local check).
    arch = "gpu" if os.environ.get("SPINSLICER_SELFTEST_ARCH") == "gpu" else "cpu"
    report["gpu_kernels"] = selftest_gpu_kernels(arch) if report["taichi_version"] else None
    workers = []  # Keep all QThreads alive until the controller has joined them.

    def finish(error=None):
        if error:
            report["error"] = str(error)
        report["shutdown_ok"] = window._job_controller.shutdown()
        report["ok"] = (not error and report["shutdown_ok"] and not report["simulation_loaded"]
                        and report["tabs"] == ["SlicerTab", "ProjectorTab", "ProcessSettingsPanel"]
                        and report.get("frame_count") == report.get("video_frame_count") == 8
                        and report.get("manifest_complete") is True and report.get("output_forwarded") is True)
        report["ok"] = (report["ok"] and report.get("dose_verified") is True
                        and (not getattr(sys, "frozen", False) or report["native_backend"]))
        (output / "self-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            window.close()
        app.exit(0 if report["ok"] else 1)

    def exported(path):
        capture = cv2.VideoCapture(path)
        count = 0
        try:
            while capture.read()[0]:
                count += 1
        finally:
            capture.release()
        report["video_frame_count"] = count
        report["video_bytes"] = Path(path).stat().st_size
        finish()

    def assembled(frames):
        worker = VideoExportWorker(frames, 8, str(output / "preview.mp4"), window)
        workers.append(worker)
        window._job_controller.register(worker)
        worker.finished_ok.connect(exported)
        worker.failed.connect(finish)
        worker.start()

    def generated(count, path):
        try:
            info = FrameRepository.validate(path)
            manifest = load_manifest(info.resolved_dir)
            report["frame_count"] = info.frame_count
            ran = ((manifest.slice_parameters if manifest else {}).get("compute") or {})
            report["device_used"] = ran.get("device_used")
            report["gpu_error"] = ran.get("gpu_error")
            report["manifest_complete"] = bool(manifest and manifest.complete)
            statistics = dict(line.split("=", 1) for line in (Path(path) / "dose_stats.txt").read_text().splitlines())
            report["dose_rmse"] = float(statistics["rmse"])
            report["dose_verified"] = (math.isfinite(report["dose_rmse"])
                                       and statistics.get("source") == "exported_png"
                                       and (Path(path) / "reconstruction_dose.npy").is_file())
            window.slicer_tab.outputGenerated.emit(path)
            report["output_forwarded"] = window.projector_tab._output_dir == path
            window.projector_tab._on_assemble_clicked()
            worker = window.projector_tab._assemble_worker
            worker.finished_ok.connect(assembled)
            worker.failed.connect(finish)
        except Exception as exc:
            finish(exc)

    def start():
        try:
            source = output / "test-model.stl"
            trimesh.creation.box(extents=(10, 10, 10)).export(source)
            window.slicer_tab._on_model_loaded(str(source), trimesh.load(source, force="mesh"))
            node = window.slicer_tab._model_node
            node.set_rotation_deg(z=15)
            params = SliceParams(diameter_mm=window.settings_tab.vat_diameter_mm(), grid_res=24,
                                 output_res=32, num_frames=8, fill_holes=True,
                                 resin=ResinSettings(), output_dir=str(output / "frames"),
                                 projection_backend="physical", optimizer_iterations=3,
                                 cuvette_outer_mm=window.settings_tab.vat_diameter_mm(),
                                 cuvette_inner_mm=window.settings_tab.vat_diameter_mm()-4.0)
            context = {"source_path": str(source), "transform_matrix": node.matrix().tolist(), "units": "mm"}
            worker = GenerationWorker(node.get_transformed_mesh(), params, window, context)
            workers.append(worker)
            window._job_controller.register(worker)
            worker.finished_ok.connect(generated)
            worker.failed.connect(finish)
            worker.start()
        except Exception as exc:
            finish(exc)

    QTimer.singleShot(0, start)
    QTimer.singleShot(30000, lambda: finish("Release smoke check timed out"))
    return app.exec()
