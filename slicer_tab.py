"""
slicer_tab.py
=============
Вкладка "Слайсер": 3D-вьюпорт с колбой на PyVista, компактное управление
трансформацией объекта и генерация проекций в фоновом QThread.

Логика идентична прежней главной версии окна — она просто перенесена
внутрь QWidget, чтобы стать одной из вкладок QTabWidget. Локальные
кнопки загрузки и генерации находятся тут же, наверху вкладки, а не в
общем тулбаре — они осмысленны только в контексте этой вкладки.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import trimesh
from PyQt6.QtCore import QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from constants import FILL_FRACTION, VAT_HEIGHT_RATIO
from i18n import tr
from job_controller import JobController
from model_node import ModelNode
from slicing_engine import ResinSettings, SliceParams
from ui_panels import ProcessSettingsPanel, TransformToolbar
from validation import format_preflight_report, preflight_mesh
from viewport import Viewport3D
from workers import GenerationWorker, LoadMeshWorker


def open_local_directory(target: str) -> bool:
    """Open a directory through Qt's local-file URL handling.

    The Slicer button was removed from the UI, but this safe helper remains a
    small compatibility surface for integrations and security regression tests.
    """

    return QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(target)))


class SlicerTab(QWidget):
    progress = pyqtSignal(float, str)
    logMessage = pyqtSignal(str)
    # Сигнализирует вкладке "Проектор", в какой папке
    # появились свежие кадры — чтобы не заставлять пользователя каждый раз
    # указывать её вручную.
    outputGenerated = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        job_controller: Optional[JobController] = None,
        process_panel: Optional[ProcessSettingsPanel] = None,
    ):
        super().__init__(parent)

        self._job_controller = job_controller or JobController()
        self._process_panel = (
            process_panel if process_panel is not None else ProcessSettingsPanel(self)
        )
        self._model_node: Optional[ModelNode] = None
        self._load_worker: Optional[LoadMeshWorker] = None
        self._gen_worker: Optional[GenerationWorker] = None
        self._generation_cancel_requested = False
        self._last_output_dir: Optional[str] = None

        self._vat_timer = QTimer(self)
        self._vat_timer.setSingleShot(True)
        self._vat_timer.timeout.connect(self._apply_vat_diameter)

        self._build_ui()
        self._wire_signals()

        self._viewport.update_vat(
            self._process_panel.vat_diameter_mm(),
            self._process_panel.container_mode_value(),
        )
        self.progress.emit(0.0, "Готово к работе. Загрузите STL-модель, чтобы начать.")

    # =======================================================================
    # Построение интерфейса
    # =======================================================================
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        toolbar = QHBoxLayout()
        self.load_btn = QPushButton(tr("Загрузить STL"))
        self.load_btn.setToolTip(tr("Открыть STL-файл модели"))
        self.load_btn.clicked.connect(self._on_load_clicked)
        toolbar.addWidget(self.load_btn)

        self.generate_btn = QPushButton(tr("Сгенерировать проекции"))
        self.generate_btn.setObjectName("generateButton")
        self.generate_btn.setToolTip(tr("Запустить расчёт проекций в фоновом потоке"))
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        self.generate_btn.setEnabled(False)
        toolbar.addWidget(self.generate_btn)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        # Инструменты трансформации относятся к модели; управление камерой
        # доступно мышью и через куб ориентации во вьюпорте.
        self._transform_toolbar = TransformToolbar()
        root.addWidget(self._transform_toolbar)

        self._viewport = Viewport3D()
        root.addWidget(self._viewport, 1)

    def _wire_signals(self) -> None:
        self._process_panel.diameterChanged.connect(self._on_diameter_changed)

        self._transform_toolbar.modeChanged.connect(self._on_transform_mode_changed)
        self._transform_toolbar.scale_controls.scalePercentChanged.connect(
            self._on_scale_percent_changed
        )
        self._transform_toolbar.scale_controls.sizeChanged.connect(
            self._on_scale_size_changed
        )
        self._transform_toolbar.centerRequested.connect(self._on_center)
        self._transform_toolbar.autoFitRequested.connect(self._on_autofit)
        self._transform_toolbar.resetRequested.connect(self._on_reset_transform)
        self._viewport.transformChanged.connect(self._on_viewport_transform_changed)

    # =======================================================================
    # Колба (не зависит от модели)
    # =======================================================================
    def _on_diameter_changed(self, _value: float) -> None:
        self._vat_timer.start(120)

    def _apply_vat_diameter(self) -> None:
        self._viewport.update_vat(
            self._process_panel.vat_diameter_mm(),
            self._process_panel.container_mode_value()
        )

    # =======================================================================
    # Загрузка STL
    # =======================================================================
    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Выбрать STL"), "", tr("STL файлы (*.stl)")
        )
        if not path:
            return

        self.load_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self._transform_toolbar.set_enabled_state(False)
        self.progress.emit(0.1, "Загрузка модели...")

        self._load_worker = LoadMeshWorker(path, self)
        self._job_controller.register(self._load_worker)
        self._load_worker.loaded.connect(self._on_model_loaded)
        self._load_worker.failed.connect(self._on_model_load_failed)
        self._load_worker.finished.connect(lambda: self.load_btn.setEnabled(True))
        self._load_worker.start()

    def _on_model_loaded(self, path: str, mesh: trimesh.Trimesh) -> None:
        node = ModelNode(mesh, path)
        diameter = self._process_panel.vat_diameter_mm()
        node.fit_to_diameter(diameter, FILL_FRACTION)
        self._model_node = node

        self._viewport.set_model(node.original_mesh)
        self._viewport.update_vat(diameter, self._process_panel.container_mode_value())
        self._viewport.update_model_transform(node.matrix())
        self._viewport.reset_camera()
        self._transform_toolbar.sync_from_model(self._model_node)

        self._transform_toolbar.set_enabled_state(True)
        self.generate_btn.setEnabled(True)

        self.progress.emit(1.0, "Модель загружена.")
        self.logMessage.emit(
            f"Загружено: {os.path.basename(path)} "
            f"({node.vertex_count:,} верш., {node.face_count:,} гран.)"
        )

    def _on_model_load_failed(self, msg: str) -> None:
        self.generate_btn.setEnabled(self._model_node is not None)
        self._transform_toolbar.set_enabled_state(self._model_node is not None)
        self.progress.emit(0.0, "Ошибка загрузки.")
        self.logMessage.emit("ОШИБКА загрузки: " + msg.splitlines()[0])
        QMessageBox.critical(self, tr("Ошибка"), msg)

    def update_language(self) -> None:
        """Restore state-dependent button text after static translations run."""

        if self._gen_worker is not None and self._gen_worker.isRunning():
            label = "Остановка..." if self._generation_cancel_requested else "⏹ Отменить генерацию"
        else:
            label = "Сгенерировать проекции"
        self.generate_btn.setText(tr(label))
        self._viewport.update_language()


    # =======================================================================
    # Трансформации объекта
    # =======================================================================
    def _on_transform_mode_changed(self, mode: str) -> None:
        self._viewport.set_transform_mode(mode)

    def _on_viewport_transform_changed(self, matrix) -> None:
        node = self._model_node
        if node is None:
            return
        node.set_matrix(matrix, uniform=self._transform_toolbar.is_uniform())
        self._sync_and_redraw()

    def _on_scale_percent_changed(self, axis: int, value: float) -> None:
        node = self._model_node
        if node is None or axis not in (0, 1, 2):
            return
        scale = np.asarray(node.transform.scale, dtype=np.float64).copy()
        target_scale = max(float(value) / 100.0, 1e-6)
        if self._transform_toolbar.is_uniform():
            current_axis_scale = max(float(scale[axis]), 1e-9)
            scale *= target_scale / current_axis_scale
        else:
            scale[axis] = target_scale
        node.transform.scale = np.maximum(scale, 1e-6)
        self._sync_and_redraw()

    def _on_scale_size_changed(self, axis: int, value: float) -> None:
        node = self._model_node
        if node is None or axis not in (0, 1, 2):
            return
        current_size = np.asarray(node.current_size_mm(), dtype=np.float64)
        base_extents = np.asarray(node.base_extents, dtype=np.float64)
        scale = np.asarray(node.transform.scale, dtype=np.float64).copy()
        target_size = max(float(value), 1e-6)
        if self._transform_toolbar.is_uniform():
            current_axis_size = max(float(current_size[axis]), 1e-9)
            scale *= target_size / current_axis_size
        else:
            scale[axis] = target_size / max(float(base_extents[axis]), 1e-9)
        node.transform.scale = np.maximum(scale, 1e-6)
        self._sync_and_redraw()

    def _on_reset_transform(self) -> None:
        node = self._model_node
        if node is None:
            return
        node.transform.translation.fill(0.0)
        node.transform.rotation_deg.fill(0.0)
        node.transform.scale.fill(1.0)
        self._transform_toolbar.sync_from_model(node)
        self._sync_and_redraw()

    def _on_center(self) -> None:
        if self._model_node is None:
            return
        self._model_node.center_xy()
        self._sync_and_redraw()

    def _on_autofit(self) -> None:
        if self._model_node is None:
            return
        diameter = self._process_panel.vat_diameter_mm()
        self._model_node.fit_to_vat(
            diameter, diameter * VAT_HEIGHT_RATIO, FILL_FRACTION,
        )
        self._sync_and_redraw()
        size = self._model_node.current_size_mm()
        self.logMessage.emit(
            f"Авто-фит: модель вписана в колбу ∅{diameter:.0f} мм — "
            f"размер {size[0]:.2f} × {size[1]:.2f} × {size[2]:.2f} мм."
        )

    def _sync_and_redraw(self) -> None:
        node = self._model_node
        if node is None:
            return
        self._viewport.update_model_transform(node.matrix())
        self._transform_toolbar.sync_from_model(node)

    # =======================================================================
    # Генерация проекций
    # =======================================================================
    def _on_generate_clicked(self) -> None:
        if self._gen_worker is not None and self._gen_worker.isRunning():
            self._generation_cancel_requested = True
            self._gen_worker.request_cancel()
            self.generate_btn.setEnabled(False)
            self.generate_btn.setText(tr("Остановка..."))
            return

        if self._model_node is None:
            QMessageBox.warning(self, tr("Внимание"), tr("Сначала загрузите STL-модель."))
            return

        diameter = self._process_panel.vat_diameter_mm()
        resin = ResinSettings(
            base_exposure=self._process_panel.exposure.value(),
            intensity=self._process_panel.intensity.value(),
            threshold=self._process_panel.threshold.value(),
        )

        if self._model_node.source_path:
            base_dir = os.path.dirname(os.path.abspath(self._model_node.source_path))
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, "output_frames")

        params = SliceParams(
            diameter_mm=diameter,
            grid_res=self._process_panel.grid_res.value_int(),
            output_res=self._process_panel.output_res.value_int(),
            num_frames=self._process_panel.frames.value_int(),
            fill_holes=self._process_panel.fill_holes.isChecked(),
            resin=resin,
            output_dir=out_dir,
            projection_backend=self._process_panel.projection_backend_name(),
            optimizer_iterations=self._process_panel.optimizer_iterations_value(),
            preserve_internal_voids=self._process_panel.preserve_internal_voids.isChecked(),
            container_mode=self._process_panel.container_mode_value(),
            cuvette_outer_mm=diameter,
            cuvette_inner_mm=diameter - 2 * self._process_panel.glass_thickness_mm(),
            n_air=1.0,
            n_glass=self._process_panel.n_glass_value(),
            n_liquid=self._process_panel.n_liquid_value(),
            n_resin=self._process_panel.n_resin_value(),
            alpha=self._process_panel.alpha_absorption_value()
        )
        vam_conda, vam_environment = self._process_panel.vam_runtime_config()
        params.vam_conda_executable = vam_conda
        params.vam_environment_name = vam_environment or params.vam_environment_name

        # Генерация всегда использует полностью трансформированную копию —
        # original_mesh внутри ModelNode остаётся нетронутым.
        mesh_to_process = self._model_node.get_transformed_mesh()

        preflight = preflight_mesh(
            mesh_to_process,
            params,
            vat_height_mm=diameter * VAT_HEIGHT_RATIO,
        )
        report_text = format_preflight_report(preflight)
        self.logMessage.emit("Preflight отчёт:\n" + report_text)
        if not preflight.ok:
            self.progress.emit(0.0, tr("Проверка модели не пройдена."))
            self.logMessage.emit(f"{tr('ОШИБКА проверки модели: ')}{'; '.join(preflight.errors)}")
            QMessageBox.critical(self, tr("Модель не готова к слайсингу"), report_text)
            return
        if preflight.warnings:
            self.logMessage.emit("Предупреждения preflight: " + "; ".join(preflight.warnings))

        manifest_context = {
            "source_path": self._model_node.source_path,
            "transform_matrix": self._model_node.matrix().tolist(),
            "units": "mm",
        }
        self._gen_worker = GenerationWorker(
            mesh_to_process,
            params,
            self,
            manifest_context=manifest_context,
        )
        self._job_controller.register(self._gen_worker)
        self._gen_worker.progress.connect(self._on_generation_progress)
        self._gen_worker.finished_ok.connect(self._on_generation_done)
        self._gen_worker.failed.connect(self._on_generation_failed)
        self._gen_worker.cancelled.connect(self._on_generation_cancelled)
        self._gen_worker.finished.connect(self._on_generation_thread_finished)

        self._generation_cancel_requested = False
        self.generate_btn.setText(tr("⏹ Отменить генерацию"))
        self.load_btn.setEnabled(False)
        self._transform_toolbar.set_enabled_state(False)
        self._process_panel.setEnabled(False)
        self.logMessage.emit("Запуск генерации проекций...")
        self._gen_worker.start()

    def _on_generation_progress(self, frac: float, msg: str) -> None:
        self.progress.emit(frac, msg)

    def _on_generation_done(self, num_frames: int, out_dir: str) -> None:
        self._last_output_dir = out_dir
        self.progress.emit(1.0, "Готово!")
        self.logMessage.emit(f"Сохранено {num_frames} кадров в: {out_dir}")
        self.outputGenerated.emit(out_dir)
        QMessageBox.information(self, tr("Успех"), f"{tr('Сохранено')} {num_frames} {tr('кадров в:')}\n{out_dir}")

    def _on_generation_failed(self, msg: str) -> None:
        self.progress.emit(0.0, "Ошибка генерации.")
        self.logMessage.emit("ОШИБКА генерации: " + msg.splitlines()[0])
        QMessageBox.critical(self, tr("Ошибка генерации проекций"), msg)

    def _on_generation_cancelled(self) -> None:
        self.progress.emit(0.0, "Отменено пользователем.")
        self.logMessage.emit("Генерация отменена пользователем.")

    def _on_generation_thread_finished(self) -> None:
        self._generation_cancel_requested = False
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText(tr("Сгенерировать проекции"))
        self.load_btn.setEnabled(True)
        self._transform_toolbar.set_enabled_state(self._model_node is not None)
        self._process_panel.setEnabled(True)

    def closeEvent(self, event) -> None:  # noqa: N802 (имя метода задано Qt)
        if self._job_controller.shutdown():
            event.accept()
        else:
            event.ignore()
