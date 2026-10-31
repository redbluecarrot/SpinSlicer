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

import json
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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from constants import APP_ORG, APP_TITLE, FILL_FRACTION, VAT_HEIGHT_RATIO
from i18n import num, tr
from job_controller import JobController
from model_node import ModelNode
from print_plan import format_print_plan
from slicing_engine import ResinSettings, SliceParams
from ui_kit import EmptyScene
from ui_panels import ProcessSettingsPanel
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
    # New shell (header + side panel) listens to these instead of message boxes.
    modelInfo = pyqtSignal(str, object, int, float)
    fitChanged = pyqtSignal(bool, str)
    runningChanged = pyqtSignal(bool)
    generationProgress = pyqtSignal(float)
    resultsReady = pyqtSignal(object, str)
    virtualAvailable = pyqtSignal(bool)

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
        self._calibration_run = False
        self._last_output_dir: Optional[str] = None

        self._vat_timer = QTimer(self)
        self._vat_timer.setSingleShot(True)
        self._vat_timer.timeout.connect(self._apply_vat_diameter)

        self._build_ui()
        self._wire_signals()

        self._viewport.update_vat(self._process_panel.vat_diameter_mm())
        self.progress.emit(0.0, "Готово к работе. Загрузите STL-модель, чтобы начать.")

    # =======================================================================
    # Построение интерфейса
    # =======================================================================
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        toolbar = QHBoxLayout()
        self.load_btn = QPushButton(tr("Загрузить STL"))
        self.load_btn.setToolTip(tr("Открыть STL-файл модели"))
        self.load_btn.clicked.connect(self._on_load_clicked)
        toolbar.addWidget(self.load_btn)

        self.generate_btn = QPushButton(tr("Сгенерировать проекции"))
        self.generate_btn.setObjectName("generateButton")
        self.generate_btn.setToolTip(tr("Запустить расчёт проекций в фоновом потоке"))
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        toolbar.addWidget(self.generate_btn)

        self.virtual_btn = QPushButton(tr("Виртуальная печать"))
        self.virtual_btn.setCheckable(True)
        self.virtual_btn.setEnabled(False)
        self.virtual_btn.setToolTip(tr("Показать, что затвердеет из этих кадров (без принтера)"))
        self.virtual_btn.toggled.connect(self._on_virtual_toggled)
        toolbar.addWidget(self.virtual_btn)

        self.calibrate_btn = QPushButton(tr("Калибровка"))
        self.calibrate_btn.setToolTip(tr("Пробная печать, которая подберёт экспозицию под вашу смолу"))
        self.calibrate_btn.clicked.connect(self._on_calibrate_clicked)
        toolbar.addWidget(self.calibrate_btn)
        toolbar.addStretch(1)
        # The actions live in the side panel now; the buttons stay as the
        # programmatic surface (release check, tests) but are not shown.
        self._toolbar_box = QWidget()
        self._toolbar_box.setLayout(toolbar)
        self._toolbar_box.setVisible(False)
        root.addWidget(self._toolbar_box)

        # Настройки принтера находятся на отдельной вкладке, поэтому слайсер
        # оставляет вьюпорту всю ширину окна.
        self._viewport = Viewport3D()
        self._empty = EmptyScene()
        self._empty.openRequested.connect(self._on_load_clicked)
        self._empty.sampleRequested.connect(self.load_test_nut)
        self._empty.fileDropped.connect(self.load_path)
        self._scene_stack = QStackedWidget()
        self._scene_stack.addWidget(self._empty)


        self._scene_stack.addWidget(self._viewport)
        self._scene_stack.setCurrentWidget(self._empty)
        root.addWidget(self._scene_stack, 1)

    def _wire_signals(self) -> None:
        self._process_panel.diameterChanged.connect(self._on_diameter_changed)

        tp = self._process_panel.transform_panel
        tp.positionChanged.connect(self._on_position_changed)
        tp.rotationChanged.connect(self._on_rotation_changed)
        tp.scaleSizeChanged.connect(self._on_scale_size_changed)
        tp.centerRequested.connect(self._on_center)
        tp.autoFitRequested.connect(self._on_autofit)
        tp.resetRequested.connect(self._on_reset_transform)

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
        self.load_path(path)

    def load_path(self, path: str) -> None:
        self.load_btn.setEnabled(False)
        self.progress.emit(0.1, "Загрузка модели...")

        self._load_worker = LoadMeshWorker(path, self)
        self._job_controller.register(self._load_worker)
        self._load_worker.loaded.connect(self._on_model_loaded)
        self._load_worker.failed.connect(self._on_model_load_failed)
        self._load_worker.finished.connect(lambda: self.load_btn.setEnabled(True))
        self._load_worker.start()

    def load_test_nut(self) -> None:
        from threaded_nut import create_default_threaded_nut
        self.progress.emit(0.3, "Создаю тестовую гайку M20...")
        mesh = create_default_threaded_nut()
        folder = os.path.join(os.path.expanduser("~"), "SpinSlicer")
        os.makedirs(folder, exist_ok=True)
        self._on_model_loaded(os.path.join(folder, "test_nut_M20.stl"), mesh)

    def _on_model_loaded(self, path: str, mesh: trimesh.Trimesh) -> None:
        self._scene_stack.setCurrentWidget(self._viewport)
        self._reset_virtual_print()
        node = ModelNode(mesh, path)
        diameter = self._process_panel.vat_diameter_mm()
        node.fit_to_vat(self._inner_diameter(), self._printable_height(), FILL_FRACTION)
        self._model_node = node

        self._viewport.set_model(node.original_mesh)
        self._viewport.update_vat(diameter, self._process_panel.container_mode_value())
        self._viewport.update_model_transform(node.matrix())
        self._viewport.reset_camera()
        self._process_panel.transform_panel.sync_from_model(self._model_node)

        self._process_panel.transform_panel.setEnabled(True)

        self._emit_model_state()
        self.progress.emit(1.0, "Модель загружена.")
        self.logMessage.emit(
            f"{tr('Загружено:')} {os.path.basename(path)} "
            f"({node.vertex_count:,} {tr('верш.')}, {node.face_count:,} {tr('гран.')})"
        )

    def _on_model_load_failed(self, msg: str) -> None:
        self.progress.emit(0.0, "Ошибка загрузки.")
        self.logMessage.emit("ОШИБКА загрузки: " + msg.splitlines()[0])
        QMessageBox.critical(self, tr("Ошибка"), msg)


    # =======================================================================
    # Трансформации объекта
    # =======================================================================
    def _on_transform_mode_changed(self, mode: str) -> None:
        self._viewport.set_transform_mode(mode)

    def _on_viewport_transform_changed(self, matrix) -> None:
        node = self._model_node
        if node is None: return
        # A click on the gizmo without a real move must not throw away the last result.
        if np.allclose(np.asarray(matrix, dtype=np.float64), node.matrix(), atol=1e-6):
            return
        node.set_matrix(matrix, uniform=False)
        self._process_panel.transform_panel.sync_from_model(node)
        self._sync_and_redraw()

    def _on_position_changed(self, axis: int, value: float) -> None:
        node = self._model_node
        if node is None: return
        pos = np.asarray(node.transform.translation, dtype=np.float64).copy()
        pos[axis] = value
        node.transform.translation = pos
        self._process_panel.transform_panel.sync_from_model(node)
        self._sync_and_redraw()

    def _on_rotation_changed(self, axis: int, value: float) -> None:
        node = self._model_node
        if node is None: return
        rot = np.asarray(node.transform.rotation_deg, dtype=np.float64).copy()
        rot[axis] = value
        node.transform.rotation_deg = rot
        self._process_panel.transform_panel.sync_from_model(node)
        self._sync_and_redraw()

    def _on_scale_size_changed(self, axis: int, value: float) -> None:
        node = self._model_node
        if node is None: return
        base_extents = np.asarray(node.base_extents, dtype=np.float64)
        scale = np.asarray(node.transform.scale, dtype=np.float64).copy()
        target_size = max(float(value), 1e-6)
        scale[axis] = target_size / max(float(base_extents[axis]), 1e-9)
        node.transform.scale = np.maximum(scale, 1e-6)
        self._process_panel.transform_panel.sync_from_model(node)
        self._sync_and_redraw()

    def _on_reset_transform(self) -> None:
        node = self._model_node
        if node is None: return
        node.transform.reset()
        self._process_panel.transform_panel.sync_from_model(node)
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
        self._model_node.fit_to_vat(self._inner_diameter(), self._printable_height(), FILL_FRACTION)
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
        self._reset_virtual_print()
        self._viewport.update_model_transform(node.matrix())
        self._process_panel.transform_panel.sync_from_model(self._model_node)
        self._emit_model_state()

    # =======================================================================
    # Model state for the side panel (size, scale, does it fit)
    # =======================================================================
    def _inner_diameter(self) -> float:
        panel = self._process_panel
        return max(panel.vat_diameter_mm() - 2 * panel.glass_thickness_mm(), 1.0)

    def _printable_height(self) -> float:
        panel = self._process_panel
        if hasattr(panel, "printable_height_mm"):
            return float(panel.printable_height_mm())
        return panel.vat_diameter_mm() * VAT_HEIGHT_RATIO

    def model_fit(self) -> tuple[bool, str]:
        node = self._model_node
        if node is None:
            return True, ""
        verts = np.asarray(node.original_mesh.vertices, dtype=np.float64)
        if len(verts) > 20000:
            verts = verts[:: max(1, len(verts) // 20000)]
        m = node.matrix()
        world = verts @ m[:3, :3].T + m[:3, 3]
        r_max = float(np.max(np.hypot(world[:, 0], world[:, 1])))
        height = float(np.ptp(world[:, 2]))
        half_vial = self._process_panel.vat_diameter_mm() * VAT_HEIGHT_RATIO / 2.0
        inner_r = self._inner_diameter() / 2.0
        limit_h = self._printable_height()
        f = num
        if r_max > inner_r:
            return False, f"Шире колбы: деталь уходит на {f(r_max)} мм от оси, внутри {f(inner_r)}"
        if height > limit_h + 1e-6 or np.max(np.abs(world[:, 2])) > half_vial:
            return False, f"Выше поля: {f(height)} мм при доступных {f(limit_h)}"
        edge = " — у самой стенки, лучше ближе к оси" if r_max > inner_r * FILL_FRACTION else ""
        return True, f"Влезает: {f(2 * r_max)} мм из {f(2 * inner_r)}, по высоте {f(height)} из {f(limit_h)}{edge}"

    def _emit_model_state(self) -> None:
        node = self._model_node
        if node is None:
            self.modelInfo.emit("", None, 0, 100.0)
            return
        size = [float(x) for x in node.current_size_mm()]
        scale_pct = float(np.mean(node.transform.scale)) * 100.0
        self.modelInfo.emit(os.path.basename(node.source_path) or "модель", size, node.face_count, scale_pct)
        ok, text = self.model_fit()
        self.fitChanged.emit(ok, text)

    def refresh_model_state(self) -> None:
        self._emit_model_state()

    def set_scale_pct(self, pct: float) -> None:
        node = self._model_node
        if node is None:
            return
        node.transform.scale = np.full(3, max(pct, 0.1) / 100.0, dtype=np.float64)
        self._sync_and_redraw()

    def set_orientation(self, index: int) -> None:
        node = self._model_node
        if node is None:
            return
        node.transform.rotation_deg = np.array([(0.0, 0.0, 0.0), (90.0, 0.0, 0.0), (0.0, 90.0, 0.0)][index % 3])
        self._sync_and_redraw()

    @property
    def viewport(self) -> Viewport3D:
        return self._viewport

    # =======================================================================
    # Генерация проекций
    # =======================================================================
    def _slice_params(self, out_dir: str) -> SliceParams:
        """Current printer, resin and slicing settings."""
        diameter = self._process_panel.vat_diameter_mm()
        resin = ResinSettings(
            base_exposure=self._process_panel.exposure.value(),
            intensity=self._process_panel.intensity.value(),
            threshold=self._process_panel.threshold.value(),
        )

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
            alpha=self._process_panel.alpha_absorption_value(),
            projector_gamma=self._process_panel.projector_gamma.value(),
            projector_irradiance_mw_cm2=self._process_panel.projector_irradiance.value(),
            projector_contrast=self._process_panel.projector_contrast.value(),
            projector_throw_mm=self._process_panel.projector_throw.value(),
            projector_blur_mm=(self._process_panel.effective_blur_mm()
                               if hasattr(self._process_panel, "effective_blur_mm")
                               else self._process_panel.projector_blur.value()),
            resin_gel_dose_mj_cm3=self._process_panel.gel_dose.value(),
        )
        if hasattr(self._process_panel, "frame_rate_hz_value"):
            params.frame_rate_hz = float(self._process_panel.frame_rate_hz_value())
        vam_conda, vam_environment = self._process_panel.vam_runtime_config()
        params.vam_conda_executable = vam_conda
        params.vam_environment_name = vam_environment or params.vam_environment_name
        return params

    def _on_generate_clicked(self) -> None:
        if self._gen_worker is not None and self._gen_worker.isRunning():
            self._gen_worker.request_cancel()
            self.generate_btn.setEnabled(False)
            self.generate_btn.setText(tr("Остановка..."))
            return
        self._reset_virtual_print()

        if self._model_node is None:
            QMessageBox.warning(self, tr("Внимание"), tr("Сначала загрузите STL-модель."))
            return

        if self._model_node.source_path:
            base_dir = os.path.dirname(os.path.abspath(self._model_node.source_path))
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        params = self._slice_params(os.path.join(base_dir, "output_frames"))
        diameter = params.diameter_mm
        # Генерация всегда использует полностью трансформированную копию —
        # original_mesh внутри ModelNode остаётся нетронутым.
        mesh_to_process = self._model_node.get_transformed_mesh()

        preflight = preflight_mesh(
            mesh_to_process,
            params,
            vat_height_mm=min(diameter * VAT_HEIGHT_RATIO, self._printable_height()),
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
        self._start_generation(mesh_to_process, params, manifest_context)

    def _start_generation(self, mesh: trimesh.Trimesh, params: SliceParams,
                          manifest_context: dict, calibration: object = None) -> None:
        self._calibration_run = calibration is not None
        self._gen_worker = GenerationWorker(
            mesh,
            params,
            self,
            manifest_context=manifest_context,
            calibration=calibration,
        )
        self._job_controller.register(self._gen_worker)
        self._gen_worker.progress.connect(self._on_generation_progress)
        self._gen_worker.finished_ok.connect(self._on_generation_done)
        self._gen_worker.failed.connect(self._on_generation_failed)
        self._gen_worker.cancelled.connect(self._on_generation_cancelled)
        self._gen_worker.finished.connect(self._on_generation_thread_finished)

        self.generate_btn.setText(tr("⏹ Отменить генерацию"))
        self.runningChanged.emit(True)
        self.load_btn.setEnabled(False)
        self.logMessage.emit("Запуск генерации проекций...")
        self._gen_worker.start()

    # =======================================================================
    # Калибровка экспозиции
    # =======================================================================
    def _on_calibrate_clicked(self) -> None:
        from PyQt6.QtCore import QSettings

        from calibration_dialog import CalibrationDialog
        panel = self._process_panel
        dialog = CalibrationDialog(QSettings(APP_ORG, APP_TITLE), self._start_calibration,
                                   panel.gel_dose.value, panel.gel_dose.setValue, self)
        dialog.exec()

    def _start_calibration(self) -> None:
        from calibration import Ladder
        from optics import fastcal
        if self._gen_worker is not None and self._gen_worker.isRunning():
            return
        out_dir = os.path.join(os.path.expanduser("~"), "SpinSlicer", "calibration")
        params = self._slice_params(out_dir)
        angles = np.deg2rad((params.angle_offset_deg + params.angle_direction
                             * np.linspace(0.0, 360.0, params.num_frames, endpoint=False)) % 360.0)
        if params.projection_backend != "physical" or not fastcal.supports(params.container_mode, angles):
            QMessageBox.warning(self, tr("Калибровка экспозиции"),
                                tr("Калибровка работает с физическим движком и круглой колбой."))
            return
        self._reset_virtual_print()
        ladder = Ladder()
        self._start_generation(ladder.mesh(), params, {"model_type": "calibration_ladder", "units": "mm"},
                               calibration=ladder)

    # =======================================================================
    # Виртуальная печать
    # =======================================================================
    def _reset_virtual_print(self) -> None:
        """The preview belongs to the last run of the current placement."""
        if self.virtual_btn.isChecked():
            self.virtual_btn.setChecked(False)
        if self.virtual_btn.isEnabled():
            self.virtual_btn.setEnabled(False)
            self.virtualAvailable.emit(False)

    def _on_virtual_toggled(self, checked: bool) -> None:
        if not checked:
            self._viewport.clear_virtual_print()
            return
        import virtual_print
        path = virtual_print.mesh_path(self._last_output_dir) if self._last_output_dir else None
        if path is None:
            self.virtual_btn.setChecked(False)
            self.virtual_btn.setEnabled(False)
            return
        try:
            mesh = trimesh.load(path, force="mesh")
            self._viewport.show_virtual_print(mesh)
        except Exception as exc:  # display only: never break the run
            self.logMessage.emit(f"virtual_print: {exc}")
            self.virtual_btn.setChecked(False)

    def _on_generation_progress(self, frac: float, msg: str) -> None:
        self.progress.emit(frac, msg)
        self.generationProgress.emit(frac)

    def _on_generation_done(self, num_frames: int, out_dir: str) -> None:
        self._last_output_dir = out_dir
        if getattr(self, "_calibration_run", False):
            from PyQt6.QtCore import QSettings

            from calibration_dialog import SETTINGS_KEY
            QSettings(APP_ORG, APP_TITLE).setValue(SETTINGS_KEY, os.path.join(out_dir, "print_plan.json"))
        self.progress.emit(1.0, "Готово!")
        self.logMessage.emit(f"{tr('Сохранено')} {num_frames} {tr('кадров в:')} {out_dir}")
        plan_text = ""
        plan_path = os.path.join(out_dir, "print_plan.json")
        if os.path.isfile(plan_path):
            try:
                with open(plan_path, encoding="utf-8") as handle:
                    plan_text = format_print_plan(json.load(handle))
            except (OSError, ValueError, KeyError) as exc:
                plan_text = f"print_plan.json: {exc}"
            for line in plan_text.splitlines():
                self.logMessage.emit(line)
        self.outputGenerated.emit(out_dir)
        import virtual_print
        has_virtual = virtual_print.mesh_path(out_dir) is not None
        self.virtual_btn.setEnabled(has_virtual)
        plan = None
        if os.path.isfile(plan_path):
            try:
                with open(plan_path, encoding="utf-8") as handle:
                    plan = json.load(handle)
            except (OSError, ValueError):
                plan = None
        self.virtualAvailable.emit(has_virtual)
        self.resultsReady.emit(plan, out_dir)
        if not getattr(self, "_calibration_run", False) and self.receivers(self.resultsReady) > 0:
            return
        message = f"{tr('Сохранено')} {num_frames} {tr('кадров в:')}\n{out_dir}"
        if plan_text:
            message += "\n\n" + plan_text
        if has_virtual:
            message += "\n\n" + tr("Кнопка «Виртуальная печать» покажет деталь, которая затвердеет из этих кадров.")
        if getattr(self, "_calibration_run", False):
            message += "\n\n" + tr("Напечатайте эти кадры, затем откройте «Калибровка» и укажите самый короткий затвердевший столбик.")
        QMessageBox.information(self, tr("Успех"), message)

    def _on_generation_failed(self, msg: str) -> None:
        self.progress.emit(0.0, "Ошибка генерации.")
        self.logMessage.emit("ОШИБКА генерации: " + msg.splitlines()[0])
        QMessageBox.critical(self, tr("Ошибка генерации проекций"), msg)

    def _on_generation_cancelled(self) -> None:
        self.progress.emit(0.0, "Отменено пользователем.")
        self.logMessage.emit("Генерация отменена пользователем.")

    def _on_generation_thread_finished(self) -> None:
        self.runningChanged.emit(False)
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText(tr("Сгенерировать проекции"))
        self.load_btn.setEnabled(True)

    def closeEvent(self, event) -> None:  # noqa: N802 (имя метода задано Qt)
        if self._job_controller.shutdown():
            event.accept()
        else:
            event.ignore()
