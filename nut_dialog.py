"""Qt dialog for the parametric threaded-nut generator."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from threaded_nut import ThreadedNutParameters


class ThreadedNutDialog(QDialog):
    """Collect printable nut dimensions before building the mesh."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Create a threaded nut")
        self.setMinimumWidth(440)

        root = QVBoxLayout(self)
        intro = QLabel(
            "Generate a watertight nut with a helical internal thread. "
            "Dimensions are in millimetres; clearance is radial."
        )
        intro.setWordWrap(True)
        intro.setObjectName("hintLabel")
        root.addWidget(intro)

        form = QFormLayout()
        self.outer_diameter = self._double(24.0, 0.1, 1000.0, 2)
        self.bore_diameter = self._double(12.0, 0.1, 999.0, 2)
        self.nut_height = self._double(10.0, 0.1, 1000.0, 2)
        self.pitch = self._double(2.0, 0.05, 100.0, 2)
        self.thread_depth = self._double(0.75, 0.01, 50.0, 2)
        self.clearance = self._double(0.15, 0.0, 10.0, 2)
        self.segments_per_turn = self._integer(128, 24, 1024)
        self.axial_segments_per_pitch = self._integer(24, 6, 256)

        form.addRow("Outer diameter, mm", self.outer_diameter)
        form.addRow("Thread bore diameter, mm", self.bore_diameter)
        form.addRow("Nut height, mm", self.nut_height)
        form.addRow("Thread pitch, mm", self.pitch)
        form.addRow("Thread depth (radial), mm", self.thread_depth)
        form.addRow("Radial clearance, mm", self.clearance)
        form.addRow("Angular samples / turn", self.segments_per_turn)
        form.addRow("Axial samples / pitch", self.axial_segments_per_pitch)
        root.addLayout(form)

        self.note = QLabel(
            "For a first print, keep at least 2–3 voxels across the thread depth "
            "and use a pitch larger than the voxel size."
        )
        self.note.setWordWrap(True)
        self.note.setObjectName("hintLabel")
        root.addWidget(self.note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @staticmethod
    def _double(value: float, minimum: float, maximum: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.05 if decimals == 2 else 0.1)
        spin.setValue(value)
        return spin

    @staticmethod
    def _integer(value: int, minimum: int, maximum: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def parameters(self) -> ThreadedNutParameters:
        return ThreadedNutParameters(
            outer_diameter_mm=self.outer_diameter.value(),
            bore_diameter_mm=self.bore_diameter.value(),
            height_mm=self.nut_height.value(),
            pitch_mm=self.pitch.value(),
            thread_depth_mm=self.thread_depth.value(),
            clearance_mm=self.clearance.value(),
            segments_per_turn=self.segments_per_turn.value(),
            axial_segments_per_pitch=self.axial_segments_per_pitch.value(),
        )

    def _validate_and_accept(self) -> None:
        try:
            self.parameters().validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid nut parameters", str(exc))
            return
        self.accept()
