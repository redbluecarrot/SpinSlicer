"""Exposure calibration dialog: make the ladder print, then read the result back."""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from calibration import PILLARS, Ladder, gel_dose_factor
from i18n import tr

SETTINGS_KEY = "calibration/last_plan"


class CalibrationDialog(QDialog):
    """Two steps: generate the exposure ladder, then enter which pillars cured."""

    def __init__(self, settings: QSettings, generate: Callable[[], None],
                 gel_dose: Callable[[], float], set_gel_dose: Callable[[float], None],
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(tr("Калибровка экспозиции"))
        self._settings = settings
        self._gel_dose = gel_dose
        self._set_gel_dose = set_gel_dose
        heights = Ladder().heights()
        root = QVBoxLayout(self)

        intro = QLabel(tr(
            "Одна пробная печать вместо подбора времени наугад.\n\n"
            "1. Сгенерируйте калибровку: 9 столбиков по кругу, каждый получает свою дозу — "
            "от −12 % до +12 % с шагом 3 %. Столбик 1 самый короткий и самый тусклый, "
            "столбик 9 самый высокий и яркий.\n"
            "2. Напечатайте кадры с временем из плана печати.\n"
            "3. Отмойте и посмотрите, какой самый короткий столбик затвердел. "
            "Выберите его ниже и нажмите «Применить» — слайсер поправит дозу гелеобразования."
        ))
        intro.setWordWrap(True)
        root.addWidget(intro)

        self.generate_btn = QPushButton(tr("Сгенерировать калибровку"))
        self._generate = generate
        self.generate_btn.clicked.connect(self._on_generate)
        root.addWidget(self.generate_btn)

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Самый короткий затвердевший столбик:")))
        self.result = QComboBox()
        for i in range(PILLARS):
            self.result.addItem(f"{i + 1}  ({heights[i]:.0f} {tr('мм')})", i + 1)
        self.result.addItem(tr("Ни один не затвердел"), 0)
        self.result.setCurrentIndex(PILLARS // 2)
        row.addWidget(self.result, 1)
        root.addLayout(row)

        self.apply_btn = QPushButton(tr("Применить"))
        self.apply_btn.clicked.connect(self._apply)
        root.addWidget(self.apply_btn)
        self._plan = self._load_plan()
        self.apply_btn.setEnabled(self._plan is not None)
        if self._plan is None:
            self.apply_btn.setToolTip(tr("Сначала сгенерируйте калибровку."))

    def _on_generate(self) -> None:
        self._generate()
        self.accept()

    def _load_plan(self) -> Optional[dict]:
        path = self._settings.value(SETTINGS_KEY, "")
        if not isinstance(path, str) or not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle).get("calibration")
        except (OSError, ValueError):
            return None

    def factor(self) -> float:
        assert self._plan is not None
        choice = int(self.result.currentData())
        return gel_dose_factor(self._plan, choice if choice > 0 else None)

    def _apply(self) -> None:
        if self._plan is None:
            return
        f = self.factor()
        old = float(self._gel_dose())
        if old > 0:
            self._set_gel_dose(old * f)
            text = tr("Доза гелеобразования: {old:.1f} → {new:.1f} мДж/см³. Время печати изменится в {f:.2f} раза.")
            text = text.format(old=old, new=old * f, f=f)
        else:
            text = tr("Доза гелеобразования не задана: умножьте время печати на {f:.2f}.").format(f=f)
        QMessageBox.information(self, tr("Калибровка экспозиции"), text)
        self.accept()
