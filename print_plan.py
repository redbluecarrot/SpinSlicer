"""Turn a relative dose plan into an exposure plan for a real print.

FastCAL dose D (Cartesian-pixel units, see optics.fastcal.VialDosePlanner)
relates to absorbed optical energy per volume as

    E [mJ/cm^3] = 1000 * alpha[1/mm] * I0[mW/mm^2] * t[s] * (w_det / h^2)[1/mm] * D[mm]

where I0 is the full-white irradiance at the vial axis, t the accumulated
display time of each frame (all turns together), w_det the width of one
detector bin at the axis plane and h the voxel pitch. Every attenuated
photon is assumed to be absorbed by the photoinitiator (scattering ignored).

Given the resin's gel (critical) dose Dc, the printed boundary sits where
E = Dc. Without a virtual print the plan puts Dc at the centre of the process
window: the dose gap between the dimmest in-part voxel and the brightest
out-of-part voxel. With one (virtual_print.py) the exposure is the one whose
cured part matches the model best, and the tolerance is the exposure range
that keeps the part's size within half a voxel.
"""
from __future__ import annotations

import math
from typing import Any, Optional


def compute_print_plan(stats: dict[str, float], *, alpha_per_mm: float, irradiance_mw_cm2: float,
                       gel_dose_mj_cm3: float, detector_bin_mm: float, voxel_pitch_mm: float,
                       frame_rate_hz: float, num_frames: int, coverage_radius_mm: float,
                       part_radius_mm: float, part_fraction_outside: float,
                       projector_contrast: float,
                       exposure: dict[str, Any] | None = None) -> dict[str, Any]:
    threshold = float(stats["threshold"])
    window = float(stats["window"])
    low, high = 1.0 - max(window, 0.0), 1.0 + max(window, 0.0)
    tolerance = max(0.0, 100.0 * window)
    if exposure is not None:
        # Exposure factor e on the window-centre dose: gel threshold at dose threshold / e.
        e = float(exposure["best_exposure"])
        threshold /= e
        low, high = float(exposure["exposure_low"]) / e, float(exposure["exposure_high"]) / e
        tolerance = float(exposure["tolerance_pct"])
    plan: dict[str, Any] = {
        "relative_threshold_dose": threshold,
        "process_window": window,
        "exposure_tolerance_pct": tolerance,
        "misprinted_volume_pct": 100.0 * float(stats["misprinted_fraction"]),
        "full_coverage_radius_mm": coverage_radius_mm,
        "part_radius_mm": part_radius_mm,
        "warnings": [],
    }
    warnings: list[str] = plan["warnings"]
    revolution_s = num_frames / frame_rate_hz
    plan["revolution_s"] = revolution_s
    if alpha_per_mm > 0 and irradiance_mw_cm2 > 0 and gel_dose_mj_cm3 > 0 and threshold > 0:
        per_second = 1000.0 * alpha_per_mm * (irradiance_mw_cm2 / 100.0) * (detector_bin_mm / voxel_pitch_mm**2)
        frame_time_s = gel_dose_mj_cm3 / (per_second * threshold)
        revolutions = frame_time_s * frame_rate_hz
        plan.update(
            frame_exposure_total_s=frame_time_s,
            revolutions=revolutions,
            print_time_s=revolutions * revolution_s,
            print_time_min_s=revolutions * revolution_s * low,
            print_time_max_s=revolutions * revolution_s * high,
        )
        if revolutions < 3:
            warnings.append(
                "Меньше 3 оборотов: ошибки старта/остановки и синхронизации станут "
                "заметными. Уменьшите яркость или мощность проектора."
            )
    else:
        warnings.append(
            "Время печати не рассчитано: задайте поглощение смолы (затухание), "
            "облучённость проектора и дозу гелеобразования смолы."
        )
    if window <= 0:
        warnings.append(
            "Нет экспозиции, при которой вся деталь печатается чисто (окно процесса "
            "отрицательное). Увеличьте число кадров или сетку либо используйте иммерсию."
        )
    if part_fraction_outside > 0.001:
        warnings.append(
            f"{100 * part_fraction_outside:.1f}% детали лежит дальше r = "
            f"{coverage_radius_mm:.1f} мм от оси: из-за преломления туда приходят лучи "
            "не со всех сторон. Сдвиньте деталь к оси или используйте иммерсию."
        )
    if projector_contrast > 0:
        black = float(stats["background_per_unit_black"]) / projector_contrast
        plan["black_level_dose_pct_of_threshold"] = 100.0 * black / threshold if threshold > 0 else math.inf
        if threshold > 0 and black / threshold > 0.2:
            warnings.append("Чёрный уровень проектора даёт больше 20% дозы гелеобразования по всему объёму.")
    return plan


def format_print_plan(plan: dict[str, Any], english: Optional[bool] = None) -> str:
    """Short human-readable summary for the log, in the interface language."""
    if english is None:
        try:
            from i18n import language
            english = language() == "en"
        except ImportError:  # pragma: no cover - i18n ships with the app
            english = False
    try:
        from i18n import tr
    except ImportError:  # pragma: no cover
        def tr(text: str) -> str:
            return text

    def t(ru: str, en: str) -> str:
        return en if english else ru

    lines = []
    if "print_time_s" in plan:
        lines.append(t(
            f"Время печати: {plan['print_time_s']:.1f} с ({plan['revolutions']:.1f} об., "
            f"оборот {plan['revolution_s']:.1f} с)",
            f"Print time: {plan['print_time_s']:.1f} s ({plan['revolutions']:.1f} turns, "
            f"{plan['revolution_s']:.1f} s per turn)"))
        if plan["exposure_tolerance_pct"] > 0:
            lines.append(t(
                f"Допуск экспозиции: ±{plan['exposure_tolerance_pct']:.1f}% "
                f"({plan['print_time_min_s']:.1f}–{plan['print_time_max_s']:.1f} с)",
                f"Exposure tolerance: ±{plan['exposure_tolerance_pct']:.1f}% "
                f"({plan['print_time_min_s']:.1f}–{plan['print_time_max_s']:.1f} s)"))
    else:
        lines.append(t(f"Окно процесса: {100 * plan['process_window']:+.1f}%",
                       f"Process window: {100 * plan['process_window']:+.1f}%"))
    if not plan.get("calibration"):
        lines.append(t(f"Ошибочно засвеченный объём (модель): {plan['misprinted_volume_pct']:.2f}%",
                       f"Wrongly cured volume (model): {plan['misprinted_volume_pct']:.2f}%"))
    cal = plan.get("calibration")
    if cal:
        rel = ", ".join(f"{i + 1}: {100 * r:.0f}%" for i, r in enumerate(cal["relative_dose"]))
        h0, h1 = cal["heights_mm"][0], cal["heights_mm"][-1]
        lines.append(t(
            f"Калибровочная печать: время рассчитано на столбик {cal['reference_pillar']}. "
            f"Доза столбиков (высота {h0:.0f}–{h1:.0f} мм): {rel}",
            f"Calibration print: the time is set for pillar {cal['reference_pillar']}. "
            f"Pillar doses (height {h0:.0f}–{h1:.0f} mm): {rel}"))
    vp = plan.get("virtual_print")
    if vp:
        verdicts = {"ok": ("✅ печатается с запасом", "✅ prints with margin"),
                    "tight": ("⚠ печатается, но экспозицию нужно попасть точно",
                              "⚠ prints, but the exposure has to be hit precisely"),
                    "bad": ("❌ деталь получится с ошибками", "❌ the part will come out with errors")}
        verdict = t(*verdicts.get(vp.get("verdict", ""), ("", "")))
        best = vp.get("best", {})
        extra, missing = best.get("extra_pct", 0), best.get("missing_pct", 0)
        lines.append(t(
            f"Виртуальная печать: {verdict}. Совпадение с моделью {100 * vp['iou']:.1f}% "
            f"(лишнее {extra:.2f}%, недосвет {missing:.2f}% объёма)",
            f"Virtual print: {verdict}. Match with the model {100 * vp['iou']:.1f}% "
            f"(extra {extra:.2f}%, under-cured {missing:.2f}% of the volume)"))
        lo, hi = vp.get("minus5", {}), vp.get("plus5", {})
        if lo and hi:
            lines.append(t(
                f"Если экспозиция ошибётся на 5%: деталь {lo['offset_mm']:+.2f} / {hi['offset_mm']:+.2f} мм "
                f"по поверхности (недосвет / пересвет)",
                f"If the exposure is off by 5%: the part changes by {lo['offset_mm']:+.2f} / "
                f"{hi['offset_mm']:+.2f} mm at the surface (under / over)"))
    for warning in plan["warnings"]:
        lines.append("⚠ " + (tr(warning) if english else warning))
    return "\n".join(lines)
