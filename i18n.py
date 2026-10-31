"""Small runtime language layer for the desktop UI.

The original prototype was written in Russian.  Russian remains the source
text so existing log messages and saved workflows stay readable; the default
application language is now Russian and this module translates the static Qt
surface without requiring a Qt Linguist build step.
"""
from __future__ import annotations

import re
from typing import Any

LANGUAGES = {"en": "English", "ru": "Русский"}
_language = "ru"

# Exact source-string translations.  Keys are deliberately the existing
# Russian UI strings so applying the English layer does not alter the Russian
# fallback or the public API.
_EN: dict[str, str] = {
    "Отсечение яркости кадров, %": "Frame gray cutoff, %",
    "Порог серого в PNG-кадрах; не критическая доза отверждения смолы.":
        "Gray cutoff in PNG frames; not the resin's critical curing dose.",
    "RMSE относительной дозы: —": "Relative dose RMSE: —",
    "RMSE относительной дозы:": "Relative dose RMSE:",
    "Ошибка формы дозы после подбора общего масштаба. Не прогноз отверждения смолы.":
        "Dose shape error after fitting a common scale. Not a resin curing prediction.",
    "Настройки процесса": "Process settings",
    "Настройки принтера и процесса": "Printer and process settings",
    "Настройки": "Settings",
    "Настройки принтера, процесса и альтернативного движка":
        "Printer, process, and alternate engine settings",
    "Печатная колба": "Print vat",
    "Диаметр колбы, мм": "Vat diameter, mm",
    "Параметры фотополимера": "Photopolymer parameters",
    "Базовое время засветки, с": "Base exposure time, s",
    "Интенсивность источника, %": "Light intensity, %",
    "Порог полимеризации, %": "Polymerization threshold, %",
    "Порог и интенсивность определяют итоговую\nконтрастность и яркость кадров.":
        "Threshold and intensity control the final\ncontrast and brightness of the frames.",
    "Разрешение и кадры": "Resolution and frames",
    "Разрешение сетки (Voxel Grid)": "Voxel grid resolution",
    "Разрешение кадра, px": "Frame resolution, px",
    "Количество кадров": "Number of frames",
    "Сплошная заливка (ремонт сетки)": "Fill holes (mesh repair)",
    "Preserve internal voids (bores / threads)": "Preserve internal voids (bores / threads)",
    "Объект": "Object",
    "Модель": "Model",
    "Файл не выбран": "No file selected",
    "Размер, мм": "Size, mm",
    "X, мм": "X, mm",
    "Y, мм": "Y, mm",
    "Z, мм": "Z, mm",
    "мм": "mm",
    "Вершин:": "Vertices:",
    "Граней:": "Faces:",
    "Исходные размеры:": "Original size:",
    "Трансформация": "Transform",
    "Инструмент:": "Tool:",
    "Перемещение": "Move",
    "Вращение": "Rotate",
    "Масштаб": "Scale",
    "Позиция": "Position",
    "Размер": "Size",
    "Тяни цветные стрелки, кольца или квадратные ручки прямо на модели.":
        "Drag the colored arrows, rings, or square handles directly on the model.",
    "Uniform Scale (сохранять пропорции)": "Uniform scale (keep proportions)",
    "Пропорционально": "Keep proportions",
    "Равномерное масштабирование": "Uniform scaling",
    "Сохранять пропорции при масштабировании": "Keep proportions while scaling",
    "Переместить модель в центр XY": "Move the model to the XY center",
    "Вписать модель в текущий размер колбы": "Fit the model to the current vat size",
    "Поворот, °": "Rotation, °",
    "Позиция, мм": "Position, mm",
    "Сдвиг X": "Offset X",
    "Сдвиг Y": "Offset Y",
    "Сдвиг Z": "Offset Z",
    "Центрировать": "Center",
    "Авто-фит под колбу": "Fit to vat",
    "Сбросить": "Reset",
    "📂 Загрузить STL": "📂 Load STL",
    "↺ Сбросить": "↺ Reset",
    "▶ Сгенерировать проекции": "▶ Generate projections",
    "📁 Открыть папку": "📁 Open output folder",
    "Настройка модели и генерация проекций": "Set up the model and generate projections",
    "Проигрывание и экспорт готовых кадров в MP4": "Play and export generated frames to MP4",
    "Обратная реконструкция геометрии по кадрам": "Reconstruct geometry from projection frames",
    "VAMToolbox — альтернативный движок": "VAMToolbox — alternate engine",
    "Скачать и установить VAMToolbox": "Download and install VAMToolbox",
    "VAMToolbox уже доступен": "VAMToolbox is already available",
    "Проверить окружение VAMToolbox": "Verify VAMToolbox environment",
    "🧊 Слайсер": "🧊 Slicer",
    "🎬 Проектор (Видео)": "🎬 Projector (Video)",
    "Язык:": "Language:",
    "Готово к работе.": "Ready.",
    "Готово к работе. Загрузите STL-модель на вкладке «Слайсер», чтобы начать.":
        "Ready. Load an STL model on the Slicer tab to begin.",
    "Папка кадров не выбрана — сначала сгенерируйте проекции на вкладке «Слайсер».":
        "No frame folder selected — generate projections on the Slicer tab first.",
    "Нажмите «Собрать и воспроизвести»,\nчтобы увидеть анимацию проекций.":
        "Click “Assemble and play”\nto preview the projection animation.",
    "▶ Собрать и воспроизвести": "▶ Assemble and play",
    "Прочитать все кадры из папки и запустить проигрывание":
        "Read all frames from the folder and start playback",
    "⏸ Пауза": "⏸ Pause",
    "💾 Сохранить в MP4": "💾 Save as MP4",
    "Экспортировать уже собранные кадры в видеофайл":
        "Export assembled frames to a video file",
    "Скорость воспроизведения": "Playback speed",
    "Открыть STL-файл модели": "Open an STL model file",
    "Сбросить трансформацию и заново вписать модель в колбу":
        "Reset the transform and fit the model to the vat again",
    "Запустить расчёт проекций в фоновом потоке":
        "Start projection generation in a background thread",
    "Открыть последнюю папку output_frames в проводнике":
        "Open the latest output_frames folder in Explorer",
    "Папка вывода ещё не создана.": "The output folder has not been created yet.",
    "Сначала загрузите STL-модель.": "Load an STL model first.",
    "Выбрать STL": "Choose an STL file",
    "STL файлы (*.stl)": "STL files (*.stl)",
    "Успех": "Success",
    "Ошибка": "Error",
    "Внимание": "Warning",
    "Projection engine": "Projection engine",
    "SpinSlicer internal Radon (recommended)":
        "SpinSlicer internal Radon (recommended)",
    "Auto: VAMToolbox CAL → SpinSlicer fallback (experimental)":
        "Auto: VAMToolbox CAL → SpinSlicer fallback (experimental)",
    "SpinSlicer internal Radon is the default engine. VAMToolbox is available only when explicitly selected.":
        "SpinSlicer internal Radon is the default engine. VAMToolbox is available only when explicitly selected.",
    "Auto: VAMToolbox CAL → Radon fallback": "Auto: VAMToolbox CAL → Radon fallback",
    "Internal Radon": "Internal Radon",
    "VAMToolbox CAL (optional)": "VAMToolbox CAL (optional)",
    "VAM optimizer iterations": "VAM optimizer iterations",
    "Auto uses VAMToolbox when installed and falls back to the internal "
    "Radon projector otherwise.":
        "Auto uses VAMToolbox when installed and falls back to the internal "
        "Radon projector otherwise.",
    "Перезапустите приложение, чтобы применить язык.":
        "Restart the application to apply the selected language.",
    "Язык изменён": "Language changed",
    "Parametric threaded nut": "Parametric threaded nut",

    # Missing strings from UI panels
    "VAMToolbox — альтернативный движок": "VAMToolbox — alternative engine",
    "Движок проекций": "Projection engine",
    "Оптический режим:": "Optical mode:",
    "🔬 Идеальный цилиндр (Radon, без преломления)": "🔬 Ideal cylinder (Radon, no refraction)",
    "🧊 Квадратная кювета (преломление по Снеллиусу)": "🧊 Square vat (Snell refraction)",
    "🧪 Виала в иммерсионной ванне (цилиндр + жидкость)": "🧪 Vial in immersion bath (cylinder + liquid)",
    "⚡ Встроенный Radon (быстрый, рекомендуется)": "⚡ Internal Radon (fast, recommended)",
    "📦 VAMToolbox CAL (внешний, опционально)": "📦 VAMToolbox CAL (external, optional)",
    "🔄 Auto: VAMToolbox → Radon fallback (экспериментально)": "🔄 Auto: VAMToolbox → Radon fallback (experimental)",
    "🔬 Физическая оптика + SIRT (точный, медленнее)": "🔬 Physical optics + SIRT (accurate, slower)",
    "Диаметр колбы (или внутр. сторона), мм": "Vat diameter (or inner side), mm",
    "Толщина стекла кюветы, мм": "Vat glass thickness, mm",
    "Показатель преломления (Стекло)": "Refractive index (Glass)",
    "Показатель преломления (Жидкость)": "Refractive index (Liquid)",
    "Показатель преломления (Смола)": "Refractive index (Resin)",
    "Затухание (Бугер-Ламберт) 1/мм": "Attenuation (Beer-Lambert) 1/mm",
    "Итерации оптимизатора (SIRT / VAM)": "Optimizer iterations (SIRT / VAM)",
    "Сбросить": "Reset",
    "Центрировать": "Center",
    "Авто-фит под колбу": "Auto-fit to vat",
    "Вписать модель в текущий размер колбы": "Fit model to current vat size",
    "Переместить модель в центр XY": "Move model to XY center",
    "Сохранять соотношение сторон модели": "Keep model aspect ratio",
    "Сохранять пропорции при масштабировании": "Keep proportions when scaling",
    "Слайсер (3D)": "Slicer (3D)",
    "Проектор (Кадры)": "Projector (Frames)",
    "Трансформация модели": "Model Transform",
    "Вращение, °": "Rotation, °",
    "Центр": "Center",
    "Авто-фит": "Auto-fit",
    "Сброс": "Reset",
    "Колба (без кюветы и жидкости)": "Flask (no cuvette/liquid)",
    "Виала (в иммерсионной ванне)": "Vial (in immersion bath)",
    "Движок проекций (Собственный C++ SIRT)": "Projection Engine (Native C++ SIRT)",
    "Движок проекций (FastCAL)": "Projection Engine (FastCAL)",
    "Проектор": "Projector",
    "Доза гелеобразования, мДж/см³ (0 = неизвестно)": "Gel dose, mJ/cm³ (0 = unknown)",
    "Поглощённая энергия на единицу объёма, при которой смола становится гелем. Вместе с затуханием и облучённостью даёт время печати.": "Absorbed energy per volume at which the resin gels. Together with attenuation and irradiance it gives the print time.",
    "Гамма проектора (1.0 = линейный)": "Projector gamma (1.0 = linear)",
    "Свет = (серый/255)^гамма. Обычный проектор в видеорежиме: около 2.2; DLP в режиме прямого вывода паттернов: 1.0.": "Light = (gray/255)^gamma. A typical projector in video mode: about 2.2; DLP in direct pattern mode: 1.0.",
    "Облучённость на оси колбы, мВт/см² (0 = неизвестно)": "Irradiance at the vial axis, mW/cm² (0 = unknown)",
    "Измеренная мощность белого кадра на месте колбы.": "Measured power of a white frame at the vial position.",
    "Контраст проектора, :1 (0 = идеальный чёрный)": "Projector contrast, :1 (0 = ideal black)",
    "Расстояние проектор → ось, мм (0 = телецентрический)": "Projector to axis distance, mm (0 = telecentric)",
    "Для обычного объектива лучи расходятся. 0 — параллельный пучок. В иммерсионной ванне указывайте кажущееся расстояние в жидкости.": "With an ordinary lens the rays diverge. 0 means a parallel beam. In an immersion tank enter the apparent distance in the liquid.",
    "FastCAL: лучи трассируются для одного ракурса, повороты считаются через FFT; "
    "проекции подбираются под порог отверждения, а не под точную дозу. "
    "Учитывает преломление света по закону Снеллиуса и поглощение по Бугеру-Ламберту.": (
        "FastCAL: rays are traced for one view and rotations are computed with FFT; "
        "projections are fitted to the cure threshold rather than to an exact dose. "
        "Accounts for Snell refraction and Beer-Lambert attenuation."
    ),
    "Итерации оптимизатора": "Optimizer iterations",
    "Реконструкция дозы (Tomo)": "Dose Reconstruction (Tomo)",
    "Реконструкция дозы": "Dose Reconstruction",
    "Средний срез (Mid-Slice)": "Mid-Slice",
    "Яркость проекции (Live Intensity)": "Projection Brightness (Live Intensity)",
    "Pepe CAL Lab (Easter Egg)": "Pepe CAL Lab (Easter Egg)",
    "Лаборатория Пепе 🐸⚡": "Pepe Lab 🐸⚡",
    "🎲 Другая цитата": "🎲 Another quote",
    "Нет данных\n(Сгенерируйте срез)": "No data\n(Generate slice)",
    "Потери дозы (L2): —\nДвижок: OpenCAL SIRT": "Dose loss (L2): —\nEngine: OpenCAL SIRT",
    "Потери дозы (L2 Loss):": "Dose loss (L2 Loss):",
    "Оптимизация: OpenCAL SIRT + Нелинейный отклик": "Optimization: OpenCAL SIRT + Non-linear response",
    "Потери дозы (L2):": "Dose loss (L2):",
    "Движок: OpenCAL SIRT": "Engine: OpenCAL SIRT",
    "Загрузить STL": "Load STL",
    "Сгенерировать проекции": "Generate projections",
    "Виртуальная печать": "Virtual print",
    "Калибровка": "Calibrate",
    "Калибровка экспозиции": "Exposure calibration",
    "Пробная печать, которая подберёт экспозицию под вашу смолу": "A test print that finds the exposure for your resin",
    "Сгенерировать калибровку": "Generate calibration",
    "Самый короткий затвердевший столбик:": "Shortest pillar that cured:",
    "Ни один не затвердел": "None cured",
    "Применить": "Apply",
    "мм": "mm",
    "Сначала сгенерируйте калибровку.": "Generate the calibration first.",
    "Калибровка работает с физическим движком и круглой колбой.": "Calibration needs the physical engine and a round vial.",
    "Напечатайте эти кадры, затем откройте «Калибровка» и укажите самый короткий затвердевший столбик.": "Print these frames, then open “Calibrate” and pick the shortest pillar that cured.",
    "Кнопка «Виртуальная печать» покажет деталь, которая затвердеет из этих кадров.": "The “Virtual print” button shows the part these frames would cure.",
    "Показать, что затвердеет из этих кадров (без принтера)": "Show what these frames would cure (no printer needed)",
    "Модель загружена.": "Model loaded.",
    "Сохранять внутренние полости (отверстия / резьба)": "Preserve internal cavities (holes / threads)",
    "Встроенный сверхбыстрый физический движок (SIRT + C++ OpenMP). ": "High-performance native physical engine (SIRT + C++ OpenMP). ",
    "Учитывает преломление света по закону Снеллиуса и поглощение по Бугеру-Ламберту.": "Accounts for Snell refraction and Beer-Lambert attenuation.",
    "Коэффициент оптического затухания света смолой (для CAL обычно 0.000 - 0.030 мм⁻¹)": "Resin optical attenuation coefficient (typically 0.000 - 0.030 mm⁻¹ for CAL)",
    "⚙ Настройки процесса": "⚙ Process settings",
    "Остановка...": "Stopping...",
    "⏹ Отменить генерацию": "⏹ Cancel generation",
    "▶ Воспроизведение": "▶ Play",
    "⏸ Пауза": "⏸ Pause",
    "Видео собрано.": "Video assembled.",
    "кадров для воспроизведения.": "frames for playback.",
    "Сборка видео не удалась.": "Video assembly failed.",
    "Модель не готова к слайсингу": "Model is not ready for slicing",
    "Ошибка генерации проекций": "Projection generation failed",
    "Сохранено": "Saved",
    "кадров в:": "frames to:",
    "Лаборатория Пепе 🐸": "Pepe Lab 🐸",
    "Режим:": "Mode:",
    "🧊 STL Модель": "🧊 STL Model",
    "🔮 3D Поле дозы": "🔮 3D Dose Field",
    "✨ Наложение (STL+Доза)": "✨ Hybrid (STL+Dose)",
    "Порог геля:": "Gel Threshold:",
    "Студия проектора и печатного процесса (HDMI Host)": "Projector Studio & Print Process (HDMI Host)",
    "📺 Выбор монитора / Проектора (HDMI)": "📺 Display / Projector Selection (HDMI)",
    "📺 Включить проектор": "📺 Enable Projector",
    "⏹ Выключить проектор": "⏹ Disable Projector",
    "🎯 Оптическая калибровка и фокус": "🎯 Optical Calibration & Focus",
    "🎯 Прицел и диаметр колбы": "🎯 Reticle & Vat Diameter",
    "🏁 Шахматка резкости фокуса": "🏁 Focus Sharpness Checkerboard",
    "💡 Белое поле (Интенсивность)": "💡 White Field (Intensity)",
    "🌑 Чёрное поле (Затемнение)": "🌑 Black Field (Blanking)",
    "🔌 Поворотный столик и УФ-затвор": "🔌 Turntable & UV Shutter",
    "⚪ Статус: Отключено (Доступен эмулятор)": "⚪ Status: Disconnected (Emulator Ready)",
    "🟢 Подключено:": "🟢 Connected:",
    "⚪ Статус: Отключено": "⚪ Status: Disconnected",
    "🌑 UV Затвор: ЗАКРЫТ": "🌑 UV Shutter: CLOSED",
    "🔆 UV Затвор: ОТКРЫТ": "🔆 UV Shutter: OPEN",
    "↷ Шаг 90°": "↷ Step 90°",
    "🏠 В ноль (0°)": "🏠 Home (0°)",
    "Физический угол столика: 0.0°": "Physical Turntable Angle: 0.0°",
    "Физический угол столика:": "Physical Turntable Angle:",
    "🚀 Запуск синхронизированной CAL 3D-печати": "🚀 Launch Synchronized CAL 3D Print",
    "Готовых кадров для проекции:": "Ready frames for projection:",
    "◑ Инвертировать цвет (Негатив)": "◑ Invert Colors (Negative)",
    "Готово к синхронной экспозиции.": "Ready for synchronized exposure.",
    "▶ НАЧАТЬ ОБЪЁМНУЮ ПЕЧАТЬ": "▶ START VOLUMETRIC PRINT",
    "⏸ Приостановить печать": "⏸ Pause Print",
    "⏹ Стоп": "⏹ Stop",
    "Идёт экспозиция:": "Exposing:",
    "🎉 Объёмная CAL печать успешно завершена!": "🎉 Volumetric CAL print successfully completed!",
    "Печать остановлена пользователем.": "Print stopped by user.",
    "Нажмите ESC или дважды кликните, чтобы выйти": "Press ESC or double-click to exit",
    "Библиотека согласования преломления:": "Refractive Index Matching Library:",
    "Пользовательский (Ручная настройка)": "Custom (Manual Setup)",
    "Боросиликат (Pyrex) + Глицерин + PEGDA (Δn < 0.002)": "Borosilicate (Pyrex) + Glycerol + PEGDA (Δn < 0.002)",
    "Кварц + Минеральное масло + Акрилат (Δn < 0.005)": "Quartz + Mineral Oil + Acrylate (Δn < 0.005)",
    "ПММА / Акрил + Пропиленгликоль + SLA (Δn ~ 0.06)": "PMMA / Acrylic + Propylene Glycol + SLA (Δn ~ 0.06)",
    "Стекло 1.51 + Вода 1.33 (Высокое преломление)": "Glass 1.51 + Water 1.33 (High Refraction)",
    "Фотокинетика и расчёт экспозиции": "Photokinetics & Exposure Calculation",
    "Критическая доза геля (Ec), мДж/см²": "Critical gel dose (Ec), mJ/cm²",
    "Мощность проектора (I₀), мВт/см²": "Projector irradiance (I₀), mW/cm²",
    "📺 Студия проектора (HDMI)": "📺 Projector Studio (HDMI)",
    "Прямой вывод на проектор, шаблоны фокусировки и аппаратный столик": "Direct projector output, focus patterns and turntable sync",
    "Основной экран": "Primary Screen",
    "Проектор / HDMI": "Projector / HDMI",
    "Оконный режим (Для тестирования)": "Windowed Mode (For Testing)",
    "Эмулятор (Virtual Turntable)": "Emulator (Virtual Turntable)",
    "Успех печати": "Print Success",
    "🔄 Обновить": "🔄 Refresh",
    "Подключить": "Connect",
    "Отключить": "Disconnect",
    "Монитор / Проектор (HDMI)": "Display / Projector (HDMI)",
    "Оптическая калибровка": "Optical Calibration",
    "Столик и UV-затвор": "Turntable + UV Shutter",
    "🚀 Синхронизированная CAL 3D-печать": "🚀 CAL 3D Print",
    "🎯 Прицел и диаметр колбы": "🎯 Reticle + Vat Diameter",
}

_RU: dict[str, str] = {
    "Projection engine": "Движок проекций",
    "SpinSlicer internal Radon (recommended)":
        "Внутренний Radon SpinSlicer (рекомендуется)",
    "Auto: VAMToolbox CAL → SpinSlicer fallback (experimental)":
        "Авто: VAMToolbox CAL → запасной SpinSlicer (экспериментально)",
    "SpinSlicer internal Radon is the default engine. VAMToolbox is available only when explicitly selected.":
        "Внутренний Radon SpinSlicer используется по умолчанию. VAMToolbox доступен только при явном выборе.",
    "Auto: VAMToolbox CAL → Radon fallback": "Авто: VAMToolbox CAL → запасной Radon",
    "Internal Radon": "Внутренний Radon",
    "VAMToolbox CAL (optional)": "VAMToolbox CAL (опционально)",
    "VAM optimizer iterations": "Итерации оптимизатора VAM",
    "Auto uses VAMToolbox when installed and falls back to the internal "
    "Radon projector otherwise.":
        "Авто использует VAMToolbox при наличии и иначе переключается на "
        "внутренний Radon-проектор.",
    "Preserve internal voids (bores / threads)":
        "Сохранять внутренние пустоты (отверстия / резьба)",
    "Generate a parametric nut with a helical internal thread":
        "Создать параметрическую гайку с винтовой внутренней резьбой",
    "Generate a watertight nut with a helical internal thread. "
    "Dimensions are in millimetres; clearance is radial.":
        "Создать герметичный корпус гайки с винтовой внутренней резьбой. "
        "Размеры указаны в миллиметрах; зазор — радиальный.",
    "Outer diameter, mm": "Наружный диаметр, мм",
    "Thread bore diameter, mm": "Диаметр резьбового отверстия, мм",
    "Nut height, mm": "Высота гайки, мм",
    "Thread pitch, mm": "Шаг резьбы, мм",
    "Thread depth (radial), mm": "Глубина резьбы (радиальная), мм",
    "Radial clearance, mm": "Радиальный зазор, мм",
    "Angular samples / turn": "Угловых отсчётов / оборот",
    "Axial samples / pitch": "Осевых отсчётов / шаг",
    "For a first print, keep at least 2–3 voxels across the thread depth "
    "and use a pitch larger than the voxel size.":
        "Для первой печати оставьте минимум 2–3 вокселя по глубине резьбы "
        "и задайте шаг больше размера вокселя.",
    "Create a threaded nut": "Создать гайку с резьбой",
    "Invalid nut parameters": "Некорректные параметры гайки",
    "Parametric threaded nut": "Параметрическая гайка с резьбой",
    "Load STL": "Загрузить STL",
    "Generate projections": "Сгенерировать проекции",
    "⏹ Cancel generation": "⏹ Отменить генерацию",
    "Model is not ready for slicing": "Модель не готова к слайсингу",
    "Projection generation failed": "Ошибка генерации проекций",
    "Dose Reconstruction": "Реконструкция дозы",
    "Mid-Slice": "Средний срез (Mid-Slice)",
    "Pepe Lab 🐸": "Лаборатория Пепе 🐸",
    "Pepe Lab 🐸⚡": "Лаборатория Пепе 🐸⚡",
    "🎲 Another quote": "🎲 Другая цитата",
    "No data\n(Generate slice)": "Нет данных\n(Сгенерируйте срез)",
    "Dose loss (L2): —\nEngine: OpenCAL SIRT": "Потери дозы (L2): —\nДвижок: OpenCAL SIRT",
    "Assemble and play": "Собрать и воспроизвести",
    "Pause": "Пауза",
    "Save as MP4": "Сохранить в MP4",
    "Playback speed": "Скорость воспроизведения",
    "Projection Brightness (Live Intensity)": "Яркость проекции (Live Intensity)",
}

# Auto-populate reverse mappings so English strings translate to Russian seamlessly
for ru_str, en_str in _EN.items():
    if en_str not in _RU:
        _RU[en_str] = ru_str

_PREFIX_EN: tuple[tuple[str, str], ...] = (
    ("Папка кадров: ", "Frames folder: "),
    ("Загружено: ", "Loaded: "),
    ("Сохранено ", "Saved "),
    ("Собрано ", "Assembled "),
    ("ОШИБКА", "ERROR"),
    ("Кадры готовы: ", "Frames ready: "),
    ("ОШИБКА генерации: ", "Generation ERROR: "),
    ("ОШИБКА загрузки: ", "Loading ERROR: "),
    ("ОШИБКА проверки модели: ", "Model check ERROR: "),
    ("Запуск генерации проекций...", "Starting projection generation..."),
    ("Загрузка модели...", "Loading model..."),
    ("Готово к работе. Загрузите STL-модель, чтобы начать.", "Ready. Load an STL model to begin."),
    ("Готово к работе.", "Ready."),
    ("Готово!", "Done!"),
    ("Отменено пользователем.", "Cancelled by user."),
    ("Остановка...", "Stopping..."),
    ("Генерация отменена пользователем.", "Generation cancelled by user."),
)

_PREFIX_RU: tuple[tuple[str, str], ...] = (
    ("Frames folder: ", "Папка кадров: "),
    ("Loaded: ", "Загружено: "),
    ("Saved ", "Сохранено "),
    ("Assembled ", "Собрано "),
    ("ERROR: ", "ОШИБКА: "),
    ("ERROR", "ОШИБКА"),
    ("Frames ready: ", "Кадры готовы: "),
    ("Starting projection generation...", "Запуск генерации проекций..."),
    ("Loading model...", "Загрузка модели..."),
    ("Ready. Load an STL model to begin.", "Готово к работе. Загрузите STL-модель, чтобы начать."),
    ("Ready.", "Готово к работе."),
    ("Done!", "Готово!"),
    ("Cancelled by user.", "Отменено пользователем."),
    ("Stopping...", "Остановка..."),
    ("Generation cancelled by user.", "Генерация отменена пользователем."),
)


# --- New interface (2026-10): exact strings -------------------------------------------
_EN.update({
    "Принтер и смола": "Printer and resin", "Свет": "Light", "Печать": "Print", "Модель": "Model",
    "НАСТРОЙКА · ОДИН РАЗ": "SETUP · ONCE",
    "показать ▾": "show ▾", "скрыть ▴": "hide ▴", "мм": "mm", "мВт/см²": "mW/cm²", "мм⁻¹": "mm⁻¹",
    "мДж/см³": "mJ/cm³", "°/с": "°/s",
    "Диаметр колбы": "Vial diameter", "Стенка": "Wall", "Колба стоит": "The vial sits",
    "Высота поля": "Field height", "Разрешение проектора": "Projector resolution",
    "Колба на экране": "Vial on screen", "Лучи проектора": "Projector rays", "Размытие пятна": "Spot blur",
    "Поглощение смолы": "Resin absorption", "Преломление смолы": "Resin refraction", "Доза геля": "Gel dose",
    "Калибровка": "Calibration", "Сдвиг картинки по X": "Picture shift X", "Сдвиг картинки по Y": "Picture shift Y",
    "Гамма": "Gamma", "Контраст": "Contrast", "Облучённость": "Irradiance", "n стекла": "Glass n",
    "n жидкости ванны": "Bath liquid n", "Подсказки маркером": "Marker hints",
    "для тех, кто печатает впервые": "for first-time printers", "Язык": "Language",
    "загрузить STL": "load STL", "Модель не загружена": "No model loaded", "STL-файл детали": "the part's STL file",
    "Масштаб": "Scale", "Ориентация": "Orientation", "Отверстия и резьба": "Holes and threads",
    "Положение": "Position", "Точная позиция": "Exact position", "Качество": "Quality",
    "Кадров на оборот": "Frames per turn", "Компенсация размытия": "Blur compensation",
    "Виртуальная печать": "Virtual print", "показать на сцене, что затвердеет": "show on the scene what will cure",
    "совпадение": "match", "запас экспоз.": "exposure margin", "печать": "print",
    "Пока считается — засветите деталь сами": "While it computes — cure the part yourself",
    "Свет готов. Доиграйте или идите печатать": "Light is ready. Finish the round or go print",
    "Куда выводить": "Output to", "Скорость вращения": "Rotation speed", "Настроить проектор": "Set up the projector",
    "разметка покажет, где должны быть стенки колбы; тест серого — все уровни яркости":
        "the pattern shows where the vial walls must be; the gray test shows every brightness level",
    "Сначала рассчитайте свет.": "Compute the light first.",
    "двигает картинку вверх/вниз, чтобы деталь попала в колбу": "moves the picture up/down so the part lands in the vial",
    "преломление на стенке колбы": "refraction at the vial wall", "преломление на стенке ванны": "refraction at the bath wall",
    "Опишите свой принтер и смолу. Это делается один раз — дальше только детали.":
        "Describe your printer and resin. You do this once — after that, just parts.",
    "Покрутите деталь мышью и проверьте, как она стоит. Если всё ок — дальше.":
        "Turn the part with the mouse and check how it sits. If it looks right — next.",
    "Выберите качество и нажмите «Рассчитать свет».": "Pick the quality and press “Compute light”.",
    "Загрузите STL: перетащите файл на сцену слева или нажмите кнопку внизу.":
        "Load an STL: drop the file on the scene or press the button below.",
    "Деталь не помещается — уменьшите масштаб или нажмите «вписать в колбу».":
        "The part does not fit — scale it down or press “fit into vial”.",
    "Считаю кадры для каждого угла. Можно пока поиграть ↓": "Computing frames for every angle. Play meanwhile ↓",
    "Готово. Посмотрите совпадение и запас — и переходите к печати.":
        "Done. Check the match and the margin — then go to printing.",
    "Проверьте разметку колбы на проекторе и запускайте.": "Check the vial pattern on the projector and start.",
    "сколько пикселей проектора по ширине занимает колба — проверьте разметкой":
        "how many projector pixels the vial spans — check with the pattern",
    "не знаете? напечатайте шкалу ↓": "not sure? print the ladder ↓", "напечатать шкалу →": "print the ladder →",
    "ниже деталь — меньше слоёв и точнее": "a lower part means fewer layers and more accuracy",
    "в центр": "to centre", "вписать в колбу": "fit into vial", "для первой печати хватит «Норма»": "“Normal” is enough for a first print",
    "разметка колбы": "vial pattern", "тест серого": "gray test",
    "движок трассирует расходящиеся лучи (как у обычного объектива)": "the engine traces diverging rays (a normal lens)",
    "телецентрический объектив или проектор далеко": "telecentric lens or a far-away projector",
    "проектор резкий — кадры как есть": "sharp projector — frames as they are",
    "время печати считается из неё": "print time is computed from it",
    "из неё и мощности проектора считается время печати": "print time comes from it and the projector power",
    "без подрезки мелкие детали размоет": "without sharpening fine details blur",
    "Готово — к модели": "Done — to the model", "всё сохраняется само": "everything saves itself",
    "нет": "none", "ошибиться сложно": "hard to get wrong", "нужны доза и мощность": "needs dose and power",
    "почти идеально": "nearly perfect", "можно точнее": "could be sharper", "нужна калибровка": "calibrate",
    "увеличьте сетку": "raise the grid", "Время засветки": "Exposure time", "Обороты": "Revolutions",
    "Время": "Time", "задайте поглощение, облучённость и дозу": "set absorption, irradiance and dose",
    "Запас экспозиции": "Exposure margin", "Частота кадров": "Frame rate", "заменить": "replace",
    "или перетащите файл на сцену": "or drop the file on the scene", "Сначала уместите деталь": "Make the part fit first",
    "уменьшите масштаб или возьмите колбу шире": "scale it down or use a wider vial", "Дальше — свет →": "Next — light →",
    "проверка модели пройдена": "model check passed", "Остановить": "Stop", "считаю кадры…": "computing frames…",
    "сначала рассчитайте свет": "compute the light first", "Дальше — печать →": "Next — print →",
    "Рассчитать свет": "Compute light", "Показать на проекторе": "Show on projector", "Сохранить видео": "Save video",
    "Открыть папку кадров": "Open frames folder", "Esc на проекторе — остановить": "Esc on the projector stops it",
    "кадры готовы": "frames ready", "сначала загрузите модель": "load a model first",
    "полноэкранно на второй экран, пиксель в пиксель": "full screen on the second display, pixel for pixel",
    "папка PNG для своего софта показа": "a PNG folder for your own player",
    "в ванне": "in a bath", "в воздухе": "in air", "параллельные": "parallel", "конусом": "cone",
    "Колба": "Vial", "Проектор": "Projector", "Смола": "Resin", "Ещё": "More", "Для специалистов": "For experts",
    "как в файле": "as in file", "на бок": "on its side", "на торец": "on its end",
    "Быстро": "Fast", "Норма": "Normal", "Точно": "Fine",
    "на проектор": "projector", "видео MP4": "MP4 video", "кадры PNG": "PNG frames",
    "Мини-игра «Засветка»": "Mini-games", "Мини-игры": "Mini-games",
    "Новый проект": "New project", "модель не загружена": "no model loaded",
    "Принтер и смола — настраиваются один раз": "Printer and resin — set up once",
    "журнал ▴": "log ▴", "журнал ▾": "log ▾", "Принтер": "Printer", "зона": "zone", "воксель": "voxel",
    "кадров": "frames", "запас": "margin", "День / ночь": "Day / night",
    "Перетащите STL сюда": "Drop an STL here", "файл детали в формате STL, в миллиметрах": "the part as an STL file, in millimetres",
    "Выбрать файл": "Choose file", "Кадры появятся после расчёта света": "Frames appear after the light is computed",
    "Загрузка кадров...": "Loading frames...", "Экспорт в MP4...": "Exporting MP4...",
    "Экспорт тестовой шкалы...": "Exporting the gray test...", "Видео сохранено.": "Video saved.",
    "Экспорт не удался.": "Export failed.", "Экспорт не удался": "Export failed", "Экран проектора": "Projector screen",
    "Загружаю кадры…": "Loading frames…", "Не удалось прочитать кадры": "Could not read the frames",
    "Кадры выводятся на второй экран. Esc — остановить.": "Frames go to the second screen. Esc stops.",
    "Готово": "Done", "Видео сохранено": "Video saved", "Сохранить видео ": "Save video",
    "Второй экран не найден. Подключите проектор как второй монитор (расширенный рабочий стол) или сохраните видео.":
        "No second screen found. Connect the projector as a second monitor (extended desktop) or save a video.",
    "Показывайте его на проекторе во весь экран, без масштабирования.": "Play it full screen on the projector, without scaling.",
    "Сначала рассчитайте свет.": "Compute the light first.", "Проектор": "Projector",
    "Создаю тестовую гайку M20...": "Building the M20 test nut...", "модель": "model",
    "мышь — двигать, клик — старт": "mouse — move, click — start",
    "Засветка": "Cure", "Слои": "Layers", "Фотон": "Photon",
    "кирпичи — силуэт вашей детали; фотон засвечивает воксели": "bricks are your part's silhouette; the photon cures voxels",
    "укладывайте воксели в колбу без щелей": "stack voxels in the vial without gaps",
    "проведите фотон через стенки, не задев стекло": "guide the photon through the walls without touching the glass",
    "←/→ — двигать, ↑ — повернуть, ↓ — быстрее": "←/→ move, ↑ rotate, ↓ drop",
    "клик или пробел — подлететь вверх": "click or Space — fly up",
    "появится после расчёта света": "appears after the light is computed",
    "результат устарел — пересчитайте свет": "result is out of date — compute the light again",
    "показать в 3D, что затвердеет": "show in 3D what will cure",
})

# Phrases with numbers: the numbers are matched in any form and put back in place.
_EN_TEMPLATES: dict[str, str] = {
    "ШАГ {} ИЗ {}": "STEP {} OF {}",
    "зона печати Ø {} мм · воксель {} мм": "print zone Ø {} mm · voxel {} mm",
    "внутри Ø {} мм — столько занимает смола": "inside Ø {} mm — that much is resin",
    "в жидкости n = {} луч почти не гнётся: кадры без искажений": "in liquid n = {} the beam hardly bends: frames are undistorted",
    "в воздухе колба работает как линза: кадры специально растянуты, печатается только центр Ø {} мм. Ванна лучше.":
        "in air the vial acts as a lens: frames are stretched on purpose and only the centre Ø {} mm prints. A bath is better.",
    "печатать можно до {} мм в высоту": "parts up to {} mm tall",
    "печатать можно до {} мм в высоту (ограничивает колба {} мм)": "parts up to {} mm tall (the vial limits it to {} mm)",
    "видео и вывод на проектор — {}×{}, пиксель в пиксель": "video and projector output — {}×{}, pixel for pixel",
    "пиксель {} мм на оси колбы": "pixel {} mm at the vial axis",
    "пиксель {} мм на оси колбы — шире экрана проектора!": "pixel {} mm at the vial axis — wider than the projector screen!",
    "до оси ≈ {} мм · по высоте у краёв детали размытие до {} мм": "to the axis ≈ {} mm · top and bottom of the part blur up to {} mm",
    "до оси ≈ {} мм · по высоте у краёв детали размытие до {} мм — поставьте проектор дальше":
        "to the axis ≈ {} mm · top and bottom of the part blur up to {} mm — move the projector farther",
    "до оси {} мм — ближе колбы, так не бывает: проверьте числа": "to the axis {} mm — closer than the vial, impossible: check the numbers",
    "кадры подрезкиваются под размытие {} мм": "frames are pre-sharpened for {} mm blur",
    "картинка на экране: x = {} px": "picture on screen: x = {} px",
    "серый → свет по степени {}; кадры пересчитываются": "gray → light with power {}; frames are converted",
    "{} — идеальный чёрный": "{} — ideal black",
    "чёрный пиксель светит {} % от белого": "a black pixel gives {} % of white",
    "{} — время печати не считается": "{} — print time is not computed",
    "{} — свет не слабеет, время печати не считается": "{} — light does not fade, print time is not computed",
    "до дальней стенки доходит {} % света": "{} % of the light reaches the far wall",
    "луч гнётся на стенке на {}° (с ванной)": "the beam bends {}° at the wall (with bath)",
    "луч гнётся на стенке на {}° (без ванны)": "the beam bends {}° at the wall (no bath)",
    "колба в воздухе — луч гнётся на стенке на {}°, движок это учтёт": "vial in air — the beam bends {}° at the wall, the engine accounts for it",
    "{} — неизвестно, время печати не считается": "{} — unknown, print time is not computed",
    "сетка {} · воксель {} мм · расчёт ≈ {} с": "grid {} · voxel {} mm · compute ≈ {} s",
    "сетка {} · воксель {} мм · расчёт ≈ {}:{}": "grid {} · voxel {} mm · compute ≈ {}:{}",
    "шаг {}° между кадрами": "{}° between frames",
    "подрезка кадров под {} мм": "frames pre-sharpened for {} mm",
    "кадр каждые {} мс · проектор {} Гц": "a frame every {} ms · projector {} Hz",
    "кадр каждые {} мс · проектор {} Гц — больше {} Гц, проверьте проектор": "a frame every {} ms · projector {} Hz — over {} Hz, check the projector",
    "файл {}×{} — показывать во весь экран без масштабирования": "a {}×{} file — play full screen without scaling",
    "{} с": "{} s", "{} оборота": "{} revolutions", "{} Гц": "{} Hz", "{} мм": "{} mm",
    "{} × {} × {} мм": "{} × {} × {} mm",
    "{} × {} × {} мм · {} треугольников": "{} × {} × {} mm · {} triangles",
    "по высоте {} мм из {}": "height {} mm of {}",
    "готово · {} кадров": "done · {} frames",
    "≈ {} с на обычном компьютере": "≈ {} s on an ordinary computer",
    "≈ {}:{} на обычном компьютере": "≈ {}:{} on an ordinary computer",
    "Ø {} × {} мм": "Ø {} × {} mm", "{} · шаг {}°": "{} · step {}°",
    "Ø {} мм · {}×{}": "Ø {} mm · {}×{}", "Ø {} мм · {}×{} · ✓": "Ø {} mm · {}×{} · ✓",
    "{}×{} · кадр {}×{} · {} Гц": "{}×{} · frame {}×{} · {} Hz",
    "{}×{} · кадр {}×{} — не помещается! · {} Гц": "{}×{} · frame {}×{} — does not fit! · {} Hz",
    "{}×{} · {} Гц": "{}×{} · {} Hz",
    "Влезает: {} мм из {}, по высоте {} из {}": "Fits: {} mm of {}, height {} of {}",
    "Влезает: {} мм из {}, по высоте {} из {} — у самой стенки, лучше ближе к оси":
        "Fits: {} mm of {}, height {} of {} — right at the wall, better closer to the axis",
    "Шире колбы: деталь уходит на {} мм от оси, внутри {}": "Wider than the vial: the part reaches {} mm from the axis, inside is {}",
    "Выше поля: {} мм при доступных {}": "Taller than the field: {} mm, available {}",
    "{} столбиков с дозой {}–{} %: выбираете лучший, доза обновится сама": "{} pillars at {}–{} % dose: pick the best one, the dose updates itself",
    "или открыть тестовую гайку M{}": "or open the M{} test nut",
    "есть в паспорте проектора, обычно {},{}–{},{}": "in the projector datasheet, usually {}.{}–{}.{}",
    "есть в паспорте проектора, обычно {}–{}": "in the projector datasheet, usually {}–{}",
    "не знаете? оставьте {}": "not sure? leave {}",
    "Сначала рассчитайте свет на шаге {}.": "Compute the light in step {} first.",
    "BEST {}": "BEST {}",
}
_EN.update({
    "Готово.": "Done.", "Компенсация размытия проектора...": "Pre-sharpening for projector blur...",
    "Нарезка сечений...": "Slicing layers...", "План печати...": "Print plan...", "Подготовка модели...": "Preparing the model...",
    "Сохранение кадров...": "Saving frames...", "Физическая оптика: трассировка лучей и оптимизация...":
        "Physical optics: ray tracing and optimisation...",
    "FastCAL: трассировка лучей одного ракурса...": "FastCAL: tracing rays for one view...",
    "ffmpeg недоступен: MP4 пишется кодеком mp4v с потерей качества.": "ffmpeg missing: MP4 is written with the lossy mp4v codec.",
    "Видео сохранено.": "Video saved.", "Загрузка кадров...": "Loading frames...", "Модель загружена.": "Model loaded.",
    "Загрузка модели...": "Loading model...", "Создаю тестовую гайку M20...": "Building the M20 test nut...",
    "Ошибка загрузки.": "Loading failed.", "Ошибка генерации.": "Computation failed.", "Видео собрано.": "Frames loaded.",
    "Процесс расчёта завершился с кодом": "The compute process exited with code",
    "Время печати не рассчитано: задайте поглощение смолы (затухание), облучённость проектора и дозу гелеобразования смолы.":
        "Print time not computed: set resin absorption, projector irradiance and the resin gel dose.",
    "Нет экспозиции, при которой вся деталь печатается чисто (окно процесса отрицательное). Увеличьте число кадров или сетку либо используйте иммерсию.":
        "No exposure prints the whole part cleanly (negative process window). Use more frames, a finer grid or immersion.",
})
_EN_TEMPLATES.update({
    "Меньше {} оборотов: ошибки старта/остановки и синхронизации станут заметными. Уменьшите яркость или мощность проектора.":
        "Fewer than {} revolutions: start/stop and sync errors become visible. Lower the projector brightness or power.",
    "{}% детали лежит дальше r = {} мм от оси: из-за преломления туда приходят лучи не со всех сторон. "
    "Сдвиньте деталь к оси или используйте иммерсию.":
        "{}% of the part lies beyond r = {} mm from the axis: refraction keeps some directions from reaching it. "
        "Move the part towards the axis or use immersion.",
    "Чёрный уровень проектора даёт больше {}% дозы гелеобразования по всему объёму.":
        "The projector's black level gives over {}% of the gel dose everywhere.",
    "Загрузка кадра {}/{}...": "Loading frame {}/{}...", "Кодирование кадра {}/{}...": "Encoding frame {}/{}...",
    "Открытая сетка: {} сечений восстановлено.": "Open mesh: {} layers repaired.",
    "Сохранено {} кадров в: ": "Saved {} frames to: ",
})
_EN.update({
    "Доза в смоле": "Dose in the resin", "средний слой детали": "middle layer of the part",
    "розовое — затвердеет · пунктир — модель": "bright — will cure · dashed — the model",
    "экспозиция по плану": "exposure as planned", "нет данных": "no data",
    "летите сквозь свет вашей печати": "fly through the light of your print",
})
_EN_TEMPLATES.update({
    "экспозиция +{} %": "exposure +{} %", "экспозиция -{} %": "exposure -{} %",
    "не досвечено {} % · лишнее {} %": "under-cured {} % · extra {} %",
    "ошибка формы дозы {}": "dose shape error {}",
})
_EN.update({"Сдвиг, мм": "Move, mm", "Поворот, °": "Rotate, °", "Размер, мм": "Size, mm", "Центр": "Centre",
            "Вписать": "Fit", "Сброс": "Reset"})

_NUMBER = re.compile(r"\d+(?:[  ]\d{3})*(?:[.,]\d+)?")


def _template_en(text: str) -> str | None:
    key = _NUMBER.sub("{}", text)
    if key == text or key not in _EN_TEMPLATES:
        return None
    numbers = [n.replace(",", ".") for n in _NUMBER.findall(text)]
    try:
        return _EN_TEMPLATES[key].format(*numbers)
    except (IndexError, KeyError):
        return None


def set_language(language: str) -> None:
    """Select ``en`` or ``ru`` for subsequently created/updated UI text."""

    global _language
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    _language = language


def language() -> str:
    return _language


_SOURCE: dict[str, str] = {}      # shown text -> the source string it was translated from


def remember(shown: str, source: str) -> None:
    """Record that `shown` on screen came from `source` (for switching language later)."""
    if shown and shown != source and _SOURCE.get(source) != shown:
        if len(_SOURCE) > 50_000:          # numbers make templated strings unique; keep it bounded
            _SOURCE.clear()
        _SOURCE[shown] = source


def source_of(text: str) -> str:
    """The source string behind a text currently on screen (itself if unknown)."""
    seen = 0
    while text in _SOURCE and seen < 3:     # a translation of a translation, at worst
        text = _SOURCE[text]
        seen += 1
    return text


def _translate(text: str) -> str:
    if _language == "ru":
        if text in _RU:
            return _RU[text]
        for source, translated in _PREFIX_RU:
            if text.startswith(source):
                return translated + text[len(source):]
        return text
    if text in _EN:
        return _EN[text]
    templated = _template_en(text)
    if templated is not None:
        return templated
    for source, translated in _PREFIX_EN:
        if text.startswith(source):
            return translated + text[len(source):]
    return text


def tr(text: str) -> str:
    """Translate a source string between Russian and English seamlessly.

    Every translation is remembered, so a text already on screen can be traced back
    to its source and translated again when the language changes.
    """
    if not text:
        return text
    source = source_of(text)
    out = _translate(source)
    remember(out, source)
    return out


def _retranslated(text: str) -> str:
    if not text or "<" in text:          # rich text is rebuilt by its owner (chips, plan table)
        return text
    return tr(source_of(text))


def apply_translations(root: Any) -> None:
    """Translate every text of a window into the current language.

    Works from whatever is on screen now: each text is traced back to its source
    (see remember/source_of) and translated again, so switching back and forth any
    number of times never leaves a stale or half-translated label. Dynamic texts
    (effects, plans, chips) are additionally rebuilt by their owners afterwards.
    """
    from PyQt6.QtCore import QObject
    from PyQt6.QtWidgets import QAbstractButton, QComboBox, QGroupBox, QLabel, QTabWidget

    widgets = [root, *root.findChildren(QObject)]
    for widget in widgets:
        if not hasattr(widget, "property"):
            continue
        if isinstance(widget, QLabel):
            if widget.pixmap() is not None and not widget.pixmap().isNull():
                continue
            if widget.property("_spinslicer_fixed_text"):
                continue
            new = _retranslated(widget.text())
            if widget.property("_i18n_upper"):
                new = tr(str(widget.property("_i18n_upper"))).upper()
            if new != widget.text():
                widget.setText(new)
        elif isinstance(widget, QAbstractButton):
            new = _retranslated(widget.text())
            if new != widget.text():
                widget.setText(new)
        elif isinstance(widget, QGroupBox):
            widget.setTitle(_retranslated(widget.title()))

        for prop in ("windowTitle", "toolTip", "statusTip", "whatsThis"):
            getter = getattr(widget, prop, None)
            setter = getattr(widget, f"set{prop[0].upper()}{prop[1:]}", None)
            if not callable(getter) or not callable(setter):
                continue
            current = getter()
            if current:
                new = _retranslated(str(current))
                if new != current:
                    setter(new)

        if isinstance(widget, QComboBox):
            for index in range(widget.count()):
                widget.setItemText(index, _retranslated(widget.itemText(index)))

        if isinstance(widget, QTabWidget):
            for index in range(widget.count()):
                widget.setTabText(index, _retranslated(widget.tabText(index)))
                tip = widget.tabToolTip(index)
                if tip:
                    widget.setTabToolTip(index, _retranslated(tip))


# Print page: brightness, exposure, timed projection; log details (2026-10-31).
_EN.update({
    "Яркость проектора": "Projector brightness",
    "полная яркость — самая короткая печать": "full brightness — the shortest print",
    "Остановить засветку": "Stop exposure",
    "идёт засветка · Esc на проекторе — стоп": "exposing · Esc on the projector stops",
    "Экспозиция": "Exposure", "Яркость": "Brightness",
    "Засветка завершена: доза набрана.": "Exposure finished: the dose is delivered.",
    "Загружено:": "Loaded:", "верш.": "vertices", "гран.": "faces",
    "Preflight отчёт:": "Preflight report:",
    "Сохранено": "Saved", "кадров в:": "frames to:",
})
_EN_TEMPLATES.update({
    "кадры в {} раза тусклее, засветка во столько же раз дольше — доза та же; полезно для быстрой смолы":
        "frames {}× dimmer, exposure that much longer — same dose; useful for fast resin",
    "идёт засветка {} с из {}:{}": "exposing {} s of {}:{}",
    "идёт засветка {}:{} из {}:{}": "exposing {}:{} of {}:{}",
    "идёт засветка {} с из {} с": "exposing {} s of {} s",
    "идёт засветка {}:{} из {} с": "exposing {}:{} of {} s",
    "{} % от плана": "{} % of plan",
    "Сохранение кадра {}/{}...": "Saving frame {}/{}...",
})


def num(value: float, digits: int = 1) -> str:
    """A number with the interface's decimal separator (comma in Russian, point in English)."""
    text = f"{value:.{digits}f}"
    return text.replace(".", ",") if _language == "ru" else text

_EN.update({
    "Настройки или деталь изменились — кадры устарели. Пересчитайте свет.":
        "Settings or the part changed — the frames are out of date. Compute the light again.",
    "Пересчитать свет": "Compute light again",
    "старые кадры не подходят к новым настройкам": "the old frames do not match the new settings",
})
