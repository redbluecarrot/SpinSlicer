"""Small runtime language layer for the desktop UI.

The original prototype was written in Russian.  Russian remains the source
text so existing log messages and saved workflows stay readable; the default
application language is now Russian and this module translates the static Qt
surface without requiring a Qt Linguist build step.
"""
from __future__ import annotations

from typing import Any

LANGUAGES = {"en": "English", "ru": "Русский"}
_language = "ru"

# Exact source-string translations.  Keys are deliberately the existing
# Russian UI strings so applying the English layer does not alter the Russian
# fallback or the public API.
_EN: dict[str, str] = {
    "Настройки процесса": "Process settings",
    "Настройки принтера и процесса": "Printer and process settings",
    "Дополнительные настройки процесса": "Advanced process settings",
    "Быстрые настройки": "Quick settings",
    "Настройки": "Settings",
    "Настройки принтера, процесса и альтернативного движка":
        "Printer, process, and alternate engine settings",
    "Печатная колба": "Print vat",
    "Оптические параметры колбы": "Vat optical properties",
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
    "Сохранять полости": "Preserve internal cavities",
    "Сохраняет отверстия и резьбу при растеризации STL-сетки.":
        "Preserves holes and threads when rasterizing the STL mesh.",
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
    "Вернуть исходное положение и масштаб модели":
        "Restore the model's original position and scale",
    "Сброс камеры": "Reset camera",
    "Вернуть камеру к исходному изометрическому виду":
        "Return the camera to the default isometric view",
    "Домашнее положение камеры": "Home camera view",
    "ЛКМ — вращение · ПКМ — панорама · колёсико — масштаб":
        "Left-drag to orbit · right-drag to pan · wheel to zoom",
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
    "Сначала выберите папку кадров (output_frames).":
        "Select a frame folder first (output_frames).",
    "Сначала соберите видео.": "Assemble the video first.",
    "Сборка видео отменена.": "Video assembly cancelled.",
    "Экспорт видео отменён.": "Video export cancelled.",
    "Сохранить видео": "Save video",
    "Видео MP4 (*.mp4)": "MP4 video (*.mp4)",
    "Видео сохранено.": "Video saved.",
    "Видео сохранено:": "Video saved:",
    "Ошибка экспорта видео.": "Video export failed.",
    "Не удалось прочитать статистику реконструкции.":
        "Could not read reconstruction statistics.",
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
    "Толщина стенки колбы, мм": "Vial wall thickness, mm",
    "Показатель преломления (Стекло)": "Refractive index (Glass)",
    "Показатель преломления (Жидкость)": "Refractive index (Liquid)",
    "Показатель преломления (Смола)": "Refractive index (Resin)",
    "Затухание (Бугер-Ламберт) 1/мм": "Attenuation (Beer-Lambert) 1/mm",
    "Итерации оптимизатора (SIRT / VAM)": "Optimizer iterations (SIRT / VAM)",
    "Сохранять соотношение сторон модели": "Keep model aspect ratio",
    "Слайсер (3D)": "Slicer (3D)",
    "Проектор (Кадры)": "Projector (Frames)",
    "Трансформация модели": "Model Transform",
    "Вращение, °": "Rotation, °",
    "Центр": "Center",
    "Авто-фит": "Auto-fit",
    "Сброс": "Reset",
    "Колба (без кюветы и жидкости)": "Flask (no cuvette/liquid)",
    "Колба (в воздухе)": "Vial (in air)",
    "Виала (в иммерсионной ванне)": "Vial (in immersion bath)",
    "Движок проекций (Собственный C++ SIRT)": "Projection Engine (Native C++ SIRT)",
    "Итерации оптимизатора": "Optimizer iterations",
    "Реконструкция дозы (Tomo)": "Dose Reconstruction (Tomo)",
    "Реконструкция дозы": "Dose Reconstruction",
    "Средний срез (Mid-Slice)": "Mid-Slice",
    "Яркость проекции (Live Intensity)": "Projection Brightness (Live Intensity)",
    "Нет данных\n(Сгенерируйте срез)": "No data\n(Generate slice)",
    "Потери дозы (L2): —\nДвижок: OpenCAL SIRT": "Dose loss (L2): —\nEngine: OpenCAL SIRT",
    "Потери дозы (L2 Loss):": "Dose loss (L2 Loss):",
    "Ошибка дозы (RMSE):": "Dose error (RMSE):",
    "Оптимизация: OpenCAL SIRT + Нелинейный отклик": "Optimization: OpenCAL SIRT + Non-linear response",
    "Потери дозы (L2):": "Dose loss (L2):",
    "Движок: OpenCAL SIRT": "Engine: OpenCAL SIRT",
    "Загрузить STL": "Load STL",
    "Сгенерировать проекции": "Generate projections",
    "Модель загружена.": "Model loaded.",
    "Сохранять внутренние полости (отверстия / резьба)": "Preserve internal cavities (holes / threads)",
    "Встроенный физический движок SIRT на C++ (OpenMP в Linux). Учитывает преломление по Снеллиусу и поглощение по Бугеру-Ламберту.": "Built-in C++ physical SIRT engine (OpenMP on Linux). Accounts for Snell's law and Beer-Lambert attenuation.",
    "Коэффициент оптического затухания света смолой (для CAL обычно 0.000 - 0.030 мм⁻¹)": "Resin optical attenuation coefficient (typically 0.000 - 0.030 mm⁻¹ for CAL)",
    "⚙ Настройки процесса": "⚙ Process settings",
    "Остановка...": "Stopping...",
    "⏹ Отменить генерацию": "⏹ Cancel generation",
    "▶ Воспроизведение": "▶ Play",
    "Видео собрано.": "Video assembled.",
    "кадров для воспроизведения.": "frames for playback.",
    "Сборка видео не удалась.": "Video assembly failed.",
    "Модель не готова к слайсингу": "Model is not ready for slicing",
    "Ошибка генерации проекций": "Projection generation failed",
    "Сохранено": "Saved",
    "кадров в:": "frames to:",
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


def set_language(language: str) -> None:
    """Select ``en`` or ``ru`` for subsequently created/updated UI text."""

    global _language
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    _language = language


def language() -> str:
    return _language


def tr(text: str) -> str:
    """Translate a source string between Russian and English seamlessly."""

    if _language == "ru":
        if text in _RU:
            return _RU[text]
        for source, translated in _PREFIX_RU:
            if text.startswith(source):
                return translated + text[len(source):]
        return text
    if text in _EN:
        return _EN[text]
    for source, translated in _PREFIX_EN:
        if text.startswith(source):
            return translated + text[len(source):]
    return text


def apply_translations(root: Any) -> None:
    """Translate static Qt widget text after a window has been built."""

    from PyQt6.QtCore import QObject
    from PyQt6.QtWidgets import QAbstractButton, QComboBox, QGroupBox, QLabel, QTabWidget

    widgets = [root, *root.findChildren(QObject)]
    for widget in widgets:
        if not hasattr(widget, "property"):
            continue

        if isinstance(widget, QLabel):
            if widget.property("_spinslicer_skip_text_translation") or (widget.pixmap() is not None and not widget.pixmap().isNull()):
                continue
            source = widget.property("_spinslicer_i18n_text")
            if source is None:
                source = widget.text()
                widget.setProperty("_spinslicer_i18n_text", source)
            widget.setText(tr(str(source)))
        elif isinstance(widget, QAbstractButton):
            source = widget.property("_spinslicer_i18n_text")
            if source is None:
                source = widget.text()
                widget.setProperty("_spinslicer_i18n_text", source)
            widget.setText(tr(str(source)))
        elif isinstance(widget, QGroupBox):
            source = widget.property("_spinslicer_i18n_title")
            if source is None:
                source = widget.title()
                widget.setProperty("_spinslicer_i18n_title", source)
            widget.setTitle(tr(str(source)))

        for prop in ("windowTitle", "toolTip", "statusTip", "whatsThis"):
            getter = getattr(widget, prop, None)
            setter = getattr(widget, f"set{prop[0].upper()}{prop[1:]}", None)
            if not callable(getter) or not callable(setter):
                continue
            property_name = f"_spinslicer_i18n_{prop}"
            source = widget.property(property_name)
            if source is None:
                source = getter()
                widget.setProperty(property_name, source)
            if source:
                setter(tr(str(source)))

        if isinstance(widget, QComboBox):
            sources = widget.property("_spinslicer_i18n_items")
            if sources is None:
                sources = [widget.itemText(i) for i in range(widget.count())]
                widget.setProperty("_spinslicer_i18n_items", sources)
            for index, source in enumerate(sources):
                widget.setItemText(index, tr(str(source)))

        if isinstance(widget, QTabWidget):
            sources = widget.property("_spinslicer_i18n_tabs")
            if sources is None:
                sources = [widget.tabText(i) for i in range(widget.count())]
                widget.setProperty("_spinslicer_i18n_tabs", sources)
            for index, source in enumerate(sources):
                widget.setTabText(index, tr(str(source)))

            tooltip_sources = widget.property("_spinslicer_i18n_tab_tooltips")
            if tooltip_sources is None:
                tooltip_sources = [widget.tabToolTip(i) for i in range(widget.count())]
                widget.setProperty("_spinslicer_i18n_tab_tooltips", tooltip_sources)
            for index, source in enumerate(tooltip_sources):
                if source:
                    widget.setTabToolTip(index, tr(str(source)))
