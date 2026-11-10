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
    "Настройки процесса": "Process settings",
    "Настройки": "Settings",
    "Параметры фотополимера": "Photopolymer parameters",
    "Объект": "Object",
    "Модель": "Model",
    "Файл не выбран": "No file selected",
    "Размер, мм": "Size, mm",
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
    "Центрировать": "Center",
    "Авто-фит под колбу": "Fit to vat",
    "Сбросить": "Reset",
    "Готово к работе.": "Ready.",
    "Открыть STL-файл модели": "Open an STL model file",
    "Запустить расчёт проекций в фоновом потоке":
        "Start projection generation in a background thread",
    "Сначала загрузи STL-модель.": "Load an STL model first.",
    "Выбрать STL": "Choose an STL file",
    "STL файлы (*.stl)": "STL files (*.stl)",
    "Успех": "Success",
    "Ошибка": "Error",
    "Внимание": "Warning",
    "Язык изменён": "Language changed",

    # Missing strings from UI panels
    "Сбросить": "Reset",
    "Центрировать": "Center",
    "Авто-фит под колбу": "Auto-fit to vat",
    "Вписать модель в текущий размер колбы": "Fit model to current vat size",
    "Переместить модель в центр XY": "Move model to XY center",
    "Сохранять соотношение сторон модели": "Keep model aspect ratio",
    "Сохранять пропорции при масштабировании": "Keep proportions when scaling",
    "Центр": "Center",
    "Авто-фит": "Auto-fit",
    "Сброс": "Reset",
    "Проектор": "Projector",
    "Загрузить STL": "Load STL",
    "Сгенерировать проекции": "Generate projections",
    "Виртуальная печать": "Virtual print",
    "Калибровка": "Calibrate",
    "Калибровка экспозиции": "Exposure calibration",
    "Пробная печать, которая подберёт экспозицию под твою смолу": "A test print that finds the exposure for your resin",
    "Сгенерировать калибровку": "Generate calibration",
    "Самый короткий затвердевший столбик:": "Shortest pillar that cured:",
    "Ни один не затвердел": "None cured",
    "Применить": "Apply",
    "мм": "mm",
    "Сначала сгенерируй калибровку.": "Generate the calibration first.",
    "Калибровка работает с физическим движком и круглой колбой.": "Calibration needs the physical engine and a round vial.",
    "Напечатай эти кадры, затем открой «Калибровка» и укажи самый короткий затвердевший столбик.": "Print these frames, then open “Calibrate” and pick the shortest pillar that cured.",
    "Кнопка «Виртуальная печать» покажет деталь, которая затвердеет из этих кадров.": "The “Virtual print” button shows the part these frames would cure.",
    "Показать, что затвердеет из этих кадров (без принтера)": "Show what these frames would cure (no printer needed)",
    "Модель загружена.": "Model loaded.",
    "Остановка...": "Stopping...",
    "⏹ Отменить генерацию": "⏹ Cancel generation",
    "Видео собрано.": "Video assembled.",
    "кадров для воспроизведения.": "frames for playback.",
    "Сборка видео не удалась.": "Video assembly failed.",
    "Модель не готова к слайсингу": "Model is not ready for slicing",
    "Ошибка генерации проекций": "Projection generation failed",
    "Сохранено": "Saved",
    "кадров в:": "frames to:",
}

_RU: dict[str, str] = {
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
    ("Готово к работе. Загрузи STL-модель, чтобы начать.", "Ready. Load an STL model to begin."),
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
    ("Ready. Load an STL model to begin.", "Готово к работе. Загрузи STL-модель, чтобы начать."),
    ("Ready.", "Готово к работе."),
    ("Done!", "Готово!"),
    ("Cancelled by user.", "Отменено пользователем."),
    ("Stopping...", "Остановка..."),
    ("Generation cancelled by user.", "Генерация отменена пользователем."),
)


# --- New interface (2026-10): exact strings -------------------------------------------
_EN.update({
    "Принтер, проектор и смола — настраиваются один раз": "Printer, projector and resin — set up once",
    "Частота проектора": "Projector refresh rate", "Оптика": "Optics", "Гц": "Hz",
    "Показатели преломления": "Refractive indices",
    "герцовка экрана: кадры не могут сменяться чаще": "screen refresh rate: frames cannot change faster",
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
    "Линии": "Lines", "прямые": "straight", "от руки": "hand-drawn", "живые": "alive",
    "прямые — по линейке, модель без точек; от руки — неподвижный набросок; живые — набросок «дышит»":
        "straight — ruled, model without dots; hand-drawn — a still sketch; alive — the sketch breathes", "затвердело как задумано": "cured as intended",
    "лишняя смола": "extra resin", "не хватает": "missing",
    "Влезает в колбу": "Fits the vial", "влезает": "fits", "не влезает": "does not fit",
    "Ширина": "Width", "Высота": "Height", "В центр": "Centre", "Вписать в колбу": "Fit to the vial",
    "показать, что затвердеет из этих кадров": "show what will cure from these frames",
    "Деталь": "Part", "Проверка": "Check",
    "Показать": "Show", "Вернуться к модели": "Back to the model",
    "Интерфейс": "Interface", "Тонкая настройка": "Fine tuning",
    "конус проектора проходит линзу Френеля и становится параллельным":
        "the projector cone passes a Fresnel lens and becomes parallel",
    "загрузить STL": "load STL", "Модель не загружена": "No model loaded", "STL-файл детали": "the part's STL file",
    "Масштаб": "Scale", "Ориентация": "Orientation", "Отверстия и резьба": "Holes and threads",
    "Положение": "Position", "Точная позиция": "Exact position", "Качество": "Quality",
    "Кадров на оборот": "Frames per turn", "Компенсация размытия": "Blur compensation",
    "Виртуальная печать": "Virtual print", "показать на сцене, что затвердеет": "show on the scene what will cure",
    "совпадение": "match", "запас экспоз.": "exposure margin", "печать": "print",
    "Пока считается — засвети деталь сами": "While it computes — cure the part yourself",
    "Свет готов. Доиграй или иди печатать": "Light is ready. Finish the round or go print",
    "Куда выводить": "Output to", "Скорость вращения": "Rotation speed", "Настроить проектор": "Set up the projector",
    "разметка покажет, где должны быть стенки колбы; тест серого — все уровни яркости":
        "the pattern shows where the vial walls must be; the gray test shows every brightness level",
    "Сначала рассчитай свет.": "Compute the light first.",
    "двигает картинку вверх/вниз, чтобы деталь попала в колбу": "moves the picture up/down so the part lands in the vial",
    "преломление на стенке колбы": "refraction at the vial wall", "преломление на стенке ванны": "refraction at the bath wall",
    "Опиши свой принтер и смолу. Это делается один раз — дальше только детали.":
        "Describe your printer and resin. You do this once — after that, just parts.",
    "Покрути деталь мышью и проверь, как она стоит. Если всё ок — дальше.":
        "Turn the part with the mouse and check how it sits. If it looks right — next.",
    "Выбери качество и нажми «Рассчитать свет».": "Pick the quality and press “Compute light”.",
    "Загрузи STL: перетащи файл на сцену слева или нажми кнопку внизу.":
        "Load an STL: drop the file on the scene or press the button below.",
    "Деталь не помещается — уменьши масштаб или нажми «вписать в колбу».":
        "The part does not fit — scale it down or press “fit into vial”.",
    "Считаю кадры для каждого угла. Можно пока поиграть ↓": "Computing frames for every angle. Play meanwhile ↓",
    "Готово. Посмотри совпадение и запас — и переходи к печати.":
        "Done. Check the match and the margin — then go to printing.",
    "Проверь разметку колбы на проекторе и запускай.": "Check the vial pattern on the projector and start.",
    "сколько пикселей проектора по ширине занимает колба — проверь разметкой":
        "how many projector pixels the vial spans — check with the pattern",
    "не знаешь? напечатай шкалу ↓": "not sure? print the ladder ↓", "напечатать шкалу →": "print the ladder →",
    "натуральный логарифм: α = 2,303·A на 1 мм = 1 / Dp":
        "natural log: α = 2.303·A per 1 mm = 1 / Dp",
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
    "увеличь сетку": "raise the grid", "Время засветки": "Exposure time", "Обороты": "Revolutions",
    "Время": "Time", "задай поглощение, облучённость и дозу": "set absorption, irradiance and dose",
    "Запас экспозиции": "Exposure margin", "Частота кадров": "Frame rate", "заменить": "replace",
    "или перетащи файл на сцену": "or drop the file on the scene", "Сначала умести деталь": "Make the part fit first",
    "уменьши масштаб или возьми колбу шире": "scale it down or use a wider vial", "Дальше — свет →": "Next — light →",
    "проверка модели пройдена": "model check passed", "Остановить": "Stop", "считаю кадры…": "computing frames…",
    "сначала рассчитай свет": "compute the light first", "Дальше — печать →": "Next — print →",
    "Рассчитать свет": "Compute light", "Показать на проекторе": "Show on projector", "Сохранить видео": "Save video",
    "Открыть папку кадров": "Open frames folder", "Esc на проекторе — остановить": "Esc on the projector stops it",
    "кадры готовы": "frames ready", "сначала загрузи модель": "load a model first",
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
    "Перетащи STL сюда": "Drop an STL here", "файл детали в формате STL, в миллиметрах": "the part as an STL file, in millimetres",
    "Выбрать файл": "Choose file", "Кадры появятся после расчёта света": "Frames appear after the light is computed",
    "Загрузка кадров...": "Loading frames...", "Экспорт в MP4...": "Exporting MP4...",
    "Экспорт тестовой шкалы...": "Exporting the gray test...", "Видео сохранено.": "Video saved.",
    "Экспорт не удался.": "Export failed.", "Экспорт не удался": "Export failed", "Экран проектора": "Projector screen",
    "Загружаю кадры…": "Loading frames…", "Не удалось прочитать кадры": "Could not read the frames",
    "Кадры выводятся на второй экран. Esc — остановить.": "Frames go to the second screen. Esc stops.",
    "Готово": "Done", "Видео сохранено": "Video saved", "Сохранить видео ": "Save video",
    "Второй экран не найден. Подключи проектор как второй монитор (расширенный рабочий стол) или сохрани видео.":
        "No second screen found. Connect the projector as a second monitor (extended desktop) or save a video.",
    "Показывай его на проекторе во весь экран, без масштабирования.": "Play it full screen on the projector, without scaling.",
    "Сначала рассчитай свет.": "Compute the light first.", "Проектор": "Projector",
    "Создаю тестовую гайку M20...": "Building the M20 test nut...", "модель": "model",
    "мышь — двигать, клик — старт": "mouse — move, click — start",
    "Засветка": "Cure", "Слои": "Layers", "Фотон": "Photon",
    "кирпичи — силуэт твоей детали; фотон засвечивает воксели": "bricks are your part's silhouette; the photon cures voxels",
    "укладывай воксели в колбу без щелей": "stack voxels in the vial without gaps",
    "проведи фотон через стенки, не задев стекло": "guide the photon through the walls without touching the glass",
    "←/→ — двигать, ↑ — повернуть, ↓ — быстрее": "←/→ move, ↑ rotate, ↓ drop",
    "клик или пробел — подлететь вверх": "click or Space — fly up",
    "появится после расчёта света": "appears after the light is computed",
    "результат устарел — пересчитай свет": "result is out of date — compute the light again",
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
    "до оси ≈ {} мм · по высоте у краёв детали размытие до {} мм — поставь проектор дальше":
        "to the axis ≈ {} mm · top and bottom of the part blur up to {} mm — move the projector farther",
    "до оси {} мм — ближе колбы, так не бывает: проверь числа": "to the axis {} mm — closer than the vial, impossible: check the numbers",
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
    "кадр каждые {} мс · проектор {} Гц — больше {} Гц, проверь проектор": "a frame every {} ms · projector {} Hz — over {} Hz, check the projector",
    "кадр каждые {} мс · проектор {} Гц — выше частоты проектора ({} Гц)": "a frame every {} ms · projector {} Hz — above the projector refresh rate ({} Hz)",
    "кадр на экране ≥ {} мс": "a frame stays on screen ≥ {} ms",
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
    "{} столбиков с дозой {}–{} %: выбираешь лучший, доза обновится сама": "{} pillars at {}–{} % dose: pick the best one, the dose updates itself",
    "или открыть тестовую гайку M{}": "or open the M{} test nut",
    "есть в паспорте проектора, обычно {},{}–{},{}": "in the projector datasheet, usually {}.{}–{}.{}",
    "есть в паспорте проектора, обычно {}–{}": "in the projector datasheet, usually {}–{}",
    "не знаешь? оставь {}": "not sure? leave {}",
    "Сначала рассчитай свет на шаге {}.": "Compute the light in step {} first.",
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
    "Время печати не рассчитано: задай поглощение смолы (затухание), облучённость проектора и дозу гелеобразования смолы.":
        "Print time not computed: set resin absorption, projector irradiance and the resin gel dose.",
    "Нет экспозиции, при которой вся деталь печатается чисто (окно процесса отрицательное). Увеличь число кадров или сетку либо используй иммерсию.":
        "No exposure prints the whole part cleanly (negative process window). Use more frames, a finer grid or immersion.",
})
_EN_TEMPLATES.update({
    "Меньше {} оборотов: ошибки старта/остановки и синхронизации станут заметными. Уменьши яркость или мощность проектора.":
        "Fewer than {} revolutions: start/stop and sync errors become visible. Lower the projector brightness or power.",
    "{}% детали лежит дальше r = {} мм от оси: из-за преломления туда приходят лучи не со всех сторон. "
    "Сдвинь деталь к оси или используй иммерсию.":
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
    "летишь сквозь свет твоей печати": "fly through the light of your print",
})
_EN_TEMPLATES.update({
    "экспозиция +{} %": "exposure +{} %", "экспозиция -{} %": "exposure -{} %",
    "не досвечено {} % · лишнее {} %": "under-cured {} % · extra {} %",
    "ошибка формы дозы {}": "dose shape error {}",
})
_EN.update({"Сдвиг, мм": "Move, mm", "Поворот, °": "Rotate, °", "Размер, мм": "Size, mm", "Центр": "Centre",
            "Вписать": "Fit", "Сброс": "Reset"})

# The .spin print file and the network printer (spin_format, printer_link).
_EN.update({
    "не выбран": "none", "Принтер": "Printer",
    "SpinPrinter в сети: его профиль заполнит колбу и проектор":
        "a SpinPrinter on the network: its profile fills in the vial and the projector",
    "найти в сети": "find on the network", "ввести адрес": "enter address",
    "экспорт .spin": "export .spin", "отправить на принтер": "send to printer",
    "Файл для принтера": "Printer file",
    "видео, расчёт и превью одним файлом: принтер сам знает обороты":
        "video, plan and preview in one file: the printer knows the revolutions",
    "диаметр колбы": "vial diameter", "стенка": "wall", "разрешение проектора": "projector resolution",
    "частота проектора": "projector refresh rate", "облучённость": "irradiance", "контраст": "contrast",
    "n стекла": "glass n", "колба на экране": "vial on screen", "сдвиг по X": "X offset",
    "сдвиг по Y": "Y offset",
    "Подожди: идёт экспорт.": "Please wait: an export is running.",
    "Экспорт для принтера (.spin)": "Export for printer (.spin)",
    "Файл для принтера сохранён.": "Printer file saved.", "Файл для принтера сохранён": "Printer file saved",
    "Скопируй его на флешку принтера или отправь по сети.":
        "Copy it to the printer's USB stick or send it over the network.",
    "Отправлено на принтер.": "Sent to the printer.", "Отправлено на принтер": "Sent to printer",
    "Не удалось подготовить файл для принтера.": "Could not prepare the printer file.",
    "Не удалось отправить на принтер": "Could not send to the printer",
    "Упаковка файла .spin...": "Packing the .spin file...",
    "Отправка на принтер": "Sending to printer",
    "Поиск по mDNS недоступен (pip install zeroconf): проверяю spinslicer.local и spinprinter.local.":
        "mDNS search is not available (pip install zeroconf): checking spinslicer.local and spinprinter.local.",
    "Ищу принтеры в сети…": "Looking for printers on the network…",
    "Найдено принтеров:": "Printers found:",
    "Принтер в сети не найден. Введи его адрес.": "No printer found on the network. Enter its address.",
    "Адрес принтера": "Printer address",
    "IP-адрес или имя принтера, например spinprinter.local": "IP address or name of the printer, e.g. spinprinter.local",
    "Принтер недоступен:": "Printer not reachable:",
    "принтер не отдаёт профиль, настройки остаются как есть.":
        "the printer does not provide a profile, the settings stay as they are.",
    "Разрешения проектора нет в списке:": "The projector resolution is not in the list:",
    "Профиль принтера": "Printer profile", "ничего не изменилось": "nothing changed",
})
_EN_TEMPLATES.update({"Отправка на принтер {}%": "Sending to printer {}%"})

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
    "Настройки или деталь изменились — кадры устарели. Пересчитай свет.":
        "Settings or the part changed — the frames are out of date. Compute the light again.",
    "Пересчитать свет": "Compute light again",
    "старые кадры не подходят к новым настройкам": "the old frames do not match the new settings",
})

# The printer: connect, write back, print with a yes on the printer, USB stick.
_EN.update({
    "Обнови принтер": "Update the printer",
    "Программа этого принтера не спрашивает «Печатать?» на его экране и начала бы печать "
    "сразу. Обнови SpinPrinter (на Raspberry Pi ещё раз команду установки) или "
    "печатай с флешки.":
        "This printer's software does not ask \"Print?\" on its screen and would start at once. "
        "Update SpinPrinter (run the install command on the Raspberry Pi again) or print from a USB stick.",
    "Проверяю принтер…": "Checking the printer…",
    "Принтер сейчас печатает. Дождись конца или останови печать на принтере.":
        "The printer is printing. Wait until it is done or stop the print on the printer.",
    "Деталь": "Part", "Положение": "Position", "Расчёт": "Calculation", "Проверка": "Check", "Доза": "Dose",
    "Оптика и сдвиг": "Optics and shift", "Результат": "Result",
    "Качество": "Quality", "сетка": "grid", "шаг": "step",
    "Подключить": "Connect", "Отключить": "Disconnect", "Принтер отключён.": "Printer disconnected.",
    # the printer's refusals by code, the print step's progress card, checks and the answer
    "Принтер занят другой печатью.": "The printer is busy with another print.",
    "На принтере нет такого файла.": "The printer has no such file.",
    "Файл на принтере повреждён: отправь его ещё раз.": "The file on the printer is damaged: send it again.",
    "Экран принтера не подключён: подтвердить печать негде.": "The printer's screen is not connected: nowhere to confirm.",
    "Проектор не подключён к принтеру (HDMI).": "The projector is not connected to the printer (HDMI).",
    "Видео для другого направления вращения: на принтере нужен движок angle.":
        "The video turns the other way: the printer needs the angle engine.",
    "Прошивка платы принтера не спрашивает «Печатать?»: обнови её.":
        "The printer's board firmware does not ask \"Print?\": update it.",
    "Принтер не понял запрос.": "The printer did not understand the request.",
    "Принтер отказал запросу из этой программы.": "The printer refused a request from this program.",
    "Ждёт подтверждения на экране принтера": "Waits for a yes on the printer's screen",
    "джойстик вправо — печатать": "joystick right to print",
    "Разгон": "Spin-up", "Прогрев проектора": "Projector warm-up", "Пауза до света": "Pause before the light",
    "Подготовка": "Getting ready", "разгон, прогрев, пауза до света": "spin-up, warm-up, pause before the light",
    "оборот": "turn", "из": "of", "осталось": "left",
    "кадры": "frames", "проектор": "projector", "проектор принтера": "the printer's projector",
    "профиль этого принтера": "this printer's profile", "профиль другого принтера": "another printer's profile",
    "об.": "turns", "по расчёту": "as planned", "доза по расчёту": "the dose as planned", "вместо": "instead of",
    "место на принтере": "space on the printer", "ГБ": "GB", "Стоп": "Stop",
    "Как получилась деталь?": "How did the part come out?",
    "поправка смолы на принтере сброшена": "the printer's correction for this resin is reset",
    "Поправка на принтере не сброшена:": "The printer's correction was not reset:",
    "применить": "apply", "по": "from", "печатям на этом принтере": "prints on this printer",
    "Недосвет": "Under", "Пересвет": "Over",
    "Не знаю, пропустить": "Not sure, skip", "Ответ отправлен на принтер": "Answer sent to the printer",
    "поправка": "correction", "Ответ не отправлен:": "Answer not sent:",
    "Подключаюсь к": "Connecting to", "Подключён:": "Connected:",
    "Не подключён:": "Not connected:", "не подключён": "not connected", "не отвечает": "does not answer",
    "из принтера:": "from the printer:", "ничего нового": "nothing new",
    "Настройки этого слайсера": "This slicer's settings", "записать в принтер": "write to the printer",
    "колба, проектор и калибровка, изменённые здесь, сохранятся в принтере":
        "the vial, projector and calibration changed here are stored on the printer",
    "Записать в принтер": "Write to the printer", "В принтере изменится:": "On the printer this changes:",
    "В принтере уже те же настройки.": "The printer already has the same settings.",
    "Настройки записаны в принтер.": "Settings written to the printer.",
    "Не удалось записать в принтер": "Could not write to the printer",
    "Принтер не подключён": "No printer connected",
    "Подключи принтер: Настройки → Принтер → «Подключить».": "Connect a printer: Settings → Printer → Connect.",
    "Подключи принтер: Настройки → Принтер": "Connect a printer: Settings → Printer",
    "Печать на SpinPrinter": "Print on a SpinPrinter", "Напечатать": "Print", "На флешку": "To USB stick",
    "стоп": "stop", "сохранить .spin в папку…": "save the .spin to a folder…",
    "файл .spin уходит на принтер, его экран спросит «Печатать?»":
        "the .spin goes to the printer, its screen asks \"Print?\"",
    "Готовлю файл и отправляю на принтер…": "Preparing the file and sending it to the printer…",
    "Принтер не назвал загруженный файл.": "The printer did not name the uploaded file.",
    "Принтер не принял печать:": "The printer refused the print:",
    "Принтер уже печатает отправленную деталь.": "The printer is already printing the part sent.",
    "Подтверди печать на экране принтера: джойстик вправо.":
        "Confirm on the printer's screen: push the joystick right.",
    "Принтер готовится: разгон, прогрев, пауза до света…": "The printer gets ready: spin-up, warm-up, dark pause…",
    "оборот": "turn", "из": "of", "осталось": "left",
    "Печать отменена на экране принтера.": "Print cancelled on the printer's screen.",
    "Принтер не дождался ответа на экране (10 минут).": "Nobody answered on the printer's screen (10 minutes).",
    "На принтере запустили другую печать.": "Another print was started on the printer.",
    "Готово! Ответь на принтере, как получилось.": "Done! Tell the printer how it came out.",
    "Печать остановлена.": "Print stopped.", "Ошибка печати:": "Print error:",
    "Нет связи с принтером:": "Lost the printer:", "Остановить печать?": "Stop the print?",
    "Свет погаснет, деталь останется недосвеченной.": "The light goes off, the part stays underexposed.",
    "Не удалось остановить": "Could not stop", "Файл не отправлен.": "The file was not sent.",
    "Флешка не найдена": "No USB stick found",
    "Вставь флешку в компьютер. Пока сохраню файл туда, куда скажешь.":
        "Plug a USB stick into the computer. For now I will save the file where you say.",
    "На какую флешку?": "Which USB stick?", "Флешка": "USB stick", "Файл уже есть": "The file is already there",
    "На флешке уже есть": "The stick already has", "Заменить?": "Replace it?",
    "Записываю на флешку": "Writing to the USB stick", "Записано на флешку": "Written to the USB stick",
    "Файл записан на флешку": "The file is on the USB stick",
    "Извлечь флешку? Потом вставь её в Raspberry Pi принтера.":
        "Eject the stick? Then plug it into the printer's Raspberry Pi.",
    "Флешку можно вынимать.": "The stick can be removed.",
    "Не удалось извлечь: извлеки её сами.": "Could not eject: eject it yourself.",
})

# The interface rework: Settings in tabs, Light in three stages, Print in two ways.
_EN.update({
    "Тема": "Theme", "светлая": "light", "тёмная": "dark", "Внешний вид": "Appearance",
    "Подключение": "Connection", "Калибровка на проекторе": "Calibration on the projector", "Мотор": "Motor",
    "Калибровка смолы": "Resin calibration", "Шкала дозы": "Dose scale", "напечатать →": "print →",
    "Проверить на проекторе": "Check on the projector", "Яркость при печати": "Brightness while printing",
    "подключён": "connected", "настройки принтера": "printer settings",
    "время": "time", "оборотов": "turns", "кадры": "frames", "запас экспозиции": "exposure margin",
    "время печати": "print time",
    "Деталь или настройки изменились: прошлый результат устарел.":
        "The part or the settings changed: the last result is out of date.",
    "пересчитать с другими параметрами": "compute again with other parameters",
    "Пока ждёте — засвети деталь сами": "While you wait, cure the part yourself",
    "Параметры расчёта": "Calculation parameters", "Игра": "Game",
    "скрыть замечания": "hide the notes", "замечаний": "notes", "показать": "show", "скрыть": "hide",
    "журнал": "log", "скрыть журнал": "hide log", "остановить печать": "stop the print",
    "на проектор этого компьютера": "on this computer's projector", ".spin в папку": ".spin to a folder",
    "Рассчитать свет →": "Compute the light →", "без кадров печатать нечего": "nothing to print without frames",
    "/мм": "/mm", "кадры устарели — пересчитай свет": "the frames are out of date: compute the light again",
    "принтер не подключён: Настройки → Принтер. На флешку можно и так":
        "no printer connected: Settings → Printer. The USB stick works anyway",
    "файл уйдёт на принтер, его экран спросит «Печатать?»":
        "the file goes to the printer, its screen asks \"Print?\"",
    "нужны поглощение, облучённость и доза: Настройки": "needs absorption, irradiance and dose: Settings",
    "нужны данные смолы": "needs the resin data",
    "на оборот": "per turn", "оборота": "turns", "экспозиция": "exposure", "яркость": "brightness",
})
