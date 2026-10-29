"""
constants.py
============
Единая точка правды для дефолтных значений и цветовой палитры.
Меняя цифры здесь, не нужно лазить по всему проекту.
"""

APP_TITLE = "SpinSlicer"
APP_ORG = "SpinSlicer"

# --- Дефолты слайсинга (перенесены из прототипа без изменений) -------------
DEFAULT_GRID_RESOLUTION = 256
DEFAULT_OUTPUT_RESOLUTION = 720
DEFAULT_NUM_FRAMES = 360
DEFAULT_DIAMETER_MM = 60.0
DEFAULT_FRAME_RATE_HZ = 30.0  # divides the 60 Hz projector refresh evenly
DEFAULT_ANGLE_OFFSET_DEG = 90.0
# Projector gamma accepted by machine profiles and applied by the MP4 export.
DEFAULT_PROJECTOR_GAMMA = 1.0
MIN_PROJECTOR_GAMMA = 0.5
MAX_PROJECTOR_GAMMA = 4.0
FILL_FRACTION = 0.92
ANTI_ALIAS_FACTOR = 4

# --- Защита вычислительного pipeline ---------------------------------------
# Значения намеренно немного шире текущих UI-диапазонов: публичные классы
# ядра могут вызываться без интерфейса, поэтому безопасность не должна
# зависеть от того, каким способом были получены параметры.
MIN_DIAMETER_MM = 1.0
MAX_DIAMETER_MM = 1000.0
MIN_GRID_RESOLUTION = 16
MAX_GRID_RESOLUTION = 512
MIN_OUTPUT_RESOLUTION = 16
MAX_OUTPUT_RESOLUTION = 4096
MIN_NUM_FRAMES = 2
MAX_NUM_FRAMES = 720
MAX_LAYERS = 4096
MAX_STL_FILE_BYTES = 512 * 1024 * 1024
MAX_MESH_TRIANGLES = 5_000_000
MAX_ESTIMATED_MEMORY_BYTES = 2 * 1024 * 1024 * 1024

# Ограничения frame-set применяются до полного декодирования изображений.
MAX_FRAME_FILE_BYTES = 64 * 1024 * 1024
MAX_FRAME_DIMENSION = 8192
MAX_FRAME_TOTAL_PIXELS = 1_000_000_000
MAX_FRAME_TOTAL_BYTES = 1_500_000_000
MAX_VIDEO_PREVIEW_BYTES = 768 * 1024 * 1024
MAX_METADATA_FILE_BYTES = 4 * 1024 * 1024

# --- Вьюпорт -----------------------------------------------------------
# Колба всегда абсолютного размера: диаметр — из настроек, высота — фиксированное
# отношение к диаметру. Она НЕ пересчитывается при трансформации модели.
VAT_HEIGHT_RATIO = 1.6
VAT_RESOLUTION = 96
VIEWPORT_WORKPLANE_SPAN_RATIO = 3.0
VIEWPORT_WORKPLANE_MIN_SPAN_MM = 120.0
VIEWPORT_WORKPLANE_RESOLUTION = 32
# Size of the last-resort native marker; the smaller footprint keeps it tight
# to the upper-left corner when the custom marker is unavailable.
VIEWPORT_ORIENTATION_WIDGET_SIZE = 96
# Normalized VTK viewport for the custom ViewCube: tight to the upper-right
# corner while still leaving room for the three edge-continuation arrows.
VIEWPORT_ORIENTATION_VIEWPORT = (0.85, 0.72, 1.0, 0.995)
VIEWPORT_ORIENTATION_CUBE_SCALE = 1.35
VIEWPORT_ORIENTATION_AXIS_LENGTH = 0.90
# Each positive axis starts at its own outer cube vertex. This keeps the
# arrowhead visibly attached to the vertex instead of emerging from a face.
VIEWPORT_ORIENTATION_AXIS_ORIGINS = (
    (
        0.5 * VIEWPORT_ORIENTATION_CUBE_SCALE,
        -0.5 * VIEWPORT_ORIENTATION_CUBE_SCALE,
        -0.5 * VIEWPORT_ORIENTATION_CUBE_SCALE,
    ),
    (
        -0.5 * VIEWPORT_ORIENTATION_CUBE_SCALE,
        0.5 * VIEWPORT_ORIENTATION_CUBE_SCALE,
        -0.5 * VIEWPORT_ORIENTATION_CUBE_SCALE,
    ),
    (
        -0.5 * VIEWPORT_ORIENTATION_CUBE_SCALE,
        -0.5 * VIEWPORT_ORIENTATION_CUBE_SCALE,
        0.5 * VIEWPORT_ORIENTATION_CUBE_SCALE,
    ),
)
VIEWPORT_ORIENTATION_AXIS_TIP_LENGTH = 0.14
# Absolute world-space radius for the small arrowhead cone.
VIEWPORT_ORIENTATION_AXIS_CONE_RADIUS = 0.10
VIEWPORT_ORIENTATION_AXIS_CONE_RESOLUTION = 3
# Keep the edge-continuation arrows readable at the compact marker size.
VIEWPORT_ORIENTATION_AXIS_LINE_WIDTH = 3.0
VIEWPORT_ORIENTATION_DRAG_SENSITIVITY = 0.70

# Децимация полигонов — только для показа во вьюпорте (60 FPS на тяжёлых STL).
# На расчёт проекций это не влияет: генерация всегда использует исходный меш.
MAX_VIEWPORT_TRIANGLES = 400_000

# --- Палитра -------------------------------------------------------------
ACCENT_GREEN = "#2fa572"
ACCENT_GREEN_HOVER = "#247d57"
ACCENT_BLUE = "#3d8bfd"
ACCENT_BLUE_HOVER = "#2f6fd1"
DANGER_RED = "#e05c5c"

VAT_COLOR = "#6fb8cc"
MODEL_COLOR = "#00b4d8"
VIEWPORT_BG_TOP = "#232733"
VIEWPORT_BG_BOTTOM = "#121319"

PANEL_RADIUS = 12
BUTTON_RADIUS = 8
