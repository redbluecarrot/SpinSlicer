<p align="center">
  <img src="assets/spinslicer.svg" alt="SpinSlicer" width="160">
</p>

<h1 align="center">SpinSlicer</h1>

<p align="center">
  Software local para fabricación aditiva volumétrica tomográfica (VAM).
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/🇬🇧_English-0b5fff?style=for-the-badge" alt="English"></a>
  <a href="README.ru.md"><img src="https://img.shields.io/badge/🇷🇺_Русский-d52b1e?style=for-the-badge" alt="Русский"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/🇪🇸_Español-f1a208?style=for-the-badge" alt="Español"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/🇨🇳_中文-de2910?style=for-the-badge" alt="中文"></a>
</p>

SpinSlicer es un prototipo de investigación local para la fabricación aditiva
volumétrica tomográfica (VAM). Combina una interfaz de escritorio en PyQt6, un
visor 3D con PyVista/VTK, un generador de proyecciones y un reproductor de
fotogramas.

La aplicación prepara proyecciones y previsualiza los fotogramas generados. No
controla un proyector real, una cubeta de resina ni una platina giratoria.

## Características

La versión estable v0.3.2 incluye **Slicer (3D)**, **Projector (Video)** y la barra lateral **Settings**.
Consulta las [notas de la versión](docs/releases/v0.3.2.md).

- **Óptica dispersa con memoria acotada:** las matrices CSR de rayos a resolución completa y la optimización por lotes en Z conservan la malla de 256×256 y 360 ángulos. Las condiciones medidas y el uso de recursos se recogen en el [informe de validación numérica](docs/validation/physical-optics-v0.3.1.md).
- **Motor SIRT paralelo nativo con OpenMP:** el backend multihilo en C++ (`sirtsolver.dll` / `sirtsolver.so`) funciona en cualquier CPU estándar sin necesidad de GPU dedicadas costosas.
- **Espacio de trabajo CAD ergonómico:** visor 3D interactivo amplio a la izquierda y parámetros de acceso rápido a la derecha.
- **Visor 3D limpio:** órbita trackball, desplazamiento y zoom sin cubos de vista que estorben.
- **Carga y transformación de modelos STL** en milímetros físicos.
- **Pipeline de óptica física:** trazado de rayos para cubetas cilíndricas y rectangulares con adaptación de índices ($n_{glass}, n_{liquid}, n_{resin}$).
- **Conserva taladros internos y relieve de rosca** en los modelos STL cargados por defecto.
- **Previsualiza los fotogramas** como vídeo giratorio y expórtalos a MP4.
- **Interfaz bilingüe:** cambio instantáneo entre ruso (Русский) e inglés.

## Instalación y paquetes

### Paquetes precompilados (GitHub Releases)

Descarga los binarios desde [Releases](https://github.com/redbluecarrot/SpinSlicer/releases):

- **Windows 10 / 11:**
  - **Instalador:** descarga y ejecuta `SpinSlicer-Setup-0.3.2.exe`
  - **Portable:** descarga `SpinSlicer-windows-x64.zip`, extráelo y ejecuta `SpinSlicer.exe`
- **Arch Linux / Manjaro:**
  ```bash
  sudo pacman -U SpinSlicer-0.3.2-1-x86_64.pkg.tar.zst
  ```
- **Ubuntu / Debian / Mint:**
  ```bash
  sudo apt install ./SpinSlicer-0.3.2-amd64.deb
  ```
- **Fedora / RHEL / openSUSE:**
  ```bash
  sudo dnf install ./SpinSlicer-0.3.2-1.x86_64.rpm
  ```
- **Linux portable genérico:**
  ```bash
  tar -xzf SpinSlicer-linux-x64.tar.gz
  ./SpinSlicer
  ```

### Entorno de desarrollo desde el código fuente

SpinSlicer es compatible con Python 3.11 y 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

python -m pip install -r requirements.txt -r requirements-dev.txt
python build_sirt.py
```

Ejecuta la aplicación con:

```bash
python SpinSlicer.py
```

## Modelos STL roscados

Carga la tuerca como un STL normal mediante **Load STL**. Los taladros internos
y el relieve de rosca se conservan por defecto durante la rasterización de
secciones, incluso con la opción de reparación de malla activada. Mantén el paso
y la profundidad de la rosca con varios vóxeles de ancho a la resolución de
malla elegida; ningún proyector basado en vóxeles puede representar de forma
fiable detalles demasiado pequeños.

## Backends de proyección

El motor de proyección sigue las mismas etapas conceptuales que Tomo y
VAMToolbox:

1. voxelizar la geometría objetivo;
2. generar u optimizar una secuencia de proyecciones sobre ángulos equiespaciados;
3. aplicar una escala común de proyección y exportar una secuencia de fotogramas para su reproducción.

El backend por defecto de la interfaz traza rayos refractados según Snell a
través del vial o la cubeta, integra los cruces de vóxeles y la atenuación de
Beer–Lambert, y después ajusta proyecciones de dosis relativa no negativas con
SIRT. La biblioteca nativa opcional acelera el mismo cálculo con C++/OpenMP. La
API de Python conserva además rutas internas de Radon y de compatibilidad
opcional con VAMToolbox.

La previsualización reconstruye los valores de gris de los PNG exportados e
informa del RMSE de forma de la dosis relativa tras ajustar una escala común.
No es una predicción de curado calibrada. La reflexión de Fresnel, la
dispersión, la respuesta de curado del material y la dosis absoluta del hardware
requieren calibración adicional. Consulta el [modelo y las comprobaciones
reproducibles](docs/validation/physical-optics-v0.3.1.md).

Las cubetas cuadradas usan un campo de proyección fijo igual a la diagonal
exterior en cada ángulo. Este valor y el paso físico de píxel se registran en el
manifiesto; la calibración del proyector debe usar ese valor. Consulta la
[corrección del campo cuadrado y las comprobaciones de cobertura de
esquinas](docs/validation/square-detector-field.md).

## Formato de salida

Cada generación completada crea un directorio de ejecución único:

```text
output_frames/
└── run-20260903T120000Z-a1b2c3d4/
    ├── frame_0000.png
    ├── frame_0001.png
    ├── slice_meta.json
    ├── dose_stats.txt
    ├── reconstruction_dose.npy
    ├── reconstruction_target.npy
    ├── reconstruction_preview.png
    └── manifest.json
```

`manifest.json` almacena la identidad de la fuente (cuando es un STL), el tipo y
los parámetros del modelo, la matriz de transformación, el backend de
proyección, los ajustes de resina, las dimensiones de los fotogramas y
`complete: true`. El repositorio de fotogramas valida el número de imágenes,
dimensiones, metadatos, límites de bytes y estado de finalización antes de que
un reproductor pueda consumir una ejecución. Las carpetas planas heredadas con
`frame_*.png` siguen siendo legibles.

## Estructura del proyecto

| Archivo | Propósito |
| --- | --- |
| `SpinSlicer.py` | Ventana principal, selector de idioma, barra de estado compartida y registro. |
| `assets/spinslicer.svg` | Icono de la aplicación para la interfaz y los paquetes. |
| `slicer_tab.py` | Carga y transformación de STL, e interfaz de generación de proyecciones. |
| `nut_dialog.py` | Diálogo heredado de tuerca roscada, conservado por compatibilidad de API. |
| `threaded_nut.py` | Generador paramétrico heredado de tuercas, conservado por compatibilidad de API. |
| `ui_panels.py` | Ajustes de impresora/proceso y controles compactos de transformación del modelo. |
| `slicing_engine.py` | Rasterización de secciones, planificación de proyecciones y exportación PNG verificada. |
| `optics/` | Rayos de Snell, integrales exactas de trayectoria por vóxel y SIRT de dosis relativa no negativa. |
| `vam_backend.py` | Adaptador opcional de VAMToolbox CAL y normalización del diseño del sinograma. |
| `video_tab.py` | Reproducción de fotogramas y exportación a MP4. |
| `frame_io.py` | Almacenamiento validado de conjuntos de fotogramas, metadatos y manifiestos. |
| `profiles.py` | Ajustes validados de máquina y resina registrados con cada ejecución. |
| `validation.py` | Comprobación previa de malla/recursos y presupuestos de carga. |
| `model_node.py` | Malla original y estado de transformación apto para GPU. |
| `viewport.py` | Visor 3D con PyVista/VTK. |
| `workers.py` | Workers Qt en segundo plano para carga de modelos, generación y vídeo. |
| `synthetic_shapes.py` | Fantomas canónicos para validación numérica. |
| `tests/` | Pruebas de geometría, proyección, almacenamiento, seguridad y validación. |

## Validación y seguridad

Antes de generar, SpinSlicer comprueba la finitud de la malla, el número de
triángulos, los límites, el ajuste a la cubeta, el número de capas, la memoria
estimada, las dimensiones de los fotogramas y los presupuestos de salida. Las
tareas largas se ejecutan en workers `QThread` y admiten cancelación. Los
directorios de salida solo se confirman cuando todos los archivos PNG y los
metadatos superan la validación. Las rutas locales se abren mediante la API de
escritorio de Qt, no con un comando de shell.

## Comprobaciones de desarrollo

```bash
python -m pytest
python -m ruff check .
python -m mypy
```

GitHub Actions ejecuta las mismas comprobaciones en cada push y pull request.

## Versiones

Para publicar compilaciones de escritorio descargables, actualiza la versión en
`pyproject.toml` y crea y sube la etiqueta de versión correspondiente:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

El workflow `Build release artifacts` publica los archivos en la [página de
GitHub Releases](https://github.com/redbluecarrot/SpinSlicer/releases):

- `SpinSlicer-windows-x64.zip` — contiene `SpinSlicer.exe` para Windows de 64 bits;
- `SpinSlicer-Setup-<version>.exe` — instalador de Windows por usuario con accesos directos en el menú Inicio y, opcionalmente, en el escritorio;
- `SpinSlicer.exe` — ejecutable independiente de Windows;
- `SpinSlicer-linux-x64.tar.gz` — ejecutable portable de Linux de 64 bits;
- `SpinSlicer-<version>-amd64.deb` — paquete Debian/Ubuntu;
- `SpinSlicer-<version>-1.x86_64.rpm` — paquete compatible con Fedora/RHEL.

En Linux, descomprime el archivo y ejecuta `chmod +x SpinSlicer && ./SpinSlicer`.
También puedes instalar el paquete nativo con `sudo apt install ./SpinSlicer-<version>-amd64.deb`
o `sudo dnf install ./SpinSlicer-<version>-1.x86_64.rpm`. Ambos paquetes añaden
`SpinSlicer` al menú de aplicaciones. Las compilaciones de Linux son para x86_64.

## Referencias científicas

- [Tomo — Print Preparation Software](https://opencal-org.readthedocs.io/en/stable/software/tomo/)
- [Documentación de VAMToolbox](https://vamtoolbox.readthedocs.io/en/latest/)
- [Repositorio de VAMToolbox](https://github.com/computed-axial-lithography/VAMToolbox)
- [Automatic Exposure Volumetric Additive Manufacturing](https://doi.org/10.1002/admt.202402168)

El artículo del DOI describe la VAM tomográfica y el control automático de
exposición mediante retroalimentación de luz dispersa en tiempo real.
SpinSlicer solo implementa por ahora la generación de proyecciones y la
previsualización de fotogramas; la retroalimentación por hardware para la
exposición automática queda fuera del alcance de este prototipo de escritorio.
