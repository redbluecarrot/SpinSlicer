<p align="center">
  <img src="assets/spinslicer.svg" alt="SpinSlicer" width="160">
</p>

<h1 align="center">SpinSlicer</h1>

<p align="center">
  Local software for tomographic volumetric additive manufacturing (VAM).
</p>

SpinSlicer is a local research prototype for tomographic volumetric additive
manufacturing (VAM). It combines a PyQt6 desktop interface, a PyVista/VTK
viewport, a projection generator, and a frame player.

The application prepares projections and previews generated frames. It does
not drive a real projector, resin vat, or rotation stage.

## Features

Stable v0.2.1 includes **Slicer**, **Projector (Video)** and **Settings**.
The experimental simulator is maintained separately and is not included in
this release. See [release notes](docs/releases/v0.2.1.md).

- Load and transform STL models in physical millimetres.
- Keep printer and process parameters on a dedicated Settings tab.
- Edit the model in a full-width 3D viewport with a visible vat bottom plane
  and an interactive Fusion-style view cube.
- Generate projection frames with the deterministic SpinSlicer internal Radon
  backend, which is the default engine.
- Optionally use the external VAMToolbox CAL optimizer when it is installed;
  it is never selected automatically by the default UI mode.
- Preserve internal bores and thread relief in loaded STL models by default.
- Preview generated frames as a rotating video and export them to MP4.
- Keep every completed run in its own directory with validated metadata and a
  manifest; incomplete runs are not published to the player.
- Store a deterministic projection schedule and the parameters used to create
  each run in its manifest.
- Russian is the default UI language. English remains available from the
  language selector and is stored with the application settings.

## Installation

SpinSlicer supports Python 3.11 and 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

python -m pip install -r requirements.txt -r requirements-dev.txt
```

The package named `pyqtdarktheme` is imported as `qdarktheme`. Video support
uses `opencv-python-headless`.

Run the application with:

```bash
python SpinSlicer.py
```

## Threaded STL models

Load the nut as a regular STL through **Load STL**. Internal bores and thread
relief are preserved by default during section rasterization, including when
the mesh-repair option is enabled. Keep the thread pitch and depth several
voxels wide at the selected grid resolution; very small features cannot be
represented reliably by any voxel-based projector.

## Projection backends

The projection engine follows the same conceptual stages as Tomo and
VAMToolbox:

1. voxelize the target geometry;
2. generate or optimize a projection sequence over evenly spaced angles;
3. normalize the dose response and export a frame sequence for playback.

The internal SpinSlicer backend uses a robust mesh-section → rasterization →
Radon path and has no additional installation requirements. The optional
VAMToolbox backend adapts a voxel target to its `TargetGeometry` and
parallel-ray `ProjectionGeometry`, then requests the CAL optimizer. It is not
pinned in `requirements.txt` because the external project uses a separate
conda-based installation and has its own distribution terms. The default UI
mode is the internal SpinSlicer projector; `Auto` is retained only as an
experimental compatibility mode.

On Windows, open **Settings → VAMToolbox — alternate engine** to create the
isolated `spinslicer-vam` Conda environment and verify the installation. This
does not modify SpinSlicer's own `.venv`. If Conda is not detected, install
Miniconda or Anaconda first, then restart the application. The official
installation details and platform limitations are documented in the
[VAMToolbox getting-started guide](https://vamtoolbox.readthedocs.io/en/latest/_docs/gettingstarted.html).

## Output format

Each completed generation creates a unique run directory:

```text
output_frames/
└── run-20260903T120000Z-a1b2c3d4/
    ├── frame_0000.png
    ├── frame_0001.png
    ├── slice_meta.json
    └── manifest.json
```

`manifest.json` stores the source identity (when the source is an STL), model
type and parameters, transform matrix, projection backend, resin settings,
frame dimensions, and `complete: true`. The frame repository validates image
counts, dimensions, metadata, byte budgets, and completion status before a
player can consume a run. Legacy flat folders containing
`frame_*.png` remain readable.

## Project layout

| File | Purpose |
| --- | --- |
| `SpinSlicer.py` | Main window, language selector, shared status bar, and log. |
| `assets/spinslicer.svg` | Application icon used by the desktop UI and packages. |
| `slicer_tab.py` | STL loading, transformation, and projection generation UI. |
| `nut_dialog.py` | Legacy threaded-nut dialog kept for API compatibility. |
| `threaded_nut.py` | Legacy parametric nut generator kept for API compatibility. |
| `ui_panels.py` | Printer/process settings and compact model transform controls. |
| `slicing_engine.py` | Mesh-section rasterization, Radon projection, and export. |
| `vam_backend.py` | Optional VAMToolbox CAL adapter and sinogram layout normalization. |
| `video_tab.py` | Frame playback and MP4 export. |
| `frame_io.py` | Validated frame-set storage, metadata, and manifests. |
| `profiles.py` | Validated machine and resin settings recorded with each projection run. |
| `validation.py` | Mesh/resource preflight and workload budgets. |
| `model_node.py` | Original mesh plus GPU-friendly transform state. |
| `viewport.py` | PyVista/VTK 3D viewport. |
| `workers.py` | Background Qt workers for model loading, generation, and video. |
| `synthetic_shapes.py` | Canonical numerical-validation phantoms. |
| `tests/` | Geometry, projection, storage, security, and validation tests. |

## Validation and safety

Before generation, SpinSlicer checks mesh finiteness, triangle count, bounds,
vat fit, layer count, estimated memory, frame dimensions, and output budgets.
Long-running work executes in `QThread` workers and supports cancellation.
Output directories are committed only after all PNG files and metadata pass
validation. Local paths are opened through Qt's desktop API rather than a
shell command.

## Development checks

```bash
python -m pytest
python -m ruff check .
python -m mypy
```

GitHub Actions runs the same checks on pushes and pull requests.

## Releases

To publish downloadable desktop builds, update the version in `pyproject.toml`,
then create and push a matching version tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

The `Build release artifacts` workflow publishes assets to the [GitHub Releases
page](https://github.com/redbluecarrot/SpinSlicer/releases):

- `SpinSlicer-windows-x64.zip` — contains `SpinSlicer.exe` for 64-bit Windows;
- `SpinSlicer-Setup-<version>.exe` — per-user Windows installer with Start Menu
  and optional desktop shortcuts;
- `SpinSlicer.exe` — standalone Windows executable;
- `SpinSlicer-linux-x64.tar.gz` — contains the portable 64-bit Linux executable;
- `SpinSlicer-<version>-amd64.deb` — Debian/Ubuntu package;
- `SpinSlicer-<version>-1.x86_64.rpm` — Fedora/RHEL-compatible package.

On Linux, unpack the archive and run `chmod +x SpinSlicer && ./SpinSlicer`.
Alternatively, install the native package with `sudo apt install ./SpinSlicer-<version>-amd64.deb`
or `sudo dnf install ./SpinSlicer-<version>-1.x86_64.rpm`. Both packages add
`SpinSlicer` to the desktop application menu. Linux builds target x86_64.

## Scientific references

- [Tomo — Print Preparation Software](https://opencal-org.readthedocs.io/en/stable/software/tomo/)
- [VAMToolbox documentation](https://vamtoolbox.readthedocs.io/en/latest/)
- [VAMToolbox source repository](https://github.com/computed-axial-lithography/VAMToolbox)
- [Automatic Exposure Volumetric Additive Manufacturing](https://doi.org/10.1002/admt.202402168)

The DOI paper describes tomographic VAM and automatic exposure control through
real-time scattered-light feedback. SpinSlicer currently implements projection
generation and frame preview only; automatic exposure hardware feedback is
outside the scope of this desktop prototype.
