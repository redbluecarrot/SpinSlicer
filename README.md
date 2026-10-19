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

Stable v0.3.2 includes **Slicer (3D)**, **Projector (Video)**, and **Settings** sidebar.
See [release notes](docs/releases/v0.3.2.md).

- **Sparse, bounded-memory optics:** Full-resolution CSR ray matrices and Z-batched optimization retain the 256×256 grid and 360 angles. Measured conditions and resource use are recorded in the [numerical validation report](docs/validation/physical-optics-v0.3.1.md).
- **Native OpenMP Parallel SIRT Engine:** Multi-threaded C++ backend (`sirtsolver.dll` / `sirtsolver.so`) runs on all standard CPUs without requiring expensive discrete GPUs.
- **Ergonomic CAD Workspace:** Side-by-side layout with wide interactive 3D viewport on the left and quick-access parameters on the right.
- **Clean 3D Viewport:** Streamlined trackball orbit, pan, and zoom without obstructing viewport cubes.
- **Load and transform STL models** in physical millimetres.
- **Physical Optics Pipeline:** Ray-tracing support for cylindrical and rectangular vats with index-matching ($n_{glass}, n_{liquid}, n_{resin}$).
- **Preserve internal bores and thread relief** in loaded STL models by default.
- **Preview generated frames** as a rotating video and export to MP4.
- **Bilingual Interface:** Instant language switching between Russian (Русский) and English.

## Installation & Packages

### Pre-built Binary Packages (GitHub Releases)

Download pre-built binaries directly from [Releases](https://github.com/redbluecarrot/SpinSlicer/releases):

- **Windows 10 / 11:**
  - **Installer:** Download and run `SpinSlicer-Setup-0.3.2.exe`
  - **Portable:** Download `SpinSlicer-windows-x64.zip`, extract and run `SpinSlicer.exe`
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
- **Generic Linux Portable:**
  ```bash
  tar -xzf SpinSlicer-linux-x64.tar.gz
  ./SpinSlicer
  ```

### Development Setup from Source

SpinSlicer supports Python 3.11 and 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

python -m pip install -r requirements.txt -r requirements-dev.txt
python build_sirt.py
```

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
3. apply a common projection scale and export a frame sequence for playback.

The default UI backend traces Snell-refracted rays through the vial or cuvette,
integrates voxel crossings and Beer–Lambert attenuation, then fits nonnegative
relative-dose projections with SIRT. The optional native library accelerates the
same calculation with C++/OpenMP. The Python API also retains internal Radon
and optional VAMToolbox compatibility paths.

The preview reconstructs the exported PNG gray values and reports the relative
dose shape RMSE after fitting a common scale. This is not a calibrated curing
prediction. Fresnel reflection, scattering, material cure response and absolute
hardware dose require additional calibration. See the [model and reproducible
checks](docs/validation/physical-optics-v0.3.1.md).

Square cuvettes use a fixed projection field equal to the outer diagonal at
every angle. This span and the physical pixel pitch are recorded in the manifest;
projector calibration must use that span. See the [square-field correction and
corner-coverage checks](docs/validation/square-detector-field.md).

## Output format

Each completed generation creates a unique run directory:

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
| `slicing_engine.py` | Mesh-section rasterization, projection planning, and verified PNG export. |
| `optics/` | Snell rays, exact voxel path integrals, and nonnegative relative-dose SIRT. |
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
