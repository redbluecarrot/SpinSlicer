<p align="center">
  <img src="assets/spinslicer.svg" alt="SpinSlicer" width="160">
</p>

<h1 align="center">SpinSlicer</h1>

<p align="center">
  Local software for tomographic volumetric additive manufacturing (VAM).
</p>

<p align="center">
  <a href="README.ru.md"><img src="assets/lang-ru.svg" alt="Русский" height="36"></a>
  <a href="README.es.md"><img src="assets/lang-es.svg" alt="Español" height="36"></a>
  <a href="README.zh-CN.md"><img src="assets/lang-zh.svg" alt="中文" height="36"></a>
</p>

SpinSlicer prepares prints for tomographic volumetric additive manufacturing
(VAM / CAL): a vial of resin turns in front of a projector, and the light from
hundreds of angles adds up to cure the whole part at once. SpinSlicer computes
those projector frames from an STL file, shows what they will cure, and plays
them on the projector for exactly the planned exposure.

It is built for both CAL researchers and people building their own printer:
every setting shows what it physically changes, and every number comes from a
model of the optics, not from a rule of thumb.

## Features

Version 0.5.0 — see the [release notes](docs/releases/v0.5.0.md).

- **Three steps:** *Model* (load, place and check the STL), *Light* (compute the
  frames), *Print* (show them on the projector, save an MP4 or PNG frames).
  The printer itself is described once under *Printer*.
- **FastCAL engine:** the round vial makes every projection angle optically the
  same, so rays are traced for one angle and the problem splits into small
  per-frequency systems. Refraction (Snell), attenuation (Beer–Lambert), Fresnel
  losses, the projector's pixel profile and cone of rays (throw ratio) and the
  rotation during a frame are all in the model. Typical parts take seconds to a
  minute, not hours ([benchmarks](docs/validation/resources-2026-10-31.md),
  [comparison with OSMO, PM, BCLP and classic CAL](docs/validation/engine-judge-2026-10-31.md)).
- **Virtual print:** the frames as the projector will really show them (8-bit,
  gamma, contrast, row blur) are turned back into dose; the cured part is shown in
  3D next to the model, with the match (IoU) and the exposure margin.
- **Print plan:** print time and number of turns from the resin's gel dose,
  absorption and the projector's irradiance; exposure calibration print for a new resin.
- **Dose card:** the light dose in the middle layer and what cures at the chosen
  exposure; its slider is the print exposure.
- **Projector output, ready to print:** frames placed 1:1 on the projector's
  native screen (no stretching), one frame row per layer, the frame shown chosen
  from the clock (no drift against the turning vial), frames held in memory, live
  brightness, and an automatic stop when the planned dose is delivered.
- **Smooth on modest hardware:** the computation runs in its own process at low
  priority and uses all CPU cores; the window stays responsive and uses no CPU when idle.
- **English and Russian**, day and night themes.

## Workflow

1. **Printer** — vial diameter and wall, immersion bath, projector resolution,
   throw ratio, gamma and irradiance; resin absorption and gel dose.
2. **Model** — drop an STL on the scene; move, rotate and scale it. Parts that do
   not fit the printable zone are flagged before anything is computed.
3. **Light** — pick the quality (grid 128 / 256 / 384) and compute. A mini-game
   keeps you company meanwhile. Check the virtual print and the exposure margin.
4. **Print** — check the vial alignment pattern on the projector, set the exposure
   on the dose card and press *Show on projector*. The projector stops by itself.

## Installation & Packages

### Pre-built Binary Packages (GitHub Releases)

Download pre-built binaries directly from [Releases](https://github.com/redbluecarrot/SpinSlicer/releases):

- **Windows 10 / 11:**
  - **Installer:** Download and run `SpinSlicer-Setup-0.5.0.exe`
  - **Portable:** Download `SpinSlicer-windows-x64.zip`, extract and run `SpinSlicer.exe`
- **Arch Linux / Manjaro:**
  ```bash
  sudo pacman -U SpinSlicer-0.5.0-1-x86_64.pkg.tar.zst
  ```
- **Ubuntu / Debian / Mint:**
  ```bash
  sudo apt install ./SpinSlicer-0.5.0-amd64.deb
  ```
- **Fedora / RHEL / openSUSE:**
  ```bash
  sudo dnf install ./SpinSlicer-0.5.0-1.x86_64.rpm
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

## Projection engine

The engine follows the usual stages of Tomo and VAMToolbox:

1. voxelize the part (exact coverage, non-zero winding; open meshes are repaired
   layer by layer), with layers at the projector's row pitch;
2. plan nonnegative projections that put the dose above the gel threshold inside
   the part and below it outside (a band objective with a soft cap), solved by a
   projected L-BFGS with angular smoothing (no frame flicker), coarse-to-fine
   across layers and two-grid across resolution;
3. apply one common gain, encode gray levels for the projector's gamma once, and
   export the frames; then run the virtual print and the print plan.

FastCAL needs a round vial and equally spaced angles over a full turn. Square
cuvettes use the sparse SIRT path (`optics/optimization.py`, optional C++/OpenMP
library); the Python API also keeps the Radon and VAMToolbox paths.

The dose model is relative: the absolute print time depends on the resin data
and the projector's irradiance you enter, and on how well they are measured. The
exposure calibration print is the way to pin it down for a new resin.

Square cuvettes use a fixed projection field equal to the outer diagonal at
every angle. See the [square-field correction and corner-coverage
checks](docs/validation/square-detector-field.md).

## Performance

Measured on purpose on a weak machine (2 cores at 2.1 GHz, no GPU); a typical
desktop CPU is faster. Details: [resources report](docs/validation/resources-2026-10-31.md).

| Quality | Frames | Time | Peak RAM (compute) |
| --- | --- | --- | --- |
| Fast (grid 128) | 360 | ~10 s | 0.5 GB |
| Normal (grid 256) | 360 | ~34 s | 1.1 GB |
| Fine (grid 384) | 360 | ~69 s | 1.7 GB |

The engine adapts to the machine: it uses every core (`SPINSLICER_THREADS` to
limit it) and up to a fifth of the free memory for wider batches
(`SPINSLICER_ENGINE_MB` to set it).

## Output format

Each completed computation creates a unique run directory:

```text
output_frames/
└── run-20261031T120000Z-a1b2c3d4/
    ├── frame_0000.png …          one frame per angle, 8-bit, one row per layer
    ├── print_plan.json           print time, exposure margin, virtual-print metrics
    ├── virtual_print.stl         the part the frames would cure
    ├── virtual_print_slices.png  a few layers: cured / extra / missing
    ├── reconstruction_dose.npy   dose of the middle layer from the saved PNGs
    ├── reconstruction_target.npy
    ├── reconstruction_preview.png
    ├── dose_stats.txt
    ├── slice_meta.json
    └── manifest.json
```

`manifest.json` stores the source identity, parameters, transform matrix,
engine, frame geometry (detector span, pixel pitch, Z pitch) and `complete: true`.
A run is published only after every PNG and the metadata pass validation.

## Project layout

| File | Purpose |
| --- | --- |
| `SpinSlicer.py` | Main window: steps, header, log; also starts the compute process. |
| `ui_panels.py` | The settings panel: printer, model, light and print pages. |
| `ui_kit.py`, `theme.py` | Hand-drawn widgets, palettes and the day/night switch. |
| `slicer_tab.py`, `viewport.py`, `sketch_scene.py` | 3D scene: the part plus a live sketch of the printer. |
| `video_tab.py`, `projector_canvas.py` | Print screen, dose card, full-screen projector output, MP4 export. |
| `engine_process.py`, `workers.py` | The compute process and background workers. |
| `slicing_engine.py`, `voxelizer.py` | Voxelization, planning, frame export. |
| `optics/fastcal.py` | FastCAL engine for round vials. |
| `optics/optimization.py`, `optics/ray_tracing.py` | Sparse SIRT path (square cuvettes) and ray tracing. |
| `optics/presharpen.py` | Compensation of the projector's row blur. |
| `virtual_print.py`, `print_plan.py`, `calibration.py` | Virtual print, print time, exposure calibration. |
| `pixel_game.py` | Mini-games shown while frames are computed. |
| `frame_io.py`, `profiles.py`, `validation.py` | Frame storage and manifests, machine/resin profiles, preflight. |
| `i18n.py` | English / Russian strings. |
| `tests/` | Geometry, optics, engine, storage, interface and security tests. |

## Validation and safety

Before computing, SpinSlicer checks mesh finiteness, triangle count, bounds, fit
in the vial, layer count, memory and output budgets. The computation runs in a
separate process and can be cancelled at any time. Output directories are
committed only after all PNG files and metadata pass validation. Local paths are
opened through Qt's desktop API rather than a shell command.

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
real-time scattered-light feedback. SpinSlicer times the exposure from the planned
dose and stops the projector by itself; camera-based feedback is not implemented yet.
