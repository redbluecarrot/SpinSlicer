# Physical optics validation — v0.3.1

This report concerns numerical correctness of a relative linear dose planner.
It does not validate a printed object or infer resin chemistry from a mesh.

## Confirmed errors in v0.3.0

- Rays started at the resin entrance but were sampled over a distance interval
  centred around that entrance. Most of the far half of the vat was omitted.
  A central test chord of 46 mm produced 23.4375 mm of matrix weight.
- Matrix columns used `y*N+x`, whereas the rasterized target is `[x,y,z]`.
  Asymmetric targets were therefore projected with the wrong axis convention.
- Circular clipping used the grid's radius instead of the resin's inner radius;
  the same circular mask incorrectly clipped square-cuvette corners.
- The vial-in-air mode ignored refractive indices and used ideal straight rays.
- Rounding absorption/cache parameters to four decimals reused a matrix for
  physically different inputs, including alpha=0 and alpha=0.00001 /mm.
- The nonlinear optimizer introduced a sigmoid with fixed slope 15 and threshold
  0.65, normalized by a changing dose maximum, and did not differentiate that
  normalization. Neither coefficient came from resin calibration.
- Export added percentile-based saturation, while the preview used pre-export
  values. Its reported loss therefore did not describe the played PNG frames.
- A 0.5 mm entity outline changed closed section boundaries; Z smoothing also
  modified the target without a corresponding physical sampling model.

## Corrected numerical model

For each refracted ray and each intersected voxel, the matrix coefficient is

`A[r,i] = integral exp(-alpha*l) dl` within the resin segment of that voxel.

Grid-line crossings split the full resin chord into exact segments. A segment
of length h starting at depth d contributes h for alpha=0 and
`exp(-alpha*d) * (-expm1(-alpha*h)) / alpha` otherwise. Depth begins at the resin
entrance, not the external glass or grid boundary. Detector rays pass through
pixel centres. Voxel indices follow `x*N+y`; backprojection is exactly `A.T`.

The delivered relative dose is `d = A.T @ p`. With `B=A.T`, the nonnegative SIRT
step is `p <- max(0, p + C @ B.T @ R @ (target - B @ p))`, where R and C contain
reciprocals of B's row/column sums. Zero-sum rows/columns receive zero weight.
The projected Ram-Lak field is only an initializer, fitted by least squares;
it is not claimed to analytically invert curved rays. Rays that miss the target
are excluded separately for each Z layer. Python and C++ execute the same update.

One global peak normalization preserves relative layer amplitudes. Export then
applies the explicit gray cutoff and user gain, resizes floating gray values,
and quantizes once. There is no percentile clipping or unrequested Z blur.

The preview reconstructs a middle detector row from the actual exported PNGs.
RMSE measures relative dose shape after fitting one nonnegative least-squares
scale. It is resolution independent; it is not a probability of curing. Raw
`reconstruction_dose.npy`, `reconstruction_target.npy`, and `relative_scale`
in `dose_stats.txt` allow the score to be reproduced without interpreting colors.

## Reproduction and measurements

Measurements made on 2026-10-19, Windows 11, Python 3.12.10, an Intel CPU with
12 logical processors. Matrix benchmark: 256×256 grid, 256 detector bins,
360 angles, outer/inner diameters 60/56 mm, equal refractive indices, alpha=0.
Peak memory is Windows process peak resident working set, including Python and
SciPy but excluding the GUI, STL parsing and other processes. MiB = 2^20 bytes.
These observations are not hardware-independent runtime or memory guarantees.

| Measurement | v0.3.0 | Corrected |
| --- | ---: | ---: |
| Matrix construction | 1.69 s | 2.34 s |
| Matrix-phase process peak | 519.1 MiB | 383.7 MiB |
| CSR matrix + transpose payload | 315.1 MiB | 315.4 MiB |
| Nonzero matrix entries | 20,610,856 | 20,633,736 |

The corrected matrix covers the complete chord, so this is not an equal-work
speed comparison. The previous 250 MB claim was not reproduced on this setup;
the matrix pair alone requires approximately 315 MiB here.

The full 256×256×256 target benchmark with 360 angles and 8 native iterations
took 12.63 s for the solver, following the 2.34 s matrix build. Peak process
working set reached 563.9 MiB. Dense temporary buffers are processed in Z
batches; neither grid resolution nor angle count is reduced to obtain this.

An asymmetric ring, offset rectangle and empty layer (64×64, 90 angles,
8 iterations) were evaluated using the corrected operator. Relative dose RMSE
was 0.172716 for v0.3.0 and 0.127877 after correction. These are synthetic,
model-based measurements, not experimental printing accuracy.

Run from the project root, using a separate process for each memory benchmark:

```bash
python tools/build_sirt.py
python -m pytest tests/test_physical_optics.py tests/test_physical_export.py
python tools/benchmark_optics.py --layers 256
python tools/validate_optics.py --baseline-ref v0.3.0
```

On Windows run the native build from an x64 Visual Studio developer shell.
The reference comparison reads the actual historical sources from Git and runs
them in a temporary directory. No reference score is hardcoded in the tool.

## Limits and references

This model assumes parallel incident rays and homogeneous refractive indices
and absorption. Fresnel reflection, scattering, projector radiometric response,
mechanical timing and measured material dose-to-cure response are not included.
Absolute dose and curing require calibration; relative-dose RMSE cannot replace it.

- [Siddon: exact radiological path calculation](https://aapm.onlinelibrary.wiley.com/doi/10.1118/1.595715).
- [ASTRA: SIRT and nonnegative constraints](https://astra-toolbox.com/docs/algs/SIRT.html).
- [PBRT: refraction and Snell's law](https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission).
- [PBRT: transmittance and Beer’s law](https://www.pbr-book.org/4ed/Volume_Scattering/Transmittance).
