# Engine stress test — FastCAL vs SIRT compute modes vs OSMO (2026-10-30)

## What was compared

| Name | What it is |
| --- | --- |
| FastCAL | `optics/fastcal.py`: rotation-symmetric (block-circulant) operator, band objective, projected L-BFGS, coarse-to-fine Z. Run with `LEGACY_OPTICS` so its physics equals the judge's. |
| SIRT economy / maximum | The sparse-matrix SIRT of v0.3.2, as the economy (1 thread, lean matrix) and maximum (all cores, C++) compute modes of the v0.4.0 fork. Both give identical frames. The GPU mode (Taichi) runs the same arithmetic on a GPU and was not available in the test machine. |
| SIRT + dose window | Maximum mode plus 8 sigmoid "dose window" steps. |
| OSMO | VAMToolbox `minimizeOSMO`, verbatim update rule and defaults (50 iterations, d_h 0.8, d_l 0.7, ram-lak target filter), with SpinSlicer's exact sparse Siddon operator (refraction included) in place of ASTRA. |

Targets: threaded nut 36 × 24 mm offset 5 mm from the axis, grid 256, 288 layers, sliced
with `voxelizer.py`; a synthetic stress volume (gear, cross, 0.6 mm rings, pins near the wall,
hollow box) whose shape changes every 9 layers, 90 layers. 360 frames, vial 60/56 mm.
`imm` = immersion bath (n 1.45), `air` = vial in air.

Judge: the exact Cartesian Siddon operator on a **2× finer grid (512)**, applied to 8-bit
frames, against a 512 target sliced independently. No engine optimises on this grid, so no
engine is judged by its own model. Machine: 2 CPU cores, 7 GB RAM, no GPU.

## Results

| Case | Engine | Time | Peak RAM | IoU | Wrong voxels | Exposure window | VER |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nut, imm | **FastCAL** | **15.9 s** | **566 MB** | **0.9912** | **0.31 %** | +3.6 % | 0.026 |
| | OSMO | 182.9 s | 1116 MB | 0.9898 | 0.36 % | **+4.1 %** | **0.021** |
| | SIRT + window | 44.6 s | 1600 MB | 0.9808 | 0.69 % | +0.6 % | 0.263 |
| | SIRT max / eco | 32.9 / 57.5 s | 1599 / 658 MB | 0.9405 | 2.17 % | −9.3 % | 0.571 |
| nut, air | **FastCAL** | **14.3 s** | **567 MB** | **0.9766** | **0.83 %** | **−0.5 %** | **0.061** |
| | OSMO | 183.6 s | 1164 MB | 0.9293 | 2.53 % | −13.9 % | 0.130 |
| | SIRT + window | 49.7 s | 1699 MB | 0.9666 | 1.20 % | −3.8 % | 0.099 |
| | SIRT max / eco | 34.0 / 63.1 s | 1700 / 708 MB | 0.8559 | 5.70 % | −34.6 % | 0.293 |
| stress, imm | **FastCAL** | 17.9 s | **365 MB** | **0.9905** | **0.17 %** | **+11.9 %** | **0.040** |
| | OSMO | 67.6 s | 664 MB | 0.9786 | 0.39 % | +3.9 % | 0.048 |
| | SIRT + window | 17.6 s | 959 MB | 0.9414 | 1.10 % | −7.1 % | 0.119 |
| | SIRT max | 12.3 s | 960 MB | 0.8274 | 3.29 % | −134.9 % | 0.370 |

IoU and wrong voxels at the best single threshold. Exposure window: gap between the 0.5 %
dimmest in-part and 0.5 % brightest out-of-part voxel (away from partial-volume edges),
relative to the in-part level; positive means one exposure prints the part cleanly.
VER (VAMToolbox): fraction of void voxels at or above the dimmest gel voxel.

## Conclusions

- The three compute modes of the fork are one algorithm; they differ in speed and memory only.
  The SIRT objective (least squares on dose) is the limit, not the hardware.
- The dose window helps SIRT a lot but stays behind OSMO and FastCAL and is slower than FastCAL.
- FastCAL matches OSMO on the index-matched nut (better IoU and wrong-voxel rate, slightly
  narrower exposure window and VER) and beats it clearly in air and on the stress volume,
  12–13× faster with half the memory, on a CPU. OSMO keeps an edge on single-voxel
  worst-case metrics in the index-matched case.
- FastCAL tuning from this study: 30 iterations for layers solved from scratch, longer
  warm-start tail `((16,30),(8,10),(4,6),(2,4),(1,3))`, soft in-part cap 1.1, float16 frame store.

## Round 2 — FastCAL v2 (same day)

Changes, each kept only if the 512 judge agreed:

- **Two-grid start.** Layers solved from scratch are first solved at N/2 (a quarter of the
  work), the frames are upsampled and rescaled, then 8 iterations at full N finish them.
  Faster *and* better than 30 iterations from a cold start.
- **Best-neighbour warm start.** Each layer picks the best of three starts (interpolated,
  lower neighbour, upper neighbour). Layers that barely change (≤ 2 %) and sit at stride 1
  copy interpolated frames directly; dose is linear in the frames.
- **Operator cache** on disk (`~/.cache/spinslicer/fastcal`, keyed by geometry and optics;
  `SPINSLICER_CACHE=""` disables it). Repeat jobs skip ray tracing and the step-size estimate.
- **Lean memory.** uint8 voxel target straight from the voxelizer, float16 frame store,
  per-frequency least squares solved on the fly instead of a second operator.
- **Parallel PNG export** (up to 4 threads).
- **Preflight** now counts FastCAL's real footprint, so grids 384 and 512 are allowed on a
  round vial (the UI slider goes to 512).

### Quality, 512 judge

| Case | Engine | IoU | Wrong voxels | Exposure window | VER | Strict PW |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nut, imm | **FastCAL v2** | **0.9936** | **0.23 %** | +3.9 % | 0.033 | −0.140 |
| | FastCAL v1 | 0.9912 | 0.31 % | +3.6 % | 0.026 | −0.148 |
| | OSMO | 0.9895 | 0.37 % | **+4.1 %** | **0.027** | **−0.101** |
| nut, air | **FastCAL v2** | **0.9800** | **0.71 %** | **+0.2 %** | **0.042** | **−0.123** |
| | OSMO | 0.9301 | 2.49 % | −13.8 % | 0.143 | −0.211 |
| stress, imm | **FastCAL v2** | **0.9910** | **0.16 %** | **+11.7 %** | **0.017** | −0.168 |
| | OSMO | 0.9786 | 0.39 % | +3.9 % | 0.048 | **−0.151** |

Grid 384 (nut, imm): IoU 0.9946, wrong voxels 0.19 %, window +4.1 %.

### Print-success Monte Carlo

24 virtual prints per engine (every 6th layer) with random exposure (±10 %), axis offset (±0.1 mm) and
angle phase (±0.5°); a print passes at IoU ≥ 0.97 and ≤ 1 % wrong voxels.

| Case | Engine | Pass rate | Mean IoU | Worst IoU |
| --- | --- | ---: | ---: | ---: |
| nut, imm | FastCAL v2 | 33 % | 0.963 | 0.923 |
| | OSMO | 33 % | 0.952 | 0.874 |
| nut, air | FastCAL v2 | 25 % | — | — |
| | OSMO | 0 % | — | — |

Exposure error dominates every failure: within ±3–4 % of the right dose both engines pass
almost always. That is a property of the resin threshold, not of the frames, so the remedy
is calibration (an exposure ladder) and stopping on camera feedback, not more iterations.

### Whole pipeline (button → PNG frames), nut, 360 frames, 720 px, 2 cores

| | v0.3.2 SIRT | FastCAL v1 | **FastCAL v2** |
| --- | ---: | ---: | ---: |
| grid 256, first run | 33–58 s¹ | 20.8 s | **13.5 s** |
| grid 256, repeat (cache) | — | — | **11.8 s** |
| grid 256, one core | 58 s¹ | — | **13.1 s** |
| peak RAM, grid 256 | 0.66–1.6 GB | 569 MB | **491 MB** |
| grid 384 | blocked | — | 34 s, 781 MB |
| grid 512 (1024 px) | blocked | — | 91 s, 1.4 GB |

¹ Optimiser only (fork's maximum / economy modes); slicing and export come on top.

## Round 3 — clean frames (same day)

Complaint: on a 3DBenchy the frames looked noisy, blocky and flickered when played.

Cause: the band objective only asks the dose to stay inside a band, so many frame sets
are equally good. The solver landed on a rough one: frames changed by ~29 % between
neighbouring angles (the plain projection of the part changes by ~2 %). On organic parts
almost every layer was also re-planned from scratch (target change > 3 %), so neighbouring
rows came from unrelated solutions: blocks.

Fix:
- `smooth_theta = 0.03` (default): a small penalty on frame change between neighbouring
  angles, in units of the dose Lipschitz constant. It selects the smooth solution among
  equally good ones.
- `change_tol = 0.15`: sloped layers start from their neighbours instead of from scratch.

Frame cleanliness (mean change between neighbouring angles, relative to the mean frame):

| Part | before | after | OSMO | plain projection |
| --- | ---: | ---: | ---: | ---: |
| nut, imm | 0.188 | **0.024** | 0.068 | 0.006 |
| stress, imm | 0.350 | **0.092** | 0.076 | 0.025 |
| Benchy, imm | 0.288 | **0.062** | 0.075 | 0.021 |
| Benchy, app PNG frames (air) | 0.138 | **0.042** | — | — |

Quality, 512 judge:

| Part | Engine | IoU | Wrong voxels | Window | VER | Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nut, imm | **v3** | **0.9946** | **0.19 %** | 4.0 % | 0.031 | 10.7 s |
| | v2 | 0.9936 | 0.23 % | 3.9 % | 0.033 | 10.8 s |
| | OSMO | 0.9895 | 0.37 % | 4.1 % | 0.027 | 183 s |
| nut, air | **v3** | **0.9816** | **0.66 %** | **+0.5 %** | 0.033 | 10.3 s |
| | v2 | 0.9800 | 0.71 % | +0.2 % | 0.042 | |
| stress, imm | v3 | 0.9895 | 0.19 % | 8.9 % | 0.020 | 5.3 s |
| | v2 | **0.9910** | **0.16 %** | **11.7 %** | **0.017** | 5.1 s |
| Benchy, imm | **v3** | **0.9816** | **0.14 %** | 9.0 % | 0.017 | 10.0 s |
| | v2 | 0.9812 | 0.15 % | **10.2 %** | **0.012** | 14.0 s |
| | OSMO | 0.9672 | 0.26 % | 7.4 % | 0.724 | 149 s |

With projector blur (0.7 row, 0.5 pixel), exposure ±5 %, axis ±0.1 mm and per-frame timing
jitter ±1° (12 virtual prints, Benchy): mean IoU v3 0.958, v2 0.963, OSMO 0.934. The
smooth frames cost about half a point of IoU in this stress test on organic parts and gain
it back on the nut.

Rejected in this round: smoothing along the detector (blurs thin walls); red-black
coherence passes between layers (fewer seams, but −0.7 % IoU and +40 % time); solving every
second or third projector row and interpolating (−0.4 to −0.6 % IoU at the row pitch).

App pipeline, Benchy at 75 %, grid 256, 360 frames, 720 px: 29.3 s → 21.1 s.

## Virtual print (same day)

Without a printer the frames are checked by a virtual print (`virtual_print.py`): the
frames as displayed (8-bit, gamma, contrast) go through the FastCAL optical model into a
whole-volume dose; the resin cures above the gel threshold. The exposure is swept from 0.80
to 1.25 of the window-centre dose: the recommended exposure is the one with the best IoU,
and the tolerance is the range where the mean surface offset ((cured − model volume) / model
surface) stays within half a voxel. The print plan uses both. Cost: about 3 s on a 432-layer
Benchy, no extra peak memory.

| Job (app, 256 grid, 360 frames, 720 px) | IoU | Extra / missing volume | ±5 % exposure: surface offset | Tolerance |
| --- | ---: | ---: | ---: | ---: |
| Benchy 75 %, immersion | 98.8 % | 0.78 / 0.43 % | −0.07 / +0.09 mm | ±6.1 % |
| Benchy 75 %, air vial | 98.2 % | 0.82 / 0.99 % | −0.08 / +0.08 mm | ±6.3 % |

The window-centre exposure over-cured the air-vial Benchy by 6 % of its volume; the
virtual print moves the recommended time 5 % lower. The virtual print uses the planning
model, so it checks the export chain and the dose plan, not the model itself; the model was
checked against the independent 512 judge above.

Detector-direction smoothing was retried with an edge-preserving (Huber/TV) penalty to
remove the remaining thin vertical stripes: −0.05 to −0.6 % IoU on the 512 judge, so the
stripes stay. They are how the frames make sharp dose edges.
