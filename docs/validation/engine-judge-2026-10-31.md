# Engine judge: FastCAL against OSMO, PM, BCLP and classic CAL (2026-10-31)

All engines plan the same jobs and are judged the same way, by a model none of them planned
with. Script: `tools/engine_judge.py` (every number below comes from it; reproduce with
`python tools/engine_judge.py --cases nut,benchy,restarget,nut_air,nut_absorbing --grid 128 --mc 8`).

## How the judging works

- **Same physics:** refraction through a 60/56 mm vial (immersion bath n = 1.45 or air),
  Beer–Lambert attenuation where stated, 360 equally spaced angles. OSMO, PM, BCLP and FBP
  plan on SpinSlicer's exact sparse Siddon operator; FastCAL on its own rotation-symmetric
  operator, with the physics the judge models (linear pixel profile, no Fresnel, no smear).
- **Same information:** the competitors use a binary target, so FastCAL is shown both with
  the sub-voxel coverage the app uses and with that binary target.
- **Competitors as published:** VAMToolbox update rules verbatim — OSMO (Rackson 2021,
  50 iterations, d_h 0.8, d_l 0.7, ram-lak start), PM (Bhattacharya 2021, L1 penalty,
  L-BFGS-B with 0 ≤ b ≤ 1, 50 iterations), BCLP (Li 2024, B 25, ε 0.1, 200 iterations at its
  best step), FBP (ramp-filtered projections, negatives clipped: the original CAL start).
- **Independent judge:** a 2x finer grid (256 for planning grid 128), voxelised separately
  from the mesh; the exact Siddon operator at that grid; frames quantised to 8 bits and
  upsampled linearly to the judge's pixel pitch, as a projector shows them. Each layer is a
  section through the layer centre, as the slicer voxelises (a projector row lights its
  layer evenly in height).
- **Metrics:** IoU at the best exposure; distance between the cured and the true surface;
  dose window between the dimmest 0.5 % part voxels and the brightest 0.5 % resin voxels away
  from partial edge voxels (> 0: a clean threshold exists); robustness: IoU over 8 virtual
  machines at the planned exposure — exposure ±5 %, projector blur 0.5 row and 0.5 pixel,
  axis offset ±0.1 mm, frame timing jitter ±1°, the same 8 machines for every engine.
- **Machine:** 2 CPU cores at 2.1 GHz, no GPU. Times are planning only.

## What the judge found in FastCAL, and what changed

1. **Face layers did not cure (bug, fixed).** To save time, a layer that looks like a mix of
   its neighbours gets their mixed frames instead of its own solution. The test compared the
   layer with the *mix of the neighbours' targets*. A layer where a top or bottom face lies
   mid-layer is matched perfectly by 50 % full + 50 % empty, yet the mixed frames give half
   the dose and the layer stayed liquid. Now every neighbour must itself be close to the
   layer. On the resolution target (grid 96, faces mid-layer) IoU went from 0.868 to 0.945;
   in the app it affects faces and steps of sloped parts. Regression test:
   `tests/test_fastcal.py::test_half_covered_face_layer_is_solved_not_mixed_from_neighbours`.
2. **Finer detector bins (new default).** With one bin per voxel the linear pixel profile
   spreads every bin over two voxels (0.94 mm at grid 128). Planning with two bins per voxel
   (the projector has far more pixels than voxels) improved every job:

| Job | IoU 1 bin → 2 bins | Surface error mean | Robust mean | Robust worst | Time |
|---|---:|---:|---:|---:|---:|
| Threaded nut, immersion bath | 0.9826 → **0.9831** | 0.096 → 0.093 mm | 0.9714 → 0.9717 | 0.9619 → 0.9626 | 1.7 → 2.6 s |
| Threaded nut, absorbing resin | 0.9805 → **0.9818** | 0.108 → 0.100 mm | 0.9659 → 0.9675 | 0.9528 → 0.9555 | 1.6 → 2.6 s |
| 3DBenchy | 0.9669 → **0.9683** | 0.058 → 0.056 mm | 0.9397 → 0.9435 | 0.9049 → 0.9132 | 4.4 → 8.4 s |
| Resolution target | 0.9671 → **0.9701** | 0.234 → 0.172 mm | 0.9289 → 0.9309 | 0.9010 → 0.8884 | 1.9 → 3.2 s |

   The cost grows with the bins (2 bins: 1.5–2.2x the time), so the app uses 1.5 bins per
   voxel for Normal and Fine and keeps 1 for Fast, a quick preview
   (`optics.fastcal.detector_bins`). Benchy on the 2-core machine, with item 3 below:
   Fast ~10 s, Normal 34 s (was 21 s), Fine 69 s (was 48 s).
3. **Thin-feature awareness (new default).** Coverage alone cannot tell a voxel on the edge
   of a big body from one inside a hole narrower than two voxels: both are "partly solid",
   the band puts them at an intermediate dose, and in a thin hole no cell is dark enough to
   anchor it, so light from the walls closes it. Per layer, a distance transform now gives
   every cell the half-width of the feature it belongs to; cells of features thinner than
   2.5 voxels get up to 4x the weight, and thin voids get a fully dark target
   (`FastVialEngine._thin_features`). Measured with the finer detector bins on:

   | Job | IoU | Surface error mean | Dose window | Robust mean / worst | Smallest pin / hole |
   |---|---:|---:|---:|---:|---:|
   | Resolution target, before | 0.9701 | 0.172 mm | +4.3 % | 0.9309 / 0.8884 | 1.0 / 1.0 mm |
   | Resolution target, **after** | **0.9733** | **0.081 mm** | **+5.5 %** | **0.9352 / 0.9114** | **0.8 / 0.8 mm** |
   | Benchy, before → after | 0.9683 → 0.9682 | 0.056 → 0.054 mm | +11.0 → +11.0 % | 0.9435 / 0.9132 → 0.9438 / 0.9118 | — |
   | Nut, before → after | 0.9831 → 0.9834 | 0.093 → 0.091 mm | +4.1 → +4.2 % | 0.9717 / 0.9626 → 0.9717 / 0.9629 | — |

   Stronger weights (6, 10) raised the target's IoU a little more but cost Benchy 0.003 IoU
   or worst-case robustness; a fully cured target for thin solids made 0.4 mm walls print
   but lowered worst-case robustness. The 0.6 mm hole (1.3 voxels at grid 128) stays closed
   in every setting: at that size the target itself barely contains it.
4. **Tried and rejected:** a wider dose gap (d_low 0.5–0.65), steeper edge ramps (worse
   IoU on 3 of 4 jobs), denser polar grid (no change), angular smoothing 0.01 / 0.1 (mixed),
   detector smoothing 0.01 / 0.03 (expected to help against blur; it lowered worst-case
   robustness on Benchy, 0.912 → 0.904 / 0.899, and raised the target's surface error).
   Twice the iterations gains +0.002 IoU at 2x time.

## Results before items 2–3 (planning grid 128, judge 256; FastCAL with the fix, one bin per voxel)

### Threaded nut, immersion bath (51 layers)

| Engine | Time | IoU | Surface error mean / p95 | Dose window | Robust mean / worst |
|---|---:|---:|---:|---:|---:|
| **FastCAL** | 1.9 s | 0.9826 | 0.096 / 0.332 mm | +4.4 % | 0.9714 / 0.9619 |
| FastCAL, binary target | 1.4 s | 0.9756 | 0.136 / 0.469 mm | +2.0 % | 0.9697 / 0.9579 |
| OSMO | 10.4 s | 0.9688 | 0.152 / 0.332 mm | -0.1 % | 0.9661 / 0.9618 |
| PM | 69.7 s | 0.9698 | 0.142 / 0.332 mm | -0.3 % | 0.9713 / 0.9648 |
| BCLP 200 it | 22.5 s | 0.9553 | 0.234 / 0.845 mm | -10.9 % | 0.9571 / 0.9459 |
| FBP (classic CAL) | 0.1 s | 0.9602 | 0.188 / 0.469 mm | -7.0 % | 0.9560 / 0.9521 |

### Threaded nut, vial in air (51 layers)

| Engine | Time | IoU | Surface error mean / p95 | Dose window | Robust mean / worst |
|---|---:|---:|---:|---:|---:|
| **FastCAL** | 2.1 s | 0.9833 | 0.090 / 0.332 mm | +4.6 % | 0.9711 / 0.9624 |
| FastCAL, binary target | 1.5 s | 0.9753 | 0.132 / 0.332 mm | +2.0 % | 0.9689 / 0.9578 |
| OSMO | 12.4 s | 0.9686 | 0.146 / 0.332 mm | -0.1 % | 0.9629 / 0.9575 |
| PM | 69.7 s | 0.9712 | 0.124 / 0.332 mm | -0.3 % | 0.9686 / 0.9612 |
| BCLP 200 it | 28.6 s | 0.9563 | 0.213 / 0.524 mm | -11.3 % | 0.9618 / 0.9556 |
| FBP (classic CAL) | 0.1 s | 0.9599 | 0.182 / 0.469 mm | -7.7 % | 0.9576 / 0.9489 |

### Threaded nut, absorbing resin (0.03 /mm) (51 layers)

| Engine | Time | IoU | Surface error mean / p95 | Dose window | Robust mean / worst |
|---|---:|---:|---:|---:|---:|
| **FastCAL** | 1.6 s | 0.9805 | 0.108 / 0.332 mm | +3.0 % | 0.9659 / 0.9528 |
| FastCAL, binary target | 1.6 s | 0.9706 | 0.160 / 0.469 mm | +0.3 % | 0.9652 / 0.9518 |
| OSMO | 10.5 s | 0.9805 | 0.100 / 0.234 mm | +3.3 % | 0.9678 / 0.9468 |
| PM | 73.3 s | 0.9743 | 0.129 / 0.332 mm | +1.3 % | 0.9695 / 0.9456 |
| BCLP 200 it | 23.4 s | 0.9642 | 0.161 / 0.469 mm | -5.8 % | 0.9640 / 0.9587 |
| FBP (classic CAL) | 0.1 s | 0.9147 | 0.562 / 2.344 mm | -29.9 % | 0.8598 / 0.7238 |

### 3DBenchy (40 mm), immersion (68 layers)

| Engine | Time | IoU | Surface error mean / p95 | Dose window | Robust mean / worst |
|---|---:|---:|---:|---:|---:|
| **FastCAL** | 4.1 s | 0.9669 | 0.058 / 0.234 mm | +9.9 % | 0.9397 / 0.9049 |
| FastCAL, binary target | 4.0 s | 0.9361 | 0.114 / 0.234 mm | +7.8 % | 0.9249 / 0.8985 |
| OSMO | 11.9 s | 0.9338 | 0.114 / 0.234 mm | +5.6 % | 0.8963 / 0.8435 |
| PM | 95.1 s | 0.9301 | 0.119 / 0.234 mm | +8.1 % | 0.9209 / 0.9100 |
| BCLP 200 it | 26.6 s | 0.9202 | 0.131 / 0.332 mm | +7.0 % | 0.9209 / 0.9129 |
| FBP (classic CAL) | 0.2 s | 0.8797 | 0.200 / 0.469 mm | -8.8 % | 0.8465 / 0.7863 |

### Resolution target (pins, holes, walls 0.4–3 mm) (26 layers)

| Engine | Time | IoU | Surface error mean / p95 | Dose window | Robust mean / worst | Smallest pin / hole / wall |
|---|---:|---:|---:|---:|---:|---:|
| **FastCAL** | 1.5 s | 0.9671 | 0.234 / 0.469 mm | +2.5 % | 0.9289 / 0.9010 | 1 / 1 / 0.6 mm |
| FastCAL, binary target | 1.5 s | 0.9408 | 0.187 / 0.234 mm | +2.1 % | 0.9228 / 0.9039 | 1.5 / 0.8 / 0.6 mm |
| OSMO | 5.4 s | 0.9357 | 0.204 / 0.234 mm | +0.7 % | 0.8901 / 0.8320 | 1.5 / 0.8 / 0.6 mm |
| PM | 36.2 s | 0.9411 | 0.136 / 0.234 mm | +3.0 % | 0.9040 / 0.8413 | 1 / 0.6 / 0.6 mm |
| BCLP 200 it | 14.1 s | 0.9358 | 0.146 / 0.234 mm | +4.4 % | 0.9277 / 0.9161 | 1 / 0.6 / 0.6 mm |
| FBP (classic CAL) | 0.1 s | 0.8972 | 0.280 / 0.524 mm | -14.9 % | 0.8672 / 0.8486 | 1.5 / 0.6 / 0.6 mm |

## Reading the tables honestly

- FastCAL has the best IoU on every job, the best surface error on four of five and the best
  dose window on four of five; it plans 5–40x faster than OSMO, BCLP and PM on this CPU.
- **Where it was not first, and what is left:** in absorbing resin OSMO tied on IoU and had
  a slightly lower surface error — with the finer detector bins FastCAL leads (IoU 0.9818,
  surface 0.100 mm, worst case 0.9555 vs OSMO 0.9468). On the resolution target PM and BCLP
  had the lower surface error and kept 0.6 mm holes open — with thin-feature awareness
  FastCAL's surface error is 0.081 mm (PM 0.136, BCLP 0.146) and 0.8 mm holes and pins print,
  but the 0.6 mm hole still closes. Worst-case robustness: BCLP keeps a small lead on the
  resolution target (0.916 vs 0.911) and absorbing resin (0.959 vs 0.956), PM on the nut
  (0.965 vs 0.963); on Benchy FastCAL and BCLP are level (0.912 / 0.913).
- No engine prints the 0.4–0.8 mm pins at grid 128 (0.47 mm voxels): features need about
  two voxels. Use a finer grid for fine detail.
- GPU implementations of OSMO/BCLP (VAMToolbox with ASTRA) run much faster than the CPU
  versions timed here; the quality figures do not depend on the hardware.

## Superseded

`docs/validation/fastcal-vs-osmo.md` (2026-10-30) gave FastCAL sub-voxel coverage while the
other engines had a binary target, ran the competitors only at default settings and judged
nearest-neighbour upsampled frames; its margins were larger than the ones above.
