# How much SpinSlicer uses: CPU, memory, disk (2026-10-31)

All numbers were measured on a weak machine on purpose: **2 vCPU Intel Xeon 2.1 GHz, 8 GB RAM, no GPU**
(OpenGL in software, Mesa llvmpipe). Python 3.13, PyQt 6.11, VTK 9.7.1, PyVista 0.49.2.
On an ordinary laptop with any graphics card the interface figures are lower, because a 3D frame
costs 1–3 ms of GPU time instead of ~160 ms of CPU.

Scripts: `bench_engine.py` (runs the engine in its child process and polls its RSS every 50 ms),
`ui_res.py` (RSS and CPU time of the interface process), `compute_probe.py` (frames per second and
event-loop delays during a compute). The test parts are a 3DBenchy scaled to the vial (46 × 24 × 37 mm,
613 layers at the 0.06 mm projector row pitch) and the M20 test nut.

## The compute (child process)

> Update: after the engine quality changes of 2026-10-31 (finer detector bins, thin-feature
> awareness, see [engine judge](engine-judge-2026-10-31.md)) the same Benchy takes 34 s and
> 1.1 GB at grid 256, 69 s and 1.7 GB at grid 384; the table below is the speed work alone.

The engine runs in its own process at a lower priority (`nice 8`), so the window always gets the CPU
first. Since v0.5.0 it uses every core and up to a fifth of the free memory (at most 1.3 GB) (see "Engine speed-up" below).

| Part | Grid («Качество») | Frames | Time v0.4 → **v0.5** | CPU time | Peak RAM | On disk | IoU v0.4 → v0.5 |
|---|---|---|---|---|---|---|---|
| benchy | 128 «Быстро» | 360 | 12.2 → **9.6 s** | 13.3 s | 471 MB | 23 MB | 0.984 → 0.984 |
| benchy | 256 «Норма» | 360 | 28.1 → **21.0 s** | 30.8 s | 834 MB | 33 MB | 0.989 → 0.989 |
| benchy | 384 «Точно» | 360 | 74.7 → **47.8 s** | 71.3 s | 1308 MB | 28 MB | 0.990 → 0.990 |
| nut M20 | 256 «Норма» | 360 | 17.4 → **15.1 s** | 20.1 s | 975 MB | 51 MB | 0.998 → 0.998 |
| benchy | 256 «Норма» | 720 | 51.1 → **38.2 s** | 55.8 s | 1380 MB | 54 MB | 0.990 → 0.990 |
| **benchy** | **384 «Точно»** | **720** | **140.7 → 68.4 s** | 107.2 s | 1897 MB | 47 MB | 0.990 → 0.991 |

The last row is the heaviest setting the interface offers (finest grid, 0.5° step). Time grows with
the part's height (one layer per projector row), roughly with grid² and with frames^0.65.

### Engine speed-up (v0.5)

What was measured first: the solver (projected L-BFGS over layer batches) took ~80 % of the time,
in three parts — per-frequency complex matrix products, FFTs along the angle, and element-wise numpy
work between them — all on a single core. Changes, each checked against the previous output:

1. **Per-frequency products on all cores** (`optics/fastcal.py: batched_matmul`). numpy runs a stack of
   matrix products one after another; split over threads they scale with cores. BLAS is pinned to one
   thread in the engine process so the two never oversubscribe.
2. **Batch width.** The products run 4× faster per layer in batches of ~24 layers than in batches of 7
   (`(384×178)·(178×Z)` complex: 5.7 ms/layer at Z = 7, 1.5 ms at Z = 32). The old fixed 160 MB budget gave
   7 layers at grid 384 / 720 frames; the budget is now a fifth of free RAM (160–1280 MB). Wider than ~24
   layers is slower again: one L-BFGS step is shared by the whole batch and converges worse.
3. **Independent batches in parallel.** Batches of one coarse-to-fine level do not depend on each other;
   solving two to four at once spreads the element-wise work over the cores as well.
4. **Angular smoothness in frequency space.** The penalty λ·Σ(x[t+1] − x[t])² is circulant: its value comes
   from the spectrum the dose already computes and its gradient 2λ·c_k·X_k is added inside the existing
   inverse FFT. This replaced four `np.roll` passes (8.5 s of 82 s at grid 384). Weights are pre-multiplied,
   BLAS dot products replace a four-operand `einsum`, L-BFGS updates are in place. Relative difference of
   the frames against the old solver: 1.5·10⁻⁵ (float32 rounding).
5. **Voxelization and virtual print** run their independent layer batches on all cores.

The machine used here is memory-bandwidth bound: two threads compete for one memory bus, so going from
1 to 2 threads gains ~15 % at grid 256. A desktop CPU with dual-channel memory and 4–8 cores gains more.

## The window (interface process)

| State | RAM (RSS) | CPU |
|---|---|---|
| Just opened, no part | 506 MB | 0 % |
| Part loaded, nobody touches anything | 782 MB | **0 %** (was 171 %) |
| During a compute, mini-game waiting | — | 52 % (was 113 %) |
| During a compute, mini-game being played | — | 91 % (was 120 %) |
| Print screen, preview playing | 848 MB | 42 % |
| Print screen after the first turn | 848 MB | 0 % |
| Peak over the whole session | 848 MB | — |

RSS counts shared libraries (Qt, VTK, Mesa) in full; most of the 506 MB at start is those libraries,
loaded once. CPU % here is per one core; on a GPU machine the "playing" figures drop several times,
since nearly all of that is the software OpenGL driver.

**Worst case for the whole app at once:** about **2.8 GB RAM** (window ~0.85 GB + compute ~1.9 GB at
«Точно» / 720 frames), all cores for about 70 s on this 2-core machine, **~50 MB** on disk per
print. With the default «Норма» / 360 frames: ~1.7 GB, ~21 s, ~35 MB.

## What was done to get there

1. **The compute moved to a separate process** (`engine_process.py`). Inside one Python process the
   compute thread and the window share the GIL; the window got control only every ~0.7 s with
   stalls up to 6 s. Now the window event loop beats every 10–15 ms (median) during a compute.
2. **The 3D view no longer redraws by itself.** pyvistaqt redraws the whole scene 5×/s by default
   (`auto_update=5.0`); with it off, an untouched window does no work at all.
3. **One frame per change.** `plotter.render()` draws twice (directly and again in Qt's paint);
   every redraw is now a coalesced widget update — several changes in one event-loop turn give one frame.
4. **The hand-drawn "boil" settles.** The line jitter plays while something happens (orbiting,
   editing, a compute) and holds still 4 s after the last change.
5. **The animation budget adapts to the machine.** Each frame's cost is measured as the larger of
   wall time and CPU time of the process (a software driver renders on several threads), and the
   animation is kept to ≤ ~25 % of that budget. Progress from the compute rides those frames
   instead of asking for 15 frames a second.
6. **The rim ticks are one object** whose colours change in place, not dozens of actors rebuilt
   at every percent.
7. **Display mesh is prepared off the window thread** (decimation and normals happen in the loader).
8. **The mini-game draws only when something moves.** A waiting game costs nothing; a running round
   pauses when the window loses focus; on a machine where drawing the window is expensive, the game
   switches to 30 pictures a second with two logic steps each (same speed of play).
9. **The print preview keeps 960-px copies** of the frames (true size is still used for 1:1
   placement on the projector), shows one full turn and stops; ▶ plays in a loop.

## Known limits

- On software OpenGL a single 3D frame takes ~160 ms, so while the drawing animates the event loop
  sometimes waits that long (p99 ≈ 250 ms during a compute). With a GPU this disappears.
- The frames go to disk as PNG at full resolution; long 0.5° runs at «Точно» take ~50 MB.
