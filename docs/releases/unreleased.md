# Unreleased

## Export for OpenCAL

- **Full-range H.264.** The MP4 is encoded in full range (black 0, white 255) and flagged
  as such. In the limited range black became 16/255 (about 6 % light on the whole
  turn) and 36 gray levels were lost.
- **Printer file from the PNG frames.** The MP4 is built from the PNG files of the run at
  their original resolution, not from the preview, which is shrunk to 900 px. The
  rotation speed in the file name (`<name>_<rpm>rpm.mp4`) comes from the number of frames
  on disk and the manifest's frame rate, with at most two decimals.
  Fractional values (for example `4.17`) depend on the frame count; whether OpenCAL's file
  name parser accepts a decimal point has not been verified, so prefer a frame count that
  gives a whole number of revolutions per minute (360 frames at 30 fps is 5 rpm).
- **Projector gamma.** The default gamma is now **1.0 (linear)**, where earlier builds
  applied 2.2. Most projectors are close to 2.2, so *without a calibrated gamma the dose
  is non-linear*. Set the projector gamma in the machine profile. For a gamma other than
  1 the slicer encodes the frames in floating point before the single 8-bit rounding and
  stores them in `projector_frames/`; the PNG files in the run and the projector tab's
  preview always stay linear, and gamma reaches only the MP4.
- The frame rate defaults to 30 fps (it divides the 60 Hz projector refresh evenly).
- `output_res` must be even.

## Optimiser

- The dose-window stage is now an **option, off by default** (checkbox "Dose window").
  It is a heuristic on top of the validated linear fit: it changes the objective, roughly
  doubles the optimisation time and is not tied to a measured resin threshold.
  Level and steepness are recorded in the manifest.

## Compute modes

- Auto, Economy and Maximum. The mode changes speed and memory, not the frames for
  printing. Economy does not create the uncorrected projection (and says so in the log).
- Maximum runs on a GPU through Taichi when one is available. The optimiser steps stay on
  the device for a whole Z block. Without a GPU it uses all cores.
- The ray matrix is kept between jobs in Maximum mode and released in Economy mode.
