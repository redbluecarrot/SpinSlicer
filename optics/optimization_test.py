import time

import numpy as np

from optics.optimization import optimize_projections_sirt_volume

grid_res = 64
nz = 5
num_frames = 360

slices = np.zeros((grid_res, grid_res, nz), dtype=np.float32)
slices[20:44, 20:44, :] = 1.0

t0 = time.time()
b_opt = optimize_projections_sirt_volume(
    slices,
    np.linspace(0, 2*np.pi, num_frames, endpoint=False),
    L_out=60.0, L_in=56.0,
    grid_res=grid_res, grid_span_mm=60.0,
    iterations=4, container_mode=0
)
t1 = time.time()
print(f"Time: {t1-t0:.2f}s")
print("SIRT Result max:", b_opt.max(), "min:", b_opt.min(), "has_nan:", np.isnan(b_opt).any())
