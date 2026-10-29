"""CSR matrices that live on the GPU and multiply through a Taichi kernel.

The matrices are the same float32 ones the CPU path uses; only the product runs
on the device. Every output element is a sequential sum over one CSR row, so the
result is deterministic (no atomics) and equals SciPy's to float32 rounding.
Importing this module requires Taichi, and ``compute.init_taichi_gpu`` must have
started it first.
"""
from typing import Optional

import numpy as np
import taichi as ti

from compute import taichi_generation


def _build_kernel():
    @ti.kernel
    def spmm(
        indptr: ti.types.ndarray(ndim=1), indices: ti.types.ndarray(ndim=1),  # type: ignore[valid-type]
        data: ti.types.ndarray(ndim=1), x: ti.types.ndarray(ndim=2),  # type: ignore[valid-type]
        y: ti.types.ndarray(ndim=2),  # type: ignore[valid-type]
    ):
        for r, k in ti.ndrange(y.shape[0], y.shape[1]):
            acc = 0.0
            for idx in range(indptr[r], indptr[r + 1]):
                acc += data[idx] * x[indices[idx], k]
            y[r, k] = acc

    return spmm


_kernels: dict[int, object] = {}


def _spmm_kernel():
    """The kernel for the current Taichi start-up (it is rebuilt after a reset)."""
    generation = taichi_generation()
    if generation not in _kernels:
        _kernels.clear()
        _kernels[generation] = _build_kernel()
    return _kernels[generation]


class GpuFailure(RuntimeError):
    """A device error; the caller can redo the work on the CPU."""


def _build_step_kernels():
    @ti.kernel
    def ls_prepare(
        target: ti.types.ndarray(ndim=2), atv: ti.types.ndarray(ndim=2),  # type: ignore[valid-type]
        voxel_norm: ti.types.ndarray(ndim=1), ws: ti.types.ndarray(ndim=2),  # type: ignore[valid-type]
    ):
        for p, k in ti.ndrange(ws.shape[0], ws.shape[1]):
            ws[p, k] = (target[p, k] - atv[p, k]) * voxel_norm[p]

    @ti.kernel
    def window_prepare(
        target: ti.types.ndarray(ndim=2), dose: ti.types.ndarray(ndim=2),  # type: ignore[valid-type]
        voxel_norm: ti.types.ndarray(ndim=1), ws: ti.types.ndarray(ndim=2),  # type: ignore[valid-type]
        level: ti.f32, steepness: ti.f32,
    ):
        for p, k in ti.ndrange(ws.shape[0], ws.shape[1]):
            arg = -steepness * (dose[p, k] - level)
            arg = ti.min(ti.max(arg, -60.0), 60.0)
            ws[p, k] = (target[p, k] - 1.0 / (1.0 + ti.exp(arg))) * voxel_norm[p]

    @ti.kernel
    def update(
        values: ti.types.ndarray(ndim=2), upd: ti.types.ndarray(ndim=2),  # type: ignore[valid-type]
        ray_norm: ti.types.ndarray(ndim=1), hits: ti.types.ndarray(ndim=2),  # type: ignore[valid-type]
    ):
        for r, k in ti.ndrange(values.shape[0], values.shape[1]):
            v = ti.max(values[r, k] + upd[r, k] * ray_norm[r], 0.0)
            if hits[r, k] == 0:
                v = 0.0
            values[r, k] = v

    return ls_prepare, window_prepare, update


_step_kernels: dict[int, tuple] = {}


def _steps():
    generation = taichi_generation()
    if generation not in _step_kernels:
        _step_kernels.clear()
        _step_kernels[generation] = _build_step_kernels()
    return _step_kernels[generation]


class GpuMatrix:
    """``matrix @ dense`` on the device; ``dense`` is (columns, layers) float32."""

    def __init__(self, matrix) -> None:
        csr = matrix.tocsr()
        # Kept so that a device failure in the middle of a job falls back to the CPU.
        self._cpu = csr
        self._broken = False
        self.error: Optional[str] = None  # why the device was given up, for the log
        self.shape = csr.shape
        self.dtype = np.dtype(np.float32)
        self._indptr = self._upload(csr.indptr.astype(np.int32), ti.i32)
        self._indices = self._upload(csr.indices.astype(np.int32), ti.i32)
        self._data = self._upload(csr.data.astype(np.float32), ti.f32)
        self._buffers: dict[tuple[int, int], tuple] = {}

    @staticmethod
    def _upload(array: np.ndarray, dtype):
        field = ti.ndarray(dtype, array.shape)
        field.from_numpy(np.ascontiguousarray(array))
        return field

    def matmul(self, values: np.ndarray) -> np.ndarray:
        if not self._broken:
            try:
                return self._device_matmul(values)
            except Exception as exc:  # driver or out-of-memory trouble: same product on the CPU
                self._broken = True
                self.error = f"{type(exc).__name__}: {exc}"
        return np.asarray(self._cpu @ np.ascontiguousarray(values, dtype=np.float32))

    def _device_matmul(self, values: np.ndarray) -> np.ndarray:
        dense = np.ascontiguousarray(values, dtype=np.float32)
        squeeze = dense.ndim == 1
        if squeeze:
            dense = dense[:, None]
        if dense.shape[0] != self.shape[1]:
            raise ValueError("Matrix and vector shapes do not match.")
        key = (dense.shape[0], dense.shape[1])
        if key not in self._buffers:
            if len(self._buffers) >= 2:  # blocks come in at most two sizes
                self._buffers.clear()
            self._buffers[key] = (
                ti.ndarray(ti.f32, dense.shape),
                ti.ndarray(ti.f32, (self.shape[0], dense.shape[1])),
            )
        x, y = self._buffers[key]
        x.from_numpy(dense)
        _spmm_kernel()(self._indptr, self._indices, self._data, x, y)  # type: ignore[operator]
        result = y.to_numpy()
        return result[:, 0] if squeeze else result

    def __matmul__(self, values: np.ndarray) -> np.ndarray:
        return self.matmul(values)

    @property
    def usable(self) -> bool:
        return not self._broken

    def product_into(self, x, y) -> None:
        """``y = matrix @ x`` on device arrays, with no copy to the host."""
        _spmm_kernel()(self._indptr, self._indices, self._data, x, y)  # type: ignore[operator]

    def release(self) -> None:
        """Drop the device arrays now, in the thread that owns the GPU context."""
        self._buffers.clear()
        self._indptr = self._indices = self._data = None
        self._broken = True  # any later product runs on the CPU copy


class GpuSolver:
    """The SIRT steps of one Z block with every array kept on the device.

    The sparse products and the element-wise parts of both the linear and the
    window step run as kernels, so nothing travels to the host between steps: the
    block goes up once and the result comes back once. The arithmetic is the one
    of the CPU path, and a device error surfaces as ``GpuFailure`` before any
    host data changed, so the caller can redo the block on the CPU.
    """

    def __init__(self, a: GpuMatrix, at: GpuMatrix, ray_norm: np.ndarray, voxel_norm: np.ndarray) -> None:
        self.a, self.at = a, at
        self._ray_norm = GpuMatrix._upload(np.ascontiguousarray(ray_norm, dtype=np.float32), ti.f32)
        self._voxel_norm = GpuMatrix._upload(np.ascontiguousarray(voxel_norm, dtype=np.float32), ti.f32)
        self._buffers: dict[int, tuple] = {}
        self.disabled = False
        self.error: Optional[str] = None

    @property
    def usable(self) -> bool:
        return not self.disabled and self.a.usable and self.at.usable

    def _arrays(self, columns: int) -> tuple:
        if columns not in self._buffers:
            self._buffers.clear()
            rays, pixels = self.a.shape[0], self.a.shape[1]
            self._buffers[columns] = (
                ti.ndarray(ti.f32, (rays, columns)), ti.ndarray(ti.f32, (pixels, columns)),
                ti.ndarray(ti.u8, (rays, columns)), ti.ndarray(ti.f32, (pixels, columns)),
                ti.ndarray(ti.f32, (pixels, columns)), ti.ndarray(ti.f32, (rays, columns)),
            )
        return self._buffers[columns]

    def run(
        self, values: np.ndarray, target: np.ndarray, hits: np.ndarray,
        iterations: int, window_iterations: int, level: float, steepness: float, on_step,
    ) -> np.ndarray:
        """Linear steps, then window steps; ``on_step(kind, index)`` runs before each."""
        columns = values.shape[1]
        try:
            v, t, h, atv, ws, upd = self._arrays(columns)
            v.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
            t.from_numpy(np.ascontiguousarray(target, dtype=np.float32))
            h.from_numpy(np.ascontiguousarray(hits, dtype=np.uint8))
        except Exception as exc:
            raise GpuFailure("could not upload the block") from exc
        ls_prepare, window_prepare, update = _steps()
        schedule = [("ls", i) for i in range(iterations)] + [("window", i) for i in range(window_iterations)]
        for kind, index in schedule:
            on_step(kind, index)  # may raise the caller's cancellation; not a device error
            try:
                self.at.product_into(v, atv)
                if kind == "ls":
                    ls_prepare(t, atv, self._voxel_norm, ws)
                else:
                    window_prepare(t, atv, self._voxel_norm, ws, float(level), float(steepness))
                self.a.product_into(ws, upd)
                update(v, upd, self._ray_norm, h)
                ti.sync()
            except Exception as exc:
                raise GpuFailure("device step failed") from exc
        try:
            return v.to_numpy()
        except Exception as exc:
            raise GpuFailure("could not download the block") from exc

    def release(self) -> None:
        self._buffers.clear()
        self._ray_norm = self._voxel_norm = None
        self.disabled = True
