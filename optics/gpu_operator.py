"""CSR matrices that live on the GPU and multiply through a Taichi kernel.

The matrices are the same float32 ones the CPU path uses; only the product runs
on the device. Every output element is a sequential sum over one CSR row, so the
result is deterministic (no atomics) and equals SciPy's to float32 rounding.
Importing this module requires Taichi, and ``compute.init_taichi_gpu`` must have
started it first.
"""
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


class GpuMatrix:
    """``matrix @ dense`` on the device; ``dense`` is (columns, layers) float32."""

    def __init__(self, matrix) -> None:
        csr = matrix.tocsr()
        # Kept so that a device failure in the middle of a job falls back to the CPU.
        self._cpu = csr
        self._broken = False
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
            except Exception:  # driver or out-of-memory trouble: same product on the CPU
                self._broken = True
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

    def release(self) -> None:
        """Drop the device arrays now, in the thread that owns the GPU context."""
        self._buffers.clear()
        self._indptr = self._indices = self._data = None
        self._broken = True  # any later product runs on the CPU copy
