"""Central lifecycle management for all background Qt jobs."""
from __future__ import annotations

from time import monotonic

from PyQt6.QtCore import QThread


class JobController:
    """Track workers and stop them before the main window is destroyed."""

    def __init__(self) -> None:
        self._workers: list[QThread] = []

    def register(self, worker: QThread) -> None:
        if worker in self._workers:
            return
        self._workers.append(worker)
        worker.finished.connect(lambda worker=worker: self.unregister(worker))

    def unregister(self, worker: QThread) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    def shutdown(self, timeout_ms: int = 5000) -> bool:
        """Request cancellation and wait for every registered worker."""

        workers = list(self._workers)
        deadline = monotonic() + max(timeout_ms, 0) / 1000.0

        for worker in workers:
            if worker.isRunning():
                request_cancel = getattr(worker, "request_cancel", None)
                if callable(request_cancel):
                    request_cancel()

        all_stopped = True
        for worker in workers:
            if not worker.isRunning():
                continue
            remaining_ms = max(0, int((deadline - monotonic()) * 1000))
            if not worker.wait(remaining_ms):
                all_stopped = False

        return all_stopped
