import threading

import pytest

PyQt6 = pytest.importorskip("PyQt6")
from PyQt6.QtCore import QCoreApplication, QThread  # noqa: E402

from job_controller import JobController  # noqa: E402


class CancellableThread(QThread):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self.cancel_requested.set()

    def run(self) -> None:
        while not self.cancel_requested.wait(0.01):
            pass


@pytest.fixture(scope="module")
def qt_app():
    return QCoreApplication.instance() or QCoreApplication([])


def test_shutdown_requests_and_waits_for_every_worker(qt_app) -> None:
    controller = JobController()
    first = CancellableThread()
    second = CancellableThread()
    controller.register(first)
    controller.register(second)
    first.start()
    second.start()

    assert controller.shutdown(timeout_ms=1000)
    assert not first.isRunning()
    assert not second.isRunning()
    assert first.cancel_requested.is_set()
    assert second.cancel_requested.is_set()
