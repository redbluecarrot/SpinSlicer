"""The engine child process must talk to the UI independently of the console code page."""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_child_protocol_survives_a_non_utf8_console(tmp_path):
    # A Russian Windows console uses cp1251; the UI decodes the pipe as UTF-8.
    job = tmp_path / "job.pkl"
    job.write_bytes(pickle.dumps((None, None, {}, None)))
    env = dict(os.environ, PYTHONIOENCODING="cp1251", PYTHONUTF8="0")
    result = subprocess.run(
        [sys.executable, str(ROOT / "engine_process.py"), str(job), str(tmp_path / "cancel")],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, timeout=120,
    )
    lines = result.stdout.decode("utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.startswith("{")]
    assert events[0]["m"] == "Подготовка модели..."
    assert "error" in events[-1]
    assert result.stdout.isascii()
