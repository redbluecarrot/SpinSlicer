"""Exercise the stable GUI, worker pipeline and MP4 encoding in a clean process."""
import json
import os
import subprocess
import sys
from pathlib import Path


def test_stable_release_has_no_simulator_and_exports_valid_video(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYVISTA_OFF_SCREEN": "true", "PYTHONUTF8": "1"}
    environment.pop("DISPLAY", None)
    result = subprocess.run([sys.executable, str(root / "SpinSlicer.py"), "--self-check", str(tmp_path)],
                            cwd=root, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    report = json.loads((tmp_path / "self-check.json").read_text(encoding="utf-8"))
    assert report["ok"], report
    assert report["video_bytes"] > 0
    assert report["tabs"] == ["SlicerTab", "ProjectorTab", "ProcessSettingsPanel"]
    assert not report["simulation_loaded"]
