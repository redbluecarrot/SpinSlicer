from pathlib import Path

import numpy as np
import trimesh

from frame_io import FrameRepository, load_manifest, sha256_file
from slicing_engine import ResinSettings, SliceParams, SlicingEngine


def test_repeated_generation_keeps_completed_runs_and_manifest(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    source = tmp_path / "source.stl"
    mesh.export(source)
    output_root = tmp_path / "output_frames"
    params = SliceParams(
        diameter_mm=60.0,
        grid_res=16,
        output_res=32,
        num_frames=8,
        fill_holes=True,
        resin=ResinSettings(),
        output_dir=str(output_root),
    )
    context = {
        "source_path": str(source),
        "transform_matrix": np.eye(4, dtype=np.float64).tolist(),
        "units": "mm",
    }

    _, first_run = SlicingEngine.run(mesh, params, manifest_context=context)
    _, second_run = SlicingEngine.run(mesh, params, manifest_context=context)

    runs = sorted(path for path in output_root.iterdir() if path.is_dir())
    assert len(runs) == 2
    assert first_run != second_run
    assert all((path / "manifest.json").exists() for path in runs)

    info = FrameRepository.validate(str(output_root))
    manifest = load_manifest(info.resolved_dir)
    assert info.frame_count == 8
    assert manifest is not None and manifest.complete
    assert manifest.source_name == Path(source).name
    assert manifest.source_sha256 == sha256_file(str(source))
    assert manifest.transform_matrix[0][0] == 1.0
    assert manifest.machine_profile["name"] == "offline-unbound"
    assert manifest.resin_profile["base_exposure_s"] == 1.0
    assert manifest.frame_schedule["angles_deg"] == [90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 0.0, 45.0]
