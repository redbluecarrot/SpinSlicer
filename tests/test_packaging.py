"""Checks of the packaging pieces that cannot be built here (the installer is compiled by CI)."""
import importlib.util
import sys
from pathlib import Path

import pytest

import compute

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ the add-on folder
def test_a_gpu_addon_folder_becomes_importable_after_the_program_libraries(tmp_path, monkeypatch) -> None:
    (tmp_path / "spinslicer_fake_addon.py").write_text("VALUE = 7\n", encoding="utf-8")
    monkeypatch.setenv(compute.GPU_ADDON_ENV, str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert compute.enable_gpu_addon()
    assert sys.path[-1] == str(tmp_path)            # last: the program's own numpy always wins
    assert importlib.util.find_spec("spinslicer_fake_addon") is not None


def test_no_gpu_addon_means_nothing_changes(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(compute.GPU_ADDON_ENV, raising=False)
    before = list(sys.path)
    assert compute.gpu_addon_directory() is None
    assert not compute.enable_gpu_addon()
    assert sys.path == before
    monkeypatch.setenv(compute.GPU_ADDON_ENV, str(tmp_path / "missing"))
    assert compute.gpu_addon_directory() is None


def test_the_installed_gpu_folder_is_found_next_to_a_packaged_program(tmp_path, monkeypatch) -> None:
    (tmp_path / "gpu").mkdir()
    monkeypatch.delenv(compute.GPU_ADDON_ENV, raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "SpinSlicer.exe"))
    assert compute.gpu_addon_directory() == (tmp_path / "gpu").resolve()


# ------------------------------------------------------------------ the add-on builder
def test_the_addon_builder_drops_numpy_and_tools_but_keeps_taichi(tmp_path) -> None:
    builder = _load(ROOT / "packaging" / "build_gpu_addon.py", "build_gpu_addon")
    for name in ("taichi", "taichi-1.7.4.dist-info", "rich", "numpy", "numpy.libs", "numpy-2.5.4.dist-info",
                 "bin", "include", "share", "SPIRV-Tools-opt"):
        (tmp_path / name).mkdir()
    (tmp_path / "taichi" / "__pycache__").mkdir()
    removed = builder.prune(tmp_path)
    kept = {entry.name for entry in tmp_path.iterdir()}
    assert kept == {"taichi", "taichi-1.7.4.dist-info", "rich"}
    assert {"numpy", "numpy.libs", "bin", "include", "share", "SPIRV-Tools-opt"} <= set(removed)
    assert not (tmp_path / "taichi" / "__pycache__").exists()


def test_every_addon_package_is_pinned_so_nothing_is_resolved() -> None:
    lines = [
        line.strip() for line in (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines and all("==" in line for line in lines)
    assert not any(line.lower().startswith("numpy") for line in lines)
    assert any(line.startswith("taichi==") for line in lines)


def test_taichi_is_not_a_required_dependency() -> None:
    required = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "taichi" not in required
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    required_block, _, optional_block = project.partition("[project.optional-dependencies]")
    assert "taichi" not in required_block.split("[tool")[0]
    assert "taichi" in optional_block
    assert '"taichi"' in (ROOT / "packaging" / "SpinSlicer.spec").read_text(encoding="utf-8")  # excluded


# ------------------------------------------------------------------ the installer script
@pytest.fixture(scope="module")
def installer() -> str:
    raw = (ROOT / "packaging" / "SpinSlicer.iss").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "needs a BOM: it contains Russian text"
    return raw.decode("utf-8-sig")


def test_installer_offers_the_gpu_addon_as_an_optional_component(installer) -> None:
    assert "[Components]" in installer and 'Name: "gpu"' in installer
    assert 'Name: "app"' in installer and "Flags: fixed" in installer     # the program is not optional
    gpu_files = [line for line in installer.splitlines() if r"dist\gpu" in line and line.startswith("Source")]
    assert gpu_files and "Components: gpu" in gpu_files[0] and "skipifsourcedoesntexist" in gpu_files[0]
    program_files = [line for line in installer.splitlines() if "{#MyAppExeName}\"; DestDir" in line]
    assert program_files and "Components: app" in program_files[0]


def test_installer_explains_taichi_in_both_languages_as_a_suggestion(installer) -> None:
    for language in ("english", "russian"):
        for key in ("CompGpu", "GpuPageCaption", "GpuPageDescription", "GpuPageText",
                    "GpuPageDetected", "GpuPageNotDetected", "TypeFull", "TypeCompact", "TypeCustom"):
            assert f"{language}.{key}=" in installer, f"{language}.{key}"
    assert "Taichi" in installer
    assert "only a suggestion" in installer and "лишь предложение" in installer


def test_installer_preselects_the_addon_only_for_an_nvidia_driver(installer) -> None:
    assert "nvcuda.dll" in installer
    assert "WizardSelectComponents('!gpu')" in installer
    assert "CreateOutputMsgPage" in installer and "procedure InitializeWizard" in installer


def test_release_workflow_builds_checks_and_publishes_the_addon() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "build_gpu_addon.py" in workflow
    assert "SPINSLICER_GPU_DIR" in workflow and "taichi_version" in workflow
    assert "SpinSlicer-gpu-addon-windows-x64.zip" in workflow
    assert "requirements-gpu.txt" in (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def test_the_spec_ships_the_kernel_source_for_taichi() -> None:
    spec = (ROOT / "packaging" / "SpinSlicer.spec").read_text(encoding="utf-8")
    assert 'PROJECT_ROOT / "optics" / "gpu_operator.py"' in spec and '"optics")]' in spec
    assert "Cannot find source code" in spec     # the reason is written next to it


def test_the_release_check_runs_a_real_kernel_and_ci_requires_it() -> None:
    check = (ROOT / "release_check.py").read_text(encoding="utf-8")
    assert "selftest_gpu_kernels" in check and "gpu_kernels" in check
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "gpu_kernels.ok" in workflow


def test_the_installer_makes_no_speed_claim_that_was_measured_only_from_source(installer) -> None:
    assert "17 seconds" not in installer and "17 секунд" not in installer


def test_the_addon_builder_refuses_a_taichi_core_built_for_another_python(tmp_path) -> None:
    builder = _load(ROOT / "packaging" / "build_gpu_addon.py", "build_gpu_addon_abi")
    core = tmp_path / "taichi" / "_lib" / "core"
    core.mkdir(parents=True)
    wrong = "cp399"
    (core / f"taichi_python.{wrong}-win_amd64.pyd").write_bytes(b"")
    with pytest.raises(SystemExit, match="not built for this Python"):
        builder.check_abi(tmp_path)
    (core / f"taichi_python.{builder.extension_tag()}-win_amd64.pyd").write_bytes(b"")
    builder.check_abi(tmp_path)   # the matching core is accepted


def test_a_taichi_that_will_not_start_is_named_in_the_profile(monkeypatch) -> None:
    def failing():
        compute._taichi_state["error"] = "ImportError: cannot import name 'taichi_python'"
        return None

    monkeypatch.setattr(compute, "init_taichi_gpu", failing)
    monkeypatch.setitem(compute._taichi_state, "error", "")
    hardware = compute.Hardware(8, 16 * 1024**3, "NVIDIA test", 8 * 1024**3, True)
    profile = compute.resolve_profile("maximum", hardware)
    assert profile.device == "cpu" and "taichi_python" in profile.reason
    assert "Taichi" in profile.describe()


def test_the_addon_builder_can_be_told_which_python_the_program_uses(tmp_path) -> None:
    import subprocess

    wrong = "cp399"
    done = subprocess.run(
        [sys.executable, str(ROOT / "packaging" / "build_gpu_addon.py"), "--out", str(tmp_path / "x"),
         "--expect-tag", wrong],
        capture_output=True, text=True,
    )
    assert done.returncode != 0 and "uses cp399" in done.stderr
    assert not (tmp_path / "x").exists()          # it refused before installing anything
    assert "--expect-tag cp312" in (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
