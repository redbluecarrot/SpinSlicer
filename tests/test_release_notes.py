"""The version, the installer, the READMEs and the release notes must agree before a tag."""
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
NOTES = ROOT / "docs" / "releases" / f"v{VERSION}.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_the_installer_builds_the_same_version_by_default() -> None:
    match = re.search(r'#define MyAppVersion "([^"]+)"', _text(ROOT / "packaging" / "SpinSlicer.iss"))
    assert match and match.group(1) == VERSION


def test_the_release_workflow_publishes_the_notes_of_the_tag() -> None:
    workflow = _text(ROOT / ".github" / "workflows" / "release.yml")
    assert "body_path: docs/releases/${{ github.ref_name }}.md" in workflow
    assert NOTES.is_file(), f"docs/releases/v{VERSION}.md is what the tag v{VERSION} will publish"


@pytest.mark.parametrize(
    "readme",
    ["README.md", "docs/translations/README.es.md", "docs/translations/README.ru.md", "docs/translations/README.zh-CN.md"],
)
def test_readmes_point_at_the_current_release(readme: str) -> None:
    text = _text(ROOT / readme)
    assert f"v{VERSION}" in text and f"SpinSlicer-Setup-{VERSION}.exe" in text
    assert f"releases/v{VERSION}.md" in text
    # What a release names is its own artifacts and notes (older reports may still be cited).
    assert set(re.findall(r"SpinSlicer-Setup-([0-9.]+)\.exe", text)) == {VERSION}
    assert set(re.findall(r"SpinSlicer-([0-9.]+)-(?:1-x86_64|1\.x86_64|amd64)", text)) == {VERSION}
    assert set(re.findall(r"releases/(v[0-9.]+)\.md", text)) == {f"v{VERSION}"}


def test_the_notes_are_bilingual_and_name_the_version() -> None:
    notes = _text(NOTES)
    assert notes.startswith(f"# SpinSlicer v{VERSION}")
    assert "## English" in notes and "## Русский" in notes
    assert notes.index("## English") < notes.index("## Русский")
    assert f"SpinSlicer-Setup-{VERSION}.exe" in notes
    assert f"compare/v0.3.2...v{VERSION}" in notes


def test_the_gamma_change_is_the_first_thing_in_both_languages() -> None:
    notes = _text(NOTES)
    english = notes.split("## English", 1)[1].split("## Русский", 1)[0]
    russian = notes.split("## Русский", 1)[1]
    assert re.search(r"gamma", english.split("###", 2)[1], re.IGNORECASE)
    assert re.search(r"гамм", russian.split("###", 2)[1], re.IGNORECASE)


def test_the_notes_state_the_gamma_facts_correctly() -> None:
    """v0.3.2 ignored the profile's gamma field; the default is still linear."""
    notes = _text(NOTES)
    english = notes.split("## English", 1)[1].split("### Overview", 1)[0]
    assert "ignored" in english and "1.0" in english and "linear" in english
    assert "PNG" in english and "preview" in english              # both stay linear
    assert "30 fps" in english and "even" in english              # the other changes that alter the output
    assert "SpinSlicer-gpu-addon-windows-x64.zip" in notes        # the add-on is published too
