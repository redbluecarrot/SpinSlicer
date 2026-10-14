import pytest

from profiles import (
    MachineProfile,
    ProfileValidationError,
    ProjectorProfile,
    ResinProfile,
    VatProfile,
    load_profile,
    save_profile,
)


def test_machine_profile_json_round_trip(tmp_path) -> None:
    profile = MachineProfile(
        name="lab-vam",
        projector=ProjectorProfile(
            name="projector-a",
            width_px=1280,
            height_px=800,
            bit_depth=8,
            fps=30.0,
            gamma=2.2,
        ),
        vat=VatProfile(diameter_mm=80.0, height_mm=120.0),
        angle_offset_deg=12.5,
        angle_direction=-1,
    )
    path = tmp_path / "machine.json"

    save_profile(str(path), profile)

    loaded = load_profile(str(path))
    assert isinstance(loaded, MachineProfile)
    assert loaded == profile


def test_resin_profile_json_round_trip(tmp_path) -> None:
    profile = ResinProfile(
        name="development-resin",
        base_exposure_s=2.5,
        intensity_pct=80.0,
        threshold_pct=35.0,
        notes="Temporary calibration profile",
    )
    path = tmp_path / "resin.json"

    save_profile(str(path), profile)

    loaded = load_profile(str(path))
    assert isinstance(loaded, ResinProfile)
    assert loaded == profile


def test_projector_profile_rejects_invalid_rotation() -> None:
    profile = ProjectorProfile(rotate_deg=45)

    with pytest.raises(ProfileValidationError, match="rotate_deg"):
        profile.validate()


def test_machine_profile_rejects_invalid_direction() -> None:
    profile = MachineProfile(angle_direction=0)

    with pytest.raises(ProfileValidationError, match="angle_direction"):
        profile.validate()
