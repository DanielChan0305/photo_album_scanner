"""Camera helper tests (no hardware required for the heuristics)."""

from __future__ import annotations

from pathlib import Path

from photo_album_scanner import camera
from photo_album_scanner.camera import ControlDetails, DeviceDetails, _config_name


def test_looks_external_rejects_builtin_names():
    assert not camera._looks_external("Integrated RGB Camera: Integrat")
    assert not camera._looks_external("IR Camera")


def test_looks_external_accepts_webcam_names():
    assert camera._looks_external("HD Pro Webcam C920")
    assert camera._looks_external("USB Camera")


def test_config_name_normalization():
    assert _config_name("Focus, Absolute") == "focus_absolute"
    assert _config_name("White Balance, Automatic") == "white_balance_automatic"
    assert _config_name("Exposure Time, Absolute") == "exposure_time_absolute"


def test_find_control_matches_raw_and_normalized_names():
    control = ControlDetails(
        id=1,
        name="Auto Exposure",
        config_name=_config_name("Auto Exposure"),
        type=3,
        minimum=0,
        maximum=3,
        step=1,
        default=3,
        flags=0,
        value=3,
    )
    details = DeviceDetails(
        path=Path("/dev/video4"),
        name="Full HD webcam",
        driver="uvcvideo",
        bus_info="usb-test",
        formats=[],
        controls=[control],
    )
    assert details.find_control("auto_exposure") is control
    assert details.find_control("Auto Exposure") is control
    assert details.find_control("nonexistent") is None
