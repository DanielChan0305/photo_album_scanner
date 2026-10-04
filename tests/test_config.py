"""Config and settings tests (no camera required)."""

from __future__ import annotations

import pytest

from photo_album_scanner.config import CameraConfig, Settings


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    target = tmp_path / "data"
    monkeypatch.setenv("PAS_DATA_DIR", str(target))
    return target


def test_settings_round_trip(data_dir):
    settings = Settings(
        camera=CameraConfig(
            device="/dev/video0",
            width=1920,
            height=1080,
            pixel_format="MJPG",
            controls={"exposure_absolute": 156},
        ),
        notes="test rig",
    )
    path = settings.save()
    assert path == data_dir / "settings.json"
    assert Settings.load() == settings


def test_load_defaults_when_missing(data_dir):
    assert Settings.load() == Settings()
