"""Paths and persisted settings.

Runtime data lives in ``<repo>/data`` by default (git-ignored), overridable with
the ``PAS_DATA_DIR`` environment variable.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def data_dir() -> Path:
    """Root directory for runtime data (captures, calibration, database)."""
    override = os.environ.get("PAS_DATA_DIR")
    return Path(override).expanduser().resolve() if override else DEFAULT_DATA_DIR


def calibration_dir() -> Path:
    return data_dir() / "calibration"


def settings_path() -> Path:
    return data_dir() / "settings.json"


@dataclass
class CameraConfig:
    """Chosen capture device (path or stream URL) and locked camera controls."""

    device: str | None = None
    width: int = 1920
    height: int = 1080
    fps: int = 30
    pixel_format: str | None = None
    still_url: str | None = None
    controls: dict[str, Any] = field(default_factory=dict)


@dataclass
class Settings:
    """All persisted, calibration-derived settings."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    page_roi: list[list[int]] | None = None
    notes: str = ""

    @classmethod
    def load(cls) -> Settings:
        path = settings_path()
        if not path.exists():
            return cls()
        raw: dict[str, Any] = json.loads(path.read_text())
        return cls(
            camera=CameraConfig(**raw.get("camera", {})),
            page_roi=raw.get("page_roi"),
            notes=raw.get("notes", ""),
        )

    def save(self) -> Path:
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path
