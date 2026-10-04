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
class DetectionConfig:
    """Photo-detection sensitivity. Smaller thresholds = more sensitive.

    Defaults are tuned for full-bleed prints in plastic sleeves, where photo
    borders can be soft, fragmented by glare, or low-contrast.
    """

    min_area_ratio: float = 0.008  # fraction of the page
    max_area_ratio: float = 0.30  # larger = probably several merged photos
    min_side: int = 80  # full-resolution pixels
    max_aspect: float = 3.0
    min_rectangularity: float = 0.45  # contour area / fitted rotated-rect area
    canny_pairs: tuple[tuple[int, int], ...] = ((25, 75), (40, 120))
    clahe: bool = True
    min_edge_component: int = 40  # drop speckle edges smaller than this
    close_kernel: int = 5
    dedupe_iou: float = 0.5
    low_confidence: float = 0.5

    def __post_init__(self) -> None:
        self.canny_pairs = tuple((int(lo), int(hi)) for lo, hi in self.canny_pairs)


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
    detection: DetectionConfig = field(default_factory=DetectionConfig)
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
            detection=DetectionConfig(**raw.get("detection", {})),
            page_roi=raw.get("page_roi"),
            notes=raw.get("notes", ""),
        )

    def save(self) -> Path:
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path
