"""Shared fixtures for the test suite."""

from __future__ import annotations

import cv2
import numpy as np
import pytest


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    target = tmp_path / "data"
    monkeypatch.setenv("PAS_DATA_DIR", str(target))
    return target


@pytest.fixture()
def synthetic_page() -> np.ndarray:
    """A dark album page with two white-bordered photos."""
    page = np.full((900, 1200, 3), 45, dtype=np.uint8)

    def photo(x1: int, y1: int, x2: int, y2: int, border: int = 14) -> None:
        cv2.rectangle(page, (x1, y1), (x2, y2), (245, 245, 245), -1)
        cv2.rectangle(
            page, (x1 + border, y1 + border), (x2 - border, y2 - border), (90, 120, 160), -1
        )

    photo(80, 80, 460, 360)
    photo(560, 120, 860, 520)
    return page
