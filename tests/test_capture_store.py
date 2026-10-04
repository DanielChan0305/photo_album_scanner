"""Capture storage tests (no camera needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from photo_album_scanner import capture


def test_album_dir_and_sequence(data_dir):
    album = capture.ensure_album_dir("album_01")
    assert album == data_dir / "albums" / "album_01"
    assert capture.next_page_seq(album) == 1
    (album / "page_001").mkdir()
    (album / "page_003").mkdir()
    assert capture.next_page_seq(album) == 4


def test_invalid_album_name(data_dir):
    with pytest.raises(ValueError):
        capture.ensure_album_dir("../evil")


def test_save_capture_writes_files_and_manifest(data_dir):
    album = capture.ensure_album_dir("album_01")
    frame = np.full((36, 64, 3), 128, dtype=np.uint8)
    record = capture.save_capture(album, frame, seq=1)

    assert Path(record.raw_path).exists()
    assert Path(record.thumb_path).exists()
    assert record.seq == 1
    assert record.brightness == pytest.approx(128, abs=2)

    manifest = (album / "captures.jsonl").read_text().strip().splitlines()
    assert len(manifest) == 1
