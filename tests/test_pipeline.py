"""End-to-end pipeline tests (synthetic page, no camera)."""

from __future__ import annotations

import json

import pytest

from photo_album_scanner import capture, store
from photo_album_scanner.config import Settings
from photo_album_scanner.pipeline import process_page


@pytest.fixture()
def prepared_page(data_dir, synthetic_page):
    album = capture.ensure_album_dir("album_01")
    capture.save_capture(album, synthetic_page, 1)
    return album / "page_001"


def test_process_page_writes_crops_and_meta(prepared_page):
    result = process_page(prepared_page, Settings())
    assert len(result.photos) == 2
    assert (prepared_page / "rectified.jpg").is_file()
    assert (prepared_page / "overlay.jpg").is_file()
    for photo in result.photos:
        assert photo.filename
        assert (prepared_page / photo.filename).is_file()

    meta = json.loads((prepared_page / "meta.json").read_text())
    assert meta["edited"] is False
    assert len(meta["photos"]) == 2
    assert meta["photos"][0]["filename"] == "photo_01.jpg"


def test_edited_boxes_take_precedence(prepared_page):
    process_page(prepared_page, Settings())
    store.write_edits(
        prepared_page,
        [{"quad": [[80, 80], [460, 80], [460, 360], [80, 360]]}],
    )
    result = process_page(prepared_page, Settings())
    assert result.edited is True
    assert len(result.photos) == 1
    assert (prepared_page / "photo_01.jpg").is_file()
    assert not (prepared_page / "photo_02.jpg").exists()


def test_redetect_discards_edits(prepared_page):
    store.write_edits(
        prepared_page,
        [{"quad": [[0, 0], [500, 0], [500, 400], [0, 400]]}],
    )
    result = process_page(prepared_page, Settings(), redetect=True)
    assert result.edited is False
    assert len(result.photos) == 2
