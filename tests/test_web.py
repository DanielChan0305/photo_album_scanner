"""Web API tests with a synthetic album (no camera)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from photo_album_scanner import capture
from photo_album_scanner.config import Settings
from photo_album_scanner.pipeline import process_page
from photo_album_scanner.web.app import create_app


@pytest.fixture()
def client(data_dir, synthetic_page):
    album = capture.ensure_album_dir("album_01")
    capture.save_capture(album, synthetic_page, 1)
    process_page(album / "page_001", Settings())
    return TestClient(create_app())


def test_albums_and_pages(client):
    albums = client.get("/api/albums").json()
    assert albums == [{"name": "album_01", "pages": 1, "processed": 1}]

    pages = client.get("/api/albums/album_01/pages").json()
    assert pages[0]["seq"] == 1
    assert pages[0]["photo_count"] == 2

    detail = client.get("/api/albums/album_01/pages/1").json()
    assert detail["processed"] is True
    assert len(detail["photos"]) == 2
    assert detail["photos"][0]["url"].startswith("/api/albums/album_01/pages/1/file/")


def test_file_endpoint(client):
    detail = client.get("/api/albums/album_01/pages/1").json()
    response = client.get(detail["photos"][0]["url"])
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")


def test_boxes_endpoint_reprocesses(client):
    payload = {"photos": [{"quad": [[80, 80], [460, 80], [460, 360], [80, 360]]}]}
    detail = client.post("/api/albums/album_01/pages/1/boxes", json=payload).json()
    assert detail["edited"] is True
    assert len(detail["photos"]) == 1
    assert detail["photos"][0]["filename"] == "photo_01.jpg"


def test_reprocess_redetect(client):
    client.post("/api/albums/album_01/pages/1/boxes", json={"photos": []})
    detail = client.post("/api/albums/album_01/pages/1/process?redetect=true").json()
    assert detail["edited"] is False
    assert len(detail["photos"]) == 2


def test_delete_page(client):
    assert client.delete("/api/albums/album_01/pages/1").json() == {"deleted": True}
    assert client.get("/api/albums/album_01/pages").json() == []


def test_invalid_album_and_traversal_rejected(client):
    assert client.get("/api/albums/../etc/pages").status_code == 404
    assert client.get("/api/albums/album_01/pages/1/file/..%2Fmeta.json").status_code == 404
