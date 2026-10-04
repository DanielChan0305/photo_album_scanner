"""Capture engine tests (no camera needed)."""

from __future__ import annotations

import cv2
import numpy as np

from photo_album_scanner import capture


def test_engine_prefers_full_resolution_still(data_dir, synthetic_page, monkeypatch):
    album = capture.ensure_album_dir("album_01")
    config = capture.CaptureConfig(
        album_dir=album,
        device="http://phone.local:8080/video",
        width=1920,
        height=1080,
        fps=30,
        fourcc="MJPG",
        still_url="http://phone.local:8080/photo.jpg",
    )
    engine = capture.CaptureEngine(config)

    still = np.full_like(synthetic_page, 200)
    monkeypatch.setattr(capture.camera, "fetch_still", lambda url, timeout=5.0: still)

    engine.step(synthetic_page, now=0.0)  # prime the detector
    engine.detector.request_capture()
    event, record = engine.step(synthetic_page, now=1.0)

    assert event.capture
    assert record is not None
    saved = cv2.imread(record.raw_path)
    assert int(saved.mean()) == 200  # the still, not the 45-ish stream frame


def test_engine_falls_back_to_stream_frame_when_still_fails(
    data_dir, synthetic_page, monkeypatch
):
    album = capture.ensure_album_dir("album_01")
    config = capture.CaptureConfig(
        album_dir=album,
        device="http://phone.local:8080/video",
        width=1920,
        height=1080,
        fps=30,
        fourcc="MJPG",
        still_url="http://phone.local:8080/photo.jpg",
    )
    engine = capture.CaptureEngine(config)
    monkeypatch.setattr(capture.camera, "fetch_still", lambda url, timeout=5.0: None)

    engine.step(synthetic_page, now=0.0)
    engine.detector.request_capture()
    event, record = engine.step(synthetic_page, now=1.0)

    assert event.capture
    assert record is not None
    saved = cv2.imread(record.raw_path)
    assert int(saved.mean()) == int(synthetic_page.mean())
