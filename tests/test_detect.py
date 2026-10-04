"""Photo detection tests on synthetic pages."""

from __future__ import annotations

import cv2

from photo_album_scanner.pipeline import detect, glare


def _bbox(quad):
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return min(xs), min(ys), max(xs), max(ys)


def test_detects_two_photos_in_reading_order(synthetic_page):
    detections = detect.detect_photos(synthetic_page)
    assert len(detections) == 2

    first = _bbox(detections[0].quad)
    second = _bbox(detections[1].quad)
    assert first[0] < second[0]

    expected = [(80, 80, 460, 360), (560, 120, 860, 520)]
    for box, (ex1, ey1, ex2, ey2) in zip((first, second), expected, strict=True):
        assert abs(box[0] - ex1) <= 20
        assert abs(box[1] - ey1) <= 20
        assert abs(box[2] - ex2) <= 20
        assert abs(box[3] - ey2) <= 20


def test_glare_streak_does_not_confuse_detection(synthetic_page):
    image = synthetic_page.copy()
    cv2.rectangle(image, (500, 600), (1150, 700), (255, 255, 255), -1)
    mask = glare.glare_mask(image)
    assert glare.glare_fraction(mask) > 0

    detections = detect.detect_photos(image)
    assert len(detections) == 2
