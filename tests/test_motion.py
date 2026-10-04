"""Motion detector state-machine tests (no camera needed)."""

from __future__ import annotations

import numpy as np

from photo_album_scanner.capture import MotionConfig, MotionDetector, State


def frame(value: int = 128) -> np.ndarray:
    return np.full((18, 32, 3), value, dtype=np.uint8)


def make_config(**overrides) -> MotionConfig:
    defaults = {"stable_seconds": 0.7, "min_capture_interval": 1.0}
    defaults.update(overrides)
    return MotionConfig(**defaults)


def test_first_frame_is_ignored():
    detector = MotionDetector(make_config())
    event = detector.update(frame(), now=0.0)
    assert not event.capture
    assert event.state == State.IDLE


def test_page_turn_triggers_exactly_one_capture():
    detector = MotionDetector(make_config())
    detector.update(frame(100), now=0.0)

    time_now = 0.0
    for index, value in enumerate([200, 100, 200, 100, 200], start=1):
        time_now = 0.1 * index
        event = detector.update(frame(value), now=time_now)
    assert event.state == State.MOTION

    captures = 0
    for _ in range(30):
        time_now += 0.1
        if detector.update(frame(200), now=time_now).capture:
            captures += 1
    assert captures == 1
    assert detector.state == State.COOLDOWN


def test_second_page_turn_triggers_second_capture():
    detector = MotionDetector(make_config())
    detector.update(frame(100), now=0.0)
    time_now = 0.0

    def turn_and_settle(start: float, values: list[int]) -> float:
        current = start
        for value in values:
            current += 0.1
            detector.update(frame(value), now=current)
        for _ in range(15):
            current += 0.1
            detector.update(frame(values[-1]), now=current)
        return current

    time_now = turn_and_settle(time_now, [200, 100, 200, 100, 200])
    time_now = turn_and_settle(time_now, [100, 200, 100, 200, 100])
    assert detector.state == State.COOLDOWN


def test_manual_capture_in_idle():
    detector = MotionDetector(make_config())
    detector.update(frame(), now=0.0)
    detector.request_capture()
    assert detector.update(frame(), now=0.1).capture
    # a static page must not produce duplicates afterwards
    assert not detector.update(frame(), now=0.2).capture


def test_glare_blocks_auto_capture_until_cleared():
    detector = MotionDetector(make_config())
    detector.update(frame(100), now=0.0)
    detector.update(frame(255), now=0.1)  # change -> MOTION

    time_now = 0.1
    captured_during_glare = False
    for _ in range(20):
        time_now += 0.1
        if detector.update(frame(255), now=time_now).capture:
            captured_during_glare = True
    assert not captured_during_glare

    # glare clears with a new page; settle then capture
    detector.update(frame(120), now=time_now + 0.1)
    time_now += 0.1
    captured = False
    for _ in range(15):
        time_now += 0.1
        if detector.update(frame(120), now=time_now).capture:
            captured = True
    assert captured
