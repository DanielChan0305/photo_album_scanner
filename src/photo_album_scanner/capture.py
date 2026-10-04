"""Motion-triggered page capture.

Watches a low-resolution analysis view of the stream. While consecutive frames
differ, the page is being turned or handled; once the frame has been stable for
a short period, the full-resolution frame is captured exactly once. After a
capture the session stays in COOLDOWN until motion resumes, so a static page
never produces duplicates.

``CaptureEngine`` is the shared loop used by both the CLI (``pas capture``) and
the web UI's background capture service.
"""

from __future__ import annotations

import json
import re
import select
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np

from . import camera, store
from .config import Settings
from .pipeline import process_page

PAGE_DIR_RE = re.compile(r"^page_(\d+)$")


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def ensure_album_dir(album: str) -> Path:
    """Create (if needed) and return the data directory for an album."""
    path = store.album_dir(album)  # validates the name
    path.mkdir(parents=True, exist_ok=True)
    return path


def next_page_seq(album_dir: Path) -> int:
    """Resume-safe next page number based on existing ``page_NNN`` folders."""
    highest = 0
    for child in album_dir.glob("page_*"):
        match = re.fullmatch(r"page_(\d+)", child.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


@dataclass
class CaptureRecord:
    seq: int
    captured_at: str
    raw_path: str
    thumb_path: str
    brightness: float


def _jpg_bytes(frame: np.ndarray, quality: int) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()


def save_capture(album_dir: Path, frame: np.ndarray, seq: int) -> CaptureRecord:
    """Write ``page_NNN/raw.jpg`` + thumbnail, and append a JSONL manifest line."""
    page_dir = album_dir / f"page_{seq:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)

    raw_path = page_dir / "raw.jpg"
    tmp_path = page_dir / "raw.jpg.tmp"
    tmp_path.write_bytes(_jpg_bytes(frame, 95))
    tmp_path.replace(raw_path)

    height, width = frame.shape[:2]
    scale = 480 / max(width, height)
    thumb = (
        cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else frame
    )
    thumb_path = page_dir / "thumb.jpg"
    thumb_path.write_bytes(_jpg_bytes(thumb, 85))

    record = CaptureRecord(
        seq=seq,
        captured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        raw_path=str(raw_path),
        thumb_path=str(thumb_path),
        brightness=camera.frame_brightness(frame),
    )
    with (album_dir / "captures.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.__dict__) + "\n")
    return record


# --------------------------------------------------------------------------
# Motion detection / settle state machine
# --------------------------------------------------------------------------


class State(str, Enum):
    IDLE = "idle"
    MOTION = "motion"
    COOLDOWN = "cooldown"


@dataclass
class MotionConfig:
    analysis_width: int = 320
    analysis_height: int = 180
    frame_diff_threshold: int = 20
    motion_ratio: float = 0.005
    stable_seconds: float = 0.7
    glare_ratio: float = 0.25
    min_capture_interval: float = 1.0


@dataclass
class MotionEvent:
    capture: bool = False
    state: State = State.IDLE
    changed_ratio: float = 0.0
    glare: bool = False
    transition: bool = False


class MotionDetector:
    """Frame-to-frame motion detector with a settle timer and glare guard."""

    def __init__(self, config: MotionConfig | None = None) -> None:
        self.config = config or MotionConfig()
        self.state = State.IDLE
        self._previous: np.ndarray | None = None
        self._last_motion = 0.0
        self._last_capture = float("-inf")
        self._manual = False
        self._kernel = np.ones((3, 3), np.uint8)

    def request_capture(self) -> None:
        """Manual capture (keyboard shortcut, SIGUSR1, foot pedal, ...)."""
        self._manual = True

    def _glare_fraction(self, gray: np.ndarray) -> float:
        return float(cv2.countNonZero(cv2.inRange(gray, 250, 255))) / gray.size

    def update(self, frame: np.ndarray, now: float) -> MotionEvent:
        small = cv2.resize(
            frame,
            (self.config.analysis_width, self.config.analysis_height),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else small
        glare = self._glare_fraction(gray) >= self.config.glare_ratio

        if self._previous is None:
            self._previous = gray
            return MotionEvent(state=self.state, glare=glare)

        delta = cv2.absdiff(gray, self._previous)
        self._previous = gray
        _, mask = cv2.threshold(
            delta, self.config.frame_diff_threshold, 255, cv2.THRESH_BINARY
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        ratio = float(cv2.countNonZero(mask)) / mask.size

        previous_state = self.state
        capture = False

        if self._manual:
            self._manual = False
            if now - self._last_capture >= self.config.min_capture_interval:
                capture = True
                self._last_capture = now
                self.state = State.COOLDOWN

        if not capture:
            if ratio >= self.config.motion_ratio:
                self._last_motion = now
                self.state = State.MOTION
            elif self.state == State.MOTION:
                if glare:
                    # Blocked by glare: require a fresh stable window once it clears.
                    self._last_motion = now
                elif (
                    now - self._last_motion >= self.config.stable_seconds
                    and now - self._last_capture >= self.config.min_capture_interval
                ):
                    capture = True
                    self._last_capture = now
                    self.state = State.COOLDOWN

        return MotionEvent(
            capture=capture,
            state=self.state,
            changed_ratio=ratio,
            glare=glare,
            transition=self.state != previous_state,
        )


# --------------------------------------------------------------------------
# Interactive keyboard (single keypress, no Enter needed)
# --------------------------------------------------------------------------


class Keyboard:
    """Non-blocking single-key input when stdin is a TTY."""

    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self._termios_settings: Any = None

    def __enter__(self) -> Self:
        if self.enabled:
            import termios
            import tty

            self._termios_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc: object) -> None:
        if self.enabled and self._termios_settings is not None:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._termios_settings)

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


# --------------------------------------------------------------------------
# Capture engine
# --------------------------------------------------------------------------


@dataclass
class CaptureConfig:
    album_dir: Path
    device: str
    width: int
    height: int
    fps: int
    fourcc: str
    still_url: str | None = None
    motion: MotionConfig = field(default_factory=MotionConfig)
    analysis_fps: float = 10.0
    max_captures: int | None = None
    warmup_frames: int = 10


class CaptureEngine:
    """Single-threaded capture loop shared by the CLI and the web service."""

    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self.detector = MotionDetector(config.motion)
        self.captures = 0
        self._capture: cv2.VideoCapture | None = None
        self._failures = 0

    def open(self) -> None:
        self._capture = camera.open_capture(
            self.config.device,
            self.config.width,
            self.config.height,
            self.config.fps,
            self.config.fourcc,
        )
        for _ in range(max(0, self.config.warmup_frames)):
            self._capture.read()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def request_capture(self) -> None:
        self.detector.request_capture()

    @property
    def state(self) -> State:
        return self.detector.state

    def read(self) -> np.ndarray:
        """Read a frame, retrying briefly; raises after persistent failure."""
        if self._capture is None:
            raise RuntimeError("capture engine is not open")
        while True:
            ok, frame = self._capture.read()
            if ok:
                self._failures = 0
                return frame
            self._failures += 1
            if self._failures > 30:
                raise RuntimeError("camera stopped delivering frames")
            time.sleep(0.1)

    def step(self, frame: np.ndarray, now: float) -> tuple[MotionEvent, CaptureRecord | None]:
        event = self.detector.update(frame, now)
        record = None
        if event.capture:
            still = None
            if self.config.still_url:
                # Prefer the phone's full-resolution still over the video frame.
                still = camera.fetch_still(self.config.still_url)
            image = still if still is not None else frame
            seq = next_page_seq(self.config.album_dir)
            record = save_capture(self.config.album_dir, image, seq)
            self.captures += 1
        return event, record


def _process_captured_page(record: CaptureRecord, quiet: bool) -> None:
    try:
        result = process_page(Path(record.raw_path).parent, Settings.load())
        if not quiet:
            print(f"  processed page_{record.seq:03d}: {len(result.photos)} photo(s)")
    except Exception as exc:  # noqa: BLE001 - pipeline failures must not stop capture
        print(f"  processing page_{record.seq:03d} failed: {exc}", file=sys.stderr)


# --------------------------------------------------------------------------
# CLI capture loop
# --------------------------------------------------------------------------


def run_capture(config: CaptureConfig, quiet: bool = False, process: bool = False) -> int:
    """Run the capture loop until 'q'/Ctrl+C/``max_captures``; returns capture count."""
    engine = CaptureEngine(config)
    previous_usr1 = signal.signal(signal.SIGUSR1, lambda *_: engine.request_capture())
    try:
        engine.open()
    except RuntimeError:
        signal.signal(signal.SIGUSR1, previous_usr1)
        raise

    interval = 1.0 / config.analysis_fps if config.analysis_fps > 0 else 0.0
    last_glare_note = 0.0

    try:
        with Keyboard() as keyboard:
            while True:
                frame = engine.read()
                now = time.monotonic()
                event, record = engine.step(frame, now)

                if record is not None:
                    if not quiet:
                        print(
                            f"captured page_{record.seq:03d} "
                            f"(brightness {record.brightness:.1f}) -> {record.raw_path}"
                        )
                    if process:
                        _process_captured_page(record, quiet)
                    if config.max_captures is not None and engine.captures >= config.max_captures:
                        break
                elif event.transition and not quiet:
                    if event.state == State.MOTION:
                        print("page moving…")

                if event.glare and not event.capture and now - last_glare_note > 2.0:
                    last_glare_note = now
                    if not quiet:
                        print("glare in view — holding off capture")

                key = keyboard.poll()
                if key in ("q", "\x03", "\x1b"):  # q, Ctrl+C, Esc
                    break
                if key == "c":
                    engine.request_capture()

                elapsed = time.monotonic() - now
                if interval and elapsed < interval:
                    time.sleep(interval - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
        signal.signal(signal.SIGUSR1, previous_usr1)

    return engine.captures
