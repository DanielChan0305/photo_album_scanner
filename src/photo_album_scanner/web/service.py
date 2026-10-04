"""Background capture service driving the web UI's live view.

Runs a :class:`~photo_album_scanner.capture.CaptureEngine` in a worker thread,
publishes a small JPEG preview for the browser, processes each captured page,
and exposes a thread-safe status dict.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2

from .. import camera, capture
from ..config import Settings
from ..pipeline import process_page


class CaptureService:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._engine: capture.CaptureEngine | None = None
        self._preview: bytes | None = None
        self._started_at: float | None = None
        self._status: dict[str, Any] = {
            "running": False,
            "album": None,
            "captures": 0,
            "state": "idle",
            "glare": False,
            "error": None,
            "last_page": None,
            "uptime": 0.0,
        }

    # ------------------------------------------------------------------
    # Public control API (thread-safe)
    # ------------------------------------------------------------------

    def start(self, album: str, process: bool = True) -> dict[str, Any]:
        with self._lock:
            already_running = self._status["running"]
        if already_running:
            return self.status()

        album_dir = capture.ensure_album_dir(album)
        self._stop.clear()
        with self._lock:
            self._preview = None
            self._status.update(
                running=True,
                album=album,
                captures=0,
                state="idle",
                glare=False,
                error=None,
                last_page=None,
                uptime=0.0,
            )
        self._thread = threading.Thread(
            target=self._run,
            args=(album, album_dir, process),
            daemon=True,
            name="pas-capture",
        )
        self._thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)
        return self.status()

    def manual_capture(self) -> dict[str, Any]:
        with self._lock:
            engine = self._engine
        if engine is not None:
            engine.request_capture()
        return self.status()

    def preview(self) -> bytes | None:
        with self._lock:
            return self._preview

    def status(self) -> dict[str, Any]:
        with self._lock:
            data = dict(self._status)
            if data["running"] and self._started_at is not None:
                data["uptime"] = round(time.monotonic() - self._started_at, 1)
            return data

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _set(self, **values: Any) -> None:
        with self._lock:
            self._status.update(values)

    def _run(self, album: str, album_dir: Path, process: bool) -> None:
        try:
            settings = Settings.load()
            device = settings.camera.device or camera.autodetect_device()
            if device is None:
                self._set(error="no camera found")
                return

            config = capture.CaptureConfig(
                album_dir=album_dir,
                device=str(device),
                width=settings.camera.width,
                height=settings.camera.height,
                fps=settings.camera.fps,
                fourcc=settings.camera.pixel_format or "MJPG",
            )
            engine = capture.CaptureEngine(config)
            with self._lock:
                self._engine = engine
                self._started_at = time.monotonic()

            if settings.camera.controls:
                camera.apply_controls(config.device, settings.camera.controls)
            engine.open()

            interval = 1.0 / max(config.analysis_fps, 1.0)
            last_preview = 0.0
            while not self._stop.is_set():
                frame = engine.read()
                now = time.monotonic()
                event, record = engine.step(frame, now)
                self._set(state=event.state.value, glare=event.glare, captures=engine.captures)

                if record is not None:
                    self._set(last_page=record.seq)
                    if process:
                        try:
                            process_page(Path(record.raw_path).parent, settings)
                        except Exception as exc:  # noqa: BLE001 - keep capturing
                            self._set(error=f"processing page_{record.seq:03d}: {exc}")

                if now - last_preview >= 0.25:
                    last_preview = now
                    scale = 640 / frame.shape[1]
                    thumb = (
                        cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                        if scale < 1
                        else frame
                    )
                    ok, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        with self._lock:
                            self._preview = buffer.tobytes()

                elapsed = time.monotonic() - now
                if elapsed < interval:
                    time.sleep(interval - elapsed)
        except Exception as exc:  # noqa: BLE001
            self._set(error=str(exc))
        finally:
            with self._lock:
                engine = self._engine
                self._engine = None
                self._started_at = None
            if engine is not None:
                engine.close()
            self._set(running=False, state="idle", glare=False)
