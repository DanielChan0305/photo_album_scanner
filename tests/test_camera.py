"""Camera helper tests (no hardware required for the heuristics)."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from photo_album_scanner import camera
from photo_album_scanner.camera import ControlDetails, DeviceDetails, _config_name


def test_looks_external_rejects_builtin_names():
    assert not camera._looks_external("Integrated RGB Camera: Integrat")
    assert not camera._looks_external("IR Camera")


def test_looks_external_accepts_webcam_names():
    assert camera._looks_external("HD Pro Webcam C920")
    assert camera._looks_external("USB Camera")


def test_is_network_source():
    assert camera.is_network_source("http://192.168.1.23:8080/video")
    assert camera.is_network_source("https://phone.local/stream")
    assert camera.is_network_source("rtsp://phone.local:8554/live")
    assert not camera.is_network_source("/dev/video4")
    assert not camera.is_network_source(Path("/dev/video4"))


class _JpegHandler(BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *args):  # silence test server logs
        pass


def test_fetch_still_over_http():
    image = np.full((48, 64, 3), 200, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    _JpegHandler.payload = buffer.tobytes()

    server = ThreadingHTTPServer(("127.0.0.1", 0), _JpegHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        still = camera.fetch_still(f"http://127.0.0.1:{server.server_port}/photo.jpg")
        assert still is not None
        assert still.shape == image.shape
        assert camera.fetch_still("http://127.0.0.1:9/missing.jpg", timeout=0.5) is None
    finally:
        server.shutdown()
        server.server_close()


def test_config_name_normalization():
    assert _config_name("Focus, Absolute") == "focus_absolute"
    assert _config_name("White Balance, Automatic") == "white_balance_automatic"
    assert _config_name("Exposure Time, Absolute") == "exposure_time_absolute"


def test_find_control_matches_raw_and_normalized_names():
    control = ControlDetails(
        id=1,
        name="Auto Exposure",
        config_name=_config_name("Auto Exposure"),
        type=3,
        minimum=0,
        maximum=3,
        step=1,
        default=3,
        flags=0,
        value=3,
    )
    details = DeviceDetails(
        path=Path("/dev/video4"),
        name="Full HD webcam",
        driver="uvcvideo",
        bus_info="usb-test",
        formats=[],
        controls=[control],
    )
    assert details.find_control("auto_exposure") is control
    assert details.find_control("Auto Exposure") is control
    assert details.find_control("nonexistent") is None
