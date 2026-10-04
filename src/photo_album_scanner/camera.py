"""Camera discovery, inspection, control locking, and frame capture.

V4L2 enumeration and controls go through ``linuxpy`` (tolerant of control types
newer than the installed linuxpy release); frame capture goes through OpenCV's
V4L2 backend, which decodes MJPG/YUYV streams to BGR frames.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# V4L2 control flags / types (kernel ABI constants; stable across versions)
FLAG_READ_ONLY = 0x0001
FLAG_INACTIVE = 0x0010
FLAG_DISABLED = 0x0400
FLAG_HAS_PAYLOAD = 0x0100

_CTRL_INTEGER = 1
_CTRL_BOOLEAN = 2
_CTRL_MENU = 3
_CTRL_INTEGER64 = 5
_CTRL_BITMASK = 8
_CTRL_INTEGER_MENU = 9
_CTRL_U8 = 10
_CTRL_U16 = 11
_CTRL_U32 = 12

_READABLE_INT_TYPES = {
    _CTRL_INTEGER,
    _CTRL_BOOLEAN,
    _CTRL_MENU,
    _CTRL_INTEGER64,
    _CTRL_BITMASK,
    _CTRL_INTEGER_MENU,
    _CTRL_U8,
    _CTRL_U16,
    _CTRL_U32,
}


def _config_name(name: str) -> str:
    """linuxpy-style normalized control name ('Focus, Absolute' -> 'focus_absolute')."""
    result = name.lower()
    for char in ("(", ")"):
        result = result.replace(char, "")
    for token in (", ", " "):
        result = result.replace(token, "_")
    return result


# --------------------------------------------------------------------------
# Device listing (cheap: sysfs only, no device open)
# --------------------------------------------------------------------------


@dataclass
class VideoDevice:
    path: Path
    name: str


def list_video_devices() -> list[VideoDevice]:
    """List /dev/video* devices with their names from sysfs."""

    def sort_key(path: Path) -> int:
        digits = re.sub(r"\D", "", path.name)
        return int(digits) if digits else -1

    devices = []
    for path in sorted(Path("/dev").glob("video*"), key=sort_key):
        name_file = Path("/sys/class/video4linux") / path.name / "name"
        name = name_file.read_text().strip() if name_file.exists() else "unknown"
        devices.append(VideoDevice(path=path, name=name))
    return devices


def _looks_external(name: str) -> bool:
    """Heuristic: the laptop's built-in camera nodes are usually labeled."""
    lowered = name.lower()
    return not any(word in lowered for word in ("integrated", "rgb camera", "ir camera", "front"))


def autodetect_device() -> Path | None:
    """Prefer a plausible external/webcam device, else the first one."""
    devices = list_video_devices()
    for device in devices:
        if _looks_external(device.name):
            return device.path
    return devices[0].path if devices else None


# --------------------------------------------------------------------------
# Device inspection (formats + controls)
# --------------------------------------------------------------------------


@dataclass
class FormatDetails:
    fourcc: str
    description: str
    sizes: list[tuple[int, int]] = field(default_factory=list)
    fps: dict[tuple[int, int], list[float]] = field(default_factory=dict)


@dataclass
class ControlDetails:
    id: int
    name: str
    config_name: str
    type: int
    minimum: int
    maximum: int
    step: int
    default: int
    flags: int
    value: int | None

    @property
    def writable(self) -> bool:
        return not (self.flags & (FLAG_READ_ONLY | FLAG_DISABLED | FLAG_INACTIVE))


@dataclass
class DeviceDetails:
    path: Path
    name: str
    driver: str
    bus_info: str
    formats: list[FormatDetails]
    controls: list[ControlDetails]

    def find_control(self, *names: str) -> ControlDetails | None:
        """Find a control by raw or normalized name; first name wins."""
        for name in names:
            for control in self.controls:
                if control.name == name or control.config_name == name:
                    return control
        return None


def _frame_interval_fps(interval) -> list[float]:
    if getattr(interval.type, "name", "") == "DISCRETE" or interval.min_fps == interval.max_fps:
        return [float(interval.min_fps)]
    return [float(interval.min_fps), float(interval.max_fps)]


def inspect_device(path: str | Path) -> DeviceDetails:
    """Enumerate pixel formats, frame sizes, frame rates, and controls."""
    from linuxpy.video.device import Device, PixelFormat, iter_read_controls

    with Device(str(path)) as dev:
        info = dev.info
        fd = dev.fileno()

        formats: list[FormatDetails] = []
        for fmt in info.formats:
            pixel_format = PixelFormat(fmt.pixel_format)
            details = FormatDetails(
                fourcc=pixel_format.name,
                description=getattr(fmt, "description", "") or "",
            )
            for size in info.format_frame_sizes(fmt.pixel_format):
                if getattr(size.type, "name", "") == "DISCRETE":
                    width, height = size.info.width, size.info.height
                else:
                    width, height = size.info.max_width, size.info.max_height
                details.sizes.append((width, height))
                with contextlib.suppress(OSError, ValueError):
                    intervals = info.fps_intervals(fmt.pixel_format, width, height)
                    details.fps[(width, height)] = [
                        fps for interval in intervals for fps in _frame_interval_fps(interval)
                    ]
            formats.append(details)

        controls: list[ControlDetails] = []
        with contextlib.suppress(OSError):
            for ctrl in iter_read_controls(fd):
                value = None
                if ctrl.type in _READABLE_INT_TYPES and not (ctrl.flags & FLAG_HAS_PAYLOAD):
                    with contextlib.suppress(OSError):
                        value = _get_control(fd, ctrl.id)
                controls.append(
                    ControlDetails(
                        id=ctrl.id,
                        name=ctrl.name.decode(),
                        config_name=_config_name(ctrl.name.decode()),
                        type=ctrl.type,
                        minimum=ctrl.minimum,
                        maximum=ctrl.maximum,
                        step=ctrl.step,
                        default=ctrl.default_value,
                        flags=ctrl.flags,
                        value=value,
                    )
                )

        return DeviceDetails(
            path=Path(path),
            name=info.card,
            driver=info.driver,
            bus_info=info.bus_info,
            formats=formats,
            controls=controls,
        )


def _get_control(fd: int, control_id: int) -> int:
    from linuxpy.video.device import get_control

    return int(get_control(fd, control_id))


def _set_control(fd: int, control_id: int, value: int) -> None:
    from linuxpy.video.device import set_control

    set_control(fd, control_id, value)


def apply_controls(path: str | Path, values: dict[str, int]) -> dict[str, tuple[bool, str]]:
    """Set controls by name; returns ``name -> (ok, message)``."""
    from linuxpy.video.device import Device, iter_read_controls

    results: dict[str, tuple[bool, str]] = {}
    with Device(str(path)) as dev:
        fd = dev.fileno()
        by_name = {ctrl.name.decode(): ctrl for ctrl in iter_read_controls(fd)}
        for name, value in values.items():
            ctrl = by_name.get(name)
            if ctrl is None:
                results[name] = (False, "control not present")
                continue
            try:
                _set_control(fd, ctrl.id, int(value))
                readback = _get_control(fd, ctrl.id)
                ok = readback == int(value)
                results[name] = (ok, f"set to {readback}" + ("" if ok else f" (requested {value})"))
            except OSError as exc:
                results[name] = (False, str(exc))
    return results


# Auto-control names as exposed by uvcvideo for typical webcams
_EXPOSURE_AUTO_NAMES = ("auto_exposure", "exposure_auto")
_FOCUS_AUTO_NAMES = ("focus_automatic_continuous", "focus_auto")
_WB_AUTO_NAMES = ("white_balance_automatic", "white_balance_temperature_auto")
_WB_PRESET_NAMES = ("white_balance_auto_preset",)
_KEY_CONTROLS = (
    "auto_exposure",
    "exposure_time_absolute",
    "exposure_absolute",
    "gain",
    "focus_automatic_continuous",
    "focus_absolute",
    "white_balance_automatic",
    "white_balance_temperature",
    "brightness",
    "contrast",
    "saturation",
    "gamma",
    "sharpness",
    "power_line_frequency",
    "backlight_compensation",
)


def auto_lock_controls(path: str | Path) -> tuple[dict[str, int], list[str]]:
    """Switch auto exposure/focus/white-balance to manual and return values to persist.

    Returns ``(controls_to_save, log_lines)``. Unsupported controls are skipped
    with a note instead of failing, since webcam control sets vary widely.
    """
    details = inspect_device(path)
    log: list[str] = []
    to_set: dict[str, int] = {}

    for names, value, what in (
        (_EXPOSURE_AUTO_NAMES, 1, "exposure"),
        (_FOCUS_AUTO_NAMES, 0, "focus"),
        (_WB_AUTO_NAMES, 0, "white balance"),
    ):
        ctrl = details.find_control(*names)
        if ctrl is None:
            log.append(f"note: no manual {what} switch exposed by this camera")
        elif not ctrl.writable:
            log.append(f"note: {what} switch {ctrl.name!r} is not writable")
        else:
            to_set[ctrl.name] = value

    if to_set:
        for name, (ok, message) in apply_controls(path, to_set).items():
            log.append(f"{'locked' if ok else 'failed to lock'} {name}: {message}")

    # Read back the key control values so the calibration can persist them.
    return read_key_controls(path), log


def read_key_controls(path: str | Path) -> dict[str, int]:
    """Current values of the well-known controls, for persistence."""
    details = inspect_device(path)
    values: dict[str, int] = {}
    for name in _KEY_CONTROLS:
        ctrl = details.find_control(name)
        if ctrl is not None and ctrl.value is not None:
            values[name] = ctrl.value
    return values


# --------------------------------------------------------------------------
# Frame capture (OpenCV / V4L2)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Capture sources: V4L2 devices and network streams
# --------------------------------------------------------------------------

NETWORK_SCHEMES = ("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://")


def is_network_source(source: str | Path) -> bool:
    """True for stream URLs (phone camera apps, IP cameras) vs. /dev/video* paths."""
    return str(source).lower().startswith(NETWORK_SCHEMES)


def open_capture(
    path: str | Path,
    width: int,
    height: int,
    fps: int | None = None,
    fourcc: str | None = None,
) -> cv2.VideoCapture:
    """Open a V4L2 device or network stream.

    For network sources, OpenCV's FFMPEG backend is used; width/height/fps are
    hints only (the streaming app decides the actual resolution).
    """
    source = str(path)
    if is_network_source(source):
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not capture.isOpened():
            raise RuntimeError(f"could not open stream {source}")
        if fps:
            capture.set(cv2.CAP_PROP_FPS, fps)
        with contextlib.suppress(Exception):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video device {source}")
    if fourcc:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        capture.set(cv2.CAP_PROP_FPS, fps)
    with contextlib.suppress(Exception):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def fetch_still(url: str, timeout: float = 5.0) -> np.ndarray | None:
    """Fetch a full-resolution still over HTTP (e.g. IP Webcam's ``/photo.jpg``).

    Returns a BGR image, or None on any failure so callers can fall back to the
    video-stream frame.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = response.read()
    except OSError:
        return None
    if not data:
        return None
    buffer = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def capture_frames(
    capture: cv2.VideoCapture, count: int, warmup: int = 3
) -> list[np.ndarray]:
    """Grab ``count`` frames after a short warmup."""
    frames: list[np.ndarray] = []
    for _ in range(max(0, warmup)):
        capture.read()
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def frame_brightness(frame: np.ndarray) -> float:
    """Mean luminance of a BGR frame."""
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
