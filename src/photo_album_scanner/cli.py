"""Command-line interface for the photo album scanner."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

import cv2

from . import __version__, camera, capture
from .config import Settings, calibration_dir

EXPOSURE_NAMES = ("exposure_absolute", "exposure_time_absolute")
FOCUS_NAMES = ("focus_absolute",)
WB_NAMES = ("white_balance_temperature",)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pas",
        description="Photo Album Scanner — overhead-webcam album digitization.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    devices = sub.add_parser("devices", help="list video devices")
    devices.set_defaults(func=_cmd_devices)

    inspect = sub.add_parser("inspect", help="show formats, frame rates, and controls of a camera")
    inspect.add_argument(
        "--device",
        type=Path,
        default=None,
        help="device path (default: calibrated device, else autodetect)",
    )
    inspect.set_defaults(func=_cmd_inspect)

    calibrate = sub.add_parser("calibrate", help="lock camera controls and save test frames")
    calibrate.add_argument("--device", type=Path, default=None)
    calibrate.add_argument("--format", dest="fourcc", default=None, help="pixel format, e.g. MJPG or YUYV")
    calibrate.add_argument("--width", type=int, default=None)
    calibrate.add_argument("--height", type=int, default=None)
    calibrate.add_argument("--fps", type=int, default=None)
    calibrate.add_argument("--frames", type=int, default=10, help="number of test frames to save")
    calibrate.add_argument("--exposure", type=int, default=None, help="manual exposure value override")
    calibrate.add_argument("--focus", type=int, default=None, help="manual focus value override")
    calibrate.add_argument("--wb", type=int, default=None, help="white balance temperature override")
    calibrate.add_argument("--no-lock", action="store_true", help="skip control locking; only capture test frames")
    calibrate.set_defaults(func=_cmd_calibrate)

    capture_cmd = sub.add_parser("capture", help="auto-capture album pages as they settle")
    capture_cmd.add_argument("--album", default="album_01", help="album name (default: album_01)")
    capture_cmd.add_argument("--device", type=Path, default=None)
    capture_cmd.add_argument("--format", dest="fourcc", default=None, help="pixel format, e.g. MJPG or YUYV")
    capture_cmd.add_argument("--width", type=int, default=None)
    capture_cmd.add_argument("--height", type=int, default=None)
    capture_cmd.add_argument("--fps", type=int, default=None)
    capture_cmd.add_argument("--stable", type=float, default=0.7, help="seconds of stillness before a capture")
    capture_cmd.add_argument("--motion-ratio", type=float, default=0.005, help="changed-pixel fraction counted as motion")
    capture_cmd.add_argument("--glare-ratio", type=float, default=0.25, help="near-white fraction that blocks a capture")
    capture_cmd.add_argument("--analysis-fps", type=float, default=10.0)
    capture_cmd.add_argument("--max-captures", type=int, default=None, help="stop after N captures (for testing)")
    capture_cmd.add_argument("--no-controls", action="store_true", help="do not re-apply locked camera controls")
    capture_cmd.add_argument("--quiet", action="store_true")
    capture_cmd.set_defaults(func=_cmd_capture)

    return parser


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _resolve_device(explicit: Path | None, settings: Settings) -> Path:
    if explicit is not None:
        return explicit
    if settings.camera.device:
        return Path(settings.camera.device)
    detected = camera.autodetect_device()
    if detected is None:
        raise SystemExit("no video devices found under /dev/video*")
    return detected


def _cmd_devices(_args: argparse.Namespace) -> int:
    devices = camera.list_video_devices()
    if not devices:
        print("no video devices found")
        return 1
    for device in devices:
        tag = "" if camera._looks_external(device.name) else "  (built-in)"
        print(f"{device.path!s:<16} {device.name}{tag}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    settings = Settings.load()
    path = _resolve_device(args.device, settings)
    details = camera.inspect_device(path)
    print(f"{details.path} — {details.name}")
    print(f"driver: {details.driver}  bus: {details.bus_info}")
    print("formats:")
    for fmt in details.formats:
        print(f"  {fmt.fourcc:<6} {fmt.description}")
        for width, height in fmt.sizes:
            rates = fmt.fps.get((width, height), [])
            rate_text = ", ".join(f"{rate:g}" for rate in rates)
            suffix = f" @ {rate_text} fps" if rate_text else ""
            print(f"      {width}x{height}{suffix}")
    print("controls:")
    for ctrl in details.controls:
        rng = str(ctrl.minimum) if ctrl.minimum == ctrl.maximum else f"{ctrl.minimum}..{ctrl.maximum}"
        value = "-" if ctrl.value is None else str(ctrl.value)
        read_only = "" if ctrl.writable else "  [read-only]"
        print(f"  {ctrl.name:<32} {rng:>14}  value={value:<8}{read_only}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    settings = Settings.load()
    path = _resolve_device(args.device, settings)
    details = camera.inspect_device(path)
    print(f"camera: {path} — {details.name}")

    fourcc = (args.fourcc or settings.camera.pixel_format or "MJPG").upper()
    width = args.width or settings.camera.width or 1920
    height = args.height or settings.camera.height or 1080
    fps = args.fps or settings.camera.fps or 30

    _warn_if_unsupported(details, fourcc, width, height)

    locked: dict[str, int] = {}
    if not args.no_lock:
        print("locking auto controls:")
        locked, log = camera.auto_lock_controls(path)
        for line in log:
            print(f"  {line}")

        overrides: dict[str, int] = {}
        if args.exposure is not None:
            _add_override(overrides, details, EXPOSURE_NAMES, args.exposure, "exposure")
        if args.focus is not None:
            _add_override(overrides, details, FOCUS_NAMES, args.focus, "focus")
        if args.wb is not None:
            _add_override(overrides, details, WB_NAMES, args.wb, "white balance")
        if overrides:
            print("applying overrides:")
            for name, (ok, message) in camera.apply_controls(path, overrides).items():
                print(f"  {'set' if ok else 'failed'}: {name}: {message}")
            locked.update(camera.read_key_controls(path))
        if locked:
            summary = ", ".join(f"{name}={value}" for name, value in sorted(locked.items()))
            print(f"locked controls: {summary}")

    print(f"capturing {args.frames} test frames at {fourcc} {width}x{height} @ {fps} fps")
    capture = camera.open_capture(path, width, height, fps, fourcc)
    try:
        actual = _actual_capture_config(capture)
        print(
            f"negotiated: {actual['fourcc']} {actual['width']}x{actual['height']} "
            f"@ {actual['fps']:.1f} fps"
        )
        frames = camera.capture_frames(capture, args.frames)
    finally:
        capture.release()

    if not frames:
        print("error: no frames captured", file=sys.stderr)
        return 1

    out_dir = calibration_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{path.name}_{actual['width']}x{actual['height']}_{actual['fourcc']}"
    for index, frame in enumerate(frames, start=1):
        cv2.imwrite(str(out_dir / f"{stem}_{index:02d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"saved {len(frames)} frames to {out_dir}")

    brightness = [camera.frame_brightness(frame) for frame in frames]
    spread = max(brightness) - min(brightness)
    print(
        f"brightness: mean={statistics.fmean(brightness):.1f} "
        f"spread={spread:.1f} across {len(frames)} frames"
    )
    if spread > 4:
        print("warning: brightness varies noticeably — auto controls may still be active")
        print("         or lighting is unstable; consider re-running with --exposure/--wb values")

    settings.camera.device = str(path)
    settings.camera.pixel_format = actual["fourcc"]
    settings.camera.width = actual["width"]
    settings.camera.height = actual["height"]
    settings.camera.fps = round(actual["fps"])
    settings.camera.controls = locked
    saved_settings = settings.save()
    print(f"settings saved to {saved_settings}")
    return 0


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _cmd_capture(args: argparse.Namespace) -> int:
    settings = Settings.load()
    path = _resolve_device(args.device, settings)
    try:
        album_dir = capture.ensure_album_dir(args.album)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    fourcc = (args.fourcc or settings.camera.pixel_format or "MJPG").upper()
    width = args.width or settings.camera.width or 1920
    height = args.height or settings.camera.height or 1080
    fps = args.fps or settings.camera.fps or 30

    if settings.camera.controls and not args.no_controls:
        results = camera.apply_controls(path, settings.camera.controls)
        failed = [name for name, (ok, _message) in results.items() if not ok]
        if failed and not args.quiet:
            print(f"warning: could not re-apply controls: {', '.join(failed)}")

    config = capture.CaptureConfig(
        album_dir=album_dir,
        device=str(path),
        width=width,
        height=height,
        fps=fps,
        fourcc=fourcc,
        motion=capture.MotionConfig(
            stable_seconds=args.stable,
            motion_ratio=args.motion_ratio,
            glare_ratio=args.glare_ratio,
        ),
        analysis_fps=args.analysis_fps,
        max_captures=args.max_captures,
    )

    if not args.quiet:
        print(f"album: {args.album} -> {album_dir}")
        print(f"camera: {path} {fourcc} {width}x{height} @ {fps} fps")
        if sys.stdin.isatty():
            print("press 'c' to capture now, 'q' to quit")
        else:
            print(f"manual capture: kill -USR1 {os.getpid()}")

    try:
        count = capture.run_capture(config, quiet=args.quiet)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"done: {count} page(s) captured into {album_dir}")
    return 0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _actual_capture_config(capture: cv2.VideoCapture) -> dict[str, object]:
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    return {
        "fourcc": fourcc if fourcc.isprintable() else "?",
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }


def _add_override(
    overrides: dict[str, int],
    details: camera.DeviceDetails,
    names: tuple[str, ...],
    value: int,
    label: str,
) -> None:
    ctrl = details.find_control(*names)
    if ctrl is None:
        print(f"note: no {label} control to override")
    else:
        overrides[ctrl.name] = int(value)


def _normalize_fourcc(fourcc: str) -> str:
    upper = fourcc.upper()
    return {"MJPEG": "MJPG"}.get(upper, upper)


def _warn_if_unsupported(
    details: camera.DeviceDetails, fourcc: str, width: int, height: int
) -> None:
    wanted = _normalize_fourcc(fourcc)
    for fmt in details.formats:
        if _normalize_fourcc(fmt.fourcc) == wanted:
            if (width, height) not in fmt.sizes:
                print(f"warning: {fourcc} does not list {width}x{height}; driver may pick another size")
            return
    print(f"warning: {fourcc} is not among the reported formats; driver may pick something else")


if __name__ == "__main__":
    raise SystemExit(main())
