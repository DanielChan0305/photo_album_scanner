# Photo Album Scanner

Semi-automated digitization of physical photo albums with an overhead camera.
A human flips the pages; the app auto-captures each page as it settles, detects the
individual photos, corrects perspective and glare, applies light post-processing, and
saves clean per-photo crops — with a local web UI for reviewing and fixing results.

**Status:** M0–M3 implemented — capture, photo detection/cropping, and a web review UI
(see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full design).

## At a glance

- **Input:** phone or webcam (V4L2 device or MJPEG/RTSP stream URL) + a human flipping pages
- **Output:** `albums/<album>/page_###/photo_##.jpg` plus per-page metadata
- **Stack:** Python (OpenCV, FastAPI), local-only
- **Detection:** classical CV first; UI corrections accumulate into training data for a
  small ML detector later

## Quick start

Requires Linux with V4L2 and [uv](https://docs.astral.sh/uv/) (or any Python 3.11+).

```bash
uv venv --python 3.13
uv pip install -e '.[dev]'

source .venv/bin/activate

pas devices          # list /dev/video* devices
pas inspect          # formats, frame rates, and controls of the selected camera
pas calibrate        # lock auto exposure/focus/WB + save test frames to data/calibration/
pas capture --album album_01   # auto-capture pages as they settle
```

`pas calibrate` writes the chosen device, capture format, and locked control values to
`data/settings.json`, which later milestones read at startup.

While `pas capture` runs, press `c` for a manual capture and `q` to quit (when run
non-interactively, `kill -USR1 <pid>` does the same). Each settled page is saved once to
`data/albums/<album>/page_###/raw.jpg` (+ `thumb.jpg`), with a manifest in
`data/albums/<album>/captures.jsonl`. Use `--max-captures N` for a quick test run.

Then process and review:

```bash
pas process            # run photo detection over all captured pages
pas serve              # web UI at http://127.0.0.1:8000
```

### Using your phone as the camera

Phone cameras beat typical webcams. The easiest route needs no kernel modules or sudo:
run an app that serves an MJPEG/HTTP stream and point `--source` at its URL.

**Android — IP Webcam**

1. Install *IP Webcam*, set the video resolution to maximum, and start the server.
2. Note the URL it shows (e.g. `http://192.168.1.23:8080`).
3. Calibrate and capture:

```bash
pas calibrate --source http://192.168.1.23:8080/video \
              --still-url http://192.168.1.23:8080/photo.jpg --frames 5
pas capture --album album_01     # reuses the saved source and still URL
```

With `--still-url`, every capture fetches a **full-resolution still** (`/photo.jpg`)
instead of a compressed video frame — noticeably sharper for print copies. Set
exposure/focus in the phone app; network streams have no V4L2 controls (the scanner
skips camera locking automatically).

**iOS** — any app that serves MJPEG or RTSP works the same way:
`--source http://PHONE_IP:PORT/video` or `--source rtsp://...`.

**Android 14+ without an app** — Developer options → *USB webcam* exposes the phone as a
standard UVC device (`/dev/videoN`); then everything works exactly like a webcam,
including `pas inspect`/control locking.

DroidCam/Iriun (via `v4l2loopback`) also work since they appear as `/dev/videoN`, but
need `sudo` and a kernel module; the stream-URL route avoids that.

### Web review UI

- **Live capture:** Start/Stop and "Capture now" in the header, with a live preview while
  you flip pages. Each captured page is processed automatically and appears in the list.
- **Page browser:** albums and pages with thumbnails, photo counts, QC flags, and errors.
- **Box editor:** drag, resize, add, or delete photo boxes directly on the page image
  ("Add box" then drag; Delete removes the selected box). **Save & reprocess** regenerates
  crops using your boxes; **Re-detect** discards edits and runs detection again.
- **Output:** `rectified.jpg` (top-down page), `overlay.jpg` (detections drawn),
  `photo_XX.jpg` crops, `meta.json`, and `boxes_edited.json` for corrections. All local,
  in `data/`.

## Repository layout

See §2 "Repository layout" in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## License

TBD.
