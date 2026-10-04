# Photo Album Scanner

Semi-automated digitization of physical photo albums with a fixed overhead webcam.
A human flips the pages; the app auto-captures each page as it settles, detects the
individual photos, corrects perspective and glare, applies light post-processing, and
saves clean per-photo crops — with a local web UI for reviewing and fixing results.

**Status:** M0 in progress — camera inspection and calibration tooling implemented
(see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full design).

## At a glance

- **Input:** overhead 1080p webcam (4K upgrade path) + a human flipping album pages
- **Output:** `albums/<album>/page_###/photo_##.jpg` plus per-page metadata
- **Stack:** Python (OpenCV, FastAPI, SQLite), local-only
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

## Repository layout

See §2 "Repository layout" in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## License

TBD.
