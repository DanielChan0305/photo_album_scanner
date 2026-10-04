# Photo Album Scanner

Semi-automated digitization of physical photo albums with a fixed overhead webcam.
A human flips the pages; the app auto-captures each page as it settles, detects the
individual photos, corrects perspective and glare, applies light post-processing, and
saves clean per-photo crops — with a local web UI for reviewing and fixing results.

**Status:** planning — see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full design.

## At a glance

- **Input:** overhead 1080p webcam (4K upgrade path) + a human flipping album pages
- **Output:** `albums/<album>/page_###/photo_##.jpg` plus per-page metadata
- **Stack:** Python (OpenCV, FastAPI, SQLite), local-only
- **Detection:** classical CV first; UI corrections accumulate into training data for a
  small ML detector later

## Quick start

Coming with milestone M0 (environment setup + camera calibration). See the plan.

## Repository layout

See §2 "Repository layout" in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## License

TBD.
