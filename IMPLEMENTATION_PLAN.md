# Photo Album Scanner — Implementation Plan

**Status:** Draft v0.1
**Target machine:** Linux Mint 22.1 · Intel Iris Xe (no NVIDIA GPU) · Python 3.13 venv
**Camera:** existing 1080p USB webcam (upgrade path to 4K)
**Repo:** `github.com/DanielChan0305/photo_album_scanner` (public)

---

## 1. Project goal

Turn a physical photo album into individual, corrected, local image files with minimal human
effort. A person flips album pages under a fixed overhead webcam; the app:

1. detects when a page settles and captures it automatically,
2. finds each photo on the page (classical CV first, ML later),
3. corrects perspective and glare,
4. applies light post-processing,
5. saves each photo as a separate file,
6. and lets the operator review/fix mistakes in a local web UI — every correction doubles
   as training data for a future ML detector.

**Non-goals (v1):** cloud upload, face recognition/grouping, OCR, repairing physical damage,
archival-grade reproduction (a 4K camera upgrade path is documented instead).

**Current assumptions:**

- Camera: 1080p webcam → roughly 160–220 DPI per print depending on framing (see §3).
- Album pages: photos in **plastic sleeves** → glossy, glare is the primary risk; sleeves keep
  photos flat, which greatly reduces warping problems.
- Interface: CLI + local web review UI (FastAPI), served at `localhost:8000`.
- Detection: classical CV first; an ML detector is fine-tuned later on UI corrections.
- Platform: Linux, Python.

---

## 2. Architecture overview

```
 [webcam]
    │
    ▼
 capture daemon ────── raw page JPEG ──────► data/albums/<album>/page_###/raw.jpg
    │  (motion +                                │
    │   stability)                              ▼
    │                                  processing worker (queue)
    │                                  page detect → rectify → glare mask →
    │                                  photo detect → per-photo warp →
    │                                  inpaint → normalize → crops
    │                                           │
    ▼                                           ▼
 live preview (MJPEG)                   SQLite + per-page meta.json
    │                                           │
    └──────────────► FastAPI web UI ◄───────────┘
                    live view · review/fix boxes ·
                    reprocess · export
```

### Components

| Component | Responsibility |
|---|---|
| `camera` | Enumerate/select the webcam, lock exposure/focus/white balance via v4l2, deliver frames. |
| `capture daemon` | Motion detection at low res; capture full-res page raws on page settle; manual trigger; live preview. |
| `processor` | Run the per-page pipeline (rectify → detect → warp → glare → normalize → save); queue-based, resumable. |
| `store` | SQLite schema + file layout; idempotent writes; raw captures are immutable. |
| `web (FastAPI)` | Live MJPEG preview, status, album/page/photo browser, box editor, reprocess, export. |
| `cli` | Thin commands: `pas calibrate`, `pas capture`, `pas process`, `pas export`, `pas serve`. |

### Repository layout

```
photo_album_scanner/
├── IMPLEMENTATION_PLAN.md
├── README.md
├── pyproject.toml
├── src/photo_album_scanner/
│   ├── cli.py               # command entry points
│   ├── config.py            # paths, calibration/settings loading
│   ├── camera.py            # v4l2 controls, format negotiation, frame source
│   ├── capture.py           # motion/stability state machine, capture loop
│   ├── pipeline/
│   │   ├── rectify.py       # page ROI + perspective normalization
│   │   ├── glare.py         # glare detection, masking, inpainting
│   │   ├── detect.py        # classical photo detection
│   │   ├── warp.py          # per-photo homography + crop
│   │   └── normalize.py     # white balance, exposure, sharpen
│   ├── store.py             # SQLite + file layout
│   └── web/
│       ├── app.py           # FastAPI routes + MJPEG stream
│       └── static/          # single-page review UI (vanilla JS/canvas)
├── tests/
│   ├── fixtures/            # sample page images + expected boxes
│   └── ...
└── data/                    # git-ignored, created at runtime
    ├── albums/<album>/page_###/{raw.jpg, meta.json, photo_##.jpg}
    └── scanner.db
```

### Data model (SQLite)

```sql
albums(id, name, created_at, notes)
pages(id, album_id, seq, raw_path, captured_at, status, processed_at, error)
photos(id, page_id, box_json, crop_path, confidence, qc_flags, corrected, accepted)
labels(id, page_id, photo_id, source, payload_json)   -- corrections destined for ML training
settings(key, value_json)                              -- calibration: ROI, thresholds, reference
```

- `pages.status`: `captured | processing | processed | error | accepted`
- `photos.qc_flags`: e.g. `glare`, `low_confidence`, `touching_neighbor`, `near_border`
- Raws are never modified; reprocessing a page replaces derived crops and rows (idempotent).

---

## 3. Hardware & rig

### Camera

- Fixed overhead mount, camera plane parallel to the table; **frame one album page tightly**
  (not the whole spread) to maximize resolution.
- 1080p resolution math: a 30 cm-wide page gives ~6.4 px/mm ≈ **160 DPI** for a 15 cm print;
  framing a 22 cm page gives ~**220 DPI**. Good for digital viewing, weak for reprints.
- Upgrade path: 4K webcam doubles this with no software changes.

### Lighting & glare (the make-or-break item)

Apply in this order — each step is cheap and the first two usually suffice:

1. **Geometry:** two diffuse lamps at the *sides* of the album, angled steeply (grazing the
   page). Turn off overhead room lights. Reflection reaches the lens only when the surface
   normal bisects light and view direction — grazing light avoids that.
2. **Block ceiling reflections:** matte dark board above/behind the camera; dark mat around
   the album to reduce stray reflections and to help page-boundary detection.
3. **Polarization:** polarizing film over both lamps + a polarizer over the webcam lens,
   rotated to null the lamps. Cheap (~€20), very effective on plastic sleeves.
4. **Camera offset:** 5–10° off perpendicular pushes reflections out of view; the homography
   corrects the resulting perspective. Use only if needed.

### Page flattening

- Sleeves keep photos flat; page curl only matters near the binding.
- Optional: clear acrylic sheet to press the page flat (adds its own reflection — handled by
  the lighting above).

### Camera settings (locked after calibration)

- Manual exposure, manual focus, manual white balance (`v4l2-ctl`).
- Prefer a format without compression artifacts: YUYV at 1080p if the camera offers it at a
  workable fps; otherwise MJPEG (accept artifacts, use frame averaging/denoise).
- Watchdog/reconnect logic if the camera sleeps (UVC autosuspend).

---

## 4. Pipeline stages (detail)

### 4.0 Calibration (once per rig setup)

- Identify the webcam node; enumerate formats/frame rates; pick the capture format.
- Lock exposure/focus/WB; verify 60 s stability (mean brightness drift < 2%).
- Mark album placement with tape; save the page ROI polygon, lighting reference, and
  thresholds to `settings`.

### 4.1 Motion detection & auto-capture

- Work at 640×360 grayscale, ~10 fps.
- `cv2.absdiff` against an exponential moving-average background → threshold → morphology →
  changed-pixel ratio.
- State machine: `IDLE → MOTION → STABLE (changed < 0.5 % for 0.7 s) → CAPTURE → COOLDOWN`
  (remain in cooldown until motion resumes, preventing double captures).
- Guards: reject capture if the ROI is severely saturated (glare) or a hand-shaped blob is
  still inside the ROI; retry after the next settle.
- Manual capture always available (UI button, key, USB foot pedal) as a fallback.
- Save full-res JPEG (quality 95) + timestamp + preview thumbnail.

### 4.2 Page localization & rectification

- Default: calibrated fixed ROI → perspective warp to canonical page size.
- Optional refresh: detect the page contour against the dark mat; sanity-check against ROI;
  if far off, flag the page rather than silently warping.
- Output: rectified page image used by all later stages.

### 4.3 Glare handling

- Mask: HSV `V ≥ 235 && S ≤ 40`, morphological open/dilate; ignore specks < ~0.01 % area.
- Detection input uses a copy where masked pixels are replaced by local median (avoids fake
  edges and fake photo borders).
- Output: inpaint small regions (`cv2.inpaint`, Telea, r=3); regions > ~2 % of a photo area
  get `qc_flags=glare` and are surfaced in the UI. Optional later: LaMa deep inpainting.
- Future hardware trick: alternate left/right lights per capture and composite the pixels
  where each frame has no specularity.

### 4.4 Photo detection (classical, v1)

1. Downscale to ~1600 px long side for speed.
2. Bilateral filter → Canny (auto thresholds) → morphological close to firm up borders.
3. Contours → `approxPolyDP` (ε ≈ 2 % of perimeter) → keep quadrilaterals.
4. Filters: min area ≥ ~4 % of page; min side ≥ 150 px full-res; aspect ratio 0.4–2.5;
   rectangularity (contour area / quad area) ≥ 0.8; edge support (Canny overlap on the quad
   perimeter) above threshold.
5. Dedupe: IoU > 0.5 keep the higher score; drop boxes contained in larger ones.
6. Reading order: cluster by row (y-overlap), then sort by x.
7. Confidence: weighted edge support, area plausibility, aspect, contrast vs page.
8. Expected failure modes: photos touching with no gap; low-contrast page/photo pairs. UI
   corrections cover these; the ML phase fixes them at scale.

### 4.5 Per-photo warp & crop

- Expand each box by 1–2 % margin (preserve photo borders), clamp to page bounds.
- Homography from quad → rectangle preserving detected aspect; target long side ~1600 px
  (only upscale when source is ≥ 80 % of target).
- Save JPEG quality 92.

### 4.6 Normalization

- White balance: gray-world/white-patch over border margins when available, else page-level.
- Exposure: match page median to the calibration reference; soft highlight rolloff for
  residual glare.
- Sharpen: unsharp mask (amount ≈ 0.6, radius 1.0); optional mild denoise for MJPEG sources.

### 4.7 Save & metadata

- Per page: `meta.json` with capture time, raw path, boxes/scores, QC flags, pipeline version.
- DB rows updated atomically; reprocessing is idempotent; raws never modified.

### 4.8 Review UI

- **Live:** MJPEG preview, capture-state badge, "Capture now", session page counter.
- **Browse:** albums → pages → photos with thumbnails; QC flags highlighted.
- **Page editor:** canvas overlay to add/delete/drag/resize boxes, "Reprocess page",
  "Accept page". Corrections set `corrected=1` and emit `labels` rows.
- All offline/local; no external services.

### 4.9 Export

- `data/exports/<album>_<date>/` containing `page_###_photo_##.jpg`; optional ZIP.
- Manifest in CSV/JSON; optional contact-sheet images per album.
- Later: contact-sheet PDF; face-based grouping (out of scope v1).

---

## 5. ML phase (after corrections accumulate)

- Export corrected pages (≥ 50 suggested) as a bounding-box dataset in YOLO format —
  generated from `labels`, no separate labeling effort.
- Fine-tune a small detector (YOLOv8n / YOLO11n); 80/20 split; target mAP@0.5 ≥ 0.9.
- Integrate: detector proposals → classical quad refinement → same downstream pipeline.
- Runtime: ONNX Runtime or OpenVINO (Iris Xe / CPU), well under 1 s per page.
- Optional: LaMa inpainting for large glare areas; consider only if software glare remains
  a real problem after hardware fixes.

---

## 6. Milestones

| # | Scope | Deliverable | Acceptance criteria | Est. |
|---|---|---|---|---|
| M0 | Env + calibration | venv, deps, `pas calibrate` (camera identify/lock/formats), sample frames | Locked settings stable 60 s; test frames captured | 0.5 d |
| M1 | Auto-capture | Daemon with motion/stability trigger + manual trigger | 20 page flips → exactly 20 captures, no duplicates | 0.5 d |
| M2 | Detection v1 | Rectify + classical detection + crops + `meta.json` + contact sheet | ≥ 90 % photos found on 20 test pages (moderate glare) | 1–2 d |
| M3 | UI v1 | Live preview, page editor, reprocess, QC flags | Correct a page in < 60 s; corrections persisted | 1–2 d |
| M4 | Quality polish | Glare inpaint, normalization, export/ZIP | Full album exported; subjective QA pass | 1 d |
| M5 | ML detector (optional) | Dataset export + fine-tune + integration | mAP@0.5 ≥ 0.9; ≥ classical recall on hard pages | 1–2 d |

Each milestone leaves the tool usable on its own. Offline processing of existing images is
supported from M2 onward so development doesn't require the physical rig.

---

## 7. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Glare streaks on sleeves | Broken crops, unusable photos | Lighting geometry first; polarizers; inpaint; QC flag + recapture; alternating-light compositing later |
| Photos touching / low contrast | Missed or merged detections | UI correction; corrections feed ML training |
| MJPEG artifacts/noise | Soft images | Prefer YUYV; frame averaging; mild denoise; 4K upgrade later |
| Auto-capture false triggers | Junk pages | Stability + ROI/hand guards; manual trigger; trivial delete in UI |
| Page curl near binding | Warp errors | Sleeves mostly flat; acrylic sheet; manual re-box; later mesh dewarp (low priority) |
| Python 3.14 wheel gaps | Setup blocked | Use a Python 3.13 venv |
| Camera sleep/autosuspend | Daemon stalls | Disable UVC autosuspend; watchdog + reconnect |
| White balance drift across pages | Inconsistent colors | Lock WB; per-page normalization to calibration reference |

---

## 8. Testing strategy

- **Offline mode:** `pas process --dir <folder>` processes fixture images without a camera;
  used for all automated tests.
- **Unit tests:** glare mask, quad filtering, NMS, reading order, homography geometry,
  store idempotency.
- **Golden fixtures:** annotated sample pages (expected boxes JSON) → detection
  recall/precision regression tests.
- **Manual QA checklist:** per album — photo count matches, orientation correct, QC flags resolved.

---

## 9. Dependencies & environment setup

System packages:

```bash
sudo apt install v4l-utils python3.13-venv
```

Python (inside project venv):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

- Runtime: `opencv-python`, `numpy`, `fastapi`, `uvicorn`, `pydantic`, `pillow`.
- Dev: `pytest`, `ruff`.
- Later: `ultralytics`/`torch` (training) or `onnxruntime`/`openvino` (inference).

### Quick sanity commands

```bash
v4l2-ctl --list-devices              # identify the webcam node
v4l2-ctl -d /dev/videoX --list-formats-ext
v4l2-ctl -d /dev/videoX --set-ctrl=auto_exposure=1     # example: lock exposure
```

---

## 10. Open questions

- Album page dimensions and max photos per page (affects canonical rect size and thresholds).
- Do we want a USB foot pedal for manual capture? (nice-to-have, cheap)
- Export preferences: JPEG quality, folder naming scheme, ZIP or plain folders?
