"""Processing pipeline: rectified page -> detected photos -> crops + metadata."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .. import store
from ..config import Settings
from . import detect, glare, rectify, warp

PAGE_DIR_RE = re.compile(r"^page_(\d+)$")


@dataclass
class PhotoResult:
    index: int
    quad: list[list[float]]
    confidence: float
    flags: list[str] = field(default_factory=list)
    filename: str | None = None


@dataclass
class PageResult:
    seq: int
    width: int
    height: int
    photos: list[PhotoResult]
    glare_fraction: float
    glare_blocked: bool
    edited: bool
    processed_at: str


def render_overlay(page: np.ndarray, photos: list[PhotoResult]) -> np.ndarray:
    """Debug overlay: detected boxes and indices drawn on the page."""
    overlay = page.copy()
    for photo in photos:
        points = np.asarray(photo.quad, dtype=np.int32).reshape(4, 1, 2)
        color = (0, 200, 0) if not photo.flags else (0, 165, 255)
        cv2.polylines(overlay, [points], True, color, 3)
        x = int(min(point[0] for point in photo.quad))
        y = int(min(point[1] for point in photo.quad))
        cv2.putText(
            overlay, str(photo.index), (x + 8, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
        )
    return overlay


def _encode(frame: np.ndarray, quality: int) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()


def process_page(
    page: Path, settings: Settings | None = None, redetect: bool = False
) -> PageResult:
    """Run the full pipeline for one captured page folder.

    Edited boxes (``boxes_edited.json``) take precedence over detection unless
    ``redetect`` is set, in which case they are discarded first.
    """
    settings = settings or Settings.load()
    frame = cv2.imread(str(page / "raw.jpg"))
    if frame is None:
        raise FileNotFoundError(f"cannot read {page / 'raw.jpg'}")

    rectified, corners = rectify.rectify(frame, settings.page_roi)
    mask = glare.glare_mask(rectified)
    glare_fraction = glare.glare_fraction(mask)

    if redetect:
        store.delete_edits(page)
    edits = store.read_edits(page)

    if edits is not None:
        photos = [
            PhotoResult(
                index=index,
                quad=[[float(point[0]), float(point[1])] for point in entry["quad"]],
                confidence=float(entry.get("confidence", 1.0)),
                flags=list(entry.get("flags", [])),
            )
            for index, entry in enumerate(edits, start=1)
        ]
        edited = True
    else:
        detections = detect.detect_photos(rectified)
        photos = [
            PhotoResult(index=i, quad=d.quad, confidence=d.confidence, flags=list(d.flags))
            for i, d in enumerate(detections, start=1)
        ]
        edited = False

    cleaned, glare_blocked = glare.clean_glare(rectified, mask)
    if glare_blocked:
        for photo in photos:
            photo.flags.append("glare")

    # Remove stale crops, then write current ones.
    for old in page.glob("photo_*.jpg"):
        old.unlink()
    for photo in photos:
        try:
            crop = warp.warp_photo(cleaned, photo.quad)
        except ValueError:
            photo.flags.append("invalid_box")
            continue
        filename = f"photo_{photo.index:02d}.jpg"
        (page / filename).write_bytes(_encode(crop, 92))
        photo.filename = filename

    (page / "rectified.jpg").write_bytes(_encode(cleaned, 92))
    overlay = render_overlay(cleaned, photos)
    (page / "overlay.jpg").write_bytes(_encode(overlay, 85))

    match = PAGE_DIR_RE.match(page.name)
    seq = int(match.group(1)) if match else 0
    processed_at = datetime.now(UTC).isoformat(timespec="seconds")
    result = PageResult(
        seq=seq,
        width=cleaned.shape[1],
        height=cleaned.shape[0],
        photos=photos,
        glare_fraction=glare_fraction,
        glare_blocked=glare_blocked,
        edited=edited,
        processed_at=processed_at,
    )

    store.write_meta(
        page,
        {
            "seq": seq,
            "processed_at": processed_at,
            "width": result.width,
            "height": result.height,
            "glare_fraction": round(glare_fraction, 5),
            "glare_blocked": glare_blocked,
            "edited": edited,
            "rectify_corners": corners.tolist() if corners is not None else None,
            "photos": [asdict(photo) for photo in photos],
        },
    )
    return result
