"""Filesystem-backed album/page/photo store.

Layout: ``<data>/albums/<album>/page_NNN/{raw.jpg,thumb.jpg,meta.json,
boxes_edited.json,rectified.jpg,overlay.jpg,photo_XX.jpg}``.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import data_dir

ALBUM_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PAGE_DIR_RE = re.compile(r"^page_(\d+)$")


def albums_root() -> Path:
    return data_dir() / "albums"


def validate_album(album: str) -> str:
    if not ALBUM_NAME_RE.match(album):
        raise ValueError(f"invalid album name {album!r}: letters, digits, '.', '_', '-' only")
    return album


def album_dir(album: str) -> Path:
    return albums_root() / validate_album(album)


def page_dir(album: str, seq: int) -> Path:
    if not isinstance(seq, int) or seq < 1:
        raise ValueError(f"invalid page number: {seq!r}")
    return album_dir(album) / f"page_{seq:03d}"


def list_albums() -> list[str]:
    root = albums_root()
    if not root.is_dir():
        return []
    return [p.name for p in sorted(root.iterdir()) if p.is_dir() and ALBUM_NAME_RE.match(p.name)]


def list_pages(album: str) -> list[int]:
    directory = album_dir(album)
    if not directory.is_dir():
        return []
    seqs = []
    for child in directory.iterdir():
        match = PAGE_DIR_RE.match(child.name)
        if match and child.is_dir():
            seqs.append(int(match.group(1)))
    return sorted(seqs)


def read_meta(directory: Path) -> dict[str, Any] | None:
    path = directory / "meta.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_meta(directory: Path, meta: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    tmp.replace(directory / "meta.json")


def read_edits(directory: Path) -> list[dict[str, Any]] | None:
    path = directory / "boxes_edited.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        photos = payload.get("photos")
        return photos if isinstance(photos, list) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_edits(directory: Path, photos: list[dict[str, Any]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "boxes_edited.json"
    payload = {
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "photos": photos,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)
    return path


def delete_edits(directory: Path) -> None:
    (directory / "boxes_edited.json").unlink(missing_ok=True)


def delete_page(album: str, seq: int) -> None:
    import shutil

    shutil.rmtree(page_dir(album, seq))
