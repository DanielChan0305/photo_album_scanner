"""FastAPI application serving the review UI and JSON API."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import store
from ..config import Settings
from ..pipeline import process_page
from .service import CaptureService

STATIC_DIR = Path(__file__).resolve().parent / "static"
FILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.(jpg|jpeg|png)$")


class BoxIn(BaseModel):
    quad: list[list[float]] = Field(min_length=4, max_length=4)


class BoxesIn(BaseModel):
    photos: list[BoxIn]


class CaptureStartIn(BaseModel):
    album: str = "album_01"
    process: bool = True


def create_app() -> FastAPI:
    app = FastAPI(title="Photo Album Scanner")
    service = CaptureService()

    # ------------------------------------------------------------------
    # Static UI
    # ------------------------------------------------------------------

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ------------------------------------------------------------------
    # Albums / pages
    # ------------------------------------------------------------------

    @app.get("/api/albums")
    def albums() -> list[dict[str, Any]]:
        result = []
        for name in store.list_albums():
            pages = store.list_pages(name)
            processed = sum(1 for seq in pages if store.read_meta(store.page_dir(name, seq)))
            result.append({"name": name, "pages": len(pages), "processed": processed})
        return result

    @app.get("/api/albums/{album}/pages")
    def pages(album: str) -> list[dict[str, Any]]:
        _require_album(album)
        return [_page_summary(album, seq) for seq in store.list_pages(album)]

    @app.get("/api/albums/{album}/pages/{seq}")
    def page_detail(album: str, seq: int) -> dict[str, Any]:
        directory = _page_dir_or_404(album, seq)
        meta = store.read_meta(directory)
        photos = []
        for entry in (meta or {}).get("photos", []):
            filename = entry.get("filename")
            photos.append(
                {
                    **entry,
                    "bbox": _bbox(entry.get("quad", [])),
                    "url": _file_url(album, seq, filename) if filename else None,
                }
            )
        return {
            "album": album,
            "seq": seq,
            "processed": meta is not None,
            "edited": bool(meta and meta.get("edited")),
            "error": (meta or {}).get("error"),
            "width": (meta or {}).get("width"),
            "height": (meta or {}).get("height"),
            "glare_fraction": (meta or {}).get("glare_fraction"),
            "glare_blocked": (meta or {}).get("glare_blocked"),
            "rectify_corners": (meta or {}).get("rectify_corners"),
            "image": _file_url(album, seq, "rectified.jpg")
            if (directory / "rectified.jpg").is_file()
            else _file_url(album, seq, "raw.jpg"),
            "raw_image": _file_url(album, seq, "raw.jpg")
            if (directory / "raw.jpg").is_file()
            else None,
            "overlay": _file_url(album, seq, "overlay.jpg")
            if (directory / "overlay.jpg").is_file()
            else None,
            "photos": photos,
        }

    @app.get("/api/albums/{album}/pages/{seq}/file/{filename}")
    def page_file(album: str, seq: int, filename: str) -> FileResponse:
        if not FILE_NAME_RE.match(filename):
            raise HTTPException(status_code=404, detail="invalid file name")
        directory = _page_dir_or_404(album, seq)
        target = (directory / filename).resolve()
        if target.parent != directory.resolve() or not target.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(target)

    @app.post("/api/albums/{album}/pages/{seq}/boxes")
    def save_boxes(album: str, seq: int, payload: BoxesIn) -> dict[str, Any]:
        directory = _page_dir_or_404(album, seq)
        store.write_edits(directory, [{"quad": box.quad} for box in payload.photos])
        _run_processing(directory)
        return page_detail(album, seq)

    @app.post("/api/albums/{album}/pages/{seq}/process")
    def reprocess(album: str, seq: int, redetect: bool = False) -> dict[str, Any]:
        directory = _page_dir_or_404(album, seq)
        _run_processing(directory, redetect=redetect)
        return page_detail(album, seq)

    @app.delete("/api/albums/{album}/pages/{seq}")
    def delete_page(album: str, seq: int) -> dict[str, bool]:
        directory = _page_dir_or_404(album, seq)
        shutil.rmtree(directory)
        return {"deleted": True}

    # ------------------------------------------------------------------
    # Live capture
    # ------------------------------------------------------------------

    @app.post("/api/capture/start")
    def capture_start(payload: CaptureStartIn) -> dict[str, Any]:
        try:
            return service.start(payload.album, process=payload.process)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/capture/stop")
    def capture_stop() -> dict[str, Any]:
        return service.stop()

    @app.post("/api/capture/manual")
    def capture_manual() -> dict[str, Any]:
        return service.manual_capture()

    @app.get("/api/capture/status")
    def capture_status() -> dict[str, Any]:
        return service.status()

    @app.get("/api/capture/preview.jpg")
    def capture_preview() -> Response:
        data = service.preview()
        if data is None:
            return Response(status_code=204)
        return Response(content=data, media_type="image/jpeg")

    return app


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _run_processing(directory: Path, redetect: bool = False) -> None:
    try:
        process_page(directory, Settings.load(), redetect=redetect)
    except Exception as exc:  # noqa: BLE001 - report processing errors to the UI
        meta = store.read_meta(directory) or {}
        meta["error"] = str(exc)
        store.write_meta(directory, meta)


def _page_summary(album: str, seq: int) -> dict[str, Any]:
    directory = store.page_dir(album, seq)
    meta = store.read_meta(directory)
    photos = (meta or {}).get("photos", [])
    flags: set[str] = set()
    for photo in photos:
        flags.update(photo.get("flags", []))
    if meta and meta.get("glare_blocked"):
        flags.add("glare")
    return {
        "seq": seq,
        "processed": meta is not None,
        "edited": bool(meta and meta.get("edited")),
        "error": (meta or {}).get("error"),
        "photo_count": len(photos),
        "flags": sorted(flags),
        "thumb": _file_url(album, seq, "thumb.jpg")
        if (directory / "thumb.jpg").is_file()
        else None,
    }


def _file_url(album: str, seq: int, filename: str | None) -> str | None:
    if filename is None:
        return None
    return f"/api/albums/{album}/pages/{seq}/file/{filename}"


def _bbox(quad: list) -> list[float] | None:
    if not quad:
        return None
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _require_album(album: str) -> None:
    try:
        store.validate_album(album)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _page_dir_or_404(album: str, seq: int) -> Path:
    try:
        directory = store.page_dir(album, seq)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="page not found")
    return directory
