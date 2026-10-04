"""Per-photo perspective warp and crop."""

from __future__ import annotations

import cv2
import numpy as np

from .rectify import order_quad


def expand_quad(quad, margin: float = 0.015) -> np.ndarray:
    """Grow a quad slightly around its centroid to keep photo borders."""
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    return (center + (pts - center) * (1.0 + margin)).astype(np.float32)


def warp_photo(
    page: np.ndarray,
    quad,
    margin: float = 0.015,
    target_long_side: int = 1600,
) -> np.ndarray:
    """Warp a photo quadrilateral to a head-on crop."""
    src = order_quad(expand_quad(quad, margin))
    width = max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3]))
    height = max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1]))
    out_w, out_h = round(width), round(height)
    if out_w < 8 or out_h < 8:
        raise ValueError(f"photo box too small: {out_w}x{out_h}")

    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(
        page, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    scale = target_long_side / max(out_w, out_h)
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return crop
