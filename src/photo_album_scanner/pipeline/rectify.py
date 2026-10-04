"""Page localization and perspective rectification."""

from __future__ import annotations

import cv2
import numpy as np

MIN_PAGE_AREA_RATIO = 0.15
DETECT_LONG_SIDE = 960


def order_quad(points) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = pts[:, 1] - pts[:, 0]  # y - x
    ordered[0] = pts[np.argmin(sums)]  # top-left
    ordered[2] = pts[np.argmax(sums)]  # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def detect_page_corners(frame: np.ndarray) -> np.ndarray | None:
    """Largest plausible page quadrilateral in camera coordinates, or None."""
    height, width = frame.shape[:2]
    scale = min(1.0, DETECT_LONG_SIDE / max(height, width))
    small = (
        cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else frame
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(small.shape[0] * small.shape[1])
    best: np.ndarray | None = None
    best_area = frame_area * MIN_PAGE_AREA_RATIO
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= best_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best, best_area = approx, area

    if best is None:
        return None
    return (best.reshape(4, 2) / scale).astype(np.float32)


def rectify(
    frame: np.ndarray, roi=None, output_long_side: int = 2000
) -> tuple[np.ndarray, np.ndarray | None]:
    """Warp the page to a top-down view.

    ``roi`` is an optional calibrated quadrilateral (4 points, camera coords);
    without it the page is detected automatically. Falls back to the full frame
    when no page can be found. Returns ``(image, corners_or_None)``.
    """
    corners = (
        np.asarray(roi, dtype=np.float32).reshape(4, 2)
        if roi
        else detect_page_corners(frame)
    )
    if corners is None:
        return frame.copy(), None

    src = order_quad(corners)
    top = float(np.linalg.norm(src[1] - src[0]))
    bottom = float(np.linalg.norm(src[2] - src[3]))
    left = float(np.linalg.norm(src[3] - src[0]))
    right = float(np.linalg.norm(src[2] - src[1]))
    out_w = max(64, round(max(top, bottom)))
    out_h = max(64, round(max(left, right)))

    scale = min(1.0, output_long_side / max(out_w, out_h))
    out_w = max(64, int(out_w * scale))
    out_h = max(64, int(out_h * scale))

    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return warped, src
