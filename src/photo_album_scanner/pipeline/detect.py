"""Classical photo detection on a rectified album page.

Approach: edge extraction (Canny) at a bounded working resolution, morphological
closing to firm up borders, contour approximation to quadrilaterals, geometric
filtering (area, aspect, rectangularity, minimum side), dedupe of overlapping
and containing boxes, then reading-order sort.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DETECT_LONG_SIDE = 1600
MIN_AREA_RATIO = 0.02
MAX_AREA_RATIO = 0.85
MIN_SIDE_FULL_RES = 150
MAX_ASPECT = 2.5
MIN_RECTANGULARITY = 0.8
DEDUPE_IOU = 0.5
CONTAINMENT_RATIO = 0.85
LOW_CONFIDENCE = 0.5


@dataclass
class Detection:
    quad: list[list[float]]
    confidence: float
    flags: list[str] = field(default_factory=list)


def _bbox(quad) -> tuple[float, float, float, float]:
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return min(xs), min(ys), max(xs), max(ys)


def _quad_area(quad) -> float:
    points = np.asarray(quad, dtype=np.float32).reshape(4, 1, 2)
    return abs(float(cv2.contourArea(points)))


def _intersection_area(a_quad, b_quad) -> float:
    ax1, ay1, ax2, ay2 = _bbox(a_quad)
    bx1, by1, bx2, by2 = _bbox(b_quad)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _iou(a_quad, b_quad) -> float:
    intersection = _intersection_area(a_quad, b_quad)
    if intersection <= 0:
        return 0.0
    union = _quad_area(a_quad) + _quad_area(b_quad) - intersection
    return intersection / union if union > 0 else 0.0


def _containment(a_quad, b_quad) -> float:
    intersection = _intersection_area(a_quad, b_quad)
    smaller = min(_quad_area(a_quad), _quad_area(b_quad))
    return intersection / smaller if smaller > 0 else 0.0


def _edge_support(edges: np.ndarray, quad: np.ndarray, samples_per_edge: int = 48) -> float:
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    points = []
    for index in range(4):
        start, end = quad[index], quad[(index + 1) % 4]
        for t in np.linspace(0.0, 1.0, samples_per_edge, endpoint=False):
            points.append(start + t * (end - start))
    points = np.asarray(points, dtype=np.float32)
    xs = np.clip(points[:, 0].round().astype(int), 0, dilated.shape[1] - 1)
    ys = np.clip(points[:, 1].round().astype(int), 0, dilated.shape[0] - 1)
    return float(np.count_nonzero(dilated[ys, xs])) / len(points)


def _dedupe(detections: list[Detection]) -> list[Detection]:
    groups: list[list[Detection]] = []
    for detection in detections:
        for group in groups:
            if any(
                _iou(detection.quad, member.quad) > DEDUPE_IOU
                or _containment(detection.quad, member.quad) > CONTAINMENT_RATIO
                for member in group
            ):
                group.append(detection)
                break
        else:
            groups.append([detection])

    representatives: list[Detection] = []
    for group in groups:
        best_confidence = max(detection.confidence for detection in group)
        plausible = [
            detection for detection in group if detection.confidence >= 0.7 * best_confidence
        ]
        # Prefer the largest plausible box: a photo with its white border, not
        # just the picture content inside it.
        representatives.append(max(plausible, key=lambda d: _quad_area(d.quad)))
    return representatives


def sort_reading_order(detections: list[Detection]) -> list[Detection]:
    """Top-to-bottom, left-to-right, with tolerance for slightly uneven rows."""
    if not detections:
        return []

    def center(detection: Detection) -> tuple[float, float]:
        x1, y1, x2, y2 = _bbox(detection.quad)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    heights = [_bbox(detection.quad)[3] - _bbox(detection.quad)[1] for detection in detections]
    row_tolerance = float(np.median(heights)) * 0.6 if heights else 0.0

    rows: list[list[Detection]] = []
    for detection in sorted(detections, key=lambda d: center(d)[1]):
        if rows and abs(center(rows[-1][0])[1] - center(detection)[1]) <= row_tolerance:
            rows[-1].append(detection)
        else:
            rows.append([detection])

    ordered: list[Detection] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda d: center(d)[0]))
    return ordered


def detect_photos(page: np.ndarray) -> list[Detection]:
    """Detect photo quadrilaterals in a rectified page image."""
    height, width = page.shape[:2]
    scale = min(1.0, DETECT_LONG_SIDE / max(height, width))
    small = (
        cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else page
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    page_area = float(small.shape[0] * small.shape[1])
    candidates: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / page_area
        if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        quad_small = approx.reshape(4, 2).astype(np.float32)
        quad_area = cv2.contourArea(quad_small.reshape(4, 1, 2))
        if quad_area <= 0:
            continue
        rectangularity = area / quad_area
        if rectangularity < MIN_RECTANGULARITY:
            continue

        (_, (rect_w, rect_h), _) = cv2.minAreaRect(quad_small.reshape(4, 1, 2))
        if rect_w <= 0 or rect_h <= 0:
            continue
        long_side, short_side = max(rect_w, rect_h), min(rect_w, rect_h)
        if long_side / short_side > MAX_ASPECT:
            continue
        if short_side / scale < MIN_SIDE_FULL_RES:
            continue

        support = _edge_support(edges, quad_small)
        confidence = round(
            0.6 * support + 0.25 * min(1.0, rectangularity) + 0.15 * min(1.0, area_ratio / 0.10),
            3,
        )
        flags = ["low_confidence"] if confidence < LOW_CONFIDENCE else []
        candidates.append(
            Detection(quad=(quad_small / scale).astype(float).tolist(), confidence=confidence, flags=flags)
        )

    return sort_reading_order(_dedupe(candidates))
