"""Classical photo detection on a rectified album page.

Approach: several edge maps (raw / CLAHE-boosted grayscale × two Canny threshold
pairs) are cleaned of speckle by filtering small connected components. Contours
are fitted with a rotated rectangle (``minAreaRect``) rather than requiring an
exactly-4-vertex polygon — soft or glare-fragmented borders rarely approximate to
four points. Candidates are filtered by area, aspect, minimum side, and how much
of the fitted rectangle the contour fills, then deduped and sorted in reading
order.

Sensitivity is controlled by :class:`~photo_album_scanner.config.DetectionConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import DetectionConfig

DETECT_LONG_SIDE = 1600


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


def _edge_maps(gray: np.ndarray, config: DetectionConfig) -> list[np.ndarray]:
    bases = [gray]
    if config.clahe:
        bases.append(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray))
    maps = []
    for base in bases:
        for low, high in config.canny_pairs:
            maps.append(cv2.Canny(base, low, high))
    return maps


def _clean_edges(edges: np.ndarray, min_component: int, close_kernel: int) -> np.ndarray:
    """Drop speckle edge components (interior texture) while keeping long lines."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    clean = np.zeros_like(edges)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_component:
            clean[labels == label] = 255
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    return clean


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


def _dedupe(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    groups: list[list[Detection]] = []
    for detection in detections:
        for group in groups:
            if any(
                _iou(detection.quad, member.quad) > iou_threshold
                or _containment(detection.quad, member.quad) > 0.85
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
        representative = max(plausible, key=lambda d: d.confidence)
        # Prefer a slightly larger enclosing box (photo with its border), but not
        # a much larger one (that is a merged multi-photo region).
        area = _quad_area(representative.quad)
        containers = [
            detection
            for detection in plausible
            if _containment(detection.quad, representative.quad) > 0.85
            and _quad_area(detection.quad) <= 1.6 * area
        ]
        if containers:
            representative = max(containers, key=lambda d: _quad_area(d.quad))
        representatives.append(representative)
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


def _candidates_from_edges(
    edges: np.ndarray,
    support_map: np.ndarray,
    page_area: float,
    scale: float,
    config: DetectionConfig,
) -> list[Detection]:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / page_area
        if area_ratio < config.min_area_ratio or area_ratio > config.max_area_ratio:
            continue

        rect = cv2.minAreaRect(contour)
        (rect_w, rect_h) = rect[1]
        if rect_w <= 0 or rect_h <= 0:
            continue
        rectangularity = area / (rect_w * rect_h)
        if rectangularity < config.min_rectangularity:
            continue

        long_side, short_side = max(rect_w, rect_h), min(rect_w, rect_h)
        if long_side / short_side > config.max_aspect:
            continue
        if short_side / scale < config.min_side:
            continue

        quad_small = cv2.boxPoints(rect).astype(np.float32)
        support = _edge_support(support_map, quad_small)
        confidence = round(
            0.6 * support
            + 0.25 * min(1.0, rectangularity)
            + 0.15 * min(1.0, area_ratio / 0.10),
            3,
        )
        flags = ["low_confidence"] if confidence < config.low_confidence else []
        candidates.append(
            Detection(
                quad=(quad_small / scale).astype(float).tolist(),
                confidence=confidence,
                flags=flags,
            )
        )
    return candidates


def detect_photos(
    page: np.ndarray, config: DetectionConfig | None = None
) -> list[Detection]:
    """Detect photo quadrilaterals in a rectified page image."""
    config = config or DetectionConfig()

    height, width = page.shape[:2]
    scale = min(1.0, DETECT_LONG_SIDE / max(height, width))
    small = (
        cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else page
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    cleaned_maps = [
        _clean_edges(edges, config.min_edge_component, config.close_kernel)
        for edges in _edge_maps(gray, config)
    ]
    support_map = np.zeros_like(cleaned_maps[0])
    for edges in cleaned_maps:
        support_map = cv2.bitwise_or(support_map, edges)

    page_area = float(small.shape[0] * small.shape[1])
    candidates: list[Detection] = []
    for edges in cleaned_maps:
        candidates.extend(
            _candidates_from_edges(edges, support_map, page_area, scale, config)
        )

    return sort_reading_order(_dedupe(candidates, config.dedupe_iou))
