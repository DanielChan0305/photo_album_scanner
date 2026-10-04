"""Specular-glare detection, masking, and small-area inpainting."""

from __future__ import annotations

import cv2
import numpy as np


def glare_mask(
    image: np.ndarray,
    v_threshold: int = 250,
    s_threshold: int = 25,
    min_area: int = 64,
) -> np.ndarray:
    """Boolean-ish (0/255) mask of bright, desaturated specular pixels.

    Thresholds sit close to true sensor clipping so that white photo borders
    (typically ~240-248) are not mistaken for glare.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, v_threshold), (179, s_threshold, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == label] = 255
    return keep


def glare_fraction(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / mask.size


def clean_glare(
    image: np.ndarray, mask: np.ndarray, max_inpaint_fraction: float = 0.05
) -> tuple[np.ndarray, bool]:
    """Inpaint small glare patches.

    Returns ``(image, blocked)`` where ``blocked`` is True when the glare area is
    too large to inpaint plausibly and should instead be flagged for review.
    """
    fraction = glare_fraction(mask)
    if fraction == 0.0:
        return image, False
    if fraction > max_inpaint_fraction:
        return image, True
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA), False
