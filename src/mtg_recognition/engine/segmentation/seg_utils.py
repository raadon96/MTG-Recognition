"""Shared geometric helpers for card segmentation."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import find_corners_lines_intersect, perspective_transform

# MTG standard card dimensions: 63 mm × 88 mm (portrait)
CARD_ASPECT_RATIO = 63.0 / 88.0  # ≈ 0.716


def _corners_from_binary_mask(
    mask: np.ndarray, erode: bool = True
) -> np.ndarray | None:
    """
    Extract 4 card corners from a binary mask.

    Optionally erodes the mask to tighten the boundary away from soft rembg
    edge pixels, then applies polygon approximation with a Hough
    line-intersection fallback.

    Args:
        mask: Binary uint8 mask (0/255), same spatial size as the source image.
        erode: If True, apply dilate x 1 + erode x 3 (net erosion) before contour
               extraction. Recommended for rembg masks which tend to be bloated.

    Returns:
        np.ndarray of shape (4, 1, 2), or None if 4 corners could not be found.
    """
    if erode:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True)

    if len(corners) == 4:
        return corners

    corners = find_corners_lines_intersect(contour)
    if corners is not None:
        return corners

    return None


def warp_card(
    img: np.ndarray,
    corners: np.ndarray,
    target_size: tuple[int, int] = (488, 680),
) -> np.ndarray:
    """Perspective-transform, orient to portrait, and resize to target_size."""
    warped = perspective_transform(img, corners)
    h, w = warped.shape[:2]
    if w > h:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(warped, target_size, interpolation=cv2.INTER_LANCZOS4)


def valid_aspect_ratio(corners: np.ndarray, tolerance: float = 0.2) -> bool:
    """Return True if the quad formed by corners has an MTG-card aspect ratio."""
    pts = corners.reshape(4, 2).astype(float)
    sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    pair_a = (sides[0] + sides[2]) / 2
    pair_b = (sides[1] + sides[3]) / 2
    if max(pair_a, pair_b) < 1e-6:
        return False
    ratio = min(pair_a, pair_b) / max(pair_a, pair_b)
    lo = CARD_ASPECT_RATIO * (1 - tolerance)
    hi = CARD_ASPECT_RATIO * (1 + tolerance)
    return bool(lo <= ratio <= hi)
