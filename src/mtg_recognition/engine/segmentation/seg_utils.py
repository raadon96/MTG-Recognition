"""Shared geometric helpers and result types for card segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import find_corners_lines_intersect, perspective_transform

# MTG standard card dimensions: 63 mm × 88 mm (portrait)
CARD_ASPECT_RATIO = 63.0 / 88.0  # ≈ 0.716

# Canonical minimum output size for a perspective-corrected card (width × height),
# consistent with the Scryfall "normal" card size (488 × 680) and the precomputed
# pHash index.
CARD_IMAGE_MIN_SIZE: tuple[int, int] = (488, 680)

# Maximum output size for a card image. Natural warp resolution is clamped to
# this ceiling so that close-up phone photos don't produce unbounded images.
# 2× CARD_IMAGE_MIN_SIZE keeps per-card memory and temp-file sizes predictable.
CARD_IMAGE_MAX_SIZE: tuple[int, int] = (976, 1360)


@dataclass
class SegmentedCard:
    """One card found by segmentation: its card image."""

    card_image: np.ndarray  # BGR, resized to target_size


@dataclass
class SegmentationResult:
    """Segmentation result for one query image. May contain multiple cards."""

    cards: list[SegmentedCard] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.cards)


def corners_from_binary_mask(mask: np.ndarray, erode: bool = True) -> np.ndarray | None:
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


def clamp_size(
    size: tuple[int, int],
    min_size: tuple[int, int],
    max_size: tuple[int, int],
) -> tuple[int, int]:
    """Clamp a (width, height) size to be within the given min and max bounds."""
    w, h = size
    min_w, min_h = min_size
    max_w, max_h = max_size
    out_w = min(max(w, min_w), max_w)
    out_h = min(max(h, min_h), max_h)
    return out_w, out_h


def warp_card(
    img: np.ndarray,
    corners: np.ndarray,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Perspective-transform, orient to portrait, and size the output.

    If target_size is None (default), the output resolution is the natural size
    derived from the corner distances (as computed by perspective_transform),
    clamped between CARD_IMAGE_MIN_SIZE (minimum) and CARD_IMAGE_MAX_SIZE (maximum).
    Pass an explicit target_size to force a specific output resolution.
    """
    card_image = perspective_transform(img, corners)
    h, w = card_image.shape[:2]
    if w > h:
        card_image = cv2.rotate(card_image, cv2.ROTATE_90_CLOCKWISE)
        h, w = w, h

    if target_size is None:
        out_w, out_h = clamp_size((w, h), CARD_IMAGE_MIN_SIZE, CARD_IMAGE_MAX_SIZE)
        if (out_w, out_h) == (w, h):
            return card_image
        return cv2.resize(card_image, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    return cv2.resize(card_image, target_size, interpolation=cv2.INTER_LANCZOS4)


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
