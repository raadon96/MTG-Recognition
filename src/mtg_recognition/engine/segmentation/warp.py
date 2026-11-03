"""Straightening a card image out of the query image."""

import cv2
import numpy as np

# Canonical minimum output size for a perspective-corrected card (width × height),
# consistent with the Scryfall "normal" card size (488 × 680) and the precomputed
# pHash index.
CARD_IMAGE_MIN_SIZE: tuple[int, int] = (488, 680)

# Maximum output size for a card image. Natural warp resolution is clamped to
# this ceiling so that close-up phone photos don't produce unbounded images.
# 2× CARD_IMAGE_MIN_SIZE keeps per-card memory and temp-file sizes predictable.
CARD_IMAGE_MAX_SIZE: tuple[int, int] = (976, 1360)


def warp_card(query_image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """The card image inside four corners of the query image: straightened,
    upright, and at its natural size clamped between CARD_IMAGE_MIN_SIZE and
    CARD_IMAGE_MAX_SIZE.

    Args:
        query_image: BGR query image.
        corners: Integer corners in any order, shape (4, 1, 2).
    """
    # Order the corners: the top-left has the smallest x + y, the bottom-right
    # the largest; the top-right has the smallest y - x, the bottom-left the
    # largest
    points = corners.reshape(4, 2).astype(np.intp)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    tl, tr, br, bl = (
        points[np.argmin(sums)],
        points[np.argmin(diffs)],
        points[np.argmax(sums)],
        points[np.argmax(diffs)],
    )

    # Natural size: the longer side of each pair of opposite sides, truncated
    width = max(int(_side_length(br, bl)), int(_side_length(tr, tl)))
    height = max(int(_side_length(tr, br)), int(_side_length(tl, bl)))

    # Straighten
    source = np.array([tl, tr, br, bl], dtype=np.float32)
    destination = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    card_image = cv2.warpPerspective(query_image, transform, (width, height))

    # Turn upright and clamp the size
    h, w = card_image.shape[:2]
    if w > h:
        card_image = cv2.rotate(card_image, cv2.ROTATE_90_CLOCKWISE)
        h, w = w, h

    out_w, out_h = clamp_size((w, h), CARD_IMAGE_MIN_SIZE, CARD_IMAGE_MAX_SIZE)
    if (out_w, out_h) == (w, h):
        return card_image
    return cv2.resize(card_image, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


def _side_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum()))


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
