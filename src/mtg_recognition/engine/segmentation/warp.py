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

    return upright_card_image(card_image)


def _side_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum()))


def upright_card_image(card_image: np.ndarray) -> np.ndarray:
    """A card image turned upright, its size clamped between CARD_IMAGE_MIN_SIZE
    and CARD_IMAGE_MAX_SIZE, each axis on its own.
    """
    if card_image.shape[1] > card_image.shape[0]:
        card_image = cv2.rotate(card_image, cv2.ROTATE_90_CLOCKWISE)
    h, w = card_image.shape[:2]
    out_w = min(max(w, CARD_IMAGE_MIN_SIZE[0]), CARD_IMAGE_MAX_SIZE[0])
    out_h = min(max(h, CARD_IMAGE_MIN_SIZE[1]), CARD_IMAGE_MAX_SIZE[1])
    if (out_w, out_h) == (w, h):
        return card_image
    return cv2.resize(card_image, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
