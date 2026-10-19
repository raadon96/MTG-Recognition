"""
Classical-CV card detector using box detection.

Used when YOLO returns 0 detections on images that have a uniform background
(e.g. scans with scanner-bed background).  The uniform background makes Otsu
thresholding extremely reliable, so this sub-millisecond path handles those
edge cases well.
"""

import cv2
import numpy as np

from .seg_utils import (
    CardSegment,
    corners_from_binary_mask,
    valid_aspect_ratio,
    warp_card,
)

_MIN_SOLIDITY = 0.85


def detect_cards_cv(
    img: np.ndarray,
    min_area_fraction: float = 0.01,
    max_area_fraction: float = 0.90,
    aspect_ratio_tolerance: float = 0.2,
    warp_img: np.ndarray | None = None,
) -> list[CardSegment]:
    """
    Detect card(s) in an image using classical CV thresholding.

    Designed for images with uniform backgrounds (scanner-bed scans).
    Uses Otsu thresholding, which works best when the histogram is bimodal
    (background colour vs. card content).

    Args:
        img: BGR input image (used for detection; may be preprocessed).
        min_area_fraction: Contours whose area is below this fraction of the
                           total image area are discarded as noise.
        max_area_fraction: Contours whose area exceeds this fraction are
                           discarded (avoids whole-image false positives).
        aspect_ratio_tolerance: Allowed relative deviation from MTG card ratio.
        warp_img: If provided, the perspective warp is computed from this image
                  instead of ``img``. Pass the original (unprocessed) image here
                  when ``img`` has been preprocessed for detection.

    Returns:
        List of CardSegment objects (may be empty).
    """
    warp_from = warp_img if warp_img is not None else img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    total_area = img.shape[0] * img.shape[1]
    min_area = min_area_fraction * total_area
    max_area = max_area_fraction * total_area
    segments: list[CardSegment] = []

    for candidate in (binary, cv2.bitwise_not(binary)):
        contours, _ = cv2.findContours(
            candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area < 1e-6:
                continue
            if area / hull_area < _MIN_SOLIDITY:
                continue

            blob_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)

            corners = corners_from_binary_mask(blob_mask, erode=False)
            if corners is None:
                continue

            if not valid_aspect_ratio(corners, aspect_ratio_tolerance):
                continue

            segments.append(CardSegment(warped_image=warp_card(warp_from, corners)))

        if segments:
            break  # don't try inverted binary if forward pass found cards

    return segments
