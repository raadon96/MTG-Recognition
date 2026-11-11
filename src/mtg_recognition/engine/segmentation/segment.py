"""Segmentation: each card in a query image, as a card image.

The perspective warp is always computed from the query image itself; the
contrast-enhanced copy is only for finding the cards.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from mtg_recognition.settings import settings

from .corners import (
    CARD_ASPECT_RATIO,
    card_corners_from_outline,
    corners_from_binary_mask,
    valid_aspect_ratio,
)
from .warp import upright_card_image, warp_card
from .yolo import load_yolo_session, yolo_outlines

if TYPE_CHECKING:
    import onnxruntime as ort
    from rembg.sessions.base import BaseSession

SegmentationMode = Literal["auto", "yolo", "rembg"]

# Allow ±3 % from exact ratio for the scan bypass
_BYPASS_RATIO_TOLERANCE = 0.03
_BYPASS_LO = CARD_ASPECT_RATIO * (1 - _BYPASS_RATIO_TOLERANCE)  # ≈ 0.694
_BYPASS_HI = CARD_ASPECT_RATIO * (1 + _BYPASS_RATIO_TOLERANCE)  # ≈ 0.737
# Outer border strip width as a fraction of the smaller image dimension
_BYPASS_BORDER_PCT = 0.05
# Grayscale mean threshold for the border strip:
# scans cropped to the card have a dark card border (mean ≈ 23–52);
# scans with the scanner bed have a bright border (mean ≈ 148–240).
# A threshold of 100 sits in the middle of the 96-point gap between groups.
_BYPASS_BORDER_MAX_MEAN = 100

# "silueta" separates cards from the background best of rembg's models
_REMBG_MODEL = "silueta"

# Blobs and contours below this fraction of the query image area are noise
_MIN_AREA_FRACTION = 0.01
# Box contours above this fraction are the whole image, not a card
_BOX_MAX_AREA_FRACTION = 0.90
_BOX_MIN_SOLIDITY = 0.85

# enhance_contrast brightness thresholds (L-channel mean, 0–255)
_L_VERY_DARK = 50
_L_DARK = 100
_L_VERY_BRIGHT = 200
_L_GAMMA_BRIGHT = 180

# CLAHE parameters per brightness band: (clip_limit, tile_grid_size)
_CLAHE_VERY_DARK = (4.0, 4)
_CLAHE_DARK = (3.5, 6)
_CLAHE_VERY_BRIGHT = (2.5, 8)
_CLAHE_NORMAL = (3.0, 6)

# Gamma correction per brightness band
_GAMMA_DARK = 1.15
_GAMMA_BRIGHT = 0.95
_GAMMA_NORMAL = 1.05


def segment(
    query_image: np.ndarray, mode: SegmentationMode = "auto"
) -> list[np.ndarray]:
    """The card image of each card in a BGR query image; empty if none is found.

    "auto" takes the scan bypass, then YOLO, then, when YOLO finds no card, the
    box fallback and then rembg. "yolo" stops after YOLO; "rembg" runs only
    rembg. Query images are segmented one at a time: YOLO batch inference
    changes results for mixed-size batches (docs/adr/0002).
    """
    if mode == "rembg":
        return rembg_card_images(query_image)

    # Scan bypass: the query image is already the card image
    if takes_scan_bypass(query_image):
        return [upright_card_image(query_image)]

    # YOLO
    enhanced_image = enhance_contrast(query_image)
    card_images = []
    for outline in yolo_outlines(_yolo_session(), enhanced_image):
        corners = card_corners_from_outline(query_image, outline)
        if corners is not None:
            card_images.append(warp_card(query_image, corners))

    match mode:
        case "yolo":
            return card_images
        case "auto":
            if card_images:
                return card_images
            # Fallbacks for the rare query image YOLO finds no card in: the box
            # fallback is fast and reliable on plain backgrounds; rembg works
            # well on photos but is slow to load and run, so it comes last
            return box_card_images(enhanced_image, query_image) or rembg_card_images(
                query_image
            )


def takes_scan_bypass(query_image: np.ndarray) -> bool:
    """Whether a query image is already a card image: card ratio, dark border."""
    h, w = query_image.shape[:2]
    ratio = min(w, h) / max(w, h)
    if not (_BYPASS_LO <= ratio <= _BYPASS_HI):
        return False
    # Secondary check: the outer border must be dark (card's black border).
    # Scanner-bed images have a bright border; card-only scans do not.
    bh = max(1, int(h * _BYPASS_BORDER_PCT))
    bw = max(1, int(w * _BYPASS_BORDER_PCT))
    gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    border = np.concatenate(
        [
            gray[:bh, :].ravel(),
            gray[-bh:, :].ravel(),
            gray[:, :bw].ravel(),
            gray[:, -bw:].ravel(),
        ]
    )
    return bool(np.mean(border) < _BYPASS_BORDER_MAX_MEAN)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Enhance contrast and brightness for YOLO inference and the box fallback.

    Applies adaptive CLAHE on the L channel (LAB) followed by gamma correction.
    Dimensions are preserved — the returned image is the same size as the input.

    Args:
        img: BGR image (uint8).

    Returns:
        Enhanced BGR image (uint8), same dimensions as input.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    l_mean = float(np.mean(l_channel))
    if l_mean < _L_VERY_DARK:
        clip, tile = _CLAHE_VERY_DARK
    elif l_mean < _L_DARK:
        clip, tile = _CLAHE_DARK
    elif l_mean > _L_VERY_BRIGHT:
        clip, tile = _CLAHE_VERY_BRIGHT
    else:
        clip, tile = _CLAHE_NORMAL

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l_enhanced = clahe.apply(l_channel)

    gamma = (
        _GAMMA_DARK
        if l_mean < _L_DARK
        else (_GAMMA_BRIGHT if l_mean > _L_GAMMA_BRIGHT else _GAMMA_NORMAL)
    )
    l_gamma = np.power(l_enhanced / 255.0, gamma) * 255.0
    l_gamma = np.clip(l_gamma, 0, 255).astype(np.uint8)

    merged = cv2.merge([l_gamma, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


@cache
def _yolo_session() -> ort.InferenceSession:
    return load_yolo_session(str(settings.YOLO_SEG_MODEL_PATH), settings.THREAD_COUNT)


def box_card_images(
    enhanced_image: np.ndarray, query_image: np.ndarray
) -> list[np.ndarray]:
    """The card images found by thresholding the contrast-enhanced query image.

    Made for plain backgrounds, such as a scanner bed, where the histogram is
    bimodal (background vs. card) and Otsu thresholding separates them reliably.
    The card images are warped from query_image.
    """
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    total_area = enhanced_image.shape[0] * enhanced_image.shape[1]
    min_area = _MIN_AREA_FRACTION * total_area
    max_area = _BOX_MAX_AREA_FRACTION * total_area

    # cards darker than the background, then lighter
    for candidate in (binary, cv2.bitwise_not(binary)):
        card_images = []
        contours, _ = cv2.findContours(
            candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if hull_area < 1e-6 or area / hull_area < _BOX_MIN_SOLIDITY:
                continue

            blob_mask = np.zeros(enhanced_image.shape[:2], dtype=np.uint8)
            cv2.drawContours(blob_mask, [contour], -1, (255,), thickness=cv2.FILLED)
            corners = corners_from_binary_mask(blob_mask)
            if corners is None or not valid_aspect_ratio(corners):
                continue
            card_images.append(warp_card(query_image, corners))

        if card_images:
            return card_images
    return []


def rembg_card_images(query_image: np.ndarray) -> list[np.ndarray]:
    """The card images found by rembg background removal: one per foreground blob
    that is large enough and card-shaped.
    """
    from rembg import remove

    mask = remove(
        query_image,
        session=_rembg_session(),
        only_mask=True,
        post_process_mask=True,
        alpha_matting=False,
    )
    _, binary = cv2.threshold(np.array(mask), 127, 255, cv2.THRESH_BINARY)

    min_area = _MIN_AREA_FRACTION * query_image.shape[0] * query_image.shape[1]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    card_images = []
    for label in range(1, num_labels):  # label 0 is background
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            continue
        blob_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        corners = rembg_blob_corners(blob_mask)
        if corners is None or not valid_aspect_ratio(corners):
            continue
        card_images.append(warp_card(query_image, corners))
    return card_images


@cache
def _rembg_session() -> BaseSession:
    from rembg import new_session

    return new_session(_REMBG_MODEL)


def rembg_blob_corners(blob_mask: np.ndarray) -> np.ndarray | None:
    """The four corners of one rembg foreground blob, as corners_from_binary_mask."""
    # rembg masks spill past the card's edges, so shrink the blob first
    # (dilate x 1 + erode x 3, a net erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(blob_mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=3)
    return corners_from_binary_mask(mask)
