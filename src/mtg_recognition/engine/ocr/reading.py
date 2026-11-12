"""Reading a card: the raw text OCR finds in a card image's name crop and
collector information crop (see Reading in GLOSSARY.md).

Nothing is extracted from the text here; scoring looks for each candidate's
details in it (`engine/ocr/scoring.py`, ADR 0008).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from mtg_recognition.engine.ocr import pp_ocr

# OCR regions as fractions of the card image's height and width:
# (top, bottom, left, right)
# Name: horizontal text area at top of card
NAME_ROI = (0.04, 0.11, 0.06, 0.9)
# Collector information (CR 213): set code, collector number, illustration credit,
# copyright year; P/T sits at its right edge.
COLLECTOR_INFORMATION_ROI = (0.88, 1.0, 0.05, 0.95)

# PP-OCR recognition's native input height is 48 px; text below ~32 px reads worse.
# RATIO_TEXT_TO_CROP: about how much of the collector information crop's height a
# text line takes up.
TARGET_TEXT_HEIGHT = 48
RATIO_TEXT_TO_CROP = 0.15
MIN_UPSCALE = 1.0
MAX_UPSCALE = 4.0

# Text detection needs both sides >= 32 px, and misses the name's text below about
# 64 px; detection_input only scales down, so the crops are scaled up here.
_MIN_SIDE = 64


@dataclass(frozen=True)
class Reading:
    """The raw text of one card image's name crop and collector information crop,
    each crop's tokens in reading order with their confidences and boxes.
    """

    name: pp_ocr.TextLines
    collector_information: pp_ocr.TextLines
    # The collector information crop OCR read, upscaled (BGR), which its boxes are
    # in; kept only when read_card is called with debug=True. Left out of ==, which
    # an array can't answer with one bool.
    collector_information_crop: np.ndarray | None = field(default=None, compare=False)


def read_card(
    ocr: pp_ocr.PPOCR, card_image: np.ndarray, *, debug: bool = False
) -> Reading:
    """Read the name and the collector information off a card image.

    Args:
        ocr: The models from pp_ocr.load_pp_ocr.
        card_image: Card image (BGR format from OpenCV).
        debug: When True, the reading keeps the collector information crop.

    Raises:
        RuntimeError: If OCR fails.
    """
    # crop the name and the collector information
    name_crop = _crop(card_image, NAME_ROI)
    collector_information_crop = _crop(card_image, COLLECTOR_INFORMATION_ROI)

    # upscale the collector information so its small text reaches
    # TARGET_TEXT_HEIGHT; the name's text is already large enough
    h, w = collector_information_crop.shape[:2]
    scale = TARGET_TEXT_HEIGHT / (h * RATIO_TEXT_TO_CROP)
    scale = max(MIN_UPSCALE, min(scale, MAX_UPSCALE))
    collector_information_crop = cv2.resize(
        collector_information_crop,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )

    # make both crops large enough for text detection
    name_crop = _ensure_min_dim(name_crop, _MIN_SIDE)
    collector_information_crop = _ensure_min_dim(collector_information_crop, _MIN_SIDE)

    # read both crops; onnxruntime's errors don't derive from RuntimeError
    try:
        name_lines = pp_ocr.read_text(ocr, name_crop)
        collector_information_lines = pp_ocr.read_text(ocr, collector_information_crop)
    except Exception as e:
        raise RuntimeError(f"{type(e).__name__}: {e}") from e

    return Reading(
        name=name_lines,
        collector_information=collector_information_lines,
        collector_information_crop=collector_information_crop if debug else None,
    )


def _crop(card_image: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    """The pixels of card_image inside roi, given as (top, bottom, left, right)
    fractions of its height and width.
    """
    top, bottom, left, right = roi
    h, w = card_image.shape[:2]
    return card_image[int(h * top) : int(h * bottom), int(w * left) : int(w * right)]


def _ensure_min_dim(img: np.ndarray, min_dim: int) -> np.ndarray:
    """Upscale img so both dimensions are >= min_dim (aspect ratio preserved)."""
    h, w = img.shape[:2]
    if h >= min_dim and w >= min_dim:
        return img
    scale = max(min_dim / h, min_dim / w)
    new_h = max(min_dim, round(h * scale))
    new_w = max(min_dim, round(w * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
