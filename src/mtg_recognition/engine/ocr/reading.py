"""Reading a card: what PaddleOCR finds printed on a card image (see Reading in
GLOSSARY.md).

The reading reports what the card shows; comparing it with printings is scoring's
job (`engine/ocr/scoring.py`).
"""

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from mtg_recognition.engine.ocr import ocr_extraction, onnx_ocr
from mtg_recognition.settings import settings

logger = logging.getLogger(__name__)

# OCR regions as fractions of the card image's height and width:
# (top, bottom, left, right)
# Name: horizontal text area at top of card
NAME_ROI = (0.04, 0.11, 0.06, 0.9)
# Collector information (CR 213): set code, collector number, illustration credit,
# copyright year; P/T sits at its right edge.
COLLECTOR_INFORMATION_ROI = (0.88, 1.0, 0.05, 0.95)

# for flavor text bleedthrough filtering in the collector information crop; the y
# threshold is kept small so artist names near the top of the crop aren't discarded
Y_TRESH_FRAC = 0.08
X_RIGHT_FRAC = 0.90

# PP-OCRv5_mobile_rec's native input height is 48 px; text below ~32 px reads worse.
# RATIO_TEXT_TO_CROP: about how much of the collector information crop's height a
# text line takes up.
TARGET_TEXT_HEIGHT = 48
RATIO_TEXT_TO_CROP = 0.15
MIN_UPSCALE = 1.0
MAX_UPSCALE = 4.0

# PaddleOCR's text detection model requires both dimensions >= 32 px.
_MIN_DIM = 32


@dataclass(frozen=True)
class ReadingDebug:
    """OCR artefacts from read_card for mismatch root-cause investigation, attached
    to a Reading only when read_card is called with debug=True.
    """

    # Annotated collector information crop from PaddleOCR
    # (side-by-side: original + text labels). None if PaddleOCR doesn't provide it.
    annotated_image: Image.Image | None
    # Raw OCR tokens of the collector information (flavor-text filtered), with their
    # confidences and boxes.
    texts: list[str]
    scores: list[float]
    boxes: list[tuple[int, int, int, int]]


@dataclass(frozen=True)
class Reading:
    """The details read off one card's image (see GLOSSARY.md), named after the
    Printing fields they are compared with. None when nothing was read.
    """

    name: str | None = None
    # Every token that may be the set code: OCR can't tell which one it is.
    possible_set_codes: list[str] = field(default_factory=list)
    collector_number: str | None = None
    base_set_size: str | None = None
    artist: str | None = None
    copyright_year: str | None = None
    power: str | None = None
    toughness: str | None = None
    debug: ReadingDebug | None = None


def load_paddleocr() -> onnx_ocr.OwnOCR:
    """The PP-OCR ONNX exports that read card images, on onnxruntime."""
    return onnx_ocr.load(
        settings.OCR_DETECTION_MODEL_DIR, settings.OCR_RECOGNITION_MODEL_DIR, threads=4
    )


def read_card(
    paddleocr: onnx_ocr.OwnOCR, card_image: np.ndarray, *, debug: bool = False
) -> Reading:
    """Read the name and the collector information off a card image.

    Args:
        paddleocr: The model from load_paddleocr.
        card_image: Card image (BGR format from OpenCV).
        debug: When True, the reading holds a ReadingDebug.

    Raises:
        RuntimeError: If PaddleOCR fails.
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

    # make both crops large enough for PaddleOCR's text detection
    name_crop = _ensure_min_dim(name_crop, _MIN_DIM)
    collector_information_crop = _ensure_min_dim(collector_information_crop, _MIN_DIM)

    # PaddleX's OCR pipeline defaults, with today's detection threshold
    name_lines = onnx_ocr.read_text(paddleocr, name_crop, 0.05, 0.6, 1.5)
    lines = onnx_ocr.read_text(paddleocr, collector_information_crop, 0.05, 0.6, 1.5)
    texts_name: list[str] = name_lines.texts
    texts: list[str] = lines.texts
    boxes: list[tuple[int, int, int, int]] = lines.boxes.tolist()
    scores: list[float] = lines.scores

    # drop flavor-text tokens: tokens near the top of the crop and not in the
    # far-right column (where P/T is) are flavor text that bled into it
    texts, boxes, scores = ocr_extraction.filter_flavor_text_zone(
        texts,
        boxes,
        scores,
        y_thresh_frac=Y_TRESH_FRAC,
        x_right_frac=X_RIGHT_FRAC,
    )

    # extract the fields
    collector_number, base_set_size = (
        ocr_extraction.extract_collector_number_and_base_set_size(texts)
    )
    power, toughness = ocr_extraction.extract_power_toughness(texts)

    return Reading(
        name=ocr_extraction.extract_card_name(texts_name),
        possible_set_codes=ocr_extraction.extract_possible_set_codes(texts),
        collector_number=collector_number,
        base_set_size=base_set_size,
        artist=ocr_extraction.extract_artist_name(texts),
        copyright_year=ocr_extraction.extract_copyright_year(texts),
        power=power,
        toughness=toughness,
        debug=(
            ReadingDebug(
                annotated_image=None,
                texts=texts,
                scores=scores,
                boxes=boxes,
            )
            if debug
            else None
        ),
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
