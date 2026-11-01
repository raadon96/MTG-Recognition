"""Reading a card: what PaddleOCR finds printed on a card image (see Reading in
GLOSSARY.md).

The reading reports what the card shows; comparing it with printings is scoring's
job (`engine/ocr/scoring.py`).
"""

from dataclasses import dataclass, field

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

from mtg_recognition.engine.image import _ensure_min_dim
from mtg_recognition.engine.ocr import ocr_extraction
from mtg_recognition.engine.ocr.enhance import upscale_for_ocr
from mtg_recognition.settings import settings

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

# PaddleOCR's text detection model requires both dimensions >= 32 px;
# upscale any crop that falls below this threshold before inference.
_MIN_DIM = 32


@dataclass(frozen=True)
class ReadingDebug:
    """Reader artefacts for mismatch root-cause investigation, attached to a Reading
    only when the reader is called with debug=True.
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


class PaddleCardReader:
    """Reads a card image with PaddleOCR."""

    def __init__(self):
        self._paddleocr = PaddleOCR(
            # text image rectification (performs geometric transformations on images
            # to correct distortions, inclinations, perspective deformations)
            use_doc_unwarping=False,
            doc_unwarping_model_name=None,
            # textline orientation classification
            # (corrects the orientation of text lines)
            use_textline_orientation=False,
            # text detection (locating and marking text-containing regions in images)
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_detection_model_dir=str(settings.OCR_DETECTION_MODEL_DIR),
            # text recognition
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            text_recognition_model_dir=str(settings.OCR_RECOGNITION_MODEL_DIR),
            # other params
            device="cpu",
            enable_hpi=False,  # enable high-performance inference
            precision="fp16",  # set to "fp16" for acceleration
            lang="en",
        )

    def read(self, card_image: np.ndarray, *, debug: bool = False) -> Reading:
        """Read the name and the collector information off a card image.

        Args:
            card_image: Card image (BGR format from OpenCV).
            debug: When True, the reading holds a ReadingDebug.

        Raises:
            RuntimeError: If PaddleOCR fails.
        """
        name_crop = _crop(card_image, NAME_ROI)
        collector_information_crop = upscale_for_ocr(
            _crop(card_image, COLLECTOR_INFORMATION_ROI)
        )

        # Perform OCR on both crops in a single batched predict call.
        # Paddle's text-detection model requires both dimensions >= 32 px; upscale
        # any crop that falls below this threshold before inference.

        name_crop = _ensure_min_dim(name_crop, _MIN_DIM)
        collector_information_crop = _ensure_min_dim(
            collector_information_crop, _MIN_DIM
        )

        raw_results = self._paddleocr.predict(
            [name_crop, collector_information_crop], text_det_thresh=0.05
        )
        texts_name: list[str] = raw_results[0]["rec_texts"]
        texts: list[str] = raw_results[1]["rec_texts"]
        boxes: list[tuple[int, int, int, int]] = raw_results[1]["rec_boxes"].tolist()
        scores: list[float] = raw_results[1]["rec_scores"]

        # Remove tokens in the flavor-text bleedthrough zone before extraction.
        # Tokens near the TOP of the crop and not in the far-right column
        # (where P/T lives) are flavor text that bled into it.
        texts, boxes, scores = ocr_extraction.filter_flavor_text_zone(
            texts,
            boxes,
            scores,
            y_thresh_frac=Y_TRESH_FRAC,
            x_right_frac=X_RIGHT_FRAC,
        )

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
                    annotated_image=_annotated_image(raw_results),
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


def _annotated_image(raw_results) -> Image.Image | None:
    """PaddleOCR's annotated collector information crop, if any."""
    try:
        return raw_results[1].img["ocr_res_img"]
    except Exception as e:
        print(
            f"Failed to extract annotated image from OCR result: {e}\n"
            "Continuing without annotated image..."
        )
        return None
