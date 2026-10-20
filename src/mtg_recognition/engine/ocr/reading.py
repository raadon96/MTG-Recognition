"""Reading a card: what PaddleOCR finds printed on a warped card image (see Reading
in GLOSSARY.md).

The reading reports what the card shows; comparing it with printings is scoring's
job (`engine/ocr/scoring.py`).
"""

from dataclasses import dataclass, field

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

from mtg_recognition.core.image_utils import _ensure_min_dim
from mtg_recognition.engine.ocr import ocr_extraction
from mtg_recognition.engine.ocr.enhance import upscale_for_ocr

# Regions as fractions of image dimensions (H x W)
# Name: horizontal text area at top of card
NAME_TOP = 0.04
NAME_BOTTOM = 0.11
NAME_LEFT = 0.06
NAME_RIGHT = 0.9

# Information below the text box (CR 213): set code, collector number,
# illustration credit, copyright year; P/T sits at its right edge.
BELOW_TEXT_BOX_TOP = 0.88
BELOW_TEXT_BOX_BOTTOM = 1.0
BELOW_TEXT_BOX_LEFT = 0.05
BELOW_TEXT_BOX_RIGHT = 0.95

# for flavor text bleedthrough filtering below the text box; the y threshold is kept
# small so artist names near the top of the region aren't discarded
Y_TRESH_FRAC = 0.08
X_RIGHT_FRAC = 0.90

# PaddleOCR's text detection model requires both dimensions >= 32 px;
# upscale any region that falls below this threshold before inference.
_MIN_DIM = 32


@dataclass(frozen=True)
class ReadingDebug:
    """Reader artefacts for mismatch root-cause investigation, attached to a Reading
    only when the reader is called with debug=True.
    """

    # Annotated crop of the information below the text box from PaddleOCR
    # (side-by-side: original + text labels). None if PaddleOCR doesn't provide it.
    annotated_image: Image.Image | None
    # Raw OCR tokens below the text box (flavor-text filtered), with their
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
    set_code_candidates: list[str] = field(default_factory=list)
    number: str | None = None
    base_set_size: str | None = None
    artist: str | None = None
    set_release_year: str | None = None
    power: str | None = None
    toughness: str | None = None
    debug: ReadingDebug | None = None


class PaddleCardReader:
    """Reads a warped card image with PaddleOCR."""

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
            text_detection_model_dir="src/mtg_recognition/engine/ocr/models/prod/PP-OCRv5_mobile_det",
            # text recognition
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            text_recognition_model_dir="src/mtg_recognition/engine/ocr/models/prod/PP-OCRv5_mobile_rec",
            # other params
            device="cpu",
            enable_hpi=False,  # enable high-performance inference
            precision="fp16",  # set to "fp16" for acceleration
            lang="en",
        )

    def read(self, card_image: np.ndarray, *, debug: bool = False) -> Reading:
        """Read the name and the information below the text box off a card image.

        Args:
            card_image: Warped card image (BGR format from OpenCV).
            debug: When True, the reading carries a ReadingDebug.
        """
        h, w, _ = card_image.shape

        name_img = card_image[
            int(h * NAME_TOP) : int(h * NAME_BOTTOM),
            int(w * NAME_LEFT) : int(w * NAME_RIGHT),
        ]
        below_text_box_img = card_image[
            int(h * BELOW_TEXT_BOX_TOP) : int(h * BELOW_TEXT_BOX_BOTTOM),
            int(w * BELOW_TEXT_BOX_LEFT) : int(w * BELOW_TEXT_BOX_RIGHT),
        ]

        below_text_box_img = upscale_for_ocr(below_text_box_img)

        # Perform OCR on both regions in a single batched predict call.
        # Paddle's text-detection model requires both dimensions >= 32 px; upscale
        # any region that falls below this threshold before inference.

        name_img = _ensure_min_dim(name_img, _MIN_DIM)
        below_text_box_img = _ensure_min_dim(below_text_box_img, _MIN_DIM)

        texts_name: list[str] = []
        texts: list[str] = []
        boxes: list[tuple[int, int, int, int]] = []
        scores: list[float] = []
        raw_results = None

        try:
            raw_results = self._paddleocr.predict(
                [name_img, below_text_box_img], text_det_thresh=0.05
            )
            texts_name = raw_results[0]["rec_texts"]
            texts = raw_results[1]["rec_texts"]
            boxes = raw_results[1]["rec_boxes"].tolist()
            scores = raw_results[1]["rec_scores"]
        except RuntimeError as e:
            print(f"OCR prediction failed: {e}\nProceeding with empty OCR results...")
            pass  # leave all text lists empty; the reading will be empty

        # Remove tokens in the flavor-text bleedthrough zone before extraction.
        # Tokens near the TOP of the region and not in the far-right column
        # (where P/T lives) are flavor text that bled into it.
        texts, boxes, scores = ocr_extraction.filter_flavor_text_zone(
            texts,
            boxes,
            scores,
            y_thresh_frac=Y_TRESH_FRAC,
            x_right_frac=X_RIGHT_FRAC,
        )

        number, base_set_size = ocr_extraction.extract_number_and_base_set_size(texts)
        power, toughness = ocr_extraction.extract_power_toughness(texts)

        return Reading(
            name=ocr_extraction.extract_card_name(texts_name),
            set_code_candidates=ocr_extraction.extract_setcode_candidates(texts),
            number=number,
            base_set_size=base_set_size,
            artist=ocr_extraction.extract_artist_name(texts),
            set_release_year=ocr_extraction.extract_set_release_year(texts),
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


def _annotated_image(raw_results) -> Image.Image | None:
    """PaddleOCR's annotated crop of the information below the text box, if any."""
    if raw_results is None:
        return None
    try:
        return raw_results[1].img["ocr_res_img"]
    except Exception as e:
        print(
            f"Failed to extract annotated image from OCR result: {e}\n"
            "Continuing without annotated image..."
        )
        return None
