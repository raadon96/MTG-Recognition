"""Reading a card: what OCR finds printed on a warped card image (see Reading in
GLOSSARY.md). PaddleOcr is the OCR step used in production.

The reading reports what the card shows; comparing it with printings is scoring's
job (`engine/ocr/scoring.py`).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

from mtg_recognition.engine.ocr import ocr_extraction

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

# Adaptive upscaling constants for the information below the text box.
# PP-OCRv5_mobile_rec native input height is 48 px; text below ~32 px degrades.
# RATIO_TEXT_TO_CROP: approximate fraction of the crop height occupied by text lines.
TARGET_TEXT_HEIGHT: int = 48
RATIO_TEXT_TO_CROP: float = 0.15
MIN_UPSCALE: float = 1.0
MAX_UPSCALE: float = 4.0


@dataclass(frozen=True)
class ReadingDebug:
    """Reader artefacts for mismatch root-cause investigation, attached to a Reading
    only when the reader is called with debug=True.
    """

    # Annotated crop of the information below the text box from the OCR step
    # (side-by-side: original + text labels). None if the OCR step doesn't provide it.
    annotated_image: Image.Image | None
    # Raw OCR tokens below the text box (flavor-text filtered), with their
    # confidences and boxes (in the pixels of the crop).
    texts: list[str]
    scores: list[float]
    boxes: list[tuple[float, float, float, float]]


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


@dataclass(frozen=True)
class OcrResult:
    """What the OCR step finds in one crop."""

    texts: list[str]
    # [x1, y1, x2, y2] in the pixels of the crop the OCR step was given
    boxes: list[tuple[float, float, float, float]]
    scores: list[float]
    annotated_image: Image.Image | None = None


# The OCR step: reads the name crop then the crop below the text box, one result
# per crop, in order.
Ocr = Callable[[list[np.ndarray]], list[OcrResult]]


class CardReader:
    """Reads a warped card image: crops the name and the information below the text
    box, OCRs them and extracts the Reading's fields.

    Args:
        ocr: The OCR step (default: PaddleOcr).
    """

    def __init__(self, ocr: Ocr | None = None):
        self._ocr: Ocr = ocr if ocr is not None else PaddleOcr()

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

        name, below_text_box = self._ocr([name_img, below_text_box_img])

        # Remove tokens in the flavor-text bleedthrough zone before extraction.
        # Tokens near the TOP of the region and not in the far-right column
        # (where P/T lives) are flavor text that bled into it.
        texts, boxes, scores = ocr_extraction.filter_flavor_text_zone(
            below_text_box.texts,
            below_text_box.boxes,
            below_text_box.scores,
            y_thresh_frac=Y_TRESH_FRAC,
            x_right_frac=X_RIGHT_FRAC,
        )

        number, base_set_size = ocr_extraction.extract_number_and_base_set_size(texts)
        power, toughness = ocr_extraction.extract_power_toughness(texts)

        return Reading(
            name=ocr_extraction.extract_card_name(name.texts),
            set_code_candidates=ocr_extraction.extract_setcode_candidates(texts),
            number=number,
            base_set_size=base_set_size,
            artist=ocr_extraction.extract_artist_name(texts),
            set_release_year=ocr_extraction.extract_set_release_year(texts),
            power=power,
            toughness=toughness,
            debug=(
                ReadingDebug(
                    annotated_image=below_text_box.annotated_image,
                    texts=texts,
                    scores=scores,
                    boxes=boxes,
                )
                if debug
                else None
            ),
        )


class PaddleOcr:
    """The OCR step run with PaddleOCR.

    Upscales the crop below the text box (its text is small), pads both crops to
    PaddleOCR's minimum size, runs one predict on both and maps the boxes back to
    the pixels of the crops it was given. A failed predict gives empty results, so
    the image keeps its pHash ranking.

    Args:
        paddleocr: Anything with PaddleOCR's predict (default: PaddleOCR built
            with the production models).
    """

    def __init__(self, paddleocr=None):
        self._paddleocr = paddleocr if paddleocr is not None else _build_paddleocr()

    def __call__(self, crops: list[np.ndarray]) -> list[OcrResult]:
        name_img, below_text_box_img = crops
        prepared = [
            _ensure_min_dim(name_img, _MIN_DIM),
            _ensure_min_dim(upscale_for_ocr(below_text_box_img), _MIN_DIM),
        ]

        try:
            raw_results = self._paddleocr.predict(prepared, text_det_thresh=0.05)
        except RuntimeError as e:
            print(f"OCR prediction failed: {e}\nProceeding with empty OCR results...")
            return [OcrResult([], [], []) for _ in crops]

        return [
            OcrResult(
                texts=list(raw["rec_texts"]),
                boxes=_to_crop_pixels(
                    np.asarray(raw["rec_boxes"]).tolist(), crop.shape, sent.shape
                ),
                scores=list(raw["rec_scores"]),
                annotated_image=_annotated_image(raw),
            )
            for raw, crop, sent in zip(raw_results, crops, prepared, strict=True)
        ]


def _build_paddleocr() -> PaddleOCR:
    return PaddleOCR(
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


def upscale_for_ocr(img: np.ndarray) -> np.ndarray:
    """Adaptively upscale a card ROI crop so its text reaches TARGET_TEXT_HEIGHT px.

    The scale factor is clamped to [MIN_UPSCALE, MAX_UPSCALE]. Meant for the info
    ROI (small text); the name ROI's text is already large enough.

    Args:
        img: BGR image (numpy array)

    Returns:
        Upscaled BGR image
    """
    h, w = img.shape[:2]
    s = TARGET_TEXT_HEIGHT / (h * RATIO_TEXT_TO_CROP)
    s = max(MIN_UPSCALE, min(s, MAX_UPSCALE))
    new_w, new_h = int(round(w * s)), int(round(h * s))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _ensure_min_dim(img: np.ndarray, min_dim: int) -> np.ndarray:
    """Upscale img so both dimensions are >= min_dim (aspect ratio preserved)."""
    h, w = img.shape[:2]
    if h >= min_dim and w >= min_dim:
        return img
    scale = max(min_dim / h, min_dim / w)
    new_h = max(min_dim, round(h * scale))
    new_w = max(min_dim, round(w * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def _to_crop_pixels(
    boxes: list[list[int]], crop_shape: tuple[int, ...], sent_shape: tuple[int, ...]
) -> list[tuple[float, float, float, float]]:
    """Map boxes from the pixels of the image sent to PaddleOCR back to the crop's.

    Not rounded: the flavor filter compares box edges with fractions of the
    region, and rounding each edge would move tokens across its thresholds.
    """
    sx = crop_shape[1] / sent_shape[1]
    sy = crop_shape[0] / sent_shape[0]
    return [(x1 * sx, y1 * sy, x2 * sx, y2 * sy) for x1, y1, x2, y2 in boxes]


def _annotated_image(raw_result) -> Image.Image | None:
    """PaddleOCR's annotated image of one crop, if any."""
    try:
        return raw_result.img["ocr_res_img"]
    except Exception as e:
        print(
            f"Failed to extract annotated image from OCR result: {e}\n"
            "Continuing without annotated image..."
        )
        return None
