import os
from pathlib import Path
from typing import NamedTuple

import click
import cv2
import numpy as np
import rapidfuzz
from dotenv import load_dotenv
from icecream import ic
from paddlex.inference import load_pipeline_config
from sqlalchemy.engine.row import Row

from mtg_recognition.engine.ocr import ocr_utils

load_dotenv()
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
from paddleocr import PaddleOCR  # noqa: E402

# Attribute mapping configuration for fuzzy matching
ATTRIBUTE_MAPPINGS: list[dict] = [
    {"db_attr": "name", "ocr_property": "name", "weight": 1.0},
    {"db_attr": "setCode", "ocr_property": "setcode_candidates", "weight": 2.0},
    {"db_attr": "number", "ocr_property": "number", "weight": 1.0},
    {"db_attr": "baseSetSize", "ocr_property": "base_set_size", "weight": 1.0},
    {"db_attr": "artist", "ocr_property": "artist", "weight": 0.5},
    {"db_attr": "setReleaseDate", "ocr_property": "set_release_date", "weight": 0.5},
]


class Properties(NamedTuple):
    """Extracted card properties from OCR results."""

    # Raw PaddleOCR results on bottom crop (ROI for setcode, collector, year, artist)
    rec_texts_bot: list[str] | None
    rec_scores_bot: list[float] | None
    # Raw PaddleOCR results on top crop (ROI for card name)
    rec_texts_top: list[str] | None
    rec_scores_top: list[float] | None
    # Extracted candidate setcodes
    setcode_candidates: list[str]
    # Extracted properties from raw PaddleOCR texts
    name: str | None
    number: str | None
    base_set_size: str | None
    artist: list[str]
    set_release_date: str | None


class CardOCR:
    """OCR-based card matcher using PaddleOCR for text extraction."""

    # ROI coordinates as fractions of image dimensions (H x W)
    # Name ROI: horizontal text area at top of card
    NAME_ROI_TOP = 0.04
    NAME_ROI_BOTTOM = 0.11
    NAME_ROI_LEFT = 0.06
    NAME_ROI_RIGHT = 0.9

    # Info ROI: card information at bottom (set code, number, artist, date)
    INFO_ROI_TOP = 0.87
    INFO_ROI_BOTTOM = 1.0
    INFO_ROI_LEFT = 0.05
    INFO_ROI_RIGHT = 0.76

    def __init__(self, enable_hpi=True, det_mobile=False):
        enable_hpi = enable_hpi
        cfg = None

        if enable_hpi:
            cfg = load_pipeline_config("OCR")

            # Use high performance onnx backend runtime for text-detection
            cfg["SubModules"]["TextDetection"]["use_hpip"] = True
            cfg["SubModules"]["TextDetection"]["hpi_config"] = {
                "backend": "onnxruntime",
            }
            # Keep normal paddlex backend runtime for text-recognition (onnx backend does not work for rec)
            cfg["SubModules"]["TextRecognition"]["use_hpip"] = False
            cfg["SubModules"]["TextRecognition"]["hpi_config"] = {
                "backend": "paddle",
                "auto_paddle2onnx": False,  # optional but useful to prevent ONNX conversion attempts
            }

        self._paddleocr = PaddleOCR(
            paddlex_config=cfg,
            # document orientation correction (corrects orientation of document images)
            use_doc_orientation_classify=False,
            doc_orientation_classify_model_name=None,
            # text image rectification (performs geometric transformations on images to correct distortions, inclinations, perspective deformations)
            use_doc_unwarping=False,
            doc_unwarping_model_name=None,
            # textline orientation classification (corrects the orientation of text lines)
            use_textline_orientation=False,
            textline_orientation_model_name="PP-LCNet_x0_25_textline_ori",
            # text detection (locating and marking text-containing regions in images)
            text_detection_model_name="PP-OCRv5_server_det",
            text_detection_model_dir="src/mtg_recognition/engine/ocr/models/prod/PP-OCRv5_server_det",
            # text_detection_model_name="PP-OCRv5_mobile_det",
            # text_detection_model_dir="src/mtg_recognition/engine/ocr/models/test/PP-OCRv5_mobile_det",
            # text recognition
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            text_recognition_model_dir="src/mtg_recognition/engine/ocr/models/prod/PP-OCRv5_mobile_rec",
            # other params
            device="cpu",
            enable_hpi=enable_hpi,  # enable high-performance inference
            precision="fp16",  # set to "fp16" for acceleration
            lang="en",
        )

    def predict(
        self, img: np.ndarray, thresh: float = 0.0
    ) -> tuple[list[str], list[float]]:
        # Predict with PaddleOCR
        result = self._paddleocr.predict(img, text_det_thresh=thresh)

        # Extract raw results
        rec_texts = result[0]["rec_texts"]
        rec_scores = result[0]["rec_scores"]  # scores from paddleOCR
        return rec_texts, rec_scores

    def extract_properties(self, query_img: np.ndarray) -> Properties:
        """
        Extract card properties from an image using OCR on specific ROIs.

        Args:
            query_img: Card image (BGR format from OpenCV)

        Returns:
            Properties namedtuple with extracted card information
        """
        h, w, _ = query_img.shape

        # Extract ROIs using class constants
        name_roi_img = query_img[
            int(h * self.NAME_ROI_TOP) : int(h * self.NAME_ROI_BOTTOM),
            int(w * self.NAME_ROI_LEFT) : int(w * self.NAME_ROI_RIGHT),
        ]
        info_roi_img = query_img[
            int(h * self.INFO_ROI_TOP) : int(h * self.INFO_ROI_BOTTOM),
            int(w * self.INFO_ROI_LEFT) : int(w * self.INFO_ROI_RIGHT),
        ]

        # Perform OCR on each ROI
        texts_name_roi, scores_name_roi = self.predict(name_roi_img)
        texts_info_roi, scores_info_roi = self.predict(info_roi_img)

        # Extract structured properties from OCR texts
        name = ocr_utils.extract_card_name(texts_name_roi)
        setcodes = ocr_utils.extract_setcode_candidates(texts_info_roi)
        number, base_set_size = ocr_utils.extract_number_and_base_set_size(
            texts_info_roi
        )
        artist = texts_info_roi
        set_release_date = ocr_utils.extract_copyright_year(texts_info_roi)

        properties = Properties(
            # Raw PaddleOCR results
            rec_texts_bot=texts_info_roi,
            rec_scores_bot=scores_info_roi,
            rec_texts_top=texts_name_roi,
            rec_scores_top=scores_name_roi,
            # Extracted properties
            setcode_candidates=setcodes,
            name=name,
            number=number,
            base_set_size=base_set_size,
            artist=artist,
            set_release_date=set_release_date,
        )

        return properties

    def find_best_matches(
        self,
        query_img: np.ndarray,
        candidates: list[Row],
        properties: Properties | None = None,
        k: int | None = 10,
        thresh: float = 0.0,
    ) -> tuple[list[Row], list[float]]:
        """
        Find best matching cards by comparing OCR-extracted properties against candidates.

        Uses fuzzy string matching to compare card attributes (name, set code, number, etc.)
        from OCR results against a list of candidate cards.

        Args:
            query_img: Card image (BGR format from OpenCV)
            candidates: List of candidate card Row objects to match against
            properties: Pre-extracted properties (if None, will extract from query_img)
            k: Number of top matches to return (None = return all)
            thresh: Minimum score threshold (0.0 to 100.0)

        Returns:
            Tuple of (best_matches, best_scores)
        """
        if properties is None:
            properties = self.extract_properties(query_img)

        n_candidates = len(candidates)

        # Compute fuzzy match scores for each attribute
        scores_list = []
        weights = []

        for mapping in ATTRIBUTE_MAPPINGS:
            db_attr = str(mapping["db_attr"])
            ocr_property = str(mapping["ocr_property"])
            weight = mapping["weight"]

            # Get values from database candidates
            choices = [getattr(candidate, db_attr) for candidate in candidates]

            # Get values from OCR properties (normalize to list)
            queries = getattr(properties, ocr_property)
            queries = queries if isinstance(queries, list) else [queries]

            # Compute fuzzy matching scores
            if queries and choices:
                # Use cdist for efficient pairwise comparison
                dist_matrix = rapidfuzz.process.cdist(
                    queries=queries,  # cdist correctly handles [None] or [""]
                    choices=choices,
                    scorer=rapidfuzz.fuzz.ratio,
                    score_cutoff=0,
                    workers=-1,
                )
                # Take max score across all query variants (e.g., multiple setcode candidates)
                scores = np.amax(dist_matrix, axis=0)
            else:
                scores = np.zeros(n_candidates, dtype=np.float32)

            scores_list.append(scores)
            weights.append(weight)

        # Aggregate scores with weighted average
        scores_matrix = np.vstack(scores_list)
        scores_aggregate = np.average(a=scores_matrix, axis=0, weights=weights)

        # Sort candidates based on score (descending)
        top_scores, top_indices = ocr_utils.sort_scores_with_indices(
            scores_aggregate, k, thresh
        )

        # Filter top candidates
        if not (top_scores and top_indices):
            print(
                f"No candidates found with k={k} and threshold={thresh}. "
                "Try lowering k or thresh.\nReturning input candidates without filtering..."
            )
            return candidates, scores_aggregate.tolist()

        top_candidates = [candidates[i] for i in top_indices]

        return top_candidates, top_scores


@click.command()
@click.option(
    "-i",
    "--input",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    default="benchmarks/dataset/images/ready",
    help="Path to input image or directory of images (supports .jpg)",
)
@click.option(
    "--enable-hpi",
    type=bool,
    is_flag=True,
    help="Enable high-performance inference (HPI) for OCR models",
)
def main(input, enable_hpi):
    if input.is_file():
        img_paths = [input]
    elif input.is_dir():
        img_paths = sorted(list(input.glob("*.jpg")))

    if not img_paths:
        print(f"No .jpg images found in: {input}")
        return

    ocr = CardOCR(enable_hpi=enable_hpi)
    from ..matching.phash_matcher import PHashCardMatcher

    matcher = PHashCardMatcher(auto_build=True)

    for i, img_path in enumerate(img_paths):
        query_img = cv2.imread(str(img_path))
        if query_img is None:
            print(f"Failed to read image: {img_path}")
            continue

        # Find candidates with deep embeddings
        candidates_dem, scores_dem = matcher.find_best_matches(query_img, k=100)
        properties = ocr.extract_properties(query_img)
        print(
            f"{properties.setcode_candidates} \t {properties.number} \t {properties.base_set_size} \t {properties.name} \t {properties.set_release_date} \t {properties.rec_texts_bot}"
        )

        # Find best matches with OCR
        success = False
        candidates_ocr, scores_ocr = ocr.find_best_matches(
            query_img, candidates_dem, properties, k=10, thresh=0.0
        )

        if candidates_ocr:
            success = candidates_ocr[0].uuid == img_path.stem
            ic(candidates_ocr[0], scores_ocr[0])

        print(f"{i} {success} \t {img_path.stem}")
        print("\n ------------------------------------------- \n")


if __name__ == "__main__":
    main()

# uv run python -m mtg_recognition.engine.ocr.ocr --enable-hpi
