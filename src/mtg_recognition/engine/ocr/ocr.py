"""OCR-based card property extraction and fuzzy matching against pHash candidates.

Use `engine/pipeline.py` (or `python -m mtg_recognition.engine.pipeline`) for
the full end-to-end pipeline including segmentation and pHash coarse search.
"""

from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd
import rapidfuzz
from paddleocr import PaddleOCR
from paddlex.inference import load_pipeline_config
from PIL import Image
from sqlalchemy.engine.row import Row

from mtg_recognition.core.image_utils import _ensure_min_dim
from mtg_recognition.engine.ocr import ocr_utils
from mtg_recognition.engine.ocr.enhance import preprocess_for_ocr

# Attribute mapping configuration for fuzzy matching
ATTRIBUTE_MAPPINGS: list[dict] = [
    {"db_attr": "name", "ocr_property": "name", "weight": 1.0},
    {"db_attr": "setCode", "ocr_property": "setcode_candidates", "weight": 1.0},
    {"db_attr": "number", "ocr_property": "number", "weight": 1.0},
    {"db_attr": "baseSetSize", "ocr_property": "base_set_size", "weight": 1.0},
    {"db_attr": "artist", "ocr_property": "artist", "weight": 0.5},
    {"db_attr": "setReleaseYear", "ocr_property": "set_release_year", "weight": 1.0},
    {"db_attr": "power", "ocr_property": "power", "weight": 1.0},
    {"db_attr": "toughness", "ocr_property": "toughness", "weight": 1.0},
]

_MIN_DIM = 32


class Properties(NamedTuple):
    """Card properties extracted from OCR results."""

    setcode_candidates: list[str]
    name: str | None
    number: str | None
    base_set_size: str | None
    artist: str | None
    set_release_year: str | None
    power: str | None
    toughness: str | None


@dataclass(frozen=True)
class OCRDebugData:
    """Intermediate OCR artefacts for mismatch root-cause investigation.

    Only populated when extract_properties / find_best_matches is called with
    debug=True. Never used in the production code path.
    """

    # Annotated info ROI image from PaddleOCR (side-by-side: original + text labels).
    # None if .img API is unavailable for this PaddleOCR version.
    annotated_info: Image.Image | None
    # Raw info-ROI OCR output: columns ["text", "score"], one row per detected fragment.
    raw_texts_df: pd.DataFrame
    # Extracted card properties (for Table 2: extracted vs. best match comparison).
    properties: "Properties"
    # Per-attribute fuzzy scores for the top-k candidates.
    # Index = candidate card names, columns = ATTRIBUTE_MAPPINGS db_attr keys.
    # None until find_best_matches populates it.
    score_matrix: pd.DataFrame | None


class CardOCR:
    """OCR-based card matcher using PaddleOCR for text extraction."""

    # ROI coordinates as fractions of image dimensions (H x W)
    # Name ROI: horizontal text area at top of card
    NAME_ROI_TOP = 0.04
    NAME_ROI_BOTTOM = 0.11
    NAME_ROI_LEFT = 0.06
    NAME_ROI_RIGHT = 0.9

    # Info ROI: card information at bottom (set code, number, artist, date)
    INFO_ROI_TOP = 0.88
    INFO_ROI_BOTTOM = 1.0
    INFO_ROI_LEFT = 0.05
    INFO_ROI_RIGHT = 0.95

    def __init__(self, enable_hpi=False, det_mobile=False, use_preprocessing=False):
        cfg = None
        self._use_preprocessing = use_preprocessing

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

        if det_mobile:
            det_model_name = "PP-OCRv5_mobile_det"
            det_model_dir = (
                "src/mtg_recognition/engine/ocr/models/prod/PP-OCRv5_mobile_det"
            )
        else:
            det_model_name = "PP-OCRv5_server_det"
            det_model_dir = (
                "src/mtg_recognition/engine/ocr/models/prod/PP-OCRv5_server_det"
            )

        self._paddleocr = PaddleOCR(
            paddlex_config=cfg,
            # document orientation correction (corrects orientation of document images)
            use_doc_orientation_classify=False,
            doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
            doc_orientation_classify_model_dir="src/mtg_recognition/engine/ocr/models/test/PP-LCNet_x1_0_doc_ori",
            # text image rectification (performs geometric transformations on images to correct distortions, inclinations, perspective deformations)
            use_doc_unwarping=False,
            doc_unwarping_model_name=None,
            # textline orientation classification (corrects the orientation of text lines)
            use_textline_orientation=False,
            # text detection (locating and marking text-containing regions in images)
            text_detection_model_name=det_model_name,
            text_detection_model_dir=det_model_dir,
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

    def extract_properties(
        self, query_img: np.ndarray, debug: bool = False
    ) -> tuple[Properties, OCRDebugData | None]:
        """
        Extract card properties from an image using OCR on specific ROIs.

        Args:
            query_img: Card image (BGR format from OpenCV)
            debug: When True, also returns an OCRDebugData with the annotated
                   info ROI image and raw OCR texts/scores. score_matrix is
                   None here — it is populated by find_best_matches.

        Returns:
            Tuple of (Properties, OCRDebugData | None). Second element is None
            when debug=False.
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

        if self._use_preprocessing:
            name_roi_img = preprocess_for_ocr(
                name_roi_img, upscale=False, enhance=False, deblur=False
            )
            info_roi_img = preprocess_for_ocr(
                info_roi_img, upscale=True, enhance=False, deblur=False
            )

        # Perform OCR on both ROIs in a single batched predict call.
        # Paddle's text-detection model requires both dimensions >= 32 px; upscale
        # any ROI that falls below this threshold before inference.

        name_roi_img = _ensure_min_dim(name_roi_img, _MIN_DIM)
        info_roi_img = _ensure_min_dim(info_roi_img, _MIN_DIM)

        texts_name_roi: list[str] = []
        texts_info_roi: list[str] = []
        scores_info_roi: list[float] = []
        batch_result = None

        try:
            batch_result = self._paddleocr.predict(
                [name_roi_img, info_roi_img], text_det_thresh=0.0
            )
            texts_name_roi = batch_result[0]["rec_texts"]
            texts_info_roi = batch_result[1]["rec_texts"]
            scores_info_roi = batch_result[1]["rec_scores"]
        except RuntimeError as e:
            print(f"OCR prediction failed: {e}\nProceeding with empty OCR results...")
            pass  # leave all text lists empty; scoring will use zero vectors

        # Extract structured properties from OCR texts
        name = ocr_utils.extract_card_name(texts_name_roi)
        setcodes = ocr_utils.extract_setcode_candidates(texts_info_roi)
        number, base_set_size = ocr_utils.extract_number_and_base_set_size(
            texts_info_roi
        )
        artist = ocr_utils.extract_artist_name(texts_info_roi)
        set_release_year = ocr_utils.extract_set_release_year(texts_info_roi)
        power, toughness = ocr_utils.extract_power_toughness(texts_info_roi)

        properties = Properties(
            setcode_candidates=setcodes,
            name=name,
            number=number,
            base_set_size=base_set_size,
            artist=artist,
            set_release_year=set_release_year,
            power=power,
            toughness=toughness,
        )

        if not debug:
            return properties, None

        annotated_info: Image.Image | None = None
        if batch_result is not None:
            try:
                annotated_info = batch_result[1].img["ocr_res_img"]
            except Exception as e:
                print(
                    f"Failed to extract annotated image from OCR result: {e}\nContinuing without annotated image..."
                )
                pass

        raw_texts_df = pd.DataFrame({"text": texts_info_roi, "score": scores_info_roi})

        ocr_debug = OCRDebugData(
            annotated_info=annotated_info,
            raw_texts_df=raw_texts_df,
            properties=properties,
            score_matrix=None,  # populated by find_best_matches
        )
        return properties, ocr_debug

    def _compute_scores(
        self, properties: Properties, candidates: list[Row]
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Compute per-attribute fuzzy scores for all candidates.

        Returns:
            scores_matrix: np.ndarray of shape [n_attributes, n_candidates]
            attribute_names: list of db_attr names from ATTRIBUTE_MAPPINGS
            candidate_labels: list of candidate card names for display
        """
        n_candidates = len(candidates)
        scores_list: list[np.ndarray] = []
        attribute_names: list[str] = []

        for mapping in ATTRIBUTE_MAPPINGS:
            db_attr = str(mapping["db_attr"])
            ocr_property = str(mapping["ocr_property"])

            choices = [getattr(candidate, db_attr) for candidate in candidates]
            queries = getattr(properties, ocr_property)
            queries = queries if isinstance(queries, list) else [queries]

            if queries and choices:
                dist_matrix = rapidfuzz.process.cdist(
                    queries=queries,
                    choices=choices,
                    scorer=rapidfuzz.fuzz.ratio,
                    score_cutoff=0,
                    workers=-1,
                )
                scores = np.amax(dist_matrix, axis=0)
            else:
                scores = np.zeros(n_candidates, dtype=np.float32)

            scores_list.append(scores)
            attribute_names.append(db_attr)

        scores_matrix = np.vstack(scores_list)  # [n_attrs, n_candidates]
        candidate_labels = [str(c.name) for c in candidates]
        return scores_matrix, attribute_names, candidate_labels

    def find_best_matches(
        self,
        query_img: np.ndarray,
        candidates: list[Row],
        properties: Properties | None = None,
        k: int | None = 10,
        thresh: float = 0.0,
        debug: bool = False,
    ) -> tuple[list[Row], list[float], OCRDebugData | None]:
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
            debug: When True, returns a populated OCRDebugData as the third element.

        Returns:
            Tuple of (best_matches, best_scores, OCRDebugData | None). Third element
            is None when debug=False.
        """
        ocr_debug: OCRDebugData | None = None
        if properties is None:
            properties, ocr_debug = self.extract_properties(query_img, debug=debug)

        weights = [float(m["weight"]) for m in ATTRIBUTE_MAPPINGS]
        scores_matrix, attribute_names, candidate_labels = self._compute_scores(
            properties, candidates
        )
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
            return candidates, scores_aggregate.tolist(), ocr_debug

        top_candidates = [candidates[i] for i in top_indices]

        if debug and ocr_debug is not None:
            # Prefix each label with its rank so duplicate card names (reprints)
            # don't produce a non-unique index, which breaks pandas Styler.
            top_labels = [
                f"#{rank + 1} {candidate_labels[i]}"
                for rank, i in enumerate(top_indices)
            ]
            top_scores_matrix = scores_matrix[:, top_indices]
            score_matrix_df = pd.DataFrame(
                top_scores_matrix.T,
                index=top_labels,
                columns=attribute_names,
            )
            ocr_debug = OCRDebugData(
                annotated_info=ocr_debug.annotated_info,
                raw_texts_df=ocr_debug.raw_texts_df,
                properties=ocr_debug.properties,
                score_matrix=score_matrix_df,
            )

        return top_candidates, top_scores, ocr_debug
