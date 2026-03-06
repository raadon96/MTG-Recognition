import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from pathlib import Path

import numpy as np
from rich import print
from sqlalchemy.engine.row import Row

from .matching.phash_matcher import PHashCardMatcher
from .ocr.ocr import CardOCR
from .segmentation.segment import CardSegmentor, SegmentationResult


class Pipeline:
    def __init__(
        self,
        auto_build: bool = True,
        segmentor: CardSegmentor = CardSegmentor("yoloe-11s-seg", 0),
        phash_matcher: PHashCardMatcher = PHashCardMatcher(),
        ocr: CardOCR = CardOCR(),
    ):
        self.segmentor = segmentor
        self.phash_matcher = phash_matcher
        self.ocr = ocr
        self.auto_build = auto_build

        if auto_build:
            self.phash_matcher.build_index(force=False)

    def segment_card(self, query_img: np.ndarray, **kwargs) -> SegmentationResult:
        """Segment the card from the query image and return the warped image."""
        result = self.segmentor.segment(query_img, **kwargs)
        return result

    def find_best_matches(
        self,
        query_img: np.ndarray,
        k_phash_candidates: int = 100,  # top k candidates from PHash Matcher
        k_final_results: int = 1,  # top k final results after OCR
        apply_segmentation: bool = True,
    ) -> tuple[list[Row], list[float]]:
        """
        Find best matching cards using a two-stage pipeline:
        1. Perceptual hashing for fast coarse search
        2. OCR-based fuzzy matching for accurate refinement

        Args:
            query_img: Query image (BGR format from OpenCV)
            k_phash_candidates: Number of candidates to retrieve from perceptual hash stage
            k_final_results: Number of final results to return after OCR refinement
            apply_segmentation: Whether to apply card segmentation first

        Returns:
            Tuple of (matches, scores) where matches are Row objects and scores are floats
        """
        # I. Segmentation
        if apply_segmentation:
            result = self.segment_card(query_img, return_warped=True)
            img_warped = result.warped_image
        else:
            img_warped = query_img

        if img_warped is None:
            print("No image detected")
            return [], []

        # II. Matching
        assert k_final_results <= k_phash_candidates, (
            "k_final_results must be <= k_phash_candidates"
        )

        # Stage 1: Coarse search with perceptual hashing
        matches_ph, scores_ph = self.phash_matcher.find_best_matches(
            img_warped, k=k_phash_candidates
        )

        if len(matches_ph) == 0:
            return [], []
        elif len(matches_ph) == 1:
            return matches_ph, scores_ph

        # Stage 2: Accurate OCR search on reduced search space
        matches_ocr, scores_ocr = self.ocr.find_best_matches(
            img_warped, matches_ph, k=k_final_results, thresh=0.0
        )

        if len(matches_ocr) == 0:
            return [], []

        return matches_ocr, scores_ocr

    def _load_index(self):
        """Load indices for all matchers."""
        # self.deep_matcher._load_index()  # Commented - keeping for future use
        self.phash_matcher._load_index()

    def build_index(self, force: bool = False):
        """Build indices for all matchers."""
        # self.deep_matcher.build_index(force)  # Commented - keeping for future use
        self.phash_matcher.build_index(force)


def main(run_single_img=True, run_multiple_imgs=True):
    import cv2

    pipeline = Pipeline(auto_build=True)
    pipeline.build_index(force=False)

    if run_single_img:
        # Test with a sample segmented test image
        query_image_path = Path(
            "benchmarks/dataset/images/raw/0a2f85cd-f66d-5330-a9d5-314554f871a1.jpg"
        )
        assert query_image_path.exists()

        k = 1
        query_img = cv2.imread(str(query_image_path))
        best_matches, best_scores = pipeline.find_best_matches(
            query_img, k_final_results=k, k_phash_candidates=100
        )

        # ic(list(zip(best_matches, best_scores)))

        print(f"\nTop {k} matches:")
        for i, match in enumerate(best_matches):
            print(f"  UUID: {match.uuid}, Name: {match.name}, Score: {best_scores[i]}")

    if run_multiple_imgs:
        img_dir = Path("benchmarks/dataset/images/raw")
        img_paths = sorted(list(img_dir.glob("*.jpg")))

        for i, img_path in enumerate(img_paths):
            query_img = cv2.imread(str(img_path))
            try:
                best_matches, best_scores = pipeline.find_best_matches(
                    query_img, k_final_results=1, k_phash_candidates=100
                )

                success = best_matches[0].uuid == img_path.stem
                print(f"{i} {success} \t {img_path.stem} \t {best_matches}\n")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    main(run_single_img=True, run_multiple_imgs=True)

# uv run python -m mtg_recognition.engine.pipeline
