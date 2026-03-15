import time
from pathlib import Path

import click
import numpy as np
from rich import print
from sqlalchemy.engine.row import Row

from mtg_recognition.engine.matching.phash_matcher import PHashCardMatcher
from mtg_recognition.engine.ocr.ocr import CardOCR
from mtg_recognition.engine.segmentation.segment import (
    CardSegmentor,
    SegmentationResult,
)


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
        """Segment the card from a query image."""
        return self.segmentor.segment(query_img, **kwargs)

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
        matches, scores, _ = self.find_best_matches_with_diagnostics(
            query_img=query_img,
            k_phash_candidates=k_phash_candidates,
            k_final_results=k_final_results,
            apply_segmentation=apply_segmentation,
        )
        return matches, scores

    def find_best_matches_with_diagnostics(
        self,
        query_img: np.ndarray,
        k_phash_candidates: int = 100,
        k_final_results: int = 1,
        apply_segmentation: bool = True,
    ) -> tuple[list[Row], list[float], dict[str, float]]:
        """Find best matches and return stage timing diagnostics in milliseconds."""
        total_start = time.perf_counter()

        diagnostics = {
            "segmentation_ms": 0.0,
            "phash_ms": 0.0,
            "ocr_ms": 0.0,
            "total_ms": 0.0,
        }

        # I. Segmentation
        if apply_segmentation:
            segmentation_start = time.perf_counter()
            result = self.segment_card(query_img, return_warped=True)
            img_warped = result.warped_image
            diagnostics["segmentation_ms"] = (
                time.perf_counter() - segmentation_start
            ) * 1000.0
        else:
            img_warped = query_img

        if img_warped is None:
            print("No image detected")
            diagnostics["total_ms"] = (time.perf_counter() - total_start) * 1000.0
            return [], [], diagnostics

        # II. Matching
        assert k_final_results <= k_phash_candidates, (
            "k_final_results must be <= k_phash_candidates"
        )

        # Stage 1: Coarse search with perceptual hashing
        phash_start = time.perf_counter()
        matches_ph, scores_ph = self.phash_matcher.find_best_matches(
            img_warped, k=k_phash_candidates
        )
        diagnostics["phash_ms"] = (time.perf_counter() - phash_start) * 1000.0

        if len(matches_ph) == 0:
            diagnostics["total_ms"] = (time.perf_counter() - total_start) * 1000.0
            return [], [], diagnostics
        elif len(matches_ph) == 1:
            diagnostics["total_ms"] = (time.perf_counter() - total_start) * 1000.0
            return matches_ph, scores_ph, diagnostics

        # Stage 2: Accurate OCR search on reduced search space
        ocr_start = time.perf_counter()
        matches_ocr, scores_ocr = self.ocr.find_best_matches(
            img_warped, matches_ph, k=k_final_results, thresh=0.0
        )
        diagnostics["ocr_ms"] = (time.perf_counter() - ocr_start) * 1000.0

        if len(matches_ocr) == 0:
            diagnostics["total_ms"] = (time.perf_counter() - total_start) * 1000.0
            return [], [], diagnostics

        diagnostics["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        return matches_ocr, scores_ocr, diagnostics

    def _load_index(self):
        """Load indices for all matchers."""
        # self.deep_matcher._load_index()  # Commented - keeping for future use
        self.phash_matcher._load_index()

    def build_index(self, force: bool = False):
        """Build indices for all matchers."""
        # self.deep_matcher.build_index(force)  # Commented - keeping for future use
        self.phash_matcher.build_index(force)


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
    default="benchmarks/dataset/images/raw",
    help="Path to input image or directory of images (supports .jpg)",
)
@click.option(
    "-k",
    type=click.INT,
    default=1,
    help="Number of final matches to return after OCR refinement",
)
@click.option(
    "-n",
    type=click.INT,
    default=100,
    help="Number of candidates to retrieve from perceptual hash stage",
)
def main(input: Path, k: int, n: int):
    if input.is_file():
        img_paths = [input]
    elif input.is_dir():
        img_paths = sorted(list(input.glob("*.jpg")))

    if not img_paths:
        print(f"No .jpg images found in: {input}")
        return

    import cv2

    pipeline = Pipeline(auto_build=True)
    pipeline.build_index(force=False)

    for i, img_path in enumerate(img_paths):
        query_img = cv2.imread(str(img_path))
        try:
            best_matches, best_scores = pipeline.find_best_matches(
                query_img, k_final_results=k, k_phash_candidates=n
            )

            success = best_matches[0].uuid == img_path.stem
            print(f"{i} {success} \t {img_path.stem}")
            print(f"Top {k} matches:")
            for i, match in enumerate(best_matches):
                print(
                    f"  UUID: {match.uuid}, Name: {match.name}, Score: {best_scores[i]}\n"
                )
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    main()

# uv run python -m mtg_recognition.engine.pipeline
# uv run python -m mtg_recognition.engine.pipeline -i benchmarks/dataset/images/raw -k 1 -n 100
