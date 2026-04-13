import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import click
import cv2
import numpy as np
from rich import print
from sqlalchemy.engine.row import Row

from mtg_recognition.core.image_utils import load_and_resize_image
from mtg_recognition.engine.matching.phash_matcher import PHashCardMatcher
from mtg_recognition.engine.ocr.ocr import CardOCR, OCRDebugData
from mtg_recognition.engine.segmentation.seg_utils import ImageSegmentationResult
from mtg_recognition.engine.segmentation.yolo import YoloCardSegmentor


@dataclass
class PipelineDiagnostics:
    """Per-stage timing for a single pipeline run (milliseconds)."""

    segmentation_ms: float = 0.0
    phash_ms: float = 0.0
    ocr_ms: float = 0.0
    total_ms: float = 0.0


@contextmanager
def _timer(diag: PipelineDiagnostics, attr: str) -> Generator[None, None, None]:
    """Context manager that measures elapsed time and stores it on a PipelineDiagnostics field."""
    start = time.perf_counter()
    try:
        yield
    finally:
        setattr(diag, attr, (time.perf_counter() - start) * 1000.0)


class Pipeline:
    def __init__(
        self,
        auto_build: bool = True,
        segmentor: YoloCardSegmentor | None = None,
        phash_matcher: PHashCardMatcher = PHashCardMatcher(),
        ocr: CardOCR = CardOCR(
            det_mobile=True, enable_hpi=False, use_preprocessing=True
        ),
    ):
        self.segmentor = segmentor if segmentor is not None else YoloCardSegmentor()
        self.phash_matcher = phash_matcher
        self.ocr = ocr
        self.auto_build = auto_build

        if auto_build:
            self.phash_matcher.build_index(force=False)

    def segment_card(self, query_img: np.ndarray) -> ImageSegmentationResult:
        """Segment cards from a query image."""
        return self.segmentor.segment(query_img)

    def find_best_matches(
        self,
        query_img: np.ndarray,
        k_phash_candidates: int = 20,
        k_final_results: int = 1,
        apply_segmentation: bool = True,
        debug: bool = False,
    ) -> tuple[list[Row], list[float], OCRDebugData | None, PipelineDiagnostics]:
        """Find best matching cards using a two-stage pipeline.

        Stage 1: Perceptual hashing for fast coarse candidate retrieval.
        Stage 2: OCR-based fuzzy matching for accurate refinement.

        Timing diagnostics are always collected (near-zero overhead).
        OCRDebugData is only populated when debug=True.

        Args:
            query_img: Query image (BGR format from OpenCV).
            k_phash_candidates: Number of candidates from the pHash stage.
            k_final_results: Number of final results after OCR refinement.
            apply_segmentation: Whether to run card segmentation first.
            debug: When True, ocr_debug in the return value is populated.

        Returns:
            Tuple of (matches, scores, ocr_debug, diag).
            ocr_debug is None when debug=False.
            Use _ to discard either trailing value when not needed.
        """
        total_start = time.perf_counter()
        diag = PipelineDiagnostics()

        # I. Segmentation
        if apply_segmentation:
            with _timer(diag, "segmentation_ms"):
                seg_result = self.segment_card(query_img)
            if not seg_result.success:
                print("No card detected")
                diag.total_ms = (time.perf_counter() - total_start) * 1000.0
                return [], [], None, diag
            img_warped = seg_result.segments[0].warped_image
        else:
            img_warped = query_img

        # II. Matching
        assert k_final_results <= k_phash_candidates, (
            "k_final_results must be <= k_phash_candidates"
        )

        # Stage 1: Coarse search with perceptual hashing
        with _timer(diag, "phash_ms"):
            matches_ph, scores_ph = self.phash_matcher.find_best_matches(
                img_warped, k=k_phash_candidates
            )

        if len(matches_ph) == 0:
            diag.total_ms = (time.perf_counter() - total_start) * 1000.0
            return [], [], None, diag
        elif len(matches_ph) == 1:
            diag.total_ms = (time.perf_counter() - total_start) * 1000.0
            return matches_ph, scores_ph, None, diag

        # Stage 2: Accurate OCR search on reduced search space
        with _timer(diag, "ocr_ms"):
            matches_ocr, scores_ocr, ocr_debug = self.ocr.find_best_matches(
                img_warped, matches_ph, k=k_final_results, thresh=0.0, debug=debug
            )

        if len(matches_ocr) == 0:
            diag.total_ms = (time.perf_counter() - total_start) * 1000.0
            return [], [], ocr_debug if debug else None, diag

        diag.total_ms = (time.perf_counter() - total_start) * 1000.0
        return matches_ocr, scores_ocr, ocr_debug if debug else None, diag

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
    else:
        print(f"Input path does not exist: {input}")
        return

    if not img_paths:
        print(f"No .jpg images found in: {input}")
        return

    pipeline = Pipeline(auto_build=True)
    pipeline.build_index(force=False)

    for i, img_path in enumerate(img_paths):
        query_img = load_and_resize_image(img_path)
        print(f"image size = {query_img.shape[:2]} (h x w)")
        try:
            best_matches, best_scores, _, _ = pipeline.find_best_matches(
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
# uv run python -m mtg_recognition.engine.pipeline -i benchmarks/dataset/images/raw -k 1 -n 20
