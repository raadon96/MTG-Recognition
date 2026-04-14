import json
import time
import uuid as _uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import click
import cv2
import numpy as np
from rich import print
from sqlalchemy.engine.row import Row

from mtg_recognition.core import image_utils
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


def _stem_uuid(stem: str) -> str | None:
    """Return stem if it is a valid UUID string, else None."""
    try:
        _uuid.UUID(stem)
        return stem
    except ValueError:
        return None


@click.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=True, readable=True, path_type=Path
    ),
    required=True,
    help="Path to a single image file or a directory of image files.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Directory where JSON result files are written (one per image). "
    "When omitted, results are printed to the terminal only.",
)
@click.option(
    "--k-phash",
    type=click.INT,
    default=20,
    show_default=True,
    help="Number of pHash candidates for the coarse retrieval stage.",
)
@click.option(
    "-k",
    type=click.INT,
    default=1,
    show_default=True,
    help="Number of top matches to return after OCR refinement.",
)
def main(input_path: Path, output_dir: Path | None, k_phash: int, k: int) -> None:
    """Run the full MTG card recognition pipeline on one image or a directory.

    Always segments the input image (the segmentor's bypass path handles
    pre-cropped scans without running any model). Results are printed
    incrementally. Pass -o to also save a JSON file per image.

    Examples:

    \b
    uv run python -m mtg_recognition.engine.pipeline -i card.jpg
    uv run python -m mtg_recognition.engine.pipeline -i benchmarks/dataset/images/raw -k 3 --k-phash 50
    uv run python -m mtg_recognition.engine.pipeline -i data/test/images -o output/results
    """
    if input_path.is_file():
        img_paths = [input_path]
    else:
        img_paths = image_utils.get_image_files_in_directory(input_path)

    if not img_paths:
        click.echo(f"No image files found in: {input_path}", err=True)
        raise SystemExit(1)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(auto_build=True)
    total = len(img_paths)

    for idx, img_path in enumerate(img_paths, 1):
        img_abs = str(img_path.resolve())
        try:
            img_display = str(img_path.resolve().relative_to(Path.cwd()))
        except ValueError:
            img_display = img_abs

        query_img = load_and_resize_image(img_path)
        if query_img is None:
            click.echo(
                f"[{idx}/{total}] {img_display}  ERROR: failed to load image", err=True
            )
            continue

        matches, scores, ocr_debug, diag = pipeline.find_best_matches(
            query_img,
            k_phash_candidates=k_phash,
            k_final_results=k,
            apply_segmentation=True,
            debug=True,
        )

        # UUID-based success check
        stem_uuid = _stem_uuid(img_path.stem)
        if stem_uuid is None:
            success: bool | None = None
        elif matches:
            success = matches[0].uuid == stem_uuid
        else:
            success = False

        # Status label
        if success is None:
            status = "—"
        elif success:
            status = "✓ MATCH"
        else:
            got = matches[0].uuid if matches else "no match"
            status = f"✗ MISS  (expected: {stem_uuid[:8]}…, got: {got[:8]}…)"

        # Raw OCR texts and scores
        raw_texts: list[str] = []
        raw_scores: list[float] = []
        if ocr_debug is not None:
            raw_texts = ocr_debug.raw_texts_df["text"].tolist()
            raw_scores = [
                round(float(s), 3) for s in ocr_debug.raw_texts_df["score"].tolist()
            ]

        # --- Terminal output (incremental) ---
        click.echo(f"[{idx}/{total}] {img_display}  {status}")
        if not matches:
            click.echo("  no matches found")
        for rank, (match, score) in enumerate(zip(matches, scores), 1):
            click.echo(
                f"  #{rank}  {match.name}  |  {match.setCode} · {match.number}  |  score: {score:.1f}"
            )
        click.echo(f"  raw_texts:  {raw_texts}")
        click.echo(f"  raw_scores: {raw_scores}")
        click.echo(
            f"  timings: seg={diag.segmentation_ms:.0f}ms  "
            f"phash={diag.phash_ms:.0f}ms  "
            f"ocr={diag.ocr_ms:.0f}ms  "
            f"total={diag.total_ms:.0f}ms"
        )
        click.echo("")

        # --- JSON output ---
        if output_dir is not None:
            result: dict = {
                "image": img_abs,
                "success": success,
                "matches": [
                    {
                        "rank": rank,
                        "name": match.name,
                        "uuid": match.uuid,
                        "setCode": match.setCode,
                        "number": match.number,
                        "score": round(score, 1),
                    }
                    for rank, (match, score) in enumerate(zip(matches, scores), 1)
                ],
                "raw_texts": raw_texts,
                "raw_scores": raw_scores,
                "timings_ms": {
                    "segmentation_ms": round(diag.segmentation_ms, 1),
                    "phash_ms": round(diag.phash_ms, 1),
                    "ocr_ms": round(diag.ocr_ms, 1),
                    "total_ms": round(diag.total_ms, 1),
                },
            }
            out_path = output_dir / f"{img_path.stem}.json"
            out_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
