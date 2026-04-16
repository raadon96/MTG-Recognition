import json
import time
import uuid as _uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import click
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


@dataclass(frozen=True)
class SegmentResult:
    """Recognition result for a single detected card segment."""

    matches: list[Row]
    scores: list[float]
    ocr_debug: OCRDebugData | None
    diag: PipelineDiagnostics


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
        ocr: CardOCR = CardOCR(use_preprocessing=True),
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
    ) -> list[SegmentResult]:
        """Find best matching cards using a two-stage pipeline.

        Stage 1: Perceptual hashing for fast coarse candidate retrieval.
        Stage 2: OCR-based fuzzy matching for accurate refinement.

        Args:
            query_img: Query image (BGR format from OpenCV).
            k_phash_candidates: Number of candidates from the pHash stage.
            k_final_results: Number of final results after OCR refinement.
            apply_segmentation: When True, runs YOLO segmentation first and
                returns one SegmentResult per detected card. When False, treats
                query_img as a pre-warped single card and returns a list of one.
            debug: When True, ocr_debug inside each SegmentResult is populated.

        Returns:
            List of SegmentResult, one per detected card segment.
            Empty list when no cards are detected.
        """
        assert k_final_results <= k_phash_candidates, (
            "k_final_results must be <= k_phash_candidates"
        )

        if apply_segmentation:
            seg_total_start = time.perf_counter()
            seg_result = self.segment_card(query_img)
            segmentation_ms = (time.perf_counter() - seg_total_start) * 1000.0

            if not seg_result.success:
                print("No card detected")
                return []

            warped_images = [s.warped_image for s in seg_result.segments]
        else:
            segmentation_ms = 0.0
            warped_images = [query_img]

        results: list[SegmentResult] = []
        for img_warped in warped_images:
            results.append(
                self._match_one(
                    img_warped,
                    k_phash_candidates,
                    k_final_results,
                    debug,
                    segmentation_ms,
                )
            )
        return results

    def _match_one(
        self,
        img_warped: np.ndarray,
        k_phash_candidates: int,
        k_final_results: int,
        debug: bool,
        segmentation_ms: float,
    ) -> SegmentResult:
        """Run pHash + OCR matching on a single pre-warped card image."""
        diag = PipelineDiagnostics(segmentation_ms=segmentation_ms)
        total_start = time.perf_counter()

        # Stage 1: Coarse search with perceptual hashing
        with _timer(diag, "phash_ms"):
            matches_ph, scores_ph = self.phash_matcher.find_best_matches(
                img_warped, k=k_phash_candidates
            )

        if len(matches_ph) == 0:
            diag.total_ms = (
                time.perf_counter() - total_start
            ) * 1000.0 + segmentation_ms
            return SegmentResult(matches=[], scores=[], ocr_debug=None, diag=diag)

        if len(matches_ph) == 1:
            diag.total_ms = (
                time.perf_counter() - total_start
            ) * 1000.0 + segmentation_ms
            return SegmentResult(
                matches=matches_ph, scores=scores_ph, ocr_debug=None, diag=diag
            )

        # Stage 2: Accurate OCR search on reduced search space
        with _timer(diag, "ocr_ms"):
            matches_ocr, scores_ocr, ocr_debug = self.ocr.find_best_matches(
                img_warped, matches_ph, k=k_final_results, thresh=0.0, debug=debug
            )

        diag.total_ms = (time.perf_counter() - total_start) * 1000.0 + segmentation_ms
        return SegmentResult(
            matches=matches_ocr,
            scores=scores_ocr,
            ocr_debug=ocr_debug if debug else None,
            diag=diag,
        )

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
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to a JSON file where all results are written as a list. "
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
def main(input_path: Path, output_file: Path | None, k_phash: int, k: int) -> None:
    """Run the full MTG card recognition pipeline on one image or a directory.

    Always segments the input image (the segmentor's bypass path handles
    pre-cropped scans without running any model). Results are printed
    incrementally. Pass -o to also save all results to a single JSON file.

    Examples:

    \b
    uv run python -m mtg_recognition.engine.pipeline -i card.jpg
    uv run python -m mtg_recognition.engine.pipeline -i benchmarks/dataset/images/raw -k 3 --k-phash 50
    uv run python -m mtg_recognition.engine.pipeline -i data/test/images -o output/results.json
    """
    if input_path.is_file():
        img_paths = [input_path]
    else:
        img_paths = image_utils.get_image_files_in_directory(input_path)

    if not img_paths:
        click.echo(f"No image files found in: {input_path}", err=True)
        raise SystemExit(1)

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(auto_build=True)
    total = len(img_paths)
    all_results: list[dict] = []

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

        segment_results = pipeline.find_best_matches(
            query_img,
            k_phash_candidates=k_phash,
            k_final_results=k,
            apply_segmentation=True,
            debug=True,
        )

        stem_uuid = _stem_uuid(img_path.stem)
        n_segs = len(segment_results)
        click.echo(f"[{idx}/{total}] {img_display}  ({n_segs} segment(s) detected)")

        segments_json: list[dict] = []
        for seg_idx, seg in enumerate(segment_results):
            # UUID-based success check (only meaningful for single-card images)
            if stem_uuid is None:
                success: bool | None = None
            elif seg.matches:
                success = seg.matches[0].uuid == stem_uuid
            else:
                success = False

            # Status label
            if success is None:
                status = "—"
            elif success:
                status = "✓ MATCH"
            else:
                got = seg.matches[0].uuid if seg.matches else "no match"
                status = f"✗ MISS  (expected: {stem_uuid[:8]}…, got: {got[:8]}…)"

            # Raw OCR texts and scores
            raw_texts: list[str] = []
            raw_scores: list[float] = []
            if seg.ocr_debug is not None:
                raw_texts = seg.ocr_debug.raw_texts_df["text"].tolist()
                raw_scores = [
                    round(float(s), 3)
                    for s in seg.ocr_debug.raw_texts_df["score"].tolist()
                ]

            # Terminal output
            prefix = f"  [seg {seg_idx}]" if n_segs > 1 else " "
            click.echo(f"{prefix} {status}")
            if not seg.matches:
                click.echo(f"{prefix}  no matches found")
            for rank, (match, score) in enumerate(zip(seg.matches, seg.scores), 1):
                click.echo(
                    f"{prefix}  #{rank}  {match.name}  |  {match.setCode} · {match.number}  |  score: {score:.1f}"
                )
            click.echo(f"{prefix}  raw_texts:  {raw_texts}")
            click.echo(f"{prefix}  raw_scores: {raw_scores}")
            click.echo(
                f"{prefix}  timings: seg={seg.diag.segmentation_ms:.0f}ms  "
                f"phash={seg.diag.phash_ms:.0f}ms  "
                f"ocr={seg.diag.ocr_ms:.0f}ms  "
                f"total={seg.diag.total_ms:.0f}ms"
            )

            if output_file is not None:
                segments_json.append(
                    {
                        "segment_idx": seg_idx,
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
                            for rank, (match, score) in enumerate(
                                zip(seg.matches, seg.scores), 1
                            )
                        ],
                        "raw_texts": raw_texts,
                        "raw_scores": raw_scores,
                        "timings_ms": {
                            "segmentation_ms": round(seg.diag.segmentation_ms, 1),
                            "phash_ms": round(seg.diag.phash_ms, 1),
                            "ocr_ms": round(seg.diag.ocr_ms, 1),
                            "total_ms": round(seg.diag.total_ms, 1),
                        },
                    }
                )

        click.echo("")

        if output_file is not None:
            all_results.append({"image": img_abs, "segments": segments_json})

    if output_file is not None:
        output_file.write_text(json.dumps(all_results, indent=2))
        click.echo(f"Results written to {output_file}")


if __name__ == "__main__":
    main()
