import json
import uuid as _uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import click
import numpy as np

from mtg_recognition.catalog import CardCatalog, RankedPrinting, SqliteCardCatalog
from mtg_recognition.core import image_utils
from mtg_recognition.core.image_utils import ImageSource, prepare_image
from mtg_recognition.engine.matching.phash_matcher import PHashCardMatcher
from mtg_recognition.engine.ocr import scoring
from mtg_recognition.engine.ocr.reading import PaddleCardReader, Reading
from mtg_recognition.engine.segmentation.seg_utils import ImageSegmentationResult
from mtg_recognition.engine.segmentation.yolo import YoloCardSegmentor
from mtg_recognition.reference import ReferenceData


class CardSegmentor(Protocol):
    """Finds the cards in an image and warps each one flat (default:
    YoloCardSegmentor).
    """

    def segment(self, img: np.ndarray) -> ImageSegmentationResult: ...


class CandidateMatcher(Protocol):
    """Coarse search over the reference index by printing UUID (default:
    PHashCardMatcher).
    """

    @property
    def index_loaded(self) -> bool: ...

    def find_best_matches(
        self, query_image: np.ndarray, *, k: int
    ) -> tuple[list[str], list[float]]: ...


class CardReader(Protocol):
    """Reads a warped card image (default: PaddleCardReader)."""

    def read(self, card_image: np.ndarray, *, debug: bool) -> Reading: ...


@dataclass(frozen=True)
class SegmentResult:
    """Recognition result for one card found in the image."""

    warped_image: np.ndarray
    ranked: list[RankedPrinting]  # best first
    reading: Reading | None  # None when no candidate was left to compare it with


@dataclass(frozen=True)
class RecognitionResult:
    """Recognition result for one input image."""

    segments: list[SegmentResult]  # empty = no card found
    error: str | None


class Pipeline:
    """The engine's recognition module: image in, ranked Printings per card out.

    Segments the image, finds pHash candidates, resolves them to Printings
    through the catalog, reads each card and ranks the candidates by scoring the
    reading against them.

    Args:
        k_phash_candidates: Number of candidates from the pHash stage.
        k_final_results: Number of printings ranked per card.
        catalog: Resolves printing UUIDs to Printings (default: MobileCache SQLite).
        segmentor, phash_matcher, reader: Stage implementations (defaults built
            here).

    Raises:
        ValueError: If the k parameters are invalid or the pHash index is not loaded.
    """

    def __init__(
        self,
        k_phash_candidates: int = 20,
        k_final_results: int = 1,
        catalog: CardCatalog | None = None,
        segmentor: CardSegmentor | None = None,
        phash_matcher: CandidateMatcher | None = None,
        reader: CardReader | None = None,
    ):
        if not 1 <= k_final_results <= k_phash_candidates:
            raise ValueError(
                f"Need 1 <= k_final_results <= k_phash_candidates, "
                f"got {k_final_results} and {k_phash_candidates}"
            )
        self.k_phash_candidates = k_phash_candidates
        self.k_final_results = k_final_results

        self.catalog = (
            catalog
            if catalog is not None
            else SqliteCardCatalog(ReferenceData.default().db_path)
        )
        self.segmentor: CardSegmentor = (
            segmentor if segmentor is not None else YoloCardSegmentor()
        )
        self.phash_matcher: CandidateMatcher = (
            phash_matcher if phash_matcher is not None else PHashCardMatcher()
        )
        self.reader: CardReader = reader if reader is not None else PaddleCardReader()

        if not self.phash_matcher.index_loaded:
            raise ValueError("pHash index not loaded. Run 'mtg build-index' first.")

    def recognize(self, image: ImageSource, debug: bool = False) -> RecognitionResult:
        """Recognize every card in an image.

        The image is prepared with prepare_image. Failures, an unreadable image
        included, are returned in RecognitionResult.error rather than raised.

        Args:
            image: A path, encoded bytes, a file-like upload, a PIL image or an
                already-decoded BGR array.
            debug: When True, each reading carries the reader's debug artefacts.
        """
        try:
            segmentation = self.segmentor.segment(prepare_image(image))
            segments = [
                self._match_one(s.warped_image, debug) for s in segmentation.segments
            ]
        except Exception as e:
            return RecognitionResult(segments=[], error=f"{type(e).__name__}: {e}")
        return RecognitionResult(segments=segments, error=None)

    def _match_one(self, warped_image: np.ndarray, debug: bool) -> SegmentResult:
        """Find pHash candidates for one warped card and rank them by its reading."""

        # Stage 1: coarse search with perceptual hashing
        uuids, _ = self.phash_matcher.find_best_matches(
            warped_image, k=self.k_phash_candidates
        )
        printings = self.catalog.lookup(uuids)
        candidates = [printings[u] for u in uuids if u in printings]

        if not candidates:
            return SegmentResult(warped_image, ranked=[], reading=None)

        # Stage 2: read the card and score the reading against the candidates
        reading = self.reader.read(warped_image, debug=debug)
        ranked = scoring.rank(reading, candidates, k=self.k_final_results)
        return SegmentResult(warped_image, ranked=ranked, reading=reading)


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
    uv run python -m mtg_recognition.engine.pipeline -i card.jpg -o runs/output.json
    uv run python -m mtg_recognition.engine.pipeline \\
        -i data/test/6_benchmark/images/raw -k 3 --k-phash 50
    uv run python -m mtg_recognition.engine.pipeline \\
        -i data/test/images -o output/results.json
    """
    img_paths = (
        [input_path]
        if input_path.is_file()
        else image_utils.get_image_files_in_directory(input_path)
    )

    if not img_paths:
        click.echo(f"No image files found in: {input_path}", err=True)
        raise SystemExit(1)

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(k_phash_candidates=k_phash, k_final_results=k)
    total = len(img_paths)
    all_results: list[dict] = []

    for idx, img_path in enumerate(img_paths, 1):
        img_abs = str(img_path.resolve())
        try:
            img_display = str(img_path.resolve().relative_to(Path.cwd()))
        except ValueError:
            img_display = img_abs

        recognition = pipeline.recognize(img_path, debug=True)
        if recognition.error is not None:
            click.echo(
                f"[{idx}/{total}] {img_display}  ERROR: {recognition.error}", err=True
            )
        segment_results = recognition.segments

        stem_uuid = _stem_uuid(img_path.stem)
        n_segs = len(segment_results)
        if recognition.error is None:
            click.echo(f"[{idx}/{total}] {img_display}  ({n_segs} segment(s) detected)")

        segments_json: list[dict] = []
        for seg_idx, seg in enumerate(segment_results):
            # UUID-based success check (only meaningful for single-card images)
            if stem_uuid is None:
                success: bool | None = None
            elif seg.ranked:
                success = seg.ranked[0].printing.uuid == stem_uuid
            else:
                success = False

            # Status label
            if success is None:
                status = "—"
            elif success:
                status = "✓ MATCH"
            else:
                got = seg.ranked[0].printing.uuid if seg.ranked else "no match"
                status = f"✗ MISS  (expected: {stem_uuid[:8]}…, got: {got[:8]}…)"

            # Raw OCR texts and scores
            raw_texts: list[str] = []
            raw_scores: list[float] = []
            raw_boxes: list[tuple[int, int, int, int]] = []
            reading_debug = seg.reading.debug if seg.reading is not None else None
            if reading_debug is not None:
                raw_texts = reading_debug.texts
                raw_scores = reading_debug.scores
                raw_boxes = reading_debug.boxes

            # Terminal output
            prefix = f"  [seg {seg_idx}]" if n_segs > 1 else " "
            click.echo(f"{prefix} {status}")
            if not seg.ranked:
                click.echo(f"{prefix}  no matches found")
            for rank, entry in enumerate(seg.ranked, 1):
                p = entry.printing
                click.echo(
                    f"{prefix}  #{rank}  {p.name}  |  {p.set_code} · {p.number}  |  "
                    f"score: {entry.score:.1f}"
                )
            click.echo(f"{prefix}  raw_texts:  {raw_texts}")
            click.echo(f"{prefix}  raw_scores: {raw_scores}")

            if output_file is not None:
                out = {
                    "segment_idx": seg_idx,
                    "success": success,
                    "ranked": [
                        {
                            "rank": rank,
                            "name": entry.printing.name,
                            "uuid": entry.printing.uuid,
                            "setCode": entry.printing.set_code,
                            "number": entry.printing.number,
                            "score": round(entry.score, 1),
                        }
                        for rank, entry in enumerate(seg.ranked, 1)
                    ],
                    "debug": {
                        "raw_texts": raw_texts,
                        "raw_scores": raw_scores,
                        "raw_boxes": raw_boxes,
                    },
                }
                segments_json.append(out)

        click.echo("")

        if output_file is not None:
            all_results.append({"image": img_abs, "segments": segments_json})

    if output_file is not None:
        output_file.write_text(json.dumps(all_results, indent=2))
        click.echo(f"Results written to {output_file}")


if __name__ == "__main__":
    main()
