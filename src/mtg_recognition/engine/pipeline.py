from dataclasses import dataclass

import numpy as np

from mtg_recognition.engine.image import ImageSource, prepare_image
from mtg_recognition.engine.matching.phash import load_phash_index, shortlist
from mtg_recognition.engine.ocr import scoring
from mtg_recognition.engine.ocr.reading import PaddleCardReader, Reading
from mtg_recognition.engine.ranking import RankedPrinting
from mtg_recognition.engine.segmentation.yolo import YoloCardSegmentor
from mtg_recognition.reference.catalog import SqliteCardCatalog
from mtg_recognition.reference.layout import ReferenceData


@dataclass(frozen=True)
class SegmentResult:
    """Recognition result for one card found in the image."""

    warped_image: np.ndarray
    ranked: list[RankedPrinting]  # best first
    # None when no candidate was left to compare it with, or OCR failed
    reading: Reading | None
    # Set when OCR failed and the ranking fell back to pHash order
    ocr_error: str | None = None


@dataclass(frozen=True)
class RecognitionResult:
    """Recognition result for one input image."""

    segments: list[SegmentResult]  # empty = no card found
    error: str | None


class Pipeline:
    """The engine's recognition module: image in, a ranking per card out.

    Segments the image, finds pHash candidates, resolves them to printings
    through the SQLite catalog, reads each card and ranks the candidates by
    scoring the reading against them.

    Args:
        k_phash_candidates: Number of candidates per card: printings shortlisted
            by pHash similarity, each scored against the card's reading.
        k_final_results: Number of printings ranked per card.

    Raises:
        ValueError: If the k parameters are invalid, or the pHash index files hold
            different numbers of hashes and UUIDs.
        FileNotFoundError: If a pHash index file is missing.
    """

    def __init__(self, k_phash_candidates: int = 20, k_final_results: int = 1):
        if not 1 <= k_final_results <= k_phash_candidates:
            raise ValueError(
                f"Need 1 <= k_final_results <= k_phash_candidates, "
                f"got {k_final_results} and {k_phash_candidates}"
            )
        self.k_phash_candidates = k_phash_candidates
        self.k_final_results = k_final_results

        reference = ReferenceData.default()
        self.phash_index = load_phash_index(reference)
        self.catalog = SqliteCardCatalog(reference.catalog_path)
        self.segmentor = YoloCardSegmentor()
        self.reader = PaddleCardReader()

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
            # prepare and segment the image
            segmentation = self.segmentor.segment(prepare_image(image))
            segments: list[SegmentResult] = []
            for card in segmentation.segments:
                warped = card.warped_image

                # shortlist pHash candidates
                uuids, _ = shortlist(
                    self.phash_index, warped, k=self.k_phash_candidates
                )

                # resolve them through the catalog, keeping pHash order and
                # dropping UUIDs not in it
                printings = self.catalog.lookup(uuids)
                candidates = [printings[u] for u in uuids if u in printings]

                # no candidate left: empty ranking, no reading
                if not candidates:
                    segments.append(SegmentResult(warped, ranked=[], reading=None))
                    continue

                # read the card; when OCR fails, the empty reading scores every
                # candidate 0 and ties keep pHash order, so the ranking falls
                # back to pHash alone
                reading: Reading | None = None
                ocr_error: str | None = None
                try:
                    reading = self.reader.read(warped, debug=debug)
                except RuntimeError as e:
                    ocr_error = f"{type(e).__name__}: {e}"

                # rank the candidates by the card's reading
                ranked = scoring.rank(
                    reading if reading is not None else Reading(),
                    candidates,
                    k=self.k_final_results,
                )
                segments.append(
                    SegmentResult(
                        warped, ranked=ranked, reading=reading, ocr_error=ocr_error
                    )
                )
        except Exception as e:
            return RecognitionResult(segments=[], error=f"{type(e).__name__}: {e}")
        return RecognitionResult(segments=segments, error=None)
