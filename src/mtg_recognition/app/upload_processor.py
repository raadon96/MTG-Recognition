"""Recognition of uploaded card images for the Streamlit app."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np

from mtg_recognition.catalog import RankedPrinting
from mtg_recognition.engine.ocr.reading import ReadingDebug
from mtg_recognition.engine.pipeline import Pipeline, RecognitionResult

# Quality of the warped card's display copy; recognition never sees it.
_DISPLAY_JPEG_QUALITY = 90


@dataclass(frozen=True)
class Upload:
    """One uploaded image file: its name and its encoded bytes, as uploaded."""

    name: str
    data: bytes


class UploadStatus(StrEnum):
    SUCCESS = "success"
    NO_MATCH = "no_match"
    ERROR = "error"


@dataclass
class UploadResult:
    """Result for one card found in an uploaded image, or for an image with no card or
    an error.
    """

    uploaded_filename: str
    original_image: bytes  # the upload itself, as uploaded
    ranked: list[RankedPrinting] = field(default_factory=list)  # best first
    error_message: str | None = None
    warped_image_jpeg: bytes | None = None  # the warped card, for display only
    reading_debug: ReadingDebug | None = None

    @property
    def status(self) -> UploadStatus:
        """Derived from the recognition result."""
        if self.error_message is not None:
            return UploadStatus.ERROR
        return UploadStatus.SUCCESS if self.ranked else UploadStatus.NO_MATCH

    @property
    def best(self) -> RankedPrinting | None:
        """The top-ranked entry, or None if there is no match."""
        return self.ranked[0] if self.ranked else None


# Exported fields of the best printing: (JSON key, CSV/Excel column, value).
_BEST_PRINTING_FIELDS: list[tuple[str, str, Callable[[RankedPrinting], object]]] = [
    ("card_name", "Card Name", lambda b: b.printing.name),
    ("uuid", "UUID", lambda b: b.printing.uuid),
    ("score", "Score", lambda b: b.score),
    ("set_code", "Set Code", lambda b: b.printing.set_code),
    ("card_number", "Card Number", lambda b: b.printing.number),
    ("base_set_size", "Base Set Size", lambda b: b.printing.base_set_size),
    ("artist", "Artist", lambda b: b.printing.artist),
    ("scryfall_url", "Scryfall URL", lambda b: b.printing.image_url),
]


def _best_printing_fields(result: UploadResult, as_columns: bool) -> dict[str, object]:
    """The exported fields of the result's best printing (all None without a match).

    Keyed by JSON key or column name.
    """
    best = result.best
    return {
        (column if as_columns else key): (get(best) if best is not None else None)
        for key, column, get in _BEST_PRINTING_FIELDS
    }


class UploadProcessor:
    """Recognize uploaded card images one at a time and export the results."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def process(
        self,
        uploads: list[Upload],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[UploadResult]:
        """
        Recognize each uploaded image.

        Args:
            uploads: The uploaded image files.
            progress_callback: Optional callback(images_done, total).

        Returns:
            UploadResults in input order: one per card found, or one per image
            that has no card or failed.
        """
        results: list[UploadResult] = []
        total = len(uploads)

        for done, upload in enumerate(uploads, 1):
            results.extend(self._process_one(upload))
            if progress_callback:
                progress_callback(done, total)

        return results

    def _process_one(self, upload: Upload) -> list[UploadResult]:
        name, data = upload.name, upload.data
        recognition: RecognitionResult = self.pipeline.recognize(data, debug=True)

        if recognition.error is not None:
            return [UploadResult(name, data, error_message=recognition.error)]
        if not recognition.segments:
            return [UploadResult(name, data)]

        return [
            UploadResult(
                name,
                data,
                ranked=segment.ranked,
                warped_image_jpeg=_encode_jpeg(segment.warped_image),
                reading_debug=(
                    segment.reading.debug if segment.reading is not None else None
                ),
            )
            for segment in recognition.segments
        ]

    def get_result_dataframe(self, results: list[UploadResult]):
        """
        Convert upload results to pandas DataFrame.

        Args:
            results: List of UploadResult objects

        Returns:
            pandas DataFrame with results
        """
        import pandas as pd

        data = []
        for result in results:
            row = {
                "Uploaded Filename": result.uploaded_filename,
                "Status": result.status.value,
                **_best_printing_fields(result, as_columns=True),
            }
            data.append(row)

        return pd.DataFrame(data)

    def export_to_csv(self, results: list[UploadResult], filepath: str) -> None:
        """
        Export results to CSV file.

        Args:
            results: List of UploadResult objects
            filepath: Output file path
        """
        df = self.get_result_dataframe(results)
        df.to_csv(filepath, index=False)

    def export_to_excel(self, results: list[UploadResult], filepath: str) -> None:
        """
        Export results to Excel file with optional formatting.

        Args:
            results: List of UploadResult objects
            filepath: Output file path

        Note:
            Requires openpyxl.
        """
        df = self.get_result_dataframe(results)
        df.to_excel(filepath, index=False, sheet_name="Results", engine="openpyxl")

    def export_to_json(self, results: list[UploadResult], filepath: str) -> None:
        """
        Export results to JSON file.

        Args:
            results: List of UploadResult objects
            filepath: Output file path
        """
        import json

        data = []
        for result in results:
            item = {
                "uploaded_filename": result.uploaded_filename,
                "status": result.status.value,
                **_best_printing_fields(result, as_columns=False),
                "error_message": result.error_message,
            }
            data.append(item)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


def _encode_jpeg(img: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _DISPLAY_JPEG_QUALITY]
    )
    if not ok:
        raise ValueError("Failed to encode the warped card as JPEG")
    return buffer.tobytes()
