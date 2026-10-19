"""Recognition of uploaded card images for the Streamlit app."""

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

from mtg_recognition.catalog import RankedPrinting
from mtg_recognition.core.image_utils import load_image
from mtg_recognition.engine.ocr.ocr import OCRDebugData
from mtg_recognition.engine.pipeline import Pipeline, RecognitionResult


class UploadStatus(StrEnum):
    SUCCESS = "success"
    NO_MATCH = "no_match"
    ERROR = "error"


@dataclass
class UploadResult:
    """Result for one card found in an uploaded image, or for an image with no card or an error."""

    uploaded_filename: str
    original_image_path: str
    matches: list[RankedPrinting] = field(default_factory=list)  # best first
    error_message: str | None = None
    warped_image_temp_path: str | None = None  # Temp file path for warped image
    original_image_temp_path: str | None = None  # Temp file path for original image
    ocr_debug: OCRDebugData | None = None

    @property
    def status(self) -> UploadStatus:
        """Derived from the recognition result."""
        if self.error_message is not None:
            return UploadStatus.ERROR
        return UploadStatus.SUCCESS if self.matches else UploadStatus.NO_MATCH

    @property
    def best(self) -> RankedPrinting | None:
        """The top-ranked entry, or None if there is no match."""
        return self.matches[0] if self.matches else None


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
    """The exported fields of the result's best printing (all None without a match), by JSON key or column name."""
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
        image_paths: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[UploadResult]:
        """
        Recognize each uploaded image.

        Args:
            image_paths: Paths of the uploaded image files.
            progress_callback: Optional callback(images_done, total).

        Returns:
            UploadResults in input order: one per card found, or one per image
            that has no card or failed.
        """
        results: list[UploadResult] = []
        total = len(image_paths)

        for done, image_path in enumerate(image_paths, 1):
            results.extend(self._process_one(Path(image_path)))
            if progress_callback:
                progress_callback(done, total)

        return results

    def _process_one(self, image_file: Path) -> list[UploadResult]:
        name, path = image_file.name, str(image_file)
        try:
            img = load_image(image_file)
        except Exception as e:
            return [UploadResult(name, path, error_message=str(e))]

        original_tmp = _write_temp_png(img)
        recognition: RecognitionResult = self.pipeline.recognize(img, debug=True)

        if recognition.error is not None:
            return [UploadResult(name, path, error_message=recognition.error, original_image_temp_path=original_tmp)]
        if not recognition.segments:
            return [UploadResult(name, path, original_image_temp_path=original_tmp)]

        return [
            UploadResult(
                name,
                path,
                matches=segment.matches,
                warped_image_temp_path=_write_temp_png(segment.warped_image),
                original_image_temp_path=original_tmp,
                ocr_debug=segment.ocr_debug,
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


def _write_temp_png(img: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        return tmp.name
