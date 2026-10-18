"""Recognition of uploaded card images for the Streamlit app."""

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mtg_recognition.catalog import Printing
from mtg_recognition.core.image_utils import load_image
from mtg_recognition.engine.ocr.ocr import OCRDebugData
from mtg_recognition.engine.pipeline import Pipeline, RecognitionResult


@dataclass
class UploadResult:
    """Result for one card found in an uploaded image, or for an image with no card or an error."""

    uploaded_filename: str
    original_image_path: str
    matches: list[Printing] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    error_message: str | None = None
    warped_image_temp_path: str | None = None  # Temp file path for warped image
    original_image_temp_path: str | None = None  # Temp file path for original image
    ocr_debug: OCRDebugData | None = None

    @property
    def status(self) -> str:
        """One of "success", "no_match" or "error", derived from the recognition result."""
        if self.error_message is not None:
            return "error"
        return "success" if self.matches else "no_match"


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
                scores=segment.scores,
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
            best = result.matches[0] if result.matches else None
            row = {
                "Uploaded Filename": result.uploaded_filename,
                "Status": result.status,
                "Card Name": best.name if best else None,
                "UUID": best.uuid if best else None,
                "Score": result.scores[0] if result.scores else None,
                "Set Code": best.set_code if best else None,
                "Card Number": best.number if best else None,
                "Base Set Size": best.base_set_size if best else None,
                "Artist": best.artist if best else None,
                "Scryfall URL": best.image_url if best else None,
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
            best = result.matches[0] if result.matches else None
            item = {
                "uploaded_filename": result.uploaded_filename,
                "status": result.status,
                "card_name": best.name if best else None,
                "uuid": best.uuid if best else None,
                "score": result.scores[0] if result.scores else None,
                "set_code": best.set_code if best else None,
                "card_number": best.number if best else None,
                "base_set_size": best.base_set_size if best else None,
                "artist": best.artist if best else None,
                "scryfall_url": best.image_url if best else None,
                "error_message": result.error_message,
            }
            data.append(item)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


def _write_temp_png(img: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        return tmp.name
