"""Batch image processing for MTG card recognition."""

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from sqlalchemy.engine.row import Row
from sqlmodel import Session

from mtg_recognition.engine.pipeline import Pipeline


@dataclass
class BatchResult:
    """Result of batch processing a single image."""

    uploaded_filename: str
    original_image_path: str
    warped_image: np.ndarray | None
    best_match: Row | None
    best_match_score: float | None
    status: str  # "success", "no_match", "error"
    error_message: str | None = None
    warped_image_temp_path: str | None = None  # Temp file path for warped image
    original_image_temp_path: str | None = None  # Temp file path for original image
    timings_ms: dict[str, float] | None = None


class BatchProcessor:
    """Process multiple card images in batch mode."""

    def __init__(
        self,
        pipeline: Pipeline,
        engine,
    ):
        """
        Initialize batch processor.

        Args:
            pipeline: Pipeline instance for card recognition
            engine: SQLAlchemy engine for database access
        """
        self.pipeline = pipeline
        self.engine = engine

    def process_batch(
        self,
        image_paths: list[str],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[BatchResult]:
        """
        Process a batch of images.

        Args:
            image_paths: List of file paths to process
            progress_callback: Optional callback for progress updates.
                              Called with (current_index, total, status_message)

        Returns:
            List of BatchResult objects
        """
        results = []

        for idx, image_path in enumerate(image_paths):
            if progress_callback:
                progress_callback(
                    idx, len(image_paths), f"Processing {Path(image_path).name}"
                )

            try:
                results.extend(self._process_single_image(image_path))
            except Exception as e:
                results.append(BatchResult(
                    uploaded_filename=Path(image_path).name,
                    original_image_path=str(image_path),
                    warped_image=None,
                    best_match=None,
                    best_match_score=None,
                    status="error",
                    error_message=str(e),
                ))

        if progress_callback:
            progress_callback(
                len(image_paths), len(image_paths), "Processing complete!"
            )

        return results

    def _process_single_image(self, image_path: str) -> list[BatchResult]:
        """
        Process a single image file. Returns one BatchResult per detected card.

        For multi-card images each detected card becomes its own BatchResult,
        all sharing the same uploaded_filename and original_image_temp_path.

        Args:
            image_path: Path to image file.

        Returns:
            List of BatchResult objects. Contains a single no_match result if
            no cards were detected, otherwise one result per card.
        """
        image_file = Path(image_path)

        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")

        input_img = cv2.imread(str(image_file))
        if input_img is None:
            raise ValueError(f"Failed to read image: {image_file}")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, input_img)
            original_img_path = tmp_file.name

        segmentation_start = time.perf_counter()
        segmentation_result = self.pipeline.segment_card(input_img)
        segmentation_ms = (time.perf_counter() - segmentation_start) * 1000.0

        if not segmentation_result.success:
            return [BatchResult(
                uploaded_filename=image_file.name,
                original_image_path=str(image_file),
                warped_image=None,
                best_match=None,
                best_match_score=None,
                status="no_match",
                original_image_temp_path=original_img_path,
                timings_ms={
                    "segmentation_ms": segmentation_ms,
                    "phash_ms": 0.0,
                    "ocr_ms": 0.0,
                    "total_ms": segmentation_ms,
                },
            )]

        results = []
        for segment in segmentation_result.segments:
            warped_img = segment.warped_image
            warped_img_path = None

            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    cv2.imwrite(tmp_file.name, warped_img)
                    warped_img_path = tmp_file.name

                matches, scores, diagnostics = (
                    self.pipeline.find_best_matches_with_diagnostics(
                        warped_img,
                        k_phash_candidates=100,
                        k_final_results=5,
                        apply_segmentation=False,
                    )
                )

                timings_ms = {
                    "segmentation_ms": segmentation_ms,
                    "phash_ms": diagnostics["phash_ms"],
                    "ocr_ms": diagnostics["ocr_ms"],
                    "total_ms": segmentation_ms + diagnostics["total_ms"],
                }

                if matches:
                    best_match = matches[0]
                    best_score = float(scores[0])
                    status = "success"
                else:
                    best_match = None
                    best_score = None
                    status = "no_match"

                results.append(BatchResult(
                    uploaded_filename=image_file.name,
                    original_image_path=str(image_file),
                    warped_image=None,
                    best_match=best_match,
                    best_match_score=best_score,
                    status=status,
                    original_image_temp_path=original_img_path,
                    warped_image_temp_path=warped_img_path,
                    timings_ms=timings_ms,
                ))
            except Exception:
                if warped_img_path and Path(warped_img_path).exists():
                    Path(warped_img_path).unlink()
                raise

        return results

    def get_result_dataframe(
        self, results: list[BatchResult], with_diagnostics: bool = False
    ):
        """
        Convert batch results to pandas DataFrame.

        Args:
            results: List of BatchResult objects

        Returns:
            pandas DataFrame with results
        """
        import pandas as pd

        data = []
        with Session(self.engine) as session:
            for result in results:
                row = {
                    "Uploaded Filename": result.uploaded_filename,
                    "Status": result.status,
                    "Card Name": result.best_match.name if result.best_match else None,
                    "UUID": result.best_match.uuid if result.best_match else None,
                    "Score": result.best_match_score,
                    "Set Code": result.best_match.setCode
                    if result.best_match
                    else None,
                    "Card Number": result.best_match.number
                    if result.best_match
                    else None,
                    "Base Set Size": result.best_match.baseSetSize
                    if result.best_match
                    else None,
                    "Artist": result.best_match.artist if result.best_match else None,
                    "Scryfall URL": result.best_match.url
                    if result.best_match
                    else None,
                }
                if with_diagnostics and result.timings_ms:
                    diagnostics_row = {
                        "Segmentation Time (ms)": result.timings_ms.get(
                            "segmentation_ms", None
                        ),
                        "pHash Time (ms)": result.timings_ms.get("phash_ms", None),
                        "OCR Time (ms)": result.timings_ms.get("ocr_ms", None),
                        "Total Time (ms)": result.timings_ms.get("total_ms", None),
                    }
                    row.update(diagnostics_row)
                data.append(row)

        return pd.DataFrame(data)

    def export_to_csv(
        self, results: list[BatchResult], filepath: str, with_diagnostics: bool = False
    ) -> None:
        """
        Export results to CSV file.

        Args:
            results: List of BatchResult objects
            filepath: Output file path
        """
        df = self.get_result_dataframe(results, with_diagnostics=with_diagnostics)
        df.to_csv(filepath, index=False)

    def export_to_excel(
        self, results: list[BatchResult], filepath: str, with_diagnostics: bool = False
    ) -> None:
        """
        Export results to Excel file with optional formatting.

        Args:
            results: List of BatchResult objects
            filepath: Output file path

        Note:
            Requires openpyxl.
        """
        df = self.get_result_dataframe(results, with_diagnostics=with_diagnostics)
        df.to_excel(filepath, index=False, sheet_name="Results", engine="openpyxl")

    def export_to_json(
        self, results: list[BatchResult], filepath: str, with_diagnostics: bool = False
    ) -> None:
        """
        Export results to JSON file.

        Args:
            results: List of BatchResult objects
            filepath: Output file path
        """
        import json

        data = []
        for result in results:
            item = {
                "uploaded_filename": result.uploaded_filename,
                "status": result.status,
                "card_name": result.best_match.name if result.best_match else None,
                "uuid": result.best_match.uuid if result.best_match else None,
                "score": result.best_match_score,
                "set_code": result.best_match.setCode if result.best_match else None,
                "card_number": result.best_match.number if result.best_match else None,
                "base_set_size": result.best_match.baseSetSize
                if result.best_match
                else None,
                "artist": result.best_match.artist if result.best_match else None,
                "scryfall_url": result.best_match.url if result.best_match else None,
                "error_message": result.error_message,
            }
            if with_diagnostics and result.timings_ms:
                item.update(
                    {
                        # "timings_ms": result.timings_ms,
                        "segmentation_ms": result.timings_ms.get(
                            "segmentation_ms", None
                        ),
                        "phash_ms": result.timings_ms.get("phash_ms", None),
                        "ocr_ms": result.timings_ms.get("ocr_ms", None),
                        "total_ms": result.timings_ms.get("total_ms", None),
                    }
                )
            data.append(item)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
