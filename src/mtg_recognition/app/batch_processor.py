"""Batch image processing for MTG card recognition."""

import tempfile
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
            # Update progress
            if progress_callback:
                progress_callback(
                    idx, len(image_paths), f"Processing {Path(image_path).name}"
                )

            try:
                result = self._process_single_image(image_path)
                results.append(result)
            except Exception as e:
                # Create error result
                result = BatchResult(
                    uploaded_filename=Path(image_path).name,
                    original_image_path=str(image_path),
                    warped_image=None,
                    best_match=None,
                    best_match_score=None,
                    status="error",
                    error_message=str(e),
                )
                results.append(result)

        # Final progress update
        if progress_callback:
            progress_callback(
                len(image_paths), len(image_paths), "Processing complete!"
            )

        return results

    def _process_single_image(self, image_path: str) -> BatchResult:
        """
        Process a single image file.

        Args:
            image_path: Path to image file

        Returns:
            BatchResult object with segmentation and matching results
        """
        image_path = Path(image_path)

        # Read image
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        input_img = cv2.imread(str(image_path))
        if input_img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Save original image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, input_img)
            original_img_path = tmp_file.name

        # Segment card
        segmentation_result = self.pipeline.segment_card(
            input_img, return_annotated=False, return_warped=True
        )

        if not segmentation_result.success or segmentation_result.warped_image is None:
            # No card detected
            return BatchResult(
                uploaded_filename=image_path.name,
                original_image_path=str(image_path),
                warped_image=None,
                best_match=None,
                best_match_score=None,
                status="no_match",
                original_image_temp_path=original_img_path,
            )

        # Save warped image to temp file (reload from temp files as requested)
        warped_img = segmentation_result.warped_image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, warped_img)
            warped_img_path = tmp_file.name

        try:
            # Find best matches
            matches, scores = self.pipeline.find_best_matches(
                warped_img,
                k_phash_candidates=100,
                k_final_results=5,
                apply_segmentation=False,
            )

            if matches is None or len(matches) == 0:
                # No matches found - still return the warped image path
                return BatchResult(
                    uploaded_filename=image_path.name,
                    original_image_path=str(image_path),
                    warped_image=None,
                    best_match=None,
                    best_match_score=None,
                    status="no_match",
                    original_image_temp_path=original_img_path,
                    warped_image_temp_path=warped_img_path,
                )

            # Create result with best match
            best_match = matches[0]
            best_score = scores[0]

            return BatchResult(
                uploaded_filename=image_path.name,
                original_image_path=str(image_path),
                warped_image=None,  # Don't keep in memory
                best_match=best_match,
                best_match_score=float(best_score),
                status="success",
                original_image_temp_path=original_img_path,
                warped_image_temp_path=warped_img_path,
            )
        except Exception as e:
            # Clean up temp files on error
            if Path(original_img_path).exists():
                Path(original_img_path).unlink()
            if Path(warped_img_path).exists():
                Path(warped_img_path).unlink()
            raise

    def get_result_dataframe(self, results: list[BatchResult]):
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
                data.append(row)

        return pd.DataFrame(data)

    def export_to_csv(self, results: list[BatchResult], filepath: str) -> None:
        """
        Export results to CSV file.

        Args:
            results: List of BatchResult objects
            filepath: Output file path
        """
        df = self.get_result_dataframe(results)
        df.to_csv(filepath, index=False)

    def export_to_excel(self, results: list[BatchResult], filepath: str) -> None:
        """
        Export results to Excel file with optional formatting.

        Args:
            results: List of BatchResult objects
            filepath: Output file path

        Note:
            Requires openpyxl.
        """
        df = self.get_result_dataframe(results)
        df.to_excel(filepath, index=False, sheet_name="Results", engine="openpyxl")

    def export_to_json(self, results: list[BatchResult], filepath: str) -> None:
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
            data.append(item)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
