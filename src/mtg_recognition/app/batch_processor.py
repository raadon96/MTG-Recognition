"""Batch image processing for MTG card recognition."""

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

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


class UploadedImageLike(Protocol):
    """Minimal interface required from uploaded file objects."""

    name: str

    def getbuffer(self) -> memoryview: ...


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

    def process_uploaded_batch(
        self,
        uploaded_files: Sequence[UploadedImageLike],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[BatchResult]:
        """Process a batch of uploaded file-like objects.

        Args:
            uploaded_files: List of uploaded file-like objects.
            progress_callback: Optional callback for progress updates.
                              Called with (current_index, total, status_message)

        Returns:
            List of BatchResult objects
        """
        results = []

        for idx, uploaded_file in enumerate(uploaded_files):
            if progress_callback:
                progress_callback(
                    idx,
                    len(uploaded_files),
                    f"Processing {Path(uploaded_file.name).name}",
                )

            try:
                result = self._process_single_uploaded_file(uploaded_file)
                results.append(result)
            except Exception as e:
                result = BatchResult(
                    uploaded_filename=Path(uploaded_file.name).name,
                    original_image_path=uploaded_file.name,
                    warped_image=None,
                    best_match=None,
                    best_match_score=None,
                    status="error",
                    error_message=str(e),
                )
                results.append(result)

        if progress_callback:
            progress_callback(
                len(uploaded_files),
                len(uploaded_files),
                "Processing complete!",
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
        image_file = Path(image_path)

        # Read image
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")

        input_img = cv2.imread(str(image_file))
        if input_img is None:
            raise ValueError(f"Failed to read image: {image_file}")

        return self._process_decoded_image(
            input_img=input_img,
            uploaded_filename=image_file.name,
            original_image_path=str(image_file),
        )

    def _process_single_uploaded_file(
        self,
        uploaded_file: UploadedImageLike,
    ) -> BatchResult:
        """Process a single uploaded file directly from its in-memory buffer."""
        image_array = np.frombuffer(uploaded_file.getbuffer(), dtype=np.uint8)
        input_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if input_img is None:
            raise ValueError(f"Failed to decode uploaded image: {uploaded_file.name}")

        return self._process_decoded_image(
            input_img=input_img,
            uploaded_filename=Path(uploaded_file.name).name,
            original_image_path=uploaded_file.name,
        )

    def _process_decoded_image(
        self,
        input_img: np.ndarray,
        uploaded_filename: str,
        original_image_path: str,
    ) -> BatchResult:
        """Run segmentation and matching for an already decoded image."""

        # Save original image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, input_img)
            original_img_path = tmp_file.name

        # Segment card
        segmentation_start = time.perf_counter()
        segmentation_result = self.pipeline.segment_card(
            input_img, return_annotated=False, return_warped=True
        )
        segmentation_ms = (time.perf_counter() - segmentation_start) * 1000.0

        if not segmentation_result.success or segmentation_result.warped_image is None:
            # No card detected
            return BatchResult(
                uploaded_filename=uploaded_filename,
                original_image_path=original_image_path,
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
            )

        # Save warped image to temp file (reload from temp files as requested)
        warped_img = segmentation_result.warped_image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, warped_img)
            warped_img_path = tmp_file.name

        try:
            # Find best matches
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

            # Create result with best match
            if matches is None or len(matches) == 0:
                best_match = None
                best_score = None
                status = "no_match"
            else:
                best_match = matches[0]
                best_score = float(scores[0])
                status = "success"

            return BatchResult(
                uploaded_filename=uploaded_filename,
                original_image_path=original_image_path,
                warped_image=None,  # Don't keep in memory
                best_match=best_match,
                best_match_score=best_score,
                status=status,
                original_image_temp_path=original_img_path,
                warped_image_temp_path=warped_img_path,
                timings_ms=timings_ms,
            )
        except Exception:
            # Clean up temp files on error
            if Path(original_img_path).exists():
                Path(original_img_path).unlink()
            if Path(warped_img_path).exists():
                Path(warped_img_path).unlink()
            raise

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
