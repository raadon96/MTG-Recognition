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

        Segmentation is performed in one batched YOLO call (mini-batches of 20)
        rather than per-image, which significantly reduces inference time when
        multiple images are provided.

        Args:
            image_paths: List of file paths to process.
            progress_callback: Optional callback(current_index, total, message).

        Returns:
            List of BatchResult objects (one per detected card segment).
        """
        if not image_paths:
            return []

        total = len(image_paths)

        # -- Stage A: load all images --------------------------------------
        if progress_callback:
            progress_callback(0, total, "Loading images…")

        loaded: list[tuple[int, str, np.ndarray | None, str | None, str | None]] = []
        # (original_index, image_path, img_array, original_temp_path, error)

        for idx, image_path in enumerate(image_paths):
            image_file = Path(image_path)
            try:
                if not image_file.exists():
                    raise FileNotFoundError(f"Image file not found: {image_file}")
                img = cv2.imread(str(image_file))
                if img is None:
                    raise ValueError(f"Failed to read image: {image_file}")
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    cv2.imwrite(tmp.name, img)
                    orig_tmp = tmp.name
                loaded.append((idx, image_path, img, orig_tmp, None))
            except Exception as e:
                loaded.append((idx, image_path, None, None, str(e)))

        # -- Stage B: batch segmentation -----------------------------------
        valid_entries = [(i, p, img, orig) for i, p, img, orig, err in loaded if img is not None]
        imgs_to_segment = [img for _, _, img, _ in valid_entries]

        seg_results = []
        if imgs_to_segment:
            if progress_callback:
                progress_callback(0, total, "Segmenting cards…")
            seg_start = time.perf_counter()
            seg_results = self.pipeline.segmentor.segment_batch(imgs_to_segment)
            batch_seg_ms = (time.perf_counter() - seg_start) * 1000.0
            amortized_seg_ms = batch_seg_ms / len(imgs_to_segment)
        else:
            amortized_seg_ms = 0.0

        # -- Stage C: per-segment matching ---------------------------------
        results: list[BatchResult] = []

        # First, emit error results for images that failed to load
        error_by_idx = {
            i: (p, err) for i, p, img, orig, err in loaded if img is None
        }
        for idx, (image_path, err) in sorted(error_by_idx.items()):
            results.append(BatchResult(
                uploaded_filename=Path(image_path).name,
                original_image_path=str(image_path),
                warped_image=None,
                best_match=None,
                best_match_score=None,
                status="error",
                error_message=err,
            ))

        for seg_local_idx, (orig_idx, image_path, img, orig_tmp) in enumerate(valid_entries):
            image_file = Path(image_path)
            if progress_callback:
                progress_callback(seg_local_idx, total, f"Matching {image_file.name}")

            seg_result = seg_results[seg_local_idx]

            if not seg_result.success:
                results.append(BatchResult(
                    uploaded_filename=image_file.name,
                    original_image_path=str(image_file),
                    warped_image=None,
                    best_match=None,
                    best_match_score=None,
                    status="no_match",
                    original_image_temp_path=orig_tmp,
                    timings_ms={
                        "segmentation_ms": amortized_seg_ms,
                        "phash_ms": 0.0,
                        "ocr_ms": 0.0,
                        "total_ms": amortized_seg_ms,
                    },
                ))
                continue

            try:
                results.extend(
                    self._match_segments(
                        seg_result.segments,
                        image_file,
                        orig_tmp,
                        amortized_seg_ms,
                    )
                )
            except Exception as e:
                results.append(BatchResult(
                    uploaded_filename=image_file.name,
                    original_image_path=str(image_file),
                    warped_image=None,
                    best_match=None,
                    best_match_score=None,
                    status="error",
                    error_message=str(e),
                    original_image_temp_path=orig_tmp,
                ))

        if progress_callback:
            progress_callback(total, total, "Processing complete!")

        return results

    def _match_segments(
        self,
        segments,
        image_file: Path,
        original_img_path: str | None,
        segmentation_ms: float,
    ) -> list[BatchResult]:
        """Run pHash + OCR matching for each card segment from one image."""
        results = []
        for segment in segments:
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
