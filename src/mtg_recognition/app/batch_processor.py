"""Batch image processing for MTG card recognition."""

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mtg_recognition.catalog import Printing
from mtg_recognition.core.image_utils import load_and_resize_image
from mtg_recognition.engine.ocr.ocr import OCRDebugData
from mtg_recognition.engine.pipeline import Pipeline

# Pipeline retrieval parameters shared between batch processing and the detail view.
# Keep these in one place so both code paths always agree.
K_PHASH_CANDIDATES = 50  # pHash candidates passed to OCR refinement
K_TOP_MATCHES = 5  # final matches returned after OCR
_CHUNK_SIZE = 20  # images loaded and matched per chunk


@dataclass
class BatchResult:
    """Result of batch processing a single image."""

    uploaded_filename: str
    original_image_path: str
    matches: list[Printing] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    status: str = "no_match"  # "success", "no_match", "error"
    error_message: str | None = None
    warped_image_temp_path: str | None = None  # Temp file path for warped image
    original_image_temp_path: str | None = None  # Temp file path for original image
    ocr_debug: OCRDebugData | None = None


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
        Process a batch of images in memory-bounded mini-batches.

        Images are loaded, segmented, and matched in chunks of
        ``_CHUNK_SIZE`` so that peak memory usage is
        O(_CHUNK_SIZE × image_size) regardless of how many images are uploaded.

        Within each mini-batch:
          1. Load & resize images (longest edge ≤ MAX_IMAGE_DIMENSION).
          2. Run YOLO batch segmentation.
          3. Run pHash + OCR matching per segment.

        Results are emitted in the same order as ``image_paths``, with errors
        appearing inline at their original position.

        Args:
            image_paths: List of file paths to process.
            progress_callback: Optional callback(current_index, total, message)
                               where current_index counts images processed so far.

        Returns:
            List of BatchResult objects (one per detected card segment).
        """
        if not image_paths:
            return []

        total = len(image_paths)
        results: list[BatchResult] = []
        processed = 0

        for batch_start in range(0, total, _CHUNK_SIZE):
            batch_paths = image_paths[batch_start : batch_start + _CHUNK_SIZE]

            # -- Stage A: load mini-batch --------------------------------------
            loaded: list[tuple[str, np.ndarray | None, str | None, str | None]] = []
            # (image_path, img_array, original_temp_path, error)

            for image_path in batch_paths:
                image_file = Path(image_path)
                try:
                    img = load_and_resize_image(image_file)
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as tmp:
                        cv2.imwrite(tmp.name, img)
                        orig_tmp = tmp.name
                    loaded.append((image_path, img, orig_tmp, None))
                except Exception as e:
                    loaded.append((image_path, None, None, str(e)))

            # -- Stage B: batch segmentation -----------------------------------
            valid = [(p, img, orig) for p, img, orig, err in loaded if img is not None]
            imgs_to_segment = [img for _, img, _ in valid]

            seg_results = [self.pipeline.segmentor.segment(img) for img in imgs_to_segment]

            # -- Stage C: emit results in input order --------------------------
            seg_idx = 0
            for image_path, img, orig_tmp, err in loaded:
                image_file = Path(image_path)
                processed += 1
                if progress_callback:
                    progress_callback(processed, total, f"Matching {image_file.name}")

                if err is not None:
                    results.append(
                        BatchResult(
                            uploaded_filename=image_file.name,
                            original_image_path=str(image_path),
                            status="error",
                            error_message=err,
                        )
                    )
                    continue

                seg_result = seg_results[seg_idx]
                seg_idx += 1

                if not seg_result.success:
                    results.append(
                        BatchResult(
                            uploaded_filename=image_file.name,
                            original_image_path=str(image_path),
                            status="no_match",
                            original_image_temp_path=orig_tmp,
                        )
                    )
                    continue

                try:
                    results.extend(
                        self._match_segments(
                            seg_result.segments,
                            image_file,
                            orig_tmp,
                        )
                    )
                except Exception as e:
                    results.append(
                        BatchResult(
                            uploaded_filename=image_file.name,
                            original_image_path=str(image_path),
                            status="error",
                            error_message=str(e),
                            original_image_temp_path=orig_tmp,
                        )
                    )

        if progress_callback:
            progress_callback(total, total, "Processing complete!")

        return results

    def _match_segments(
        self,
        segments,
        image_file: Path,
        original_img_path: str | None,
    ) -> list[BatchResult]:
        """Run pHash + OCR matching for each card segment from one image."""
        results = []
        for segment in segments:
            warped_img = segment.warped_image
            warped_img_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as tmp_file:
                    cv2.imwrite(tmp_file.name, warped_img)
                    warped_img_path = tmp_file.name

                seg_results = self.pipeline.find_best_matches(
                    warped_img,
                    k_phash_candidates=K_PHASH_CANDIDATES,
                    k_final_results=K_TOP_MATCHES,
                    apply_segmentation=False,
                    debug=True,
                )
                seg = seg_results[0]

                results.append(
                    BatchResult(
                        uploaded_filename=image_file.name,
                        original_image_path=str(image_file),
                        matches=seg.matches,
                        scores=[float(s) for s in seg.scores],
                        status="success" if seg.matches else "no_match",
                        original_image_temp_path=original_img_path,
                        warped_image_temp_path=warped_img_path,
                        ocr_debug=seg.ocr_debug,
                    )
                )
            except Exception:
                if warped_img_path and Path(warped_img_path).exists():
                    Path(warped_img_path).unlink()
                raise

        return results

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
