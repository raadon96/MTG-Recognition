"""
Generate downscaled image fixtures for rembg segmentation regression tests.

Run once from the project root after changing the test dataset or resizing
settings:

    uv run python tests/regression/segmentation/generate_fixtures.py

Produces under tests/regression/segmentation/fixtures/:

  0_failure_cases/
    images/       - downscaled failure-case images (JPEG)
    masks/        - corresponding birefnet-general-lite masks (PNG)
    (no corners - these are all xfail scenarios)

  1_photos_single_card/
    images/       - 20 UUID images + card_and_pen, downscaled (JPEG)
    masks/        - corresponding birefnet-general-lite masks (PNG)
    corners.json  - golden corners (one 4-point set per image, largest blob)

  2_photos_multiple_cards/
    images/       - 2 multi-card images, downscaled (JPEG)
    masks/        - corresponding birefnet-general-lite masks (PNG)
    corners.json  - golden corners per blob, sorted by centroid (top→bottom,
                    left→right); value is a list of 4-point sets
"""

import json
from pathlib import Path

import cv2
import numpy as np

from mtg_recognition.core.image_utils import load_and_resize_image
from mtg_recognition.engine.segmentation.seg_utils import corners_from_binary_mask

SRC_BASE = Path("data/test")
FIXTURE_DIR = Path("tests/regression/segmentation/fixtures")

N_SINGLE_CARD = 20  # number of UUID images to include
MIN_AREA_FRACTION = 0.01  # blobs smaller than this fraction are ignored


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resize_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize a binary mask using INTER_NEAREST to preserve 0/255 values."""
    resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return binary


def _sorted_blobs(mask: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    """
    Return per-blob binary masks sorted by centroid (top-to-bottom, left-to-right).

    Each element is (cy, cx, blob_mask).  Blobs smaller than MIN_AREA_FRACTION
    of the image area are discarded.
    """
    min_area = MIN_AREA_FRACTION * mask.shape[0] * mask.shape[1]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    blobs = []
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            continue
        blob_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        cx, cy = centroids[label]
        blobs.append((cy, cx, blob_mask))
    blobs.sort(key=lambda b: (b[0], b[1]))
    return blobs


def _single_corners(mask: np.ndarray) -> list[list[int]] | None:
    """Corners from the largest contour in mask (single-card case)."""
    corners = corners_from_binary_mask(mask, erode=True)
    if corners is None:
        return None
    return corners.reshape(4, 2).tolist()


def _per_blob_corners(mask: np.ndarray) -> list[list[list[int]]]:
    """Corners per blob, sorted by centroid. Returns [] if any blob yields no
    corners.
    """
    result = []
    for _, _, blob_mask in _sorted_blobs(mask):
        corners = corners_from_binary_mask(blob_mask, erode=True)
        if corners is None:
            return []
        result.append(corners.reshape(4, 2).tolist())
    return result


# ---------------------------------------------------------------------------
# Image-set processor
# ---------------------------------------------------------------------------


def _process(
    src_images: Path,
    src_masks: Path,
    fixture_subdir: str,
    stems: list[str],
    *,
    corners_mode: str = "none",  # "single" | "per_blob" | "none"
) -> None:
    """
    Downscale images and masks from src_* into
    tests/regression/segmentation/fixtures/<fixture_subdir>/.

    Args:
        corners_mode:
            "single"   - one 4-point set per image (largest blob)
            "per_blob" - list of 4-point sets, one per blob (sorted by centroid)
            "none"     - do not produce corners.json
    """
    out = FIXTURE_DIR / fixture_subdir
    img_out = out / "images"
    mask_out = out / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    corners_data: dict = {}
    skipped: list[str] = []

    for stem in stems:
        img_path = next(
            (
                src_images / f"{stem}{ext}"
                for ext in (".jpg", ".jpeg")
                if (src_images / f"{stem}{ext}").exists()
            ),
            None,
        )
        if img_path is None:
            print(f"  SKIP {stem}: image not found")
            skipped.append(stem)
            continue

        mask_path = src_masks / f"{stem}.png"
        if not mask_path.exists():
            print(f"  SKIP {stem}: no reference mask")
            skipped.append(stem)
            continue

        img = load_and_resize_image(img_path)
        h, w = img.shape[:2]

        ref_mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        ref_mask = _resize_mask(ref_mask_full, w, h)

        # Always save as .jpg for consistency
        cv2.imwrite(str(img_out / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(mask_out / f"{stem}.png"), ref_mask)

        if corners_mode == "single":
            corners = _single_corners(ref_mask)
            if corners is not None:
                corners_data[stem] = corners
                print(f"  OK  {stem}  {w}x{h}")
            else:
                print(f"  WARN {stem}: no corners found")
                skipped.append(stem)

        elif corners_mode == "per_blob":
            blobs = _per_blob_corners(ref_mask)
            if blobs:
                corners_data[stem] = blobs
                print(f"  OK  {stem}  {w}x{h}  ({len(blobs)} blobs)")
            else:
                print(f"  WARN {stem}: no per-blob corners found")
                skipped.append(stem)

        else:
            print(f"  OK  {stem}  {w}x{h}")

    if corners_mode != "none":
        corners_path = out / "corners.json"
        with open(corners_path, "w") as f:
            json.dump(corners_data, f, indent=2)
        print(f"  corners: {len(corners_data)} entries in corners.json")

    print(f"  images : {len(list(img_out.glob('*.jpg')))}")
    print(f"  masks  : {len(list(mask_out.glob('*.png')))}")
    if skipped:
        print(f"  skipped: {skipped}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # --- 0_failure_cases ---
    print("\n=== 0_failure_cases ===")
    src0 = SRC_BASE / "0_failure_cases"
    failure_stems = [p.stem for p in sorted((src0 / "images").glob("*.jpg"))]
    _process(
        src0 / "images",
        src0 / "masks" / "birefnet-general-lite",
        "0_failure_cases",
        failure_stems,
        corners_mode="none",
    )

    # --- 1_photos_single_card ---
    print("\n=== 1_photos_single_card ===")
    src1 = SRC_BASE / "1_photos_single_card"
    uuid_stems = [
        p.stem
        for p in sorted((src1 / "images").glob("*.jpg"))
        if p.stem != "card_and_pen"
    ][:N_SINGLE_CARD]
    single_stems = uuid_stems + ["card_and_pen"]
    _process(
        src1 / "images",
        src1 / "masks" / "birefnet-general-lite",
        "1_photos_single_card",
        single_stems,
        corners_mode="single",
    )

    # --- 2_photos_multiple_cards ---
    print("\n=== 2_photos_multiple_cards ===")
    src2 = SRC_BASE / "2_photos_multiple_cards"
    multi_stems = [p.stem for p in sorted((src2 / "images").glob("*.jpg"))]
    _process(
        src2 / "images",
        src2 / "masks" / "birefnet-general-lite",
        "2_photos_multiple_cards",
        multi_stems,
        corners_mode="per_blob",
    )


if __name__ == "__main__":
    main()
