"""rembg-based card segmentor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from rembg import new_session, remove
from tqdm import tqdm

from mtg_recognition.core.image_utils import load_and_resize_image

from .seg_utils import _corners_from_binary_mask, valid_aspect_ratio, warp_card


@dataclass
class CardSegment:
    """A single card extracted and perspective-corrected from a source image."""

    warped_image: np.ndarray  # BGR, resized to target_size


@dataclass
class ImageSegmentationResult:
    """Segmentation results for one input image. May contain multiple cards."""

    segments: list[CardSegment] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.segments)


class RembgCardSegmentor:
    """
    Segments Magic: The Gathering cards from images using background removal.

    Pipeline per image:
        rembg mask → binary threshold → connected components →
        per-blob corner detection → aspect ratio validation →
        perspective transform

    Handles multi-card images: each connected foreground blob is treated as a
    separate card candidate. Blobs are filtered by minimum area and aspect ratio.
    The rembg session is loaded once and reused across all calls, making
    segment_batch efficient for processing multiple images.
    """

    def __init__(
        self,
        model_name: str = "silueta",
        target_size: tuple[int, int] = (488, 680),
        min_area_fraction: float = 0.01,
        aspect_ratio_tolerance: float = 0.2,
    ):
        """
        Args:
            model_name: rembg model name. "silueta" is recommended for cards.
            target_size: Output (width, height) of each warped card image.
            min_area_fraction: Blobs smaller than this fraction of the total
                               image area are discarded as noise.
            aspect_ratio_tolerance: Allowed relative deviation from the MTG card
                                    aspect ratio (0.716). 0.2 accepts [0.57, 0.86].
        """
        self._session = new_session(model_name)
        self.target_size = target_size
        self.min_area_fraction = min_area_fraction
        self.aspect_ratio_tolerance = aspect_ratio_tolerance

    def segment(self, img: np.ndarray) -> ImageSegmentationResult:
        """
        Segment all cards from a single image.

        Args:
            img: Input image in BGR format.

        Returns:
            ImageSegmentationResult with one CardSegment per detected card.
            segments is empty if no cards were found.
        """
        mask = remove(
            img,
            session=self._session,
            only_mask=True,
            post_process_mask=True,
            alpha_matting=False,
        )
        _, binary = cv2.threshold(np.array(mask), 127, 255, cv2.THRESH_BINARY)

        min_area = self.min_area_fraction * img.shape[0] * img.shape[1]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        segments = []
        for label in range(1, num_labels):  # label 0 is background
            if stats[label, cv2.CC_STAT_AREA] < min_area:
                continue

            blob_mask = np.where(labels == label, 255, 0).astype(np.uint8)
            corners = _corners_from_binary_mask(blob_mask, erode=True)
            if corners is None:
                continue

            if not valid_aspect_ratio(corners, self.aspect_ratio_tolerance):
                continue

            segments.append(
                CardSegment(warped_image=warp_card(img, corners, self.target_size))
            )

        return ImageSegmentationResult(segments=segments)

    def segment_batch(self, imgs: list[np.ndarray]) -> list[ImageSegmentationResult]:
        """
        Segment cards from multiple images, reusing the rembg session.

        Args:
            imgs: List of BGR images.

        Returns:
            One ImageSegmentationResult per input image, in the same order.
        """
        return [self.segment(img) for img in imgs]

    def segment_from_dir(self, src_dir: str | Path, dst_dir: str | Path) -> None:
        src_dir = Path(src_dir)
        if not src_dir.exists():
            raise ValueError(f"source directory does not exist: {src_dir}")

        dst_dir = Path(dst_dir)
        assert src_dir != dst_dir
        dst_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(src_dir.glob("*.jpg"))
        for img_path in tqdm(image_paths, desc="Segmenting images"):
            img = load_and_resize_image(img_path)
            result = self.segment(img)
            if not result.success:
                print(f"no card detected: {img_path}")
                continue
            for i, segment in enumerate(result.segments):
                stem = (
                    img_path.stem
                    if len(result.segments) == 1
                    else f"{img_path.stem}_{i}"
                )
                cv2.imwrite(str(dst_dir / f"{stem}.jpg"), segment.warped_image)


# # Backward-compatible alias
# CardSegmentor = RembgCardSegmentor
