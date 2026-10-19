"""rembg-based card segmentor."""

from __future__ import annotations

import cv2
import numpy as np
from rembg import new_session, remove

from .seg_utils import (
    CardSegment,
    ImageSegmentationResult,
    corners_from_binary_mask,
    valid_aspect_ratio,
    warp_card,
)


class RembgCardSegmentor:
    """
    Segments Magic: The Gathering cards from images using background removal.

    Pipeline per image:
        rembg mask → binary threshold → connected components →
        per-blob corner detection → aspect ratio validation →
        perspective transform

    Handles multi-card images: each connected foreground blob is treated as a
    separate card candidate. Blobs are filtered by minimum area and aspect ratio.
    The rembg session is loaded once and reused across all calls.
    """

    def __init__(
        self,
        model_name: str = "silueta",
        min_area_fraction: float = 0.01,
        aspect_ratio_tolerance: float = 0.2,
    ):
        """
        Args:
            model_name: rembg model name. "silueta" is recommended for cards.
            min_area_fraction: Blobs smaller than this fraction of the total
                               image area are discarded as noise.
            aspect_ratio_tolerance: Allowed relative deviation from the MTG card
                                    aspect ratio (0.716). 0.2 accepts [0.57, 0.86].
        """
        self._session = new_session(model_name)
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
            corners = corners_from_binary_mask(blob_mask, erode=True)
            if corners is None:
                continue

            if not valid_aspect_ratio(corners, self.aspect_ratio_tolerance):
                continue

            segments.append(CardSegment(warped_image=warp_card(img, corners)))

        return ImageSegmentationResult(segments=segments)
