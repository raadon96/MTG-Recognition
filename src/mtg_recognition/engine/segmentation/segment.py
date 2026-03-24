from dataclasses import dataclass, field
from pathlib import Path

import click
import cv2
import numpy as np
from rembg import new_session, remove
from tqdm import tqdm

from mtg_recognition.core.image_utils import load_and_resize_image

from .geometry import find_corners_lines_intersect, perspective_transform

# MTG standard card dimensions: 63 mm × 88 mm (portrait)
_CARD_ASPECT_RATIO = 63.0 / 88.0  # ≈ 0.716


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


def _corners_from_binary_mask(
    mask: np.ndarray, erode: bool = True
) -> np.ndarray | None:
    """
    Extract 4 card corners from a binary mask.

    Optionally erodes the mask to tighten the boundary away from soft rembg
    edge pixels, then applies polygon approximation with a Hough
    line-intersection fallback.

    Args:
        mask: Binary uint8 mask (0/255), same spatial size as the source image.
        erode: If True, apply dilate x 1 + erode x 3 (net erosion) before contour
               extraction. Recommended for rembg masks which tend to be bloated.

    Returns:
        np.ndarray of shape (4, 1, 2), or None if 4 corners could not be found.
    """
    if erode:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True)

    if len(corners) == 4:
        return corners

    corners = find_corners_lines_intersect(contour)
    if corners is not None:
        return corners

    return None


class CardSegmentor:
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

            if not self._valid_aspect_ratio(corners):
                continue

            segments.append(CardSegment(warped_image=self._warp_card(img, corners)))

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

    def _warp_card(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        warped = perspective_transform(img, corners)
        h, w = warped.shape[:2]
        if w > h:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        return cv2.resize(warped, self.target_size, interpolation=cv2.INTER_LANCZOS4)

    def _valid_aspect_ratio(self, corners: np.ndarray) -> bool:
        pts = corners.reshape(4, 2).astype(float)
        sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        # Opposite side pairs from the contour-ordered quad: (0,2) and (1,3)
        pair_a = (sides[0] + sides[2]) / 2
        pair_b = (sides[1] + sides[3]) / 2
        if max(pair_a, pair_b) < 1e-6:
            return False
        ratio = min(pair_a, pair_b) / max(pair_a, pair_b)
        lo = _CARD_ASPECT_RATIO * (1 - self.aspect_ratio_tolerance)
        hi = _CARD_ASPECT_RATIO * (1 + self.aspect_ratio_tolerance)
        return bool(lo <= ratio <= hi)


@click.command()
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    default="benchmarks/dataset/images/raw",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default="output/engine/segmentation/CardSegmentor",
)
@click.option("-m", "--model-name", type=str, default="silueta")
def main(input: Path, output_dir: Path, model_name: str):
    segmentor = CardSegmentor(model_name=model_name)
    if input.is_file():
        img = load_and_resize_image(input)
        result = segmentor.segment(img)
        if result.success:
            print(f"Found {len(result.segments)} card(s)")
            for i, segment in enumerate(result.segments):
                cv2.imshow(f"Card {i}", segment.warped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No card detected")
    else:
        segmentor.segment_from_dir(input, output_dir)


if __name__ == "__main__":
    main()

# uv run python -m mtg_recognition.engine.segmentation.segment
# uv run python -m mtg_recognition.engine.segmentation.segment -i benchmarks/dataset/images/raw -o output/engine/segmentation/CardSegmentor
