from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from random import choices

from dotenv import load_dotenv

load_dotenv()

import click
import cv2
import numpy as np
from rembg import new_session, remove
from tqdm import tqdm

from .geometry import (
    crop_img,
    find_corners_lines_intersect,
    perspective_transform,
)
from .yolomodel import YOLOModel


@dataclass
class SegmentationResult:
    """Result of card segmentation operation."""

    corners: np.ndarray | None
    warped_image: np.ndarray | None
    annotated_image: np.ndarray | None
    success: bool


class SegmentStrategy(ABC):
    """Abstract base for card corner detection strategies."""

    @abstractmethod
    def detect_corners(self, img: np.ndarray) -> np.ndarray | None:
        """
        Detect the 4 card corners from an image.

        Returns:
            np.ndarray of shape (4, 1, 2), or None if detection failed.
        """
        ...


def _corners_from_binary_mask(
    mask: np.ndarray, erode: bool = True
) -> np.ndarray | None:
    """
    Extract 4 card corners from a binary mask bitmap.

    Optionally erodes the mask before finding the contour to tighten the
    boundary away from residual background pixels, then applies polygon
    approximation and line-intersection refinement.

    Args:
        mask: Binary mask (uint8, 0/255) in the same spatial dimensions as the source image.
        erode: Whether to apply morphological erosion before corner extraction.

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


def _corners_from_img(img: np.ndarray) -> np.ndarray | None:
    """Extract card corners from an image via Canny edge detection and contour approximation."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 15, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    mask_binary = np.zeros_like(edges)
    mask_binary = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(mask_binary, True)
    corners = cv2.approxPolyDP(mask_binary, epsilon, True)

    if len(corners) == 4:
        return corners

    corners = find_corners_lines_intersect(contour)
    if corners is not None:
        return corners

    return None


class MaskStrategy(SegmentStrategy):
    """
    Corner detection via segmentation mask.

    Pipeline: img → YOLO segmentation → mask contour → 4 corners.
    Best for clean backgrounds with a single card in frame.
    """

    def __init__(
        self,
        model_name: str = "yoloe-11s-seg",
        param_set_idx: int = 0,
        erode: bool = True,
    ):
        self.model = YOLOModel(model_name, parameter_set_idx=param_set_idx)
        self.erode = erode

    def detect_corners(self, img: np.ndarray) -> np.ndarray | None:
        mask_contour = self.model.predict(img)
        if mask_contour is None:
            return None

        h, w = img.shape[:2]
        mask_binary = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_binary, [mask_contour], -1, 255, -1)
        return _corners_from_binary_mask(mask_binary, self.erode)


class BBoxStrategy(SegmentStrategy):
    """
    Corner detection via object detection bounding box.

    Pipeline: img → YOLO detection → padded bbox crop → Canny edges → 4 corners.
    Robust to cluttered scenes with multiple objects in frame.

    An optional fallback strategy can be provided for cases where the Canny-based
    approach fails. The fallback receives the bbox crop (not the full image), and its
    corners are mapped back to full-image coordinates automatically.

    Example:
        BBoxStrategy(fallback=RembgMaskStrategy())
    """

    def __init__(
        self,
        model_name: str = "yolov8s-worldv2",
        param_set_idx: int = 0,
        padding_pct: float = 0.1,
        blur_kernel: tuple[int, int] = (11, 11),
        canny_thresh: tuple[int, int] = (15, 150),
        fallback: SegmentStrategy | None = None,
    ):
        self.model = YOLOModel(model_name, parameter_set_idx=param_set_idx)
        self.padding_pct = padding_pct
        self.blur_kernel = blur_kernel
        self.canny_thresh = canny_thresh
        self.fallback = fallback or RembgMaskStrategy()

    def detect_corners(self, img: np.ndarray) -> np.ndarray | None:
        bbox = self.model.predict(img)
        if bbox is None:
            return None

        crop, (cx1, cy1, _, _) = crop_img(img, bbox, self.padding_pct)

        # corners = self._corners_from_crop(crop)
        # if corners is None and self.fallback is not None:
        #     corners = self.fallback.detect_corners(crop)
        # if corners is None:
        #     return None

        corners = None
        if self.fallback is not None:
            corners = self.fallback.detect_corners(crop)

        if corners is None:
            corners = _corners_from_img(crop)

        if corners is None:
            return None

        return corners + [cx1, cy1]


class RembgMaskStrategy(SegmentStrategy):
    """
    Corner detection via background removal (rembg u2netp).

    Pipeline: img → rembg u2netp → alpha mask → 4 corners.
    Slower but robust fallback when edge-based methods fail.

    Can be used standalone or as the fallback in BBoxStrategy:
        BBoxStrategy(fallback=RembgMaskStrategy())
    When used as a fallback, it receives the bbox crop rather than the full image.
    """

    def __init__(self, model_name: str = "u2netp"):

        self._session = new_session(model_name=model_name)

    def detect_corners(self, img: np.ndarray) -> np.ndarray | None:
        mask = remove(
            img,
            alpha_matting=False,
            only_mask=True,
            post_process_mask=True,
            session=self._session,
        )
        mask = np.array(mask)
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return _corners_from_binary_mask(mask_binary, erode=False)


class CardSegmentor:
    """
    Segments a Magic: The Gathering card from an image.

    Delegates corner detection to a SegmentStrategy, then applies a
    perspective transform to produce a canonical card image.
    """

    def __init__(
        self,
        strategy: SegmentStrategy | None = None,
        target_size: tuple[int, int] = (488, 680),
    ):
        self.strategy = strategy or MaskStrategy()
        self.target_size = target_size

    def segment(
        self,
        img: np.ndarray,
        return_annotated: bool = False,
        return_warped: bool = False,
    ) -> SegmentationResult:
        """
        Detect card corners and optionally warp/annotate the image.

        Args:
            img: Input image in BGR format
            return_annotated: Include annotated image with drawn corners
            return_warped: Include perspective-corrected card image

        Returns:
            SegmentationResult with corners, optional images, and success flag
        """
        corners = self.strategy.detect_corners(img)
        if corners is None:
            return SegmentationResult(
                corners=None,
                warped_image=None,
                annotated_image=img if return_annotated else None,
                success=False,
            )

        return SegmentationResult(
            corners=corners,
            warped_image=self._warp_card(img, corners) if return_warped else None,
            annotated_image=self._annotate_corners(img, corners)
            if return_annotated
            else None,
            success=True,
        )

    def _warp_card(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        img_warped = perspective_transform(img, corners)
        h, w = img_warped.shape[:2]
        if w > h:
            img_warped = cv2.rotate(img_warped, cv2.ROTATE_90_CLOCKWISE)
        return cv2.resize(
            img_warped, self.target_size, interpolation=cv2.INTER_LANCZOS4
        )

    def _annotate_corners(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        annotated = img.copy()
        cv2.drawContours(annotated, [corners], -1, (0, 255, 0), 2)
        return annotated

    def segment_from_dir(self, src_dir: str | Path, dst_dir: str | Path) -> None:
        src_dir = Path(src_dir)
        if not src_dir.exists():
            raise ValueError(f"query dir {src_dir} does not exist")

        dst_dir = Path(dst_dir)
        assert src_dir != dst_dir

        dst_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(src_dir.glob("*.jpg"))
        for img_path in tqdm(
            image_paths, total=len(image_paths), desc="Segmenting images"
        ):
            img = cv2.imread(str(img_path))
            result = self.segment(img, return_warped=True)
            if result.warped_image is None:
                print(f"image {img_path} could not be segmented")
                continue
            cv2.imwrite(str(dst_dir / img_path.name), result.warped_image)


_STRATEGY_MAP: dict[str, type[SegmentStrategy]] = {
    "yoloe-11s-seg": MaskStrategy,
    "yolov8s-worldv2": BBoxStrategy,
}


@click.command()
@click.option(
    "-i",
    "--input",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    default="benchmarks/dataset/images/raw",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    default="output/engine/segmentation/CardSegmentor",
)
@click.option(
    "-m",
    "--model-name",
    type=click.Choice(list(_STRATEGY_MAP.keys())),
    default="yoloe-11s-seg",
)
def main(input, output_dir, model_name):
    segmentor = CardSegmentor(_STRATEGY_MAP[model_name]())
    if input.is_file():
        img = cv2.imread(str(input))
        result = segmentor.segment(img, return_warped=True)
        if result.warped_image is not None:
            print("Success!")
            cv2.imshow("Warped", result.warped_image)
            key = cv2.waitKey(0)
            if key == ord("q"):
                cv2.destroyAllWindows()
        else:
            print("No image detected")
    else:
        segmentor.segment_from_dir(input, output_dir)


if __name__ == "__main__":
    main()

# uv run python -m mtg_recognition.engine.segmentation.segment
# uv run python -m mtg_recognition.engine.segmentation.segment -i benchmarks/dataset/images/raw -o output/engine/segmentation/CardSegmentor
