from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import click
import cv2
import numpy as np
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
        mask_points = self.model.predict(img)
        if mask_points is None:
            return None
        return self._corners_from_mask(mask_points, img.shape)

    def _corners_from_mask(
        self, mask_points: np.ndarray, img_shape: tuple
    ) -> np.ndarray | None:
        contour = mask_points

        if self.erode:
            h, w = img_shape[:2]
            mask_bitmap = np.zeros((h, w), dtype=np.uint8)
            mask_bitmap = cv2.drawContours(mask_bitmap, [mask_points], -1, 255, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_bitmap = cv2.dilate(mask_bitmap, kernel, iterations=1)
            mask_eroded = cv2.erode(mask_bitmap, kernel, iterations=3)
            contours, _ = cv2.findContours(
                mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour = max(contours, key=cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, epsilon, True)

        if len(corners) != 4 and hasattr(cv2, "approxPolyN"):
            corners = (
                cv2.approxPolyN(contour, 4, True)
                .astype(np.int32)
                .reshape(4, 1, 2)
            )

        if len(corners) != 4:
            corners = find_corners_lines_intersect(contour, corners)

        if len(corners) != 4:
            return None

        return corners


class BBoxStrategy(SegmentStrategy):
    """
    Corner detection via object detection bounding box.

    Pipeline: img → YOLO detection → padded bbox crop → Canny edges → 4 corners.
    Robust to cluttered scenes with multiple objects in frame.
    """

    def __init__(
        self,
        model_name: str = "yolov8s-worldv2",
        param_set_idx: int = 0,
        padding_pct: float = 0.1,
        blur_kernel: tuple[int, int] = (11, 11),
        canny_thresh: tuple[int, int] = (15, 150),
    ):
        self.model = YOLOModel(model_name, parameter_set_idx=param_set_idx)
        self.padding_pct = padding_pct
        self.blur_kernel = blur_kernel
        self.canny_thresh = canny_thresh

    def detect_corners(self, img: np.ndarray) -> np.ndarray | None:
        bbox = self.model.predict(img)
        if bbox is None:
            return None

        crop, offset = crop_img(img, bbox, self.padding_pct)
        corners = self._corners_from_crop(crop)
        if corners is None:
            return None

        cx1, cy1, _, _ = offset
        return corners + [cx1, cy1]

    def _corners_from_crop(self, crop: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        edges = cv2.Canny(blurred, *self.canny_thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        corners = cv2.approxPolyDP(hull, epsilon, True)

        if len(corners) == 4:
            return corners

        if hasattr(cv2, "approxPolyN"):
            corners = (
                cv2.approxPolyN(contour, 4, True)
                .astype(np.int32)
                .reshape(4, 1, 2)
            )

        if len(corners) != 4:
            corners = find_corners_lines_intersect(contour, corners)

        return corners


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
            annotated_image=self._annotate_corners(img, corners) if return_annotated else None,
            success=True,
        )

    def _warp_card(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        img_warped = perspective_transform(img, corners)
        return cv2.resize(img_warped, self.target_size, interpolation=cv2.INTER_LANCZOS4)

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
        for img_path in tqdm(image_paths, total=len(image_paths), desc="Segmenting images"):
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
        exists=True,
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
    type=click.Choice(list(_STRATEGY_MAP)),
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
