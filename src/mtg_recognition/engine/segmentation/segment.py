from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
from tqdm import tqdm

from .geometry import (
    crop_img,
    find_corners_lines_intersect,
    perspective_transform,
    sharpen,
)
from .yolomodel import YOLOModel


@dataclass
class SegmentationResult:
    """Result of card segmentation operation."""

    corners: Optional[np.ndarray]
    warped_image: Optional[np.ndarray]
    annotated_image: Optional[np.ndarray]
    success: bool


class CardSegmentor:
    def __init__(
        self,
        model_name: str = "yoloe-11s-seg",
        param_set_idx: int = 0,
        target_size: tuple[int, int] = (488, 680),
        padding_pct: float = 0.1,
        blur_kernel: tuple[int, int] = (11, 11),
        canny_thresh: tuple[int, int] = (15, 150),
    ):
        # Set model
        self.model = YOLOModel(model_name, parameter_set_idx=param_set_idx)

        # Set target size (default is img size of ref MTG-cards stored in ref db)
        self.target_size = target_size

        # Preprocessing attributes is using object detection model (with bbox)
        self.padding_pct = padding_pct
        self.blur_kernel = blur_kernel
        self.canny_thresh = canny_thresh

        # see https://towardsdatascience.com/background-removal-with-python-b61671d1508a/
        self.blur = 21

    def detect_corners(
        self, img: np.ndarray, refine: bool = False
    ) -> Optional[np.ndarray]:
        """
        Detect card corners from an image using the configured model.

        Args:
            img: Input image in BGR format

        Returns:
            np.ndarray of corner points with shape (4, 1, 2), or None if detection failed
        """
        match self.model.task:
            case "segment":
                mask_points = self.model.predict(img)
                if mask_points is None:
                    return None
                return self.find_corners_from_mask(
                    mask_points,
                    img.shape,
                    erode=True,
                    refine=refine,
                )

            case "detect":
                bbox_points = self.model.predict(img)
                if bbox_points is None:
                    return None

                crop, offset = crop_img(img, bbox_points, self.padding_pct)
                corner_points = self.find_corners_from_bbox(crop)
                if corner_points is None:
                    return None

                cx1, cy1, _, _ = offset
                return corner_points + [cx1, cy1]

    def find_corners_from_bbox(
        self, crop: np.ndarray, refine=False
    ) -> np.ndarray | None:
        """
        Detects the largest quadrilateral in the cropped image using detection bbox.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        edges = cv2.Canny(blurred, *self.canny_thresh)

        # Dilate to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Sort by area (descending) to check largest first
        contour_points = max(contours, key=cv2.contourArea)

        hull = cv2.convexHull(contour_points)
        epsilon = 0.02 * cv2.arcLength(hull, True)  # hull
        corner_points = cv2.approxPolyDP(hull, epsilon, True)  # hull

        if len(corner_points) == 4:
            return corner_points

        # Fallback 1: Force 4 corners if approximation failed
        if hasattr(cv2, "approxPolyN"):
            corner_points = (
                cv2.approxPolyN(contour_points, 4, True)
                .astype(np.int32)
                .reshape(4, 1, 2)
            )

        if refine or len(corner_points) != 4:
            corner_points = find_corners_lines_intersect(contour_points, corner_points)

        return corner_points

    def find_corners_from_mask(
        self,
        mask_points: np.ndarray,
        img_shape: tuple,
        erode: bool = True,
        refine: bool = False,
    ) -> np.ndarray | None:
        """
        Approximate 4 corners of a card from a raw contour.

        Parameters
        ----------
        mask_points : np.ndarray
            Raw contour points.
        img_shape : tuple
            Shape of the original image.
        erode : bool, optional
            If True, apply Morphological Erosion to the mask before finding the contour.
            Aims to eliminate potential residual background pixels from the mask.
            Defaults to True.
        refine : bool, optional
            If True, refine the corners to be more precise.
            Aims to eliminate potential residual background pixels from the mask.
            Defaults to False.

        Returns
        -------
        np.ndarray | None
            The 4 corner points of the card, or None if failed.
        """

        # 1. Morphological Erosion
        if erode:
            # 1.1 Convert raw contour (points) to binary mask (bitmap)
            h, w = img_shape[:2]
            mask_bitmap = np.zeros((h, w), dtype=np.uint8)
            mask_bitmap = cv2.drawContours(
                mask_bitmap, [mask_points], -1, 255, -1
            )  # Fill contour

            # 1.2 Apply Morphological Erosion
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_bitmap = cv2.dilate(mask_bitmap, kernel, iterations=1)
            mask_eroded = cv2.erode(mask_bitmap, kernel, iterations=3)

            # 1.3 Find the new, tighter contour from the eroded mask
            contours, _ = cv2.findContours(
                mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            mask_points = max(contours, key=cv2.contourArea)

        # 2. Polygon Approximation
        epsilon = 0.02 * cv2.arcLength(mask_points, True)
        corner_points = cv2.approxPolyDP(mask_points, epsilon, True)

        # Fallback: use approxPolyN if approxPolyDP fails
        if len(corner_points) != 4:
            if hasattr(cv2, "approxPolyN"):
                corner_points = (
                    cv2.approxPolyN(mask_points, 4, True)
                    .astype(np.int32)
                    .reshape(4, 1, 2)
                )

        # 3. Refine (Line Intersection)
        if refine or len(corner_points) != 4:
            corner_points = find_corners_lines_intersect(mask_points, corner_points)

        if len(corner_points) != 4:
            return None

        return corner_points

    def warp_card(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Warp and resize the card based on detected corner points.

        Args:
            img: Input image in BGR format
            corners: Corner points with shape (4, 1, 2)

        Returns:
            Warped and resized image as np.ndarray
        """
        img_warped = perspective_transform(img, corners)
        img_warped = cv2.resize(
            img_warped,
            self.target_size,
            interpolation=cv2.INTER_LANCZOS4,
        )
        return img_warped

    def annotate_corners(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Annotate the image with detected corner points.

        Args:
            img: Input image in BGR format
            corners: Corner points with shape (4, 1, 2)

        Returns:
            Annotated image with corners drawn in green
        """
        img_annotated = img.copy()
        cv2.drawContours(img_annotated, [corners], -1, (0, 255, 0), 2)
        return img_annotated

    def segment(
        self,
        img: np.ndarray,
        return_annotated: bool = False,
        return_warped: bool = False,
        refine: bool = False,
    ) -> SegmentationResult:
        """
        Main pipeline for card segmentation: Corner Detection -> Optional Annotation -> Optional Warp.

        Args:
            img: Input image in BGR format
            return_annotated: Whether to return annotated image with corners
            return_warped: Whether to return warped card image

        Returns:
            SegmentationResult containing corners, annotated image (if requested),
            warped image (if requested), and success status
        """
        # Detect corners
        corners = self.detect_corners(img, refine=False)
        if corners is None:
            return SegmentationResult(
                corners=None,
                warped_image=None,
                annotated_image=img if return_annotated else None,
                success=False,
            )

        # Generate requested outputs
        annotated_image = (
            self.annotate_corners(img, corners) if return_annotated else None
        )
        warped_image = self.warp_card(img, corners) if return_warped else None

        return SegmentationResult(
            corners=corners,
            warped_image=warped_image,
            annotated_image=annotated_image,
            success=True,
        )

    def segment_from_dir(self, src_dir: str | Path, dst_dir: str | Path) -> None:
        src_dir = Path(src_dir)
        if not src_dir.exists():
            raise ValueError(f"query dir {src_dir} does not exist")

        # prevent overwrite images
        dst_dir = Path(dst_dir)
        assert src_dir != dst_dir

        # create save dir if non-existing
        dst_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(src_dir.glob("*.jpg"))

        for img_path in tqdm(
            image_paths, total=len(image_paths), desc="Segmenting images"
        ):
            # read image
            img = cv2.imread(img_path)

            # segment image
            result = self.segment(img, return_warped=True)
            img_warped = result.warped_image
            if img_warped is None:
                print(f"image {img_path} could not be segmented")
                continue

            # save image
            img_path_dst = dst_dir / img_path.name
            cv2.imwrite(img_path_dst, img_warped)


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
    "--model-name",
    type=click.Choice(
        [
            "yoloe-11s-seg",
            "yolov8s-worldv2",
            # "yoloe-26n-seg",
        ]
    ),
    default="yoloe-11s-seg",
)
def main(input, output_dir, model_name):
    segmentor = CardSegmentor(model_name, param_set_idx=0)
    if input.is_file():
        # Segment one raw test image
        img = cv2.imread(str(input))

        result = segmentor.segment(img, return_warped=True)
        img_warped = result.warped_image

        if img_warped is not None:
            print("Success!")
            cv2.imshow("Warped", img_warped)

            key = cv2.waitKey(0)
            if key == ord("q"):
                cv2.destroyAllWindows()
            # elif key == ord("s"):
            #     img_path_dst = Path(f"database/test/images/ready/{img_path.name}")
            #     img_path_dst.parent.mkdir(parents=True, exist_ok=True)
            #     cv2.imwrite(img_path_dst, img_warped)
            #     cv2.destroyAllWindows()
        else:
            print("No image detected")
    else:
        # Segment all raw test images
        # src_dir = "benchmarks/dataset/images/raw"
        # # dst_dir = "scratch/test_segmentation/approxPolyDP/images/ready"
        # dst_dir = "scratch/test_segmentation/onnx/images/ready"
        segmentor.segment_from_dir(input, output_dir)


if __name__ == "__main__":
    main()

# uv run python -m mtg_recognition.engine.segmentation.segment
# uv run python -m mtg_recognition.engine.segmentation.segment -i benchmarks/dataset/images/raw -o output/engine/segmentation/CardSegmentor
