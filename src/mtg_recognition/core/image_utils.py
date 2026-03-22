from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from mtg_recognition.settings import settings

# All types accepted by load_image
ImageSource = Union[
    str,  # file path
    Path,  # file path
    bytes,  # raw encoded bytes
    bytearray,  # raw encoded bytes
    np.ndarray,  # already decoded BGR array (passthrough)
    Image.Image,  # PIL Image
    # Any file-like with .read() — BytesIO, UploadedFile, open(), etc.
]


def _resize_if_needed(img: np.ndarray, max_dimension: int) -> np.ndarray:
    """Downscale img so its longest edge does not exceed max_dimension.

    Returns the original array unchanged if it is already within bounds.
    Uses INTER_AREA interpolation, which is optimal for downscaling.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_dimension:
        return img
    scale = max_dimension / max(h, w)
    return cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a BGR uint8 numpy array."""
    pil_img = ImageOps.exif_transpose(pil_img)
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _to_ndarray(source: ImageSource) -> np.ndarray:
    """Convert any supported image source to a BGR uint8 numpy array."""

    # Already a numpy array — passthrough
    if isinstance(source, np.ndarray):
        return source

    # Already a PIL Image
    if isinstance(source, Image.Image):
        return _pil_to_bgr(source)

    # Raw bytes / bytearray — wrap in BytesIO for PIL
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    # Path or file-like (BytesIO, UploadedFile, open(), …)
    try:
        pil_img = Image.open(source)
        return _pil_to_bgr(pil_img)
    except Exception as e:
        raise ValueError(
            f"Failed to decode image from {type(source).__name__}: {e}"
        ) from e


def load_and_resize_image(
    source: ImageSource,
    max_dimension: int = settings.MAX_IMAGE_DIMENSION,
) -> np.ndarray:
    """Load an image from any supported source and apply app-standard preprocessing.

    Accepts file paths, raw bytes, file-like objects (BytesIO, UploadedFile, …),
    PIL Images, and numpy arrays. Always returns a BGR uint8 numpy array resized
    so its longest edge does not exceed max_dimension, matching the preprocessing
    applied by the Streamlit app before images reach the pipeline.

    EXIF orientation is applied automatically, so smartphone photos taken in
    portrait mode are correctly rotated before reaching the pipeline.

    Args:
        source: Image source — see ImageSource type alias for accepted types.
        max_dimension: Constrain the longest edge to this many pixels.
                       Defaults to app_settings.MAX_IMAGE_DIMENSION (1280).

    Returns:
        BGR uint8 numpy array, downscaled if necessary.

    Raises:
        FileNotFoundError: If source is a path and the file does not exist.
        ValueError: If the image cannot be decoded.
        TypeError: If source is an unsupported type.
    """
    return _resize_if_needed(_to_ndarray(source), max_dimension)
