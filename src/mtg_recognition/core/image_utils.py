from __future__ import annotations

import io
from pathlib import Path
from typing import IO

import cv2
import numpy as np
from PIL import Image, ImageOps

from mtg_recognition.settings import settings

# All types accepted by prepare_image
ImageSource = (
    str  # file path
    | Path  # file path
    | bytes  # raw encoded bytes
    | bytearray  # raw encoded bytes
    | IO[bytes]  # file-like (BytesIO, UploadedFile, open(), …)
    | np.ndarray  # already decoded BGR array (passthrough)
    | Image.Image  # PIL Image
)


def get_image_files_in_directory(
    directory: Path, extensions: list[str] | None = None
) -> list[Path]:
    """Find all image files in a directory, matching settings.IMG_EXT."""
    if extensions is None:
        extensions = settings.IMG_EXT
    img_paths = []
    for ext in extensions:
        img_paths.extend(directory.glob(ext))
    return sorted(img_paths)


def resize_if_needed(img: np.ndarray, max_dimension: int) -> np.ndarray:
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


def prepare_image(source: ImageSource) -> np.ndarray:
    """Produce the engine's input pixels from any supported image source.

    Decodes the source, rotates it by its EXIF orientation, then shrinks it so its
    longest edge is at most settings.MAX_IMAGE_DIMENSION (never upscaling). An
    already-decoded array is taken as is, then shrunk.

    Returns:
        BGR uint8 numpy array.

    Raises:
        ValueError: If the source cannot be decoded.
    """
    return resize_if_needed(_decode(source), settings.MAX_IMAGE_DIMENSION)


def _decode(source: ImageSource) -> np.ndarray:
    """Decode any supported image source to a BGR uint8 numpy array, applying EXIF
    orientation.
    """

    # Already a numpy array — passthrough
    if isinstance(source, np.ndarray):
        return source

    # Already a PIL Image
    if isinstance(source, Image.Image):
        return _pil_to_bgr(source)

    # Raw bytes / bytearray — wrap in BytesIO for PIL; paths and file-likes as is
    stream = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source

    try:
        pil_img = Image.open(stream)
        return _pil_to_bgr(pil_img)
    except Exception as e:
        raise ValueError(
            f"Failed to decode image from {type(source).__name__}: {e}"
        ) from e
