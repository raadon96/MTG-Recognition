import cv2
import numpy as np

# Adaptive upscaling constants for the collector information crop.
# PP-OCRv5_mobile_rec native input height is 48 px; text below ~32 px degrades.
# RATIO_TEXT_TO_CROP: approximate fraction of the crop height occupied by text lines.
TARGET_TEXT_HEIGHT: int = 48
RATIO_TEXT_TO_CROP: float = 0.15
MIN_UPSCALE: float = 1.0
MAX_UPSCALE: float = 4.0


def upscale_for_ocr(img: np.ndarray) -> np.ndarray:
    """Adaptively upscale an OCR crop so its text reaches TARGET_TEXT_HEIGHT px.

    The scale factor is clamped to [MIN_UPSCALE, MAX_UPSCALE]. Meant for the
    collector information crop (small text); the name crop's text is already large
    enough.

    Args:
        img: BGR image (numpy array)

    Returns:
        Upscaled BGR image
    """
    h, w = img.shape[:2]
    s = TARGET_TEXT_HEIGHT / (h * RATIO_TEXT_TO_CROP)
    s = max(MIN_UPSCALE, min(s, MAX_UPSCALE))
    new_w, new_h = int(round(w * s)), int(round(h * s))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
