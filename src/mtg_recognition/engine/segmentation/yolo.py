"""
YOLO-primary card segmentor with three-path architecture:

    Path 1 — Bypass   : image aspect ratio ≈ MTG card ratio → resize only
                         (covers scans where background was already stripped)
    Path 2 — YOLO     : yoloe-11s-seg batch inference (mini-batches of 20)
    Path 3 — Fallback : CV box segmentor first, then rembg if CV finds nothing
                         (covers the rare <1 % of images YOLO misses)

Preprocessing (CLAHE + gamma) is applied before YOLO inference and before the
CV box segmentor to improve detection quality.  The perspective warp is always
computed from the original, unprocessed image.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .box import detect_cards_cv
from .rembg import CardSegment, ImageSegmentationResult, RembgCardSegmentor
from .seg_utils import _corners_from_binary_mask, valid_aspect_ratio, warp_card

# MTG card portrait ratio  63 / 88 ≈ 0.716
_CARD_RATIO = 63.0 / 88.0
# Allow ±3 % from exact ratio for bypass detection
_BYPASS_LO = _CARD_RATIO * 0.97  # ≈ 0.694
_BYPASS_HI = _CARD_RATIO * 1.03  # ≈ 0.737
# Outer border strip width as a fraction of the smaller image dimension
_BYPASS_BORDER_PCT = 0.05
# Grayscale mean threshold for the border strip:
# scans-without-background have a dark card border (mean ≈ 23–52);
# scans-with-background have a bright scanner bed (mean ≈ 148–240).
# A threshold of 100 sits in the middle of the 96-point gap between groups.
_BYPASS_BORDER_MAX_MEAN = 100

_DEFAULT_PREDICT_KWARGS: dict = dict(
    conf=0.1,  # Higher confidence to filter weak overlaps
    iou=0.0,  # Lower IoU for more aggressive suppression
    agnostic_nms=True,  # Prevent overlaps between different classes
    overlap_mask=False,  # Ensure masks do not overlap at the pixel level
    max_det=50,
    verbose=False,
)


class YoloCardSegmentor:
    """
    Segments MTG cards using YOLO instance segmentation as the primary method.

    Three-path pipeline per image:
        1. Aspect-ratio bypass  → immediate resize (no model)
        2. YOLO                 → batch inference, polygon mask → corners → warp
        3. Fallback             → CV box segmentor, then rembg if CV fails

    Images are preprocessed (CLAHE + gamma) before YOLO and CV detection, but
    the perspective warp is always applied to the original, unprocessed image.

    The rembg session is lazy-loaded on first use to avoid loading it when the
    fallback path is never exercised (which is the common case).

    Args:
        model_path: Path to yoloe-11s-seg.pt (PyTorch, not ONNX).
        classes: Open-vocabulary class names for yoloe set_classes.
        target_size: Output (width, height) of each warped card image.
        aspect_ratio_tolerance: Tolerance for YOLO mask aspect-ratio validation.
        batch_size: Number of images per YOLO predict call.
        fallback_enabled: If False, skip CV and rembg fallback when YOLO finds
                          nothing (useful for --mode yolo CLI option).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        classes: list[str] | None = None,
        target_size: tuple[int, int] = (488, 680),
        aspect_ratio_tolerance: float = 0.2,
        batch_size: int = 20,
        fallback_enabled: bool = True,
    ):
        from mtg_recognition.settings import settings

        _model_path = Path(model_path) if model_path else settings.YOLO_SEG_MODEL_PATH
        _classes = classes if classes is not None else settings.YOLO_SEG_CLASSES

        from ultralytics import YOLO

        self._model = YOLO(str(_model_path), task="segment")
        self._model.set_classes(_classes)

        self.target_size = target_size
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.batch_size = batch_size
        self.fallback_enabled = fallback_enabled

        self._rembg_segmentor: RembgCardSegmentor | None = None  # lazy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, img: np.ndarray) -> ImageSegmentationResult:
        """Segment all cards from a single image."""
        return self.segment_batch([img])[0]

    def segment_batch(
        self,
        imgs: list[np.ndarray],
        batch_size: int | None = None,
    ) -> list[ImageSegmentationResult]:
        """
        Segment cards from multiple images, leveraging YOLO batch inference.

        Args:
            imgs: List of BGR images.
            batch_size: Mini-batch size for YOLO predict calls. Defaults to
                        self.batch_size (20).

        Returns:
            One ImageSegmentationResult per input image, in the same order.
        """
        if not imgs:
            return []

        _batch_size = batch_size if batch_size is not None else self.batch_size
        n = len(imgs)
        results: list[ImageSegmentationResult | None] = [None] * n

        # -- Pass 1: bypass for scan-without-background images --------------------
        yolo_indices: list[int] = []
        for i, img in enumerate(imgs):
            if self._is_scan_without_bg(img):
                results[i] = ImageSegmentationResult(segments=[self._bypass_warp(img)])
            else:
                yolo_indices.append(i)

        # -- Pass 2: YOLO batch inference ----------------------------------
        fallback_indices: list[int] = []
        for start in range(0, len(yolo_indices), _batch_size):
            batch_idx = yolo_indices[start : start + _batch_size]
            batch_imgs = [imgs[i] for i in batch_idx]
            proc_imgs = [preprocess_for_yolo(img) for img in batch_imgs]

            yolo_results = self._model.predict(proc_imgs, **_DEFAULT_PREDICT_KWARGS)

            for global_i, yolo_result in zip(batch_idx, yolo_results):
                segments = self._process_yolo_result(imgs[global_i], yolo_result)
                if segments:
                    results[global_i] = ImageSegmentationResult(segments=segments)
                else:
                    fallback_indices.append(global_i)

        # -- Pass 3: fallback (CV → rembg) for 0-detection images ---------
        for i in fallback_indices:
            if self.fallback_enabled:
                proc_img = preprocess_for_yolo(imgs[i])
                segments = self._fallback_segment(imgs[i], proc_img)
            else:
                segments = []
            results[i] = ImageSegmentationResult(segments=segments)

        return results  # type: ignore[return-value]  # all slots filled

    # ------------------------------------------------------------------
    # Path 1 — Bypass
    # ------------------------------------------------------------------

    def _is_scan_without_bg(self, img: np.ndarray) -> bool:
        h, w = img.shape[:2]
        ratio = min(w, h) / max(w, h)
        if not (_BYPASS_LO <= ratio <= _BYPASS_HI):
            return False
        # Secondary check: the outer border must be dark (card's black border).
        # Scanner-bed images have a bright border; card-only scans do not.
        bh = max(1, int(h * _BYPASS_BORDER_PCT))
        bw = max(1, int(w * _BYPASS_BORDER_PCT))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        border = np.concatenate(
            [
                gray[:bh, :].ravel(),
                gray[-bh:, :].ravel(),
                gray[:, :bw].ravel(),
                gray[:, -bw:].ravel(),
            ]
        )
        return bool(np.mean(border) < _BYPASS_BORDER_MAX_MEAN)

    def _bypass_warp(self, img: np.ndarray) -> CardSegment:
        h, w = img.shape[:2]
        out = img
        if w > h:
            out = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        out = cv2.resize(out, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        return CardSegment(warped_image=out)

    # ------------------------------------------------------------------
    # Path 2 — YOLO mask → corners → warp
    # ------------------------------------------------------------------

    def _process_yolo_result(self, img: np.ndarray, result) -> list[CardSegment]:
        if result.masks is None or len(result.masks.xy) == 0:
            return []

        segments: list[CardSegment] = []
        h, w = img.shape[:2]

        for polygon in result.masks.xy:
            if len(polygon) < 4:
                continue

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)

            corners = _corners_from_binary_mask(mask, erode=False)
            if corners is None:
                continue

            if not valid_aspect_ratio(corners, self.aspect_ratio_tolerance):
                continue

            segments.append(
                CardSegment(warped_image=warp_card(img, corners, self.target_size))
            )

        return segments

    # ------------------------------------------------------------------
    # Path 3 — Fallback: CV box segmentor, then rembg
    # ------------------------------------------------------------------

    def _fallback_segment(
        self, img: np.ndarray, proc_img: np.ndarray
    ) -> list[CardSegment]:
        # Sub-path A: classical CV (sub-millisecond, good for uniform backgrounds)
        segments = detect_cards_cv(
            proc_img,
            self.target_size,
            aspect_ratio_tolerance=self.aspect_ratio_tolerance,
            warp_img=img,
        )
        if segments:
            return segments

        # Sub-path B: rembg (works very well for photos with clear background,
        # but very slow to load and run, so only used as a last resort)
        rembg_result = self._get_rembg_segmentor().segment(img)
        return rembg_result.segments

    def _get_rembg_segmentor(self) -> RembgCardSegmentor:
        if self._rembg_segmentor is None:
            self._rembg_segmentor = RembgCardSegmentor(
                target_size=self.target_size,
                aspect_ratio_tolerance=self.aspect_ratio_tolerance,
            )
        return self._rembg_segmentor


def preprocess_for_yolo(img: np.ndarray) -> np.ndarray:
    """
    Enhance contrast and brightness for YOLO inference and CV box detection.

    Applies adaptive CLAHE on the L channel (LAB) followed by gamma correction.
    Dimensions are preserved — the returned image is the same size as the input.

    Args:
        img: BGR image (uint8).

    Returns:
        Enhanced BGR image (uint8), same dimensions as input.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l_mean = float(np.mean(l))
    if l_mean < 50:  # very dark
        clip, tile = 4.0, 4
    elif l_mean < 100:  # dark
        clip, tile = 3.5, 6
    elif l_mean > 200:  # very bright
        clip, tile = 2.5, 8
    else:  # normal
        clip, tile = 3.0, 6

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l_enhanced = clahe.apply(l)

    gamma = 1.15 if l_mean < 100 else (0.95 if l_mean > 180 else 1.05)
    l_gamma = np.power(l_enhanced / 255.0, gamma) * 255.0
    l_gamma = np.clip(l_gamma, 0, 255).astype(np.uint8)

    lab = cv2.merge([l_gamma, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
