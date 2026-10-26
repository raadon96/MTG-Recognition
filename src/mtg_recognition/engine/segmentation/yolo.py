"""
YOLO-primary card segmentor with three-path architecture:

    Path 1 — Bypass   : image aspect ratio ≈ MTG card ratio → resize only
                         (covers scans where background was already stripped)
    Path 2 — YOLO     : yoloe-11s-seg inference, one image at a time
    Path 3 — Fallback : CV box segmentor first, then rembg if CV finds nothing
                         (covers the rare <1 % of images YOLO misses)

Preprocessing (CLAHE + gamma) is applied before YOLO inference and before the
CV box segmentor to improve detection quality.  The perspective warp is always
computed from the original, unprocessed image.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from mtg_recognition.settings import settings

from .box import detect_cards_cv
from .rembg import RembgCardSegmentor
from .seg_utils import (
    CARD_ASPECT_RATIO,
    WARPED_MAX_SIZE,
    WARPED_MIN_SIZE,
    CardSegment,
    ImageSegmentationResult,
    clamp_size,
    corners_from_binary_mask,
    valid_aspect_ratio,
    warp_card,
)

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

# Allow ±3 % from exact ratio for bypass detection
_RATIO_TOLERANCE = 0.03
_BYPASS_LO = CARD_ASPECT_RATIO * (1 - _RATIO_TOLERANCE)  # ≈ 0.694
_BYPASS_HI = CARD_ASPECT_RATIO * (1 + _RATIO_TOLERANCE)  # ≈ 0.737
# Outer border strip width as a fraction of the smaller image dimension
_BYPASS_BORDER_PCT = 0.05
# Grayscale mean threshold for the border strip:
# scans-without-background have a dark card border (mean ≈ 23–52);
# scans-with-background have a bright scanner bed (mean ≈ 148–240).
# A threshold of 100 sits in the middle of the 96-point gap between groups.
_BYPASS_BORDER_MAX_MEAN = 100

YOLO_SEG_CLASSES: list[str] = ["book cover", "art work"]

_DEFAULT_PREDICT_KWARGS: dict[str, object] = {
    "conf": 0.1,  # Higher confidence to filter weak overlaps
    "iou": 0.0,  # Lower IoU for more aggressive suppression
    "agnostic_nms": True,  # Prevent overlaps between different classes
    "overlap_mask": False,  # Ensure masks do not overlap at the pixel level
    "max_det": 50,
    "verbose": False,
}

# enhance_contrast brightness thresholds (L-channel mean, 0–255)
_L_VERY_DARK = 50
_L_DARK = 100
_L_VERY_BRIGHT = 200
_L_GAMMA_BRIGHT = 180

# CLAHE parameters per brightness band: (clip_limit, tile_grid_size)
_CLAHE_VERY_DARK = (4.0, 4)
_CLAHE_DARK = (3.5, 6)
_CLAHE_VERY_BRIGHT = (2.5, 8)
_CLAHE_NORMAL = (3.0, 6)

# Gamma correction per brightness band
_GAMMA_DARK = 1.15
_GAMMA_BRIGHT = 0.95
_GAMMA_NORMAL = 1.05


class YoloCardSegmentor:
    """
    Segments MTG cards using YOLO instance segmentation as the primary method.

    Three-path pipeline per image:
        1. Aspect-ratio bypass  → immediate resize (no model)
        2. YOLO                 → inference, polygon mask → corners → warp
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
        fallback_enabled: If False, skip CV and rembg fallback when YOLO finds
                          nothing (useful for --mode yolo CLI option).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        classes: list[str] | None = None,
        aspect_ratio_tolerance: float = 0.2,
        fallback_enabled: bool = True,
    ):

        _model_path = Path(model_path) if model_path else settings.YOLO_SEG_MODEL_PATH
        _classes = classes if classes is not None else YOLO_SEG_CLASSES

        from ultralytics import YOLO

        self._model = YOLO(str(_model_path), task="segment")
        if self._model.model is None:
            raise ValueError(f"Failed to load YOLO model from {_model_path}")
        self._model.set_classes(_classes)

        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.fallback_enabled = fallback_enabled

        self._rembg_segmentor: RembgCardSegmentor | None = None  # lazy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, img: np.ndarray) -> ImageSegmentationResult:
        """
        Segment all cards from a single image.

        Images are segmented one at a time: YOLO batch inference changes results
        for mixed-size batches (see docs/adr/0002-recognise-one-image-at-a-time.md).

        Args:
            img: BGR input image.

        Returns:
            ImageSegmentationResult with one CardSegment per detected card.
            segments is empty if no cards were found.
        """
        # -- Path 1: bypass for scan-without-background images --------------
        if is_scan_without_bg(img):
            return ImageSegmentationResult(segments=[bypass_warp(img)])

        # -- Path 2: YOLO ----------------------------------------------------
        proc_img = enhance_contrast(img)
        yolo_result = self._model.predict(proc_img, **_DEFAULT_PREDICT_KWARGS)[0]
        segments = self._process_yolo_result(img, yolo_result)

        # -- Path 3: fallback (CV → rembg) for 0-detection images -----------
        if not segments and self.fallback_enabled:
            segments = self._fallback_segment(img, proc_img)

        return ImageSegmentationResult(segments=segments)

    # ------------------------------------------------------------------
    # Path 2 — YOLO mask → corners → warp
    # ------------------------------------------------------------------

    def _process_yolo_result(
        self, img: np.ndarray, result: Results
    ) -> list[CardSegment]:
        if result.masks is None or len(result.masks.xy) == 0:
            return []

        segments: list[CardSegment] = []
        h, w = img.shape[:2]

        for polygon in result.masks.xy:
            if len(polygon) < 4:
                continue

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)

            corners = corners_from_binary_mask(mask, erode=False)
            if corners is None:
                continue

            if not valid_aspect_ratio(corners, self.aspect_ratio_tolerance):
                continue

            segments.append(CardSegment(warped_image=warp_card(img, corners)))

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
                aspect_ratio_tolerance=self.aspect_ratio_tolerance,
            )
        return self._rembg_segmentor


def is_scan_without_bg(img: np.ndarray) -> bool:
    """Whether a query image is already a card image: card ratio, dark border."""
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


def bypass_warp(img: np.ndarray) -> CardSegment:
    """The card image of a scan without background: upright, size clamped."""
    out = img
    if out.shape[1] > out.shape[0]:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    h, w = out.shape[:2]
    out_w, out_h = clamp_size((w, h), WARPED_MIN_SIZE, WARPED_MAX_SIZE)
    if (out_w, out_h) != (w, h):
        out = cv2.resize(out, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    return CardSegment(warped_image=out)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
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
    l_channel, a, b = cv2.split(lab)

    l_mean = float(np.mean(l_channel))
    if l_mean < _L_VERY_DARK:
        clip, tile = _CLAHE_VERY_DARK
    elif l_mean < _L_DARK:
        clip, tile = _CLAHE_DARK
    elif l_mean > _L_VERY_BRIGHT:
        clip, tile = _CLAHE_VERY_BRIGHT
    else:
        clip, tile = _CLAHE_NORMAL

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l_enhanced = clahe.apply(l_channel)

    gamma = (
        _GAMMA_DARK
        if l_mean < _L_DARK
        else (_GAMMA_BRIGHT if l_mean > _L_GAMMA_BRIGHT else _GAMMA_NORMAL)
    )
    l_gamma = np.power(l_enhanced / 255.0, gamma) * 255.0
    l_gamma = np.clip(l_gamma, 0, 255).astype(np.uint8)

    merged = cv2.merge([l_gamma, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
