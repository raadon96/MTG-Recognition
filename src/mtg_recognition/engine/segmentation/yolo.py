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
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from mtg_recognition.settings import settings

from .box import detect_cards_cv
from .geometry import compute_intersection
from .rembg import RembgSegmentor
from .seg_utils import (
    CARD_ASPECT_RATIO,
    CARD_IMAGE_MAX_SIZE,
    CARD_IMAGE_MIN_SIZE,
    SegmentationResult,
    SegmentedCard,
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

# YOLO's masks have 160 cells along the long side of the image it sees
_YOLO_MASK_CELLS = 160
# Fraction of each side's length left out at both ends when finding the side,
# where the outline turns the corner
_SIDE_TRIM = 0.1
# Distance in px between the points along a side where its edge is looked for
_EDGE_SAMPLE_SPACING = 3.0

YOLO_SEG_CLASSES: list[str] = ["book cover", "art work"]

_DEFAULT_PREDICT_KWARGS: dict[str, Any] = {
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


class YoloSegmentor:
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
        target_size: Output (width, height) of each card image.
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

        self._rembg_segmentor: RembgSegmentor | None = None  # lazy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, query_image: np.ndarray) -> SegmentationResult:
        """
        Segment all cards from a query image.

        Images are segmented one at a time: YOLO batch inference changes results
        for mixed-size batches (see docs/adr/0002-recognise-one-image-at-a-time.md).

        Args:
            query_image: BGR query image.

        Returns:
            SegmentationResult with one SegmentedCard per card found.
            cards is empty if no cards were found.
        """
        # -- Path 1: bypass for scan-without-background images --------------
        if is_scan_without_bg(query_image):
            return SegmentationResult(cards=[bypass_warp(query_image)])

        # -- Path 2: YOLO ----------------------------------------------------
        proc_img = enhance_contrast(query_image)
        yolo_result = self._model.predict(proc_img, **_DEFAULT_PREDICT_KWARGS)[0]
        cards = self._process_yolo_result(query_image, yolo_result)

        # -- Path 3: fallback (CV → rembg) for 0-detection images -----------
        if not cards and self.fallback_enabled:
            cards = self._fallback_segment(query_image, proc_img)

        return SegmentationResult(cards=cards)

    # ------------------------------------------------------------------
    # Path 2 — YOLO mask → corners → warp
    # ------------------------------------------------------------------

    def _process_yolo_result(
        self, query_image: np.ndarray, result: Results
    ) -> list[SegmentedCard]:
        if result.masks is None or len(result.masks.xy) == 0:
            return []

        cards: list[SegmentedCard] = []
        for outline in result.masks.xy:
            corners = card_corners_from_outline(
                query_image, outline, self.aspect_ratio_tolerance
            )
            if corners is None:
                continue
            cards.append(SegmentedCard(card_image=warp_card(query_image, corners)))

        return cards

    # ------------------------------------------------------------------
    # Path 3 — Fallback: CV box segmentor, then rembg
    # ------------------------------------------------------------------

    def _fallback_segment(
        self, query_image: np.ndarray, proc_img: np.ndarray
    ) -> list[SegmentedCard]:
        # Sub-path A: classical CV (sub-millisecond, good for uniform backgrounds)
        cards = detect_cards_cv(
            proc_img,
            aspect_ratio_tolerance=self.aspect_ratio_tolerance,
            warp_img=query_image,
        )
        if cards:
            return cards

        # Sub-path B: rembg (works very well for photos with clear background,
        # but very slow to load and run, so only used as a last resort)
        rembg_result = self._get_rembg_segmentor().segment(query_image)
        return rembg_result.cards

    def _get_rembg_segmentor(self) -> RembgSegmentor:
        if self._rembg_segmentor is None:
            self._rembg_segmentor = RembgSegmentor(
                aspect_ratio_tolerance=self.aspect_ratio_tolerance,
            )
        return self._rembg_segmentor


def card_corners_from_outline(
    query_image: np.ndarray, outline: np.ndarray, aspect_ratio_tolerance: float
) -> np.ndarray | None:
    """The four corners of the card that YOLO outlined, on the card's edges.

    YOLO computes its masks on a coarse grid, so its outline is a staircase and
    corners taken straight from it can sit a step off the card. The sides are
    fitted to the outline, then each is moved onto the light/dark edge nearest
    to it in the query image.

    Args:
        query_image: BGR query image, not the contrast-enhanced copy YOLO sees.
        outline: YOLO's outline of one card, shape (N, 2), in query image pixels.
        aspect_ratio_tolerance: As for valid_aspect_ratio.

    Returns:
        Corners of shape (4, 1, 2) int32, or None if the outline isn't card-shaped.
        When the sides can't be fitted, the corners of the outline itself; when
        the moved sides fail the aspect ratio check, the corners of the fitted
        sides.
    """
    if len(outline) < 4:
        return None

    # Rough corners from the staircase outline
    h, w = query_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [outline.astype(np.int32)], (255,))
    rough = corners_from_binary_mask(mask, erode=False)
    if rough is None or not valid_aspect_ratio(rough, aspect_ratio_tolerance):
        return None

    # Fit the sides: split the outline at the rough corners and fit a straight
    # line through each piece, leaving out its ends, where the outline turns
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    rough_xy = rough.reshape(4, 2)
    splits = sorted(
        int(np.argmin(np.linalg.norm(contour - corner, axis=1))) for corner in rough_xy
    )
    if len(set(splits)) < 4:
        return rough
    pieces = np.split(
        np.roll(contour, -splits[0], axis=0), [i - splits[0] for i in splits[1:]]
    )
    fitted_sides = []
    for piece in pieces:
        trim = int(len(piece) * _SIDE_TRIM)
        points = piece[trim : len(piece) - trim].astype(np.float32)
        if len(points) < 2:
            return rough
        fitted_sides.append(cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten())
    fitted = _corners_of_sides(fitted_sides)
    if fitted is None or not valid_aspect_ratio(
        _rounded_corners(fitted), aspect_ratio_tolerance
    ):
        return rough

    # Move each side onto the card's edge in the query image, looking up to one
    # step of YOLO's outline either way
    step = max(h, w) / _YOLO_MASK_CELLS
    gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    moved_sides = [
        _side_on_edge(gray, fitted[k], fitted[(k + 1) % 4], step) for k in range(4)
    ]
    moved = _corners_of_sides(moved_sides)
    if moved is None or not valid_aspect_ratio(
        _rounded_corners(moved), aspect_ratio_tolerance
    ):
        return _rounded_corners(fitted)
    return _rounded_corners(moved)


def _side_on_edge(
    gray: np.ndarray, start: np.ndarray, end: np.ndarray, reach: float
) -> np.ndarray:
    """The line along the strongest light/dark edge within reach of a side.

    Looks across the side, up to `reach` px either way, at points spread along
    it, takes the strongest edge at each point and fits a line through them.
    Keeps the side where it is if no such line stays within reach of it.
    """
    direction = end - start
    length = float(np.linalg.norm(direction))
    direction /= length
    side = np.array([*direction, *start])
    normal = np.array([-direction[1], direction[0]])

    # Brightness profiles across the side, a few px apart along it
    count = max(2, int(length * (1 - 2 * _SIDE_TRIM) / _EDGE_SAMPLE_SPACING))
    along = np.linspace(_SIDE_TRIM, 1 - _SIDE_TRIM, count) * length
    # out to 0.5 px past the search distance, so that the change over 1 px can
    # be measured at its ends
    across = np.arange(-reach - 0.5, reach + 0.5 + 0.25, 0.5)
    centres = start + along[:, None] * direction
    xs = centres[:, None, 0] + across[None, :] * normal[0]
    ys = centres[:, None, 1] + across[None, :] * normal[1]
    profiles = cv2.remap(
        gray,
        xs.astype(np.float32),
        ys.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # The strongest edge on each profile: the largest change over 1 px
    change = np.abs(profiles[:, 2:] - profiles[:, :-2])
    at = np.argmax(change, axis=1)
    strength = change[np.arange(count), at]
    edge_points = centres + across[at + 1][:, None] * normal
    # where the edge is faint (glare, a card border as light as the table) the
    # strongest change is noise, so keep only the clear ones
    clear = strength >= 0.5 * np.median(strength)
    points = edge_points[clear].astype(np.float32)
    if len(points) < 2:
        return side
    # Huber, so the odd point on the art or a reflection pulls the line less
    moved = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()

    # Check: a line through edges found within reach stays within reach at both
    # ends. One that leaves the band mixes two edges, such as the card's edge
    # and the inner edge of its border where YOLO's outline stops short.
    moved_normal = np.array([-moved[1], moved[0]])
    for end_point in (start, end):
        if abs((end_point - moved[2:]) @ moved_normal) > reach:
            return side
    return moved


def _corners_of_sides(sides: list[np.ndarray]) -> np.ndarray | None:
    """The four corners where neighbouring sides cross, or None if two are parallel."""
    corners = []
    for k in range(4):
        point = compute_intersection(sides[(k - 1) % 4], sides[k])
        if point is None:
            return None
        corners.append(point)
    return np.array(corners, dtype=np.float64)


def _rounded_corners(corners: np.ndarray) -> np.ndarray:
    """(4, 2) float corners in the (4, 1, 2) int32 shape warp_card takes."""
    return np.round(corners).astype(np.int32).reshape(4, 1, 2)


def is_scan_without_bg(query_image: np.ndarray) -> bool:
    """Whether a query image is already a card image: card ratio, dark border."""
    h, w = query_image.shape[:2]
    ratio = min(w, h) / max(w, h)
    if not (_BYPASS_LO <= ratio <= _BYPASS_HI):
        return False
    # Secondary check: the outer border must be dark (card's black border).
    # Scanner-bed images have a bright border; card-only scans do not.
    bh = max(1, int(h * _BYPASS_BORDER_PCT))
    bw = max(1, int(w * _BYPASS_BORDER_PCT))
    gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    border = np.concatenate(
        [
            gray[:bh, :].ravel(),
            gray[-bh:, :].ravel(),
            gray[:, :bw].ravel(),
            gray[:, -bw:].ravel(),
        ]
    )
    return bool(np.mean(border) < _BYPASS_BORDER_MAX_MEAN)


def bypass_warp(query_image: np.ndarray) -> SegmentedCard:
    """The card image of a scan without background: upright, size clamped."""
    out = query_image
    if out.shape[1] > out.shape[0]:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    h, w = out.shape[:2]
    out_w, out_h = clamp_size((w, h), CARD_IMAGE_MIN_SIZE, CARD_IMAGE_MAX_SIZE)
    if (out_w, out_h) != (w, h):
        out = cv2.resize(out, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    return SegmentedCard(card_image=out)


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
