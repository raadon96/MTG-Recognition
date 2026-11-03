"""The corners of a card from YOLO's outline of it."""

import cv2
import numpy as np

from .geometry import compute_intersection
from .seg_utils import corners_from_binary_mask, valid_aspect_ratio

# YOLO's masks have 160 cells along the long side of the image it sees
_YOLO_MASK_CELLS = 160
# Fraction of each side's length left out at both ends when finding the side,
# where the outline turns the corner
_SIDE_TRIM = 0.1
# Distance in px between the points along a side where its edge is looked for
_EDGE_SAMPLE_SPACING = 3.0


def card_corners_from_outline(
    query_image: np.ndarray, outline: np.ndarray
) -> np.ndarray | None:
    """The four corners of the card that YOLO outlined, on the card's edges.

    YOLO computes its masks on a coarse grid, so its outline is a staircase and
    corners taken straight from it can sit a step off the card. The sides are
    fitted to the outline, then each is moved onto the light/dark edge nearest
    to it in the query image.

    Args:
        query_image: BGR query image, not the contrast-enhanced copy YOLO sees.
        outline: YOLO's outline of one card, shape (N, 2), in query image pixels.

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
    rough = corners_from_binary_mask(mask)
    if rough is None or not valid_aspect_ratio(rough):
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
    if fitted is None or not valid_aspect_ratio(_rounded_corners(fitted)):
        return rough

    # Move each side onto the card's edge in the query image, looking up to one
    # step of YOLO's outline either way
    step = max(h, w) / _YOLO_MASK_CELLS
    gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    moved_sides = [
        _side_on_edge(gray, fitted[k], fitted[(k + 1) % 4], step) for k in range(4)
    ]
    moved = _corners_of_sides(moved_sides)
    if moved is None or not valid_aspect_ratio(_rounded_corners(moved)):
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
