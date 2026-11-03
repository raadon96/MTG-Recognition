"""The four corners of a card: from a mask, or from YOLO's outline of it.

All coordinates follow OpenCV's ``(x, y)`` convention. Corner arrays use the
OpenCV contour format: shape ``(N, 1, 2)`` int32 unless stated otherwise.
"""

import cv2
import numpy as np

# MTG standard card dimensions: 63 mm × 88 mm (portrait)
CARD_ASPECT_RATIO = 63.0 / 88.0  # ≈ 0.716

# Allowed relative deviation from CARD_ASPECT_RATIO for a found card: 0.2
# accepts ratios in [0.57, 0.86]
ASPECT_RATIO_TOLERANCE = 0.2

# YOLO's masks have 160 cells along the long side of the image it sees
_YOLO_MASK_CELLS = 160
# Fraction of each side's length left out at both ends when finding the side,
# where the outline turns the corner
_SIDE_TRIM = 0.1
# Distance in px between the points along a side where its edge is looked for
_EDGE_SAMPLE_SPACING = 3.0

_MIN_CONTOUR_POINTS: int = 20  # below this, Hough corner detection is unreliable
_HOUGH_MIN_LENGTH_RATIO: float = (
    0.10  # minLineLength as a fraction of the canvas diagonal
)
_HOUGH_MAX_GAP_RATIO: float = 0.05  # maxLineGap as a fraction of the canvas diagonal
_ARC_TRIM_FRACTION: float = 0.15  # fraction of arc endpoints trimmed before line fit
_PARALLEL_DET_THRESHOLD: float = 1e-6  # determinant threshold for near-parallel lines


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


def corners_from_binary_mask(mask: np.ndarray) -> np.ndarray | None:
    """
    Extract 4 card corners from a binary mask.

    Applies polygon approximation, with a Hough line-intersection fallback and
    then a farthest-point split.

    Args:
        mask: Binary uint8 mask (0/255), same spatial size as the source image.

    Returns:
        np.ndarray of shape (4, 1, 2), or None if 4 corners could not be found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True)

    if len(corners) == 4:
        return corners

    # Fallback: fit the four sides with Hough lines, else by splitting the
    # contour at its farthest points
    contour_xy = contour.reshape(-1, 2)
    if len(contour_xy) < _MIN_CONTOUR_POINTS:
        return None
    corners = _corners_via_hough(contour_xy)
    if corners is None:
        corners = _corners_via_farthest_split(contour_xy)
    return corners


def _corners_via_hough(contour_xy: np.ndarray) -> np.ndarray | None:
    """
    Fit 4 edge lines to the contour via HoughLinesP and return their intersections.

    Strategy:
      1. Draw the contour on a local canvas.
      2. Run HoughLinesP to detect line segments.
      3. Cluster segments into 2 direction groups (by angular gap).
      4. Within each group split into 2 parallel lines by perpendicular projection.
      5. Intersect each line from group 0 with each line from group 1 → 4 corners.
    """
    x_min, y_min = contour_xy.min(axis=0)
    local_xy = (contour_xy - [x_min, y_min]).astype(np.int32).reshape(-1, 1, 2)
    x_max, y_max = local_xy.reshape(-1, 2).max(axis=0)

    if x_max < 2 or y_max < 2:
        return None

    canvas = np.zeros((y_max + 1, x_max + 1), dtype=np.uint8)
    cv2.drawContours(canvas, [local_xy], -1, (255,), 1)

    diag = np.sqrt(x_max**2 + y_max**2)
    raw = cv2.HoughLinesP(
        canvas,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=int(diag * _HOUGH_MIN_LENGTH_RATIO),
        maxLineGap=int(diag * _HOUGH_MAX_GAP_RATIO),
    )
    if raw is None or len(raw) < 4:
        return None

    segments = raw.reshape(-1, 4).astype(float)
    angles = (
        np.arctan2(
            segments[:, 3] - segments[:, 1],
            segments[:, 2] - segments[:, 0],
        )
        % np.pi
    )  # undirected angle in [0, pi)

    # Find the largest angular gap to split into 2 direction groups.
    sorted_idx = np.argsort(angles)
    sorted_angles = angles[sorted_idx]
    gaps = np.diff(sorted_angles)
    wrap_gap = sorted_angles[0] + np.pi - sorted_angles[-1]
    gaps = np.append(gaps, wrap_gap)
    split = np.argmax(gaps)

    group0_idx = sorted_idx[: split + 1]
    group1_idx = sorted_idx[split + 1 :]

    if len(group0_idx) < 1 or len(group1_idx) < 1:
        return None

    edge_lines = []
    for group_idx in [group0_idx, group1_idx]:
        group_segs = segments[group_idx]
        group_angles = angles[group_idx]

        # Perpendicular direction to the median angle of this group.
        perp = np.median(group_angles) + np.pi / 2
        px, py = np.cos(perp), np.sin(perp)

        midpoints = group_segs[:, :2] * 0.5 + group_segs[:, 2:] * 0.5
        projections = midpoints[:, 0] * px + midpoints[:, 1] * py
        median_proj = np.median(projections)

        for mask in [projections <= median_proj, projections > median_proj]:
            selected = group_segs[mask] if np.any(mask) else group_segs
            pts = np.vstack([selected[:, :2], selected[:, 2:]]).astype(np.float32)
            line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            # Shift line origin back to original image coordinates.
            line[2] += x_min
            line[3] += y_min
            edge_lines.append(line)

    if len(edge_lines) != 4:
        return None

    # Intersect each line from group 0 (indices 0,1) with each from group 1 (2,3).
    corners = []
    for i in range(2):
        for j in range(2, 4):
            pt = compute_intersection(edge_lines[i], edge_lines[j])
            if pt is None:
                return None
            corners.append(pt)

    return np.array(corners, dtype=np.int32).reshape(4, 1, 2)


def _corners_via_farthest_split(contour_xy: np.ndarray) -> np.ndarray | None:
    """
    Find 4 corners by recursively splitting the contour at its geometrically farthest
    points, then fitting one line per edge segment and intersecting adjacent lines.

    Strategy:
      1. Find the 2 contour points farthest apart (diameter) via the convex hull.
      2. Split the contour into 2 arcs at those points.
      3. In each arc find the point farthest from the diameter line.
      4. Partition the contour at the 4 resulting corner candidates.
      5. Fit a line per segment and intersect adjacent pairs.
    """
    n = len(contour_xy)

    # Diameter search on the convex hull vertices (much smaller than full contour).
    hull_idx = (
        cv2.convexHull(
            contour_xy.astype(np.float32).reshape(-1, 1, 2), returnPoints=False
        )
        .flatten()
        .astype(np.intp)
    )
    hull_pts = contour_xy[hull_idx].astype(float)
    diffs = hull_pts[:, np.newaxis, :] - hull_pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs**2, axis=2)
    hi, hj = np.unravel_index(np.argmax(dists_sq), dists_sq.shape)
    i, j = sorted([int(hull_idx[hi]), int(hull_idx[hj])])

    A, B = contour_xy[i].astype(float), contour_xy[j].astype(float)
    AB = B - A
    AB_len = np.linalg.norm(AB)
    if AB_len < 1e-6:
        return None

    def farthest_in_arc(arc_idx: np.ndarray) -> int:
        pts = contour_xy[arc_idx].astype(float)
        AP = pts - A
        cross = np.abs(AP[:, 0] * AB[1] - AP[:, 1] * AB[0])
        return int(arc_idx[np.argmax(cross)])

    arc1 = np.arange(i, j + 1)
    arc2 = np.concatenate([np.arange(j, n), np.arange(0, i + 1)])
    p1 = farthest_in_arc(arc1)
    p2 = farthest_in_arc(arc2)

    split_indices = sorted([i, j, p1, p2])
    if len(set(split_indices)) < 4:
        return None

    # Partition the contour and fit one line per segment.
    shift = split_indices[0]
    contour_rolled = np.roll(contour_xy, -shift, axis=0)
    adj_indices = [idx - shift for idx in split_indices]
    segments = np.split(contour_rolled, adj_indices[1:])

    lines = []
    for seg in segments:
        if len(seg) < 2:
            return None
        trim = int(len(seg) * _ARC_TRIM_FRACTION)
        seg_t = seg[trim : len(seg) - trim] if len(seg) > 2 * trim + 2 else seg
        line = cv2.fitLine(seg_t.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        lines.append(line.flatten())

    # Start at line 1, so that each corner is compute_intersection(line k,
    # line k + 1): swapping its arguments can change the last bit, and the
    # int32 truncation can turn that into a 1 px change in the card image
    corners = _corners_of_sides(lines[1:] + lines[:1])
    if corners is None:
        return None
    return corners.astype(np.int32).reshape(4, 1, 2)


def _corners_of_sides(sides: list[np.ndarray]) -> np.ndarray | None:
    """The four corners where neighbouring sides cross, or None if two are parallel."""
    corners = []
    for k in range(4):
        point = compute_intersection(sides[(k - 1) % 4], sides[k])
        if point is None:
            return None
        corners.append(point)
    return np.array(corners, dtype=np.float64)


def compute_intersection(line1: np.ndarray, line2: np.ndarray) -> list[float] | None:
    """
    Finds the intersection point of two lines.
    Each line is given by (vx, vy, x0, y0) from cv2.fitLine.

    Returns
    -------
    [x, y] intersection point, or None if the lines are parallel.
    """
    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2

    # The system of equations:
    # P1 = (x1, y1) + t1 * (vx1, vy1)
    # P2 = (x2, y2) + t2 * (vx2, vy2)
    # We want P1 = P2.
    # This leads to: t1*vx1 - t2*vx2 = x2 - x1
    #                t1*vy1 - t2*vy2 = y2 - y1

    # Solving using determinants (Cramer's Rule equivalent)
    det = vx1 * -vy2 - vy1 * -vx2

    if abs(det) < _PARALLEL_DET_THRESHOLD:
        return None  # Parallel lines

    dx = x2 - x1
    dy = y2 - y1

    t1 = (dx * -vy2 - dy * -vx2) / det

    x = x1 + t1 * vx1
    y = y1 + t1 * vy1

    return [float(x), float(y)]


def _rounded_corners(corners: np.ndarray) -> np.ndarray:
    """(4, 2) float corners in the (4, 1, 2) int32 shape warp_card takes."""
    return np.round(corners).astype(np.int32).reshape(4, 1, 2)


def valid_aspect_ratio(corners: np.ndarray) -> bool:
    """Return True if the quad formed by corners has an MTG-card aspect ratio."""
    pts = corners.reshape(4, 2).astype(float)
    sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    pair_a = (sides[0] + sides[2]) / 2
    pair_b = (sides[1] + sides[3]) / 2
    if max(pair_a, pair_b) < 1e-6:
        return False
    ratio = min(pair_a, pair_b) / max(pair_a, pair_b)
    lo = CARD_ASPECT_RATIO * (1 - ASPECT_RATIO_TOLERANCE)
    hi = CARD_ASPECT_RATIO * (1 + ASPECT_RATIO_TOLERANCE)
    return bool(lo <= ratio <= hi)
