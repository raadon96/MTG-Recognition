"""Geometric helpers for card corner detection and perspective correction.

All coordinates follow OpenCV's ``(x, y)`` convention. Corner arrays use the
OpenCV contour format: shape ``(N, 1, 2)`` int32 unless stated otherwise.
The canonical four-corner order throughout is: top-left, top-right,
bottom-right, bottom-left (as produced by :func:`order_points`).
"""

from collections.abc import Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_MIN_CONTOUR_POINTS: int = 20  # below this, Hough corner detection is unreliable
_HOUGH_MIN_LENGTH_RATIO: float = (
    0.10  # minLineLength as a fraction of the canvas diagonal
)
_HOUGH_MAX_GAP_RATIO: float = 0.05  # maxLineGap as a fraction of the canvas diagonal
_ARC_TRIM_FRACTION: float = 0.15  # fraction of arc endpoints trimmed before line fit
_PARALLEL_DET_THRESHOLD: float = 1e-6  # determinant threshold for near-parallel lines


def find_corners_lines_intersect(contour_points: np.ndarray) -> np.ndarray | None:
    """
    Detect the 4 card corners from a contour using Hough line detection.
    Falls back to a farthest-point recursive split if Hough fails.

    Parameters
    ----------
    contour_points : np.ndarray
        Contour in OpenCV format, shape (N, 1, 2) or (N, 2).

    Returns
    -------
    np.ndarray of shape (4, 1, 2) int32, or None if detection fails.
    """
    contour_xy = contour_points.reshape(-1, 2)
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

    corners = []
    for k in range(4):
        pt = compute_intersection(lines[k], lines[(k + 1) % 4])
        if pt is None:
            return None
        corners.append(pt)

    return np.array(corners, dtype=np.int32).reshape(4, 1, 2)


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


def order_points(pts: Sequence[Sequence[float]]) -> list[list[int]]:
    """
    Rearrange coordinates to order: top-left, top-right, bottom-right, bottom-left

    Args:
        pts (Sequence[Sequence[float]]): 4 points in the format of (x, y)

    Returns:
        List[List[int]]: List of 4 points in the format of
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    pts_arr = np.array(pts)
    s = pts_arr.sum(axis=1)

    # Top-left has smallest sum, Bottom-right has largest
    rect[0] = pts_arr[np.argmin(s)]
    rect[2] = pts_arr[np.argmax(s)]

    diff = np.diff(pts_arr, axis=1).ravel()
    # Top-right has smallest diff, Bottom-left has largest
    rect[1] = pts_arr[np.argmin(diff)]
    rect[3] = pts_arr[np.argmax(diff)]

    return rect.astype(np.intp).tolist()


def find_dest(pts: list[list[int]]) -> list[list[int]]:
    """
    Calculate the destination dimensions (width/height) for perspective transform.

    Parameters
    ----------
    pts : list[list[int]]
        Four corner points in top-left, top-right, bottom-right, bottom-left order
        (as returned by :func:`order_points`).

    Returns
    -------
    list[list[int]]
        Destination corners in the same canonical order.
    """
    (tl, tr, br, bl) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    destination_corners = [
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight],
    ]
    # order_points is redundant here (corners are already canonical), but kept
    # to ensure the output contract is satisfied even for degenerate inputs.
    return order_points(destination_corners)


def perspective_transform(
    img: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """
    Warps the perspective of the found contour from the original image.

    Parameters
    ----------
    img : np.ndarray
        Input image
    corners : np.ndarray
        4 corner points of the contour in the image, shape (4, 1, 2).

    Returns
    -------
    np.ndarray
        Warped image
    """
    if corners.shape != (4, 1, 2):
        raise ValueError(f"corners must have shape (4, 1, 2), got {corners.shape}")

    ordered_corners = order_points(np.concatenate(corners).tolist())

    dest_corners = find_dest(ordered_corners)
    w, h = dest_corners[2][0], dest_corners[2][1]

    M = cv2.getPerspectiveTransform(
        np.array(ordered_corners, dtype=np.float32),
        np.array(dest_corners, dtype=np.float32),
    )
    warped = cv2.warpPerspective(img, M, (w, h))

    return warped
