import cv2
import numpy as np


def find_corners_lines_intersect(
    contour_points: np.ndarray, corner_points: np.ndarray
) -> np.ndarray:
    # Flatten arrays for vectorized operations
    # contour_xy: (N, 2), corners_xy: (4, 2)
    contour_xy = contour_points.reshape(-1, 2)
    corners_xy = corner_points.reshape(-1, 2)

    # print(f"len(contour_xy) = {len(contour_xy)}")
    if len(contour_xy) < 20:  # Not enough points to refine meaningfully
        return corner_points

    # OPTIMIZATION 1: Vectorized Distance Calculation
    # Instead of looping 4 times, we calculate distances in one shot using broadcasting.
    # diff shape: (4, N, 2) -> 4 corners, N contour points, 2 coords (x,y)
    diff = contour_xy[np.newaxis, :, :] - corners_xy[:, np.newaxis, :]

    # Squared Euclidean distance (avoiding sqrt for speed)
    # dists shape: (4, N)
    dists_sq = np.sum(diff**2, axis=2)

    # Get indices of the closest contour point for each corner
    # indicies shape: (4,)
    indices = np.argmin(dists_sq, axis=1)
    indices.sort()  # Ensure they are in order of the contour

    # OPTIMIZATION 2: Vectorized Split (Rotation instead of wrapping loops)
    # We rotate the array so the first segment starts at index 0
    shift = indices[0]
    contour_rolled = np.roll(contour_xy, -shift, axis=0)

    # Adjust indices relative to the new start
    split_indices = indices - shift

    # Split into 4 segments.
    # Note: split_indices[0] is 0, so the first split is empty (dummy),
    # but np.split handles the rest correctly.
    # We effectively want ranges: [0:idx1], [idx1:idx2], [idx2:idx3], [idx3:]
    # Since we rolled, the first corner is at 0. The segments are between corners.
    segments = np.split(contour_rolled, split_indices[1:])

    # segments[0]: Corner 0 to Corner 1
    # segments[1]: Corner 1 to Corner 2
    # segments[2]: Corner 2 to Corner 3
    # segments[3]: Corner 3 to Corner 0 (wrapped end)

    refined_corners = []
    lines = []

    # Process each segment (Fitting lines)
    for i, seg in enumerate(segments):
        # Safety checks for tiny segments
        if len(seg) < 10:
            vx, vy = seg[-1] - seg[0]
            lines.append((vx, vy, seg[0][0], seg[0][1]))
            continue

        # Remove rounded corners (15% trim)
        trim = int(len(seg) * 0.15)
        if trim > 0 and len(seg) > 2 * trim:
            seg_trimmed = seg[trim:-trim]
        else:
            seg_trimmed = seg

        # fitLine expects points in a specific shape, usually (N, 1, 2) or (N, 2)
        # We ensure float32 for OpenCV compatibility
        line = cv2.fitLine(seg_trimmed.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        lines.append(line.flatten())

    # Compute intersections
    for i in range(4):
        line1 = lines[i]
        line2 = lines[(i + 1) % 4]

        pt = compute_intersection(line1, line2)
        if pt is None:
            return corner_points  # Fallback
        refined_corners.append(pt)

    corner_points = np.array(refined_corners, dtype=np.int32).reshape(4, 1, 2)
    return corner_points


def compute_intersection(line1, line2):
    """
    Finds the intersection point of two lines.
    Each line is given by (vx, vy, x0, y0) from cv2.fitLine.
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

    if abs(det) < 1e-6:
        return None  # Parallel lines

    dx = x2 - x1
    dy = y2 - y1

    t1 = (dx * -vy2 - dy * -vx2) / det

    x = x1 + t1 * vx1
    y = y1 + t1 * vy1

    return [float(x), float(y)]


def crop_img(img, bbox, padding_pct=0.1):
    """
    Crops the image with padding based on the bbox.
    """
    # bbox = results[0].boxes.xyxy.numpy().ravel().astype(np.int64)

    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    pad_w = int(padding_pct * w)
    pad_h = int(padding_pct * h)

    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(w_img, x2 + pad_w)
    cy2 = min(h_img, y2 + pad_h)

    # Return crop and the offset coordinates (to map back later)
    return img[cy1:cy2, cx1:cx2].copy(), (cx1, cy1, cx2, cy2)


def order_points(pts: list[tuple[float, float]]) -> list[list[int]]:
    """
    Rearrange coordinates to order: top-left, top-right, bottom-right, bottom-left

    Args:
        pts (List[Tuple[float, float]]): List of 4 points in the format of (x, y)

    Returns:
        List[List[int]]: List of 4 points in the format of [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)

    # Top-left has smallest sum, Bottom-right has largest
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right has smallest diff, Bottom-left has largest
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect.astype("int").tolist()


def find_dest(pts):
    """
    Calculate the destination dimensions (width/height) for perspective transform
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
    destination_corners = order_points(destination_corners)

    return destination_corners


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
        4 corner points of the contour in the image

    Returns
    -------
    np.ndarray
        Warped image
    """
    assert corners.shape == (4, 1, 2)

    corners = sorted(np.concatenate(corners).tolist())
    corners = order_points(corners)

    dest_corners = find_dest(corners)
    w, h = dest_corners[2][0], dest_corners[2][1]

    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(dest_corners))
    warped = cv2.warpPerspective(img, M, (w, h))

    return warped


def sharpen(img_bgr):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_bgr = cv2.filter2D(img_bgr, -1, kernel)
    return img_bgr
