"""Throwaway prototype for the ticket "Test PP-OCR in onnxruntime without
paddlepaddle" (#199) of the map #189. Never merged.

Reads text off an image with the PP-OCR ONNX exports and onnxruntime alone: no
paddlepaddle, paddlex, paddleocr or rapidocr. It reproduces what PaddleX's OCR
pipeline does with today's settings (no document preprocessing, no textline
orientation), so its size is what production would carry instead of a library.
Needs onnxruntime, numpy, OpenCV, pyclipper and PyYAML.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
import yaml

# PaddleOCR's OCR pipeline settings (paddlex/configs/pipelines/OCR.yaml)
DET_LIMIT_SIDE_LEN = 64
DET_MAX_SIDE_LIMIT = 4000
DET_MAX_CANDIDATES = 1000
DET_MIN_SIZE = 3
DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
REC_HEIGHT = 48
REC_MIN_WIDTH = 320
REC_MAX_WIDTH = 3200
REC_BATCH = 6
# boxes whose tops are this close (px) count as one line when sorting
SAME_LINE_PX = 10


@dataclass(slots=True)
class OwnOCR:
    det: ort.InferenceSession
    rec: ort.InferenceSession
    # index 0 is the CTC blank; the last is the space PaddleX appends
    characters: list[str]


@dataclass(slots=True)
class TextLines:
    texts: list[str]
    scores: list[float]
    # (x1, y1, x2, y2): each detected quadrilateral's bounding rectangle
    boxes: np.ndarray


def load(det_dir: Path, rec_dir: Path, threads: int) -> OwnOCR:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.log_severity_level = 3
    config = yaml.safe_load((rec_dir / "inference.yml").read_text())
    return OwnOCR(
        det=ort.InferenceSession(str(det_dir / "inference.onnx"), options),
        rec=ort.InferenceSession(str(rec_dir / "inference.onnx"), options),
        characters=["blank", *config["PostProcess"]["character_dict"], " "],
    )


def read_text(
    ocr: OwnOCR,
    image: np.ndarray,
    thresh: float,
    box_thresh: float,
    unclip_ratio: float,
) -> TextLines:
    """Detect the text lines of a BGR image and read each one, top to bottom,
    left to right.
    """
    quads = detect(ocr.det, image, thresh, box_thresh, unclip_ratio)
    lines = recognize(ocr, [_crop_quad(image, quad) for quad in quads])
    texts = [text for text, _ in lines]
    scores = [score for _, score in lines]
    boxes = (
        np.concatenate([quads.min(axis=1), quads.max(axis=1)], axis=1).astype(np.int16)
        if len(quads)
        else np.zeros((0, 4), dtype=np.int16)
    )
    return TextLines(texts=texts, scores=scores, boxes=boxes)


def detect(
    det: ort.InferenceSession,
    image: np.ndarray,
    thresh: float,
    box_thresh: float,
    unclip_ratio: float,
) -> np.ndarray:
    """The text lines' quadrilaterals, (N, 4, 2) in image pixels, in reading order."""
    # pad tiny images, then scale so the short side reaches the limit, both sides
    # multiples of 32
    h, w = image.shape[:2]
    if h + w < 64:
        padded = np.zeros((max(32, h), max(32, w), 3), np.uint8)
        padded[:h, :w] = image
        image = padded
    src_h, src_w = image.shape[:2]
    ratio = (
        DET_LIMIT_SIDE_LEN / min(src_h, src_w)
        if min(src_h, src_w) < DET_LIMIT_SIDE_LEN
        else 1.0
    )
    resize_h, resize_w = int(src_h * ratio), int(src_w * ratio)
    if max(resize_h, resize_w) > DET_MAX_SIDE_LIMIT:
        ratio = DET_MAX_SIDE_LIMIT / max(resize_h, resize_w)
        resize_h, resize_w = int(resize_h * ratio), int(resize_w * ratio)
    resize_h = max(round(resize_h / 32) * 32, 32)
    resize_w = max(round(resize_w / 32) * 32, 32)
    if (resize_h, resize_w) != (src_h, src_w):
        image = cv2.resize(image, (resize_w, resize_h))

    x = (image.astype(np.float32) / 255 - DET_MEAN) / DET_STD
    x = x.transpose(2, 0, 1)[None]
    pred = det.run(None, {det.get_inputs()[0].name: x})[0][0, 0]

    # DB post-processing: each connected region of the text map above thresh, kept
    # when its mean score passes box_thresh, then grown by unclip_ratio
    bitmap = ((pred > thresh) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    scale_x, scale_y = src_w / pred.shape[1], src_h / pred.shape[0]
    quads = []
    for contour in contours[:DET_MAX_CANDIDATES]:
        points, short_side = _min_area_quad(contour)
        if short_side < DET_MIN_SIZE:
            continue
        if _mean_score(pred, points) < box_thresh:
            continue
        area = cv2.contourArea(points)
        distance = area * unclip_ratio / cv2.arcLength(points, True)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        grown = np.array(offset.Execute(distance)).reshape(-1, 1, 2)
        quad, short_side = _min_area_quad(grown)
        if short_side < DET_MIN_SIZE + 2:
            continue
        quad[:, 0] = np.clip(np.round(quad[:, 0] * scale_x), 0, src_w)
        quad[:, 1] = np.clip(np.round(quad[:, 1] * scale_y), 0, src_h)
        quads.append(quad.astype(np.int16))
    if not quads:
        return np.zeros((0, 4, 2), dtype=np.int16)

    # reading order: by top-left corner, then swap neighbours on one line (tops
    # within SAME_LINE_PX) into left-to-right order, as PaddleX does
    quads.sort(key=lambda q: (q[0][1], q[0][0]))
    for i in range(len(quads) - 1):
        for j in range(i, -1, -1):
            a, b = quads[j], quads[j + 1]
            if abs(int(b[0][1]) - int(a[0][1])) < SAME_LINE_PX and b[0][0] < a[0][0]:
                quads[j], quads[j + 1] = b, a
            else:
                break
    return np.array(quads)


def recognize(ocr: OwnOCR, lines: list[np.ndarray]) -> list[tuple[str, float]]:
    """The text of each cropped text line, and its mean character confidence."""
    # PaddleX reads lines in batches of REC_BATCH sorted by width, each padded to
    # the widest; the padding changes the output, so it is kept
    order = np.argsort([line.shape[1] / line.shape[0] for line in lines], kind="stable")
    results: list[tuple[str, float]] = [("", 0.0)] * len(lines)
    for start in range(0, len(lines), REC_BATCH):
        batch = [_rec_input(lines[i]) for i in order[start : start + REC_BATCH]]
        width = max(x.shape[2] for x in batch)
        x = np.zeros((len(batch), 3, REC_HEIGHT, width), dtype=np.float32)
        for k, item in enumerate(batch):
            x[k, :, :, : item.shape[2]] = item
        probs = ocr.rec.run(None, {ocr.rec.get_inputs()[0].name: x})[0]

        # CTC decoding: drop repeats, then blanks
        for k, line_probs in enumerate(probs):
            indices = line_probs.argmax(axis=-1)
            keep = np.ones(len(indices), dtype=bool)
            keep[1:] = indices[1:] != indices[:-1]
            keep &= indices != 0
            text = "".join(ocr.characters[i] for i in indices[keep])
            confidences = line_probs.max(axis=-1)[keep]
            score = float(confidences.mean()) if len(confidences) else 0.0
            results[order[start + k]] = (text, score)
    return results


def _rec_input(line: np.ndarray) -> np.ndarray:
    """One line scaled to 48 px high and normalised, padded on the right to at
    least 320 px.
    """
    h, w = line.shape[:2]
    width = int(REC_HEIGHT * max(REC_MIN_WIDTH / REC_HEIGHT, w / h))
    if width > REC_MAX_WIDTH:
        resized, resized_w, width = (
            cv2.resize(line, (REC_MAX_WIDTH, REC_HEIGHT)),
            REC_MAX_WIDTH,
            REC_MAX_WIDTH,
        )
    else:
        resized_w = min(width, math.ceil(REC_HEIGHT * w / h))
        resized = cv2.resize(line, (resized_w, REC_HEIGHT))
    x = np.zeros((3, REC_HEIGHT, width), dtype=np.float32)
    x[:, :, :resized_w] = (
        resized.astype(np.float32).transpose(2, 0, 1) / 255 - 0.5
    ) / 0.5
    return x


def _min_area_quad(contour: np.ndarray) -> tuple[np.ndarray, float]:
    """The contour's minimum-area rectangle as 4 corners, clockwise from the top
    left, and its short side.
    """
    rect = cv2.minAreaRect(contour)
    p = sorted(cv2.boxPoints(rect), key=lambda point: point[0])
    # on equal heights PaddleX takes the second point as the top one
    top_left, bottom_left = (p[0], p[1]) if p[1][1] > p[0][1] else (p[1], p[0])
    top_right, bottom_right = (p[2], p[3]) if p[3][1] > p[2][1] else (p[3], p[2])
    quad = np.array([top_left, top_right, bottom_right, bottom_left])
    return quad, min(rect[1])


def _mean_score(pred: np.ndarray, quad: np.ndarray) -> float:
    """The mean of the text map inside quad, over its bounding rectangle."""
    h, w = pred.shape
    x_min = int(np.clip(math.floor(quad[:, 0].min()), 0, w - 1))
    x_max = int(np.clip(math.ceil(quad[:, 0].max()), 0, w - 1))
    y_min = int(np.clip(math.floor(quad[:, 1].min()), 0, h - 1))
    y_max = int(np.clip(math.ceil(quad[:, 1].max()), 0, h - 1))
    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    shifted = quad - [x_min, y_min]
    cv2.fillPoly(mask, shifted.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(pred[y_min : y_max + 1, x_min : x_max + 1], mask)[0]


def _crop_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """The text line inside quad, straightened; lines taller than wide by 1.5 are
    turned on their side.
    """
    quad, _ = _min_area_quad(quad.astype(np.int32))
    quad = quad.astype(np.float32)
    width = int(
        max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3]))
    )
    height = int(
        max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2]))
    )
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    crop = cv2.warpPerspective(
        image,
        cv2.getPerspectiveTransform(quad, target),
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if crop.shape[0] / crop.shape[1] >= 1.5:
        crop = np.rot90(crop)
    return crop
