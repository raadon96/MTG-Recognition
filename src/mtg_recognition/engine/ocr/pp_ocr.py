"""PP-OCR on onnxruntime: the text lines in an image, and what each says.

The models are the official ONNX exports of PP-OCRv5_mobile_det and
PP-OCRv6_tiny_rec (#218). The pre- and post-processing are ours, copied from
PaddleX's OCR pipeline with the settings PaddleOCR ran with, so the texts, scores
and boxes were the same as it gave (#199, docs/adr/0007), until #219 changed the
detection resize: resize and normalise,
detect, DB post-processing, reading order, straighten each line, recognise in
batches, CTC decoding.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
import yaml

from mtg_recognition.settings import settings

# Detection input: the long side scaled down to at most 960 px (PaddleX's
# limit_side_len 960, limit_type max), both sides rounded to multiples of 32;
# ImageNet normalisation. Detecting on the upscaled collector information crop at
# full size changed no benchmark result and was slower (docs/adr/0007);
# recognition still reads each line from the full-size image.
_DET_MAX_SIDE = 960
_DET_STRIDE = 32
_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# DB post-processing: text-map pixels above _THRESH form regions; a region is a
# text line when its mean score reaches _BOX_THRESH, and it is grown by
# _UNCLIP_RATIO since the model predicts shrunk lines. 0.05 instead of PaddleX's
# 0.3 finds the faint collector information.
_THRESH = 0.05
_BOX_THRESH = 0.6
_UNCLIP_RATIO = 1.5
_MAX_CANDIDATES = 1000
_MIN_SIZE = 3
# lines whose tops are this close (px) count as one line in reading order
_SAME_LINE_PX = 10

# Recognition input: 48 px high, at least 320 px and at most 3200 px wide
_REC_HEIGHT = 48
_REC_MIN_WIDTH = 320
_REC_MAX_WIDTH = 3200
_REC_BATCH = 6


@dataclass(slots=True)
class PPOCR:
    """The two models, and the characters recognition picks from."""

    detection: ort.InferenceSession
    recognition: ort.InferenceSession
    # index 0 is the CTC blank; the last is the space PaddleX appends
    characters: list[str]


@dataclass(slots=True)
class TextLines:
    """What read_text found, in reading order."""

    texts: list[str]
    scores: list[float]
    # (x1, y1, x2, y2): each line's bounding rectangle, in image pixels
    boxes: list[tuple[int, int, int, int]]


def load_pp_ocr() -> PPOCR:
    """The production models, on THREAD_COUNT threads."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = settings.THREAD_COUNT
    options.inter_op_num_threads = 1
    # errors only: the exports hold initializers no node uses, which onnxruntime
    # warns about on every load
    options.log_severity_level = 3
    detection_dir = settings.OCR_DETECTION_MODEL_DIR
    recognition_dir = settings.OCR_RECOGNITION_MODEL_DIR
    config = yaml.safe_load((recognition_dir / "inference.yml").read_text())
    return PPOCR(
        detection=ort.InferenceSession(
            detection_dir / "inference.onnx",
            options,
            providers=["CPUExecutionProvider"],
        ),
        recognition=ort.InferenceSession(
            recognition_dir / "inference.onnx",
            options,
            providers=["CPUExecutionProvider"],
        ),
        characters=["blank", *config["PostProcess"]["character_dict"], " "],
    )


def read_text(ocr: PPOCR, image: np.ndarray) -> TextLines:
    """Find the text lines of a BGR image and read each one."""
    # detect
    tensor = detection_input(image)
    session = ocr.detection
    [text_map] = session.run(None, {session.get_inputs()[0].name: tensor})
    quads = quads_from_text_map(np.asarray(text_map)[0, 0], image.shape[:2])

    # recognise each line, straightened
    lines = read_lines(ocr, [_crop_quad(image, quad) for quad in quads])
    return TextLines(
        texts=[text for text, _ in lines],
        scores=[score for _, score in lines],
        boxes=[
            (
                int(quad[:, 0].min()),
                int(quad[:, 1].min()),
                int(quad[:, 0].max()),
                int(quad[:, 1].max()),
            )
            for quad in quads
        ],
    )


def detection_input(image: np.ndarray) -> np.ndarray:
    """The image as the detection model's input: resized and normalised,
    (1, 3, H, W) float32.
    """
    # pad tiny images to 32 px with black
    h, w = image.shape[:2]
    if h + w < 64:
        padded = np.zeros((max(32, h), max(32, w), 3), np.uint8)
        padded[:h, :w] = image
        image = padded

    # scale, then round both sides to multiples of the stride
    h, w = image.shape[:2]
    ratio = min(1.0, _DET_MAX_SIDE / max(h, w))
    resize_h, resize_w = int(h * ratio), int(w * ratio)
    resize_h = max(round(resize_h / _DET_STRIDE) * _DET_STRIDE, _DET_STRIDE)
    resize_w = max(round(resize_w / _DET_STRIDE) * _DET_STRIDE, _DET_STRIDE)
    if (resize_h, resize_w) != (h, w):
        image = cv2.resize(image, (resize_w, resize_h))

    # BGR HWC uint8 to normalised CHW float, channels kept in BGR order as PaddleX
    # does
    tensor = (image.astype(np.float32) / 255 - _DET_MEAN) / _DET_STD
    return tensor.transpose(2, 0, 1)[None]


def quads_from_text_map(
    text_map: np.ndarray, image_shape: tuple[int, int]
) -> np.ndarray:
    """The text lines' quadrilaterals, (N, 4, 2) int16 in image pixels, clockwise
    from the top left, in reading order.
    """
    # DB post-processing: each region of the text map above _THRESH, kept when its
    # mean score reaches _BOX_THRESH, then grown by _UNCLIP_RATIO
    image_h, image_w = image_shape
    bitmap = ((text_map > _THRESH) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    scale_x, scale_y = image_w / text_map.shape[1], image_h / text_map.shape[0]
    quads = []
    for contour in contours[:_MAX_CANDIDATES]:
        points, short_side = _min_area_quad(contour)
        if short_side < _MIN_SIZE:
            continue
        if _mean_score(text_map, points) < _BOX_THRESH:
            continue
        distance = cv2.contourArea(points) * _UNCLIP_RATIO / cv2.arcLength(points, True)
        # pyclipper's names come from a star import of its extension, which ty
        # can't see
        offset = pyclipper.PyclipperOffset()  # ty: ignore[unresolved-attribute]
        offset.AddPath(
            points,
            pyclipper.JT_ROUND,  # ty: ignore[unresolved-attribute]
            pyclipper.ET_CLOSEDPOLYGON,  # ty: ignore[unresolved-attribute]
        )
        grown = np.array(offset.Execute(distance)).reshape(-1, 1, 2)
        quad, short_side = _min_area_quad(grown)
        if short_side < _MIN_SIZE + 2:
            continue
        quad[:, 0] = np.clip(np.round(quad[:, 0] * scale_x), 0, image_w)
        quad[:, 1] = np.clip(np.round(quad[:, 1] * scale_y), 0, image_h)
        quads.append(quad.astype(np.int16))
    if not quads:
        return np.zeros((0, 4, 2), dtype=np.int16)

    # reading order: by top-left corner, then neighbours on one line (tops within
    # _SAME_LINE_PX) swapped into left-to-right order, as PaddleX does
    quads.sort(key=lambda q: (q[0][1], q[0][0]))
    for i in range(len(quads) - 1):
        for j in range(i, -1, -1):
            a, b = quads[j], quads[j + 1]
            if abs(int(b[0][1]) - int(a[0][1])) < _SAME_LINE_PX and b[0][0] < a[0][0]:
                quads[j], quads[j + 1] = b, a
            else:
                break
    return np.array(quads)


def read_lines(ocr: PPOCR, lines: list[np.ndarray]) -> list[tuple[str, float]]:
    """The text of each straightened text line, and its score."""
    # PaddleX reads lines in batches sorted by aspect ratio, each padded to its
    # widest line; the padding changes the output, so it is kept
    order = np.argsort([line.shape[1] / line.shape[0] for line in lines], kind="stable")
    session = ocr.recognition
    results: list[tuple[str, float]] = [("", 0.0)] * len(lines)
    for start in range(0, len(lines), _REC_BATCH):
        batch = [
            _recognition_input(lines[i]) for i in order[start : start + _REC_BATCH]
        ]
        width = max(x.shape[2] for x in batch)
        tensor = np.zeros((len(batch), 3, _REC_HEIGHT, width), dtype=np.float32)
        for k, x in enumerate(batch):
            tensor[k, :, :, : x.shape[2]] = x
        probs = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        for k, line_probs in enumerate(np.asarray(probs)):
            results[order[start + k]] = ctc_decode(line_probs, ocr.characters)
    return results


def ctc_decode(probs: np.ndarray, characters: list[str]) -> tuple[str, float]:
    """One line's text from its character probabilities, (steps, characters), and
    the mean probability of the characters kept.
    """
    # the most likely character at each step; repeats, then blanks, dropped
    indices = probs.argmax(axis=-1)
    keep = np.ones(len(indices), dtype=bool)
    keep[1:] = indices[1:] != indices[:-1]
    keep &= indices != 0
    text = "".join(characters[i] for i in indices[keep])
    confidences = probs.max(axis=-1)[keep]
    score = float(confidences.mean()) if len(confidences) else 0.0
    return text, score


def _recognition_input(line: np.ndarray) -> np.ndarray:
    """One line scaled to 48 px high and normalised to [-1, 1], (3, 48, W), padded
    on the right to at least 320 px.
    """
    h, w = line.shape[:2]
    width = int(_REC_HEIGHT * max(_REC_MIN_WIDTH / _REC_HEIGHT, w / h))
    if width > _REC_MAX_WIDTH:
        resized_w = width = _REC_MAX_WIDTH
    else:
        resized_w = min(width, math.ceil(_REC_HEIGHT * w / h))
    resized = cv2.resize(line, (resized_w, _REC_HEIGHT))
    x = np.zeros((3, _REC_HEIGHT, width), dtype=np.float32)
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
    return np.array([top_left, top_right, bottom_right, bottom_left]), min(rect[1])


def _mean_score(text_map: np.ndarray, quad: np.ndarray) -> float:
    """The mean of the text map inside quad."""
    h, w = text_map.shape
    x_min = int(np.clip(math.floor(quad[:, 0].min()), 0, w - 1))
    x_max = int(np.clip(math.ceil(quad[:, 0].max()), 0, w - 1))
    y_min = int(np.clip(math.floor(quad[:, 1].min()), 0, h - 1))
    y_max = int(np.clip(math.ceil(quad[:, 1].max()), 0, h - 1))
    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    shifted = quad - [x_min, y_min]
    cv2.fillPoly(mask, [shifted.astype(np.int32)], (1,))
    return cv2.mean(text_map[y_min : y_max + 1, x_min : x_max + 1], mask)[0]


def _crop_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """The text line inside quad, straightened; a line at least 1.5 times taller
    than wide is turned on its side.
    """
    quad, _ = _min_area_quad(quad.astype(np.int32))
    quad = quad.astype(np.float32)
    width = int(
        max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3]))
    )
    height = int(
        max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2]))
    )
    target = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
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
