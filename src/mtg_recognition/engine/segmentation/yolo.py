"""YOLOE-11s on onnxruntime: the outline of each object it finds in an image.

The model is the ONNX export written by scripts/export_segmentation.py, with
the prompts fused in. The pre- and post-processing are ours, copied from
ultralytics 8.4.21 so the outlines are the same as it gave (docs/adr/0007):
letterbox, run, filter by confidence, NMS, decode the masks, trace and scale
the outlines back to the image.
"""

from functools import cache

import cv2
import numpy as np
import onnxruntime as ort

from mtg_recognition.settings import settings

# The model's input: the long side scaled to 640, the short side padded with
# grey to the next multiple of the model's stride
_INPUT_SIZE = 640
_STRIDE = 32
_PAD_VALUE = 114

# 0.1 missed cards on the 9-card scan with border gaps (#195)
_CONF = 0.05
# NMS drops any detection that overlaps a kept one, whatever its prompt: a card
# found as both "book cover" and "art work" counts once
_IOU = 0.0
_MAX_DETECTIONS = 50


@cache
def yolo_session() -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = settings.THREAD_COUNT
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        settings.YOLO_SEG_MODEL_PATH, options, providers=["CPUExecutionProvider"]
    )


def yolo_outlines(session: ort.InferenceSession, image: np.ndarray) -> list[np.ndarray]:
    """Each object's outline in a BGR image, (N, 2) float32 in image pixels."""
    padded = letterbox(image)
    # BGR HWC uint8 to RGB CHW float in [0, 1]
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    output, protos = session.run(None, {session.get_inputs()[0].name: tensor})
    return outlines_from_output(
        np.asarray(output)[0],
        np.asarray(protos)[0],
        padded.shape[:2],
        image.shape[:2],
    )


def letterbox(image: np.ndarray) -> np.ndarray:
    """The image as the model's input: scaled and padded, still BGR uint8."""
    h, w = image.shape[:2]
    gain = min(_INPUT_SIZE / h, _INPUT_SIZE / w)
    new_w, new_h = round(w * gain), round(h * gain)
    if (new_w, new_h) != (w, h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # split the padding between both sides; the ±0.1 puts the odd pixel below
    # and right, as ultralytics does
    pad_w = (_INPUT_SIZE - new_w) % _STRIDE / 2
    pad_h = (_INPUT_SIZE - new_h) % _STRIDE / 2
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    return cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(_PAD_VALUE,) * 3
    )


def outlines_from_output(
    output: np.ndarray,
    protos: np.ndarray,
    input_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> list[np.ndarray]:
    """Each detection's outline, scaled back from the model's input to the image.

    output is [4 box + 1 score per prompt + 32 coefficients, anchors], boxes as
    centre and size in input pixels; protos is [32, input / 4].
    """
    # detections above conf, boxes as corners
    detections = output.T
    prompts = detections.shape[1] - 4 - protos.shape[0]
    scores = detections[:, 4 : 4 + prompts].max(axis=1)
    keep = scores > _CONF
    detections, scores = detections[keep], scores[keep]
    xy, wh = detections[:, :2], detections[:, 2:4]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)

    kept = _nms(boxes, scores)[:_MAX_DETECTIONS]
    boxes, coefficients = boxes[kept], detections[kept, 4 + prompts :]

    n_protos, mh, mw = protos.shape
    ih, iw = input_shape
    h, w = image_shape
    gain = min(ih / h, iw / w)
    outlines = []
    for box, coeffs in zip(boxes, coefficients, strict=True):
        # mask: coefficients times prototypes, zeroed outside the box at
        # prototype size, upsampled to the input size, thresholded at logit 0
        mask = (coeffs @ protos.reshape(n_protos, -1)).reshape(mh, mw)
        x1, y1, x2, y2 = np.round(box * [mw / iw, mh / ih, mw / iw, mh / ih]).astype(
            int
        )
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)
        cropped = np.zeros_like(mask)
        cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        binary = (
            cv2.resize(cropped, (iw, ih), interpolation=cv2.INTER_LINEAR) > 0
        ).astype(np.uint8)
        if not binary.any():
            continue

        # outline: a mask in several parts is joined into one outline
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 1:
            outline = np.concatenate(
                _merge_multi_segment([contour.reshape(-1, 2) for contour in contours])
            ).astype(np.float32)
        else:
            outline = contours[0].reshape(-1, 2).astype(np.float32)

        # back to image pixels: remove the padding, undo the scaling
        outline[:, 0] = np.clip((outline[:, 0] - (iw - w * gain) / 2) / gain, 0, w)
        outline[:, 1] = np.clip((outline[:, 1] - (ih - h * gain) / 2) / gain, 0, h)
        outlines.append(outline)
    return outlines


def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    """Greedy NMS across prompts, as torchvision.ops.nms: the indices of the
    boxes kept, best score first, dropping a box whose IoU with a kept box is
    above _IOU."""
    order = np.argsort(-scores, kind="stable")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kept = []
    while len(order):
        i = order[0]
        kept.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        overlap = inter / (areas[i] + areas[rest] - inter)
        order = rest[overlap <= _IOU]
    return kept


def _merge_multi_segment(segments: list[np.ndarray]) -> list[np.ndarray]:
    """Copied unchanged from ultralytics 8.4.21 (ultralytics.data.converter),
    AGPL-3.0 like the model (docs/adr/0007): joins several contours into one by
    connecting their closest points."""
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]
    for i in range(1, len(segments)):
        dis = ((segments[i - 1][:, None, :] - segments[i][None, :, :]) ** 2).sum(-1)
        idx1, idx2 = np.unravel_index(np.argmin(dis, axis=None), dis.shape)
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)
    for k in range(2):
        if k == 0:
            for i, idx in enumerate(idx_list):
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]
                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])
        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s
