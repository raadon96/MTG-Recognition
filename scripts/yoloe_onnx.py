"""Throwaway: run a YOLOE ONNX export in onnxruntime without ultralytics, for the
ticket "Compare segmentation candidates: YOLOE-11 and YOLOE-26, as .pt and ONNX"
(#195). Never merged.

Copies ultralytics 8.4.21's steps: letterbox, normalise, run, filter (and NMS
for YOLOE-11's one-to-many output), decode the masks from the prototypes, find
the outlines, scale them back to the query image. `predict` returns an object
shaped like ultralytics' result, so `segment()` runs unchanged on it.
"""

from types import SimpleNamespace

import cv2
import numpy as np
import onnxruntime as ort

STRIDE = 32
PAD_VALUE = 114


class OnnxYoloe:
    def __init__(self, path: str, threads: int = 0) -> None:
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            path, options, providers=["CPUExecutionProvider"]
        )
        shape = self.session.get_inputs()[0].shape
        # a static export has fixed sizes; a dynamic one has names
        self.static_size = shape[2] if isinstance(shape[2], int) else None

    def predict(self, image: np.ndarray, **kwargs) -> list:
        outlines = yoloe_outlines(
            self.session,
            image,
            conf=kwargs.get("conf", 0.25),
            iou=kwargs.get("iou", 0.7),
            imgsz=self.static_size or kwargs.get("imgsz", 640),
            static=self.static_size is not None,
            max_det=kwargs.get("max_det", 50),
        )
        masks = SimpleNamespace(xy=outlines) if outlines else None
        return [SimpleNamespace(masks=masks)]


def yoloe_outlines(
    session: ort.InferenceSession,
    image: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    static: bool,
    max_det: int,
) -> list[np.ndarray]:
    """Each found object's outline, (N, 2) float32 in query image pixels."""
    # letterbox: scale the long side to imgsz, pad to a square (static) or to
    # the next multiple of the stride (dynamic, as ultralytics' rect mode)
    h, w = image.shape[:2]
    gain = min(imgsz / h, imgsz / w)
    new_w, new_h = round(w * gain), round(h * gain)
    pad_w, pad_h = imgsz - new_w, imgsz - new_h
    if not static:
        pad_w, pad_h = pad_w % STRIDE, pad_h % STRIDE
    pad_w, pad_h = pad_w / 2, pad_h / 2
    resized = image
    if (w, h) != (new_w, new_h):
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )

    # BGR HWC uint8 to RGB CHW float in [0, 1]
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    output, protos = session.run(None, {session.get_inputs()[0].name: tensor})
    protos = protos[0]

    # rows of [x1, y1, x2, y2, score, 32 coefficients]
    # one-to-many output is [rows, 8400 anchors]; one-to-one is [max_det, row]
    if output.shape[1] > output.shape[2]:
        # one-to-one head: already top-k, boxes as corners, scores as sigmoid
        rows = output[0]
        rows = rows[rows[:, 4] > conf]
        boxes, coefficients = rows[:, :4], rows[:, 6:]
    else:
        # one-to-many head: [4 + classes + 32, anchors], boxes as centre and size
        preds = output[0].T
        classes = preds.shape[1] - 4 - protos.shape[0]
        scores = preds[:, 4 : 4 + classes].max(axis=1)
        keep = scores > conf
        preds, scores = preds[keep], scores[keep]
        xy, wh = preds[:, :2], preds[:, 2:4]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
        kept = _nms(boxes, scores, iou)[:max_det]
        boxes, coefficients = boxes[kept], preds[kept, 4 + classes :]

    # masks: coefficients times prototypes, zeroed outside the box at prototype
    # size, upsampled to the input size, thresholded at logit 0
    c, mh, mw = protos.shape
    ph, pw = padded.shape[:2]
    outlines = []
    for box, coeffs in zip(boxes, coefficients, strict=True):
        mask = (coeffs @ protos.reshape(c, -1)).reshape(mh, mw)
        x1, y1, x2, y2 = np.round(box * [mw / pw, mh / ph, mw / pw, mh / ph]).astype(
            int
        )
        cropped = np.zeros_like(mask)
        cropped[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = mask[
            max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)
        ]
        binary = (
            cv2.resize(cropped, (pw, ph), interpolation=cv2.INTER_LINEAR) > 0
        ).astype(np.uint8)
        if not binary.any():
            continue
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
            0
        ]
        if len(contours) > 1:
            from ultralytics.data.converter import merge_multi_segment

            outline = np.concatenate(
                merge_multi_segment([x.reshape(-1, 2) for x in contours])
            ).astype(np.float32)
        else:
            outline = contours[0].reshape(-1, 2).astype(np.float32)
        # back to the query image, as ultralytics' scale_coords
        gain_back = min(ph / h, pw / w)
        outline[:, 0] = np.clip(
            (outline[:, 0] - (pw - w * gain_back) / 2) / gain_back, 0, w
        )
        outline[:, 1] = np.clip(
            (outline[:, 1] - (ph - h * gain_back) / 2) / gain_back, 0, h
        )
        outlines.append(outline)
    return outlines


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy NMS across classes, as torchvision.ops.nms: drops a box whose IoU
    with a kept box is above `iou`."""
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
        order = rest[overlap <= iou]
    return kept
