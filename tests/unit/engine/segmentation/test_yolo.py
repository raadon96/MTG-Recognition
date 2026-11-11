"""YOLO's pre- and post-processing around the ONNX model, on plain arrays: the
letterboxed input, and the outlines decoded from the model's outputs.
"""

import cv2
import numpy as np

from mtg_recognition.engine.segmentation.yolo import letterbox, outlines_from_output

# A 1280 x 960 query image letterboxes to a 640 x 480 input, at half scale; the
# model's mask prototypes are a quarter of the input size
_QUERY_SHAPE = (960, 1280)
_INPUT_SHAPE = (480, 640)
_PROTOS_SHAPE = (32, 120, 160)


def test_the_model_input_has_a_long_side_of_640_and_grey_padding_to_a_multiple_of_32():
    query_image = np.zeros((530, 1000, 3), dtype=np.uint8)

    padded = letterbox(query_image)

    # 530 * 0.64 = 339 rows, padded to 352: 6 grey rows above, 7 below
    assert padded.shape == (352, 640, 3)
    assert (padded[:6] == 114).all()
    assert (padded[6:345] == 0).all()
    assert (padded[345:] == 114).all()


def test_an_image_already_at_the_model_input_size_is_left_unchanged():
    query_image = np.random.default_rng(0).integers(0, 256, (640, 480, 3), np.uint8)

    assert np.array_equal(letterbox(query_image), query_image)


def test_a_detection_above_conf_gives_its_mask_outline_in_query_image_pixels():
    # one prototype positive everywhere: the mask fills the box
    protos = np.zeros(_PROTOS_SHAPE, np.float32)
    protos[0] = 1.0
    output = _output([(160, 100, 320, 300, 0.9, 0)])

    [outline] = outlines_from_output(output, protos, _INPUT_SHAPE, _QUERY_SHAPE)

    x, y, w, h = cv2.boundingRect(outline.round().astype(np.int32))
    assert np.allclose([x, y, x + w, y + h], [320, 200, 640, 600], atol=4)


def test_detections_at_or_below_conf_give_no_outlines():
    protos = np.zeros(_PROTOS_SHAPE, np.float32)
    protos[0] = 1.0
    output = _output([(160, 100, 320, 300, 0.05, 0), (400, 100, 500, 300, 0.01, 0)])

    assert outlines_from_output(output, protos, _INPUT_SHAPE, _QUERY_SHAPE) == []


def test_overlapping_detections_count_once_whatever_their_prompt():
    protos = np.zeros(_PROTOS_SHAPE, np.float32)
    protos[0] = 1.0
    # the second overlaps the first by a few pixels, and scores lower
    output = _output([(160, 100, 320, 300, 0.9, 0), (310, 100, 400, 300, 0.8, 0)])
    output[4:6, 1] = [0.0, 0.8]  # found as the other prompt

    [outline] = outlines_from_output(output, protos, _INPUT_SHAPE, _QUERY_SHAPE)

    x, _, w, _ = cv2.boundingRect(outline.round().astype(np.int32))
    assert np.allclose([x, x + w], [320, 640], atol=4)


def test_separate_detections_each_give_an_outline():
    protos = np.zeros(_PROTOS_SHAPE, np.float32)
    protos[0] = 1.0
    output = _output([(160, 100, 320, 300, 0.9, 0), (400, 100, 500, 300, 0.8, 0)])

    outlines = outlines_from_output(output, protos, _INPUT_SHAPE, _QUERY_SHAPE)

    assert len(outlines) == 2


def test_a_mask_in_two_parts_gives_one_outline_around_both():
    # a prototype positive in two columns of the box, with a gap between them
    protos = np.zeros(_PROTOS_SHAPE, np.float32)
    protos[1, :, 40:55] = 1.0
    protos[1, :, 65:80] = 1.0
    output = _output([(160, 100, 320, 300, 0.9, 1)])

    [outline] = outlines_from_output(output, protos, _INPUT_SHAPE, _QUERY_SHAPE)

    x, _, w, _ = cv2.boundingRect(outline.round().astype(np.int32))
    assert np.allclose([x, x + w], [320, 640], atol=4)


def _output(
    detections: list[tuple[float, float, float, float, float, int]],
) -> np.ndarray:
    """The model's first output for these detections, (x1, y1, x2, y2, score,
    prototype) in input pixels: one column per anchor of [centre x, centre y,
    width, height, score per prompt, 32 coefficients], with the coefficient of
    the given prototype set to 1. Every other anchor scores 0.
    """
    output = np.zeros((4 + 2 + 32, 8400), np.float32)
    for anchor, (x1, y1, x2, y2, score, proto) in enumerate(detections):
        output[:4, anchor] = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        output[4, anchor] = score
        output[6 + proto, anchor] = 1.0
    return output
