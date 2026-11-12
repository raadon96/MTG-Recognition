"""PP-OCR's pre- and post-processing around the ONNX models, on plain arrays: the
detection input, the text lines found in a text map, and the text decoded from
the recognition model's character probabilities.
"""

import numpy as np
import pytest

from mtg_recognition.engine.ocr.pp_ocr import (
    ctc_decode,
    detection_input,
    quads_from_text_map,
)

# index 0 is the CTC blank
_CHARACTERS = ["blank", "a", "b", " "]


def _probabilities(steps: list[tuple[int, float]]) -> np.ndarray:
    """One line's character probabilities: at each step, the given character
    with the given probability, the rest shared out among the others.
    """
    probs = np.zeros((len(steps), len(_CHARACTERS)), dtype=np.float32)
    for t, (index, p) in enumerate(steps):
        probs[t] = (1 - p) / (len(_CHARACTERS) - 1)
        probs[t, index] = p
    return probs


def test_repeated_characters_count_once_unless_a_blank_separates_them():
    probs = _probabilities([(1, 0.9), (1, 0.9), (0, 0.9), (1, 0.9), (2, 0.9)])

    text, _ = ctc_decode(probs, _CHARACTERS)

    assert text == "aab"


def test_the_score_is_the_mean_probability_of_the_characters_kept():
    # the blank's and the repeat's probabilities don't count
    probs = _probabilities([(1, 0.8), (1, 0.5), (0, 0.9), (2, 0.6)])

    _, score = ctc_decode(probs, _CHARACTERS)

    assert score == pytest.approx(0.7)


def test_a_line_of_blanks_reads_as_empty_with_score_0():
    probs = _probabilities([(0, 0.9), (0, 0.9)])

    assert ctc_decode(probs, _CHARACTERS) == ("", 0.0)


def test_a_crop_longer_than_960_px_is_scaled_down_to_960_in_multiples_of_32():
    # about the size of the upscaled collector information crop
    crop = np.zeros((320, 1700, 3), dtype=np.uint8)

    tensor = detection_input(crop)

    # 320 * 960 / 1700 = 180.7, rounded to 192
    assert tensor.shape == (1, 3, 192, 960)


def test_a_small_crop_keeps_its_size_rounded_to_multiples_of_32():
    crop = np.zeros((40, 300, 3), dtype=np.uint8)

    assert detection_input(crop).shape == (1, 3, 32, 288)


def test_a_bright_region_of_the_text_map_gives_a_grown_line_in_image_pixels():
    # the text map is at the detection input's size: half the image's
    text_map = np.zeros((64, 320), dtype=np.float32)
    text_map[20:30, 40:200] = 0.9

    [quad] = quads_from_text_map(text_map, (128, 640))

    # the region's outline is 159 x 9 px, grown by 1431 * 1.5 / 336 = 6.4 px on
    # each side: x 33.6 to 205.4, y 13.6 to 35.4, then doubled
    x1, y1 = quad.min(axis=0)
    x2, y2 = quad.max(axis=0)
    assert np.allclose([x1, y1, x2, y2], [67, 27, 411, 71], atol=4)


def test_a_region_whose_mean_score_is_below_0_6_is_not_a_line():
    text_map = np.zeros((64, 320), dtype=np.float32)
    text_map[20:30, 40:200] = 0.5

    assert len(quads_from_text_map(text_map, (64, 320))) == 0


def test_lines_on_one_row_read_left_to_right_then_the_next_row():
    text_map = np.zeros((96, 320), dtype=np.float32)
    # the right line sits 4 px higher than the left one: still the same row
    text_map[10:20, 200:300] = 0.9
    text_map[14:24, 20:120] = 0.9
    text_map[60:70, 20:120] = 0.9

    quads = quads_from_text_map(text_map, (96, 320))

    lefts_and_tops = [(int(q[:, 0].min()), int(q[:, 1].min())) for q in quads]
    assert [x < 150 for x, _ in lefts_and_tops] == [True, False, True]
    assert [y < 40 for _, y in lefts_and_tops] == [True, True, False]
