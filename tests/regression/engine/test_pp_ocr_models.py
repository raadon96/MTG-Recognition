"""The production OCR models, with the dictionary that ships with the recognition
export, on rendered lines of card text.
"""

import cv2
import numpy as np
import pytest

from mtg_recognition.engine.ocr.pp_ocr import load_pp_ocr, read_text


@pytest.fixture(scope="module")
def ocr():
    return load_pp_ocr()


@pytest.mark.slow
def test_recognition_has_one_output_per_character_of_the_dictionary(ocr):
    # a dictionary from another model would decode to the wrong characters
    [output] = ocr.recognition.get_outputs()

    assert output.shape[-1] == len(ocr.characters)


@pytest.mark.slow
@pytest.mark.parametrize(
    "text", ["KHM 214/285", "Skyshroud Elf", "2021 Wizards of the Coast"]
)
def test_a_rendered_line_reads_as_its_text(ocr, text):
    lines = read_text(ocr, _rendered_line(text))

    assert lines.texts == [text]


def _rendered_line(text: str) -> np.ndarray:
    """Black text on a white strip, 60 px high."""
    image = np.full((60, len(text) * 22 + 20, 3), 255, dtype=np.uint8)
    cv2.putText(
        image, text, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA
    )
    return image
