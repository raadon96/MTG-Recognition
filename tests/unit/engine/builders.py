"""Printings, readings and card images built by hand for engine and app tests."""

import cv2
import numpy as np

from mtg_recognition.engine.ocr.pp_ocr import TextLines
from mtg_recognition.engine.ocr.reading import Reading
from mtg_recognition.reference.catalog import Printing


def printing(uuid: str) -> Printing:
    return Printing(
        uuid=uuid,
        name=f"Card {uuid}",
        set_code="TST",
        collector_number="1",
        base_set_size="100",
        artist="Artist",
        set_release_year="2020",
        power="1",
        toughness="1",
        printed_set_code="TST",
        printed_collector_number="1",
        printed_base_set_size="100",
        printed_year="2020",
        is_plain=True,
    )


def reading_of(p: Printing) -> Reading:
    """A reading that shows every detail the printing shows, so it scores 100."""
    tokens = [
        f"{p.printed_collector_number}/{p.printed_base_set_size}",
        f"{p.printed_set_code}·EN",
        f"{p.artist}",
        f"™ & © {p.printed_year} Wizards of the Coast",
        f"{p.power}/{p.toughness}",
    ]
    return reading(p.name, tokens)


def reading(name: str = "", collector_information: list[str] | None = None) -> Reading:
    """A reading of `name` in the name crop and the tokens `collector_information`
    in the collector information crop.
    """
    return Reading(
        name=text_lines([name] if name else []),
        collector_information=text_lines(collector_information or []),
    )


def text_lines(texts: list[str]) -> TextLines:
    """OCR tokens with made-up confidences and boxes."""
    return TextLines(
        texts=texts,
        scores=[0.9] * len(texts),
        boxes=[(0, 10 * i, 100, 10 * i + 9) for i in range(len(texts))],
    )


def card_image(value: int = 0) -> np.ndarray:
    return np.full((680, 488, 3), value, dtype=np.uint8)


def card_png(value: int = 0) -> bytes:
    """card_image(value) encoded as PNG, as an uploaded or saved file holds it."""
    ok, buffer = cv2.imencode(".png", card_image(value))
    assert ok
    return buffer.tobytes()
