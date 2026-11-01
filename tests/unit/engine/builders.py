"""Printings, readings and card images built by hand for engine and app tests."""

import cv2
import numpy as np

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
    )


def reading_of(p: Printing) -> Reading:
    """A reading that agrees with every field of the printing, so it scores 100."""
    return Reading(
        name=p.name,
        possible_set_codes=[p.set_code],
        collector_number=p.collector_number,
        base_set_size=p.base_set_size,
        artist=p.artist,
        copyright_year=p.set_release_year,
        power=p.power,
        toughness=p.toughness,
    )


def card_image(value: int = 0) -> np.ndarray:
    return np.full((680, 488, 3), value, dtype=np.uint8)


def card_png(value: int = 0) -> bytes:
    """card_image(value) encoded as PNG, as an uploaded or saved file holds it."""
    ok, buffer = cv2.imencode(".png", card_image(value))
    assert ok
    return buffer.tobytes()
