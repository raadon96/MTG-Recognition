"""Fake recognition stages and printings for tests that build a Pipeline without
models.
"""

import cv2
import numpy as np

from mtg_recognition.catalog import InMemoryCardCatalog, Printing
from mtg_recognition.engine.ocr.reading import Reading
from mtg_recognition.engine.pipeline import Pipeline
from mtg_recognition.engine.segmentation.seg_utils import (
    CardSegment,
    ImageSegmentationResult,
)


def printing(uuid: str) -> Printing:
    return Printing(
        uuid=uuid,
        name=f"Card {uuid}",
        set_code="TST",
        number="1",
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
        set_code_candidates=[p.set_code],
        number=p.number,
        base_set_size=p.base_set_size,
        artist=p.artist,
        set_release_year=p.set_release_year,
        power=p.power,
        toughness=p.toughness,
    )


PRINTINGS = [printing(f"p{i}") for i in range(10)]
CATALOG = InMemoryCardCatalog(PRINTINGS)


def card_image(value: int = 0) -> np.ndarray:
    return np.full((680, 488, 3), value, dtype=np.uint8)


def card_png(value: int = 0) -> bytes:
    """card_image(value) encoded as PNG, as an uploaded or saved file holds it."""
    ok, buffer = cv2.imencode(".png", card_image(value))
    assert ok
    return buffer.tobytes()


class FakeSegmentor:
    """Finds the given warped images in every input image."""

    def __init__(self, warped_images: list[np.ndarray]):
        self.warped_images = warped_images
        self.inputs: list[np.ndarray] = []

    def segment(self, img: np.ndarray) -> ImageSegmentationResult:
        self.inputs.append(img)
        return ImageSegmentationResult(
            segments=[CardSegment(warped_image=w) for w in self.warped_images]
        )


class FakePHashMatcher:
    """Returns the printings whose image value matches first.

    The rest of the index follows in order.
    """

    index_loaded = True

    def __init__(self, index: list[str] | None = None):
        self.index = index if index is not None else [p.uuid for p in PRINTINGS]
        self.ks: list[int] = []

    def find_best_matches(
        self, query_image: np.ndarray, k: int = 5
    ) -> tuple[list[str], list[float]]:
        self.ks.append(k)
        value = int(query_image[0, 0, 0])
        ranked = sorted(self.index, key=lambda uuid: uuid != f"p{value}")[:k]
        return ranked, [1.0 - 0.01 * i for i in range(len(ranked))]


class FakeReader:
    """Reads the given reading off every card image (default: a reading of p0)."""

    def __init__(self, reading: Reading | None = None):
        self.reading = reading if reading is not None else reading_of(PRINTINGS[0])
        self.debugs: list[bool] = []

    def read(self, card_image: np.ndarray, *, debug: bool = False) -> Reading:
        self.debugs.append(debug)
        return self.reading


def make_pipeline(
    segmentor: FakeSegmentor,
    k_phash_candidates: int = 20,
    k_final_results: int = 1,
    phash_matcher: FakePHashMatcher | None = None,
    reader: FakeReader | None = None,
    catalog: InMemoryCardCatalog = CATALOG,
) -> Pipeline:
    return Pipeline(
        k_phash_candidates=k_phash_candidates,
        k_final_results=k_final_results,
        catalog=catalog,
        segmentor=segmentor,
        phash_matcher=phash_matcher
        if phash_matcher is not None
        else FakePHashMatcher(),
        reader=reader if reader is not None else FakeReader(),
    )
