"""Fake recognition stages and printings for tests that build a Pipeline without models."""

import numpy as np

from mtg_recognition.catalog import InMemoryCardCatalog, Printing, RankedPrinting
from mtg_recognition.engine.ocr.ocr import OCRDebugData
from mtg_recognition.engine.pipeline import Pipeline
from mtg_recognition.engine.segmentation.seg_utils import CardSegment, ImageSegmentationResult


def printing(uuid: str) -> Printing:
    return Printing(uuid=uuid, name=f"Card {uuid}", set_code="TST", number="1", base_set_size="100")


PRINTINGS = [printing(f"p{i}") for i in range(10)]
CATALOG = InMemoryCardCatalog(PRINTINGS)


def card_image(value: int = 0) -> np.ndarray:
    return np.full((680, 488, 3), value, dtype=np.uint8)


class FakeSegmentor:
    """Finds the given warped images in every input image."""

    def __init__(self, warped_images: list[np.ndarray]):
        self.warped_images = warped_images
        self.inputs: list[np.ndarray] = []

    def segment(self, img: np.ndarray) -> ImageSegmentationResult:
        self.inputs.append(img)
        return ImageSegmentationResult(segments=[CardSegment(warped_image=w) for w in self.warped_images])


class FakePHashMatcher:
    """Returns the printings whose image value matches first, then the rest of the index in order."""

    index_loaded = True

    def __init__(self, index: list[str] | None = None):
        self.index = index if index is not None else [p.uuid for p in PRINTINGS]
        self.ks: list[int] = []

    def find_best_matches(self, query_image: np.ndarray, k: int = 5) -> tuple[list[str], list[float]]:
        self.ks.append(k)
        value = int(query_image[0, 0, 0])
        ranked = sorted(self.index, key=lambda uuid: uuid != f"p{value}")[:k]
        return ranked, [1.0 - 0.01 * i for i in range(len(ranked))]

    def build_index(self, force: bool = False) -> None:
        pass


class FakeOCR:
    """Reverses the pHash order, so its effect on the ranking is visible."""

    def __init__(self):
        self.ks: list[int | None] = []

    def find_best_matches(
        self,
        query_img: np.ndarray,
        candidates: list[Printing],
        k: int | None = 10,
        thresh: float = 0.0,
        debug: bool = False,
    ) -> tuple[list[RankedPrinting], OCRDebugData | None]:
        self.ks.append(k)
        reranked = list(reversed(candidates))[:k]
        return [RankedPrinting(p, 90.0 - i) for i, p in enumerate(reranked)], None


def make_pipeline(
    segmentor: FakeSegmentor,
    k_phash_candidates: int = 20,
    k_final_results: int = 1,
    phash_matcher: FakePHashMatcher | None = None,
    ocr: FakeOCR | None = None,
) -> Pipeline:
    return Pipeline(
        k_phash_candidates=k_phash_candidates,
        k_final_results=k_final_results,
        catalog=CATALOG,
        segmentor=segmentor,
        phash_matcher=phash_matcher if phash_matcher is not None else FakePHashMatcher(),
        ocr=ocr if ocr is not None else FakeOCR(),
    )
