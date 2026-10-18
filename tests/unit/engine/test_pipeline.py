"""Pipeline.recognize with fake segmentor, pHash matcher and OCR plus the in-memory catalog."""

import numpy as np
import pytest

from mtg_recognition.catalog import InMemoryCardCatalog, Printing
from mtg_recognition.engine.pipeline import Pipeline
from mtg_recognition.engine.segmentation.seg_utils import CardSegment, ImageSegmentationResult


def _printing(uuid: str) -> Printing:
    return Printing(uuid=uuid, name=f"Card {uuid}", set_code="TST", number="1", base_set_size="100")


PRINTINGS = [_printing(f"p{i}") for i in range(10)]
CATALOG = InMemoryCardCatalog(PRINTINGS)


def _card_image(value: int = 0) -> np.ndarray:
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

    def find_best_matches(self, query_img, candidates, k=10, thresh=0.0, debug=False):
        self.ks.append(k)
        reranked = list(reversed(candidates))[:k]
        return reranked, [90.0 - i for i in range(len(reranked))], None


def _pipeline(segmentor: FakeSegmentor, **kwargs) -> Pipeline:
    kwargs.setdefault("phash_matcher", FakePHashMatcher())
    kwargs.setdefault("ocr", FakeOCR())
    return Pipeline(catalog=CATALOG, segmentor=segmentor, **kwargs)


def test_no_card_found_gives_no_segments_and_no_error():
    result = _pipeline(FakeSegmentor([])).recognize(_card_image())

    assert result.segments == []
    assert result.error is None


def test_one_card_is_matched_by_phash_then_reranked_by_ocr():
    warped = _card_image(3)

    result = _pipeline(FakeSegmentor([warped]), k_phash_candidates=2, k_final_results=2).recognize(_card_image())

    assert result.error is None
    [segment] = result.segments
    assert segment.warped_image is warped
    # pHash ranks p3 then p0; the fake OCR reverses that order.
    assert segment.matches == [_printing("p0"), _printing("p3")]
    assert segment.scores == [90.0, 89.0]


def test_several_cards_give_one_segment_each_in_segmentation_order():
    segmentor = FakeSegmentor([_card_image(1), _card_image(2), _card_image(3)])

    result = _pipeline(segmentor, k_phash_candidates=1, k_final_results=1).recognize(_card_image())

    assert [s.matches for s in result.segments] == [[_printing("p1")], [_printing("p2")], [_printing("p3")]]


def test_k_parameters_are_passed_to_phash_and_bound_the_results():
    phash, ocr = FakePHashMatcher(), FakeOCR()
    pipeline = _pipeline(
        FakeSegmentor([_card_image(5)]), k_phash_candidates=7, k_final_results=3, phash_matcher=phash, ocr=ocr
    )

    [segment] = pipeline.recognize(_card_image()).segments

    assert phash.ks == [7]
    assert ocr.ks == [3]
    assert len(segment.matches) == len(segment.scores) == 3


def test_results_are_truncated_to_k_even_if_ocr_returns_more():
    class AllCandidatesOCR(FakeOCR):
        def find_best_matches(self, query_img, candidates, k=10, thresh=0.0, debug=False):
            return candidates, [0.0] * len(candidates), None

    pipeline = _pipeline(
        FakeSegmentor([_card_image()]), k_phash_candidates=5, k_final_results=2, ocr=AllCandidatesOCR()
    )

    [segment] = pipeline.recognize(_card_image()).segments

    assert len(segment.matches) == len(segment.scores) == 2


def test_candidates_missing_from_the_catalog_are_dropped_with_their_scores():
    phash = FakePHashMatcher(index=["p4", "not-in-catalog", "p5"])
    pipeline = _pipeline(FakeSegmentor([_card_image(4)]), k_phash_candidates=3, k_final_results=2, phash_matcher=phash)

    [segment] = pipeline.recognize(_card_image()).segments

    assert segment.matches == [_printing("p5"), _printing("p4")]


def test_a_single_phash_candidate_is_returned_without_ocr():
    ocr = FakeOCR()
    pipeline = _pipeline(FakeSegmentor([_card_image(6)]), k_phash_candidates=1, k_final_results=1, ocr=ocr)

    [segment] = pipeline.recognize(_card_image()).segments

    assert segment.matches == [_printing("p6")]
    assert segment.scores == [1.0]
    assert ocr.ks == []


def test_an_exception_while_processing_becomes_the_error():
    class BrokenSegmentor(FakeSegmentor):
        def segment(self, img):
            raise RuntimeError("model exploded")

    result = _pipeline(BrokenSegmentor([])).recognize(_card_image())

    assert result.segments == []
    assert result.error == "RuntimeError: model exploded"


def test_large_images_are_downscaled_before_segmentation():
    segmentor = FakeSegmentor([])

    _pipeline(segmentor).recognize(np.zeros((4000, 3000, 3), dtype=np.uint8))

    assert segmentor.inputs[0].shape == (2048, 1536, 3)


def test_small_images_reach_the_segmentor_unchanged():
    segmentor = FakeSegmentor([])
    image = _card_image()

    _pipeline(segmentor).recognize(image)

    assert segmentor.inputs[0] is image


@pytest.mark.parametrize(("k_phash", "k_final"), [(5, 6), (5, 0), (0, 0)])
def test_invalid_k_parameters_raise(k_phash, k_final):
    with pytest.raises(ValueError):
        _pipeline(FakeSegmentor([]), k_phash_candidates=k_phash, k_final_results=k_final)


def test_missing_phash_index_raises():
    phash = FakePHashMatcher()
    phash.index_loaded = False

    with pytest.raises(ValueError, match="pHash index"):
        _pipeline(FakeSegmentor([]), phash_matcher=phash)
