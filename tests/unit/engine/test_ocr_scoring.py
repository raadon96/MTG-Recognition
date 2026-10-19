"""Fuzzy scoring of OCR-extracted properties against candidate printings."""

import numpy as np
import pytest

from mtg_recognition.catalog import Printing
from mtg_recognition.engine.ocr.ocr import CardOCR, Properties

BLANK_CARD = np.zeros((680, 488, 3), dtype=np.uint8)

BOLT_M10 = Printing(
    uuid="bolt-m10",
    name="Lightning Bolt",
    set_code="M10",
    number="146",
    base_set_size="249",
    artist="Christopher Moeller",
    set_release_year="2009",
)
BOLT_M11 = Printing(
    uuid="bolt-m11",
    name="Lightning Bolt",
    set_code="M11",
    number="149",
    base_set_size="249",
    artist="Christopher Moeller",
    set_release_year="2010",
)
SHOCK = Printing(
    uuid="shock-m19",
    name="Shock",
    set_code="M19",
    number="156",
    base_set_size="280",
    artist="Jason Rainville",
    set_release_year="2018",
)
GRIZZLY = Printing(
    uuid="grizzly-m10",
    name="Grizzly Bears",
    set_code="M10",
    number="186",
    base_set_size="249",
    artist="D. J. Cleland-Hura",
    set_release_year="2009",
    power="2",
    toughness="2",
)


def _properties(**overrides) -> Properties:
    fields = dict(
        setcode_candidates=[],
        name=None,
        number=None,
        base_set_size=None,
        artist=None,
        set_release_year=None,
        power=None,
        toughness=None,
    )
    fields.update(overrides)
    return Properties(**fields)


@pytest.fixture(scope="module")
def ocr() -> CardOCR:
    return CardOCR()


def test_name_alone_ranks_matching_card_above_other_cards(ocr):
    ranked, _ = ocr.find_best_matches(
        BLANK_CARD, [SHOCK, BOLT_M10], properties=_properties(name="LIGHTNING BOLT")
    )

    assert ranked[0].printing == BOLT_M10


def test_set_code_and_number_pick_the_right_printing_among_reprints(ocr):
    properties = _properties(
        name="Lightning Bolt",
        setcode_candidates=["M11"],
        number="149",
        base_set_size="249",
    )

    ranked, _ = ocr.find_best_matches(
        BLANK_CARD, [BOLT_M10, SHOCK, BOLT_M11], properties=properties
    )

    assert ranked[0].printing == BOLT_M11


def test_power_and_toughness_count_towards_the_score(ocr):
    properties = _properties(setcode_candidates=["M10"], power="2", toughness="2")

    ranked, _ = ocr.find_best_matches(
        BLANK_CARD, [BOLT_M10, GRIZZLY], properties=properties
    )

    assert ranked[0].printing == GRIZZLY


def test_returns_k_ranked_printings_with_descending_scores(ocr):
    properties = _properties(
        name="Lightning Bolt", setcode_candidates=["M10"], number="146"
    )

    ranked, _ = ocr.find_best_matches(
        BLANK_CARD, [SHOCK, BOLT_M11, GRIZZLY, BOLT_M10], properties=properties, k=2
    )

    assert [r.printing for r in ranked] == [BOLT_M10, BOLT_M11]
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert 0 < scores[0] <= 100
