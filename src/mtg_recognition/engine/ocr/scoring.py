"""Scoring: how well a card's reading agrees with each candidate printing (see Score
in GLOSSARY.md).

FIELDS is the only place where Reading and Printing fields are paired.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz, utils

from mtg_recognition.engine.ocr.reading import Reading
from mtg_recognition.engine.ranking import FieldScore, RankedPrinting
from mtg_recognition.reference.catalog import Printing


def _as_is(value: str) -> str:
    return value


def _without_leading_zeros(number: str) -> str:
    """'0146' -> '146' and '0000' -> '0', but '0' stays '0'."""
    return re.sub(r"^0+(?=\d)", "", number)


@dataclass(frozen=True)
class Field:
    """One Reading field compared with one Printing field."""

    name: str
    weight: float
    read: Callable[[Reading], str | list[str] | None]
    printed: Callable[[Printing], str | None]
    # Applied to both sides before comparing them.
    normalise: Callable[[str], str] = _as_is


FIELDS: list[Field] = [
    Field("name", 2.0, lambda r: r.name, lambda p: p.name, utils.default_process),
    Field("set_code", 1.0, lambda r: r.set_code_candidates, lambda p: p.set_code),
    Field(
        "number", 1.0, lambda r: r.number, lambda p: p.number, _without_leading_zeros
    ),
    Field("base_set_size", 1.0, lambda r: r.base_set_size, lambda p: p.base_set_size),
    Field("artist", 0.5, lambda r: r.artist, lambda p: p.artist, utils.default_process),
    Field(
        "set_release_year",
        1.0,
        lambda r: r.set_release_year,
        lambda p: p.set_release_year,
    ),
    Field("power", 1.0, lambda r: r.power, lambda p: p.power),
    Field("toughness", 1.0, lambda r: r.toughness, lambda p: p.toughness),
]


def _field_score(
    field: Field, read: str | list[str] | None, printed: str | None
) -> float:
    """Best fuzzy ratio (0-100) between any read value and the printed one; 0 when
    either side is missing.
    """
    queries = read if isinstance(read, list) else [] if read is None else [read]
    if printed is None or not queries:
        return 0.0
    choice = field.normalise(printed)
    return max(fuzz.ratio(field.normalise(q), choice) for q in queries)


def _score(reading: Reading, printing: Printing) -> RankedPrinting:
    breakdown = [
        FieldScore(
            field=f.name,
            read=(read := f.read(reading)),
            printed=(printed := f.printed(printing)),
            score=_field_score(f, read, printed),
        )
        for f in FIELDS
    ]
    total = sum(f.weight * s.score for f, s in zip(FIELDS, breakdown, strict=True))
    weights = sum(f.weight for f in FIELDS)
    return RankedPrinting(printing, total / weights, breakdown)


def rank(
    reading: Reading, candidates: Sequence[Printing], k: int
) -> list[RankedPrinting]:
    """The k candidates that best agree with the reading, best first.

    Each score is the weighted average of the field scores (0-100). Tied candidates
    keep their order in `candidates`.
    """
    scored = [_score(reading, p) for p in candidates]
    return sorted(scored, key=lambda r: -r.score)[:k]
