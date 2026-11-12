"""Scoring: how well a card's reading agrees with each candidate (see Score in
GLOSSARY.md).

Each detail a candidate shows on the card is looked for in the reading's raw text;
a detail it doesn't show is left out of its score (ADR 0008).
"""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

from rapidfuzz import fuzz, utils

from mtg_recognition.engine.ocr.reading import Reading
from mtg_recognition.engine.ranking import FieldScore, RankedPrinting
from mtg_recognition.reference.catalog import Printing

# From early manual tests (ADR 0003)
WEIGHTS = {
    "name": 2.0,
    "set_code": 1.0,
    "collector_number": 1.0,
    "base_set_size": 1.0,
    "artist": 0.5,
    "year": 1.0,
    "power_toughness": 1.0,
}

# OCR reads 0 as O or o next to a digit ("199o", "6o/350")
_ZERO = re.compile(r"(?<=\d)[Oo]|[Oo](?=\d)")
_DIGIT_RUN = re.compile(r"\d+")
_WORD = re.compile(r"[A-Z0-9]+")
# OCR reads the slash between power and toughness as any of these
_PT_SLASH = r"\s*[/|\\Il1)]\s*"


@dataclass(slots=True)
class _PreparedReading:
    """A reading's text, prepared once for every candidate."""

    name: str  # the name crop's text
    name_processed: str  # lower-case, punctuation as spaces
    # The collector information's tokens, O next to a digit read as 0, and their
    # runs of digits (leading zeros dropped) and upper-case words, each with its
    # token, in reading order
    tokens: list[str]
    runs: list[tuple[str, str]]
    words: list[tuple[str, str]]
    # the tokens joined, and the same lower-cased with punctuation as spaces,
    # character for character, so a stretch of one is the same stretch of the other
    collector_information: str
    collector_information_folded: str


def rank(
    reading: Reading, candidates: Sequence[Printing], k: int
) -> list[RankedPrinting]:
    """The k candidates that best agree with the reading, best first.

    A candidate's score is the weighted average (0-100) of the scores of the
    details it shows. Among equal scores the plain printing comes first, then
    the order of `candidates`.
    """
    text = _prepare(reading)
    scored: list[RankedPrinting] = []
    for printing in candidates:
        breakdown = [
            _name(printing, text),
            _set_code(printing, text),
            _whole_run("collector_number", printing.printed_collector_number, text),
            _base_set_size(printing, text),
            _artist(printing, text),
            _whole_run("year", printing.printed_year, text),
            _power_toughness(printing, text),
        ]
        # the name is always shown, so the weights never sum to 0
        shown = [(WEIGHTS[s.field], s.score) for s in breakdown if s.score is not None]
        score = sum(w * s for w, s in shown) / sum(w for w, _ in shown)
        scored.append(RankedPrinting(printing, score, breakdown))

    # tie-break: the plain printing first, then candidate order (sorted is stable)
    scored.sort(key=lambda r: (-r.score, not r.printing.is_plain))
    counts = Counter(r.score for r in scored)
    return [replace(r, decided_by_tie_break=counts[r.score] > 1) for r in scored[:k]]


def _prepare(reading: Reading) -> _PreparedReading:
    name = " ".join(reading.name.texts)
    tokens = [_read_zeros(t) for t in reading.collector_information.texts]
    collector_information = " ".join(tokens)
    return _PreparedReading(
        name=name,
        name_processed=utils.default_process(name),
        tokens=tokens,
        runs=[(r.lstrip("0") or "0", t) for t in tokens for r in _DIGIT_RUN.findall(t)],
        words=[(w, t) for t in tokens for w in _WORD.findall(t.upper())],
        collector_information=collector_information,
        collector_information_folded="".join(
            c.lower() if c.isalnum() else " " for c in collector_information
        ),
    )


def _read_zeros(token: str) -> str:
    # each pass only sees the original neighbours, so "OO1" takes two
    while (fixed := _ZERO.sub("0", token)) != token:
        token = fixed
    return token


def _name(printing: Printing, text: _PreparedReading) -> FieldScore:
    """The closest of the printing's names: each face of `A // B`, and the flavor
    name, which is the large name on the card when there is one.
    """
    names = printing.name.split(" // ")
    if printing.flavor_name:
        names.append(printing.flavor_name)
    scores = {
        n: fuzz.ratio(utils.default_process(n), text.name_processed) for n in names
    }
    best = max(names, key=scores.__getitem__)
    return FieldScore("name", best, scores[best], text.name or None)


def _set_code(printing: Printing, text: _PreparedReading) -> FieldScore:
    """Found alone, or glued to a two-letter language (`THBDE`, `WAREN`)."""
    code = printing.printed_set_code
    if code is None:
        return _not_shown("set_code")
    pattern = re.compile(rf"{re.escape(code)}(?:[A-Z]{{2}})?")
    found = next((t for w, t in text.words if pattern.fullmatch(w)), None)
    return _found("set_code", code, found)


def _whole_run(field: str, value: str | None, text: _PreparedReading) -> FieldScore:
    """Found as a whole run of digits, so `14` isn't found in `149`; a range
    year's end year (`1993-2001`) is one.
    """
    if value is None:
        return _not_shown(field)
    found = next((t for run, t in text.runs if run == value), None)
    return _found(field, value, found)


def _base_set_size(printing: Printing, text: _PreparedReading) -> FieldScore:
    """Found in the run right after the collector number, or glued to it when
    OCR lost the slash or read it as 1 (`751350`, `7511350`).
    """
    number = printing.printed_collector_number
    size = printing.printed_base_set_size
    # the size is shown only after a number
    if size is None or number is None:
        return _not_shown("base_set_size")
    glued = re.compile(rf"{re.escape(number)}1?0*{re.escape(size)}")
    # the last run is followed by nothing
    for (run, token), (next_run, next_token) in pairwise([*text.runs, ("", "")]):
        if (run, next_run) == (number, size):
            found = token if token == next_token else f"{token} {next_token}"
            return _found("base_set_size", size, found)
        if glued.fullmatch(run):
            return _found("base_set_size", size, token)
    return _found("base_set_size", size, None)


def _artist(printing: Printing, text: _PreparedReading) -> FieldScore:
    """The closest stretch of the collector information to the artist's name,
    which may be preceded by `Illus.`.
    """
    if printing.artist is None:
        return _not_shown("artist")
    alignment = fuzz.partial_ratio_alignment(
        utils.default_process(printing.artist), text.collector_information_folded
    )
    # None only below a score_cutoff, which isn't given
    assert alignment is not None
    matched = text.collector_information[alignment.dest_start : alignment.dest_end]
    return FieldScore(
        "artist", printing.artist, alignment.score, matched.strip() or None
    )


def _power_toughness(printing: Printing, text: _PreparedReading) -> FieldScore:
    power, toughness = printing.power, printing.toughness
    if power is None or toughness is None:
        return _not_shown("power_toughness")
    pattern = re.compile(
        rf"(?<![\dX*]){re.escape(power)}{_PT_SLASH}{re.escape(toughness)}(?!\d)"
    )
    found = next((t for t in text.tokens if pattern.search(t)), None)
    return _found("power_toughness", f"{power}/{toughness}", found)


def _not_shown(field: str) -> FieldScore:
    return FieldScore(field, None, None, None)


def _found(field: str, value: str, text: str | None) -> FieldScore:
    """A detail that is either found (100) or not (0)."""
    return FieldScore(field, value, 0.0 if text is None else 100.0, text)
