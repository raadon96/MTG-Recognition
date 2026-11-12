"""Ranked candidates: the printings recognition proposes for one card, with their
scores.
"""

from dataclasses import dataclass, field

from mtg_recognition.reference.catalog import Printing


@dataclass(frozen=True)
class FieldScore:
    """How well one detail a printing shows on the card was found in a card's
    reading.
    """

    field: str
    # the printing's value looked for; None when the printing doesn't show it
    value: str | None
    # 0-100; None when the printing doesn't show the detail, which leaves it out
    # of the score
    score: float | None
    # the reading's text the value was found in; None when it wasn't found
    text: str | None


@dataclass(frozen=True)
class RankedPrinting:
    """A candidate printing with its score (0-100, higher is better; see GLOSSARY.md)
    and the field scores it was computed from.
    """

    printing: Printing
    score: float
    breakdown: list[FieldScore] = field(default_factory=list)
    # another candidate has the same score, so the tie-break (the plain printing
    # first, then pHash order) decided this one's place
    decided_by_tie_break: bool = False
