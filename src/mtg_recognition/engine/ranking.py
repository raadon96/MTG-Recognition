"""Ranked candidates: the printings recognition proposes for one card, with their
scores.
"""

from dataclasses import dataclass, field

from mtg_recognition.reference.catalog import Printing


@dataclass(frozen=True)
class FieldScore:
    """A field score: how well one field of a card's reading agrees with a
    printing's.
    """

    field: str
    read: str | list[str] | None  # as read off the card
    printed: str | None  # the printing's value
    score: float  # 0-100


@dataclass(frozen=True)
class RankedPrinting:
    """A candidate printing with its score (0-100, higher is better; see GLOSSARY.md)
    and the per-field scores it was computed from.
    """

    printing: Printing
    score: float
    breakdown: list[FieldScore] = field(default_factory=list)
