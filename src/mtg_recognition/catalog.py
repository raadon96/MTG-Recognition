"""Card catalog: resolves printing UUIDs to Printing records.

The engine only ever sees `Printing`s; where they come from (the SQLite
MobileCache in production, a dict in tests) is the catalog adapter's concern.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from mtg_recognition.jobs.schemas import MobileCache
from mtg_recognition.reference import open_read_only


@dataclass(frozen=True)
class Printing:
    """One edition of a card in a specific set, identified by a UUID (see
    GLOSSARY.md).
    """

    uuid: str
    name: str
    set_code: str
    number: str
    base_set_size: str
    artist: str | None = None
    set_release_year: str | None = None
    power: str | None = None
    toughness: str | None = None
    image_url: str | None = None


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


class CardCatalog(Protocol):
    def lookup(self, uuids: Sequence[str]) -> dict[str, Printing]:
        """Return the known printings among `uuids`, keyed by UUID. Unknown UUIDs are
        omitted.
        """
        ...


class InMemoryCardCatalog:
    """Catalog backed by a dict; for tests."""

    def __init__(self, printings: Iterable[Printing]):
        self._printings = {p.uuid: p for p in printings}

    def lookup(self, uuids: Sequence[str]) -> dict[str, Printing]:
        return {u: self._printings[u] for u in uuids if u in self._printings}


class SqliteCardCatalog:
    """Catalog backed by the read-only MobileCache SQLite database at `db_path`,
    opened the first time it is used.
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._engine: Engine | None = None

    def lookup(self, uuids: Sequence[str]) -> dict[str, Printing]:
        if self._engine is None:
            self._engine = open_read_only(self._db_path)
        statement = select(MobileCache).where(col(MobileCache.uuid).in_(list(uuids)))
        with Session(self._engine) as session:
            rows = session.exec(statement).all()
        return {
            row.uuid: Printing(
                uuid=row.uuid,
                name=row.name,
                set_code=row.setCode,
                number=row.number,
                base_set_size=row.baseSetSize,
                artist=row.artist,
                set_release_year=row.setReleaseYear,
                power=row.power,
                toughness=row.toughness,
                image_url=row.url,
            )
            for row in rows
        }
