"""Card catalog: resolves printing UUIDs to Printing records.

The engine only ever sees `Printing`s; reading them out of the SQLite file is
the catalog's concern.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from mtg_recognition.jobs.schemas import CatalogPrinting
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


class SqliteCardCatalog:
    """Catalog read from the SQLite file at `catalog_path`, opened read-only the
    first time it is used.
    """

    def __init__(self, catalog_path: Path):
        self._catalog_path = catalog_path
        self._engine: Engine | None = None

    def lookup(self, uuids: Sequence[str]) -> dict[str, Printing]:
        if self._engine is None:
            self._engine = open_read_only(self._catalog_path)
        statement = select(CatalogPrinting).where(
            col(CatalogPrinting.uuid).in_(list(uuids))
        )
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
