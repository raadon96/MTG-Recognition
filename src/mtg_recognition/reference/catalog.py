"""Card catalog: resolves printing UUIDs to Printing records.

The engine only ever sees `Printing`s; reading them out of the SQLite file is
the catalog's concern. The catalog table schema lives here too, so `jobs`
writes the catalog with the same schema this module reads it with.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Field, Session, SQLModel, col, select

from mtg_recognition.reference.sqlite import open_read_only


@dataclass(frozen=True)
class Printing:
    """One edition of a card in a specific set, identified by a UUID (see
    GLOSSARY.md).
    """

    uuid: str
    name: str
    set_code: str
    collector_number: str
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
                collector_number=row.number,
                base_set_size=row.baseSetSize,
                artist=row.artist,
                set_release_year=row.setReleaseYear,
                power=row.power,
                toughness=row.toughness,
                image_url=row.url,
            )
            for row in rows
        }


class CatalogPrinting(
    SQLModel,
    table=True,
):
    # https://docs.sqlalchemy.org/en/13/orm/extensions/declarative/table_config.html
    """One printing in the catalog."""

    __tablename__ = "catalog"
    __table_args__ = {
        "extend_existing": True,
    }
    # __table_args__ = (
    #     {"extend_existing": True},  # Table Options
    #     Index("number", "baseSetSize"),  # Composite Index
    # )
    uuid: str = Field(primary_key=True, index=True)
    name: str = Field(default=None, index=False)
    setCode: str = Field(default=None, index=False)  # index=True

    number: str = Field(default=None, index=False)
    baseSetSize: str = Field(default=None, index=False)

    # number: str = Field(default=None) # Use with composite index
    # baseSetSize: str = Field(default=None) # Use with composite index

    setReleaseYear: str | None = Field(None, index=False)  # Copyright year
    artist: str | None = Field(default=None)
    url: str | None = Field(default=None)
    manaCost: str | None = Field(default=None)
    type: str | None = Field(default=None)
    # setName: Optional[str] = Field(None, description="Set name")
    power: str | None = Field(default=None)
    toughness: str | None = Field(default=None)
    borderColor: str | None = Field(default=None)
    # text: str | None = Field(default=None)
    # flavorText: str | None = Field(default=None)
    # scryfallId: str
    # mtgoId: str | None = Field(default=None)
    # language: str | None = Field(default=None)
    # frameVersion: str | None = Field(default=None) # removed, useless

    # # AllPrintings, table: cards
    # uuid: str | None = Field(primary_key=True)
    # name: str | None = Field(default=None)
    # manaCost: str | None = Field(default=None)
    # setCode: str | None = Field(default=None)
    # artist: str | None = Field(default=None)
    # number: str | None = Field(default=None)
    # type: str | None = Field(default=None)
    # power: str | None = Field(default=None)
    # toughness: str | None = Field(default=None)
    # borderColor: str | None = Field(default=None)
    # flavorText: str | None = Field(default=None)  # XX
    # language: str | None = Field(default=None)
    # text: str | None = Field(default=None)
    # frameVersion: str | None = Field(default=None)

    # # AllPrintings, table: sets
    # code: str = Field(
    #     primary_key=True,
    #     index=True,
    #     max_length=8,
    #     description="Unique set code",
    # ) # setCode
    # name: Optional[str] = Field(None, description="Set name") # setName
    # baseSetSize: Optional[int] = Field(
    #     default=None, description="Base size of the set"
    # )
    # setReleaseDate
    # releaseDate: Optional[str] = Field(None, description="Release date as string")

    # # AllPrintings, table: cardIdentifiers
    # uuid: str | None = Field(primary_key=True)
    # mtgoId: str | None = Field(default=None)
    # scryfallId: str | None = Field(default=None)
