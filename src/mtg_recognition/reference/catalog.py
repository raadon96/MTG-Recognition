"""Card catalog: resolves printing UUIDs to Printing records, and holds each
printing's pHash and the MTGJSON release it was built from (ADR 0006).

The engine only ever sees `Printing`s and pHashes; reading them out of the
SQLite file is the catalog's concern. The catalog's table schemas live here too,
so `jobs` writes the catalog with the same schemas this module reads it with.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine, MetaData, func, text
from sqlalchemy.exc import OperationalError
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
    # what the card shows, which differs from the values above on promos, The
    # List and retro reprints; None when the card doesn't show the detail
    printed_set_code: str | None = None
    printed_collector_number: str | None = None
    printed_base_set_size: str | None = None
    printed_year: str | None = None
    flavor_name: str | None = None
    # shows its own details and its number has no suffix (`254★`), so it comes
    # first among printings showing the same details
    is_plain: bool = False


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
                collector_number=row.collectorNumber,
                base_set_size=row.baseSetSize,
                artist=row.artist,
                set_release_year=row.setReleaseYear,
                power=row.power,
                toughness=row.toughness,
                image_url=row.url,
                printed_set_code=row.printedSetCode,
                printed_collector_number=row.printedCollectorNumber,
                printed_base_set_size=row.printedBaseSetSize,
                printed_year=row.printedYear,
                flavor_name=row.flavorName,
                is_plain=row.isPlain,
            )
            for row in rows
        }


def catalog_version(catalog_path: Path) -> str | None:
    """The MTGJSON release the catalog at `catalog_path` was built from, or `None`
    when it wasn't built: the file is missing, or has no `meta` table (catalogs from
    before ADR 0006).
    """
    if not catalog_path.exists():
        return None
    engine = open_read_only(catalog_path)
    try:
        with Session(engine) as session:
            return session.exec(select(col(CatalogMeta.version))).first()
    except OperationalError:  # no meta table
        return None
    finally:
        engine.dispose()


def read_phashes(catalog_path: Path) -> list[tuple[str, bytes]]:
    """`(uuid, phash)` of every printing in the catalog that has a pHash, in the
    order the printings were written, so the order is the same on every machine.
    """
    engine = open_read_only(catalog_path)
    try:
        statement = (
            select(CatalogPrinting.uuid, CatalogPrinting.phash)
            .where(col(CatalogPrinting.phash).is_not(None))
            .order_by(text("rowid"))
        )
        with Session(engine) as session:
            return [(uuid, phash) for uuid, phash in session.exec(statement).all()]
    finally:
        engine.dispose()


def count_phashes(catalog_path: Path) -> int:
    """Number of printings in the catalog that have a pHash."""
    engine = open_read_only(catalog_path)
    try:
        statement = select(func.count()).where(col(CatalogPrinting.phash).is_not(None))
        with Session(engine) as session:
            return session.exec(statement).one()
    finally:
        engine.dispose()


def read_image_urls(catalog_path: Path) -> dict[str, str | None]:
    """The image URL of every printing in the catalog, by UUID; `None` when the
    printing has none.
    """
    engine = open_read_only(catalog_path)
    try:
        statement = select(CatalogPrinting.uuid, CatalogPrinting.url)
        with Session(engine) as session:
            return dict(session.exec(statement).all())
    finally:
        engine.dispose()


class CatalogMeta(SQLModel, table=True):
    """The MTGJSON release the catalog was built from, copied from AllPrintings'
    `meta` table; one row.
    """

    # AllPrintings' own `meta` table (`jobs.schemas.Meta`) is registered in
    # SQLModel's shared metadata under the same name, so this one gets its own
    metadata = MetaData()
    __tablename__ = "meta"
    version: str = Field(primary_key=True)
    date: str | None = Field(default=None)


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
    # the large name on the card, when it isn't the card name
    flavorName: str | None = Field(default=None)

    # Each detail beside what the card shows of it, from the rules in
    # `jobs.printed`; a printed column is None when the card doesn't show the
    # detail. Scoring compares the reading with the printed columns.
    setCode: str = Field(default=None, index=False)  # index=True
    printedSetCode: str | None = Field(default=None)
    collectorNumber: str = Field(default=None, index=False)
    printedCollectorNumber: str | None = Field(default=None)
    baseSetSize: str = Field(default=None, index=False)
    printedBaseSetSize: str | None = Field(default=None)
    # number: str = Field(default=None) # Use with composite index
    # baseSetSize: str = Field(default=None) # Use with composite index
    # The year of the set's release date, set when the catalog is built. Scoring
    # compares it with the copyright year in a card's reading.
    setReleaseYear: str | None = Field(None, index=False)
    printedYear: str | None = Field(default=None)
    # shows its own details and its collector number has no suffix (`254★`)
    isPlain: bool = Field(default=False)

    artist: str | None = Field(default=None)
    manaCost: str | None = Field(default=None)
    type: str | None = Field(default=None)
    # setName: Optional[str] = Field(None, description="Set name")
    power: str | None = Field(default=None)
    toughness: str | None = Field(default=None)
    borderColor: str | None = Field(default=None)

    # the reference image the pHash is computed from
    url: str | None = Field(default=None)
    # The 256-bit pHash of the reference image, packed 8 bits per byte (32
    # bytes); None when the printing has no image URL, or its image failed to
    # download or hash.
    phash: bytes | None = Field(default=None)
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
