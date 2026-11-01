"""Reference data: everything built from one MTGJSON release and used to recognise
cards (the catalog, the reference images and the pHash index; see
GLOSSARY.md).

`ReferenceData` owns the on-disk layout and the sync state file, so no other
module builds these paths itself.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, create_engine, select

from mtg_recognition.jobs.schemas import CatalogPrinting
from mtg_recognition.settings import settings


def open_read_only(db_path: Path) -> Engine:
    """Engine on the SQLite database at `db_path`, opened read-only: it never
    writes and never creates a missing file. (`mode=ro` is only honoured in a
    SQLite URI, hence the `file:` URI and `uri=true`.)
    """
    return create_engine(
        f"sqlite:///{db_path.resolve().as_uri()}?mode=ro&nolock=1&uri=true"
    )


@dataclass(frozen=True)
class SyncState:
    """MTGJSON version last used to build each committed artifact; `None` means
    not built (or a build was interrupted). The reference images are a local cache,
    checked on disk (`ReferenceData.missing_images`), never recorded here.
    """

    catalog_version: str | None = None
    phash_version: str | None = None


@dataclass(frozen=True)
class ReferenceData:
    """The reference data stored under `root`."""

    root: Path

    @classmethod
    def default(cls) -> "ReferenceData":
        return cls(settings.DATA_DIR / "reference")

    @property
    def catalog_path(self) -> Path:
        """The catalog SQLite file."""
        return self.root / "catalog.sqlite"

    @property
    def phash_files(self) -> tuple[Path, Path]:
        """The pHash index: the hashes `.npy` and the index-to-UUID `.json`."""
        phash_dir = self.root / "index" / "phash"
        return phash_dir / "hashes.npy", phash_dir / "index_to_uuid.json"

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    def image_path(self, uuid: str) -> Path:
        """Where the reference image of the printing `uuid` is stored."""
        return self.images_dir / f"{uuid}.jpg"

    @property
    def allprintings_path(self) -> Path:
        """The MTGJSON AllPrintings database the catalog is built from (kept
        for exploration, not git-tracked).
        """
        return self.root / "mtgjson" / "AllPrintings.sqlite"

    @property
    def allprintings_zip_path(self) -> Path:
        """Where the AllPrintings ZIP is downloaded to, deleted once extracted."""
        return self.allprintings_path.with_name("AllPrintings.sqlite.zip")

    @property
    def log_file(self) -> Path:
        """Images the last index build failed to download or hash (absent when
        there were none).
        """
        return self.root / "image_errors.log"

    @property
    def _state_file(self) -> Path:
        return self.root / ".mtg-sync-state.json"

    def missing_images(self) -> list[tuple[str, str]]:
        """`(uuid, url)` of the printings in the catalog that have an image URL
        but no image file.
        """
        return [
            (uuid, url)
            for uuid, url in self._printing_urls()
            if url is not None and not self.image_path(uuid).exists()
        ]

    def present_images(self) -> list[tuple[str, str | None]]:
        """`(uuid, url)` of the printings in the catalog that have an image
        file.
        """
        return [
            (uuid, url)
            for uuid, url in self._printing_urls()
            if self.image_path(uuid).exists()
        ]

    def stray_images(self) -> list[Path]:
        """The image files in `images_dir` that are not a printing in the
        catalog.
        """
        uuids = {uuid for uuid, _ in self._printing_urls()}
        return sorted(
            path for path in self.images_dir.glob("*.jpg") if path.stem not in uuids
        )

    def _printing_urls(self) -> list[tuple[str, str | None]]:
        """`(uuid, url)` of every printing in the catalog."""
        engine = open_read_only(self.catalog_path)
        try:
            statement = select(CatalogPrinting.uuid, CatalogPrinting.url)
            with Session(engine) as session:
                return list(session.exec(statement).all())
        finally:
            engine.dispose()

    def load_state(self) -> SyncState:
        """Read the sync state. A missing or corrupt file is treated as a first
        run (every version `None`); unknown keys are ignored.
        """
        try:
            data = json.loads(self._state_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return SyncState()
        if not isinstance(data, dict):
            return SyncState()
        known = {f.name for f in fields(SyncState)}
        return SyncState(**{k: v for k, v in data.items() if k in known})

    def save_state(self, state: SyncState) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps(asdict(state), indent=2))
