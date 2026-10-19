"""Reference data: everything built from one MTGJSON release and used to recognise
cards (the MobileCache database, the reference images and the pHash index; see
GLOSSARY.md).

`ReferenceData` owns the on-disk layout and the sync state file, so no other
module builds these paths itself.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from sqlmodel import Session, col, create_engine, select

from mtg_recognition.jobs.schemas import MobileCache
from mtg_recognition.settings import settings


@dataclass(frozen=True)
class SyncState:
    """MTGJSON version last used to build each artifact; `None` means not built
    (or a build was interrupted).
    """

    mobilecache_version: str | None = None
    images_version: str | None = None
    phash_version: str | None = None


@dataclass(frozen=True)
class ReferenceData:
    """The reference data stored under `root`."""

    root: Path

    @classmethod
    def default(cls) -> "ReferenceData":
        return cls(settings.DATA_DIR / "reference")

    @property
    def db_path(self) -> Path:
        """The MobileCache SQLite database."""
        return self.root / "MobileCache.sqlite"

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
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    @property
    def allprintings_zip_path(self) -> Path:
        return self.tmp_dir / "AllPrintings.sqlite.zip"

    @property
    def allprintings_path(self) -> Path:
        return self.tmp_dir / "AllPrintings.sqlite"

    @property
    def log_file(self) -> Path:
        """Errors from the last image download."""
        return self.root / "download_errors.log"

    @property
    def _state_file(self) -> Path:
        return self.root / ".mtg-sync-state.json"

    def missing_images(self) -> list[tuple[str, str]]:
        """`(uuid, url)` of the printings in the database that have an image URL
        but no image file.
        """
        engine = create_engine(f"sqlite:///{self.db_path}")
        try:
            statement = select(MobileCache.uuid, MobileCache.url).where(
                col(MobileCache.url).is_not(None)
            )
            with Session(engine) as session:
                rows = session.exec(statement).all()
        finally:
            engine.dispose()
        return [
            (uuid, url)
            for uuid, url in rows
            if url is not None and not self.image_path(uuid).exists()
        ]

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
