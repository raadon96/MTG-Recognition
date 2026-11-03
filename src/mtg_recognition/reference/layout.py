"""`ReferenceData` owns the on-disk layout of the reference data, so no other
module builds these paths itself.
"""

from dataclasses import dataclass
from pathlib import Path

from mtg_recognition.settings import settings


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
        """Images that failed to download or hash, from the last `mtg update`,
        or the last `mtg download-images` with a failed download, whichever ran
        later. `mtg update` removes it when it has no failure.
        """
        return self.root / "image_errors.log"
