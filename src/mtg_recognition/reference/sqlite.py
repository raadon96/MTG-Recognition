"""Opening the reference data's SQLite files: the catalog and AllPrintings."""

from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import create_engine


def open_read_only(db_path: Path) -> Engine:
    """Engine on the SQLite database at `db_path`, opened read-only: it never
    writes and never creates a missing file. (`mode=ro` is only honoured in a
    SQLite URI, hence the `file:` URI and `uri=true`.)
    """
    return create_engine(
        f"sqlite:///{db_path.resolve().as_uri()}?mode=ro&nolock=1&uri=true"
    )
