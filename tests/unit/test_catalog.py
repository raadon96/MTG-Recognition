"""Tests for `mtg_recognition.catalog.SqliteCardCatalog`."""

import pytest
from sqlalchemy.exc import OperationalError

from mtg_recognition.catalog import SqliteCardCatalog


@pytest.mark.unit
class TestSqliteCardCatalog:
    def test_missing_database_is_not_created(self, tmp_path) -> None:
        db_path = tmp_path / "MobileCache.sqlite"
        catalog = SqliteCardCatalog(db_path)

        with pytest.raises(OperationalError):
            catalog.lookup(["some-uuid"])

        assert not db_path.exists()
