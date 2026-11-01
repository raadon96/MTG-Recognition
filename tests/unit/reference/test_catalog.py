"""Tests for `mtg_recognition.reference.catalog.SqliteCardCatalog`."""

import pytest
from sqlalchemy.exc import OperationalError

from mtg_recognition.reference.catalog import SqliteCardCatalog


@pytest.mark.unit
class TestSqliteCardCatalog:
    def test_missing_catalog_is_not_created(self, tmp_path) -> None:
        catalog_path = tmp_path / "catalog.sqlite"
        catalog = SqliteCardCatalog(catalog_path)

        with pytest.raises(OperationalError):
            catalog.lookup(["some-uuid"])

        assert not catalog_path.exists()
