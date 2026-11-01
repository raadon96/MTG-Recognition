"""Tests for `mtg_recognition.reference.ReferenceData`."""

import json

import pytest
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, SQLModel, create_engine

from mtg_recognition.jobs.schemas import CatalogPrinting
from mtg_recognition.reference import ReferenceData, SyncState


@pytest.mark.unit
class TestSyncState:
    def test_saved_state_loads_back(self, tmp_path) -> None:
        reference = ReferenceData(tmp_path / "reference")
        state = SyncState(
            catalog_version="5.3.0+20260302",
            phash_version=None,
        )

        reference.save_state(state)

        assert reference.load_state() == state

    def test_missing_state_file_is_all_none(self, tmp_path) -> None:
        reference = ReferenceData(tmp_path / "reference")

        assert reference.load_state() == SyncState(None, None)

    @pytest.mark.parametrize("content", ["{not json", "[]", '"V"'])
    def test_corrupt_state_file_is_all_none(self, tmp_path, content) -> None:
        reference = ReferenceData(tmp_path)
        (tmp_path / ".mtg-sync-state.json").write_text(content)

        assert reference.load_state() == SyncState(None, None)

    def test_unknown_keys_are_ignored(self, tmp_path) -> None:
        reference = ReferenceData(tmp_path)
        (tmp_path / ".mtg-sync-state.json").write_text(
            '{"catalog_version": "V", "phash_version": "V", "report": {"missing": 4}}'
        )

        assert reference.load_state() == SyncState("V", "V")

    def test_old_images_version_is_dropped_on_save(self, tmp_path) -> None:
        """State files written before images left the state still load, and are
        rewritten without the old key.
        """
        reference = ReferenceData(tmp_path)
        state_file = tmp_path / ".mtg-sync-state.json"
        state_file.write_text(
            '{"catalog_version": "V", "images_version": "V", "phash_version": "V"}'
        )

        reference.save_state(reference.load_state())

        assert json.loads(state_file.read_text()) == {
            "catalog_version": "V",
            "phash_version": "V",
        }


def _printing(uuid: str, url: str | None) -> CatalogPrinting:
    return CatalogPrinting(
        uuid=uuid, name="Name", setCode="SET", number="1", baseSetSize="1", url=url
    )


@pytest.mark.unit
class TestMissingImages:
    def test_lists_printings_without_an_image_file(self, tmp_path) -> None:
        reference = ReferenceData(tmp_path)
        engine = create_engine(f"sqlite:///{reference.catalog_path}")
        # SQLModel sets __table__ on table models at runtime but doesn't
        # declare it on the class.
        SQLModel.metadata.create_all(
            engine,
            tables=[CatalogPrinting.__table__],  # ty: ignore[unresolved-attribute]
        )
        with Session(engine) as session:
            session.add_all(
                [
                    _printing("on-disk", "https://img/on-disk.jpg"),
                    _printing("missing", "https://img/missing.jpg"),
                    _printing("no-url", None),
                ]
            )
            session.commit()
        engine.dispose()
        reference.images_dir.mkdir()
        (reference.images_dir / "on-disk.jpg").write_bytes(b"")

        assert reference.missing_images() == [("missing", "https://img/missing.jpg")]

    def test_missing_catalog_is_not_created(self, tmp_path) -> None:
        reference = ReferenceData(tmp_path)

        with pytest.raises(OperationalError):
            reference.missing_images()

        assert not reference.catalog_path.exists()
