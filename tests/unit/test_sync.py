"""Tests for `mtg_recognition.jobs.sync` step functions.

Predicate functions (_needs_*_rebuild) are covered by TestNeedsRebuild.
download_images() engine-passing behaviour is covered by TestDownloadImages.
The sync CLI command behaviour is covered by TestSyncCommand in test_cli.py.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestNeedsRebuild:
    """Unit tests for _needs_mobilecache_rebuild, _needs_images_rebuild, _needs_phash_rebuild."""

    # --- _needs_mobilecache_rebuild ---

    def test_mobilecache_force_always_rebuilds(self) -> None:
        """force=True returns True regardless of state or file existence."""
        from mtg_recognition.jobs.sync import _needs_mobilecache_rebuild

        state = {"mobilecache_version": "V", "images_version": "V", "phash_version": "V"}
        assert _needs_mobilecache_rebuild(state, remote_version="V", force=True) is True

    def test_mobilecache_null_version_rebuilds(self) -> None:
        """null mobilecache_version returns True (short-circuits before file check)."""
        from mtg_recognition.jobs.sync import _needs_mobilecache_rebuild

        state = {"mobilecache_version": None, "images_version": None, "phash_version": None}
        assert _needs_mobilecache_rebuild(state, remote_version="V", force=False) is True

    def test_mobilecache_version_mismatch_rebuilds(self) -> None:
        """Stale local version triggers rebuild (short-circuits before file check)."""
        from mtg_recognition.jobs.sync import _needs_mobilecache_rebuild

        state = {"mobilecache_version": "V_old", "images_version": "V_old", "phash_version": "V_old"}
        assert _needs_mobilecache_rebuild(state, remote_version="V_new", force=False) is True

    def test_mobilecache_up_to_date_skips(self, tmp_path, monkeypatch) -> None:
        """Matching version + file exists returns False."""
        from mtg_recognition.jobs.sync import _needs_mobilecache_rebuild
        from mtg_recognition.settings import settings

        db = tmp_path / "MobileCache.sqlite"
        db.write_bytes(b"")
        monkeypatch.setattr(type(settings), "LOCAL_DB_FILE", property(lambda self: db))

        state = {"mobilecache_version": "V", "images_version": "V", "phash_version": "V"}
        assert _needs_mobilecache_rebuild(state, remote_version="V", force=False) is False

    def test_mobilecache_file_missing_rebuilds(self, tmp_path, monkeypatch) -> None:
        """Matching version but LOCAL_DB_FILE absent triggers rebuild."""
        from mtg_recognition.jobs.sync import _needs_mobilecache_rebuild
        from mtg_recognition.settings import settings

        missing = tmp_path / "MobileCache.sqlite"
        monkeypatch.setattr(type(settings), "LOCAL_DB_FILE", property(lambda self: missing))

        state = {"mobilecache_version": "V", "images_version": "V", "phash_version": "V"}
        assert _needs_mobilecache_rebuild(state, remote_version="V", force=False) is True

    # --- _needs_images_rebuild ---

    def test_images_force_always_rebuilds(self) -> None:
        """force=True returns True regardless of state."""
        from mtg_recognition.jobs.sync import _needs_images_rebuild

        state = {"mobilecache_version": "V", "images_version": "V", "phash_version": "V"}
        assert _needs_images_rebuild(state, force=True) is True

    def test_images_null_version_rebuilds(self) -> None:
        """null images_version returns True."""
        from mtg_recognition.jobs.sync import _needs_images_rebuild

        state = {"mobilecache_version": "V", "images_version": None, "phash_version": None}
        assert _needs_images_rebuild(state, force=False) is True

    def test_images_version_mismatch_rebuilds(self) -> None:
        """images_version != mobilecache_version returns True."""
        from mtg_recognition.jobs.sync import _needs_images_rebuild

        state = {"mobilecache_version": "V_new", "images_version": "V_old", "phash_version": "V_old"}
        assert _needs_images_rebuild(state, force=False) is True

    def test_images_up_to_date_skips(self) -> None:
        """images_version matches mobilecache_version returns False."""
        from mtg_recognition.jobs.sync import _needs_images_rebuild

        state = {"mobilecache_version": "V", "images_version": "V", "phash_version": "V"}
        assert _needs_images_rebuild(state, force=False) is False

    # --- _needs_phash_rebuild ---

    def test_phash_force_always_rebuilds(self) -> None:
        """force=True returns True regardless of state or file existence."""
        from mtg_recognition.jobs.sync import _needs_phash_rebuild

        state = {"mobilecache_version": "V", "images_version": "V", "phash_version": "V"}
        assert _needs_phash_rebuild(state, force=True) is True

    def test_phash_null_version_rebuilds(self) -> None:
        """null phash_version returns True."""
        from mtg_recognition.jobs.sync import _needs_phash_rebuild

        state = {"mobilecache_version": "V", "images_version": "V", "phash_version": None}
        assert _needs_phash_rebuild(state, force=False) is True

    def test_phash_version_mismatch_rebuilds(self) -> None:
        """phash_version != images_version returns True."""
        from mtg_recognition.jobs.sync import _needs_phash_rebuild

        state = {"mobilecache_version": "V_new", "images_version": "V_new", "phash_version": "V_old"}
        assert _needs_phash_rebuild(state, force=False) is True

    def test_phash_files_missing_rebuilds(self, tmp_path, monkeypatch) -> None:
        """Matching versions but index files absent returns True."""
        from mtg_recognition.jobs.sync import _needs_phash_rebuild
        from mtg_recognition.settings import settings

        missing = tmp_path / "does_not_exist"
        monkeypatch.setattr(type(settings), "PHASH_INDEX_FILE", property(lambda self: missing))
        monkeypatch.setattr(type(settings), "PHASH_MAPPING_FILE", property(lambda self: missing))

        state = {"mobilecache_version": "V", "images_version": "V", "phash_version": "V"}
        assert _needs_phash_rebuild(state, force=False) is True

    def test_phash_up_to_date_skips(self, tmp_path, monkeypatch) -> None:
        """Matching versions + both index files exist returns False."""
        from mtg_recognition.jobs.sync import _needs_phash_rebuild
        from mtg_recognition.settings import settings

        phash_dir = tmp_path / "index" / "phash"
        phash_dir.mkdir(parents=True)
        hashes = phash_dir / "hashes.npy"
        mapping = phash_dir / "index_to_uuid.json"
        hashes.write_bytes(b"")
        mapping.write_text("{}")
        monkeypatch.setattr(type(settings), "PHASH_INDEX_FILE", property(lambda self: hashes))
        monkeypatch.setattr(type(settings), "PHASH_MAPPING_FILE", property(lambda self: mapping))

        state = {"mobilecache_version": "V", "images_version": "V", "phash_version": "V"}
        assert _needs_phash_rebuild(state, force=False) is False


@pytest.mark.unit
class TestDownloadImages:
    """Test download_images() standalone and engine-passing behaviour."""

    def test_download_images_creates_engine_when_none(self) -> None:
        """When dest_engine=None, download_images() must create a SQLite engine."""
        import mtg_recognition.jobs.sync as sync_module

        mock_engine = MagicMock()

        with (
            patch.object(
                sync_module, "create_engine", return_value=mock_engine
            ) as mock_ce,
            patch.object(sync_module, "download_raw_images") as mock_dl,
        ):
            sync_module.download_images(dest_engine=None)

        mock_ce.assert_called_once()
        # The URL passed to create_engine must be a SQLite URL containing MobileCache.sqlite
        call_args = mock_ce.call_args[0][0]
        assert "sqlite:///" in call_args
        assert "MobileCache.sqlite" in call_args
        mock_dl.assert_called_once_with(mock_engine)

    def test_download_images_uses_provided_engine(self) -> None:
        """When a dest_engine is provided, create_engine must NOT be called."""
        import mtg_recognition.jobs.sync as sync_module

        sentinel_engine = MagicMock(name="sentinel_engine")

        with (
            patch.object(sync_module, "create_engine") as mock_ce,
            patch.object(sync_module, "download_raw_images") as mock_dl,
        ):
            sync_module.download_images(dest_engine=sentinel_engine)

        mock_ce.assert_not_called()
        mock_dl.assert_called_once_with(sentinel_engine)
