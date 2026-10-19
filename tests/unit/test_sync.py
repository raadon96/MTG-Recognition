"""Tests for `mtg_recognition.jobs.sync` step functions.

Predicate functions (_needs_*_rebuild) are covered by TestNeedsRebuild.
download_images() logging is covered by TestDownloadImages.
The sync CLI command behaviour is covered by TestSyncCommand in test_cli.py.
"""

from unittest.mock import patch

import pytest

from mtg_recognition.reference import ReferenceData, SyncState


@pytest.fixture
def reference(tmp_path) -> ReferenceData:
    return ReferenceData(tmp_path)


@pytest.mark.unit
class TestNeedsRebuild:
    """Unit tests for _needs_mobilecache_rebuild, _needs_images_rebuild,
    _needs_phash_rebuild.
    """

    # --- _needs_mobilecache_rebuild ---

    def test_mobilecache_force_always_rebuilds(self, reference) -> None:
        """force=True returns True regardless of state or file existence."""
        from mtg_recognition.jobs.sync import _needs_mobilecache_rebuild

        state = SyncState("V", "V", "V")
        assert (
            _needs_mobilecache_rebuild(reference, state, remote_version="V", force=True)
            is True
        )

    def test_mobilecache_null_version_rebuilds(self, reference) -> None:
        """null mobilecache_version returns True (short-circuits before file check)."""
        from mtg_recognition.jobs.sync import _needs_mobilecache_rebuild

        state = SyncState(None, None, None)
        assert (
            _needs_mobilecache_rebuild(
                reference, state, remote_version="V", force=False
            )
            is True
        )

    def test_mobilecache_version_mismatch_rebuilds(self, reference) -> None:
        """Stale local version triggers rebuild (short-circuits before file check)."""
        from mtg_recognition.jobs.sync import _needs_mobilecache_rebuild

        state = SyncState("V_old", "V_old", "V_old")
        assert (
            _needs_mobilecache_rebuild(
                reference, state, remote_version="V_new", force=False
            )
            is True
        )

    def test_mobilecache_up_to_date_skips(self, reference) -> None:
        """Matching version + file exists returns False."""
        from mtg_recognition.jobs.sync import _needs_mobilecache_rebuild

        reference.db_path.write_bytes(b"")

        state = SyncState("V", "V", "V")
        assert (
            _needs_mobilecache_rebuild(
                reference, state, remote_version="V", force=False
            )
            is False
        )

    def test_mobilecache_file_missing_rebuilds(self, reference) -> None:
        """Matching version but the database file absent triggers rebuild."""
        from mtg_recognition.jobs.sync import _needs_mobilecache_rebuild

        state = SyncState("V", "V", "V")
        assert (
            _needs_mobilecache_rebuild(
                reference, state, remote_version="V", force=False
            )
            is True
        )

    def test_mobilecache_skips_without_update_db(self, reference) -> None:
        """remote_version=None (no --update-db) never rebuilds MobileCache."""
        from mtg_recognition.jobs.sync import _needs_mobilecache_rebuild

        # Version mismatch — would normally rebuild, but remote_version=None pins the DB
        state = SyncState("V_old", "V_old", "V_old")
        assert (
            _needs_mobilecache_rebuild(
                reference, state, remote_version=None, force=False
            )
            is False
        )

    # --- _needs_images_rebuild ---

    def test_images_force_always_rebuilds(self, reference) -> None:
        """force=True returns True regardless of state."""
        from mtg_recognition.jobs.sync import _needs_images_rebuild

        state = SyncState("V", "V", "V")
        assert _needs_images_rebuild(reference, state, force=True) is True

    def test_images_null_version_rebuilds(self, reference) -> None:
        """null images_version returns True."""
        from mtg_recognition.jobs.sync import _needs_images_rebuild

        state = SyncState("V", None, None)
        assert _needs_images_rebuild(reference, state, force=False) is True

    def test_images_version_mismatch_rebuilds(self, reference) -> None:
        """images_version != mobilecache_version returns True."""
        from mtg_recognition.jobs.sync import _needs_images_rebuild

        state = SyncState("V_new", "V_old", "V_old")
        assert _needs_images_rebuild(reference, state, force=False) is True

    def test_images_up_to_date_skips(self, reference) -> None:
        """images_version matches mobilecache_version and the images directory is
        non-empty returns False.
        """
        from mtg_recognition.jobs.sync import _needs_images_rebuild

        reference.images_dir.mkdir()
        reference.image_path("dummy").write_bytes(b"")

        state = SyncState("V", "V", "V")
        assert _needs_images_rebuild(reference, state, force=False) is False

    def test_images_dir_missing_rebuilds(self, reference) -> None:
        """Missing the images directory triggers rebuild even when versions match."""
        from mtg_recognition.jobs.sync import _needs_images_rebuild

        state = SyncState("V", "V", "V")
        assert _needs_images_rebuild(reference, state, force=False) is True

    def test_images_dir_empty_rebuilds(self, reference) -> None:
        """Empty the images directory triggers rebuild even when versions match."""
        from mtg_recognition.jobs.sync import _needs_images_rebuild

        reference.images_dir.mkdir()

        state = SyncState("V", "V", "V")
        assert _needs_images_rebuild(reference, state, force=False) is True

    # --- _needs_phash_rebuild ---

    def test_phash_force_always_rebuilds(self, reference) -> None:
        """force=True returns True regardless of state or file existence."""
        from mtg_recognition.jobs.sync import _needs_phash_rebuild

        state = SyncState("V", "V", "V")
        assert _needs_phash_rebuild(reference, state, force=True) is True

    def test_phash_null_version_rebuilds(self, reference) -> None:
        """null phash_version returns True."""
        from mtg_recognition.jobs.sync import _needs_phash_rebuild

        state = SyncState("V", "V", None)
        assert _needs_phash_rebuild(reference, state, force=False) is True

    def test_phash_version_mismatch_rebuilds(self, reference) -> None:
        """phash_version != images_version returns True."""
        from mtg_recognition.jobs.sync import _needs_phash_rebuild

        state = SyncState("V_new", "V_new", "V_old")
        assert _needs_phash_rebuild(reference, state, force=False) is True

    def test_phash_files_missing_rebuilds(self, reference) -> None:
        """Matching versions but index files absent returns True."""
        from mtg_recognition.jobs.sync import _needs_phash_rebuild

        state = SyncState("V", "V", "V")
        assert _needs_phash_rebuild(reference, state, force=False) is True

    def test_phash_up_to_date_skips(self, reference) -> None:
        """Matching versions + both index files exist returns False."""
        from mtg_recognition.jobs.sync import _needs_phash_rebuild

        hashes, mapping = reference.phash_files
        hashes.parent.mkdir(parents=True)
        hashes.write_bytes(b"")
        mapping.write_text("{}")

        state = SyncState("V", "V", "V")
        assert _needs_phash_rebuild(reference, state, force=False) is False


@pytest.mark.unit
class TestDownloadImages:
    """Test download_images() logging."""

    def test_download_errors_go_to_the_log_file(self, reference) -> None:
        """Errors logged during the download land in reference.log_file, and the
        file handler is detached afterwards.
        """
        import mtg_recognition.jobs.sync as sync_module

        handlers_before = list(sync_module.logger.handlers)

        def fake_download(ref) -> None:
            assert ref == reference
            sync_module.logger.error("404 Not Found: https://img/x.jpg")

        with patch.object(sync_module, "download_raw_images", fake_download):
            sync_module.download_images(reference)

        assert "404 Not Found: https://img/x.jpg" in reference.log_file.read_text()
        assert sync_module.logger.handlers == handlers_before
