"""Tests for `mtg_recognition.jobs.sync` step functions.

Confirms that `build_mobilecache()` and `download_images()` are module-level
callables and that `download_images()` handles the dest_engine=None case correctly.
The sync CLI command behaviour is covered by TestSyncCommand in test_cli.py.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestSyncImports:
    """Test that step functions are importable from the sync module."""

    def test_imports(self) -> None:
        """build_mobilecache, download_images, and main must be importable."""
        from mtg_recognition.jobs.sync import (
            build_mobilecache,
            download_images,
            main,
        )

        assert callable(build_mobilecache)
        assert callable(download_images)
        assert callable(main)


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
