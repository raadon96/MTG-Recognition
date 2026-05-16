"""Tests for the refactored `build_database.py` step functions.

Confirms that `build_mobilecache()` and `download_images()` are module-level
callables, that `main()` delegates to both, and backward-compat import is preserved.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.unit
class TestBuildDatabaseImports:
    """Test that refactored step functions are importable."""

    def test_imports(self) -> None:
        """All three symbols must be importable from the module."""
        from mtg_recognition.jobs.build_database import (
            build_mobilecache,
            download_images,
            main,
        )

        assert callable(build_mobilecache)
        assert callable(download_images)
        assert callable(main)


@pytest.mark.unit
class TestMainDelegates:
    """Test that main() calls both step functions."""

    def test_main_calls_both_steps(self) -> None:
        """main() must call build_mobilecache() and pass the result to download_images()."""
        import mtg_recognition.jobs.build_database as db_module

        mock_engine = MagicMock()

        with (
            patch.object(db_module, "build_mobilecache", return_value=mock_engine) as mock_build,
            patch.object(db_module, "download_images") as mock_download,
        ):
            db_module.main()

        assert mock_build.called, "main() must call build_mobilecache()"
        assert mock_download.called, "main() must call download_images()"
        mock_download.assert_called_once_with(mock_engine)


@pytest.mark.unit
class TestDownloadImages:
    """Test download_images() standalone and engine-passing behaviour."""

    def test_download_images_creates_engine_when_none(self) -> None:
        """When dest_engine=None, download_images() must create a SQLite engine."""
        import mtg_recognition.jobs.build_database as db_module

        mock_engine = MagicMock()

        with (
            patch.object(db_module, "create_engine", return_value=mock_engine) as mock_ce,
            patch.object(db_module, "download_raw_images") as mock_dl,
        ):
            db_module.download_images(dest_engine=None)

        mock_ce.assert_called_once()
        # The URL passed to create_engine must be a SQLite URL containing MobileCache.sqlite
        call_args = mock_ce.call_args[0][0]
        assert "sqlite:///" in call_args
        assert "MobileCache.sqlite" in call_args
        mock_dl.assert_called_once_with(mock_engine)

    def test_download_images_uses_provided_engine(self) -> None:
        """When a dest_engine is provided, create_engine must NOT be called."""
        import mtg_recognition.jobs.build_database as db_module

        sentinel_engine = MagicMock(name="sentinel_engine")

        with (
            patch.object(db_module, "create_engine") as mock_ce,
            patch.object(db_module, "download_raw_images") as mock_dl,
        ):
            db_module.download_images(dest_engine=sentinel_engine)

        mock_ce.assert_not_called()
        mock_dl.assert_called_once_with(sentinel_engine)
