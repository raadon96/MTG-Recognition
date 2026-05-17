"""Tests for the top-level `mtg` CLI group."""

import click
import pytest
from click.testing import CliRunner

from mtg_recognition.cli import cli


@pytest.mark.unit
class TestCliGroup:
    """Tests for the `cli` Click group itself."""

    def test_import_succeeds(self) -> None:
        """Importing `cli` from `mtg_recognition.cli` must not raise."""
        assert cli is not None

    def test_cli_is_click_group(self) -> None:
        """`cli` must be a Click Group instance."""
        assert isinstance(cli, click.Group)

    def test_recognize_subcommand_registered(self) -> None:
        """`cli.commands` must contain the `recognize` key."""
        assert "recognize" in cli.commands

    def test_segment_subcommand_registered(self) -> None:
        """`cli.commands` must contain the `segment` key."""
        assert "segment" in cli.commands

    def test_build_index_subcommand_registered(self) -> None:
        """`cli.commands` must contain the `build-index` key."""
        assert "build-index" in cli.commands

    def test_sync_subcommand_registered(self) -> None:
        """`cli.commands` must contain the `sync` key."""
        assert "sync" in cli.commands

    def test_recognize_is_same_object_as_pipeline_main(self) -> None:
        """The `recognize` command must be the same object as `pipeline.main`."""
        from mtg_recognition.engine.pipeline import main as _recognize_main

        assert cli.commands["recognize"] is _recognize_main

    def test_segment_is_same_object_as_segmentation_main(self) -> None:
        """The `segment` command must be the same object as `segmentation/cli.main`."""
        from mtg_recognition.engine.segmentation.cli import main as _segment_main

        assert cli.commands["segment"] is _segment_main

    def test_build_index_is_same_object_as_phash_build_index(self) -> None:
        """The `build-index` command must be the same object as `phash_matcher.build_index`."""
        from mtg_recognition.engine.matching.phash_matcher import (
            build_index as _build_index_cmd,
        )

        assert cli.commands["build-index"] is _build_index_cmd

    def test_sync_is_same_object_as_sync_main(self) -> None:
        """The `sync` command must be the same object as `jobs.sync.main`."""
        from mtg_recognition.jobs.sync import main as _sync_main

        assert cli.commands["sync"] is _sync_main


@pytest.mark.unit
class TestCliHelp:
    """Tests for `--help` output of the CLI group and subcommands."""

    def test_group_help_exits_zero(self) -> None:
        """`mtg --help` must exit with code 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_group_help_lists_recognize(self) -> None:
        """`mtg --help` stdout must list `recognize`."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "recognize" in result.output

    def test_group_help_lists_segment(self) -> None:
        """`mtg --help` stdout must list `segment`."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "segment" in result.output

    def test_recognize_help_exits_zero(self) -> None:
        """`mtg recognize --help` must exit with code 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["recognize", "--help"])
        assert result.exit_code == 0

    def test_recognize_help_contains_input_option(self) -> None:
        """`mtg recognize --help` must show the `--input` option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["recognize", "--help"])
        assert "--input" in result.output

    def test_recognize_help_contains_k_phash_option(self) -> None:
        """`mtg recognize --help` must show the `--k-phash` option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["recognize", "--help"])
        assert "--k-phash" in result.output

    def test_recognize_help_contains_k_option(self) -> None:
        """`mtg recognize --help` must show the `-k` option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["recognize", "--help"])
        assert "-k" in result.output

    def test_segment_help_exits_zero(self) -> None:
        """`mtg segment --help` must exit with code 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["segment", "--help"])
        assert result.exit_code == 0

    def test_segment_help_contains_input_option(self) -> None:
        """`mtg segment --help` must show the `--input` option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["segment", "--help"])
        assert "--input" in result.output

    def test_segment_help_contains_mode_option(self) -> None:
        """`mtg segment --help` must show the `--mode` option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["segment", "--help"])
        assert "--mode" in result.output


@pytest.mark.unit
class TestBuildIndexCommand:
    """Tests for the `mtg build-index` subcommand."""

    def test_build_index_help(self) -> None:
        """`mtg build-index --help` must exit 0 and list `--force` and `--workers`."""
        runner = CliRunner()
        result = runner.invoke(cli, ["build-index", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.output
        assert "--workers" in result.output

    def test_build_index_preflight_missing_dir(self, tmp_path, monkeypatch) -> None:
        """Exit code 1 when IMG_DIR does not exist; output contains path and 'mtg sync'."""
        nonexistent = tmp_path / "does_not_exist"
        from mtg_recognition.settings import settings

        monkeypatch.setattr(
            type(settings), "IMG_DIR", property(lambda self: nonexistent)
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["build-index"])
        assert result.exit_code == 1
        assert "data/reference/images/" in result.output
        assert "mtg sync" in result.output

    def test_build_index_preflight_empty_dir(self, tmp_path, monkeypatch) -> None:
        """Exit code 1 when IMG_DIR exists but is empty; output contains path and 'mtg sync'."""
        empty_dir = tmp_path / "images"
        empty_dir.mkdir()
        from mtg_recognition.settings import settings

        monkeypatch.setattr(type(settings), "IMG_DIR", property(lambda self: empty_dir))
        runner = CliRunner()
        result = runner.invoke(cli, ["build-index"])
        assert result.exit_code == 1
        assert "data/reference/images/" in result.output
        assert "mtg sync" in result.output

    def test_build_index_invokes_matcher_with_defaults(
        self, tmp_path, monkeypatch
    ) -> None:
        """Invokes PHashCardMatcher(auto_build=False).build_index(force=False, num_workers=<default>)."""
        from multiprocessing import cpu_count
        from unittest.mock import MagicMock, patch

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "dummy.jpg").write_bytes(b"")

        from mtg_recognition.settings import settings

        monkeypatch.setattr(type(settings), "IMG_DIR", property(lambda self: img_dir))

        mock_matcher_instance = MagicMock()
        with patch(
            "mtg_recognition.engine.matching.phash_matcher.PHashCardMatcher",
            return_value=mock_matcher_instance,
        ) as MockMatcher:
            runner = CliRunner()
            result = runner.invoke(cli, ["build-index"])
        assert result.exit_code == 0
        MockMatcher.assert_called_once_with(auto_build=False)
        mock_matcher_instance.build_index.assert_called_once_with(
            force=False, num_workers=cpu_count() - 1
        )

    def test_build_index_invokes_matcher_with_flags(
        self, tmp_path, monkeypatch
    ) -> None:
        """--force and --workers flags are passed through to build_index."""
        from unittest.mock import MagicMock, patch

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "dummy.jpg").write_bytes(b"")

        from mtg_recognition.settings import settings

        monkeypatch.setattr(type(settings), "IMG_DIR", property(lambda self: img_dir))

        mock_matcher_instance = MagicMock()
        with patch(
            "mtg_recognition.engine.matching.phash_matcher.PHashCardMatcher",
            return_value=mock_matcher_instance,
        ) as MockMatcher:
            runner = CliRunner()
            result = runner.invoke(cli, ["build-index", "--force", "--workers", "4"])
        assert result.exit_code == 0
        MockMatcher.assert_called_once_with(auto_build=False)
        mock_matcher_instance.build_index.assert_called_once_with(
            force=True, num_workers=4
        )


@pytest.mark.unit
class TestSyncCommand:
    """Tests for the `mtg sync` subcommand."""

    def test_sync_help(self) -> None:
        """`mtg sync --help` exits 0 and lists --force."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sync", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.output

    def test_sync_first_run_runs_all_steps(self, tmp_path, monkeypatch) -> None:
        """First run (no state file, all missing): all three steps execute and state file is written."""
        import json
        from unittest.mock import MagicMock, patch

        ref_dir = tmp_path / "reference"
        local_db = ref_dir / "MobileCache.sqlite"

        from mtg_recognition.settings import settings

        monkeypatch.setattr(type(settings), "REF_DIR", property(lambda self: ref_dir))
        monkeypatch.setattr(
            type(settings), "LOCAL_DB_FILE", property(lambda self: local_db)
        )

        mock_engine = MagicMock()
        mock_matcher_instance = MagicMock()

        with (
            patch(
                "mtg_recognition.jobs.sync._fetch_mtgjson_version",
                return_value="5.3.0+20260302",
            ),
            patch(
                "mtg_recognition.jobs.sync.build_mobilecache", return_value=mock_engine
            ) as mock_build,
            patch("mtg_recognition.jobs.sync.download_images") as mock_download,
            patch(
                "mtg_recognition.jobs.sync.PHashCardMatcher",
                return_value=mock_matcher_instance,
            ) as MockMatcher,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["sync"])

        assert result.exit_code == 0, (
            f"Exit code {result.exit_code}, output: {result.output}"
        )
        mock_build.assert_called_once()
        mock_download.assert_called_once()
        MockMatcher.assert_called_once_with(auto_build=False)
        mock_matcher_instance.build_index.assert_called_once()

        # State file must exist with three version strings
        state_file = ref_dir / ".mtg-sync-state.json"
        assert state_file.exists(), "State file must be written"
        state = json.loads(state_file.read_text())
        assert state["mobilecache_version"] == "5.3.0+20260302"
        assert state["images_version"] == "5.3.0+20260302"
        assert state["phash_version"] == "5.3.0+20260302"

    def test_sync_skips_when_versions_match(self, tmp_path, monkeypatch) -> None:
        """Second run with matching versions: all three steps are skipped."""
        import json
        from unittest.mock import patch

        ref_dir = tmp_path / "reference"
        ref_dir.mkdir(parents=True)
        local_db = ref_dir / "MobileCache.sqlite"
        local_db.write_bytes(b"")

        from mtg_recognition.settings import settings

        monkeypatch.setattr(type(settings), "REF_DIR", property(lambda self: ref_dir))
        monkeypatch.setattr(
            type(settings), "LOCAL_DB_FILE", property(lambda self: local_db)
        )
        state_file = ref_dir / ".mtg-sync-state.json"
        # Stub phash index files so the file-existence guard in sync.py doesn't force a rebuild
        phash_dir = ref_dir / "index" / "phash"
        phash_dir.mkdir(parents=True)
        (phash_dir / "hashes.npy").write_bytes(b"")
        (phash_dir / "index_to_uuid.json").write_text("{}")

        state_file.write_text(
            json.dumps(
                {
                    "mobilecache_version": "V",
                    "images_version": "V",
                    "phash_version": "V",
                }
            )
        )

        with (
            patch("mtg_recognition.jobs.sync._fetch_mtgjson_version", return_value="V"),
            patch("mtg_recognition.jobs.sync.build_mobilecache") as mock_build,
            patch("mtg_recognition.jobs.sync.download_images") as mock_download,
            patch("mtg_recognition.jobs.sync.PHashCardMatcher") as MockMatcher,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["sync"])

        assert result.exit_code == 0, (
            f"Exit code {result.exit_code}, output: {result.output}"
        )
        mock_build.assert_not_called()
        mock_download.assert_not_called()
        MockMatcher.assert_not_called()

        assert "skipping" in result.output
        assert result.output.count("✓") >= 3

    def test_sync_force_rebuilds_all(self, tmp_path, monkeypatch) -> None:
        """--force rebuilds all three steps even when state versions match."""
        import json
        from unittest.mock import MagicMock, patch

        ref_dir = tmp_path / "reference"
        ref_dir.mkdir(parents=True)
        local_db = ref_dir / "MobileCache.sqlite"
        state_file = ref_dir / ".mtg-sync-state.json"

        from mtg_recognition.settings import settings

        monkeypatch.setattr(type(settings), "REF_DIR", property(lambda self: ref_dir))
        monkeypatch.setattr(
            type(settings), "LOCAL_DB_FILE", property(lambda self: local_db)
        )

        state_file.write_text(
            json.dumps(
                {
                    "mobilecache_version": "V",
                    "images_version": "V",
                    "phash_version": "V",
                }
            )
        )

        mock_engine = MagicMock()
        mock_matcher_instance = MagicMock()

        with (
            patch("mtg_recognition.jobs.sync._fetch_mtgjson_version", return_value="V"),
            patch(
                "mtg_recognition.jobs.sync.build_mobilecache", return_value=mock_engine
            ) as mock_build,
            patch("mtg_recognition.jobs.sync.download_images") as mock_download,
            patch(
                "mtg_recognition.jobs.sync.PHashCardMatcher",
                return_value=mock_matcher_instance,
            ),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["sync", "--force"])

        assert result.exit_code == 0, (
            f"Exit code {result.exit_code}, output: {result.output}"
        )
        mock_build.assert_called_once()
        mock_download.assert_called_once()
        mock_matcher_instance.build_index.assert_called_once()

    def test_sync_step1_failure_leaves_null_state(self, tmp_path, monkeypatch) -> None:
        """When Step 1 raises, exit code is 1, and state file has mobilecache_version: null."""
        import json
        from unittest.mock import patch

        ref_dir = tmp_path / "reference"
        ref_dir.mkdir(parents=True)
        local_db = ref_dir / "MobileCache.sqlite"
        state_file = ref_dir / ".mtg-sync-state.json"

        from mtg_recognition.settings import settings

        monkeypatch.setattr(type(settings), "REF_DIR", property(lambda self: ref_dir))
        monkeypatch.setattr(
            type(settings), "LOCAL_DB_FILE", property(lambda self: local_db)
        )

        with (
            patch(
                "mtg_recognition.jobs.sync._fetch_mtgjson_version",
                return_value="5.3.0+20260302",
            ),
            patch(
                "mtg_recognition.jobs.sync.build_mobilecache",
                side_effect=RuntimeError("boom"),
            ),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["sync"])

        assert result.exit_code == 1
        assert "boom" in result.output

        assert state_file.exists(), "State file must exist (null written before work)"
        state = json.loads(state_file.read_text())
        assert state["mobilecache_version"] is None

    def test_sync_rebuilds_when_remote_version_changes(
        self, tmp_path, monkeypatch
    ) -> None:
        """When remote version differs from state, all three steps rebuild without --force."""
        import json
        from unittest.mock import MagicMock, patch

        ref_dir = tmp_path / "reference"
        ref_dir.mkdir(parents=True)
        local_db = ref_dir / "MobileCache.sqlite"
        local_db.write_bytes(b"")

        from mtg_recognition.settings import settings

        monkeypatch.setattr(type(settings), "REF_DIR", property(lambda self: ref_dir))
        monkeypatch.setattr(
            type(settings), "LOCAL_DB_FILE", property(lambda self: local_db)
        )

        state_file = ref_dir / ".mtg-sync-state.json"
        phash_dir = ref_dir / "index" / "phash"
        phash_dir.mkdir(parents=True)
        (phash_dir / "hashes.npy").write_bytes(b"")
        (phash_dir / "index_to_uuid.json").write_text("{}")
        state_file.write_text(
            json.dumps(
                {
                    "mobilecache_version": "V_old",
                    "images_version": "V_old",
                    "phash_version": "V_old",
                }
            )
        )

        mock_engine = MagicMock()
        mock_matcher_instance = MagicMock()

        with (
            patch(
                "mtg_recognition.jobs.sync._fetch_mtgjson_version",
                return_value="V_new",
            ),
            patch(
                "mtg_recognition.jobs.sync.build_mobilecache", return_value=mock_engine
            ) as mock_build,
            patch("mtg_recognition.jobs.sync.download_images") as mock_download,
            patch(
                "mtg_recognition.jobs.sync.PHashCardMatcher",
                return_value=mock_matcher_instance,
            ),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["sync"])

        assert result.exit_code == 0, (
            f"Exit code {result.exit_code}, output: {result.output}"
        )
        mock_build.assert_called_once()
        mock_download.assert_called_once()
        mock_matcher_instance.build_index.assert_called_once()

        state = json.loads(state_file.read_text())
        assert state["mobilecache_version"] == "V_new"
        assert state["images_version"] == "V_new"
        assert state["phash_version"] == "V_new"
