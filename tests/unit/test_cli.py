"""Tests for the top-level `mtg` CLI group."""

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from mtg_recognition.cli import cli
from mtg_recognition.reference import ReferenceData, SyncState

from .sync_fakes import V1, FakeNetwork


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

    @pytest.mark.parametrize("name", ["status", "update", "build-index"])
    def test_reference_data_subcommand_registered(self, name) -> None:
        assert name in cli.commands

    def test_sync_subcommand_removed(self) -> None:
        """The old `sync` command was replaced by `status`, `update`, `build-index`."""
        assert "sync" not in cli.commands

    def test_recognize_is_same_object_as_pipeline_main(self) -> None:
        """The `recognize` command must be the same object as `pipeline.main`."""
        from mtg_recognition.engine.pipeline import main as _recognize_main

        assert cli.commands["recognize"] is _recognize_main

    def test_segment_is_same_object_as_segmentation_main(self) -> None:
        """The `segment` command must be the same object as `segmentation/cli.main`."""
        from mtg_recognition.engine.segmentation.cli import main as _segment_main

        assert cli.commands["segment"] is _segment_main


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


def invoke(tmp_path, network: FakeNetwork, args: list[str]):
    """Run `mtg <args>` in `tmp_path`, so the default reference data is
    `tmp_path/data/reference`, with the HTTP client on the fake network.
    """
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path), network.client() as client:
        return runner.invoke(cli, args, obj=client)


def state_in(tmp_path) -> SyncState:
    """The sync state of the default reference data of the last `invoke`."""
    (cwd,) = [path for path in tmp_path.iterdir() if path.name != "fake"]
    return ReferenceData(cwd / "data" / "reference").load_state()


@pytest.mark.unit
class TestReferenceDataCommands:
    """`mtg status`, `mtg update` and `mtg build-index` on the fake network; the
    steps themselves are covered in test_sync.py.
    """

    def test_build_index_help(self) -> None:
        result = CliRunner().invoke(cli, ["build-index", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.output
        assert "--workers" in result.output

    def test_update_prints_the_summary(self, tmp_path, network) -> None:
        result = invoke(tmp_path, network, ["update", "--workers", "1"])

        assert result.exit_code == 0, result.output
        assert "Update Summary" in result.output
        assert f"MobileCache: rebuilt ({V1})" in result.output
        assert "Reference images: 3 downloaded, 0 failed" in result.output
        assert "Removed images (no printing): 0" in result.output
        assert state_in(tmp_path) == SyncState(V1, V1)

    def test_update_then_build_index_force_rebuilds(self, tmp_path, network) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path), network.client() as client:
            runner.invoke(cli, ["update", "--workers", "1"], obj=client)
            up_to_date = runner.invoke(
                cli, ["build-index", "--workers", "1"], obj=client
            )
            forced = runner.invoke(
                cli, ["build-index", "--force", "--workers", "1"], obj=client
            )

        assert up_to_date.exit_code == 0, up_to_date.output
        assert f"pHash index: up to date ({V1})" in up_to_date.output
        assert forced.exit_code == 0, forced.output
        assert f"pHash index: rebuilt ({V1})" in forced.output

    def test_build_index_reports_unreadable_images(self, tmp_path, network) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path), network.client() as client:
            runner.invoke(cli, ["update", "--workers", "1"], obj=client)
            image = ReferenceData(Path("data/reference")).image_path("uuid-bolt")
            image.write_bytes(image.read_bytes()[:200])
            image.with_stem("uuid-not-a-printing").write_bytes(image.read_bytes())
            result = runner.invoke(
                cli, ["build-index", "--force", "--workers", "1"], obj=client
            )

        assert result.exit_code == 0, result.output
        assert "Removed images (no printing): 1" in result.output
        assert "Unreadable images deleted: 1; run `mtg build-index` again" in (
            result.output
        )
        assert "Image failures written to" in result.output

    def test_status_offline_exits_zero(self, tmp_path, network) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path), network.client() as client:
            runner.invoke(cli, ["update", "--workers", "1"], obj=client)
            network.offline = True
            result = runner.invoke(cli, ["status"], obj=client)

        assert result.exit_code == 0, result.output
        assert "Latest MTGJSON release: unknown" in result.output
        assert "AllPrintings: absent" not in result.output
        assert "nothing to run" in result.output

    def test_update_offline_fails(self, tmp_path, network) -> None:
        network.offline = True

        result = invoke(tmp_path, network, ["update"])

        assert result.exit_code == 1
        assert "Error: ConnectError" in result.output
