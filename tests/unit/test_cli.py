"""Tests for the top-level `mtg` CLI group."""

import pytest
import click
from click.testing import CliRunner

from mtg_recognition.cli import cli


@pytest.mark.unit
class TestCliGroup:
    """Tests for the `cli` Click group itself."""

    def test_import_succeeds(self) -> None:
        """Importing `cli` from `mtg_recognition.cli` must not raise."""
        # If we got here, the import succeeded
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

    def test_recognize_is_same_object_as_pipeline_main(self) -> None:
        """The `recognize` command must be the same object as `pipeline.main` (identity check)."""
        from mtg_recognition.engine.pipeline import main as _recognize_main

        assert cli.commands["recognize"] is _recognize_main

    def test_segment_is_same_object_as_segmentation_main(self) -> None:
        """The `segment` command must be the same object as `segmentation/cli.main` (identity check)."""
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
