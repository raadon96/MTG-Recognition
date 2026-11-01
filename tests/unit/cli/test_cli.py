"""Tests for the top-level `mtg` CLI group."""

import pytest
from click.testing import CliRunner

from mtg_recognition.cli import cli


@pytest.mark.unit
@pytest.mark.parametrize(
    "command", ["recognize", "segment", "status", "update", "build-index", "benchmark"]
)
def test_help_lists_every_command(command) -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert command in result.output
