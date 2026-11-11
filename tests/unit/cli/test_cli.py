"""Tests for the top-level `mtg` CLI group."""

import pytest
from click.testing import CliRunner

from mtg_recognition.cli import cli


@pytest.mark.unit
@pytest.mark.parametrize(
    ("group", "commands"),
    [
        ([], ["recognize", "segment", "reference", "benchmark"]),
        (["reference"], ["status", "update", "build", "download-images"]),
    ],
)
def test_help_lists_every_command(group, commands) -> None:
    result = CliRunner().invoke(cli, [*group, "--help"])

    assert result.exit_code == 0
    for command in commands:
        assert command in result.output
