"""Tests for the top-level `mtg` CLI group."""

import subprocess
import sys

import pytest
from click.testing import CliRunner

from mtg_recognition.cli import cli


@pytest.mark.unit
@pytest.mark.parametrize(
    "command", ["recognize", "segment", "status", "update", "benchmark"]
)
def test_help_lists_every_command(command) -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert command in result.output


@pytest.mark.unit
def test_loading_the_cli_does_not_import_paddleocr() -> None:
    # importing PaddleOCR takes seconds and prints a message, which commands that
    # don't recognise anything shouldn't pay for; a subprocess, because this
    # test session may have imported it already
    check = "import sys, mtg_recognition.cli; print('paddleocr' in sys.modules)"

    result = subprocess.run(
        [sys.executable, "-c", check], capture_output=True, text=True, check=True
    )

    assert result.stdout == "False\n"
