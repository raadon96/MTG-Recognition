"""`mtg app`, with `os.execvp` patched so Streamlit never starts."""

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from mtg_recognition.cli import cli


@pytest.mark.unit
def test_app_runs_streamlit_on_the_app_script_with_extra_options(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(os, "execvp", lambda file, args: calls.append((file, args)))

    result = CliRunner().invoke(cli, ["app", "--server.port", "8600"])

    assert result.exit_code == 0, result.output
    [(file, args)] = calls
    assert file == "streamlit"
    program, command, script, *options = args
    assert (program, command) == ("streamlit", "run")
    assert Path(script).is_absolute()
    assert Path(script).name == "app.py"
    assert Path(script).is_file()
    assert options == ["--server.port", "8600"]
