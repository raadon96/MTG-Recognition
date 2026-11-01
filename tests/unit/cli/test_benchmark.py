"""`mtg benchmark update-baseline`, on results files written in the test."""

import pytest
from click.testing import CliRunner

from mtg_recognition.benchmark.results import (
    BASELINE_PATH,
    LATEST_PATH,
    read_results,
    summarise,
    write_results,
)
from mtg_recognition.cli import cli


@pytest.mark.unit
def test_update_baseline_adopts_the_latest_results(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    latest = summarise("abc123", "5.3.0", {})
    write_results(latest, LATEST_PATH)

    result = CliRunner().invoke(cli, ["benchmark", "update-baseline"])

    assert result.exit_code == 0
    assert read_results(BASELINE_PATH) == latest


@pytest.mark.unit
def test_update_baseline_without_latest_results_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["benchmark", "update-baseline"])

    assert result.exit_code == 1
    assert not BASELINE_PATH.exists()
