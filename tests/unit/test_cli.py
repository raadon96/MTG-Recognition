"""Tests for the top-level `mtg` CLI group."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from mtg_recognition.cli import cli
from mtg_recognition.reference import ReferenceData, SyncState

from .sync_fakes import V1, FakeNetwork


@pytest.mark.unit
@pytest.mark.parametrize(
    "command", ["recognize", "segment", "status", "update", "build-index"]
)
def test_help_lists_every_command(command) -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert command in result.output


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
