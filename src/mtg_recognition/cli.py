"""Top-level `mtg` CLI group.

Registers all subcommands via Click's add_command so each domain module owns
its own Click command definition:

  mtg recognize   — full recognition pipeline           (engine/pipeline.py)
  mtg segment     — card segmentation                   (engine/segmentation/cli.py)
  mtg status      — reference data versions and images  (jobs/sync.py)
  mtg update      — update to the latest MTGJSON release (jobs/sync.py)
  mtg build-index — download images, rebuild pHash index (jobs/sync.py)
"""

import click

from mtg_recognition.engine.pipeline import main as _recognize_main
from mtg_recognition.engine.segmentation.cli import main as _segment_main
from mtg_recognition.jobs.sync import (
    build_index_command,
    status_command,
    update_command,
)


@click.group()
def cli() -> None:
    """MTG card recognition toolkit."""
    pass


cli.add_command(status_command, name="status")
cli.add_command(update_command, name="update")
cli.add_command(build_index_command, name="build-index")
cli.add_command(_segment_main, name="segment")
cli.add_command(_recognize_main, name="recognize")


if __name__ == "__main__":
    cli()
