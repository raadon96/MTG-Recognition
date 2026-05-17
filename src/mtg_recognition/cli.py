"""Top-level `mtg` CLI group.

Registers all subcommands via Click's add_command so each domain module owns
its own Click command definition:

  mtg recognize  — full recognition pipeline  (engine/pipeline.py)
  mtg segment    — card segmentation          (engine/segmentation/cli.py)
  mtg build-index — pHash index rebuild       (engine/matching/phash_matcher.py)
  mtg sync        — full reference data sync  (jobs/sync.py)
"""

import click

from mtg_recognition.engine.matching.phash_matcher import (
    build_index as _build_index_main,
)
from mtg_recognition.engine.pipeline import main as _recognize_main
from mtg_recognition.engine.segmentation.cli import main as _segment_main
from mtg_recognition.jobs.sync import main as _sync_main


@click.group()
def cli() -> None:
    """MTG card recognition toolkit."""
    pass


cli.add_command(_sync_main, name="sync")
cli.add_command(_build_index_main, name="build-index")
cli.add_command(_segment_main, name="segment")
cli.add_command(_recognize_main, name="recognize")


if __name__ == "__main__":
    cli()
