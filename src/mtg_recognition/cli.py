"""Top-level `mtg` CLI group.

Wraps the existing recognition and segmentation pipelines as Click subcommands
so collaborators can use `mtg recognize` and `mtg segment` instead of the long
`uv run python -m mtg_recognition.engine.*` invocations.

Later plans attach `build-index` and `sync` to this same group via add_command().
"""

import click
from rich.console import Console

from mtg_recognition.engine.pipeline import main as _recognize_main
from mtg_recognition.engine.segmentation.cli import main as _segment_main

# Module-level Console instance — unused in this plan but stabilises the import
# surface for plans 02 and 03 (build-index, sync step headers and summary panel).
console = Console()


@click.group()
def cli() -> None:
    """MTG card recognition toolkit."""
    pass


# Register existing commands directly — zero code duplication (D-11).
# The name= argument gives the user `mtg recognize` and `mtg segment` regardless
# of what the underlying function is named inside its own module.
cli.add_command(_recognize_main, name="recognize")
cli.add_command(_segment_main, name="segment")


if __name__ == "__main__":
    cli()
