"""Top-level `mtg` CLI group.

Wraps the existing recognition and segmentation pipelines as Click subcommands
so collaborators can use `mtg recognize` and `mtg segment` instead of the long
`uv run python -m mtg_recognition.engine.*` invocations.

Later plans attach `sync` to this same group via add_command().
"""

from multiprocessing import cpu_count

import click
from rich.console import Console

from mtg_recognition.engine.matching.phash_matcher import PHashCardMatcher
from mtg_recognition.engine.pipeline import main as _recognize_main
from mtg_recognition.engine.segmentation.cli import main as _segment_main
from mtg_recognition.settings import settings

# Module-level Console instance — used by build-index and future sync step headers
# and summary panel.
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


@cli.command("build-index")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    show_default=True,
    help="Rebuild pHash index even if it already exists.",
)
@click.option(
    "--workers",
    type=click.INT,
    default=cpu_count() - 1,
    show_default=True,
    help="Number of parallel worker processes for hash computation.",
)
def build_index(force: bool, workers: int) -> None:
    """Rebuild the pHash index from reference images already on disk.

    Requires data/reference/images/ to be populated first (run 'mtg sync'
    to download reference images if you haven't already).
    """
    # Preflight check (D-10): image directory must exist and be non-empty.
    img_dir = settings.IMG_DIR
    if not img_dir.exists() or not any(img_dir.iterdir()):
        click.echo(
            "Error: data/reference/images/ is missing or empty. Run 'mtg sync' first.",
            err=True,
        )
        raise SystemExit(1)

    # Instantiate with auto_build=False so the preflight branch is the only path
    # that initiates a build — avoids a double-build when force=False and files exist.
    matcher = PHashCardMatcher(auto_build=False)
    matcher.build_index(force=force, num_workers=workers)


if __name__ == "__main__":
    cli()
