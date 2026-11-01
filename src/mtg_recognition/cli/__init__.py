"""The `mtg` command group, and the only package that imports Click.

mtg recognize   — full recognition pipeline            (cli/recognize.py)
mtg segment     — card segmentation                    (cli/segment.py)
mtg status      — reference data versions and images   (cli/reference.py)
mtg update      — update to the latest MTGJSON release (cli/reference.py)
mtg build-index — download images, rebuild pHash index (cli/reference.py)
"""

import click

from mtg_recognition.cli import recognize, reference, segment


@click.group()
def cli() -> None:
    """MTG card recognition toolkit."""


cli.add_command(reference.status)
cli.add_command(reference.update)
cli.add_command(reference.build_index)
cli.add_command(segment.segment)
cli.add_command(recognize.recognize)
