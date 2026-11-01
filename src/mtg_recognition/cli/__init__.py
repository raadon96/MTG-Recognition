"""The `mtg` command group, and the only package that imports Click."""

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
