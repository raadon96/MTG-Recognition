"""`mtg status`, `mtg update` and `mtg build-index`: run the reference data jobs
and print their results.
"""

from collections.abc import Generator
from contextlib import contextmanager

import click
from rich.console import Console
from rich.panel import Panel

from mtg_recognition.engine.matching.phash import DEFAULT_NUM_WORKERS
from mtg_recognition.jobs import sync
from mtg_recognition.reference.layout import ReferenceData

console = Console()


@contextmanager
def _exit_on_error() -> Generator[None]:
    try:
        yield
    except Exception as exc:
        click.echo(f"Error: {type(exc).__name__}: {exc}", err=True)
        raise SystemExit(1) from exc


_force_option = click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Rebuild even when the versions already match.",
)
_workers_option = click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=DEFAULT_NUM_WORKERS,
    show_default=True,
    help="Processes hashing images (1 hashes in-process).",
)


@click.command(name="status")
def status() -> None:
    """Show the reference data versions, the images on disk and the latest MTGJSON
    release, then the command to run, if any.
    """
    reference = ReferenceData.default()
    with _exit_on_error(), sync.make_client() as client:
        report = sync.status(reference, client)

    lines, next_step = status_summary(report)
    console.print(Panel("\n".join(lines), title="Reference data", expand=False))
    console.print(next_step)


@click.command(name="update")
@_force_option
@_workers_option
def update(force: bool, workers: int) -> None:
    """Update the catalog to the latest MTGJSON release, then rebuild the pHash
    index if it no longer matches the catalog.
    """
    reference = ReferenceData.default()
    with _exit_on_error(), sync.make_client() as client:
        result = sync.update(reference, client, force=force, workers=workers)

    lines = update_summary(reference, result)
    console.print(Panel("\n".join(lines), title="Update Summary", expand=False))


@click.command(name="build-index")
@_force_option
@_workers_option
def build_index(force: bool, workers: int) -> None:
    """Delete the reference images that are not a printing, download the missing
    ones, then rebuild the pHash index from the catalog's printings with an
    image on disk.
    """
    reference = ReferenceData.default()
    with _exit_on_error(), sync.make_client() as client:
        result = sync.build_index(reference, client, force=force, workers=workers)

    lines = index_summary(reference, result)
    console.print(Panel("\n".join(lines), title="Build-index Summary", expand=False))


def status_summary(report: sync.Status) -> tuple[list[str], str]:
    """The lines of the panel, and the line after it naming the command to run."""
    match = "matches" if report.index_matches else "does not match"
    lines = [
        f"Catalog: {report.catalog_version}",
        f"pHash index: {report.phash_version} ({match} the catalog)",
        f"AllPrintings: {report.allprintings_version or 'absent'}",
        f"Reference images: {report.images_present} present, "
        f"{report.images_missing} missing",
        f"Latest MTGJSON release: {report.latest_version or 'unknown'}",
    ]
    if report.next_command is None:
        return lines, "Up to date: nothing to run."
    return lines, f"Run: {report.next_command}"


def update_summary(reference: ReferenceData, result: sync.UpdateResult) -> list[str]:
    catalog = "rebuilt" if result.catalog_rebuilt else "up to date"
    lines = [f"Catalog: {catalog} ({result.catalog_version})"]
    if result.catalog_rebuilt:
        source = "downloaded" if result.allprintings_downloaded else "reused"
        lines.append(f"AllPrintings: {source} ({result.catalog_version})")
    return lines + index_summary(reference, result.index)


def index_summary(reference: ReferenceData, index: sync.IndexResult) -> list[str]:
    if index.downloads is None:
        return [f"pHash index: up to date ({index.phash_version})"]
    report = index.downloads
    lines = [
        f"Reference images: {report.downloaded} downloaded, "
        f"{len(report.failed)} failed, {report.missing_url} without URL",
        f"Removed images (no printing): {report.removed}",
        f"pHash index: rebuilt ({index.phash_version})",
    ]
    if index.unhashable:
        lines.append(
            f"Unreadable images deleted: {len(index.unhashable)}; "
            "run `mtg build-index` again"
        )
    if report.failed or index.unhashable:
        lines.append(f"Image failures written to {reference.log_file}")
    return lines
