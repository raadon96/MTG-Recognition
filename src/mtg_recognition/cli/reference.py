"""`mtg status` and `mtg update`: run the reference data jobs and print their
results.
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


@click.command(name="status")
def status() -> None:
    """Show the catalog's MTGJSON release, the printings without a pHash, the
    images on disk and the latest MTGJSON release, then the command to run, if any.
    """
    reference = ReferenceData.default()
    with _exit_on_error(), sync.make_client() as client:
        report = sync.status(reference, client)

    lines, next_step = status_summary(report)
    console.print(Panel("\n".join(lines), title="Reference data", expand=False))
    console.print(next_step)


@click.command(name="update")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Rebuild even when the catalog already is the latest release, downloading "
    "every image again and recomputing every pHash.",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=DEFAULT_NUM_WORKERS,
    show_default=True,
    help="Processes hashing images (1 hashes in-process).",
)
def update(force: bool, workers: int) -> None:
    """Build the reference data from the latest MTGJSON release: the catalog,
    then the missing reference images, then the pHashes of the printings that
    don't have one yet. Does nothing when the catalog already is that release.
    """
    reference = ReferenceData.default()
    with _exit_on_error(), sync.make_client() as client:
        result = sync.update(reference, client, force=force, workers=workers)

    lines = update_summary(reference, result)
    console.print(Panel("\n".join(lines), title="Update Summary", expand=False))


@contextmanager
def _exit_on_error() -> Generator[None]:
    try:
        yield
    except Exception as exc:
        click.echo(f"Error: {type(exc).__name__}: {exc}", err=True)
        raise SystemExit(1) from exc


def status_summary(report: sync.Status) -> tuple[list[str], str]:
    """The lines of the panel, and the line after it naming the command to run."""
    lines = [
        f"Catalog: {report.catalog_version or 'not built by `mtg update`'}",
        f"Printings without a pHash: {report.without_phash}",
        f"AllPrintings: {report.allprintings_version or 'absent'}",
        f"Reference images: {report.images_present} present, "
        f"{report.images_missing} missing",
        f"Latest MTGJSON release: {report.latest_version or 'unknown'}",
    ]
    if report.next_command is None:
        return lines, "Up to date: nothing to run."
    return lines, f"Run: {report.next_command}"


def update_summary(reference: ReferenceData, result: sync.UpdateResult) -> list[str]:
    report = result.build
    if report is None:
        return [f"Catalog: up to date ({result.catalog_version})"]
    source = "downloaded" if report.allprintings_downloaded else "reused"
    lines = [
        f"Catalog: rebuilt ({report.version})",
        f"AllPrintings: {source} ({report.version})",
        f"Reference images: {report.downloaded} downloaded, "
        f"{len(report.failed)} failed",
        f"Removed images (no printing): {report.removed}",
        f"pHashes: {report.reused} reused, {report.hashed} computed, "
        f"{report.without_phash} printings without one",
    ]
    if report.unhashable:
        lines.append(
            f"Unreadable images deleted: {len(report.unhashable)}; "
            "the next build downloads them again"
        )
    if report.failed or report.unhashable:
        lines.append(f"Image failures written to {reference.log_file}")
    return lines
