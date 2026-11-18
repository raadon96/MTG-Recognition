"""`mtg reference status`, `mtg reference update`, `mtg reference build` and
`mtg reference download-images`: run the reference data jobs and print their
results.
"""

from collections.abc import Generator
from contextlib import contextmanager

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mtg_recognition.engine.matching.phash import DEFAULT_NUM_WORKERS
from mtg_recognition.jobs import sync
from mtg_recognition.reference.layout import ReferenceData

console = Console()


@click.group(name="reference")
def reference() -> None:
    """Check, build and download the reference data."""


@reference.command(name="status")
def status() -> None:
    """Show the catalog's MTGJSON release, the printings without a pHash, the
    images on disk and the latest MTGJSON release, then whether the catalog is
    the latest release.

    Ends with one verdict: up to date, outdated (with the command to run), or
    unknown when MTGJSON can't be reached.
    """
    reference = ReferenceData.default()
    with _exit_on_error(), sync.make_client() as client:
        report = sync.status(reference, client)

    lines, verdict = status_summary(report)
    console.print(Panel("\n".join(lines), title="Reference data", expand=False))
    console.print(verdict)


workers_option = click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=DEFAULT_NUM_WORKERS,
    show_default=True,
    help="Processes hashing images (1 hashes in-process).",
)


@reference.command(name="update")
@workers_option
def update(workers: int) -> None:
    """Build the reference data from the latest MTGJSON release: the catalog,
    then the missing reference images, then the pHashes of the printings that
    don't have one yet. Does nothing when the catalog already is that release.

    AllPrintings is kept in data/reference/mtgjson/. A printing keeps its pHash
    from the previous catalog when its image URL is unchanged; only the other
    printings are hashed, and only their reference images missing from disk are
    downloaded, to data/reference/images/. Hashing every printing, with no pHash
    to reuse, downloads about 116k reference images, 12 GB.

    Reference images that fail to download or hash are written to
    data/reference/image_errors.log, which is removed when a run has none.
    """
    reference = ReferenceData.default()
    with _exit_on_error(), sync.make_client() as client:
        result = sync.update(reference, client, workers=workers)

    lines = update_summary(reference, result)
    console.print(Panel("\n".join(lines), title="Update Summary", expand=False))


@reference.command(name="build")
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Download AllPrintings and every image again and recompute every pHash.",
)
@workers_option
def build(no_cache: bool, workers: int) -> None:
    """Build the reference data again from the catalog's own MTGJSON release, for
    when the code that builds the catalog changed. Reuses the AllPrintings on
    disk without reaching MTGJSON, and the pHash of each printing whose image URL
    is unchanged. Never changes which release the catalog is.

    With --no-cache, or when the AllPrintings on disk is another release or
    absent, AllPrintings is downloaded, which works only while the catalog's
    release is MTGJSON's latest: MTGJSON publishes no older release. Otherwise
    the command fails, naming `mtg reference update`, and changes nothing.
    Reference images that fail to download or hash are written to
    data/reference/image_errors.log, which is removed when a run has none.

    Example: Scryfall replaced a preview image under the same URL, so the reused
    pHash is out of date:

    \b
    uv run mtg reference build --no-cache
    """
    reference = ReferenceData.default()
    with _exit_on_error(), sync.make_client() as client:
        report = sync.build(reference, client, no_cache=no_cache, workers=workers)

    lines = build_summary(reference, report)
    console.print(Panel("\n".join(lines), title="Build Summary", expand=False))


@reference.command(name="download-images")
def download_images() -> None:
    """Download the reference images missing from disk for the printings in the
    committed catalog. Changes nothing else: the catalog, its pHashes and
    AllPrintings stay as they are.

    Neither recognition nor the tests need the reference images, only the
    pHashes in the catalog; download them to look at them, for example to
    compare a card image with its candidates' reference images while debugging
    matching.

    The images go to data/reference/images/: about 116k images, 12 GB, on a first
    run; a second run downloads only what is still missing. Failed downloads are
    written to data/reference/download_errors.log, which is removed when a run
    has none; data/reference/image_errors.log, the log of `mtg reference update`
    and `mtg reference build`, stays as it is.
    """
    reference = ReferenceData.default()
    with _exit_on_error(), sync.make_client() as client:
        report = sync.download_missing_images(reference, client)

    lines = download_images_summary(reference, report)
    console.print(Panel("\n".join(lines), title="Download Summary", expand=False))


@contextmanager
def _exit_on_error() -> Generator[None]:
    try:
        yield
    except Exception as exc:
        click.echo(f"Error: {type(exc).__name__}: {exc}", err=True)
        raise SystemExit(1) from exc


def status_summary(report: sync.Status) -> tuple[list[str], Text]:
    """The lines of the panel, and the verdict printed after it."""
    lines = [
        f"Catalog: {report.catalog_version or "wasn't built"}",
        f"Printings without a pHash: {report.without_phash}",
        f"AllPrintings: {report.allprintings_version or 'absent'}",
        f"Reference images: {report.images_present} present, "
        f"{report.images_missing} missing",
        f"Latest MTGJSON release: {report.latest_version or 'unknown'}",
    ]
    if report.next_command is not None:
        return lines, Text(f"✗ Outdated: run {report.next_command}", style="red")
    # offline, a catalog with a version may still be outdated
    if report.latest_version is None:
        return lines, Text("? Latest release unknown (offline)", style="yellow")
    return lines, Text("✓ Up to date", style="green")


def update_summary(reference: ReferenceData, result: sync.UpdateResult) -> list[str]:
    if result.build is None:
        return [f"Catalog: up to date ({result.catalog_version})"]
    return build_summary(reference, result.build)


def build_summary(reference: ReferenceData, report: sync.BuildReport) -> list[str]:
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


def download_images_summary(
    reference: ReferenceData, report: sync.DownloadReport
) -> list[str]:
    lines = [
        f"Reference images: {report.downloaded} downloaded, {len(report.failed)} failed"
    ]
    if report.failed:
        lines.append(f"Image failures written to {reference.download_log_file}")
    return lines
