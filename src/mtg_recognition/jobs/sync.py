"""Reference data jobs: `mtg status`, `mtg update` and `mtg build-index`.

The MobileCache database and the pHash index are built from one MTGJSON release
and committed; the reference images are a local cache, downloaded only to build
the index. Every HTTP request goes through the `httpx.Client` passed to the step
functions, which the Click commands build.
"""

import io
import os
import shutil
import time
import zipfile
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import click
import httpx
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from sqlalchemy import func
from sqlmodel import Session, SQLModel, col, create_engine, select
from tqdm import tqdm

from mtg_recognition.engine.matching.phash import DEFAULT_NUM_WORKERS, build_phash_index
from mtg_recognition.reference import ReferenceData, SyncState, open_read_only
from mtg_recognition.settings import settings

from .schemas import CardIdentifiers, Cards, Meta, MobileCache, Sets

SCRYFALL_HEADERS = {
    "User-Agent": "mtg-recognition/1.0 (contact: adnen.boumessouer@gmail.com)",
}

# MTGJSON Meta endpoint for a lightweight version check.
META_URL = "https://mtgjson.com/api/v5/Meta.json"

IMAGE_DOWNLOAD_WORKERS = 25
IMAGE_DOWNLOAD_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0

console = Console()


def make_client() -> httpx.Client:
    """The HTTP client the commands use for MTGJSON and Scryfall."""
    return httpx.Client(
        headers=SCRYFALL_HEADERS,
        timeout=httpx.Timeout(30.0),
        follow_redirects=True,
    )


@contextmanager
def _replacing(path: Path) -> Generator[Path]:
    """Yield a temporary path next to `path`, moved over `path` when the block
    succeeds and deleted when it fails, so `path` is never left half-written.
    """
    part_path = path.with_name(path.name + ".part")
    part_path.unlink(missing_ok=True)
    try:
        yield part_path
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise
    os.replace(part_path, path)


# ---------------------------------------------------------------------------
# MobileCache database
# ---------------------------------------------------------------------------


def generate_image_uri(
    scryfallId,
    fileFace: str = "front",
    fileType: str = "normal",
    fileFormat: str = "jpg",
) -> str:
    # See https://mtgjson.com/faq/#scryfall-imagery
    return f"https://cards.scryfall.io/{fileType}/{fileFace}/{scryfallId[0]}/{scryfallId[1]}/{scryfallId}.{fileFormat}"


def extract_year(releaseDate: str) -> str | None:
    if not releaseDate:
        return None
    return releaseDate.split("-")[0]


def build_database(allprintings_path: Path, db_path: Path) -> int:
    """(Re)build the MobileCache database at `db_path` from AllPrintings, streaming
    the rows; an interrupted build leaves the old one in place. Returns the number
    of printings.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _replacing(db_path) as part_path:
        source_engine = open_read_only(allprintings_path)
        dest_engine = create_engine(f"sqlite:///{part_path}")
        try:
            # SQLModel sets __table__ on table models at runtime but doesn't
            # declare it on the class.
            SQLModel.metadata.create_all(
                dest_engine,
                tables=[MobileCache.__table__],  # ty: ignore[unresolved-attribute]
            )
            with (
                Session(source_engine) as source_session,
                Session(dest_engine) as dest_session,
            ):
                # sqlmodel's select overloads stop at 4 columns; this takes 13.
                statement = (
                    select(  # ty: ignore[no-matching-overload]
                        CardIdentifiers.uuid,
                        Cards.name,
                        Cards.manaCost,
                        Cards.type,
                        Cards.number,
                        Sets.baseSetSize,
                        Cards.setCode,
                        Cards.artist,
                        Cards.power,
                        Cards.toughness,
                        Cards.borderColor,
                        CardIdentifiers.scryfallId,
                        Sets.releaseDate,
                    )
                    .join(CardIdentifiers, Cards.uuid == CardIdentifiers.uuid)
                    .join(Sets, Cards.setCode == Sets.code)
                )

                # Stream results (fetch only 1000 rows at a time, crucial for memory)
                results_stream = source_session.exec(statement).yield_per(1000)

                batch = []
                batch_size = 5000
                total_migrated = 0

                for row in tqdm(results_stream, desc="Processing Cards", unit=" card"):
                    batch.append(
                        MobileCache(
                            uuid=row.uuid,
                            name=row.name,
                            manaCost=row.manaCost,
                            type=row.type,
                            number=row.number,
                            baseSetSize=row.baseSetSize,
                            setCode=row.setCode,
                            url=(
                                generate_image_uri(row.scryfallId)
                                if row.scryfallId
                                else None
                            ),
                            artist=row.artist,
                            power=row.power,
                            toughness=row.toughness,
                            borderColor=row.borderColor,
                            setReleaseYear=extract_year(row.releaseDate),
                        )
                    )

                    if len(batch) >= batch_size:
                        dest_session.add_all(batch)
                        dest_session.commit()
                        total_migrated += len(batch)
                        batch = []

                if batch:
                    dest_session.add_all(batch)
                    dest_session.commit()
                    total_migrated += len(batch)
        finally:
            source_engine.dispose()
            dest_engine.dispose()
    return total_migrated


def _count_printings(reference: ReferenceData, with_url: bool) -> int:
    """Number of printings in the database with (or without) an image URL."""
    url = col(MobileCache.url)
    engine = open_read_only(reference.db_path)
    try:
        statement = select(func.count()).where(
            url.is_not(None) if with_url else url.is_(None)
        )
        with Session(engine) as session:
            return session.exec(statement).one()
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# MTGJSON release and AllPrintings
# ---------------------------------------------------------------------------


def fetch_latest_version(client: httpx.Client) -> str:
    """The latest MTGJSON release, from `Meta.json`."""
    response = client.get(META_URL)
    response.raise_for_status()
    return response.json()["meta"]["version"]


def allprintings_version(reference: ReferenceData) -> str | None:
    """The release of the AllPrintings on disk (its `meta` table), or `None` when
    it is absent.
    """
    if not reference.allprintings_path.exists():
        return None
    engine = open_read_only(reference.allprintings_path)
    try:
        with Session(engine) as session:
            return session.exec(select(col(Meta.version))).first()
    finally:
        engine.dispose()


def ensure_allprintings(
    reference: ReferenceData, client: httpx.Client, version: str
) -> bool:
    """Make the AllPrintings on disk the release `version`: download the ZIP,
    extract it over the old file and delete it, unless the file on disk already is
    that release. Returns whether it was downloaded.
    """
    if allprintings_version(reference) == version:
        return False

    zip_path = reference.allprintings_zip_path
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _download_file(client, settings.MTGJSON_URL, zip_path)
        _extract_allprintings(zip_path, reference.allprintings_path)
    finally:
        zip_path.unlink(missing_ok=True)
    return True


def _download_file(client: httpx.Client, url: str, path: Path) -> None:
    console.print(f"Downloading {url}...")
    with client.stream("GET", url) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) or None
        with (
            open(path, "wb") as out_file,
            tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as pbar,
        ):
            for chunk in response.iter_bytes():
                out_file.write(chunk)
                pbar.update(len(chunk))


def _extract_allprintings(zip_path: Path, allprintings_path: Path) -> None:
    console.print("Extracting...")
    with (
        _replacing(allprintings_path) as part_path,
        zipfile.ZipFile(zip_path) as archive,
        archive.open(allprintings_path.name) as source,
        open(part_path, "wb") as out_file,
    ):
        shutil.copyfileobj(source, out_file)


# ---------------------------------------------------------------------------
# Reference images
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageFailure:
    """A reference image that failed to download, or to hash during the build."""

    uuid: str
    url: str | None
    step: Literal["download", "hash"]
    error: str


@dataclass(frozen=True)
class DownloadReport:
    """Outcome of downloading the missing reference images."""

    downloaded: int
    failed: tuple[ImageFailure, ...]
    missing_url: int  # printings with no image URL, so nothing to download
    removed: int  # images deleted because they are not a printing in the database


def download_images(reference: ReferenceData, client: httpx.Client) -> DownloadReport:
    """Delete the images that are not a printing in the database, then download
    the missing ones. Failures don't stop the download; they are reported.
    """
    stray = reference.stray_images()
    for path in stray:
        path.unlink()

    missing = reference.missing_images()
    reference.images_dir.mkdir(parents=True, exist_ok=True)
    failed: list[ImageFailure] = []

    with ThreadPoolExecutor(max_workers=IMAGE_DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(_download_image, client, url, reference.image_path(uuid)): (
                uuid,
                url,
            )
            for uuid, url in missing
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading Images",
            unit="img",
        ):
            error = future.result()
            if error is not None:
                uuid, url = futures[future]
                failed.append(ImageFailure(uuid, url, "download", error))

    return DownloadReport(
        downloaded=len(missing) - len(failed),
        failed=tuple(failed),
        missing_url=_count_printings(reference, with_url=False),
        removed=len(stray),
    )


def _download_image(client: httpx.Client, url: str, path: Path) -> str | None:
    """Download `url` to `path`; returns the error, or `None` on success.

    Timeouts, connection errors, 429 and 5xx are retried; other HTTP errors
    (404 above all) and a body that doesn't decode as an image are not.
    """
    error = ""
    for attempt in range(IMAGE_DOWNLOAD_ATTEMPTS):
        if attempt:
            time.sleep(RETRY_DELAY_SECONDS)
        try:
            response = client.get(url)
        except httpx.TransportError as exc:
            error = f"{type(exc).__name__}: {exc}"
            continue
        if response.status_code == 200:
            error = _decode_error(response.content)
            if error is None:
                with _replacing(path) as part_path:
                    part_path.write_bytes(response.content)
            return error
        error = f"HTTP {response.status_code}"
        if response.status_code != 429 and response.status_code < 500:
            return error
    return error


def _decode_error(content: bytes) -> str | None:
    """Why `content` doesn't decode as an image, or `None` when it does."""
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
    except Exception as exc:  # PIL raises OSError, SyntaxError, ValueError...
        return f"Undecodable image: {type(exc).__name__}: {exc}"
    return None


def hash_images(reference: ReferenceData, workers: int) -> tuple[ImageFailure, ...]:
    """Rebuild the pHash index from the printings in the database that have an
    image on disk. Images that fail to hash are deleted, so the next build
    downloads them again, and reported.
    """
    urls = dict(reference.present_images())
    errors = build_phash_index(
        {uuid: reference.image_path(uuid) for uuid in urls}, reference, workers=workers
    )
    for uuid in errors:
        reference.image_path(uuid).unlink(missing_ok=True)
    return tuple(
        ImageFailure(uuid, urls[uuid], "hash", error) for uuid, error in errors.items()
    )


def _write_image_log(reference: ReferenceData, failed: list[ImageFailure]) -> None:
    """Write the image failures of this build to `reference.log_file`, or remove
    the log of an earlier one when there are none.
    """
    if not failed:
        reference.log_file.unlink(missing_ok=True)
        return
    reference.log_file.write_text(
        "".join(f"{f.uuid}\t{f.url or ''}\t{f.step}\t{f.error}\n" for f in failed)
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexResult:
    phash_version: str | None
    downloads: DownloadReport | None  # None when the index was already up to date
    unhashable: tuple[ImageFailure, ...] = ()  # deleted, downloaded again next build

    @property
    def built(self) -> bool:
        return self.downloads is not None


@dataclass(frozen=True)
class UpdateResult:
    latest_version: str
    database_version: str | None
    database_rebuilt: bool
    allprintings_downloaded: bool
    index: IndexResult


@dataclass(frozen=True)
class Status:
    database_version: str | None
    phash_version: str | None
    index_matches: bool  # the index is on disk and built from the database's release
    allprintings_version: str | None  # None when AllPrintings is absent
    images_present: int
    images_missing: int
    latest_version: str | None  # None when MTGJSON can't be reached

    @property
    def next_command(self) -> str | None:
        """The command that brings the reference data up to date, if any."""
        if self.latest_version is not None and (
            self.latest_version != self.database_version
        ):
            return "mtg update"
        if not self.index_matches:
            return "mtg build-index"
        return None


def _index_matches(reference: ReferenceData, state: SyncState) -> bool:
    return (
        state.phash_version is not None
        and state.phash_version == state.mobilecache_version
        and all(path.exists() for path in reference.phash_files)
    )


def build_index(
    reference: ReferenceData,
    client: httpx.Client,
    force: bool = False,
    workers: int = DEFAULT_NUM_WORKERS,
) -> IndexResult:
    """Bring the reference images in line with the database (delete the strays,
    download the missing), then rebuild the pHash index from the database's
    printings with an image on disk and record it as built from the database's
    release. Without `force`, does nothing when the index already is. Image
    failures are written to `reference.log_file`.
    """
    state = reference.load_state()
    if not force and _index_matches(reference, state):
        return IndexResult(state.phash_version, downloads=None)

    state = replace(state, phash_version=None)
    reference.save_state(state)

    failed: list[ImageFailure] = []
    try:
        console.print("\n[bold]--- Reference images ---[/bold]")
        downloads = download_images(reference, client)
        failed += downloads.failed

        console.print("\n[bold]--- pHash index ---[/bold]")
        unhashable = hash_images(reference, workers)
        failed += unhashable
    finally:
        _write_image_log(reference, failed)

    state = replace(state, phash_version=state.mobilecache_version)
    reference.save_state(state)
    return IndexResult(state.phash_version, downloads, unhashable)


def update(
    reference: ReferenceData,
    client: httpx.Client,
    force: bool = False,
    workers: int = DEFAULT_NUM_WORKERS,
) -> UpdateResult:
    """Bring the reference data to the latest MTGJSON release: rebuild the database
    when its release differs (or `force`), then the index when it no longer
    matches the database. State is saved after each step, so a re-run after an
    interruption resumes.
    """
    latest = fetch_latest_version(client)
    state = reference.load_state()
    rebuilt = downloaded = False

    if force or state.mobilecache_version != latest or not reference.db_path.exists():
        console.print("\n[bold]--- MobileCache ---[/bold]")
        state = replace(state, mobilecache_version=None)
        reference.save_state(state)

        downloaded = ensure_allprintings(reference, client, latest)
        console.print("Building Local Database...")
        build_database(reference.allprintings_path, reference.db_path)

        # The release actually built, which may lag behind Meta.json
        state = replace(state, mobilecache_version=allprintings_version(reference))
        reference.save_state(state)
        rebuilt = True

    index = build_index(reference, client, force=force, workers=workers)
    return UpdateResult(
        latest_version=latest,
        database_version=state.mobilecache_version,
        database_rebuilt=rebuilt,
        allprintings_downloaded=downloaded,
        index=index,
    )


def status(reference: ReferenceData, client: httpx.Client) -> Status:
    """Read-only report on the reference data; the latest release is `None` when
    MTGJSON can't be reached.
    """
    state = reference.load_state()
    try:
        latest = fetch_latest_version(client)
    except (httpx.HTTPError, ValueError, KeyError):  # offline or a malformed reply
        latest = None
    missing = len(reference.missing_images())
    return Status(
        database_version=state.mobilecache_version,
        phash_version=state.phash_version,
        index_matches=_index_matches(reference, state),
        allprintings_version=allprintings_version(reference),
        images_present=_count_printings(reference, with_url=True) - missing,
        images_missing=missing,
        latest_version=latest,
    )


# ---------------------------------------------------------------------------
# Click commands
# ---------------------------------------------------------------------------


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
def status_command() -> None:
    """Show the reference data versions, the images on disk and the latest MTGJSON
    release, then the command to run, if any.
    """
    reference = ReferenceData.default()
    with _exit_on_error(), make_client() as client:
        report = status(reference, client)

    lines, next_step = status_summary(report)
    console.print(Panel("\n".join(lines), title="Reference data", expand=False))
    console.print(next_step)


@click.command(name="update")
@_force_option
@_workers_option
def update_command(force: bool, workers: int) -> None:
    """Update the database to the latest MTGJSON release, then rebuild the pHash
    index if it no longer matches the database.
    """
    reference = ReferenceData.default()
    with _exit_on_error(), make_client() as client:
        result = update(reference, client, force=force, workers=workers)

    lines = update_summary(reference, result)
    console.print(Panel("\n".join(lines), title="Update Summary", expand=False))


@click.command(name="build-index")
@_force_option
@_workers_option
def build_index_command(force: bool, workers: int) -> None:
    """Delete the reference images that are not a printing, download the missing
    ones, then rebuild the pHash index from the database's printings with an
    image on disk.
    """
    reference = ReferenceData.default()
    with _exit_on_error(), make_client() as client:
        result = build_index(reference, client, force=force, workers=workers)

    lines = index_summary(reference, result)
    console.print(Panel("\n".join(lines), title="Build-index Summary", expand=False))


def status_summary(report: Status) -> tuple[list[str], str]:
    """The lines of the panel, and the line after it naming the command to run."""
    match = "matches" if report.index_matches else "does not match"
    lines = [
        f"MobileCache: {report.database_version}",
        f"pHash index: {report.phash_version} ({match} the database)",
        f"AllPrintings: {report.allprintings_version or 'absent'}",
        f"Reference images: {report.images_present} present, "
        f"{report.images_missing} missing",
        f"Latest MTGJSON release: {report.latest_version or 'unknown'}",
    ]
    if report.next_command is None:
        return lines, "Up to date: nothing to run."
    return lines, f"Run: {report.next_command}"


def update_summary(reference: ReferenceData, result: UpdateResult) -> list[str]:
    database = "rebuilt" if result.database_rebuilt else "up to date"
    lines = [f"MobileCache: {database} ({result.database_version})"]
    if result.database_rebuilt:
        source = "downloaded" if result.allprintings_downloaded else "reused"
        lines.append(f"AllPrintings: {source} ({result.database_version})")
    return lines + index_summary(reference, result.index)


def index_summary(reference: ReferenceData, index: IndexResult) -> list[str]:
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
