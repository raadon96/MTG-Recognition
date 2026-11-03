"""Reference data jobs, run by `mtg status`, `mtg update` and
`mtg download-images`.

The catalog, which holds each printing's pHash and the MTGJSON release it was
built from (ADR 0006), is committed; the reference images are a local cache,
downloaded only to compute the pHashes. Every HTTP request goes through the
`httpx.Client` passed to the step functions, which the commands in
`cli.reference` build.
"""

import io
import os
import shutil
import time
import zipfile
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from PIL import Image
from rich.console import Console
from sqlalchemy import bindparam
from sqlmodel import Session, SQLModel, col, create_engine, select
from tqdm import tqdm

from mtg_recognition.engine.matching.phash import DEFAULT_NUM_WORKERS, compute_phashes
from mtg_recognition.reference.catalog import (
    CatalogMeta,
    CatalogPrinting,
    catalog_version,
    count_phashes,
    read_image_urls,
    read_phashes,
)
from mtg_recognition.reference.layout import ReferenceData
from mtg_recognition.reference.sqlite import open_read_only
from mtg_recognition.settings import settings

from .schemas import CardIdentifiers, Cards, Meta, Sets

SCRYFALL_HEADERS = {
    "User-Agent": "mtg-recognition/1.0 (contact: adnen.boumessouer@gmail.com)",
}

# MTGJSON Meta endpoint for a lightweight version check.
META_URL = "https://mtgjson.com/api/v5/Meta.json"

IMAGE_DOWNLOAD_WORKERS = 25
IMAGE_DOWNLOAD_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0

# Layouts with a physical back side, which Scryfall serves as the back image of
# the shared scryfallId. Other multi-face layouts (split, adventure, flip,
# aftermath) print both halves on the front, and a meld result has its own
# scryfallId, so their printings keep the front image.
DOUBLE_FACED_LAYOUTS = frozenset({"transform", "modal_dfc", "reversible_card"})

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
# Catalog
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


def build_catalog(allprintings_path: Path, catalog_path: Path) -> str:
    """Write the catalog at `catalog_path`, a new file, from AllPrintings,
    streaming the rows; every printing is written without a pHash. The `meta`
    table gets AllPrintings' release, which is returned.
    """
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    source_engine = open_read_only(allprintings_path)
    dest_engine = create_engine(f"sqlite:///{catalog_path}")
    try:
        # SQLModel sets __table__ on table models at runtime but doesn't
        # declare it on the class.
        SQLModel.metadata.create_all(
            dest_engine,
            tables=[
                CatalogPrinting.__table__,  # ty: ignore[unresolved-attribute]
                CatalogMeta.__table__,  # ty: ignore[unresolved-attribute]
            ],
        )
        with (
            Session(source_engine) as source_session,
            Session(dest_engine) as dest_session,
        ):
            # AllPrintings stores the date as text ("2026-11-02"), which the
            # schema's datetime column doesn't parse, so read it as text
            version, date = (
                source_session.connection()
                .exec_driver_sql("SELECT version, date FROM meta")
                .one()
            )
            dest_session.add(CatalogMeta(version=version, date=date))

            # sqlmodel's select overloads stop at 4 columns; this select takes more.
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
                    Cards.layout,
                    Cards.side,
                    Cards.faceName,
                )
                .join(CardIdentifiers, Cards.uuid == CardIdentifiers.uuid)
                .join(Sets, Cards.setCode == Sets.code)
            )

            # Stream results (fetch only 1000 rows at a time, crucial for memory)
            results_stream = source_session.exec(statement).yield_per(1000)

            batch = []
            batch_size = 5000

            for row in tqdm(results_stream, desc="Processing Cards", unit=" card"):
                # Each face of a double-faced card is a printing of its own,
                # with the face's own name and image (see GLOSSARY.md).
                double_faced = row.layout in DOUBLE_FACED_LAYOUTS
                face = "back" if double_faced and row.side == "b" else "front"
                batch.append(
                    CatalogPrinting(
                        uuid=row.uuid,
                        name=row.faceName if double_faced else row.name,
                        manaCost=row.manaCost,
                        type=row.type,
                        number=row.number,
                        baseSetSize=row.baseSetSize,
                        setCode=row.setCode,
                        url=(
                            generate_image_uri(row.scryfallId, face)
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
                    batch = []

            if batch:
                dest_session.add_all(batch)
                dest_session.commit()
    finally:
        source_engine.dispose()
        dest_engine.dispose()
    return version


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


def download_images(
    reference: ReferenceData, client: httpx.Client, urls: dict[str, str]
) -> tuple[ImageFailure, ...]:
    """Download the reference image of each printing in `urls` (UUID -> image
    URL), replacing any file already on disk. Failures don't stop the download;
    they are returned.
    """
    reference.images_dir.mkdir(parents=True, exist_ok=True)
    failed: list[ImageFailure] = []
    with ThreadPoolExecutor(max_workers=IMAGE_DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(_download_image, client, url, reference.image_path(uuid)): (
                uuid,
                url,
            )
            for uuid, url in urls.items()
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
    return tuple(failed)


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


def _write_image_log(log_file: Path, failed: list[ImageFailure]) -> None:
    """Write the image failures of this run to `log_file`, or remove the log of
    an earlier run when there are none.
    """
    if not failed:
        log_file.unlink(missing_ok=True)
        return
    log_file.write_text(
        "".join(f"{f.uuid}\t{f.url or ''}\t{f.step}\t{f.error}\n" for f in failed)
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildReport:
    """Outcome of building the reference data from one MTGJSON release."""

    version: str  # the release in AllPrintings, which may lag behind Meta.json
    allprintings_downloaded: bool
    removed: int  # images deleted because they are not a printing in the catalog
    downloaded: int
    failed: tuple[ImageFailure, ...]  # downloads that failed
    reused: int  # pHashes kept from the previous catalog
    hashed: int  # pHashes computed
    unhashable: tuple[ImageFailure, ...]  # deleted, downloaded at the next build
    without_phash: int  # printings never a candidate: no image URL, or a failure


@dataclass(frozen=True)
class DownloadReport:
    """Outcome of downloading the missing reference images of the catalog."""

    downloaded: int
    failed: tuple[ImageFailure, ...]


@dataclass(frozen=True)
class UpdateResult:
    latest_version: str
    catalog_version: str | None
    build: BuildReport | None  # None when the catalog already was the latest release


@dataclass(frozen=True)
class Status:
    catalog_version: str | None  # None when `mtg update` never built the catalog
    without_phash: int
    allprintings_version: str | None  # None when AllPrintings is absent
    images_present: int
    images_missing: int  # printings with an image URL but no image on disk
    latest_version: str | None  # None when MTGJSON can't be reached

    @property
    def next_command(self) -> str | None:
        """The command that brings the reference data up to date, if any."""
        if self.catalog_version is None or (
            self.latest_version is not None
            and self.latest_version != self.catalog_version
        ):
            return "mtg update"
        return None


def update(
    reference: ReferenceData,
    client: httpx.Client,
    force: bool = False,
    workers: int = DEFAULT_NUM_WORKERS,
) -> UpdateResult:
    """Build the reference data from the latest MTGJSON release (see `build`),
    unless the catalog already was built from it. `force` builds anyway.
    """
    latest = fetch_latest_version(client)
    built = catalog_version(reference.catalog_path)
    if not force and built == latest:
        return UpdateResult(latest, built, build=None)

    report = build(reference, client, latest, force=force, workers=workers)
    return UpdateResult(latest, report.version, report)


def build(
    reference: ReferenceData,
    client: httpx.Client,
    version: str,
    force: bool = False,
    workers: int = DEFAULT_NUM_WORKERS,
) -> BuildReport:
    """Build the reference data from the MTGJSON release `version`: AllPrintings,
    then the catalog, then the missing reference images, then the pHashes.

    The pHash of every printing already in the previous catalog is reused; only
    the other printings' images are downloaded (when not on disk) and hashed. A
    catalog without a version was never fully built, so nothing is reused from
    it. `force` downloads every image again and recomputes every pHash. The new
    catalog replaces the old one only once complete. Image failures are written
    to `reference.log_file`.

    Raises:
        ValueError: If no printing has a pHash; the previous catalog is kept
    """
    console.print("\n[bold]--- Catalog ---[/bold]")
    allprintings_downloaded = ensure_allprintings(reference, client, version)
    previous: dict[str, bytes] = {}
    if not force and catalog_version(reference.catalog_path) is not None:
        previous = dict(read_phashes(reference.catalog_path))

    failed: tuple[ImageFailure, ...] = ()
    unhashable: tuple[ImageFailure, ...] = ()
    try:
        with _replacing(reference.catalog_path) as part_path:
            console.print("Building the catalog...")
            built_version = build_catalog(reference.allprintings_path, part_path)
            urls = read_image_urls(part_path)
            # a printing that lost its image URL loses its pHash with it
            reused = {
                uuid: previous[uuid]
                for uuid, url in urls.items()
                if url is not None and uuid in previous
            }
            to_hash = {
                uuid: url
                for uuid, url in urls.items()
                if url is not None and uuid not in reused
            }

            # reference images: download those to hash
            console.print("\n[bold]--- Reference images ---[/bold]")
            to_download = to_hash if force else _missing_images(reference, to_hash)
            failed = download_images(reference, client, to_download)

            # pHashes; an image that fails to hash is deleted, so the next build
            # downloads it again
            console.print("\n[bold]--- pHashes ---[/bold]")
            on_disk = {
                uuid: reference.image_path(uuid)
                for uuid in to_hash
                if reference.image_path(uuid).exists()
            }
            phashes, errors = compute_phashes(on_disk, workers)
            for uuid in errors:
                reference.image_path(uuid).unlink()
            unhashable = tuple(
                ImageFailure(uuid, urls[uuid], "hash", error)
                for uuid, error in errors.items()
            )
            if not reused and not phashes:
                raise ValueError("No printing has a pHash; the catalog is unchanged.")
            _write_phashes(part_path, reused | phashes)
    finally:
        _write_image_log(reference.log_file, [*failed, *unhashable])

    # delete the images of no printing only now: until the new catalog replaced
    # the old one, the old one could still need them
    stray = [
        path for path in reference.images_dir.glob("*.jpg") if path.stem not in urls
    ]
    for path in stray:
        path.unlink()

    return BuildReport(
        version=built_version,
        allprintings_downloaded=allprintings_downloaded,
        removed=len(stray),
        downloaded=len(to_download) - len(failed),
        failed=failed,
        reused=len(reused),
        hashed=len(phashes),
        unhashable=unhashable,
        without_phash=len(urls) - len(reused) - len(phashes),
    )


def _missing_images(reference: ReferenceData, urls: dict[str, str]) -> dict[str, str]:
    """The printings in `urls` (UUID -> image URL) without an image on disk."""
    return {
        uuid: url
        for uuid, url in urls.items()
        if not reference.image_path(uuid).exists()
    }


def _write_phashes(catalog_path: Path, phashes: dict[str, bytes]) -> None:
    """Set the pHash of each printing in `phashes` (UUID -> packed pHash)."""
    engine = create_engine(f"sqlite:///{catalog_path}")
    try:
        # SQLModel sets __table__ on table models at runtime but doesn't
        # declare it on the class.
        table = CatalogPrinting.__table__  # ty: ignore[unresolved-attribute]
        statement = (
            table.update()
            .where(table.c.uuid == bindparam("printing"))
            .values(phash=bindparam("phash"))
        )
        with engine.begin() as connection:
            connection.execute(
                statement,
                [{"printing": uuid, "phash": phash} for uuid, phash in phashes.items()],
            )
    finally:
        engine.dispose()


def status(reference: ReferenceData, client: httpx.Client) -> Status:
    """Read-only report on the reference data; the latest release is `None` when
    MTGJSON can't be reached.
    """
    try:
        latest = fetch_latest_version(client)
    except (httpx.HTTPError, ValueError, KeyError):  # offline or a malformed reply
        latest = None
    version = catalog_version(reference.catalog_path)
    urls = read_image_urls(reference.catalog_path)
    with_url = {uuid: url for uuid, url in urls.items() if url is not None}
    missing = len(_missing_images(reference, with_url))
    # a catalog from before ADR 0006 has no phash column
    phashes = count_phashes(reference.catalog_path) if version is not None else 0
    return Status(
        catalog_version=version,
        without_phash=len(urls) - phashes,
        allprintings_version=allprintings_version(reference),
        images_present=len(with_url) - missing,
        images_missing=missing,
        latest_version=latest,
    )


def download_missing_images(
    reference: ReferenceData, client: httpx.Client
) -> DownloadReport:
    """Download the reference images missing from disk for the printings in the
    catalog. Changes nothing else in the reference data: no MTGJSON download, no
    catalog rebuild, no pHash. Failures are written to
    `reference.download_log_file`, so the log of the last build stays as it is.
    """
    urls = read_image_urls(reference.catalog_path)
    with_url = {uuid: url for uuid, url in urls.items() if url is not None}
    to_download = _missing_images(reference, with_url)
    failed = download_images(reference, client, to_download)
    _write_image_log(reference.download_log_file, list(failed))
    return DownloadReport(downloaded=len(to_download) - len(failed), failed=failed)
