"""Reference data synchronisation for MTG Recognition.

Manages the full reference data stack: MobileCache (SQLite card database),
reference images, and pHash index.  All three artifacts must stay in sync —
they are built from the same MTGJSON data release and consumed together
during card recognition.

This module can be executed as a standalone script:

    uv run python -m mtg_recognition.jobs.sync
"""

import asyncio
import json
import logging
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path
from urllib import request

import aiohttp
import click
from rich.console import Console
from rich.panel import Panel
from sqlmodel import Session, SQLModel, create_engine, select
from tqdm.asyncio import tqdm

from mtg_recognition.engine.matching.phash_matcher import PHashCardMatcher
from mtg_recognition.reference import ReferenceData, SyncState
from mtg_recognition.settings import settings

from .schemas import CardIdentifiers, Cards, MobileCache, Sets

SCRYFALL_HEADERS = {
    "User-Agent": "mtg-recognition/1.0 (contact: adnen.boumessouer@gmail.com)",
}

# MTGJSON Meta endpoint for a lightweight version check (D-04).
META_URL = "https://mtgjson.com/api/v5/Meta.json"

logger = logging.getLogger(__name__)

console = Console()


# ---------------------------------------------------------------------------
# Database helpers
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


def create_database(source_engine, dest_engine, db_path: Path, force: bool = False):
    """Migrate data from Source to Destination (the database at `db_path`) with
    streaming.
    """

    # 1. Reset Destination
    if db_path.exists():
        print(f"Local database already exists at {db_path}...")
        if not force:
            print("Skipping database creation...")
            return

        print(f"Removing existing database at {db_path}...")
        db_path.unlink()

    # Create tables (MobileCache is imported from models.py)
    SQLModel.metadata.create_all(dest_engine, tables=[MobileCache.__table__])

    print("Creating local database...")

    with Session(source_engine) as source_session, Session(dest_engine) as dest_session:
        # 2. Build Query
        statement = (
            select(
                CardIdentifiers.uuid,
                Cards.name,
                Cards.manaCost,
                Cards.type,
                Cards.number,
                Sets.baseSetSize,
                Cards.setCode,
                # Sets.name,
                Cards.artist,
                Cards.power,
                Cards.toughness,
                Cards.borderColor,
                CardIdentifiers.scryfallId,
                # Cards.text,
                # Cards.flavorText,
                # CardIdentifiers.mtgoId,
                Sets.releaseDate,
            )
            .join(CardIdentifiers, Cards.uuid == CardIdentifiers.uuid)
            .join(Sets, Cards.setCode == Sets.code)
        )

        # 3. Stream Results (fetch only 1000 rows at a time, crucial for memory)
        results_stream = source_session.exec(statement).yield_per(1000)

        batch = []
        batch_size = 5000
        total_migrated = 0

        # We iterate over the stream directly
        for row in tqdm(results_stream, desc="Processing Cards", unit=" card"):
            # Create object
            new_card = MobileCache(
                uuid=row.uuid,
                name=row.name,
                manaCost=row.manaCost,
                type=row.type,
                number=row.number,
                baseSetSize=row.baseSetSize,
                setCode=row.setCode,
                # setName=row.setName,
                url=generate_image_uri(row.scryfallId),
                artist=row.artist,
                power=row.power,
                toughness=row.toughness,
                borderColor=row.borderColor,
                setReleaseYear=extract_year(row.releaseDate),
                # text=row.text,
                # flavorText=row.flavorText,
                # mtgoId=row.mtgoId,
                # scryfallId=row.scryfallId,
            )
            batch.append(new_card)

            # Insert Batch
            if len(batch) >= batch_size:
                dest_session.add_all(batch)
                dest_session.commit()
                # Determine how many objects we just saved to update memory
                total_migrated += len(batch)
                batch = []  # Reset list

        # Final Batch
        if batch:
            dest_session.add_all(batch)
            dest_session.commit()
            total_migrated += len(batch)

    print(f"Database created at {db_path} ({total_migrated} cards)")


async def download_file_with_retry(session, url, path, retries=3):
    """Downloads a file with basic retry logic."""
    if not url:
        return

    path = Path(path)
    if path.exists():
        # skip download if file already exists
        return

    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    # Ensure dir exists (just in case)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(content)
                    return
                elif response.status == 404:
                    logger.error(f"404 Not Found: {url}")
                    # Don't retry 404s
                    return
                else:
                    # Log other HTTP errors
                    logger.warning(
                        f"HTTP {response.status}: {url} (Attempt {attempt + 1})"
                    )
        except TimeoutError:
            logger.warning(f"Timeout: {url} (Attempt {attempt + 1})")
        except aiohttp.ClientError as e:
            logger.warning(f"ClientError: {url} - {str(e)} (Attempt {attempt + 1})")
        except Exception as e:
            logger.error(f"Unexpected Error: {url} - {str(e)}")
            return  # Don't retry unexpected errors

        # Wait before retry, unless it's the last attempt
        if attempt < retries - 1:
            await asyncio.sleep(1)


async def image_worker(queue, session, pbar):
    """Worker that consumes URLs from the queue."""
    while True:
        # Get a "unit of work"
        item = await queue.get()

        # Determine strict termination condition
        if item is None:
            queue.task_done()
            break

        url, file_path = item

        await download_file_with_retry(session, url, file_path)

        pbar.update(1)
        queue.task_done()


async def download_manager(reference: ReferenceData, max_concurrent=20):
    """Manages the download queue and workers."""

    # 1. Fetch the UUID and URL of every printing whose image is not on disk
    cards_to_download = reference.missing_images()

    queue = asyncio.Queue(maxsize=1000)  # Buffer size
    total_files = len(cards_to_download)

    # 2. Setup HTTP Session
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(
        timeout=timeout, headers=SCRYFALL_HEADERS
    ) as session:
        # 3. Create Workers
        pbar = tqdm(total=total_files, desc="Downloading Images", unit="img")
        workers = [
            asyncio.create_task(image_worker(queue, session, pbar))
            for _ in range(max_concurrent)
        ]

        # 4. Fill Queue (Producer)
        for uuid, url in cards_to_download:
            await queue.put((url, reference.image_path(uuid)))

        # 5. Signal Completion
        for _ in range(max_concurrent):
            await queue.put(None)

        # 6. Wait for completion
        await queue.join()

        # Cancel workers (cleanup)
        for worker in workers:
            worker.cancel()

        pbar.close()


def download_raw_images(reference: ReferenceData):
    # Windows/Python 3.8+ specific event loop policy fix might be needed on some systems
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(
        download_manager(reference, max_concurrent=25)
    )  # TODO download webp images instead

    print("\nDone!")


def build_mobilecache(reference: ReferenceData) -> None:
    """Download AllPrintings ZIP and rebuild MobileCache."""
    reference.tmp_dir.mkdir(parents=True, exist_ok=True)
    zip_path = reference.allprintings_zip_path
    if not zip_path.exists():
        console.print(f"Downloading from {settings.MTGJSON_URL}...")
        req = request.Request(settings.MTGJSON_URL, headers=SCRYFALL_HEADERS)
        with (
            request.urlopen(req) as response,
            open(zip_path, "wb") as out_file,
        ):
            shutil.copyfileobj(response, out_file)

    console.print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(reference.tmp_dir)

    # Create cache db for memory efficient lookups during matching
    console.print("Building Local Database...")
    source_engine = create_engine(f"sqlite:///{reference.allprintings_path}")
    dest_engine = create_engine(f"sqlite:///{reference.db_path}")
    try:
        create_database(source_engine, dest_engine, reference.db_path, force=True)
    finally:
        source_engine.dispose()
        dest_engine.dispose()


def download_images(reference: ReferenceData) -> None:
    """Download the reference images missing from disk.

    Download errors are written to `reference.log_file` (overwritten on each run)
    through a file handler attached only while the download runs.
    """
    console.print("Downloading reference images...")
    reference.root.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(reference.log_file, mode="w")
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    try:
        download_raw_images(reference)
    finally:
        logger.removeHandler(handler)
        handler.close()


# ---------------------------------------------------------------------------
# Sync state helpers
# ---------------------------------------------------------------------------


def _fetch_mtgjson_version() -> str:
    """Fetch the current MTGJSON version via a lightweight META_URL GET.

    Uses SCRYFALL_HEADERS (D-04).
    Re-raises any exception so the sync orchestrator can exit non-zero (SYNC-04).
    """
    req = request.Request(META_URL, headers=SCRYFALL_HEADERS)
    with request.urlopen(req) as response:
        data = json.loads(response.read())
    return data["meta"]["version"]


# ---------------------------------------------------------------------------
# Rebuild predicates
# ---------------------------------------------------------------------------


def _needs_mobilecache_rebuild(
    reference: ReferenceData,
    state: SyncState,
    remote_version: str | None,
    force: bool,
) -> bool:
    if force:
        return True
    if remote_version is None:
        return False  # pinned mode (no --update-db): never rebuild the database
    return (
        state.mobilecache_version is None
        or state.mobilecache_version != remote_version
        or not reference.db_path.exists()
    )


def _needs_images_rebuild(
    reference: ReferenceData, state: SyncState, force: bool
) -> bool:
    return (
        force
        or state.images_version is None
        or state.images_version != state.mobilecache_version
        or not reference.images_dir.exists()
        or not any(reference.images_dir.iterdir())
    )


def _needs_phash_rebuild(
    reference: ReferenceData, state: SyncState, force: bool
) -> bool:
    return (
        force
        or state.phash_version is None
        or state.phash_version != state.images_version
        or not all(path.exists() for path in reference.phash_files)
    )


# ---------------------------------------------------------------------------
# Sync CLI command
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--force",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Bypass all version checks and rebuild MobileCache, images, and pHash index "
        "unconditionally."
    ),
)
@click.option(
    "--update-db",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Check for a new MTGJSON release and rebuild MobileCache, images, "
        "and pHash index if outdated."
    ),
)
def main(force: bool, update_db: bool) -> None:
    """Rebuild the full reference data stack in one shot.

    By default only downloads missing reference images (database is pinned to
    the committed version). Use --update-db to check for a new MTGJSON release
    and rebuild all artifacts if outdated. Use --force to bypass all version
    checks and rebuild unconditionally.
    """
    reference = ReferenceData.default()
    reference.root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, str] = {}

    try:
        remote_version = _fetch_mtgjson_version() if (update_db or force) else None
        state = reference.load_state()
        # ------------------------------------------------------------------
        # Step 1 — MobileCache (D-04, D-13)
        # ------------------------------------------------------------------
        console.print("\n[bold]--- Step 1: MobileCache ---[/bold]")

        if not _needs_mobilecache_rebuild(reference, state, remote_version, force):
            console.print(
                f"[green]✓ MobileCache is up-to-date ({state.mobilecache_version}) "
                "— skipping rebuild[/green]"
            )
            summary["MobileCache"] = f"skipped ({state.mobilecache_version})"
        else:
            # D-02: null before work begins
            state = replace(state, mobilecache_version=None)
            reference.save_state(state)

            build_mobilecache(reference)

            state = replace(state, mobilecache_version=remote_version)
            reference.save_state(state)
            summary["MobileCache"] = f"rebuilt ({remote_version})"

        # ------------------------------------------------------------------
        # Step 2 — Reference images (D-05, D-13)
        # ------------------------------------------------------------------
        console.print("\n[bold]--- Step 2: Reference Images ---[/bold]")

        if not _needs_images_rebuild(reference, state, force):
            console.print(
                f"[green]✓ Reference images are up-to-date ({state.images_version}) "
                "— skipping download[/green]"
            )
            summary["Reference images"] = f"skipped ({state.images_version})"
        else:
            # D-02: null before work begins
            state = replace(state, images_version=None)
            reference.save_state(state)

            download_images(reference)

            state = replace(state, images_version=state.mobilecache_version)
            reference.save_state(state)
            summary["Reference images"] = f"downloaded ({state.images_version})"

        # ------------------------------------------------------------------
        # Step 3 — pHash index (D-06, D-13, D-15)
        # ------------------------------------------------------------------
        console.print("\n[bold]--- Step 3: pHash Index ---[/bold]")

        if not _needs_phash_rebuild(reference, state, force):
            console.print(
                f"[green]✓ pHash index is up-to-date ({state.phash_version}) "
                "— skipping rebuild[/green]"
            )
            summary["pHash index"] = f"skipped ({state.phash_version})"
        else:
            # D-02: null before work begins
            state = replace(state, phash_version=None)
            reference.save_state(state)

            # D-15: tqdm progress bar inside build_index() passes through unchanged
            matcher = PHashCardMatcher(reference=reference)
            matcher.build_index(force=True)

            state = replace(state, phash_version=state.images_version)
            reference.save_state(state)
            summary["pHash index"] = f"rebuilt ({state.phash_version})"

    except Exception as exc:
        # SYNC-04: non-zero exit with descriptive error message
        click.echo(
            f"Error during sync: {type(exc).__name__}: {exc}",
            err=True,
        )
        raise SystemExit(1) from exc

    # ------------------------------------------------------------------
    # Summary panel (D-14)
    # ------------------------------------------------------------------
    summary_text = "\n".join(
        f"{artifact}: {outcome}" for artifact, outcome in summary.items()
    )
    console.print(Panel(summary_text, title="Sync Summary", expand=False))


if __name__ == "__main__":
    main()
