import asyncio
import logging
import shutil
import zipfile
from pathlib import Path
from urllib import request

import aiohttp
from sqlmodel import Session, SQLModel, create_engine, select
from tqdm.asyncio import tqdm

from mtg_recognition.settings import settings

from .schemas import CardIdentifiers, Cards, MobileCache, Sets

SCRYFALL_HEADERS = {
    "User-Agent": "mtg-recognition/1.0 (contact: adnen.boumessouer@gmail.com)",
}

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.ERROR,
    filename=settings.LOG_FILE,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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


def create_database(source_engine, dest_engine, force: bool = False):
    """Migrate data from Source to Destination with streaming."""

    # 1. Reset Destination
    if settings.LOCAL_DB_FILE.exists():
        print(f"Local database already exists at {settings.LOCAL_DB_FILE}.")
        if not force:
            print("Skipping database creation.")
            return

        print(f"Removing existing database at {settings.LOCAL_DB_FILE}...")
        settings.LOCAL_DB_FILE.unlink()

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

    print(f"Database created at {settings.LOCAL_DB_FILE} ({total_migrated} cards)")

    # # Cleanup source to save space
    # if settings.TMP_DIR.exists():
    #     shutil.rmtree(settings.TMP_DIR)
    #


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
                    logging.error(f"404 Not Found: {url}")
                    # Don't retry 404s
                    return
                else:
                    # Log other HTTP errors
                    logging.warning(
                        f"HTTP {response.status}: {url} (Attempt {attempt + 1})"
                    )
        except TimeoutError:
            logging.warning(f"Timeout: {url} (Attempt {attempt + 1})")
        except aiohttp.ClientError as e:
            logging.warning(f"ClientError: {url} - {str(e)} (Attempt {attempt + 1})")
        except Exception as e:
            logging.error(f"Unexpected Error: {url} - {str(e)}")
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

        uuid, url = item
        file_path = settings.IMG_DIR / f"{uuid}.jpg"

        await download_file_with_retry(session, url, file_path)

        pbar.update(1)
        queue.task_done()


async def download_manager(dest_engine, max_concurrent=20):
    """Manages the download queue and workers."""

    # 1. Fetch URLs efficiently (Stream only UUID and URL)
    with Session(dest_engine) as session:
        # Only select needed columns
        statement = select(MobileCache.uuid, MobileCache.url).where(
            MobileCache.url is not None
        )
        # Load all into memory is okay for just UUID+URL strings (approx 10MB RAM)
        # If strict on RAM, we could stream this too, but queue filling is fast.
        cards_to_download = session.exec(statement).all()

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
        for card in cards_to_download:
            await queue.put((card.uuid, card.url))

        # 5. Signal Completion
        for _ in range(max_concurrent):
            await queue.put(None)

        # 6. Wait for completion
        await queue.join()

        # Cancel workers (cleanup)
        for worker in workers:
            worker.cancel()

        pbar.close()


def download_raw_images(dest_engine):
    # Windows/Python 3.8+ specific event loop policy fix might be needed on some systems
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(
        download_manager(dest_engine, max_concurrent=25)
    )  # TODO download webp images instead

    print("\nDone!")


def build_mobilecache():
    """Download AllPrintings ZIP and rebuild MobileCache. Steps 1+2 of the original main().

    Returns the SQLAlchemy Engine pointing at settings.LOCAL_DB_FILE so the caller
    can pass it directly to download_images(dest_engine=...) without recreating it.
    """
    print("--- Step 1: Retrieving Data ---")
    if not settings.MTGO_DB_FILE_ZIP.exists():
        print(f"Downloading {settings.MTGJSON_URL}...")
        req = request.Request(settings.MTGJSON_URL, headers=SCRYFALL_HEADERS)
        with (
            request.urlopen(req) as response,
            open(settings.MTGO_DB_FILE_ZIP, "wb") as out_file,
        ):
            shutil.copyfileobj(response, out_file)

    print("Extracting...")
    with zipfile.ZipFile(settings.MTGO_DB_FILE_ZIP, "r") as zip_ref:
        zip_ref.extractall(settings.TMP_DIR)

    print("\n--- Step 2: Building Local Database ---")

    # Engines
    source_engine = create_engine(f"sqlite:///{settings.MTGO_DB_FILE}")
    dest_engine = create_engine(f"sqlite:///{settings.LOCAL_DB_FILE}")

    # create cache database for memory efficient lookups when matching
    create_database(source_engine, dest_engine, force=True)

    return dest_engine


def download_images(dest_engine=None):
    """Download reference images. Step 3 of the original main().

    If dest_engine is None, creates a new SQLAlchemy Engine pointing at
    settings.LOCAL_DB_FILE so the function is callable standalone.
    download_raw_images() is already incremental — it skips files that exist on disk.
    """
    if dest_engine is None:
        dest_engine = create_engine(f"sqlite:///{settings.LOCAL_DB_FILE}")

    print("\n--- Step 3: Downloading Images ---")
    download_raw_images(dest_engine)


def main():
    """Backward-compatible entry point. Called by 'python -m mtg_recognition.jobs.build_database'."""
    dest_engine = build_mobilecache()
    download_images(dest_engine)


if __name__ == "__main__":
    main()

# uv run python -m mtg_recognition.jobs.build_database
