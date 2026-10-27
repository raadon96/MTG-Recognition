"""A fake network for the reference data jobs: an `httpx.MockTransport` serving a
tiny `Meta.json`, a tiny AllPrintings ZIP built from `jobs/schemas.py`, and a few
small JPEGs.
"""

import io
import zipfile
from pathlib import Path

import httpx
from PIL import Image
from sqlmodel import Session, SQLModel, create_engine

from mtg_recognition.jobs.schemas import CardIdentifiers, Cards, Meta, Sets
from mtg_recognition.jobs.sync import META_URL, generate_image_uri
from mtg_recognition.settings import settings

V1 = "5.3.0+20260302"
V2 = "5.3.0+20260517"

# (uuid, name, scryfallId). The two Delver faces are a double-faced pair: two
# printings sharing one Scryfall id, so one front image.
PRINTINGS_V1 = [
    ("uuid-bolt", "Lightning Bolt", "a1111111-0000-0000-0000-000000000000"),
    ("uuid-delver", "Delver of Secrets", "b2222222-0000-0000-0000-000000000000"),
    ("uuid-aberration", "Insectile Aberration", "b2222222-0000-0000-0000-000000000000"),
]
PRINTING_V2 = ("uuid-ponder", "Ponder", "c3333333-0000-0000-0000-000000000000")


def build_allprintings(
    path: Path, version: str, printings: list[tuple[str, str, str]]
) -> None:
    """Write a tiny AllPrintings database: one set holding `printings`."""
    engine = create_engine(f"sqlite:///{path}")
    # SQLModel sets __table__ on table models at runtime but doesn't declare it
    # on the class.
    SQLModel.metadata.create_all(
        engine,
        tables=[
            Cards.__table__,  # ty: ignore[unresolved-attribute]
            CardIdentifiers.__table__,  # ty: ignore[unresolved-attribute]
            Sets.__table__,  # ty: ignore[unresolved-attribute]
            Meta.__table__,  # ty: ignore[unresolved-attribute]
        ],
    )
    with Session(engine) as session:
        session.add(Meta(version=version))
        session.add(
            Sets(
                code="ISD", name="Innistrad", baseSetSize=264, releaseDate="2011-09-30"
            )
        )
        for number, (uuid, name, scryfall_id) in enumerate(printings, start=1):
            session.add(Cards(uuid=uuid, name=name, setCode="ISD", number=str(number)))
            session.add(CardIdentifiers(uuid=uuid, scryfallId=scryfall_id))
        session.commit()
    engine.dispose()


def jpeg(seed: int) -> bytes:
    """A small JPEG, different for each seed."""
    image = Image.new("RGB", (32, 44), (seed * 70 % 256, seed * 30 % 256, 90))
    image.paste((255, 255, 255), (0, 0, 8 + seed * 4, 8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeNetwork:
    """MTGJSON and Scryfall, served from memory; records every requested URL."""

    def __init__(self, work_dir: Path) -> None:
        self._work_dir = work_dir
        self.latest_version = V1
        self.zip_bytes = b""
        self.images: dict[str, bytes] = {}
        self.offline = False
        self.requests: list[str] = []
        self.publish(V1, PRINTINGS_V1)

    def publish(
        self,
        version: str,
        printings: list[tuple[str, str, str]],
        meta_version: str | None = None,
    ) -> None:
        """Release `version` holding `printings`; the AllPrintings inside the ZIP
        says `meta_version` (default `version`).
        """
        self.latest_version = version
        db_path = self._work_dir / f"AllPrintings-{version}.sqlite"
        db_path.unlink(missing_ok=True)
        build_allprintings(db_path, meta_version or version, printings)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.write(db_path, "AllPrintings.sqlite")
        self.zip_bytes = buffer.getvalue()
        for seed, (_, _, scryfall_id) in enumerate(printings):
            self.images.setdefault(generate_image_uri(scryfall_id), jpeg(seed))

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if self.offline:
            raise httpx.ConnectError("offline", request=request)
        if url == META_URL:
            return httpx.Response(200, json={"meta": {"version": self.latest_version}})
        if url == settings.MTGJSON_URL:
            return httpx.Response(200, content=self.zip_bytes)
        if url in self.images:
            return httpx.Response(200, content=self.images[url])
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def zip_requests(self) -> int:
        return self.requests.count(settings.MTGJSON_URL)
