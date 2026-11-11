"""Tests for the reference data jobs in `mtg_recognition.jobs.sync`.

The steps run for real against `ReferenceData(tmp_path)` and the fake network of
`sync_fakes`; the pHashes are computed in-process (`workers=1`). The summaries the
commands print are tested in tests/unit/cli.
"""

import io
import sqlite3
import zipfile

import cv2
import pytest
from sqlmodel import SQLModel

from mtg_recognition.engine.matching.phash import load_phashes, shortlist
from mtg_recognition.jobs.schemas import CardIdentifiers, Cards, Sets
from mtg_recognition.jobs.sync import (
    META_URL,
    allprintings_version,
    build,
    build_catalog,
    download_missing_images,
    generate_image_uri,
    status,
    update,
)
from mtg_recognition.reference.catalog import (
    Printing,
    SqliteCardCatalog,
    catalog_version,
)
from mtg_recognition.reference.layout import ReferenceData

from .sync_fakes import (
    PRINTING_V2,
    PRINTINGS_V1,
    V1,
    V2,
    FakeNetwork,
    build_allprintings,
    jpeg,
    write_allprintings,
)

ALL_V1 = {"uuid-bolt", "uuid-delver", "uuid-aberration"}
BOLT_URL = generate_image_uri(PRINTINGS_V1[0][2])
PONDER_URL = generate_image_uri(PRINTING_V2[2])


@pytest.fixture
def reference(tmp_path) -> ReferenceData:
    return ReferenceData(tmp_path / "reference")


def uuids_with_a_phash(reference: ReferenceData) -> set[str]:
    return set(load_phashes(reference).uuids)


def catalog_uuids(reference: ReferenceData) -> set[str]:
    with sqlite3.connect(reference.catalog_path) as connection:
        return {row[0] for row in connection.execute("SELECT uuid FROM catalog")}


def image_requests(network: FakeNetwork) -> set[str]:
    return set(network.requests) & set(network.images)


def strip_to_a_catalog_without_version(reference: ReferenceData) -> None:
    """Turn the catalog into one from before ADR 0006, like the one committed
    before `mtg reference update` stored pHashes: no `phash` column, no `meta` table.
    """
    with sqlite3.connect(reference.catalog_path) as connection:
        connection.execute("ALTER TABLE catalog DROP COLUMN phash")
        connection.execute("DROP TABLE meta")


def corrupt_zip() -> bytes:
    """A ZIP whose AllPrintings fails its CRC check once read, so extraction fails
    midway, after it started writing.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("AllPrintings.sqlite", b"x" * 100_000)
    return buffer.getvalue().replace(b"x" * 100_000, b"y" * 100_000)


@pytest.mark.unit
class TestBuildCatalog:
    def test_double_faced_printing_has_its_own_face_name_and_image(
        self, reference
    ) -> None:
        reference.allprintings_path.parent.mkdir(parents=True)
        build_allprintings(reference.allprintings_path, V1, PRINTINGS_V1)

        build_catalog(reference.allprintings_path, reference.catalog_path)

        delver_id = PRINTINGS_V1[1][2]
        printings = SqliteCardCatalog(reference.catalog_path).lookup(
            ["uuid-delver", "uuid-aberration"]
        )
        assert printings["uuid-delver"].name == "Delver of Secrets"
        assert printings["uuid-delver"].image_url == generate_image_uri(
            delver_id, "front"
        )
        assert printings["uuid-aberration"].name == "Insectile Aberration"
        assert printings["uuid-aberration"].image_url == generate_image_uri(
            delver_id, "back"
        )

    def test_card_with_both_halves_on_the_front_keeps_the_front_image(
        self, reference
    ) -> None:
        """Adventure, split, flip and aftermath cards print both halves on the
        front, so neither half has a back image of its own.
        """
        name = "Bonecrusher Giant // Stomp"
        scryfall_id = "d4444444-0000-0000-0000-000000000000"
        reference.allprintings_path.parent.mkdir(parents=True)
        build_allprintings(
            reference.allprintings_path,
            V1,
            [
                (
                    "uuid-giant",
                    name,
                    scryfall_id,
                    "adventure",
                    "a",
                    "Bonecrusher Giant",
                ),
                ("uuid-stomp", name, scryfall_id, "adventure", "b", "Stomp"),
            ],
        )

        build_catalog(reference.allprintings_path, reference.catalog_path)

        printings = SqliteCardCatalog(reference.catalog_path).lookup(
            ["uuid-giant", "uuid-stomp"]
        )
        front = generate_image_uri(scryfall_id, "front")
        assert [(p.name, p.image_url) for p in printings.values()] == [
            (name, front),
            (name, front),
        ]


@pytest.mark.unit
class TestShownOnTheCard:
    def test_list_card_shows_the_original_set_code_and_number_and_its_own_year(
        self, reference
    ) -> None:
        plst = Sets(
            code="PLST", type="masters", baseSetSize=5075, releaseDate="2020-09-26"
        )
        rows = [
            khm(),
            plst,
            *printing_rows("list", "PLST", "KHM-214", originalReleaseDate="2022-03-04"),
        ]

        printing = build_and_look_up(reference, rows)["list"]

        assert printed(printing) == ("KHM", "214", "285", "2022")
        assert not printing.is_plain

    def test_promo_shows_its_parent_set_details(self, reference) -> None:
        pkhm = Sets(
            code="PKHM",
            type="promo",
            parentCode="KHM",
            baseSetSize=1,
            releaseDate="2021-02-05",
        )
        rows = [
            khm(),
            pkhm,
            *printing_rows("expansion card", "KHM", "214", name="Cosmos Elixir"),
            *printing_rows("promo", "PKHM", "214p", name="Cosmos Elixir"),
        ]

        printing = build_and_look_up(reference, rows)["promo"]

        assert printed(printing) == ("KHM", "214", "285", "2021")
        assert not printing.is_plain

    @pytest.mark.parametrize(
        ("original_artist", "expected"),
        [
            ("Kev Walker", (None, "123", "350", "1998")),
            ("Someone Else", (None, None, None, None)),
        ],
        ids=["original found", "no original"],
    )
    def test_world_championship_card_shows_its_original_printing_details(
        self, reference, original_artist, expected
    ) -> None:
        usg = Sets(
            code="USG", type="expansion", baseSetSize=350, releaseDate="1998-10-12"
        )
        wc99 = Sets(
            code="WC99", type="memorabilia", baseSetSize=1, releaseDate="1999-08-04"
        )
        old_card = {"name": "Shivan Dragon", "frameVersion": "1997"}
        rows = [
            usg,
            wc99,
            *printing_rows(
                "original", "USG", "123", artist=original_artist, **old_card
            ),
            *printing_rows("deck card", "WC99", "123", artist="Kev Walker", **old_card),
        ]

        printing = build_and_look_up(reference, rows)["deck card"]

        assert printed(printing) == expected
        assert not printing.is_plain

    @pytest.mark.parametrize(
        ("release_date", "frame", "number", "expected"),
        [
            ("1994-05-31", "1993", "10", (None, None, None, None)),
            ("1994-06-01", "1993", "10", (None, None, None, "1994")),
            ("1998-06-14", "1997", "10", (None, None, None, "1998")),
            ("1998-06-15", "1997", "10", (None, "10", "280", "1998")),
            ("2019-09-30", "2015", "290", ("NEW", "290", "280", "2019")),
            ("2019-10-01", "2015", "290", ("NEW", "290", None, "2019")),
            ("2023-03-31", "2015", "10", ("NEW", "10", "280", "2023")),
            ("2023-04-01", "2015", "10", ("NEW", "10", None, "2023")),
        ],
        ids=[
            "before Legends: no year",
            "from Legends: year",
            "before Exodus: no number",
            "from Exodus: number and size",
            "above the size before October 2019: size",
            "above the size from October 2019: no size",
            "before April 2023: size",
            "from April 2023: no size",
        ],
    )
    def test_details_shown_change_with_the_release_date(
        self, reference, release_date, frame, number, expected
    ) -> None:
        rows = [
            Sets(
                code="NEW", type="expansion", baseSetSize=280, releaseDate=release_date
            ),
            *printing_rows("printing", "NEW", number, frameVersion=frame),
        ]

        assert printed(build_and_look_up(reference, rows)["printing"]) == expected

    def test_range_year_stores_its_end_year(self, reference) -> None:
        """A 1997-frame reprint in Seventh Edition shows `© 1993-2001`."""
        seventh = Sets(
            code="7ED", type="core", baseSetSize=350, releaseDate="2001-04-11"
        )
        rows = [seventh, *printing_rows("reprint", "7ED", "10", frameVersion="1997")]

        assert build_and_look_up(reference, rows)["reprint"].printed_year == "2001"

    def test_year_comes_from_the_original_release_date(self, reference) -> None:
        sld = Sets(code="SLD", type="box", baseSetSize=900, releaseDate="2019-12-02")
        rows = [
            sld,
            *printing_rows("drop", "SLD", "10", originalReleaseDate="2023-05-01"),
        ]

        assert build_and_look_up(reference, rows)["drop"].printed_year == "2023"

    def test_retro_reprint_shows_its_number_without_base_set_size(
        self, reference
    ) -> None:
        brr = Sets(
            code="BRR", type="expansion", baseSetSize=63, releaseDate="2022-11-18"
        )
        rows = [brr, *printing_rows("retro", "BRR", "5", frameVersion="1997")]

        assert printed(build_and_look_up(reference, rows)["retro"]) == (
            None,
            "5",
            None,
            "2022",
        )

    def test_event_promo_shows_neither_set_code_nor_base_set_size(
        self, reference
    ) -> None:
        """Event promos show PRM, FNM, H19 and sizes of their own, which no field
        holds.
        """
        prm = Sets(code="PRM", type="promo", baseSetSize=5, releaseDate="2021-06-01")
        rows = [prm, *printing_rows("event promo", "PRM", "3")]

        printing = build_and_look_up(reference, rows)["event promo"]

        assert printed(printing) == (None, "3", None, "2021")

    def test_playtest_card_shows_only_its_year(self, reference) -> None:
        cmb = Sets(code="CMB1", type="funny", baseSetSize=300, releaseDate="2019-11-07")
        rows = [cmb, *printing_rows("playtest", "CMB1", "20", promoTypes="playtest")]

        printing = build_and_look_up(reference, rows)["playtest"]

        assert printed(printing) == (None, None, None, "2019")

    def test_flavor_name_is_kept_beside_the_card_name(self, reference) -> None:
        rows = [
            khm(),
            *printing_rows(
                "flavored",
                "KHM",
                "10",
                name="Mothra's Great Cocoon",
                flavorName="Mothra's Cocoon",
            ),
        ]

        printing = build_and_look_up(reference, rows)["flavored"]

        assert (printing.name, printing.flavor_name) == (
            "Mothra's Great Cocoon",
            "Mothra's Cocoon",
        )

    def test_plain_printing_shows_its_own_details_and_a_number_without_suffix(
        self, reference
    ) -> None:
        rows = [
            khm(),
            *printing_rows("plain", "KHM", "254"),
            *printing_rows("star", "KHM", "254★"),
        ]

        printings = build_and_look_up(reference, rows)

        assert printings["plain"].is_plain
        assert not printings["star"].is_plain


def khm() -> Sets:
    """Kaldheim, a set many printings in these tests show the details of."""
    return Sets(code="KHM", type="expansion", baseSetSize=285, releaseDate="2021-02-05")


def printing_rows(
    uuid: str, set_code: str, collector_number: str, **fields
) -> list[SQLModel]:
    """A printing's `cards` row, in the 2015 frame, and its `cardIdentifiers` row."""
    fields = {"name": uuid, "artist": "Artist", "frameVersion": "2015"} | fields
    return [
        Cards(uuid=uuid, setCode=set_code, number=collector_number, **fields),
        CardIdentifiers(uuid=uuid),
    ]


def build_and_look_up(
    reference: ReferenceData, rows: list[SQLModel]
) -> dict[str, Printing]:
    """Build the catalog from an AllPrintings holding `rows`, and look up each of
    its printings.
    """
    # read before writing: the rows can't be read once their session closes
    uuids = [row.uuid for row in rows if isinstance(row, Cards) and row.uuid]
    reference.allprintings_path.parent.mkdir(parents=True)
    write_allprintings(reference.allprintings_path, V1, rows)
    build_catalog(reference.allprintings_path, reference.catalog_path)
    return SqliteCardCatalog(reference.catalog_path).lookup(uuids)


def printed(printing: Printing) -> tuple[str | None, ...]:
    """The set code, collector number, base set size and year `printing` shows."""
    return (
        printing.printed_set_code,
        printing.printed_collector_number,
        printing.printed_base_set_size,
        printing.printed_year,
    )


@pytest.mark.unit
class TestUpdate:
    def test_first_update_builds_everything(self, reference, network) -> None:
        with network.client() as client:
            result = update(reference, client, workers=1)

        assert result.build is not None
        assert result.build.downloaded == 3
        assert result.build.hashed == 3
        assert catalog_uuids(reference) == ALL_V1
        assert uuids_with_a_phash(reference) == ALL_V1
        assert catalog_version(reference.catalog_path) == V1
        assert reference.allprintings_path.exists()
        assert not reference.allprintings_zip_path.exists()

    def test_printings_keep_their_order_whatever_the_allprintings_order(
        self, tmp_path
    ) -> None:
        """Candidates with the same similarity keep the catalog's order (#146), so
        that order must not change when a release stores the same printings in
        another order. (The fake numbers printings in insertion order, so the
        reversed file also has other collector numbers; UUIDs are unchanged.)
        """
        orders = []
        for name, printings in [
            ("stored", PRINTINGS_V1),
            ("reversed", PRINTINGS_V1[::-1]),
        ]:
            reference = ReferenceData(tmp_path / name / "reference")
            (tmp_path / name).mkdir()
            network = FakeNetwork(tmp_path / name)
            network.publish(V1, printings)

            with network.client() as client:
                update(reference, client, workers=1)

            orders.append(load_phashes(reference).uuids)
        assert orders == [["uuid-aberration", "uuid-bolt", "uuid-delver"]] * 2

    def test_second_update_with_the_same_release_does_nothing(
        self, reference, network
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
            network.requests.clear()

            result = update(reference, client, workers=1)

        assert result.build is None
        assert result.catalog_version == V1
        assert network.requests == [META_URL]

    def test_new_release_hashes_only_the_new_printings(
        self, reference, network
    ) -> None:
        """The images of the previous release are gone from disk, so a printing
        whose pHash was not reused would need its image downloaded again.
        """
        with network.client() as client:
            update(reference, client, workers=1)
        for path in reference.images_dir.glob("*.jpg"):
            path.unlink()
        network.publish(V2, [*PRINTINGS_V1, PRINTING_V2])
        network.requests.clear()

        with network.client() as client:
            result = update(reference, client, workers=1)

        assert image_requests(network) == {PONDER_URL}
        assert result.build is not None
        assert (result.build.reused, result.build.hashed) == (3, 1)
        assert uuids_with_a_phash(reference) == {*ALL_V1, "uuid-ponder"}
        assert catalog_version(reference.catalog_path) == V2

    def test_reused_phash_still_shortlists_its_printing(
        self, reference, network
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        bolt = cv2.imread(str(reference.image_path("uuid-bolt")))
        assert bolt is not None
        network.publish(V2, [*PRINTINGS_V1, PRINTING_V2])

        with network.client() as client:
            update(reference, client, workers=1)

        uuids, similarities = shortlist(load_phashes(reference), bolt, k=1)
        assert (uuids, similarities) == (["uuid-bolt"], [1.0])

    def test_changed_image_url_downloads_the_image_again_and_recomputes_its_phash(
        self, reference, network
    ) -> None:
        """A printing keeps its UUID but gets a new image URL, as each face of a
        double-faced card did in #155; its old image and pHash are of the wrong
        picture.
        """
        with network.client() as client:
            update(reference, client, workers=1)
        new_id = "a9999999-0000-0000-0000-000000000000"
        new_url = generate_image_uri(new_id)
        network.publish(
            V2,
            [
                ("uuid-bolt", "Lightning Bolt", new_id, "normal", None, None),
                *PRINTINGS_V1[1:],
            ],
        )
        network.images[new_url] = jpeg(8)
        network.requests.clear()

        with network.client() as client:
            result = update(reference, client, workers=1)

        assert image_requests(network) == {new_url}
        assert reference.image_path("uuid-bolt").read_bytes() == jpeg(8)
        assert result.build is not None
        assert (result.build.reused, result.build.hashed) == (2, 1)

    def test_catalog_without_version_is_built_again_from_the_images_on_disk(
        self, reference, network
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        strip_to_a_catalog_without_version(reference)
        network.requests.clear()

        with network.client() as client:
            result = update(reference, client, workers=1)

        assert image_requests(network) == set()
        assert result.build is not None
        assert (result.build.reused, result.build.hashed) == (0, 3)
        assert uuids_with_a_phash(reference) == ALL_V1
        assert catalog_version(reference.catalog_path) == V1

    def test_stale_allprintings_is_downloaded_again(self, reference, network) -> None:
        """An AllPrintings left from an older release is replaced, and the catalog
        is labelled with the release in AllPrintings' `meta` table, not Meta.json's.
        """
        reference.allprintings_path.parent.mkdir(parents=True)
        build_allprintings(reference.allprintings_path, V1, PRINTINGS_V1)
        network.publish(V2, [*PRINTINGS_V1, PRINTING_V2], meta_version="5.3.0+20260516")

        with network.client() as client:
            result = update(reference, client, workers=1)

        assert network.zip_requests() == 1
        assert result.build is not None
        assert result.build.allprintings_downloaded
        assert "uuid-ponder" in catalog_uuids(reference)
        assert catalog_version(reference.catalog_path) == "5.3.0+20260516"

    def test_current_allprintings_is_reused(self, reference, network) -> None:
        reference.allprintings_path.parent.mkdir(parents=True)
        build_allprintings(reference.allprintings_path, V1, PRINTINGS_V1)

        with network.client() as client:
            result = update(reference, client, workers=1)

        assert network.zip_requests() == 0
        assert result.build is not None
        assert not result.build.allprintings_downloaded

    def test_printing_whose_image_404s_has_no_phash(self, reference, network) -> None:
        del network.images[BOLT_URL]

        with network.client() as client:
            result = update(reference, client, workers=1)

        assert result.build is not None
        assert result.build.downloaded == 2
        assert result.build.without_phash == 1
        assert [(f.uuid, f.url, f.error) for f in result.build.failed] == [
            ("uuid-bolt", BOLT_URL, "HTTP 404")
        ]
        assert network.requests.count(BOLT_URL) == 1  # 404s are not retried
        assert reference.log_file.read_text() == (
            f"uuid-bolt\t{BOLT_URL}\tdownload\tHTTP 404\n"
        )
        assert uuids_with_a_phash(reference) == {"uuid-delver", "uuid-aberration"}

    def test_printing_without_phash_is_hashed_at_the_next_release(
        self, reference, network
    ) -> None:
        served_bolt = network.images.pop(BOLT_URL)
        with network.client() as client:
            update(reference, client, workers=1)
        network.images[BOLT_URL] = served_bolt
        network.publish(V2, PRINTINGS_V1)

        with network.client() as client:
            result = update(reference, client, workers=1)

        assert result.build is not None
        assert (result.build.reused, result.build.hashed) == (2, 1)
        assert uuids_with_a_phash(reference) == ALL_V1
        assert not reference.log_file.exists()

    def test_printing_that_loses_its_image_url_loses_its_phash(
        self, reference, network
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        bolt_without_url = (*PRINTINGS_V1[0][:2], None, *PRINTINGS_V1[0][3:])
        network.publish(V2, [bolt_without_url, *PRINTINGS_V1[1:]])

        with network.client() as client:
            result = update(reference, client, workers=1)

        assert result.build is not None
        assert result.build.without_phash == 1
        assert uuids_with_a_phash(reference) == {"uuid-delver", "uuid-aberration"}

    def test_build_with_no_phash_keeps_no_catalog_and_resumes(
        self, reference, network
    ) -> None:
        served_images = dict(network.images)
        network.images.clear()  # every image 404s: no image to hash

        with network.client() as client, pytest.raises(ValueError):
            update(reference, client, workers=1)
        assert not reference.catalog_path.exists()
        assert list(reference.root.rglob("*.part")) == []

        network.images.update(served_images)
        with network.client() as client:
            update(reference, client, workers=1)

        assert network.zip_requests() == 1
        assert uuids_with_a_phash(reference) == ALL_V1
        assert not reference.log_file.exists()

    def test_failed_build_keeps_the_images_of_the_previous_catalog(
        self, reference, network
    ) -> None:
        """Bolt is gone from the new release, but the previous catalog, which
        the failed build keeps, still holds it.
        """
        with network.client() as client:
            update(reference, client, workers=1)
        strip_to_a_catalog_without_version(reference)  # nothing to reuse
        network.publish(V2, PRINTINGS_V1[1:])
        network.images.clear()  # every image 404s: no printing gets a pHash
        for uuid in ("uuid-delver", "uuid-aberration"):
            reference.image_path(uuid).unlink()

        with network.client() as client, pytest.raises(ValueError):
            update(reference, client, workers=1)

        assert reference.image_path("uuid-bolt").exists()

    def test_failed_extraction_keeps_the_old_allprintings(
        self, reference, network
    ) -> None:
        reference.allprintings_path.parent.mkdir(parents=True)
        build_allprintings(reference.allprintings_path, V1, PRINTINGS_V1)
        network.publish(V2, PRINTINGS_V1)
        network.zip_bytes = corrupt_zip()

        with network.client() as client, pytest.raises(zipfile.BadZipFile):
            update(reference, client, workers=1)

        assert allprintings_version(reference) == V1
        assert list(reference.root.rglob("*.part")) == []
        assert not reference.allprintings_zip_path.exists()

    def test_image_with_no_printing_is_removed(self, reference, network) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        stray = reference.image_path("uuid-removed-by-mtgjson")
        stray.write_bytes(jpeg(9))
        network.publish(V2, PRINTINGS_V1)

        with network.client() as client:
            result = update(reference, client, workers=1)

        assert result.build is not None
        assert result.build.removed == 1
        assert not stray.exists()

    def test_download_that_does_not_decode_is_a_failure(
        self, reference, network
    ) -> None:
        network.images[BOLT_URL] = jpeg(0)[:200]

        with network.client() as client:
            result = update(reference, client, workers=1)

        assert result.build is not None
        assert [(f.uuid, f.step) for f in result.build.failed] == [
            ("uuid-bolt", "download")
        ]
        assert network.requests.count(BOLT_URL) == 1
        assert list(reference.images_dir.glob("uuid-bolt*")) == []
        assert reference.log_file.read_text().startswith(
            f"uuid-bolt\t{BOLT_URL}\tdownload\t"
        )
        assert uuids_with_a_phash(reference) == {"uuid-delver", "uuid-aberration"}

    def test_unreadable_image_on_disk_is_deleted_then_downloaded_again(
        self, reference, network
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        strip_to_a_catalog_without_version(reference)
        bolt = reference.image_path("uuid-bolt")
        bolt.write_bytes(bolt.read_bytes()[:200])

        with network.client() as client:
            result = update(reference, client, workers=1)

        assert result.build is not None
        assert [(f.uuid, f.step) for f in result.build.unhashable] == [
            ("uuid-bolt", "hash")
        ]
        assert not bolt.exists()
        assert "uuid-bolt" not in uuids_with_a_phash(reference)
        assert reference.log_file.read_text().startswith(
            f"uuid-bolt\t{BOLT_URL}\thash\t"
        )

        network.publish(V2, PRINTINGS_V1)
        with network.client() as client:
            result = update(reference, client, workers=1)

        assert result.build is not None
        assert result.build.downloaded == 1
        assert uuids_with_a_phash(reference) == ALL_V1


def set_catalog_image_url(reference: ReferenceData, uuid: str, url: str) -> None:
    """Give a printing another image URL in the catalog, as an older version of
    the code that builds the catalog would have.
    """
    with sqlite3.connect(reference.catalog_path) as connection:
        connection.execute("UPDATE catalog SET url = ? WHERE uuid = ?", (url, uuid))


def reference_files(reference: ReferenceData) -> dict[str, bytes]:
    """Every file of the reference data, by path relative to its root."""
    return {
        str(path.relative_to(reference.root)): path.read_bytes()
        for path in reference.root.rglob("*")
        if path.is_file()
    }


@pytest.mark.unit
class TestBuild:
    def test_current_allprintings_rebuilds_offline_reusing_every_phash(
        self, reference, network
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        network.offline = True
        network.requests.clear()

        with network.client() as client:
            report = build(reference, client, workers=1)

        assert network.requests == []
        assert (report.reused, report.hashed, report.downloaded) == (3, 0, 0)
        assert not report.allprintings_downloaded
        assert uuids_with_a_phash(reference) == ALL_V1
        assert catalog_version(reference.catalog_path) == V1

    def test_absent_allprintings_is_downloaded_when_still_the_latest_release(
        self, reference, network
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        reference.allprintings_path.unlink()
        network.requests.clear()

        with network.client() as client:
            report = build(reference, client, workers=1)

        assert network.zip_requests() == 1
        assert image_requests(network) == set()
        assert report.allprintings_downloaded
        assert (report.version, report.reused, report.hashed) == (V1, 3, 0)
        assert catalog_version(reference.catalog_path) == V1

    @pytest.mark.parametrize(
        ("allprintings", "described"),
        [(None, "AllPrintings is absent"), (V2, f"AllPrintings on disk is {V2}")],
    )
    def test_release_no_longer_published_fails_and_changes_nothing(
        self, reference, network, allprintings, described
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        reference.allprintings_path.unlink()
        if allprintings is not None:
            build_allprintings(reference.allprintings_path, allprintings, PRINTINGS_V1)
        network.publish(V2, [*PRINTINGS_V1, PRINTING_V2])
        files = reference_files(reference)
        network.requests.clear()

        with network.client() as client, pytest.raises(ValueError) as error:
            build(reference, client, workers=1)

        assert str(error.value) == (
            f"The catalog is MTGJSON release {V1}; {described}; the latest release "
            f"is {V2}. MTGJSON no longer publishes {V1}, so the catalog can't be "
            "rebuilt from it: run `mtg reference update`."
        )
        assert network.requests == [META_URL]
        assert reference_files(reference) == files

    def test_no_catalog_fails_and_suggests_update(self, reference, network) -> None:
        with network.client() as client, pytest.raises(ValueError) as error:
            build(reference, client, workers=1)

        assert "run `mtg reference update`" in str(error.value)
        assert network.requests == []

    def test_catalog_without_version_fails_and_suggests_update(
        self, reference, network
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        strip_to_a_catalog_without_version(reference)

        with network.client() as client, pytest.raises(ValueError) as error:
            build(reference, client, workers=1)

        assert "run `mtg reference update`" in str(error.value)

    def test_changed_image_url_downloads_the_image_again_and_recomputes_its_phash(
        self, reference, network
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        set_catalog_image_url(reference, "uuid-bolt", "https://old.example/bolt.jpg")
        bolt = reference.image_path("uuid-bolt")
        bolt.write_bytes(jpeg(7))
        network.requests.clear()

        with network.client() as client:
            report = build(reference, client, workers=1)

        assert image_requests(network) == {BOLT_URL}
        assert bolt.read_bytes() == network.images[BOLT_URL]
        assert (report.reused, report.hashed) == (2, 1)
        served_bolt = cv2.imread(str(bolt))
        assert served_bolt is not None
        uuids, _ = shortlist(load_phashes(reference), served_bolt, k=1)
        assert uuids == ["uuid-bolt"]

    def test_changed_image_url_that_fails_to_download_leaves_no_phash(
        self, reference, network
    ) -> None:
        """The image on disk is from the old URL, so hashing it would give the
        printing the pHash of another picture.
        """
        with network.client() as client:
            update(reference, client, workers=1)
        set_catalog_image_url(reference, "uuid-bolt", "https://old.example/bolt.jpg")
        del network.images[BOLT_URL]

        with network.client() as client:
            report = build(reference, client, workers=1)

        assert [(f.uuid, f.step) for f in report.failed] == [("uuid-bolt", "download")]
        assert not reference.image_path("uuid-bolt").exists()
        assert uuids_with_a_phash(reference) == {"uuid-delver", "uuid-aberration"}

    def test_no_cache_downloads_allprintings_and_every_image_again(
        self, reference, network
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        bolt = reference.image_path("uuid-bolt")
        bolt.write_bytes(jpeg(7))
        network.requests.clear()

        with network.client() as client:
            report = build(reference, client, no_cache=True, workers=1)

        assert network.zip_requests() == 1
        assert report.allprintings_downloaded
        assert image_requests(network) == set(network.images)
        assert bolt.read_bytes() == network.images[BOLT_URL]
        assert (report.reused, report.hashed) == (0, 3)

    def test_no_cache_with_release_no_longer_published_changes_nothing(
        self, reference, network
    ) -> None:
        """The AllPrintings on disk is the catalog's release, but `--no-cache`
        would download it again, and MTGJSON now serves another one.
        """
        with network.client() as client:
            update(reference, client, workers=1)
        network.publish(V2, [*PRINTINGS_V1, PRINTING_V2])
        files = reference_files(reference)
        network.requests.clear()

        with network.client() as client, pytest.raises(ValueError, match="no longer"):
            build(reference, client, no_cache=True, workers=1)

        assert network.requests == [META_URL]
        assert reference_files(reference) == files


@pytest.mark.unit
class TestDownloadMissingImages:
    def test_downloads_only_the_missing_images(self, reference, network) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        reference.image_path("uuid-bolt").unlink()
        network.requests.clear()

        with network.client() as client:
            report = download_missing_images(reference, client)

        assert network.requests == [BOLT_URL]
        assert (report.downloaded, report.failed) == (1, ())
        assert reference.image_path("uuid-bolt").exists()

    def test_leaves_the_catalog_unchanged(self, reference, network) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        reference.image_path("uuid-bolt").unlink()
        catalog_bytes = reference.catalog_path.read_bytes()

        with network.client() as client:
            download_missing_images(reference, client)

        assert reference.catalog_path.read_bytes() == catalog_bytes

    def test_failures_go_to_its_own_log_and_leave_the_build_log(
        self, reference, network
    ) -> None:
        del network.images[BOLT_URL]
        with network.client() as client:
            update(reference, client, workers=1)
        build_log = reference.log_file.read_text()

        with network.client() as client:
            download_missing_images(reference, client)

        assert "uuid-bolt" in reference.download_log_file.read_text()
        assert reference.log_file.read_text() == build_log

    def test_run_without_failures_removes_its_log_and_leaves_the_build_log(
        self, reference, network
    ) -> None:
        del network.images[BOLT_URL]
        with network.client() as client:
            update(reference, client, workers=1)
            download_missing_images(reference, client)
        build_log = reference.log_file.read_text()

        network.publish(V1, PRINTINGS_V1)  # Scryfall serves the image again
        with network.client() as client:
            download_missing_images(reference, client)

        assert not reference.download_log_file.exists()
        assert reference.log_file.read_text() == build_log

    def test_second_run_downloads_nothing(self, reference, network) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        for path in reference.images_dir.glob("*.jpg"):
            path.unlink()

        with network.client() as client:
            first = download_missing_images(reference, client)
            network.requests.clear()
            second = download_missing_images(reference, client)

        assert first.downloaded == 3
        assert network.requests == []
        assert (second.downloaded, second.failed) == (0, ())

    def test_failures_are_reported(self, reference, network) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        reference.image_path("uuid-bolt").unlink()
        del network.images[BOLT_URL]

        with network.client() as client:
            report = download_missing_images(reference, client)

        assert report.downloaded == 0
        assert [f.uuid for f in report.failed] == ["uuid-bolt"]


@pytest.mark.unit
class TestStatus:
    def test_offline_status_shows_latest_release_as_unknown(
        self, reference, network
    ) -> None:
        del network.images[BOLT_URL]
        with network.client() as client:
            update(reference, client, workers=1)
        (reference.images_dir / "uuid-delver.jpg").unlink()
        network.offline = True

        with network.client() as client:
            report = status(reference, client)

        assert report.latest_version is None
        assert report.catalog_version == V1
        assert report.without_phash == 1
        assert report.allprintings_version == V1
        assert (report.images_present, report.images_missing) == (1, 2)
        assert report.next_command is None

    def test_new_release_suggests_update(self, reference, network) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        network.publish(V2, PRINTINGS_V1)

        with network.client() as client:
            report = status(reference, client)

        assert report.latest_version == V2
        assert report.next_command == "mtg reference update"

    def test_catalog_without_version_suggests_update(self, reference, network) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        strip_to_a_catalog_without_version(reference)

        with network.client() as client:
            report = status(reference, client)

        assert report.catalog_version is None
        assert report.without_phash == 3
        assert report.next_command == "mtg reference update"
