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

from mtg_recognition.engine.matching.phash import load_phashes, shortlist
from mtg_recognition.jobs.sync import (
    META_URL,
    allprintings_version,
    build_catalog,
    download_missing_images,
    generate_image_uri,
    status,
    update,
)
from mtg_recognition.reference.catalog import SqliteCardCatalog, catalog_version
from mtg_recognition.reference.layout import ReferenceData

from .sync_fakes import (
    PRINTING_V2,
    PRINTINGS_V1,
    V1,
    V2,
    FakeNetwork,
    build_allprintings,
    jpeg,
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
    before `mtg update` stored pHashes: no `phash` column, no `meta` table.
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

    def test_force_downloads_every_image_again_and_recomputes_every_phash(
        self, reference, network
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        bolt = reference.image_path("uuid-bolt")
        bolt.write_bytes(jpeg(7))
        network.requests.clear()

        with network.client() as client:
            result = update(reference, client, force=True, workers=1)

        assert image_requests(network) == set(network.images)
        assert bolt.read_bytes() == network.images[BOLT_URL]
        assert result.build is not None
        assert (result.build.reused, result.build.hashed) == (0, 3)
        assert network.zip_requests() == 0  # the AllPrintings on disk is current

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

    def test_run_without_failures_keeps_the_log_of_the_last_build(
        self, reference, network
    ) -> None:
        del network.images[BOLT_URL]
        with network.client() as client:
            update(reference, client, workers=1)
        log = reference.log_file.read_text()

        network.publish(V1, PRINTINGS_V1)  # Scryfall serves the image again
        (reference.images_dir / "uuid-delver.jpg").unlink()
        with network.client() as client:
            download_missing_images(reference, client)

        assert reference.log_file.read_text() == log

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

    def test_failures_are_reported_and_logged(self, reference, network) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        reference.image_path("uuid-bolt").unlink()
        del network.images[BOLT_URL]

        with network.client() as client:
            report = download_missing_images(reference, client)

        assert report.downloaded == 0
        assert [f.uuid for f in report.failed] == ["uuid-bolt"]
        assert "uuid-bolt" in reference.log_file.read_text()


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
        assert report.next_command == "mtg update"

    def test_catalog_without_version_suggests_update(self, reference, network) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        strip_to_a_catalog_without_version(reference)

        with network.client() as client:
            report = status(reference, client)

        assert report.catalog_version is None
        assert report.without_phash == 3
        assert report.next_command == "mtg update"
