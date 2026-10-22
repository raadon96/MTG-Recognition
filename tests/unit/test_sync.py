"""Tests for the reference data jobs in `mtg_recognition.jobs.sync`.

The steps run for real against `ReferenceData(tmp_path)` and the fake network of
`sync_fakes`; the index is built in-process (`workers=1`).
"""

import io
import json
import shutil
import sqlite3
import zipfile

import pytest

from mtg_recognition.engine.matching.phash import load_phash_index
from mtg_recognition.jobs.sync import (
    META_URL,
    allprintings_version,
    build_index,
    generate_image_uri,
    status,
    update,
)
from mtg_recognition.reference import ReferenceData, SyncState

from .sync_fakes import (
    PRINTING_V2,
    PRINTINGS_V1,
    V1,
    V2,
    build_allprintings,
    jpeg,
)


@pytest.fixture
def reference(tmp_path) -> ReferenceData:
    return ReferenceData(tmp_path / "reference")


def indexed_uuids(reference: ReferenceData) -> set[str]:
    return set(load_phash_index(reference).uuids)


def database_uuids(reference: ReferenceData) -> set[str]:
    with sqlite3.connect(reference.db_path) as connection:
        return {row[0] for row in connection.execute("SELECT uuid FROM mobilecache")}


def corrupt_zip() -> bytes:
    """A ZIP whose AllPrintings fails its CRC check once read, so extraction fails
    midway, after it started writing.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("AllPrintings.sqlite", b"x" * 100_000)
    return buffer.getvalue().replace(b"x" * 100_000, b"y" * 100_000)


@pytest.mark.unit
class TestUpdate:
    def test_first_update_builds_everything(self, reference, network) -> None:
        with network.client() as client:
            result = update(reference, client, workers=1)

        assert result.database_rebuilt
        assert database_uuids(reference) == {
            "uuid-bolt",
            "uuid-delver",
            "uuid-aberration",
        }
        assert result.index.downloads is not None
        assert result.index.downloads.downloaded == 3
        assert indexed_uuids(reference) == {
            "uuid-bolt",
            "uuid-delver",
            "uuid-aberration",
        }
        assert reference.load_state() == SyncState(V1, V1)
        assert reference.allprintings_path.exists()
        assert not reference.allprintings_zip_path.exists()

    def test_second_update_with_the_same_release_does_nothing(
        self, reference, network
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
            network.requests.clear()

            result = update(reference, client, workers=1)

        assert not result.database_rebuilt
        assert not result.index.built
        assert network.requests == [META_URL]

    def test_stale_allprintings_is_downloaded_again(self, reference, network) -> None:
        """An AllPrintings left from an older release is replaced, and the database
        is labelled with the release in AllPrintings' `meta` table, not Meta.json's.
        """
        reference.allprintings_path.parent.mkdir(parents=True)
        build_allprintings(reference.allprintings_path, V1, PRINTINGS_V1)
        network.publish(V2, [*PRINTINGS_V1, PRINTING_V2], meta_version="5.3.0+20260516")

        with network.client() as client:
            result = update(reference, client, workers=1)

        assert network.zip_requests() == 1
        assert result.allprintings_downloaded
        assert "uuid-ponder" in database_uuids(reference)
        assert reference.load_state() == SyncState("5.3.0+20260516", "5.3.0+20260516")

    def test_current_allprintings_is_reused(self, reference, network) -> None:
        reference.allprintings_path.parent.mkdir(parents=True)
        build_allprintings(reference.allprintings_path, V1, PRINTINGS_V1)

        with network.client() as client:
            result = update(reference, client, workers=1)

        assert network.zip_requests() == 0
        assert not result.allprintings_downloaded
        assert reference.load_state().mobilecache_version == V1

    def test_image_404_is_reported_and_the_index_still_built(
        self, reference, network
    ) -> None:
        bolt_url = generate_image_uri(PRINTINGS_V1[0][2])
        del network.images[bolt_url]

        with network.client() as client:
            result = update(reference, client, workers=1)

        downloads = result.index.downloads
        assert downloads is not None
        assert downloads.downloaded == 2
        assert [(f.uuid, f.url, f.error) for f in downloads.failed] == [
            ("uuid-bolt", bolt_url, "HTTP 404")
        ]
        assert network.requests.count(bolt_url) == 1  # 404s are not retried
        assert reference.log_file.read_text() == (
            f"uuid-bolt\t{bolt_url}\tdownload\tHTTP 404\n"
        )
        assert indexed_uuids(reference) == {"uuid-delver", "uuid-aberration"}
        assert reference.load_state() == SyncState(V1, V1)

    def test_interrupted_update_resumes(self, reference, network) -> None:
        """When the index build fails after the database was rebuilt, the next run
        builds only the index.
        """
        served_images = dict(network.images)
        network.images.clear()  # every image 404s: no image to hash

        with network.client() as client, pytest.raises(ValueError):
            update(reference, client, workers=1)
        assert reference.load_state() == SyncState(V1, None)

        network.images.update(served_images)
        with network.client() as client:
            result = update(reference, client, workers=1)

        assert not result.database_rebuilt
        assert network.zip_requests() == 1
        assert indexed_uuids(reference) == {
            "uuid-bolt",
            "uuid-delver",
            "uuid-aberration",
        }
        assert reference.load_state() == SyncState(V1, V1)
        assert not reference.log_file.exists()

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

    def test_force_rebuilds_without_downloading_again(self, reference, network) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
            result = update(reference, client, force=True, workers=1)

        assert result.database_rebuilt
        assert result.index.built
        assert network.zip_requests() == 1


@pytest.mark.unit
class TestBuildIndex:
    def test_after_a_pull_downloads_only_the_new_printings_images(
        self, tmp_path, reference, network
    ) -> None:
        """A pull brings the database a maintainer built from a newer release and
        its state; the images of the older release are on disk.
        """
        with network.client() as client:
            update(reference, client, workers=1)
        network.publish(V2, [*PRINTINGS_V1, PRINTING_V2])
        maintainer = ReferenceData(tmp_path / "maintainer")
        with network.client() as client:
            update(maintainer, client, workers=1)

        shutil.copyfile(maintainer.db_path, reference.db_path)
        state_file = reference.root / ".mtg-sync-state.json"
        state_file.write_text(
            json.dumps({"mobilecache_version": V2, "phash_version": V1})
        )
        network.requests.clear()

        with network.client() as client:
            result = build_index(reference, client, workers=1)

        assert network.requests == [generate_image_uri(PRINTING_V2[2])]
        assert result.downloads is not None
        assert result.downloads.downloaded == 1
        assert "uuid-ponder" in indexed_uuids(reference)
        assert json.loads(state_file.read_text()) == {
            "mobilecache_version": V2,
            "phash_version": V2,
        }

    def test_image_with_no_printing_is_removed(self, reference, network) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        stray = reference.image_path("uuid-removed-by-mtgjson")
        stray.write_bytes(jpeg(9))

        with network.client() as client:
            result = build_index(reference, client, force=True, workers=1)

        assert result.downloads is not None
        assert result.downloads.removed == 1
        assert not stray.exists()
        assert indexed_uuids(reference) == database_uuids(reference)

    def test_download_that_does_not_decode_is_a_failure(
        self, reference, network
    ) -> None:
        bolt_url = generate_image_uri(PRINTINGS_V1[0][2])
        network.images[bolt_url] = jpeg(0)[:200]

        with network.client() as client:
            result = update(reference, client, workers=1)

        downloads = result.index.downloads
        assert downloads is not None
        assert [(f.uuid, f.step) for f in downloads.failed] == [
            ("uuid-bolt", "download")
        ]
        assert network.requests.count(bolt_url) == 1
        assert list(reference.images_dir.glob("uuid-bolt*")) == []
        assert reference.log_file.read_text().startswith(
            f"uuid-bolt\t{bolt_url}\tdownload\t"
        )
        assert indexed_uuids(reference) == {"uuid-delver", "uuid-aberration"}

    def test_unreadable_image_on_disk_is_deleted_then_downloaded_again(
        self, reference, network
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        bolt = reference.image_path("uuid-bolt")
        bolt.write_bytes(bolt.read_bytes()[:200])

        with network.client() as client:
            result = build_index(reference, client, force=True, workers=1)

        assert [(f.uuid, f.step) for f in result.unhashable] == [("uuid-bolt", "hash")]
        assert not bolt.exists()
        assert "uuid-bolt" not in indexed_uuids(reference)
        bolt_url = generate_image_uri(PRINTINGS_V1[0][2])
        assert reference.log_file.read_text().startswith(
            f"uuid-bolt\t{bolt_url}\thash\t"
        )

        with network.client() as client:
            result = build_index(reference, client, force=True, workers=1)

        assert result.downloads is not None
        assert result.downloads.downloaded == 1
        assert result.unhashable == ()
        assert indexed_uuids(reference) == database_uuids(reference)
        assert not reference.log_file.exists()

    def test_does_nothing_when_the_index_matches(self, reference, network) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
            network.requests.clear()

            result = build_index(reference, client, workers=1)

        assert not result.built
        assert network.requests == []


@pytest.mark.unit
class TestStatus:
    def test_offline_status_shows_latest_release_as_unknown(
        self, reference, network
    ) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        (reference.images_dir / "uuid-bolt.jpg").unlink()
        network.offline = True

        with network.client() as client:
            report = status(reference, client)

        assert report.latest_version is None
        assert report.database_version == V1
        assert report.index_matches
        assert report.allprintings_version == V1
        assert (report.images_present, report.images_missing) == (2, 1)
        assert report.next_command is None

    def test_new_release_suggests_update(self, reference, network) -> None:
        with network.client() as client:
            update(reference, client, workers=1)
        network.publish(V2, PRINTINGS_V1)

        with network.client() as client:
            report = status(reference, client)

        assert report.latest_version == V2
        assert report.next_command == "mtg update"
