"""The summaries `mtg reference status`, `mtg reference update` and `mtg reference
download-images` print, on results built in the test.
"""

from dataclasses import replace

import pytest

from mtg_recognition.cli.reference import (
    download_images_summary,
    status_summary,
    update_summary,
)
from mtg_recognition.jobs.sync import (
    BuildReport,
    DownloadReport,
    ImageFailure,
    Status,
    UpdateResult,
)
from mtg_recognition.reference.layout import ReferenceData
from tests.unit.sync_fakes import V1, V2


@pytest.fixture
def reference(tmp_path) -> ReferenceData:
    return ReferenceData(tmp_path / "reference")


BUILD = BuildReport(
    version=V1,
    allprintings_downloaded=True,
    removed=0,
    downloaded=3,
    failed=(),
    reused=0,
    hashed=3,
    unhashable=(),
    without_phash=0,
)


@pytest.mark.unit
class TestUpdateSummary:
    def test_catalog_up_to_date_is_one_line(self, reference) -> None:
        lines = update_summary(reference, UpdateResult(V1, V1, build=None))

        assert lines == [f"Catalog: up to date ({V1})"]

    def test_rebuilt_catalog_reports_the_images_and_the_phashes(
        self, reference
    ) -> None:
        report = replace(
            BUILD, allprintings_downloaded=False, reused=2, hashed=1, without_phash=1
        )

        lines = update_summary(reference, UpdateResult(V1, V1, report))

        assert lines == [
            f"Catalog: rebuilt ({V1})",
            f"AllPrintings: reused ({V1})",
            "Reference images: 3 downloaded, 0 failed",
            "Removed images (no printing): 0",
            "pHashes: 2 reused, 1 computed, 1 printings without one",
        ]

    def test_unreadable_images_are_downloaded_at_the_next_build(
        self, reference
    ) -> None:
        unhashable = (ImageFailure("uuid-bolt", None, "hash", "truncated"),)

        lines = update_summary(
            reference, UpdateResult(V1, V1, replace(BUILD, unhashable=unhashable))
        )

        assert lines[-2:] == [
            "Unreadable images deleted: 1; the next build downloads them again",
            f"Image failures written to {reference.log_file}",
        ]

    def test_failed_downloads_point_to_the_log(self, reference) -> None:
        failure = ImageFailure("uuid-bolt", "https://x", "download", "HTTP 404")

        lines = update_summary(
            reference,
            UpdateResult(V1, V1, replace(BUILD, downloaded=2, failed=(failure,))),
        )

        assert lines[2] == "Reference images: 2 downloaded, 1 failed"
        assert lines[-1] == f"Image failures written to {reference.log_file}"


STATUS = Status(
    catalog_version=V1,
    without_phash=0,
    allprintings_version=V1,
    images_present=3,
    images_missing=0,
    latest_version=V1,
)


@pytest.mark.unit
class TestStatusSummary:
    def test_latest_release_is_up_to_date(self) -> None:
        lines, verdict = status_summary(STATUS)

        assert lines == [
            f"Catalog: {V1}",
            "Printings without a pHash: 0",
            f"AllPrintings: {V1}",
            "Reference images: 3 present, 0 missing",
            f"Latest MTGJSON release: {V1}",
        ]
        assert verdict.plain == "✓ Up to date"

    def test_absent_allprintings_does_not_change_the_verdict(self) -> None:
        lines, verdict = status_summary(replace(STATUS, allprintings_version=None))

        assert lines[2] == "AllPrintings: absent"
        assert verdict.plain == "✓ Up to date"

    def test_offline_the_latest_release_is_unknown(self) -> None:
        lines, verdict = status_summary(replace(STATUS, latest_version=None))

        assert lines[-1] == "Latest MTGJSON release: unknown"
        assert verdict.plain == "? Latest release unknown (offline)"

    def test_new_release_is_outdated(self) -> None:
        _, verdict = status_summary(replace(STATUS, latest_version=V2))

        assert verdict.plain == "✗ Outdated: run mtg reference update"

    def test_catalog_without_version_is_outdated(self) -> None:
        lines, verdict = status_summary(replace(STATUS, catalog_version=None))

        assert lines[0] == "Catalog: wasn't built"
        assert verdict.plain == "✗ Outdated: run mtg reference update"

    def test_catalog_without_version_is_outdated_offline(self) -> None:
        _, verdict = status_summary(
            replace(STATUS, catalog_version=None, latest_version=None)
        )

        assert verdict.plain == "✗ Outdated: run mtg reference update"


@pytest.mark.unit
class TestDownloadImagesSummary:
    def test_no_failure_is_one_line(self, reference) -> None:
        lines = download_images_summary(reference, DownloadReport(3, failed=()))

        assert lines == ["Reference images: 3 downloaded, 0 failed"]

    def test_failed_downloads_point_to_the_log(self, reference) -> None:
        failure = ImageFailure("uuid-bolt", "https://x", "download", "HTTP 404")

        lines = download_images_summary(reference, DownloadReport(2, (failure,)))

        assert lines == [
            "Reference images: 2 downloaded, 1 failed",
            f"Image failures written to {reference.download_log_file}",
        ]
