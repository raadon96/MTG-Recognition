"""The summaries `mtg status`, `mtg update` and `mtg download-images` print, on
results built in the test.
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
    def test_offline_summary_prints_the_latest_release_as_unknown(self) -> None:
        report = replace(
            STATUS,
            without_phash=1,
            images_present=2,
            images_missing=1,
            latest_version=None,
        )

        lines, next_step = status_summary(report)

        assert lines == [
            f"Catalog: {V1}",
            "Printings without a pHash: 1",
            f"AllPrintings: {V1}",
            "Reference images: 2 present, 1 missing",
            "Latest MTGJSON release: unknown",
        ]
        assert next_step == "Up to date: nothing to run."

    def test_catalog_without_version_and_missing_allprintings_are_shown(
        self,
    ) -> None:
        lines, next_step = status_summary(
            replace(STATUS, catalog_version=None, allprintings_version=None)
        )

        assert lines[0] == "Catalog: not built by `mtg update`"
        assert lines[2] == "AllPrintings: absent"
        assert next_step == "Run: mtg update"

    def test_new_release_names_the_command_to_run(self) -> None:
        _, next_step = status_summary(replace(STATUS, latest_version=V2))

        assert next_step == "Run: mtg update"


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
            f"Image failures written to {reference.log_file}",
        ]
