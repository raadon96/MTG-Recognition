"""The summaries `mtg status`, `mtg update` and `mtg build-index` print, on
results built in the test.
"""

import pytest

from mtg_recognition.cli.reference import index_summary, status_summary, update_summary
from mtg_recognition.jobs.sync import (
    DownloadReport,
    ImageFailure,
    IndexResult,
    Status,
    UpdateResult,
)
from mtg_recognition.reference.layout import ReferenceData
from tests.unit.sync_fakes import V1, V2


@pytest.fixture
def reference(tmp_path) -> ReferenceData:
    return ReferenceData(tmp_path / "reference")


@pytest.mark.unit
class TestIndexSummary:
    def test_index_already_built_is_up_to_date(self, reference) -> None:
        lines = index_summary(reference, IndexResult(V1, downloads=None))

        assert lines == [f"pHash index: up to date ({V1})"]

    def test_rebuilt_index_reports_the_images(self, reference) -> None:
        downloads = DownloadReport(downloaded=3, failed=(), missing_url=2, removed=0)

        lines = index_summary(reference, IndexResult(V1, downloads))

        assert lines == [
            "Reference images: 3 downloaded, 0 failed, 2 without URL",
            "Removed images (no printing): 0",
            f"pHash index: rebuilt ({V1})",
        ]

    def test_unreadable_images_ask_for_another_build(self, reference) -> None:
        downloads = DownloadReport(downloaded=0, failed=(), missing_url=0, removed=1)
        unhashable = (ImageFailure("uuid-bolt", None, "hash", "truncated"),)

        lines = index_summary(reference, IndexResult(V1, downloads, unhashable))

        assert lines[1:] == [
            "Removed images (no printing): 1",
            f"pHash index: rebuilt ({V1})",
            "Unreadable images deleted: 1; run `mtg build-index` again",
            f"Image failures written to {reference.log_file}",
        ]

    def test_failed_downloads_point_to_the_log(self, reference) -> None:
        failure = ImageFailure("uuid-bolt", "https://x", "download", "HTTP 404")
        downloads = DownloadReport(
            downloaded=2, failed=(failure,), missing_url=0, removed=0
        )

        lines = index_summary(reference, IndexResult(V1, downloads))

        assert lines[0] == "Reference images: 2 downloaded, 1 failed, 0 without URL"
        assert lines[-1] == f"Image failures written to {reference.log_file}"


@pytest.mark.unit
class TestUpdateSummary:
    def test_rebuilt_catalog_names_the_allprintings_source(self, reference) -> None:
        downloads = DownloadReport(downloaded=3, failed=(), missing_url=0, removed=0)
        result = UpdateResult(
            latest_version=V1,
            catalog_version=V1,
            catalog_rebuilt=True,
            allprintings_downloaded=True,
            index=IndexResult(V1, downloads),
        )

        lines = update_summary(reference, result)

        assert lines == [
            f"Catalog: rebuilt ({V1})",
            f"AllPrintings: downloaded ({V1})",
            "Reference images: 3 downloaded, 0 failed, 0 without URL",
            "Removed images (no printing): 0",
            f"pHash index: rebuilt ({V1})",
        ]

    def test_catalog_up_to_date_leaves_out_allprintings(self, reference) -> None:
        result = UpdateResult(
            latest_version=V1,
            catalog_version=V1,
            catalog_rebuilt=False,
            allprintings_downloaded=False,
            index=IndexResult(V1, downloads=None),
        )

        lines = update_summary(reference, result)

        assert lines == [
            f"Catalog: up to date ({V1})",
            f"pHash index: up to date ({V1})",
        ]


@pytest.mark.unit
class TestStatusSummary:
    def test_offline_summary_prints_the_latest_release_as_unknown(self) -> None:
        report = Status(
            catalog_version=V1,
            phash_version=V1,
            index_matches=True,
            allprintings_version=V1,
            images_present=2,
            images_missing=1,
            latest_version=None,
        )

        lines, next_step = status_summary(report)

        assert lines == [
            f"Catalog: {V1}",
            f"pHash index: {V1} (matches the catalog)",
            f"AllPrintings: {V1}",
            "Reference images: 2 present, 1 missing",
            "Latest MTGJSON release: unknown",
        ]
        assert next_step == "Up to date: nothing to run."

    def test_missing_allprintings_and_stale_index_are_shown(self) -> None:
        report = Status(
            catalog_version=V2,
            phash_version=V1,
            index_matches=False,
            allprintings_version=None,
            images_present=3,
            images_missing=0,
            latest_version=V2,
        )

        lines, _ = status_summary(report)

        assert lines[1:3] == [
            f"pHash index: {V1} (does not match the catalog)",
            "AllPrintings: absent",
        ]

    def test_new_release_names_the_command_to_run(self) -> None:
        report = Status(
            catalog_version=V1,
            phash_version=V1,
            index_matches=True,
            allprintings_version=V1,
            images_present=3,
            images_missing=0,
            latest_version=V2,
        )

        _, next_step = status_summary(report)

        assert next_step == "Run: mtg update"
