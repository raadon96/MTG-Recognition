"""`mtg benchmark status` and `mtg benchmark update-baseline`, on results files
written in the test. The tests run in an empty directory, without model weights.
"""

import json

import pytest
from click.testing import CliRunner
from sqlmodel import Session, SQLModel, create_engine

from mtg_recognition.benchmark.results import (
    BASELINE_PATH,
    DATASET_ROOT,
    LATEST_PATH,
    MANIFEST_NAME,
    ImageOutcome,
    ImageScore,
    Totals,
    read_results,
    summarise,
    write_results,
)
from mtg_recognition.cli import cli
from mtg_recognition.cli.benchmark import totals_line
from mtg_recognition.reference.catalog import CatalogPrinting
from mtg_recognition.reference.layout import ReferenceData

BOLT = "11111111-1111-1111-1111-111111111111"
GIANT = "33333333-3333-3333-3333-333333333333"
UNKNOWN = "99999999-9999-9999-9999-999999999999"


def outcome(
    exact: bool, count: bool = True, recognised: list[str | None] | None = None
) -> ImageOutcome:
    return ImageOutcome(
        score=ImageScore(exact, count, exact),
        recognised=[BOLT] if recognised is None else recognised,
        runtime_s=1.5,
        ocr_fallbacks=0,
    )


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def status(*args: str):
    return CliRunner().invoke(cli, ["benchmark", "status", *args])


@pytest.mark.unit
def test_the_totals_line_names_each_metric() -> None:
    totals = Totals(
        images=3,
        exact_printing=1,
        card_count=2,
        same_card_name=3,
        ocr_fallbacks=4,
        runtime_s=9.0,
    )

    assert totals_line(totals) == (
        "exact printing 1/3, card count 2/3, same card name 3/3, OCR fallbacks 4"
    )


@pytest.mark.unit
class TestStatus:
    def test_summarises_the_baseline(self, in_tmp) -> None:
        write_results(
            summarise("abc123", "5.3.0", {"a": outcome(True), "b": outcome(False)}),
            BASELINE_PATH,
        )

        result = status()

        assert result.exit_code == 0, result.output
        assert result.output.splitlines() == [
            f"Baseline: {BASELINE_PATH}",
            "Commit: abc123",
            "MTGJSON: 5.3.0",
            "exact printing 1/2, card count 2/2, same card name 1/2, OCR fallbacks 0",
            "Runtime: 3.0s",
        ]

    def test_latest_summarises_the_latest_results(self, in_tmp) -> None:
        write_results(summarise("def456", "5.4.0", {}), LATEST_PATH)

        result = status("--latest")

        assert result.exit_code == 0, result.output
        assert result.output.splitlines()[:3] == [
            f"Latest results: {LATEST_PATH}",
            "Commit: def456",
            "MTGJSON: 5.4.0",
        ]

    def test_reports_latest_results_that_match_the_baseline(self, in_tmp) -> None:
        write_results(summarise("abc123", "5.3.0", {"a": outcome(True)}), BASELINE_PATH)
        write_results(summarise("def456", "5.3.0", {"a": outcome(True)}), LATEST_PATH)

        result = status()

        assert result.exit_code == 0, result.output
        assert result.output.splitlines()[-1] == (
            "latest results (at def456) match the baseline"
        )

    def test_counts_the_query_images_that_differ_from_the_baseline(
        self, in_tmp
    ) -> None:
        baseline = {"a": outcome(True), "b": outcome(True), "gone": outcome(True)}
        latest = {"a": outcome(False), "b": outcome(True), "new": outcome(True)}
        write_results(summarise("abc123", "5.3.0", baseline), BASELINE_PATH)
        write_results(summarise("def456", "5.3.0", latest), LATEST_PATH)

        result = status("--latest")

        assert result.exit_code == 0, result.output
        assert result.output.splitlines()[-1] == (
            "latest results (at def456) differ from the baseline in 3 query images"
        )

    def test_failures_list_the_label_and_the_recognised_printings(self, in_tmp) -> None:
        _write_catalog(in_tmp)
        manifest = {"a.jpg": [BOLT], "b.jpg": [BOLT, GIANT], "c.jpg": [BOLT]}
        DATASET_ROOT.mkdir(parents=True)
        (DATASET_ROOT / MANIFEST_NAME).write_text(json.dumps(manifest))
        failing = outcome(False, count=False, recognised=[None])
        failing.error = "ValueError: unreadable"
        images = {
            "a.jpg": outcome(True),
            "b.jpg": failing,
            "c.jpg": outcome(False, recognised=[UNKNOWN]),
        }
        write_results(summarise("abc123", "5.3.0", images), BASELINE_PATH)

        result = status("--failures")

        assert result.exit_code == 0, result.output
        assert result.output.splitlines()[5:] == [
            "Failures:",
            "  b.jpg",
            "    label: Lightning Bolt (M10), Realm-Cloaked Giant // Cast Off (ZNR)",
            "    recognised: nothing",
            "    error: ValueError: unreadable",
            "  c.jpg",
            "    label: Lightning Bolt (M10)",
            f"    recognised: {UNKNOWN}",
        ]

    def test_a_missing_baseline_names_the_command_that_creates_it(self, in_tmp) -> None:
        result = status()

        assert result.exit_code == 1
        assert result.output == (
            f"No baseline at {BASELINE_PATH}; run `uv run mtg benchmark run`, "
            "then `uv run mtg benchmark update-baseline`\n"
        )

    def test_missing_latest_results_name_the_command_that_creates_them(
        self, in_tmp
    ) -> None:
        write_results(summarise("abc123", "5.3.0", {}), BASELINE_PATH)

        result = status("--latest")

        assert result.exit_code == 1
        assert result.output == (
            f"No latest results at {LATEST_PATH}; run `uv run mtg benchmark run`\n"
        )


def _write_catalog(root) -> None:
    reference = ReferenceData(root / "data" / "reference")
    reference.root.mkdir(parents=True)
    engine = create_engine(f"sqlite:///{reference.catalog_path}")
    # SQLModel sets __table__ on table models at runtime but doesn't declare it
    SQLModel.metadata.create_all(
        engine,
        tables=[CatalogPrinting.__table__],  # ty: ignore[unresolved-attribute]
    )
    with Session(engine) as session:
        session.add_all(
            [
                CatalogPrinting(
                    uuid=BOLT,
                    name="Lightning Bolt",
                    setCode="M10",
                    number="1",
                    baseSetSize="1",
                ),
                CatalogPrinting(
                    uuid=GIANT,
                    name="Realm-Cloaked Giant // Cast Off",
                    setCode="ZNR",
                    number="2",
                    baseSetSize="1",
                ),
            ]
        )
        session.commit()
    engine.dispose()


@pytest.mark.unit
def test_update_baseline_adopts_the_latest_results(in_tmp) -> None:
    latest = summarise("abc123", "5.3.0", {})
    write_results(latest, LATEST_PATH)

    result = CliRunner().invoke(cli, ["benchmark", "update-baseline"])

    assert result.exit_code == 0
    assert read_results(BASELINE_PATH) == latest


@pytest.mark.unit
def test_update_baseline_without_latest_results_fails(in_tmp) -> None:
    result = CliRunner().invoke(cli, ["benchmark", "update-baseline"])

    assert result.exit_code == 1
    assert not BASELINE_PATH.exists()
