"""`mtg benchmark run`, `mtg benchmark update-baseline` and `mtg benchmark
status`: measure recognition on the labelled query images, compare it with the
baseline, and summarise either results file.
"""

import json
import shutil
import subprocess
import time
from collections.abc import Sequence

import click

from mtg_recognition.benchmark.results import (
    BASELINE_PATH,
    DATASET_ROOT,
    GATED_METRICS,
    LATEST_PATH,
    MANIFEST_NAME,
    Comparison,
    ImageOutcome,
    ManifestProblems,
    Results,
    Totals,
    compare,
    read_results,
    score_image,
    summarise,
    validate_manifest,
    write_results,
)
from mtg_recognition.engine.pipeline import Pipeline
from mtg_recognition.reference.catalog import Printing, SqliteCardCatalog
from mtg_recognition.reference.layout import ReferenceData


@click.group(name="benchmark")
def benchmark() -> None:
    """Measure recognition on the benchmark and compare it with the baseline."""


@benchmark.command(name="run")
def run() -> None:
    """Recognise every labelled query image, write the results to
    output/benchmark/latest.json and compare them with the baseline.

    Exits non-zero when the manifest doesn't match the dataset, or when the results
    differ from the baseline in exact printing or card count, better or worse.
    """
    # check the manifest before loading any model
    manifest_path = DATASET_ROOT / MANIFEST_NAME
    if not manifest_path.exists():
        click.echo(
            f"No manifest at {manifest_path}: run from the repo root, after "
            "`uv run dvc pull data/benchmark.dvc`",
            err=True,
        )
        raise SystemExit(1)
    manifest: dict[str, list[str]] = json.loads(manifest_path.read_text())
    reference = ReferenceData.default()
    labelled_uuids = sorted({uuid for label in manifest.values() for uuid in label})
    labelled_printings = SqliteCardCatalog(reference.catalog_path).lookup(
        labelled_uuids
    )
    problems = validate_manifest(DATASET_ROOT, manifest, labelled_printings)
    for heading, items in [
        ("Images without a label", problems.unlabelled_images),
        ("Labels for missing files", problems.missing_images),
        ("UUIDs not in the catalog", problems.unknown_uuids),
    ]:
        if items:
            click.echo(f"{heading}:\n" + "\n".join(f"  {i}" for i in items), err=True)
    if problems != ManifestProblems():
        click.echo(f"The manifest doesn't match the dataset: {manifest_path}", err=True)
        raise SystemExit(1)

    # recognise each query image at the pipeline's defaults
    pipeline = Pipeline()
    images: dict[str, ImageOutcome] = {}
    for idx, (image, label) in enumerate(sorted(manifest.items()), 1):
        start = time.perf_counter()
        recognition = pipeline.recognize(DATASET_ROOT / image)
        runtime_s = time.perf_counter() - start

        best = [
            card.ranked[0].printing if card.ranked else None
            for card in recognition.cards
        ]
        score = score_image([labelled_printings[uuid] for uuid in label], best)
        images[image] = ImageOutcome(
            score=score,
            recognised=[p.uuid if p is not None else None for p in best],
            runtime_s=round(runtime_s, 3),
            ocr_fallbacks=sum(card.ocr_error is not None for card in recognition.cards),
            error=recognition.error,
        )
        mark = "✓" if score.exact_printing and score.card_count else "✗"
        click.echo(f"[{idx}/{len(manifest)}] {mark} {image}  {runtime_s:.1f}s")

    results = summarise(_commit(), reference.load_state().catalog_version, images)
    write_results(results, LATEST_PATH)
    click.echo(f"\nResults written to {LATEST_PATH}")

    # compare with the baseline
    if not BASELINE_PATH.exists():
        click.echo(
            f"No baseline at {BASELINE_PATH}; adopt these results with "
            "`uv run mtg benchmark update-baseline`",
            err=True,
        )
        raise SystemExit(1)
    baseline = read_results(BASELINE_PATH)
    comparison = compare(results, baseline)
    click.echo("\n".join(_comparison_report(results, baseline, comparison)))
    if comparison.differs:
        raise SystemExit(1)


@benchmark.command(name="update-baseline")
def update_baseline() -> None:
    """Copy the latest results onto the baseline, without recognising anything.
    Commit data/benchmark-baseline.json afterwards.
    """
    if not LATEST_PATH.exists():
        click.echo(
            f"No results at {LATEST_PATH}; run `uv run mtg benchmark run`", err=True
        )
        raise SystemExit(1)
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(LATEST_PATH, BASELINE_PATH)
    click.echo(f"Baseline updated: {BASELINE_PATH}")


@benchmark.command(name="status")
@click.option(
    "--latest", is_flag=True, help="Summarise the latest results, not the baseline."
)
@click.option(
    "--failures",
    is_flag=True,
    help="List the query images that fail exact printing or card count.",
)
def status(latest: bool, failures: bool) -> None:
    """Summarise the baseline or the latest results, without recognising anything
    or writing to disk.
    """
    # read the results to summarise
    if latest:
        heading, path = "Latest results", LATEST_PATH
        missing = f"No latest results at {LATEST_PATH}; run `uv run mtg benchmark run`"
    else:
        heading, path = "Baseline", BASELINE_PATH
        missing = (
            f"No baseline at {BASELINE_PATH}; run `uv run mtg benchmark run`, "
            "then `uv run mtg benchmark update-baseline`"
        )
    if not path.exists():
        click.echo(missing, err=True)
        raise SystemExit(1)
    results = read_results(path)
    click.echo(f"{heading}: {path}")
    click.echo(f"Commit: {results.commit}")
    click.echo(f"MTGJSON: {results.mtgjson_version}")
    click.echo(totals_line(results.totals))
    click.echo(f"Runtime: {results.totals.runtime_s:.1f}s")

    # the latest results against the baseline, when both exist
    if LATEST_PATH.exists() and BASELINE_PATH.exists():
        latest_results = results if latest else read_results(LATEST_PATH)
        baseline = read_results(BASELINE_PATH) if latest else results
        comparison = compare(latest_results, baseline)
        differing = (
            {f.image for f in comparison.flips if f.metric in GATED_METRICS}
            | set(comparison.added_images)
            | set(comparison.removed_images)
        )
        latest_at = f"latest results (at {latest_results.commit})"
        if differing:
            click.echo(
                f"{latest_at} differ from the baseline in {len(differing)} query images"
            )
        else:
            click.echo(f"{latest_at} match the baseline")

    if failures:
        click.echo("\n".join(_failure_lines(results)))


def _failure_lines(results: Results) -> list[str]:
    """Each query image failing exact printing or card count, with its label from
    the manifest, the printings recognised in it and its recognition error.
    """
    failing = {
        image: outcome
        for image, outcome in results.images.items()
        if not (outcome.score.exact_printing and outcome.score.card_count)
    }
    if not failing:
        return ["Failures: none"]

    # the label is in the manifest, not the results, so it is the current label,
    # which may have changed since the run; without the dataset pulled, the
    # failures are still worth listing
    manifest_path = DATASET_ROOT / MANIFEST_NAME
    manifest: dict[str, list[str]] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )

    # one catalog query for every printing shown
    uuids = {uuid for image in failing for uuid in manifest.get(image, [])}
    uuids |= {
        uuid for outcome in failing.values() for uuid in outcome.recognised if uuid
    }
    printings = SqliteCardCatalog(ReferenceData.default().catalog_path).lookup(
        sorted(uuids)
    )

    lines = ["Failures:"]
    for image, outcome in failing.items():
        label = manifest.get(image)
        label_names = (
            _printing_names(label, printings) if label is not None else "unknown"
        )
        recognised_names = _printing_names(outcome.recognised, printings)
        lines += [
            f"  {image}",
            f"    label: {label_names}",
            f"    recognised: {recognised_names}",
        ]
        if outcome.error is not None:
            lines.append(f"    error: {outcome.error}")
    return lines


def _printing_names(uuids: Sequence[str | None], printings: dict[str, Printing]) -> str:
    """`Lightning Bolt (M10), nothing`: the card name and set code of each UUID,
    the bare UUID when the catalog lacks it, and `nothing` for a card without
    candidates.
    """
    names = []
    for uuid in uuids:
        if uuid is None:
            names.append("nothing")
        elif uuid in printings:
            names.append(f"{printings[uuid].name} ({printings[uuid].set_code})")
        else:
            names.append(uuid)
    return ", ".join(names) if names else "nothing"


def _commit() -> str:
    """The commit the results are made at, with `-dirty` when tracked files have
    uncommitted changes, which the results then reflect too.
    """
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return f"{head}-dirty" if status else head


def _comparison_report(
    results: Results, baseline: Results, comparison: Comparison
) -> list[str]:
    """Old vs new: the commits, the flips per image, the totals and the verdict."""
    lines = [
        f"Commit: {baseline.commit} (baseline) -> {results.commit} (now)",
        f"MTGJSON: {baseline.mtgjson_version} (baseline) -> "
        f"{results.mtgjson_version} (now)",
        "",
    ]

    # per-image flips
    for flip in comparison.flips:
        change = "now passes" if flip.passes else "now fails"
        lines.append(f"  {flip.metric} {change}: {flip.image}")
    lines += [f"  new image: {image}" for image in comparison.added_images]
    lines += [f"  removed image: {image}" for image in comparison.removed_images]
    if comparison.flips or comparison.added_images or comparison.removed_images:
        lines.append("")

    # totals: now, the ones that changed, then runtime, which always does
    now = results.totals
    lines.append(f"Now: {totals_line(now)}")
    for metric, (old, new) in comparison.changed_totals.items():
        lines.append(f"{metric}: {old} -> {new}")
    if not comparison.changed_totals:
        lines.append("Totals unchanged.")
    lines.append(
        f"runtime_s: {baseline.totals.runtime_s:.1f} -> {results.totals.runtime_s:.1f}"
    )
    lines.append("")

    if comparison.differs:
        lines.append(
            "Differs from the baseline. To adopt these results: "
            "`uv run mtg benchmark update-baseline`, then commit "
            f"{BASELINE_PATH}"
        )
    else:
        lines.append("Same as the baseline.")
    return lines


def totals_line(totals: Totals) -> str:
    """The pass count of each metric and the OCR fallbacks, as `run` and `status`
    print them.
    """
    return (
        f"exact printing {totals.exact_printing}/{totals.images}, "
        f"card count {totals.card_count}/{totals.images}, "
        f"same card name {totals.same_card_name}/{totals.images}, "
        f"OCR fallbacks {totals.ocr_fallbacks}"
    )
