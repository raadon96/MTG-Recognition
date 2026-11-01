"""`mtg benchmark run` and `mtg benchmark update-baseline`: measure recognition on
the labelled query images and compare it with the baseline.
"""

import json
import shutil
import subprocess
import time

import click

from mtg_recognition.benchmark.results import (
    BASELINE_PATH,
    DATASET_ROOT,
    LATEST_PATH,
    MANIFEST_NAME,
    Comparison,
    ImageOutcome,
    Results,
    compare,
    read_results,
    score_image,
    summarise,
    validate_manifest,
    write_results,
)
from mtg_recognition.engine.pipeline import Pipeline
from mtg_recognition.reference.catalog import SqliteCardCatalog
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
        click.echo(f"No manifest at {manifest_path}; run `uv run dvc pull`", err=True)
        raise SystemExit(1)
    manifest: dict[str, list[str]] = json.loads(manifest_path.read_text())
    reference = ReferenceData.default()
    labelled_uuids = sorted({uuid for label in manifest.values() for uuid in label})
    catalog = SqliteCardCatalog(reference.catalog_path).lookup(labelled_uuids)
    problems = validate_manifest(DATASET_ROOT, manifest, catalog)
    for heading, items in [
        ("Images without a label", problems.unlabelled_images),
        ("Labels for missing files", problems.missing_images),
        ("UUIDs not in the catalog", problems.unknown_uuids),
    ]:
        if items:
            click.echo(f"{heading}:\n" + "\n".join(f"  {i}" for i in items), err=True)
    if problems.unlabelled_images or problems.missing_images or problems.unknown_uuids:
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
        score = score_image([catalog[uuid] for uuid in label], best)
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
    click.echo("\n".join(comparison_report(results, baseline, comparison)))
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


def comparison_report(
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

    # totals
    old, new = baseline.totals, results.totals
    lines.append(f"{'':<16}{'baseline':>10}{'now':>10}")
    for metric in ("exact_printing", "card_count", "same_card_name"):
        lines.append(
            f"{metric:<16}{getattr(old, metric):>6}/{old.images:<3}"
            f"{getattr(new, metric):>6}/{new.images:<3}"
        )
    lines.append(f"{'ocr_fallbacks':<16}{old.ocr_fallbacks:>10}{new.ocr_fallbacks:>10}")
    lines.append(f"{'runtime_s':<16}{old.runtime_s:>10.1f}{new.runtime_s:>10.1f}")
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
