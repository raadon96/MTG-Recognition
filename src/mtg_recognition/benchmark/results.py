"""Benchmark results as plain data: validating the manifest, scoring each query
image against its label, and comparing results with the baseline. No models: the
`mtg benchmark` command runs recognition and passes the printings in.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mtg_recognition.reference.catalog import Printing

# relative to the repo root, not settings.DATA_DIR: data/benchmark.dvc fixes where
# the dataset is pulled to, and the baseline is tracked in git beside it, so each
# PR commits the baseline it is measured against
DATASET_ROOT = Path("data/benchmark")
MANIFEST_NAME = "manifest.json"
BASELINE_PATH = Path("data/benchmark-baseline.json")
LATEST_PATH = Path("output/benchmark/latest.json")


@dataclass(slots=True)
class ImageScore:
    """How the printings recognised in one query image compare with its label."""

    # the recognised printings are the labelled ones, in any order
    exact_printing: bool
    # as many cards were recognised as the label holds
    card_count: bool
    # the recognised card names are the labelled ones, in any order: true for
    # exact printings too, and for another printing or the other face of the
    # right card (both faces of a double-faced card share one name, "A // B")
    same_card_name: bool


@dataclass(slots=True)
class ManifestProblems:
    """What keeps the manifest from matching the dataset and the catalog. Paths
    are relative to the dataset root.
    """

    unlabelled_images: list[str] = field(default_factory=list)
    missing_images: list[str] = field(default_factory=list)
    unknown_uuids: list[str] = field(default_factory=list)


def validate_manifest(
    dataset_root: Path,
    manifest: Mapping[str, list[str]],
    labelled_printings: Mapping[str, Printing],
) -> ManifestProblems:
    """Check the manifest against the files under `dataset_root` and against
    `labelled_printings`, what the catalog holds for the labelled UUIDs.
    """
    images = {
        path.relative_to(dataset_root).as_posix()
        for path in dataset_root.rglob("*")
        # hidden files (.DS_Store and the like) are not query images
        if path.is_file()
        and path.name != MANIFEST_NAME
        and not path.name.startswith(".")
    }
    labelled_uuids = {uuid for label in manifest.values() for uuid in label}
    return ManifestProblems(
        unlabelled_images=sorted(images - manifest.keys()),
        missing_images=sorted(manifest.keys() - images),
        unknown_uuids=sorted(labelled_uuids - labelled_printings.keys()),
    )


def score_image(
    label: Sequence[Printing], recognised: Sequence[Printing | None]
) -> ImageScore:
    """Score the best printing of each card found in a query image (`None` for a
    card without candidates) against the printings of its label.
    """
    # labels and results are compared as unordered multisets: the manifest has
    # no order convention, and a query image can hold the same printing twice
    return ImageScore(
        exact_printing=Counter(p.uuid for p in label)
        == Counter(p.uuid if p is not None else None for p in recognised),
        card_count=len(recognised) == len(label),
        same_card_name=Counter(p.name for p in label)
        == Counter(p.name if p is not None else None for p in recognised),
    )


@dataclass(slots=True)
class ImageOutcome:
    """What the benchmark records for one query image."""

    score: ImageScore
    recognised: list[str | None]  # the best printing's UUID of each card found
    runtime_s: float
    ocr_fallbacks: int  # cards ranked by pHash alone because OCR failed
    error: str | None = None  # recognition failed on the whole query image


@dataclass(slots=True)
class Totals:
    """Aggregate metrics: the number of query images that pass each one, and the
    sums of the reported runtimes and OCR fallbacks.
    """

    images: int
    exact_printing: int  # gated
    card_count: int  # gated
    same_card_name: int
    ocr_fallbacks: int
    runtime_s: float


@dataclass(slots=True)
class Results:
    """A benchmark run, as written to the latest results and the baseline."""

    commit: str
    mtgjson_version: str | None  # the catalog's, from the reference data
    totals: Totals
    images: dict[str, ImageOutcome]  # by path relative to the dataset root


# a difference in a gated metric fails `mtg benchmark run`; the others are
# reported only
GATED_METRICS = ("exact_printing", "card_count")
REPORTED_METRICS = ("same_card_name",)


def summarise(
    commit: str, mtgjson_version: str | None, images: dict[str, ImageOutcome]
) -> Results:
    """The results of a run, with their totals."""
    outcomes = images.values()
    totals = Totals(
        images=len(images),
        exact_printing=sum(o.score.exact_printing for o in outcomes),
        card_count=sum(o.score.card_count for o in outcomes),
        same_card_name=sum(o.score.same_card_name for o in outcomes),
        ocr_fallbacks=sum(o.ocr_fallbacks for o in outcomes),
        runtime_s=round(sum(o.runtime_s for o in outcomes), 3),
    )
    return Results(commit, mtgjson_version, totals, dict(sorted(images.items())))


@dataclass(slots=True)
class Flip:
    """A metric of one query image that passes in one run and fails in the
    other.
    """

    image: str
    metric: str
    passes: bool  # in the new results; the baseline had the opposite


@dataclass(slots=True)
class Comparison:
    flips: list[Flip]
    added_images: list[str]  # in the results, not the baseline
    removed_images: list[str]  # in the baseline, not the results
    # (baseline, results) of each total that changed, runtime aside: it always does
    changed_totals: dict[str, tuple[int, int]]
    # a gated metric flipped, or the query images changed
    differs: bool


def compare(results: Results, baseline: Results) -> Comparison:
    """Compare new results with the baseline, image by image."""
    # flips in the query images both runs scored
    flips = []
    for image, now in results.images.items():
        before = baseline.images.get(image)
        if before is None:
            continue
        for metric in GATED_METRICS + REPORTED_METRICS:
            passes = getattr(now.score, metric)
            if passes != getattr(before.score, metric):
                flips.append(Flip(image, metric, passes))

    # totals that changed
    changed_totals = {}
    for metric in ("images", *GATED_METRICS, *REPORTED_METRICS, "ocr_fallbacks"):
        old = getattr(baseline.totals, metric)
        new = getattr(results.totals, metric)
        if old != new:
            changed_totals[metric] = (old, new)

    added = sorted(results.images.keys() - baseline.images.keys())
    removed = sorted(baseline.images.keys() - results.images.keys())
    gated_flip = any(flip.metric in GATED_METRICS for flip in flips)
    return Comparison(
        flips=flips,
        added_images=added,
        removed_images=removed,
        changed_totals=changed_totals,
        differs=gated_flip or bool(added) or bool(removed),
    )


def write_results(results: Results, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(results), indent=2) + "\n")


def read_results(path: Path) -> Results:
    data = json.loads(path.read_text())
    images = {
        image: ImageOutcome(
            score=ImageScore(**outcome["score"]),
            recognised=outcome["recognised"],
            runtime_s=outcome["runtime_s"],
            ocr_fallbacks=outcome["ocr_fallbacks"],
            error=outcome["error"],
        )
        for image, outcome in data["images"].items()
    }
    return Results(
        commit=data["commit"],
        mtgjson_version=data["mtgjson_version"],
        totals=Totals(**data["totals"]),
        images=images,
    )
