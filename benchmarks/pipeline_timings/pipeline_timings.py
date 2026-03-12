from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from mtg_recognition.engine.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark pipeline timings over multiple runs and export a CSV report."
        )
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=100,
        help="Number of complete runs over the dataset (default: 100).",
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=Path("benchmarks/dataset/images/raw"),
        help="Directory containing input images (default: benchmarks/dataset/images/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/pipeline_timings"),
        help="Directory where CSV report is written (default: benchmarks/pipeline_timings).",
    )
    return parser.parse_args()


def load_images(img_dir: Path) -> list[tuple[str, np.ndarray]]:
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    image_paths = sorted(img_dir.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No .jpg images found in: {img_dir}")

    loaded_images: list[tuple[str, np.ndarray]] = []
    for image_path in tqdm(image_paths):
        assert image_path.exists()
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        loaded_images.append((image_path.name, image))

    if not loaded_images:
        raise ValueError(f"No readable images found in: {img_dir}")

    return loaded_images


def build_report_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    run_summary_df = (
        raw_df.groupby("run_index")
        .agg(
            image_count=("image_name", "count"),
            segmentation_avg_ms=("segmentation_ms", "mean"),
            segmentation_p95_ms=("segmentation_ms", lambda s: s.quantile(0.95)),
            phash_avg_ms=("phash_ms", "mean"),
            phash_p95_ms=("phash_ms", lambda s: s.quantile(0.95)),
            ocr_avg_ms=("ocr_ms", "mean"),
            ocr_p95_ms=("ocr_ms", lambda s: s.quantile(0.95)),
            total_avg_ms=("total_ms", "mean"),
            total_p95_ms=("total_ms", lambda s: s.quantile(0.95)),
        )
        .reset_index()
    )
    run_summary_df.insert(0, "scope", "run")

    aggregate_row = {
        "scope": "aggregate",
        "run_index": "all",
        "image_count": int(raw_df["image_name"].count()),
        "segmentation_avg_ms": raw_df["segmentation_ms"].mean(),
        "segmentation_p95_ms": raw_df["segmentation_ms"].quantile(0.95),
        "phash_avg_ms": raw_df["phash_ms"].mean(),
        "phash_p95_ms": raw_df["phash_ms"].quantile(0.95),
        "ocr_avg_ms": raw_df["ocr_ms"].mean(),
        "ocr_p95_ms": raw_df["ocr_ms"].quantile(0.95),
        "total_avg_ms": raw_df["total_ms"].mean(),
        "total_p95_ms": raw_df["total_ms"].quantile(0.95),
    }

    report_df = pd.concat(
        [run_summary_df, pd.DataFrame([aggregate_row])], ignore_index=True
    )

    numeric_cols = [
        "segmentation_avg_ms",
        "segmentation_p95_ms",
        "phash_avg_ms",
        "phash_p95_ms",
        "ocr_avg_ms",
        "ocr_p95_ms",
        "total_avg_ms",
        "total_p95_ms",
    ]
    report_df[numeric_cols] = report_df[numeric_cols].round(4)

    return report_df


def write_report_csv(output_path: Path, report_df: pd.DataFrame) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()

    if args.num_runs <= 0:
        raise ValueError("--num-runs must be > 0")

    images = load_images(args.img_dir)
    image_count = len(images)

    pipeline = Pipeline(auto_build=True)

    rows: list[dict[str, float | int | str]] = []

    for run_index in tqdm(range(1, args.num_runs + 1)):
        for image_name, image in images:
            _, _, diagnostics = pipeline.find_best_matches_with_diagnostics(
                image,
                k_phash_candidates=100,
                k_final_results=1,
                apply_segmentation=True,
            )
            rows.append(
                {
                    "run_index": run_index,
                    "image_name": image_name,
                    "segmentation_ms": diagnostics["segmentation_ms"],
                    "phash_ms": diagnostics["phash_ms"],
                    "ocr_ms": diagnostics["ocr_ms"],
                    "total_ms": diagnostics["total_ms"],
                }
            )

        print(f"Completed run {run_index}/{args.num_runs} ({image_count} images)")

    raw_df = pd.DataFrame(rows)
    report_df = build_report_dataframe(raw_df)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"pipeline_timing_report_{timestamp}.csv"
    write_report_csv(output_path, report_df)

    print(f"Timing report written to: {output_path}")


if __name__ == "__main__":
    main()

# uv run python -m benchmarks.pipeline_timings.pipeline_timings --num-runs 2 --output-dir benchmarks/pipeline_timings/results2
