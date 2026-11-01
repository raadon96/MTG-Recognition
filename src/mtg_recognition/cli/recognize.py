"""`mtg recognize`: rank the printings of every card in the input images."""

import json
import uuid as _uuid
from pathlib import Path

import click

from mtg_recognition.cli.segment import input_images
from mtg_recognition.engine.pipeline import Pipeline


@click.command(name="recognize")
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=True, readable=True, path_type=Path
    ),
    required=True,
    help="Path to a single image file or a directory of image files.",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to a JSON file where all results are written as a list. "
    "When omitted, results are printed to the terminal only.",
)
@click.option(
    "--k-phash",
    type=click.INT,
    default=20,
    show_default=True,
    help="Number of candidates per card: printings shortlisted by pHash "
    "similarity, each scored against the card's reading.",
)
@click.option(
    "-k",
    type=click.INT,
    default=1,
    show_default=True,
    help="Number of printings in each card's ranking.",
)
def recognize(input_path: Path, output_file: Path | None, k_phash: int, k: int) -> None:
    """Run the full MTG card recognition pipeline on one image or a directory.

    Always segments the input image (the segmentor's bypass path handles
    pre-cropped scans without running any model). Results are printed
    incrementally. Pass -o to also save all results to a single JSON file.

    Examples:

    \b
    uv run mtg recognize -i card.jpg -o runs/output.json
    uv run mtg recognize \\
        -i data/benchmark/6_photos_single_card_textured_bg -k 3 --k-phash 20
    uv run mtg recognize \\
        -i data/benchmark/1_photos_single_card -o output/results.json
    """
    img_paths = input_images(input_path)
    if not img_paths:
        click.echo(f"No image files found in: {input_path}", err=True)
        raise SystemExit(1)

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(k_phash_candidates=k_phash, k_final_results=k)
    total = len(img_paths)
    all_results: list[dict] = []

    for idx, img_path in enumerate(img_paths, 1):
        img_abs = str(img_path.resolve())
        try:
            img_display = str(img_path.resolve().relative_to(Path.cwd()))
        except ValueError:
            img_display = img_abs

        recognition = pipeline.recognize(img_path, debug=True)
        if recognition.error is not None:
            click.echo(
                f"[{idx}/{total}] {img_display}  ERROR: {recognition.error}", err=True
            )
        segment_results = recognition.segments

        stem_uuid = _stem_uuid(img_path.stem)
        n_segs = len(segment_results)
        if recognition.error is None:
            click.echo(f"[{idx}/{total}] {img_display}  ({n_segs} segment(s) detected)")

        segments_json: list[dict] = []
        for seg_idx, seg in enumerate(segment_results):
            # UUID-based success check and status label (only meaningful for
            # single-card images)
            if stem_uuid is None:
                success: bool | None = None
                status = "—"
            elif seg.ranked and seg.ranked[0].printing.uuid == stem_uuid:
                success = True
                status = "✓ MATCH"
            else:
                success = False
                got = seg.ranked[0].printing.uuid if seg.ranked else "no match"
                status = f"✗ MISS  (expected: {stem_uuid[:8]}…, got: {got[:8]}…)"

            # Raw OCR texts and scores
            raw_texts: list[str] = []
            raw_scores: list[float] = []
            raw_boxes: list[tuple[int, int, int, int]] = []
            reading_debug = seg.reading.debug if seg.reading is not None else None
            if reading_debug is not None:
                raw_texts = reading_debug.texts
                raw_scores = reading_debug.scores
                raw_boxes = reading_debug.boxes

            # Terminal output
            prefix = f"  [seg {seg_idx}]" if n_segs > 1 else " "
            click.echo(f"{prefix} {status}")
            if seg.ocr_error is not None:
                click.echo(f"{prefix}  OCR failed, ranked by pHash: {seg.ocr_error}")
            if not seg.ranked:
                click.echo(f"{prefix}  no matches found")
            for rank, entry in enumerate(seg.ranked, 1):
                p = entry.printing
                click.echo(
                    f"{prefix}  #{rank}  {p.name}  |  {p.set_code} · {p.number}  |  "
                    f"score: {entry.score:.1f}"
                )
            click.echo(f"{prefix}  raw_texts:  {raw_texts}")
            click.echo(f"{prefix}  raw_scores: {raw_scores}")

            if output_file is not None:
                out = {
                    "segment_idx": seg_idx,
                    "success": success,
                    "ranked": [
                        {
                            "rank": rank,
                            "name": entry.printing.name,
                            "uuid": entry.printing.uuid,
                            "setCode": entry.printing.set_code,
                            "number": entry.printing.number,
                            "score": round(entry.score, 1),
                        }
                        for rank, entry in enumerate(seg.ranked, 1)
                    ],
                    "ocr_error": seg.ocr_error,
                    "debug": {
                        "raw_texts": raw_texts,
                        "raw_scores": raw_scores,
                        "raw_boxes": raw_boxes,
                    },
                }
                segments_json.append(out)

        click.echo("")

        if output_file is not None:
            all_results.append({"image": img_abs, "segments": segments_json})

    if output_file is not None:
        output_file.write_text(json.dumps(all_results, indent=2))
        click.echo(f"Results written to {output_file}")


def _stem_uuid(stem: str) -> str | None:
    """Return stem if it is a valid UUID string, else None."""
    try:
        _uuid.UUID(stem)
        return stem
    except ValueError:
        return None
