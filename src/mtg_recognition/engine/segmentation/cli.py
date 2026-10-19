"""
CLI for card segmentation.

Usage examples:
    uv run python -m mtg_recognition.engine.segmentation.cli \
        -i data/test/1_photos_single_card/images \
        -o output/1.0_seg/1_photos_single_card --mode auto

    uv run python -m mtg_recognition.engine.segmentation.cli \
        -i data/test/2_photos_multiple_cards/images \
        -o output/1.0_seg/2_photos_multiple_cards --mode auto

    uv run python -m mtg_recognition.engine.segmentation.cli \
        -i data/test/3_scans_single_card_with_bg/images \
        -o output/1.0_seg/3_scans_single_card_with_bg --mode auto

    uv run python -m mtg_recognition.engine.segmentation.cli \
        -i data/test/4_scans_single_card_no_bg/images \
        -o output/1.0_seg/4_scans_single_card_no_bg --mode auto

    uv run python -m mtg_recognition.engine.segmentation.cli \
        -i data/test/5_scans_multiple_cards/images \
        -o output/1.0_seg/5_scans_multiple_cards --mode auto


    uv run python -m mtg_recognition.engine.segmentation.cli \
        -i photo.jpg -o output/1.0_seg --mode yolo
"""

from __future__ import annotations

from pathlib import Path

import click
import cv2
from tqdm import tqdm

from mtg_recognition.core import image_utils


@click.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    required=True,
    help="Path to a single image file or a directory of image files.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory where segmented card images are written.",
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["auto", "yolo", "rembg"], case_sensitive=False),
    default="auto",
    show_default=True,
    help=(
        "Segmentation mode. "
        "'auto'  — YOLO primary with CV+rembg fallback (recommended). "
        "'yolo'  — YOLO only, no fallback. "
        "'rembg' — rembg background removal only."
    ),
)
def main(input_path: Path, output_dir: Path, mode: str) -> None:
    segmentor = _build_segmentor(mode)

    img_paths = (
        [input_path]
        if input_path.is_file()
        else image_utils.get_image_files_in_directory(input_path)
    )

    if not img_paths:
        click.echo(f"No image files found in: {input_path}", err=True)
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for img_path in tqdm(img_paths, desc="Segmenting"):
        # img = image_utils.load_and_resize_image(img_path)
        img = cv2.imread(img_path)
        result = segmentor.segment(img)
        if not result.success:
            click.echo(f"  no card detected: {img_path.name}", err=True)
            continue
        for idx, segment in enumerate(result.segments):
            stem = (
                img_path.stem if len(result.segments) == 1 else f"{img_path.stem}_{idx}"
            )
            cv2.imwrite(str(output_dir / f"{stem}.jpg"), segment.warped_image)
            saved += 1

    click.echo(f"Saved {saved} segment(s) to {output_dir}")


def _build_segmentor(mode: str):
    mode = mode.lower()
    if mode == "rembg":
        from .rembg import RembgCardSegmentor

        return RembgCardSegmentor()
    elif mode == "yolo":
        from .yolo import YoloCardSegmentor

        return YoloCardSegmentor(fallback_enabled=False)
    else:  # auto
        from .yolo import YoloCardSegmentor

        return YoloCardSegmentor(fallback_enabled=True)


if __name__ == "__main__":
    main()
