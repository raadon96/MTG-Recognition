"""`mtg segment`: write each card found in the query images as a card image.

Usage examples:
    uv run mtg segment -i data/benchmark/1_photos_single_card \
        -o output/1.0_seg/1_photos_single_card --mode auto

    uv run mtg segment -i photo.jpg -o output/1.0_seg --mode yolo
"""

from pathlib import Path

import click
import cv2
from tqdm import tqdm

from mtg_recognition.engine.image import prepare_image
from mtg_recognition.settings import settings


@click.command(name="segment")
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    required=True,
    help="Path to a single query image or a directory of query images.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory where card images are written.",
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
def segment(input_path: Path, output_dir: Path, mode: str) -> None:
    segmentor = _build_segmentor(mode)

    img_paths = query_image_paths(input_path)
    if not img_paths:
        click.echo(f"No image files found in: {input_path}", err=True)
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for img_path in tqdm(img_paths, desc="Segmenting"):
        try:
            img = prepare_image(img_path)
        except ValueError:
            click.echo(f"  unreadable query image, skipped: {img_path.name}", err=True)
            continue
        result = segmentor.segment(img)
        if not result.success:
            click.echo(f"  no card detected: {img_path.name}", err=True)
            continue
        for idx, card in enumerate(result.cards):
            stem = img_path.stem if len(result.cards) == 1 else f"{img_path.stem}_{idx}"
            cv2.imwrite(str(output_dir / f"{stem}.jpg"), card.card_image)
            saved += 1

    click.echo(f"Saved {saved} card image(s) to {output_dir}")


def query_image_paths(input_path: Path) -> list[Path]:
    """The file itself, or the images in the directory matching settings.IMG_EXT,
    sorted.
    """
    if input_path.is_file():
        return [input_path]
    img_paths: list[Path] = []
    for ext in settings.IMG_EXT:
        img_paths.extend(input_path.glob(ext))
    return sorted(img_paths)


def _build_segmentor(mode: str):
    # the models load only for the mode asked for
    mode = mode.lower()
    if mode == "rembg":
        from mtg_recognition.engine.segmentation.rembg import RembgSegmentor

        return RembgSegmentor()
    from mtg_recognition.engine.segmentation.yolo import YoloSegmentor

    return YoloSegmentor(fallback_enabled=mode == "auto")
