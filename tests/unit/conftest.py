import json
from pathlib import Path

import numpy as np
import pytest

from mtg_recognition.core.image_utils import load_and_resize_image
from mtg_recognition.engine.segmentation.rembg import RembgCardSegmentor

FIXTURES_DIR = Path("tests/unit/fixtures")
SEG_FIXTURE_DIR = FIXTURES_DIR / "seg"

SINGLE_CARD_DIR = SEG_FIXTURE_DIR / "1_imgs_single_card"
MULTI_CARD_DIR = SEG_FIXTURE_DIR / "2_imgs_multiple_cards"
FAILURE_CASES_DIR = SEG_FIXTURE_DIR / "0_failure_cases"

BENCHMARK_RAW_DIR = Path("benchmarks/dataset/images/raw")


@pytest.fixture(scope="session")
def segmentor() -> RembgCardSegmentor:
    """Session-scoped RembgCardSegmentor; model is loaded once for the whole suite."""
    return RembgCardSegmentor()


@pytest.fixture(scope="session")
def single_corners_golden() -> dict:
    """Golden corner coordinates for single-card fixtures, keyed by image stem."""
    path = SINGLE_CARD_DIR / "corners.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def multi_corners_golden() -> dict:
    """Golden per-blob corner coordinates for multi-card fixtures, keyed by image stem."""
    path = MULTI_CARD_DIR / "corners.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def valid_card_image() -> np.ndarray:
    return load_and_resize_image(FIXTURES_DIR / "images" / "valid_card.jpg")


@pytest.fixture(scope="session")
def no_cards_image() -> np.ndarray:
    return load_and_resize_image(FIXTURES_DIR / "images" / "no_cards.jpg")
