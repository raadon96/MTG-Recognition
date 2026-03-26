from pathlib import Path

import numpy as np
import pytest

from mtg_recognition.core.image_utils import load_and_resize_image

FIXTURES_DIR = Path("tests/unit/fixtures")


@pytest.fixture(scope="session")
def valid_card_image() -> np.ndarray:
    return load_and_resize_image(FIXTURES_DIR / "images" / "valid_card.jpg")


@pytest.fixture(scope="session")
def no_cards_image() -> np.ndarray:
    return load_and_resize_image(FIXTURES_DIR / "images" / "no_cards.jpg")
