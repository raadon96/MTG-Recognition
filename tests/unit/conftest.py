from pathlib import Path

import numpy as np
import pytest

from mtg_recognition.core.image_utils import prepare_image

from .sync_fakes import FakeNetwork

FIXTURES_DIR = Path("tests/unit/fixtures")


@pytest.fixture(scope="session")
def valid_card_image() -> np.ndarray:
    return prepare_image(FIXTURES_DIR / "images" / "valid_card.jpg")


@pytest.fixture(scope="session")
def no_cards_image() -> np.ndarray:
    return prepare_image(FIXTURES_DIR / "images" / "no_cards.jpg")


@pytest.fixture
def network(tmp_path) -> FakeNetwork:
    """The fake MTGJSON and Scryfall of `sync_fakes`, working under `tmp_path`."""
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    return FakeNetwork(fake_dir)
