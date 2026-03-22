import json
from pathlib import Path

import numpy as np
import pytest
from sqlmodel import Session, col, create_engine, select

from mtg_recognition.core.image_utils import load_and_resize_image
from mtg_recognition.engine.segmentation.segment import CardSegmentor
from mtg_recognition.jobs.schemas import MobileCache

BEST_SEGMENTOR = "yoloe-11s-seg"
BEST_PARAM_SET = 0

FIXTURES_DIR = Path("tests/unit/fixtures")
IMG_DIR = FIXTURES_DIR / "images"


@pytest.fixture(scope="session")
def segmentor() -> CardSegmentor:
    """Create CardSegmentor instance."""
    return CardSegmentor(model_name=BEST_SEGMENTOR, param_set_idx=BEST_PARAM_SET)


@pytest.fixture(scope="session")
def valid_card_image() -> np.ndarray:
    """Load test image with a single valid card."""
    return load_and_resize_image(IMG_DIR / "valid_card.jpg")


@pytest.fixture(scope="session")
def no_cards_image() -> np.ndarray:
    """Load test image with that does not contain any cards."""
    return load_and_resize_image(IMG_DIR / "no_cards.jpg")


@pytest.fixture(scope="session")
def high_contrast_card_image() -> np.ndarray:
    """Load test image with one valid card with difficult lighting conditions."""
    return load_and_resize_image(IMG_DIR / "high_contrast_card.jpg")


@pytest.fixture(scope="session")
def two_cards_image() -> np.ndarray:
    """Load test image with two cards."""
    return load_and_resize_image(IMG_DIR / "two_cards.png")


@pytest.fixture(scope="session")
def setcodes_ref() -> list[str]:
    "Load the available mtg setcodes from either cached file or database"
    engine = create_engine("sqlite:///data/reference/MobileCache.sqlite")
    with Session(engine) as session:
        statement = select(MobileCache.setCode).distinct()
        rows = session.exec(statement).all()
        # results = [row for row in rows]
        return list(rows)
