import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlmodel import Session, col, create_engine, select

from mtg_recognition.engine.segmentation.segment import CardSegmentor
from mtg_recognition.jobs.schemas import MobileCache

BEST_SEGMENTOR = "yoloe-11s-seg"
BEST_PARAM_SET = 0

FIXTURES_DIR = Path(__file__).parent / "fixtures"
IMG_DIR = FIXTURES_DIR / "images"

REF_SETCODES_PATH = "data/reference/setcodes.json"


@pytest.fixture(scope="session")
def segmentor() -> CardSegmentor:
    """Create CardSegmentor instance."""
    return CardSegmentor(model_name=BEST_SEGMENTOR, param_set_idx=BEST_PARAM_SET)


@pytest.fixture(scope="session")
def valid_card_image() -> np.ndarray:
    """Load test image with a single valid card."""
    img_path = str(IMG_DIR / "valid_card.jpg")
    return cv2.imread(img_path)


@pytest.fixture(scope="session")
def no_cards_image() -> np.ndarray:
    """Load test image with that does not contain any cards."""
    img_path = str(IMG_DIR / "no_cards.jpg")
    return cv2.imread(img_path)


@pytest.fixture(scope="session")
def high_contrast_card_image() -> np.ndarray:
    """Load test image with one valid card with difficult lighting conditions."""
    img_path = str(IMG_DIR / "high_contrast_card.jpg")
    return cv2.imread(img_path)


@pytest.fixture(scope="session")
def setcodes_ref() -> list[str]:
    "Load the available mtg setcodes from either cached file or database"
    # with open(REF_SETCODES_PATH) as file:
    #     setcodes_ref = json.load(file)
    #     return setcodes_ref

    engine = create_engine("sqlite:///data/reference/MobileCache.sqlite")
    with Session(engine) as session:
        statement = select(MobileCache.setCode).distinct()
        rows = session.exec(statement).all()
        # results = [row for row in rows]
        return list(rows)
