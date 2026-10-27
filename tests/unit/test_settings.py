import os
from pathlib import Path

from mtg_recognition.settings import Settings


def test_rembg_finds_its_models_under_models_dir(monkeypatch):
    # Settings writes U2NET_HOME to os.environ; monkeypatch restores it afterwards
    monkeypatch.setenv("U2NET_HOME", "")

    Settings(MODELS_DIR=Path("elsewhere"))

    assert os.environ["U2NET_HOME"] == "elsewhere/segmentation/prod"
