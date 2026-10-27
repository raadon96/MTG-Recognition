import os
from pathlib import Path

from mtg_recognition.settings import Settings


def test_model_paths_are_built_from_models_dir(monkeypatch):
    # Settings writes U2NET_HOME to os.environ; monkeypatch restores it afterwards
    monkeypatch.setenv("U2NET_HOME", "")

    settings = Settings(MODELS_DIR=Path("elsewhere"))

    assert (
        Path("elsewhere/segmentation/prod/yoloe-11s-seg.pt")
        == settings.YOLO_SEG_MODEL_PATH
    )
    assert (
        Path("elsewhere/ocr/prod/PP-OCRv5_mobile_det")
        == settings.OCR_DETECTION_MODEL_DIR
    )
    assert (
        Path("elsewhere/ocr/prod/PP-OCRv5_mobile_rec")
        == settings.OCR_RECOGNITION_MODEL_DIR
    )
    assert os.environ["U2NET_HOME"] == "elsewhere/segmentation/prod"
