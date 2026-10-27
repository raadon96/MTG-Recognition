import os
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # --- Upload / preprocessing ---
    IMG_EXT: list[str] = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    MAX_UPLOAD_SIZE_MB: int = 25
    MAX_IMAGE_DIMENSION: int = 2048  # Constrain longest edge

    # --- Data paths ---
    DATA_DIR: Path = Path("data")
    # DATA_DIR: Path = Path("scratch/test_download")
    MTGJSON_URL: str = "https://mtgjson.com/api/v5/AllPrintings.sqlite.zip"
    MODELS_DIR: Path = Path("models")

    # --- Third-party env vars (pushed to os.environ on instantiation) ---
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK: bool = True

    # --- Derived paths (reference data paths live in reference.ReferenceData) ---
    @property
    def SEGMENTATION_MODEL_DIR(self) -> Path:
        return self.MODELS_DIR / "segmentation" / "prod"

    @property
    def YOLO_SEG_MODEL_PATH(self) -> Path:
        return self.SEGMENTATION_MODEL_DIR / "yoloe-11s-seg.pt"

    @property
    def OCR_DETECTION_MODEL_DIR(self) -> Path:
        return self.MODELS_DIR / "ocr" / "prod" / "PP-OCRv5_mobile_det"

    @property
    def OCR_RECOGNITION_MODEL_DIR(self) -> Path:
        return self.MODELS_DIR / "ocr" / "prod" / "PP-OCRv5_mobile_rec"

    @model_validator(mode="after")
    def _apply_third_party_env_vars(self) -> "Settings":
        """Push third-party vars to os.environ so libraries read them at import time."""
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = str(
            self.PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK
        )
        # rembg looks for its model files in U2NET_HOME
        os.environ["U2NET_HOME"] = str(self.SEGMENTATION_MODEL_DIR)
        return self


settings = Settings()
