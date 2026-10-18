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
    JPEG_QUALITY: int = 90

    # --- Data paths ---
    DATA_DIR: Path = Path("data")
    # DATA_DIR: Path = Path("scratch/test_download")
    MTGJSON_URL: str = "https://mtgjson.com/api/v5/AllPrintings.sqlite.zip"

    # --- Third-party env vars (pushed to os.environ on instantiation) ---
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK: bool = True
    U2NET_HOME: str = "src/mtg_recognition/engine/segmentation/models/prod"

    # --- Derived paths ---
    @property
    def YOLO_SEG_MODEL_PATH(self) -> Path:
        return Path(
            "src/mtg_recognition/engine/segmentation/models/prod/yoloe-11s-seg.pt"
        )

    @property
    def REF_DIR(self) -> Path:
        return self.DATA_DIR / "reference"
        # return self.DATA_DIR / "reference_test"

    @property
    def LOCAL_DB_FILE(self) -> Path:
        return self.REF_DIR / "MobileCache.sqlite"

    @property
    def TMP_DIR(self) -> Path:
        return self.REF_DIR / "tmp"

    @property
    def MTGO_DB_FILE_ZIP(self) -> Path:
        return self.TMP_DIR / "AllPrintings.sqlite.zip"

    @property
    def MTGO_DB_FILE(self) -> Path:
        return self.TMP_DIR / "AllPrintings.sqlite"

    @property
    def IMG_DIR(self) -> Path:
        return self.REF_DIR / "images"

    @property
    def LOG_FILE(self) -> Path:
        return self.REF_DIR / "download_errors.log"

    @property
    def PHASH_INDEX_DIR(self) -> Path:
        return self.REF_DIR / "index" / "phash"

    @property
    def PHASH_INDEX_FILE(self) -> Path:
        return self.PHASH_INDEX_DIR / "hashes.npy"

    @property
    def PHASH_MAPPING_FILE(self) -> Path:
        return self.PHASH_INDEX_DIR / "index_to_uuid.json"

    @property
    def STATE_FILE(self) -> Path:
        """State file that tracks the MTGJSON version last used to build each artifact"""
        return self.REF_DIR / ".mtg-sync-state.json"

    def ensure_dirs(self) -> None:
        self.IMG_DIR.mkdir(parents=True, exist_ok=True)
        self.TMP_DIR.mkdir(parents=True, exist_ok=True)
        self.PHASH_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    @model_validator(mode="after")
    def _apply_third_party_env_vars(self) -> "Settings":
        """Push third-party vars to os.environ so libraries read them at import time."""
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = str(
            self.PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK
        )
        os.environ["U2NET_HOME"] = self.U2NET_HOME
        return self


settings = Settings()
settings.ensure_dirs()
