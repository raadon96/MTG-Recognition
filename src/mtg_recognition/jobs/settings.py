from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Default: Current directory. Can be overridden by env var APP_DATA_DIR
    # DATA_DIR: Path = Path(__file__).parents[3].resolve()
    DATA_DIR: Path = Path("data")

    # These automatically nest or you can define them relative to DATA_DIR in __init__
    MTGJSON_URL: str = "https://mtgjson.com/api/v5/AllPrintings.sqlite.zip"

    _REF_DIR: Path = DATA_DIR / "reference"

    @property
    def REF_DIR(self) -> Path:
        # data/reference
        return self._REF_DIR

    @REF_DIR.setter
    def REF_DIR(self, value: Path | str):
        # Allow setting REF_DIR to a different path (for testing, etc.)
        self._REF_DIR = Path(value)
        self.ensure_dirs()

    @property
    def LOCAL_DB_FILE(self) -> Path:
        # data/reference/MobileCache.sqlite
        return self.REF_DIR / "MobileCache.sqlite"

    @property
    def TMP_DIR(self) -> Path:
        # data/reference/tmp
        return self.REF_DIR / "tmp"

    @property
    def MTGO_DB_FILE_ZIP(self) -> Path:
        # data/reference/tmp/AllPrintings.sqlite.zip
        return self.TMP_DIR / "AllPrintings.sqlite.zip"

    @property
    def MTGO_DB_FILE(self) -> Path:
        # data/reference/tmp/AllPrintings.sqlite
        return self.TMP_DIR / "AllPrintings.sqlite"

    @property
    def IMG_DIR(self) -> Path:
        # data/reference/images
        return self.REF_DIR / "images"

    @property
    def LOG_FILE(self) -> Path:
        # data/reference/download_errors.log
        return self.REF_DIR / "download_errors.log"

    # Creates the directories automatically when settings are loaded
    def ensure_dirs(self):
        self.IMG_DIR.mkdir(parents=True, exist_ok=True)
        self.TMP_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
