from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    MAX_UPLOAD_SIZE_MB: int = 25
    MAX_IMAGE_DIMENSION: int = 1280  # Constrain longest edge (720p)
    JPEG_QUALITY: int = 90

    LOCAL_DB_FILE: str = "data/reference/MobileCache.sqlite"


app_settings = AppSettings()
