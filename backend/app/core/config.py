from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    PROJECT_NAME: str = "RepoMind"
    API_VERSION: str = "v1"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    REPOSITORIES_DIR: Path = DATA_DIR / "repos"
    TEMP_DIR: Path = DATA_DIR / "tmp"
    MAX_UPLOAD_SIZE_MB: int = 50
    MAX_FILE_SIZE_MB: int = 1
    MAX_REPOSITORY_FILES: int = 10_000
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )


settings = Settings()
settings.REPOSITORIES_DIR.mkdir(parents=True, exist_ok=True)
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
