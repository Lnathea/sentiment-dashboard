"""Application settings, read from environment variables (and the repo-root .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app import REPO_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env also holds frontend / other variables
    )

    app_env: str = "development"
    # Relative SQLite paths are resolved from the current working directory
    # (run uvicorn/alembic from backend/ -> backend/sentiment.sqlite3).
    database_url: str = "sqlite:///./sentiment.sqlite3"
    cors_origins: str = "http://localhost:3000"
    max_upload_mb: float = Field(default=10, gt=0)
    max_text_length: int = Field(default=1000, gt=0)
    baseline_model_path: Path = Path("ml/artifacts/tfidf_svm.joblib")
    default_model: str = "tfidf_svm"

    @field_validator("baseline_model_path")
    @classmethod
    def _resolve_from_repo_root(cls, v: Path) -> Path:
        return v if v.is_absolute() else (REPO_ROOT / v).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
