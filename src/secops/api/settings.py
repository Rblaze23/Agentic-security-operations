"""API settings. Keys and paths come from SECOPS_* environment variables, never from code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SECOPS_", env_file=".env", extra="ignore")

    model_dir: Path = Path.home() / "data" / "secops" / "models"
    detector_subdir: str = "detector"
    family_subdir: str = "family"
    api_keys: str = ""  # comma-separated; empty means every protected route is refused
    top_k: int = 5
    log_level: str = "INFO"
    host: str = "127.0.0.1"  # the Dockerfile sets SECOPS_HOST=0.0.0.0 for the container
    port: int = 8000
    max_body_bytes: int = 8 * 1024 * 1024

    @property
    def keys(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def detector_dir(self) -> Path:
        return self.model_dir / self.detector_subdir

    @property
    def family_dir(self) -> Path:
        return self.model_dir / self.family_subdir


@lru_cache(maxsize=1)
def get_api_settings() -> ApiSettings:
    return ApiSettings()
