"""Runtime settings. Paths and secrets come from the environment, never from code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SECOPS_", env_file=".env", extra="ignore")

    data_dir: Path = Path.home() / "data" / "secops"
    mlflow_tracking_uri: str | None = None
    database_url: str | None = None
    random_seed: int = 42

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def mlflow_dir(self) -> Path:
        return self.data_dir / "mlflow"

    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.data_dir / 'events.db'}"

    def resolved_tracking_uri(self) -> str:
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        return f"sqlite:///{self.mlflow_dir / 'mlflow.db'}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
