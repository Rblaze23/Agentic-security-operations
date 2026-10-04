"""Thin HTTP client for the dashboard. The API key comes from the environment, never the UI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

DEFAULT_URL = "http://localhost:8000"


class ApiClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or os.environ.get("SECOPS_API_URL", DEFAULT_URL)).rstrip("/")
        key = api_key or os.environ.get("SECOPS_API_KEY") or os.environ.get("SECOPS_API_KEYS", "")
        self.api_key = key.split(",")[0].strip()
        self.runs_dir = Path(os.environ.get("SECOPS_EVAL_RUNS_DIR", "evaluation/runs"))

    def _get(self, path: str, **params: Any) -> Any:
        r = requests.get(
            f"{self.base_url}{path}", headers={"X-API-Key": self.api_key}, params=params, timeout=30
        )
        r.raise_for_status()
        return r.json()

    def health(self) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return dict(r.json())

    def investigations(self, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._get("/investigations", limit=limit))

    def investigation(self, investigation_id: str) -> dict[str, Any]:
        return dict(self._get(f"/investigations/{investigation_id}"))

    def start_investigation(self, alert: dict[str, Any]) -> dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/investigations",
            headers={"X-API-Key": self.api_key},
            json={"alert": alert},
            timeout=30,
        )
        r.raise_for_status()
        return dict(r.json())

    def evaluation_runs(self) -> list[dict[str, Any]]:
        return list(self._get("/evaluation/runs"))

    def run_file(self, run_id: str) -> dict[str, Any] | None:
        path = self.runs_dir / f"{run_id}.json"
        return json.loads(path.read_text()) if path.exists() else None
