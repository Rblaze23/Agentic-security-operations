"""Versioned prompt templates for the investigation agent."""

from __future__ import annotations

from pathlib import Path

PROMPT_VERSION = "2026-10-04.2"
_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    return (_DIR / f"{name}.md").read_text().strip()
