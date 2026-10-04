"""Fixture files for record/replay: plain `NNN.json` or gzipped `NNN.json.gz` (the request bodies
repeat the growing history, so gzip shrinks a scenario roughly tenfold)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

SUFFIXES = (".json", ".json.gz")


def count_entries(directory: Path) -> int:
    return (
        sum(1 for p in directory.iterdir() if p.name.endswith(SUFFIXES))
        if directory.exists()
        else 0
    )


def write_entry(directory: Path, stem: str, entry: dict[str, Any], compress: bool) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(entry, indent=None if compress else 2, default=str)
    if compress:
        path = directory / f"{stem}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        path = directory / f"{stem}.json"
        path.write_text(text)
    return path


def read_entries(directory: Path) -> list[dict[str, Any]]:
    """Every entry in name order, whichever suffix each file has."""
    out: list[dict[str, Any]] = []
    if not directory.exists():
        return out
    for p in sorted(p for p in directory.iterdir() if p.name.endswith(SUFFIXES)):
        if p.name.endswith(".gz"):
            with gzip.open(p, "rt", encoding="utf-8") as f:
                out.append(json.loads(f.read()))
        else:
            out.append(json.loads(p.read_text()))
    return out
