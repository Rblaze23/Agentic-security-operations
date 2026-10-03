"""Expected raw files for the DistriNet-improved CIC-IDS-2017 release, with verification helpers.

Hashes measured 2026-10-03 from CICIDS2017_improved.zip (server mtime 2023-04-27).
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from secops.data.schema import DAY_FILES

ZIP_URL = (
    "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip"
)
ZIP_SHA256 = "97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883"
ZIP_SIZE = 343_549_013
EXTRACT_SUBDIR = "improved"


@dataclass(frozen=True)
class FileEntry:
    name: str
    size_bytes: int
    rows: int
    sha256: str


MANIFEST: dict[str, FileEntry] = {
    "monday": FileEntry(
        "monday.csv",
        207_875_155,
        371_624,
        "51fe5dc962626efb4ae70dce0303072fb780da0932822b651202ee9c2fbc1aff",
    ),
    "tuesday": FileEntry(
        "tuesday.csv",
        178_397_720,
        322_078,
        "e2a0a5b631dfc6b455cc9f9a88b944110637d70a7f74171473925f76f38b6b0c",
    ),
    "wednesday": FileEntry(
        "wednesday.csv",
        291_290_505,
        496_641,
        "bf46c5f3c792e8817381f724511229569606918eaf07ac986d7a2592b6341bc2",
    ),
    "thursday": FileEntry(
        "thursday.csv",
        189_519_159,
        362_076,
        "78a4d11eaf473d099e30e71ddb01e0f38218e844c0a9cdd36602145d674af482",
    ),
    "friday": FileEntry(
        "friday.csv",
        285_188_226,
        547_557,
        "ebd499e6f23bd59f9cb81bec28178491b02b925fa5640a24215c9437d79482d0",
    ),
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest_digest() -> str:
    joined = "|".join(f"{d}:{MANIFEST[d].sha256}" for d in sorted(MANIFEST))
    return hashlib.sha256(joined.encode()).hexdigest()


def csv_path(raw_dir: Path, day: str) -> Path:
    return raw_dir / EXTRACT_SUBDIR / DAY_FILES[day]


def verify_raw_dir(raw_dir: Path) -> list[str]:
    problems: list[str] = []
    for day, entry in MANIFEST.items():
        p = csv_path(raw_dir, day)
        if not p.exists():
            problems.append(f"{entry.name}: missing at {p}")
            continue
        size = p.stat().st_size
        if size != entry.size_bytes:
            problems.append(f"{entry.name}: size {size} != expected {entry.size_bytes}")
        digest = sha256_of(p)
        if digest != entry.sha256:
            problems.append(f"{entry.name}: sha256 {digest} != expected {entry.sha256}")
    return problems


def download_zip(url: str, dest: Path, expected_sha256: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_of(dest) == expected_sha256:
        return dest
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    digest = sha256_of(dest)
    if digest != expected_sha256:
        raise ValueError(f"downloaded zip sha256 {digest} != expected {expected_sha256}")
    return dest


def extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        z.extractall(dest_dir, members=names)
    return [dest_dir / n for n in names]


def write_verification_stamp(raw_dir: Path) -> Path:
    stamp = raw_dir / "MANIFEST_VERIFIED.json"
    payload = {
        "verified_at": datetime.now(UTC).isoformat(),
        "manifest_digest": manifest_digest(),
        "files": {d: asdict(e) for d, e in MANIFEST.items()},
    }
    stamp.write_text(json.dumps(payload, indent=2))
    return stamp
