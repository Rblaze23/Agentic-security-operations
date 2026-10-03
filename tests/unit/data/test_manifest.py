import hashlib
from pathlib import Path

from secops.data import manifest as m


def test_manifest_lists_five_days_with_hashes() -> None:
    assert set(m.MANIFEST) == {"monday", "tuesday", "wednesday", "thursday", "friday"}
    for entry in m.MANIFEST.values():
        assert len(entry.sha256) == 64
        assert entry.size_bytes > 0 and entry.rows > 0
    assert sum(e.rows for e in m.MANIFEST.values()) == 2_099_976


def test_sha256_of_small_file(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert m.sha256_of(p) == hashlib.sha256(b"hello").hexdigest()


def test_verify_raw_dir_reports_missing_and_mismatched(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "improved"
    raw.mkdir(parents=True)
    (raw / "monday.csv").write_text("not the real file")
    problems = m.verify_raw_dir(tmp_path / "raw")
    assert any("monday.csv" in p and "sha256" in p for p in problems)
    assert any("tuesday.csv" in p and "missing" in p for p in problems)


def test_manifest_digest_is_stable() -> None:
    assert m.manifest_digest() == m.manifest_digest()
    assert len(m.manifest_digest()) == 64
