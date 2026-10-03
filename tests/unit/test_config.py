from pathlib import Path

from secops.config import Settings


def test_settings_defaults_derive_subdirectories(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path)
    assert s.raw_dir == tmp_path / "raw"
    assert s.processed_dir == tmp_path / "processed"
    assert s.reports_dir == tmp_path / "reports"
    assert s.mlflow_dir == tmp_path / "mlflow"
    assert s.random_seed == 42


def test_tracking_uri_defaults_to_sqlite_under_data_dir(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path, mlflow_tracking_uri=None)
    assert s.resolved_tracking_uri() == f"sqlite:///{tmp_path / 'mlflow' / 'mlflow.db'}"


def test_tracking_uri_override_wins(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path, mlflow_tracking_uri="http://localhost:5000")
    assert s.resolved_tracking_uri() == "http://localhost:5000"
