"""secops-data command line."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from secops.config import Settings, get_settings
from secops.data import manifest as m
from secops.data.build import build as build_pipeline
from secops.data.clean import AttemptedPolicy

app = typer.Typer(help="Dataset acquisition and preprocessing.")


def _report_problems(problems: list[str]) -> None:
    for p in problems:
        typer.echo(f"PROBLEM: {p}", err=True)
    if problems:
        raise typer.Exit(code=1)


@app.command()
def download(data_dir: Path | None = None) -> None:
    """Download the DistriNet-improved CIC-IDS-2017 zip, verify, extract, verify CSVs."""
    raw = (data_dir or get_settings().data_dir) / "raw"
    zip_path = m.download_zip(m.ZIP_URL, raw / "CICIDS2017_improved.zip", m.ZIP_SHA256)
    typer.echo(f"zip verified: {zip_path}")
    m.extract_zip(zip_path, raw / m.EXTRACT_SUBDIR)
    _report_problems(m.verify_raw_dir(raw))
    stamp = m.write_verification_stamp(raw)
    typer.echo(f"all files verified; stamp written to {stamp}")


@app.command()
def verify(data_dir: Path | None = None) -> None:
    """Verify already-downloaded CSVs against the manifest."""
    raw = (data_dir or get_settings().data_dir) / "raw"
    _report_problems(m.verify_raw_dir(raw))
    typer.echo("all files verified")


@app.command()
def build(
    attempted_policy: AttemptedPolicy = AttemptedPolicy.RELABEL_BENIGN,
    data_dir: Path | None = None,
    subdir: str = "improved",
) -> None:
    """Clean, split and write processed/<policy>/flows.parquet with reports."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    st = get_settings() if data_dir is None else Settings(data_dir=data_dir)
    out = build_pipeline(
        st.raw_dir, st.processed_dir, st.reports_dir, attempted_policy, subdir=subdir
    )
    typer.echo(f"built {out}")
