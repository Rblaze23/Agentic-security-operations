# The venv lives on the Linux filesystem: /mnt/* (DrvFs) cannot create the symlinks uv needs.
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/secops

.PHONY: setup lint type test-unit test-integration test data-download data-build train-all mlflow-ui

setup:
	uv sync --all-groups
	uv run pre-commit install

lint:
	uv run ruff check . && uv run ruff format --check .

type:
	uv run mypy

test-unit:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration -m integration

test: lint type test-unit test-integration

data-download:
	uv run secops-data download

data-build:
	uv run secops-data build --attempted-policy relabel_benign
	uv run secops-data build --attempted-policy drop

train-all:
	for c in configs/models/*.yaml; do uv run secops-train run --config $$c; done

mlflow-ui:
	uv run mlflow ui --backend-store-uri "$$(uv run python -c 'from secops.config import get_settings as g; print(g().resolved_tracking_uri())')"
