# The venv lives on the Linux filesystem: /mnt/* (DrvFs) cannot create the symlinks uv needs.
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/secops

.PHONY: setup lint type test-unit test-integration test data-download data-build train-binary train-family train-heldout train-ablations train-all mlflow-ui export-models api docker-build docker-smoke compose-up compose-down load-events fetch-attack test-network

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

train-binary:
	for m in logreg xgboost lightgbm; do \
	  uv run secops-train run --config configs/models/$${m}_binary.yaml; \
	  uv run secops-train run --config configs/models/$${m}_binary.yaml --set weighting=none --set run_name=$${m}_none; \
	done
	uv run secops-train promote-best secops/detection-binary secops-detector

train-family:
	for m in lightgbm xgboost; do \
	  uv run secops-train run --config configs/models/$${m}_family.yaml; \
	  uv run secops-train run --config configs/models/$${m}_family.yaml --set weighting=none --set run_name=$${m}_family_none; \
	done
	uv run secops-train promote-best secops/detection-family secops-family-classifier --metric val_macro_f1

train-heldout:
	uv run secops-train run --config configs/models/lightgbm_binary_heldout.yaml \
	  --set reuse_threshold_from_run=$$(uv run python -c "import mlflow;from mlflow import MlflowClient;from secops.config import get_settings as g;mlflow.set_tracking_uri(g().resolved_tracking_uri());print(MlflowClient().get_model_version_by_alias('secops-detector','champion').run_id)")

train-ablations:
	uv run secops-train run --config configs/models/ablation_withport.yaml
	uv run secops-train run --config configs/models/ablation_drop.yaml

train-all: train-binary train-family train-heldout train-ablations

mlflow-ui:
	uv run mlflow ui --backend-store-uri "$$(uv run python -c 'from secops.config import get_settings as g; print(g().resolved_tracking_uri())')"

export-models:
	uv run secops-train export --model-name secops-detector --out "$${SECOPS_MODEL_DIR:-$$HOME/data/secops/models}/detector"
	uv run secops-train export --model-name secops-family-classifier --out "$${SECOPS_MODEL_DIR:-$$HOME/data/secops/models}/family"

api:
	uv run secops-api

docker-build:
	docker build --target runtime -t secops-api:local .

# Builds fixture bundles, starts the image against them and checks /health and /predict.
docker-smoke: docker-build
	uv run python scripts/docker_smoke.py secops-api:local

compose-up:
	docker compose up --build -d api

compose-down:
	docker compose down

load-events:
	uv run secops-data load-events --attempted-policy relabel_benign

fetch-attack:
	uv run python scripts/fetch_attack.py

test-network:
	uv run pytest tests/network -m network
