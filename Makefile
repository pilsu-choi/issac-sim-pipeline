UV ?= uv
ISAAC_SIM_PYTHON ?= /path/to/isaac-sim/python.sh
API_PORT ?= 8000
UI_PORT ?= 7860
UI_METRICS_PORT ?= 7861

.PHONY: install mlflow generate-data generate-data-gui convert-data train api ui monitor stack

install:
	$(UV) sync

mlflow:
	docker compose up -d mlflow

generate-data:
	$(ISAAC_SIM_PYTHON) scripts/generate_synthetic_dataset.py --config configs/dataset.yaml --overwrite

generate-data-gui:
	$(ISAAC_SIM_PYTHON) scripts/generate_synthetic_dataset.py --config configs/dataset.yaml --gui --keep-open --overwrite

convert-data:
	$(UV) run python scripts/convert_replicator_to_yolo.py --config configs/dataset.yaml --overwrite

train:
	$(UV) run python src/training/train_yolo.py --config configs/training.yaml

api:
	$(UV) run uvicorn src.api.app:app --host 0.0.0.0 --port $(API_PORT)

ui:
	UI_PORT=$(UI_PORT) UI_METRICS_PORT=$(UI_METRICS_PORT) $(UV) run python src/ui/app.py

monitor:
	docker compose up -d prometheus grafana

stack:
	docker compose up --build
