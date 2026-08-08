.PHONY: install dev lint format test data models jobs

install:
	pip install -e ".[train]"

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -q

data:     ## KITTI Eigen test split, 652 frames (~0.7 GB)
	bash scripts/download_data.sh

models:   ## all weights, for offline compute nodes
	bash scripts/download_models.sh

jobs:     ## render all Slurm scripts into slurm/generated/
	python3 slurm/generate.py --all
