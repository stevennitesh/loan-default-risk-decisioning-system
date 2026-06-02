.PHONY: setup ingest features train evaluate score calibrate explain dashboard-data dashboard-data-post-v1 pipeline-v1 pipeline-post-v1 lint format-check format test

CONFIG ?= configs/base.yaml
CONFIG_V1 ?= configs/v1.yaml
CONFIG_POST_V1 ?= configs/post_v1.yaml
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

setup:
	$(PYTHON) -m pip install -r requirements.txt

ingest:
	$(PYTHON) -m src.ingest --config $(CONFIG)

features:
	$(PYTHON) -m src.build_features --config $(CONFIG)

train:
	$(PYTHON) -m src.train --config $(CONFIG)

evaluate:
	$(PYTHON) -m src.evaluate --config $(CONFIG)

score:
	$(PYTHON) -m src.score_batch --config $(CONFIG)

calibrate:
	$(PYTHON) -m src.calibrate --config $(CONFIG)

explain:
	$(PYTHON) -m src.explain --config $(CONFIG)

lint:
	$(PYTHON) -m ruff check src tests

format-check:
	$(PYTHON) -m ruff format --check src tests

format:
	$(PYTHON) -m ruff format src tests

dashboard-data:
	$(PYTHON) -m src.evaluate --config $(CONFIG_V1) --export-dashboard-data

dashboard-data-post-v1:
	$(PYTHON) -m src.evaluate --config $(CONFIG_POST_V1) --export-dashboard-data --use-calibrated-dashboard-metrics

pipeline-v1:
	$(PYTHON) -m src.ingest --config $(CONFIG_V1)
	$(PYTHON) -m src.build_features --config $(CONFIG_V1)
	$(PYTHON) -m src.train --config $(CONFIG_V1)
	$(PYTHON) -m src.evaluate --config $(CONFIG_V1)
	$(PYTHON) -m src.score_batch --config $(CONFIG_V1)
	$(PYTHON) -m src.explain --config $(CONFIG_V1)
	$(PYTHON) -m src.evaluate --config $(CONFIG_V1) --export-dashboard-data

pipeline-post-v1:
	$(PYTHON) -m src.ingest --config $(CONFIG_POST_V1)
	$(PYTHON) -m src.build_features --config $(CONFIG_POST_V1)
	$(PYTHON) -m src.train --config $(CONFIG_POST_V1)
	$(PYTHON) -m src.evaluate --config $(CONFIG_POST_V1)
	$(PYTHON) -m src.calibrate --config $(CONFIG_POST_V1)
	$(PYTHON) -m src.score_batch --config $(CONFIG_POST_V1)
	$(PYTHON) -m src.explain --config $(CONFIG_POST_V1)
	$(PYTHON) -m src.evaluate --config $(CONFIG_POST_V1) --export-dashboard-data --use-calibrated-dashboard-metrics

test:
	$(PYTHON) -m pytest -q
