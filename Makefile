.PHONY: setup ingest features train evaluate score dashboard-data test

CONFIG ?= configs/base.yaml

setup:
	python -m pip install -r requirements.txt

ingest:
	python -m src.ingest --config $(CONFIG)

features:
	python -m src.build_features --config $(CONFIG)

train:
	python -m src.train --config $(CONFIG)

evaluate:
	python -m src.evaluate --config $(CONFIG)

score:
	python -m src.score_batch --config $(CONFIG)

dashboard-data:
	python -m src.evaluate --config $(CONFIG) --export-dashboard-data

test:
	python -m pytest -q
