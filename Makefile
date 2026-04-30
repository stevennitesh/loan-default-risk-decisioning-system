.PHONY: setup ingest features train evaluate score calibrate explain dashboard-data dashboard-data-post-v1 pipeline-v1 pipeline-post-v1 test

CONFIG ?= configs/base.yaml
CONFIG_V1 ?= configs/v1.yaml
CONFIG_POST_V1 ?= configs/post_v1.yaml

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

calibrate:
	python -m src.calibrate --config $(CONFIG)

explain:
	python -m src.explain --config $(CONFIG)

dashboard-data:
	python -m src.evaluate --config $(CONFIG_V1) --export-dashboard-data

dashboard-data-post-v1:
	python -m src.evaluate --config $(CONFIG_POST_V1) --export-dashboard-data --use-calibrated-dashboard-metrics

pipeline-v1:
	python -m src.ingest --config $(CONFIG_V1)
	python -m src.build_features --config $(CONFIG_V1)
	python -m src.train --config $(CONFIG_V1)
	python -m src.evaluate --config $(CONFIG_V1)
	python -m src.score_batch --config $(CONFIG_V1)
	python -m src.explain --config $(CONFIG_V1)
	python -m src.evaluate --config $(CONFIG_V1) --export-dashboard-data

pipeline-post-v1:
	python -m src.ingest --config $(CONFIG_POST_V1)
	python -m src.build_features --config $(CONFIG_POST_V1)
	python -m src.train --config $(CONFIG_POST_V1)
	python -m src.evaluate --config $(CONFIG_POST_V1)
	python -m src.calibrate --config $(CONFIG_POST_V1)
	python -m src.score_batch --config $(CONFIG_POST_V1)
	python -m src.explain --config $(CONFIG_POST_V1)
	python -m src.evaluate --config $(CONFIG_POST_V1) --export-dashboard-data --use-calibrated-dashboard-metrics

test:
	python -m pytest -q
