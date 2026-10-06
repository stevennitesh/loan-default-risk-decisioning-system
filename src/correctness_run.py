"""Run an isolated named pipeline and preserve phase-level success/failure evidence."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.config import is_post_v1_scope, load_config
from src.evidence import fingerprints
from src.runtime import created_at_utc, resolve_config_path


def run(config_path):
    config = load_config(config_path)
    output = resolve_config_path(config, "report_dir")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "pipeline_manifest.json"
    if path.exists():
        raise ValueError(
            "Use a fresh named output directory; pipeline manifest already exists"
        )
    manifest = {
        "status": "running",
        "created_at": created_at_utc(),
        "config": config,
        "fingerprints": fingerprints(config),
        "stages": [],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
    }
    stages = [
        "ingest",
        "build_features",
        "train",
        "evaluate",
        "calibrate",
        "score_batch",
        "explain",
        "evaluate",
    ]
    if not is_post_v1_scope(config):
        stages.remove("calibrate")
    try:
        for index, module in enumerate(stages):
            command = [
                sys.executable,
                "-m",
                f"src.{module}",
                "--config",
                str(config_path),
            ]
            if index == len(stages) - 1:
                command += ["--export-dashboard-data"]
                if is_post_v1_scope(config):
                    command += ["--use-calibrated-dashboard-metrics"]
            record = {
                "module": module,
                "command": command,
                "started_at": created_at_utc(),
                "status": "running",
            }
            manifest["stages"].append(record)
            path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Starting {module}", flush=True)
            with (output / f"stage_{index}_{module}.log").open(
                "w", encoding="utf-8"
            ) as log:
                completed = subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            record.update(
                returncode=completed.returncode,
                finished_at=created_at_utc(),
                status="complete" if completed.returncode == 0 else "failed",
            )
            if completed.returncode:
                raise RuntimeError(
                    f"Stage {module} failed; see {output / f'stage_{index}_{module}.log'}"
                )
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
