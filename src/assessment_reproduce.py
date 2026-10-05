"""Refit one declared fold recipe and compare frozen predictions and choices."""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np

from src.evidence import file_sha256
from src.feature_experiments import load_split_frames
from src.nested_assessment import FIT_ROLES, assess_workflow, fit_fold_workflow
from src.runtime import resolve_config_path


def run(output, tolerance=1e-10):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["negative_control"]:
        raise ValueError("Reproduction requires a completed original-label assessment")
    plan = manifest["folds"][0]
    config = manifest["config"]
    with duckdb.connect(
        str(resolve_config_path(config, "duckdb_path")), read_only=True
    ) as con:
        roles = load_split_frames(
            con,
            {name: plan["applicant_ids"][name] for name in FIT_ROLES},
            manifest["feature_columns"],
        )
        assessment = load_split_frames(
            con,
            {"assessment": plan["applicant_ids"]["assessment"]},
            manifest["feature_columns"],
        )["assessment"]
    refitted, _choices, _ranking = fit_fold_workflow(
        config, roles, manifest["feature_columns"], manifest["settings"]
    )
    frozen = joblib.load(
        output
        / f"fold_{plan['split_seed']}_{plan['outer_fold']}_history_selected.joblib"
    )
    context = frozen["assessment_context"]
    _, before = assess_workflow(frozen, assessment, config, context)
    _, after = assess_workflow(refitted, assessment, config, context)
    differences = {
        kind: float(
            np.max(
                np.abs(
                    np.array([row[f"{kind}_score"] for row in before])
                    - np.array([row[f"{kind}_score"] for row in after])
                )
            )
        )
        for kind in ("raw", "calibrated")
    }
    matching = (
        frozen["feature_columns"] == refitted["feature_columns"]
        and frozen["selection_row"]["feature_set"]
        == refitted["selection_row"]["feature_set"]
        and frozen["selected_candidate"]["candidate_name"]
        == refitted["selected_candidate"]["candidate_name"]
        and frozen["selected_candidate"]["params"]
        == refitted["selected_candidate"]["params"]
        and frozen["selected_calibration_method"]
        == refitted["selected_calibration_method"]
    )
    result = {
        "python_executable": sys.executable,
        "requirements_lock_sha256": file_sha256(Path("requirements.lock")),
        "dependencies": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        },
        "status": "complete"
        if matching and max(differences.values()) <= tolerance
        else "failed",
        "assessment_run_id": manifest["run_id"],
        "split_seed": plan["split_seed"],
        "outer_fold": plan["outer_fold"],
        "workflow": "history_selected",
        "absolute_prediction_tolerance": tolerance,
        "max_absolute_prediction_difference": differences,
        "same_features_parameters_calibration": matching,
        "applicant_count": len(assessment),
        "selected_feature_set": refitted["selection_row"]["feature_set"],
        "selected_candidate_name": refitted["selected_candidate"]["candidate_name"],
        "calibration_method": refitted["selected_calibration_method"],
        "limitation": "Same host, Python, locked wheels, data and random roles; does not prove portability across hardware or training resamples.",
    }
    (output / "clean_environment_reproduction.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if result["status"] != "complete":
        raise ValueError("Declared fold reproduction failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.assessment_dir)))


if __name__ == "__main__":
    main()
