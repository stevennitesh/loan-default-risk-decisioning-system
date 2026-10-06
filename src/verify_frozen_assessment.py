"""Reproduce every saved workflow's frozen predictions without fitting."""

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from src.feature_experiments import load_split_frames
from src.nested_assessment import assess_workflow
from src.runtime import resolve_config_path


def run(output, tolerance=1e-10):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Frozen verification requires a completed assessment")
    saved = pd.read_csv(output / "assessment_predictions.csv")
    if saved.duplicated(["workflow", "split_seed", "SK_ID_CURR"]).any():
        raise ValueError("Duplicate outer prediction membership")
    checked = []
    with duckdb.connect(
        str(resolve_config_path(manifest["config"], "duckdb_path")), read_only=True
    ) as connection:
        for plan in manifest["folds"]:
            outer_ids = plan["applicant_ids"]["assessment"]
            frame = load_split_frames(
                connection,
                {"assessment": outer_ids},
                manifest["feature_columns"],
            )["assessment"].set_index("SK_ID_CURR")
            for name in manifest["settings"]["workflows"]:
                observed = saved.loc[
                    (saved["workflow"] == name)
                    & (saved["split_seed"] == plan["split_seed"])
                    & (saved["outer_fold"] == plan["outer_fold"])
                ].set_index("SK_ID_CURR")
                if set(observed.index) != set(outer_ids):
                    raise ValueError("Saved workflow has incorrect outer membership")
                workflow = joblib.load(
                    output
                    / f"fold_{plan['split_seed']}_{plan['outer_fold']}_{name}.joblib"
                )
                if set(workflow["fit_applicant_ids"]) & set(outer_ids):
                    raise ValueError("Frozen artifact contains outer fitting IDs")
                ordered = frame.loc[observed.index].reset_index()
                # Negative-control labels are already frozen permutations, never
                # used for fitting here. Original-label runs must match the mart.
                if not manifest["negative_control"] and not np.array_equal(
                    ordered["TARGET"], observed["target"]
                ):
                    raise ValueError("Original-label predictions disagree with source")
                ordered["TARGET"] = observed["target"].to_numpy()
                _, reproduced = assess_workflow(
                    workflow,
                    ordered,
                    manifest["config"],
                    workflow["assessment_context"],
                )
                difference = max(
                    float(
                        np.max(
                            np.abs(
                                np.array([row[f"{kind}_score"] for row in reproduced])
                                - observed[f"{kind}_score"].to_numpy()
                            )
                        )
                    )
                    for kind in ("raw", "calibrated")
                )
                if difference > tolerance:
                    raise ValueError("Frozen predictions exceed declared tolerance")
                checked.append(
                    {
                        "workflow": name,
                        "split_seed": plan["split_seed"],
                        "outer_fold": plan["outer_fold"],
                        "applicant_count": len(observed),
                        "maximum_absolute_score_difference": difference,
                    }
                )
    if sum(row["applicant_count"] for row in checked) != len(saved):
        raise ValueError("Saved prediction file includes undeclared rows")
    result = {
        "status": "complete",
        "assessment_run_id": manifest["run_id"],
        "artifact_count": len(checked),
        "prediction_count": len(saved),
        "absolute_tolerance": tolerance,
        "checks": checked,
    }
    (output / "frozen_artifact_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.assessment_dir)))


if __name__ == "__main__":
    main()
