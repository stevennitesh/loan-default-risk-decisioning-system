"""Descriptive diagnostics on frozen outer predictions; no fitting or selection."""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from src.config import manual_review_capacity_rate
from src.metrics import probability_metrics
from src.runtime import resolve_config_path


def run(output):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Diagnostics require completed assessment")
    config = manifest["config"]
    predictions = pd.read_csv(output / "assessment_predictions.csv")
    with duckdb.connect(
        str(resolve_config_path(config, "duckdb_path")), read_only=True
    ) as con:
        segments = con.execute(
            "SELECT SK_ID_CURR, CASE WHEN installment_obligation_count=0 THEN 'no_installment_history' WHEN installment_ambiguous_obligation_count>0 OR installment_unknown_payment_obligation_count>0 THEN 'ambiguous_or_unknown_history' ELSE 'known_installment_history' END AS history_segment FROM mart_credit_risk_features WHERE source_population='application_train'"
        ).fetch_df()
    predictions = predictions.merge(segments, on="SK_ID_CURR", validate="many_to_one")
    rows = []
    for (workflow, seed, fold, segment), frame in predictions.groupby(
        ["workflow", "split_seed", "outer_fold", "history_segment"]
    ):
        for kind in ("raw", "calibrated"):
            metrics = (
                probability_metrics(
                    frame["target"],
                    frame[f"{kind}_score"].to_numpy(),
                    manual_review_capacity_rate(config),
                )
                if frame["target"].nunique() == 2
                else {}
            )
            rows.append(
                {
                    "workflow": workflow,
                    "split_seed": seed,
                    "outer_fold": fold,
                    "history_segment": segment,
                    "score_kind": kind,
                    "applicant_count": len(frame),
                    "target_rate": float(frame["target"].mean()),
                    **metrics,
                }
            )
    pd.DataFrame(rows).to_csv(output / "history_segment_metrics.csv", index=False)
    # The run declaration fixes these segments; they cannot influence selection.
    choices = pd.DataFrame(
        [
            {
                **{
                    k: row[k]
                    for k in (
                        "workflow",
                        "split_seed",
                        "outer_fold",
                        "calibration_method",
                    )
                },
                "feature_count": len(row["feature_columns"]),
                "candidate_name": row["candidate"]["candidate_name"],
            }
            for row in manifest["selected_workflows"]
        ]
    )
    choices.to_csv(output / "selection_stability.csv", index=False)
    history = [
        row
        for row in manifest["selected_workflows"]
        if row["workflow"] == "history_selected"
    ]
    inclusion = []
    for feature in manifest["feature_columns"]:
        inclusion.append(
            {
                "feature_name": feature,
                "selected_fold_count": sum(
                    feature in row["feature_columns"] for row in history
                ),
                "fold_count": len(history),
            }
        )
    pd.DataFrame(inclusion).to_csv(
        output / "feature_selection_frequency.csv", index=False
    )
    result = {
        "status": "complete",
        "assessment_run_id": manifest["run_id"],
        "history_segment_rows": len(rows),
        "segment_definition": "No obligations; otherwise any ambiguous/unknown obligation; otherwise known history. Comparisons are conditional population descriptions, not causal effects or alternate schedule inference.",
        "calibration_selection_counts": choices.groupby(
            ["workflow", "calibration_method"]
        )
        .size()
        .to_dict()
        .__str__(),
        "candidate_selection_counts": choices.groupby(["workflow", "candidate_name"])
        .size()
        .to_dict()
        .__str__(),
        "uncertainty": "Fold means and sample SD are descriptive, not confidence intervals. No pooled cross-fold average precision or bootstrap claim.",
    }
    (output / "diagnostics_manifest.json").write_text(
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
