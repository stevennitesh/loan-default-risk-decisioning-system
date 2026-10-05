"""Independent Python checks of selected real repayment and monthly source rows.

SQL remains the feature extractor. This small oracle checks selected raw/staged
records with Python collections, rather than re-executing a second SQL pipeline.
Applicant/source row evidence stays in ignored local outputs.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.config import load_config
from src.data_contracts import get_model_feature_columns, validate_data_contracts
from src.evidence import file_sha256
from src.runtime import configure_duckdb, resolve_config_path

KEY = ["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"]


def finite(value):
    return value is not None and np.isfinite(value)


def repayment_oracle(records):
    """Infer only unambiguous obligations; keep every source payment record."""
    versions = defaultdict(set)
    groups = defaultdict(list)
    for row in records:
        key = tuple(row[name] for name in KEY)
        groups[key].append(row)
        if row["NUM_INSTALMENT_VERSION"] is not None:
            versions[(key[0], key[1], key[3])].add(key[2])
    output = {}
    for key, rows in groups.items():
        days = [row["DAYS_INSTALMENT"] for row in rows]
        amounts = [row["AMT_INSTALMENT"] for row in rows]
        if not any(day is None or day < 0 for day in days):
            continue
        known = (
            all(finite(day) and day < 0 for day in days)
            and all(finite(amount) and amount > 0 for amount in amounts)
            and len(set(days)) == 1
            and len(set(amounts)) == 1
            and all(value is not None for value in key[1:])
            and len(versions[(key[0], key[1], key[3])]) == 1
        )
        unknown = any(
            not finite(row["DAYS_ENTRY_PAYMENT"])
            or (
                row["DAYS_ENTRY_PAYMENT"] < 0
                and (not finite(row["AMT_PAYMENT"]) or row["AMT_PAYMENT"] < 0)
            )
            for row in rows
        )
        daily = defaultdict(float)
        for row in rows:
            if (
                finite(row["DAYS_ENTRY_PAYMENT"])
                and row["DAYS_ENTRY_PAYMENT"] < 0
                and finite(row["AMT_PAYMENT"])
                and row["AMT_PAYMENT"] >= 0
            ):
                daily[row["DAYS_ENTRY_PAYMENT"]] += row["AMT_PAYMENT"]
        due = (
            min(day for day in days if day is not None)
            if any(day is not None for day in days)
            else None
        )
        amount = (
            min(value for value in amounts if value is not None)
            if any(value is not None for value in amounts)
            else None
        )
        completion = None
        cumulative = 0.0
        for day in sorted(daily):
            cumulative += daily[day]
            if amount is not None and cumulative >= amount and completion is None:
                completion = day
        usable = known and not unknown
        output[key] = {
            "obligation_known": known,
            "payment_unknown": unknown,
            "paid_amount": cumulative if usable else None,
            "underpaid": int(cumulative < amount) if usable else None,
            "late": int(completion is None or completion > due) if usable else None,
            "completed_delay_days": completion - due
            if usable and completion is not None
            else None,
            "arrears_age_days": -due if usable and completion is None else None,
        }
    return output


def _rate(rows, column):
    values = [row[column] for row in rows]
    if any(finite(value) and value > 0 for value in values):
        return 1.0
    if all(finite(value) and value >= 0 for value in values):
        return 0.0
    return None


def _mean(values):
    known = [value for value in values if value is not None]
    return float(np.mean(known)) if known else None


def monthly_oracle(records, card=False):
    groups = defaultdict(list)
    for row in records:
        if finite(row["MONTHS_BALANCE"]) and row["MONTHS_BALANCE"] <= -1:
            groups[row["MONTHS_BALANCE"]].append(row)
    months = [groups[month] for month in sorted(groups, reverse=True)[:3]]
    if not months:
        return {}
    if not card:
        pairs = [
            row
            for rows in months
            for row in rows
            if finite(row["CNT_INSTALMENT"])
            and row["CNT_INSTALMENT"] > 0
            and finite(row["CNT_INSTALMENT_FUTURE"])
            and row["CNT_INSTALMENT_FUTURE"] >= 0
        ]
        return {
            "pos_cash_last_3_dpd_rate": _mean(
                [_rate(rows, "SK_DPD") for rows in months]
            ),
            "pos_cash_last_3_dpd_def_rate": _mean(
                [_rate(rows, "SK_DPD_DEF") for rows in months]
            ),
            "pos_cash_last_3_future_installment_ratio": sum(
                row["CNT_INSTALMENT_FUTURE"] for row in pairs
            )
            / sum(row["CNT_INSTALMENT"] for row in pairs)
            if pairs
            else None,
        }
    result = {
        "credit_card_last_3_dpd_rate": _mean(
            [_rate(rows, "SK_DPD") for rows in months]
        ),
        "credit_card_last_3_drawing_count": _mean(
            [
                sum(row["CNT_DRAWINGS_CURRENT"] for row in rows)
                if all(
                    finite(row["CNT_DRAWINGS_CURRENT"])
                    and row["CNT_DRAWINGS_CURRENT"] >= 0
                    for row in rows
                )
                else None
                for rows in months
            ]
        ),
    }
    for numerator, denominator, feature in (
        (
            "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL",
            "credit_card_last_3_credit_utilization",
        ),
        (
            "AMT_PAYMENT_CURRENT",
            "AMT_INST_MIN_REGULARITY",
            "credit_card_last_3_payment_to_min_ratio",
        ),
    ):
        pairs = [
            row
            for rows in months
            for row in rows
            if finite(row[numerator])
            and row[numerator] >= 0
            and finite(row[denominator])
            and row[denominator] > 0
        ]
        result[feature] = (
            sum(row[numerator] for row in pairs)
            / sum(row[denominator] for row in pairs)
            if pairs
            else None
        )
    return result


def _records(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def _equal(actual, expected):
    if expected is None:
        return actual is None
    return actual is not None and np.isclose(actual, expected, rtol=1e-10, atol=1e-8)


def run(config_path):
    config = load_config(config_path)
    output = resolve_config_path(config, "report_dir") / "source_reconciliation"
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running",
        "oracle_tolerance": {"rtol": 1e-10, "atol": 1e-8},
        "repayment_cases": [],
        "monthly_cases": [],
    }
    path = output / "manifest.json"
    try:
        with duckdb.connect(
            str(resolve_config_path(config, "duckdb_path")), read_only=True
        ) as con:
            configure_duckdb(con, config)
            validate_data_contracts(con, config)
            # Deliberate edge-case selection; these labels never choose model settings.
            conditions = {
                "split_payments": "obligation_known AND NOT payment_unknown AND payment_row_count>1",
                "on_time": "obligation_known AND late=0",
                "completed_late": "completed_delay_days>0",
                "ongoing_arrears": "arrears_age_days IS NOT NULL",
                "competing_or_conflicting": "NOT obligation_known",
                "unknown_payments": "obligation_known AND payment_unknown",
            }
            selected = {}
            duplicate_examples = con.execute(
                "SELECT * , COUNT(*) AS duplicate_count FROM stg_installments_payments GROUP BY ALL HAVING COUNT(*)>1 LIMIT 3"
            ).fetch_df()
            selected["identical_payment_records"] = [
                int(value) for value in duplicate_examples["SK_ID_CURR"]
            ]
            for name, condition in conditions.items():
                selected[name] = [
                    int(row[0])
                    for row in con.execute(
                        f"SELECT DISTINCT SK_ID_CURR FROM n_installment_obligations WHERE {condition} ORDER BY SK_ID_CURR LIMIT 3"
                    ).fetchall()
                ]
            for name, condition in {
                "future_payments": "DAYS_ENTRY_PAYMENT>=0",
                "future_schedules": "DAYS_INSTALMENT>=0",
            }.items():
                selected[name] = [
                    int(row[0])
                    for row in con.execute(
                        f"SELECT DISTINCT SK_ID_CURR FROM stg_installments_payments WHERE {condition} ORDER BY SK_ID_CURR LIMIT 3"
                    ).fetchall()
                ]
            applicants = sorted(set().union(*map(set, selected.values())))
            raw = con.execute(
                "SELECT * FROM stg_installments_payments WHERE SK_ID_CURR IN (SELECT UNNEST(?))",
                [applicants],
            ).fetch_df()
            records = _records(raw)
            expected = repayment_oracle(records)
            actual_rows = _records(
                con.execute(
                    "SELECT * FROM n_installment_obligations WHERE SK_ID_CURR IN (SELECT UNNEST(?))",
                    [applicants],
                ).fetch_df()
            )
            actual = {tuple(row[name] for name in KEY): row for row in actual_rows}
            assert set(expected) == set(actual), "Source obligation keys disagree"
            checks = 0
            for key, values in expected.items():
                for name, value in values.items():
                    assert _equal(actual[key][name], value), (
                        f"Repayment mismatch {key} {name}"
                    )
                    checks += 1
            raw.to_csv(output / "repayment_source_rows.csv", index=False)
            result.update(
                repayment_selected_applicants=selected,
                repayment_source_rows=len(records),
                repayment_obligations=len(expected),
                repayment_numeric_checks=checks,
                identical_source_record_count=int(raw.duplicated().sum()),
                duplicate_policy="Retain identical payment records: no unique cashflow identifier proves duplication",
            )
            for category, ids in selected.items():
                result["repayment_cases"].append(
                    {
                        "category": category,
                        "applicant_count": len(ids),
                        "agreement": "passed" if ids else "no actual source example",
                    }
                )
            for table, card in (
                ("stg_pos_cash_balance", False),
                ("stg_credit_card_balance", True),
            ):
                ids = [
                    int(row[0])
                    for row in con.execute(
                        f"SELECT SK_ID_CURR FROM {table} WHERE MONTHS_BALANCE<=-1 GROUP BY SK_ID_CURR,MONTHS_BALANCE HAVING COUNT(DISTINCT SK_ID_PREV)>1 ORDER BY SK_ID_CURR LIMIT 5"
                    ).fetchall()
                ]
                unknown = (
                    "CNT_INSTALMENT IS NULL OR CNT_INSTALMENT_FUTURE IS NULL"
                    if not card
                    else "AMT_BALANCE IS NULL OR AMT_CREDIT_LIMIT_ACTUAL IS NULL OR AMT_PAYMENT_CURRENT IS NULL OR AMT_INST_MIN_REGULARITY IS NULL"
                )
                ids += [
                    int(row[0])
                    for row in con.execute(
                        f"SELECT DISTINCT SK_ID_CURR FROM {table} WHERE {unknown} ORDER BY SK_ID_CURR LIMIT 5"
                    ).fetchall()
                ]
                ids = sorted(set(ids))
                source = con.execute(
                    f"SELECT * FROM {table} WHERE SK_ID_CURR IN (SELECT UNNEST(?))",
                    [ids],
                ).fetch_df()
                source.to_csv(output / f"{table}_source_rows.csv", index=False)
                for applicant, group in source.groupby("SK_ID_CURR"):
                    expected_values = monthly_oracle(_records(group), card)
                    actual_values = _records(
                        con.execute(
                            "SELECT * FROM f_last_k_temporal_features WHERE SK_ID_CURR=?",
                            [int(applicant)],
                        ).fetch_df()
                    )[0]
                    for name, value in expected_values.items():
                        assert _equal(actual_values[name], value), (
                            f"Monthly mismatch {applicant} {name}"
                        )
                    result["monthly_cases"].append(
                        {
                            "table": table,
                            "SK_ID_CURR": int(applicant),
                            "source_row_count": len(group),
                            "checks": len(expected_values),
                            "values": expected_values,
                            "agreement": "passed",
                        }
                    )
                duplicates = con.execute(
                    f"SELECT COUNT(*) FROM (SELECT SK_ID_CURR,SK_ID_PREV,MONTHS_BALANCE FROM {table} WHERE MONTHS_BALANCE<=-1 GROUP BY ALL HAVING COUNT(*)>1)"
                ).fetchone()[0]
                assert duplicates == 0, "Ambiguous duplicate account-month rows"
                result[f"{table}_duplicate_account_month_groups"] = duplicates
            coverage = con.execute(
                "SELECT TARGET, COUNT(*) AS applicants, COUNT(*) FILTER (WHERE installment_ambiguous_obligation_count>0) AS ambiguous, COUNT(*) FILTER (WHERE installment_unknown_payment_obligation_count>0) AS unknown_payment, COUNT(*) FILTER (WHERE installment_obligation_count=0) AS no_installment_history FROM mart_credit_risk_features WHERE source_population='application_train' GROUP BY TARGET ORDER BY TARGET"
            ).fetch_df()
            coverage.to_csv(output / "coverage_by_target.csv", index=False)
            result["coverage_by_target"] = _records(coverage)
            result["schedule_counts"] = _records(
                con.execute(
                    "SELECT COUNT(*) AS schedule_groups, COUNT(*) FILTER (WHERE NOT obligation_known) AS ambiguous_groups,COUNT(*) FILTER (WHERE obligation_known AND payment_unknown) AS unknown_payment_groups FROM n_installment_obligations"
                ).fetch_df()
            )[0]
            features = get_model_feature_columns(con, config)
            result["feature_count"] = len(features)
            result["excluded_model_features"] = sorted(
                set().union(*map(set, config["excluded_features"].values()))
            )
            assert not set(features) & set(result["excluded_model_features"])
            result["availability_counts"] = {
                "bureau_eligible_prior_origins": con.execute(
                    "SELECT COUNT(*) FROM n_eligible_bureau"
                ).fetchone()[0],
                "bureau_day_zero_origins_excluded": con.execute(
                    "SELECT COUNT(*) FROM stg_bureau WHERE DAYS_CREDIT=0"
                ).fetchone()[0],
                "bureau_positive_origins_excluded": con.execute(
                    "SELECT COUNT(*) FROM stg_bureau WHERE DAYS_CREDIT>0 AND ISFINITE(DAYS_CREDIT)"
                ).fetchone()[0],
                "bureau_unknown_origins_excluded": con.execute(
                    "SELECT COUNT(*) FROM stg_bureau WHERE DAYS_CREDIT IS NULL OR NOT ISFINITE(DAYS_CREDIT)"
                ).fetchone()[0],
                "bureau_credit_origin_nonnegative": con.execute(
                    "SELECT COUNT(*) FROM stg_bureau WHERE DAYS_CREDIT>=0"
                ).fetchone()[0],
                "bureau_actual_end_after_application": con.execute(
                    "SELECT COUNT(*) FROM stg_bureau WHERE DAYS_ENDDATE_FACT>0"
                ).fetchone()[0],
                "previous_decision_nonnegative": con.execute(
                    "SELECT COUNT(*) FROM stg_previous_application WHERE DAYS_DECISION>=0"
                ).fetchone()[0],
                "pos_current_future_excluded": con.execute(
                    "SELECT COUNT(*) FROM stg_pos_cash_balance WHERE MONTHS_BALANCE>=0"
                ).fetchone()[0],
                "card_current_future_excluded": con.execute(
                    "SELECT COUNT(*) FROM stg_credit_card_balance WHERE MONTHS_BALANCE>=0"
                ).fetchone()[0],
                "installment_current_future_payments_excluded": con.execute(
                    "SELECT COUNT(*) FROM stg_installments_payments WHERE DAYS_ENTRY_PAYMENT>=0"
                ).fetchone()[0],
            }
            result["bureau_origin_coverage_by_target"] = _records(
                con.execute(
                    "SELECT a.TARGET,SUM(c.day_zero_loan_count) AS day_zero_loans,COUNT(*) FILTER (WHERE c.day_zero_loan_count>0) AS applicants_with_day_zero_origins,SUM(c.unknown_origin_loan_count) AS unknown_origin_loans FROM bureau_origin_coverage c JOIN stg_application_train a USING (SK_ID_CURR) GROUP BY a.TARGET ORDER BY a.TARGET"
                ).fetch_df()
            )
            result["verification_source_sha256"] = {
                name: file_sha256(Path(name))
                for name in (
                    "src/source_reconciliation.py",
                    "sql/03_feature_bureau.sql",
                    "sql/03b_feature_bureau_balance.sql",
                    "sql/05c_feature_recency_deterioration.sql",
                    "sql/05_feature_installments.sql",
                    "sql/05d_feature_last_k_temporal.sql",
                )
            }
        result["source_rows_sha256"] = {
            item.name: file_sha256(item)
            for item in sorted(output.glob("*_source_rows.csv"))
        }
        result.update(
            status="complete",
            limitations="Relative offsets support declared pre-application censoring, but no reliable calendar application dates or amendment timestamps exist. Anonymized application external-score/vendor availability cannot be independently established. Selected Python examples are independent numeric checks, not exhaustive validation of every source group.",
        )
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config)))


if __name__ == "__main__":
    main()
