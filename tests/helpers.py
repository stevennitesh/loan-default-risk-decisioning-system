from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import pandas as pd

from src.feature_selection import readable_feature_label


def read_csv_rows(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == expected_columns
        return list(reader)


def create_training_database(database_path: Path, train_rows: int = 40, test_rows: int = 6) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    train_records = [
        _mart_record(
            applicant_id=100000 + index,
            source_population="application_train",
            target=index % 2,
            index=index,
        )
        for index in range(train_rows)
    ]
    test_records = [
        _mart_record(
            applicant_id=200000 + index,
            source_population="application_test",
            target=None,
            index=train_rows + index,
        )
        for index in range(test_rows)
    ]
    mart = pd.DataFrame(train_records + test_records)
    staging_train = mart.loc[mart["source_population"] == "application_train", ["SK_ID_CURR", "TARGET"]]
    staging_test = mart.loc[mart["source_population"] == "application_test", ["SK_ID_CURR"]]
    diagnostics = mart[["SK_ID_CURR", "source_population", "TARGET"]].copy()
    diagnostics["CODE_GENDER"] = ["F" if row % 2 == 0 else "M" for row in range(len(diagnostics))]
    diagnostics["NAME_FAMILY_STATUS"] = "Married"
    diagnostics["applicant_age_years"] = 35
    diagnostics["applicant_age_band"] = "30_to_44"
    diagnostics["CNT_CHILDREN"] = 1
    diagnostics["CNT_FAM_MEMBERS"] = 3

    with duckdb.connect(str(database_path)) as connection:
        _create_table_from_frame(connection, "stg_application_train", staging_train)
        _create_table_from_frame(connection, "stg_application_test", staging_test)
        _create_table_from_frame(
            connection,
            "stg_bureau",
            pd.DataFrame(
                {
                    "SK_ID_BUREAU": range(1, len(mart) + 1),
                    "SK_ID_CURR": mart["SK_ID_CURR"],
                }
            ),
        )
        _create_table_from_frame(
            connection,
            "stg_bureau_balance",
            pd.DataFrame(
                {
                    "SK_ID_BUREAU": range(1, len(mart) + 1),
                    "MONTHS_BALANCE": [0 for _ in range(len(mart))],
                    "STATUS": ["1" if row % 4 == 0 else "0" for row in range(len(mart))],
                }
            ),
        )
        _create_table_from_frame(
            connection,
            "stg_previous_application",
            pd.DataFrame(
                {
                    "SK_ID_PREV": range(1000, 1000 + len(mart)),
                    "SK_ID_CURR": mart["SK_ID_CURR"],
                }
            ),
        )
        _create_table_from_frame(
            connection,
            "stg_pos_cash_balance",
            pd.DataFrame(
                {
                    "SK_ID_PREV": range(1000, 1000 + len(mart)),
                    "SK_ID_CURR": mart["SK_ID_CURR"],
                    "MONTHS_BALANCE": [0 for _ in range(len(mart))],
                    "CNT_INSTALMENT": [12.0 for _ in range(len(mart))],
                    "CNT_INSTALMENT_FUTURE": [6.0 for _ in range(len(mart))],
                    "NAME_CONTRACT_STATUS": [
                        "Active" if row % 2 == 0 else "Completed" for row in range(len(mart))
                    ],
                    "SK_DPD": [1 if row % 5 == 0 else 0 for row in range(len(mart))],
                    "SK_DPD_DEF": [1 if row % 7 == 0 else 0 for row in range(len(mart))],
                }
            ),
        )
        _create_table_from_frame(
            connection,
            "stg_credit_card_balance",
            pd.DataFrame(
                {
                    "SK_ID_PREV": range(2000, 2000 + len(mart)),
                    "SK_ID_CURR": mart["SK_ID_CURR"],
                    "MONTHS_BALANCE": [0 for _ in range(len(mart))],
                    "AMT_BALANCE": [100.0 + row for row in range(len(mart))],
                    "AMT_CREDIT_LIMIT_ACTUAL": [1000.0 for _ in range(len(mart))],
                    "AMT_DRAWINGS_CURRENT": [10.0 if row % 3 == 0 else 0.0 for row in range(len(mart))],
                    "AMT_INST_MIN_REGULARITY": [20.0 for _ in range(len(mart))],
                    "AMT_PAYMENT_CURRENT": [20.0 for _ in range(len(mart))],
                    "AMT_PAYMENT_TOTAL_CURRENT": [20.0 for _ in range(len(mart))],
                    "AMT_TOTAL_RECEIVABLE": [100.0 + row for row in range(len(mart))],
                    "CNT_DRAWINGS_CURRENT": [1.0 if row % 3 == 0 else 0.0 for row in range(len(mart))],
                    "NAME_CONTRACT_STATUS": [
                        "Active" if row % 2 == 0 else "Completed" for row in range(len(mart))
                    ],
                    "SK_DPD": [1 if row % 6 == 0 else 0 for row in range(len(mart))],
                    "SK_DPD_DEF": [1 if row % 8 == 0 else 0 for row in range(len(mart))],
                }
            ),
        )
        _create_table_from_frame(
            connection,
            "stg_installments_payments",
            pd.DataFrame({"SK_ID_CURR": mart["SK_ID_CURR"]}),
        )
        _create_table_from_frame(
            connection,
            "f_applicant_static",
            mart[
                [
                    "SK_ID_CURR",
                    "source_population",
                    "TARGET",
                    "credit_to_income_ratio",
                    "category_feature",
                ]
            ],
        )
        _create_table_from_frame(connection, "segment_diagnostics", diagnostics)
        _create_table_from_frame(
            connection,
            "f_bureau_agg",
            mart[["SK_ID_CURR", "bureau_credit_count"]],
        )
        _create_table_from_frame(
            connection,
            "f_bureau_balance_agg",
            mart[
                [
                    "SK_ID_CURR",
                    "bureau_balance_month_count",
                    "bureau_balance_dpd_1plus_rate",
                    "bureau_balance_recent_dpd_1plus_rate",
                ]
            ],
        )
        _create_table_from_frame(
            connection,
            "f_pos_cash_agg",
            mart[
                [
                    "SK_ID_CURR",
                    "pos_cash_month_count",
                    "pos_cash_dpd_month_rate",
                    "pos_cash_recent_dpd_month_rate",
                ]
            ],
        )
        _create_table_from_frame(
            connection,
            "f_credit_card_agg",
            mart[
                [
                    "SK_ID_CURR",
                    "credit_card_month_count",
                    "credit_card_avg_credit_utilization",
                    "credit_card_dpd_month_rate",
                ]
            ],
        )
        _create_table_from_frame(
            connection,
            "f_previous_application_agg",
            mart[["SK_ID_CURR", "previous_application_count"]],
        )
        _create_table_from_frame(
            connection,
            "f_installments_agg",
            mart[["SK_ID_CURR", "payment_amount_ratio"]],
        )
        _create_table_from_frame(connection, "mart_credit_risk_features", mart)


def write_feature_importance(path: Path, feature_columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["model_version", "feature_name", "importance_type", "importance_value", "rank"],
        )
        writer.writeheader()
        for rank, feature_name in enumerate(feature_columns, start=1):
            writer.writerow(
                {
                    "model_version": "lightgbm_credit_risk_v1",
                    "feature_name": readable_feature_label(feature_name),
                    "importance_type": "mean_abs_shap",
                    "importance_value": 1 / rank,
                    "rank": rank,
                }
            )


def _mart_record(
    applicant_id: int,
    source_population: str,
    target: int | None,
    index: int,
) -> dict[str, object]:
    return {
        "SK_ID_CURR": applicant_id,
        "source_population": source_population,
        "TARGET": target,
        "credit_to_income_ratio": 1.0 + index / 100.0,
        "bureau_credit_count": index % 5 + 1,
        "bureau_balance_month_count": index % 6 + 1,
        "bureau_balance_dpd_1plus_rate": (index % 4) / 10.0,
        "bureau_balance_recent_dpd_1plus_rate": (index % 3) / 10.0,
        "pos_cash_month_count": index % 8 + 1,
        "pos_cash_dpd_month_rate": (index % 5) / 10.0,
        "pos_cash_recent_dpd_month_rate": (index % 4) / 10.0,
        "credit_card_month_count": index % 7 + 1,
        "credit_card_avg_credit_utilization": (index % 6) / 10.0,
        "credit_card_dpd_month_rate": (index % 4) / 10.0,
        "payment_amount_ratio": 0.75 + (index % 7) / 20.0,
        "previous_application_count": index % 3 + 1,
        "category_feature": ["low", "medium", "high"][index % 3],
        "optional_numeric_feature": None if index % 6 == 0 else index / 10.0,
    }


def _create_table_from_frame(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    connection.register("table_frame", frame)
    try:
        connection.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM table_frame')
    finally:
        connection.unregister("table_frame")
