from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.calibration import fit_calibrators
from src.mart_access import load_labeled_split_frame
from src.metrics import target_class_values


@pytest.mark.parametrize("invalid_target", [0.5, 1.5, 2, None, float("nan")])
def test_target_classes_reject_invalid_labels_before_integer_conversion(
    invalid_target,
) -> None:
    with pytest.raises(ValueError, match="TARGET.*binary"):
        target_class_values(pd.Series([0, 1, invalid_target]))


def test_binary_numeric_targets_and_optional_missing_filter_remain_supported() -> None:
    assert target_class_values(pd.Series([0.0, 1.0])) == {0, 1}
    assert target_class_values(pd.Series([0, 1, None]), dropna=True) == {0, 1}


@pytest.mark.parametrize("require_both_classes", [True, False])
def test_saved_split_loader_rejects_fractional_target(
    scratch_path: Path, require_both_classes: bool
) -> None:
    with duckdb.connect(str(scratch_path / "labels.duckdb")) as connection:
        connection.execute(
            "CREATE TABLE mart_credit_risk_features AS "
            "SELECT * FROM (VALUES (1, 0.0, 'application_train'), "
            "(2, 1.0, 'application_train'), (3, 0.5, 'application_train')) "
            "AS source(SK_ID_CURR, TARGET, source_population)"
        )
        with pytest.raises(RuntimeError, match="TARGET.*binary"):
            load_labeled_split_frame(
                connection,
                [1, 2, 3],
                [],
                "validation",
                error_cls=RuntimeError,
                require_both_target_classes=require_both_classes,
            )


def test_calibrator_rejects_fractional_target_before_fitting() -> None:
    with pytest.raises(RuntimeError, match="TARGET.*binary"):
        fit_calibrators(
            np.array([0.1, 0.8, 0.3]),
            np.array([0.0, 1.0, 0.5]),
            42,
            error_cls=RuntimeError,
        )
