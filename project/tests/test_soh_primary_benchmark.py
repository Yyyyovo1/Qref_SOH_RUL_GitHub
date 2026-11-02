from __future__ import annotations

import numpy as np
import pandas as pd

from soh_primary_benchmark_repro.config import (
    ALL_EXTRACTED_FEATURES,
    PRIMARY_FEATURES,
    PROTOCOL_SHORTCUT_FEATURES,
)
from soh_primary_benchmark_repro.metrics import (
    calculate_metrics,
    summarize_cell_metrics,
)
from soh_primary_benchmark_repro.pipeline import (
    _align_targets,
    _eligible_mask,
    _feature_freeze,
    _model_rankings,
    _prepare_fold_transforms,
)


def _synthetic_matrix() -> pd.DataFrame:
    rows = []
    for index in range(5):
        row = {
            "Dataset": "HUST",
            "Cell_ID": f"cell-{index + 1}",
            "Raw_Cycle_Index": 1,
            "Fold_ID": index + 1,
            "Physical_Observation_Key": f"HUST|cell-{index + 1}|1",
        }
        row.update({feature: float(index + offset + 1) for offset, feature in enumerate(PRIMARY_FEATURES)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_primary_feature_allowlist_is_exact_a9() -> None:
    assert len(PRIMARY_FEATURES) == 9
    assert set(PRIMARY_FEATURES).isdisjoint(PROTOCOL_SHORTCUT_FEATURES)
    assert tuple(feature for feature in ALL_EXTRACTED_FEATURES if feature in PRIMARY_FEATURES) == PRIMARY_FEATURES


def test_feature_freeze_retains_but_excludes_shortcuts() -> None:
    shortcut = pd.DataFrame(
        {
            "Feature_Name": list(PROTOCOL_SHORTCUT_FEATURES),
            "Between_To_Overall_Variance_Ratio": [0.7735, 0.9967],
        }
    )
    freeze = _feature_freeze(shortcut)
    assert len(freeze) == len(ALL_EXTRACTED_FEATURES)
    included = freeze.loc[
        freeze["Primary_Status"].eq("INCLUDED_PRIMARY_FEATURE_SET_A9"),
        "Feature_Name",
    ].tolist()
    assert tuple(included) == PRIMARY_FEATURES
    excluded = freeze.loc[
        freeze["Primary_Status"].str.startswith("EXCLUDED_PRIMARY"),
        "Feature_Name",
    ].tolist()
    assert set(excluded) == set(PROTOCOL_SHORTCUT_FEATURES)


def test_complete_case_mask_is_feature_only_and_explicit() -> None:
    matrix = _synthetic_matrix()
    matrix.loc[2, "CV_Duration_s"] = np.nan
    mask = _eligible_mask(matrix, "HUST")
    assert mask["Mask_Qref_Independent"].eq("YES").all()
    assert mask["Model_Eligible"].tolist() == ["YES", "YES", "NO", "YES", "YES"]
    assert mask.loc[2, "Missing_Primary_Features"] == "CV_Duration_s"
    assert mask.loc[2, "Exclusion_Reason"] == "INCOMPLETE_PRIMARY_A9_NO_IMPUTATION"


def test_target_alignment_preserves_matrix_order_and_keys() -> None:
    matrix = _synthetic_matrix()
    label_rows = []
    for qref, qref_ah in (("nominal", 2.0), ("first_comparable", 1.9)):
        for row in matrix.iloc[::-1].itertuples(index=False):
            capacity = 1.5 + 0.01 * row.Raw_Cycle_Index
            label_rows.append(
                {
                    "Cell_ID": row.Cell_ID,
                    "Raw_Cycle_Index": row.Raw_Cycle_Index,
                    "Qref_Definition": qref,
                    "Qref_Ah": qref_ah,
                    "SOH_Fraction": capacity / qref_ah,
                    "Capacity_Ah": capacity,
                }
            )
    labels = pd.DataFrame(label_rows)
    aligned, hashes = _align_targets(
        matrix, labels, "HUST", ("nominal", "first_comparable")
    )
    assert len(set(hashes.values())) == 1
    assert aligned["nominal"]["Physical_Observation_Key"].tolist() == matrix[
        "Physical_Observation_Key"
    ].tolist()
    assert np.allclose(
        aligned["first_comparable"]["SOH_Fraction"],
        aligned["first_comparable"]["Capacity_Ah"]
        / aligned["first_comparable"]["Qref_Ah"],
    )


def test_fold_preprocessing_has_no_cell_overlap_and_train_only_statistics() -> None:
    matrix = _synthetic_matrix()
    transforms = _prepare_fold_transforms(matrix)
    assert set(transforms) == {1, 2, 3, 4, 5}
    for fold_id, transform in transforms.items():
        train_cells = set(matrix.iloc[transform.train_indices]["Cell_ID"])
        test_cells = set(matrix.iloc[transform.test_indices]["Cell_ID"])
        assert train_cells.isdisjoint(test_cells)
        expected_median = np.median(
            matrix.iloc[transform.train_indices][list(PRIMARY_FEATURES)].to_numpy(),
            axis=0,
        )
        from soh_primary_benchmark_repro.pipeline import _sha256_array

        assert transform.imputer_statistics_sha256 == _sha256_array(expected_median)
        assert set(matrix.iloc[transform.test_indices]["Fold_ID"]) == {fold_id}


def test_physical_backtransform_metrics_are_exact() -> None:
    observed_soh = np.array([0.9, 0.8])
    predicted_soh = np.array([0.85, 0.75])
    qref = 2.0
    observed_q = observed_soh * qref
    predicted_q = predicted_soh * qref
    metrics = calculate_metrics(observed_soh, predicted_soh, observed_q, predicted_q)
    assert np.isclose(metrics["MAE_SOH_fraction"], 0.05)
    assert np.isclose(metrics["MAE_Q_Ah"], 0.1)
    assert np.isclose(metrics["MAE_SOH_percentage_points"], 5.0)


def test_rankings_use_median_cell_mae_q_ascending() -> None:
    cells = pd.DataFrame(
        {
            "Analysis_ID": ["a"] * 6,
            "Dataset": ["XJTU"] * 6,
            "Qref_Definition": ["nominal"] * 6,
            "Model": [
                "Ridge Regression",
                "Ridge Regression",
                "Random Forest Regressor",
                "Random Forest Regressor",
                "Gradient Boosting Regressor",
                "Gradient Boosting Regressor",
            ],
            "Cell_ID": ["c1", "c2"] * 3,
            "Observation_Count": [1] * 6,
            "MAE_Q_Ah": [0.3, 0.3, 0.1, 0.1, 0.2, 0.2],
            "RMSE_Q_Ah": [0.3, 0.3, 0.1, 0.1, 0.2, 0.2],
            "R2_Q": [0.0] * 6,
            "MAE_SOH_fraction": [0.15, 0.15, 0.05, 0.05, 0.1, 0.1],
            "MAE_SOH_percentage_points": [15, 15, 5, 5, 10, 10],
            "RMSE_SOH_fraction": [0.15, 0.15, 0.05, 0.05, 0.1, 0.1],
            "R2_SOH": [0.0] * 6,
        }
    )
    summary = summarize_cell_metrics(cells)
    rankings = _model_rankings(summary)
    assert rankings.sort_values("Rank_Position")["Model"].tolist() == [
        "Random Forest Regressor",
        "Gradient Boosting Regressor",
        "Ridge Regression",
    ]
    assert rankings["Ranking_Metric"].eq("Median held-out-cell MAE_Q_Ah").all()
