from __future__ import annotations

import numpy as np
import pandas as pd

from soh_ml_robustness_repro.config import (
    A8_FEATURES,
    B10_FEATURES,
    BOOTSTRAP_REPLICATES,
    POSTCYCLE5_FEATURES,
)
from soh_ml_robustness_repro.pipeline import (
    _bootstrap_median_difference,
    _bootstrap_rank_frequency,
    _model_pair_bootstrap,
    _qref_paired_bootstrap,
)
from soh_primary_benchmark_repro.config import (
    MODEL_NAMES,
    PRIMARY_FEATURES,
    PROTOCOL_SHORTCUT_FEATURES,
    QREF_PAIRS,
)


def _synthetic_primary_cell_metrics() -> pd.DataFrame:
    rows = []
    qrefs = ("nominal", "first_comparable", "early_median5_comparable")
    for dataset, cells in (("XJTU", 5), ("HUST", 6)):
        for qref_index, qref in enumerate(qrefs):
            for model_index, model in enumerate(MODEL_NAMES):
                for cell_index in range(cells):
                    base = 0.01 * (model_index + 1) + 0.001 * cell_index
                    rows.append(
                        {
                            "Analysis_ID": f"{dataset}_PRIMARY",
                            "Dataset": dataset,
                            "Batch_ID": 1 if dataset == "XJTU" else "",
                            "Cell_ID": f"{dataset}_cell_{cell_index + 1}",
                            "Fold_ID": (cell_index % 5) + 1,
                            "Qref_Definition": qref,
                            "Qref_Ah": 2.0,
                            "Model": model,
                            "Observation_Count": 10,
                            "MAE_SOH_fraction": base / 2 + 0.0002 * qref_index,
                            "RMSE_SOH_fraction": base / 2 + 0.001,
                            "R2_SOH": 0.8,
                            "MAE_SOH_percentage_points": 50 * base,
                            "MAE_Q_Ah": base + 0.0005 * qref_index,
                            "RMSE_Q_Ah": base + 0.002,
                            "R2_Q": 0.8,
                            "Aggregation_Unit": "HELD_OUT_CELL_FIRST",
                        }
                    )
    return pd.DataFrame(rows)


def test_sensitivity_feature_sets_are_exact() -> None:
    assert A8_FEATURES == tuple(
        feature for feature in PRIMARY_FEATURES if feature != "CV_Duration_s"
    )
    assert B10_FEATURES == (*PRIMARY_FEATURES, "Raw_Cycle_Index")
    assert POSTCYCLE5_FEATURES == PRIMARY_FEATURES
    all_features = set(A8_FEATURES) | set(B10_FEATURES) | set(POSTCYCLE5_FEATURES)
    assert all_features.isdisjoint(PROTOCOL_SHORTCUT_FEATURES)
    assert not any("capacity" in feature.lower() for feature in all_features)
    assert not any("qref" in feature.lower() for feature in all_features)


def test_bootstrap_median_difference_is_deterministic() -> None:
    differences = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    first = _bootstrap_median_difference(differences, seed=12345)
    second = _bootstrap_median_difference(differences, seed=12345)
    assert first == second
    assert first[0] == 0.0
    assert first[1] <= first[0] <= first[2]


def test_qref_bootstrap_pairs_cells_and_has_complete_grid() -> None:
    metrics = _synthetic_primary_cell_metrics()
    result = _qref_paired_bootstrap(metrics)
    assert len(result) == 2 * len(MODEL_NAMES) * len(QREF_PAIRS) * 2
    assert result["Bootstrap_Replicates"].eq(BOOTSTRAP_REPLICATES).all()
    assert result["Bootstrap_Unit"].eq("Cell_ID").all()
    assert result["Paired_By_Cell"].eq("YES").all()
    assert result["Cycle_Level_Resampling"].eq("NO").all()
    assert set(result.loc[result["Dataset"].eq("XJTU"), "Cell_Count"]) == {5}
    assert set(result.loc[result["Dataset"].eq("HUST"), "Cell_Count"]) == {6}


def test_model_pair_bootstrap_uses_model_b_minus_model_a() -> None:
    metrics = _synthetic_primary_cell_metrics()
    result = _model_pair_bootstrap(metrics)
    assert len(result) == 2 * 3 * 3
    rf_vs_gb = result.loc[
        result["Dataset"].eq("XJTU")
        & result["Qref_Definition"].eq("nominal")
        & result["Model_A"].eq("Random Forest Regressor")
        & result["Model_B"].eq("Gradient Boosting Regressor")
    ].iloc[0]
    assert np.isclose(rf_vs_gb["Median_Difference_ModelB_minus_ModelA"], 0.01)
    assert rf_vs_gb["Paired_By_Cell"] == "YES"


def test_rank_frequency_sums_to_one_without_cycle_resampling() -> None:
    metrics = _synthetic_primary_cell_metrics()
    result = _bootstrap_rank_frequency(metrics)
    assert len(result) == 2 * 3 * len(MODEL_NAMES)
    sums = result.groupby(["Dataset", "Qref_Definition"])[
        "Bootstrap_Rank1_Frequency"
    ].sum()
    assert np.allclose(sums.to_numpy(), 1.0)
    assert result["Cycle_Level_Resampling"].eq("NO").all()
    ridge = result.loc[result["Model"].eq("Ridge Regression")]
    assert ridge["Bootstrap_Rank1_Frequency"].eq(1.0).all()
