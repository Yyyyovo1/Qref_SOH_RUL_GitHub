"""Cell-first metric calculations for the frozen SOH benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.size < 2:
        return np.nan
    centered = observed - float(np.mean(observed))
    denominator = float(np.sum(centered * centered))
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    numerator = float(np.sum((observed - predicted) ** 2))
    return 1.0 - numerator / denominator


def calculate_metrics(
    observed_soh: np.ndarray,
    predicted_soh: np.ndarray,
    observed_capacity: np.ndarray,
    predicted_capacity: np.ndarray,
) -> dict[str, float]:
    observed_soh = np.asarray(observed_soh, dtype=float)
    predicted_soh = np.asarray(predicted_soh, dtype=float)
    observed_capacity = np.asarray(observed_capacity, dtype=float)
    predicted_capacity = np.asarray(predicted_capacity, dtype=float)
    soh_error = predicted_soh - observed_soh
    capacity_error = predicted_capacity - observed_capacity
    mae_soh = float(np.mean(np.abs(soh_error)))
    rmse_soh = float(np.sqrt(np.mean(soh_error**2)))
    mae_q = float(np.mean(np.abs(capacity_error)))
    rmse_q = float(np.sqrt(np.mean(capacity_error**2)))
    return {
        "MAE_SOH_fraction": mae_soh,
        "RMSE_SOH_fraction": rmse_soh,
        "R2_SOH": safe_r2(observed_soh, predicted_soh),
        "MAE_SOH_percentage_points": 100.0 * mae_soh,
        "MAE_Q_Ah": mae_q,
        "RMSE_Q_Ah": rmse_q,
        "R2_Q": safe_r2(observed_capacity, predicted_capacity),
    }


def cell_metric_rows(
    predictions: pd.DataFrame,
    *,
    analysis_id: str,
    dataset: str,
    qref_definition: str,
    model_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell_id, group in predictions.groupby("Cell_ID", sort=True):
        metrics = calculate_metrics(
            group["Observed_SOH_Fraction"].to_numpy(dtype=float),
            group["Predicted_SOH_Fraction"].to_numpy(dtype=float),
            group["Observed_Capacity_Ah"].to_numpy(dtype=float),
            group["Predicted_Capacity_Ah"].to_numpy(dtype=float),
        )
        qref_values = group["Qref_Ah"].to_numpy(dtype=float)
        rows.append(
            {
                "Analysis_ID": analysis_id,
                "Dataset": dataset,
                "Batch_ID": group["Batch_ID"].iloc[0] if "Batch_ID" in group else "",
                "Cell_ID": cell_id,
                "Fold_ID": int(group["Fold_ID"].iloc[0]),
                "Qref_Definition": qref_definition,
                "Qref_Ah": float(qref_values[0]) if np.allclose(qref_values, qref_values[0]) else np.nan,
                "Model": model_name,
                "Observation_Count": len(group),
                **metrics,
                "Aggregation_Unit": "HELD_OUT_CELL_FIRST",
            }
        )
    return rows


def summarize_cell_metrics(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["Analysis_ID", "Dataset", "Qref_Definition", "Model"]
    for keys, group in cell_metrics.groupby(group_columns, sort=False):
        analysis_id, dataset, qref, model = keys
        mae_q = group["MAE_Q_Ah"]
        q1 = float(mae_q.quantile(0.25))
        q3 = float(mae_q.quantile(0.75))
        rows.append(
            {
                "Analysis_ID": analysis_id,
                "Dataset": dataset,
                "Qref_Definition": qref,
                "Model": model,
                "Cell_Count": int(group["Cell_ID"].nunique()),
                "Observation_Count": int(group["Observation_Count"].sum()),
                "Median_MAE_Q_Ah": float(mae_q.median()),
                "Q1_MAE_Q_Ah": q1,
                "Q3_MAE_Q_Ah": q3,
                "IQR_MAE_Q_Ah": q3 - q1,
                "Mean_MAE_Q_Ah": float(mae_q.mean()),
                "Median_RMSE_Q_Ah": float(group["RMSE_Q_Ah"].median()),
                "Median_R2_Q": float(group["R2_Q"].median()),
                "Median_MAE_SOH_fraction": float(group["MAE_SOH_fraction"].median()),
                "Median_MAE_SOH_percentage_points": float(
                    group["MAE_SOH_percentage_points"].median()
                ),
                "Median_RMSE_SOH_fraction": float(group["RMSE_SOH_fraction"].median()),
                "Median_R2_SOH": float(group["R2_SOH"].median()),
                "Primary_Ranking_Metric": "Median_MAE_Q_Ah",
                "Aggregation_Order": "CELL_FIRST_THEN_CROSS_CELL_SUMMARY",
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "calculate_metrics",
    "cell_metric_rows",
    "safe_r2",
    "summarize_cell_metrics",
]
