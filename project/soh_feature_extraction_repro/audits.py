"""Descriptive, non-exclusion audits for extracted SAFE_CORE features."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CONSTANT_ABS_TOLERANCE,
    CONSTANT_REL_TOLERANCE,
    FEATURE_NAMES,
    NEAR_CONSTANT_ABS_TOLERANCE,
    NEAR_CONSTANT_MODE_FRACTION,
    NEAR_CONSTANT_REL_IQR_TOLERANCE,
    NEAR_TARGET_PROXY_ABS_SPEARMAN,
    OUTLIER_MAD_MULTIPLIER,
    PROTOCOL_DOMINANCE_RATIO_THRESHOLD,
)


def _finite_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric[np.isfinite(numeric.to_numpy(dtype=float))]


def variation_statistics(values: pd.Series) -> dict[str, Any]:
    series = _finite_series(values)
    if series.empty:
        return {
            "Valid_Count": 0,
            "Min": np.nan,
            "Q1": np.nan,
            "Median": np.nan,
            "Q3": np.nan,
            "Max": np.nan,
            "Std": np.nan,
            "Unique_Value_Count": 0,
            "Mode_Fraction_Rounded_12dp": np.nan,
            "Variation_Flag": "NO_VALID_VALUES",
        }
    q1, median, q3 = [float(value) for value in series.quantile([0.25, 0.5, 0.75])]
    minimum = float(series.min())
    maximum = float(series.max())
    scale = max(1.0, abs(median))
    constant_tol = max(CONSTANT_ABS_TOLERANCE, CONSTANT_REL_TOLERANCE * scale)
    near_tol = max(
        NEAR_CONSTANT_ABS_TOLERANCE,
        NEAR_CONSTANT_REL_IQR_TOLERANCE * scale,
    )
    rounded = series.round(12)
    mode_fraction = float(rounded.value_counts(normalize=True, dropna=False).iloc[0])
    if maximum - minimum <= constant_tol:
        flag = "CONSTANT"
    elif (q3 - q1 <= near_tol) or mode_fraction >= NEAR_CONSTANT_MODE_FRACTION:
        flag = "NEAR_CONSTANT"
    else:
        flag = "VARIABLE"
    return {
        "Valid_Count": int(len(series)),
        "Min": minimum,
        "Q1": q1,
        "Median": median,
        "Q3": q3,
        "Max": maximum,
        "Std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        "Unique_Value_Count": int(series.nunique(dropna=True)),
        "Mode_Fraction_Rounded_12dp": mode_fraction,
        "Variation_Flag": flag,
    }


def distribution_summary(matrix: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_NAMES:
        stats = variation_statistics(matrix[feature])
        rows.append(
            {
                "Dataset": dataset,
                "Feature_Name": feature,
                "Total_Observation_Count": len(matrix),
                "Missing_Value_Count": int(matrix[feature].isna().sum()),
                "Missing_Rate": float(matrix[feature].isna().mean()),
                **stats,
                "Constant_Criterion": (
                    "range<=max(1e-12,1e-9*max(1,abs(median)))"
                ),
                "Near_Constant_Criterion": (
                    "IQR<=max(1e-9,1e-4*max(1,abs(median))) OR rounded-12dp mode fraction>=0.99"
                ),
                "Automatic_Removal": "NO",
                "CV_Duration_Sensitivity_Flag": (
                    "PRIMARY_FEATURE_WITH_PROXY_SENSITIVITY_CHECK"
                    if feature == "CV_Duration_s"
                    else "NO"
                ),
                "Planned_Ablation": (
                    "FEATURE_SET_A_MINUS_CV_DURATION"
                    if feature == "CV_Duration_s"
                    else "NOT_APPLICABLE"
                ),
            }
        )
    return pd.DataFrame(rows)


def distribution_by_xjtu_batch(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch_id, group in matrix.groupby("Batch_ID", sort=True):
        for feature in FEATURE_NAMES:
            stats = variation_statistics(group[feature])
            rows.append(
                {
                    "Dataset": "XJTU",
                    "Batch_ID": int(batch_id),
                    "Feature_Name": feature,
                    "Total_Observation_Count": len(group),
                    "Missing_Value_Count": int(group[feature].isna().sum()),
                    "Missing_Rate": float(group[feature].isna().mean()),
                    **stats,
                    "Automatic_Removal": "NO",
                }
            )
    return pd.DataFrame(rows)


def protocol_shortcut_audit(xjtu_matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_NAMES:
        frame = xjtu_matrix[["Batch_ID", feature]].copy()
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        frame = frame.dropna(subset=[feature])
        n = len(frame)
        if n < 2:
            rows.append(
                {
                    "Dataset": "XJTU",
                    "Feature_Name": feature,
                    "Valid_Count": n,
                    "Overall_Variance": np.nan,
                    "Pooled_Within_Batch_Variance_Component": np.nan,
                    "Median_Within_Batch_Variance": np.nan,
                    "Min_Within_Batch_Variance": np.nan,
                    "Max_Within_Batch_Variance": np.nan,
                    "Between_Batch_Variance_Component": np.nan,
                    "Between_To_Overall_Variance_Ratio": np.nan,
                    "Protocol_Dominance_Flag": "UNAVAILABLE",
                    "Dominance_Threshold": PROTOCOL_DOMINANCE_RATIO_THRESHOLD,
                    "Automatic_Exclusion": "NO",
                }
            )
            continue
        overall_mean = float(frame[feature].mean())
        overall_variance = float(frame[feature].var(ddof=1))
        within_ss = 0.0
        between_ss = 0.0
        batch_variances: list[float] = []
        for _, group in frame.groupby("Batch_ID", sort=True):
            count = len(group)
            mean = float(group[feature].mean())
            variance = float(group[feature].var(ddof=1)) if count > 1 else 0.0
            within_ss += max(count - 1, 0) * variance
            between_ss += count * (mean - overall_mean) ** 2
            batch_variances.append(variance)
        denominator = n - 1
        within_component = within_ss / denominator
        between_component = between_ss / denominator
        ratio = (
            between_component / overall_variance
            if np.isfinite(overall_variance) and overall_variance > 0
            else np.nan
        )
        rows.append(
            {
                "Dataset": "XJTU",
                "Feature_Name": feature,
                "Valid_Count": n,
                "Overall_Variance": overall_variance,
                "Pooled_Within_Batch_Variance_Component": within_component,
                "Median_Within_Batch_Variance": float(np.median(batch_variances)),
                "Min_Within_Batch_Variance": float(np.min(batch_variances)),
                "Max_Within_Batch_Variance": float(np.max(batch_variances)),
                "Between_Batch_Variance_Component": between_component,
                "Between_To_Overall_Variance_Ratio": ratio,
                "Protocol_Dominance_Flag": (
                    "STRONGLY_BATCH_DOMINATED"
                    if np.isfinite(ratio) and ratio >= PROTOCOL_DOMINANCE_RATIO_THRESHOLD
                    else "NOT_STRONGLY_BATCH_DOMINATED"
                ),
                "Dominance_Threshold": PROTOCOL_DOMINANCE_RATIO_THRESHOLD,
                "Automatic_Exclusion": "NO",
            }
        )
    return pd.DataFrame(rows)


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan, int(len(frame))
    rho = frame["x"].rank(method="average").corr(frame["y"].rank(method="average"))
    return float(rho), int(len(frame))


def target_association_audit(
    matrix: pd.DataFrame,
    nominal_labels: pd.DataFrame,
    dataset: str,
    key_columns: list[str],
) -> pd.DataFrame:
    labels = nominal_labels[key_columns + ["Capacity_Ah", "SOH_Fraction"]].copy()
    joined = matrix.merge(labels, on=key_columns, how="left", validate="one_to_one")
    if joined["Capacity_Ah"].isna().any() or joined["SOH_Fraction"].isna().any():
        raise RuntimeError(f"{dataset}: nominal target join missing rows")
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_NAMES:
        rho_capacity, n_capacity = _spearman(joined[feature], joined["Capacity_Ah"])
        rho_soh, n_soh = _spearman(joined[feature], joined["SOH_Fraction"])
        extreme = bool(
            (np.isfinite(rho_capacity) and abs(rho_capacity) > NEAR_TARGET_PROXY_ABS_SPEARMAN)
            or (np.isfinite(rho_soh) and abs(rho_soh) > NEAR_TARGET_PROXY_ABS_SPEARMAN)
        )
        rows.append(
            {
                "Dataset": dataset,
                "Feature_Name": feature,
                "Valid_Pair_Count_Capacity": n_capacity,
                "Spearman_Rho_vs_Primary_Capacity_Ah": rho_capacity,
                "Valid_Pair_Count_Nominal_SOH": n_soh,
                "Spearman_Rho_vs_Nominal_SOH": rho_soh,
                "Review_Threshold_Abs_Rho": NEAR_TARGET_PROXY_ABS_SPEARMAN,
                "Association_Flag": (
                    "POTENTIAL_NEAR_TARGET_PROXY_REVIEW" if extreme else "DESCRIPTIVE_ONLY"
                ),
                "Extraction_Influenced_By_Association": "NO",
                "Automatic_Leakage_Classification_Change": "NO",
            }
        )
    return pd.DataFrame(rows)


def complete_case_audit(matrix: pd.DataFrame, dataset: str) -> pd.DataFrame:
    complete = matrix[list(FEATURE_NAMES)].notna().all(axis=1)
    rows: list[dict[str, Any]] = [
        {
            "Dataset": dataset,
            "Batch_ID": "ALL" if dataset == "XJTU" else "",
            "Cell_ID": "ALL",
            "Audit_Level": "DATASET",
            "Total_Observations": len(matrix),
            "Complete_Observations_All_11": int(complete.sum()),
            "Incomplete_Observations": int((~complete).sum()),
            "Completeness_Rate": float(complete.mean()),
            "All_Observations_Complete": "YES" if complete.all() else "NO",
            "Cells_With_All_Observations_Complete": int(
                matrix.assign(_complete=complete)
                .groupby("Cell_ID")["_complete"]
                .all()
                .sum()
            ),
            "Total_Cells": int(matrix["Cell_ID"].nunique()),
            "Automatic_Row_Dropping": "NO",
        }
    ]
    temp = matrix[["Cell_ID"] + (["Batch_ID"] if dataset == "XJTU" else [])].copy()
    temp["_complete"] = complete.to_numpy()
    group_columns = (["Batch_ID"] if dataset == "XJTU" else []) + ["Cell_ID"]
    for keys, group in temp.groupby(group_columns, sort=True):
        if dataset == "XJTU":
            batch_id, cell_id = keys
        else:
            batch_id, cell_id = "", keys[0] if isinstance(keys, tuple) else keys
        rows.append(
            {
                "Dataset": dataset,
                "Batch_ID": batch_id,
                "Cell_ID": cell_id,
                "Audit_Level": "CELL",
                "Total_Observations": len(group),
                "Complete_Observations_All_11": int(group["_complete"].sum()),
                "Incomplete_Observations": int((~group["_complete"]).sum()),
                "Completeness_Rate": float(group["_complete"].mean()),
                "All_Observations_Complete": "YES" if group["_complete"].all() else "NO",
                "Cells_With_All_Observations_Complete": np.nan,
                "Total_Cells": np.nan,
                "Automatic_Row_Dropping": "NO",
            }
        )
    return pd.DataFrame(rows)


def outlier_candidates(matrix: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_NAMES:
        values = _finite_series(matrix[feature])
        if values.empty:
            continue
        median = float(values.median())
        mad = float(np.median(np.abs(values.to_numpy() - median)))
        if not np.isfinite(mad) or mad <= 0:
            continue
        lower = median - OUTLIER_MAD_MULTIPLIER * mad
        upper = median + OUTLIER_MAD_MULTIPLIER * mad
        flagged = matrix.loc[matrix[feature].notna() & ((matrix[feature] < lower) | (matrix[feature] > upper))]
        for row in flagged.itertuples(index=False):
            rows.append(
                {
                    "Dataset": dataset,
                    "Batch_ID": getattr(row, "Batch_ID", ""),
                    "Cell_ID": row.Cell_ID,
                    "Raw_Cycle_Index": row.Raw_Cycle_Index,
                    "Feature_Name": feature,
                    "Feature_Value": getattr(row, feature),
                    "Extraction_Status": "PASS",
                    "Candidate_Type": "ROBUST_EXTREME_VALUE_REVIEW",
                    "Reason_Code": "OUTSIDE_MEDIAN_PLUS_MINUS_10_MAD",
                    "Review_Lower_Bound": lower,
                    "Review_Upper_Bound": upper,
                    "Automatic_Exclusion": "NO",
                }
            )
    return pd.DataFrame(rows)


def missing_mask_sha256(matrix: pd.DataFrame) -> str:
    mask = matrix[list(FEATURE_NAMES)].isna().to_numpy(dtype=np.uint8)
    return hashlib.sha256(mask.tobytes(order="C")).hexdigest()


def feature_values_sha256(matrix: pd.DataFrame) -> str:
    values = matrix[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    normalized = values.copy()
    normalized[np.isnan(normalized)] = np.float64(np.nan)
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


__all__ = [
    "complete_case_audit",
    "distribution_by_xjtu_batch",
    "distribution_summary",
    "feature_values_sha256",
    "missing_mask_sha256",
    "outlier_candidates",
    "protocol_shortcut_audit",
    "target_association_audit",
    "variation_statistics",
]
