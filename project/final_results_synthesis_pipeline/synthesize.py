from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
FINAL_DIR = OUTPUT_ROOT / "final_results_synthesis_v1"
TZ = ZoneInfo("America/Tijuana")


SOURCES = {
    "calce_final": ROOT / "phase1" / "cycle_capacity_final_CS2_35_38.csv",
    "xjtu_qref": OUTPUT_ROOT / "xjtu_55_final_v2" / "xjtu_55_qref_summary.csv",
    "xjtu_cell": OUTPUT_ROOT / "xjtu_55_final_v2" / "xjtu_55_cell_summary.csv",
    "xjtu_regression": OUTPUT_ROOT / "xjtu_55_final_v2" / "xjtu_55_run_regression_tests.csv",
    "hust_qref": OUTPUT_ROOT / "hust_77_final_v1" / "hust_77_qref_summary.csv",
    "hust_cell": OUTPUT_ROOT / "hust_77_final_v1" / "hust_77_cell_summary.csv",
    "hust_primary": OUTPUT_ROOT / "hust_77_final_v1" / "hust_77_primary_capacity.csv",
    "hust_regression": OUTPUT_ROOT / "hust_77_final_v1" / "hust_77_run_regression_tests.csv",
    "cross_cell": OUTPUT_ROOT / "cross_dataset_summary_v1" / "cross_dataset_cell_level.csv",
    "cross_qref": OUTPUT_ROOT / "cross_dataset_summary_v1" / "cross_dataset_qref_summary.csv",
    "cross_censor": OUTPUT_ROOT / "cross_dataset_summary_v1" / "cross_dataset_censoring_summary.csv",
    "cross_rul": OUTPUT_ROOT / "cross_dataset_summary_v1" / "cross_dataset_rul_availability_summary.csv",
    "cross_transition": OUTPUT_ROOT / "cross_dataset_summary_v1" / "cross_dataset_eol_transition_summary.csv",
    "cross_effect": OUTPUT_ROOT / "cross_dataset_summary_v1" / "cross_dataset_effect_bounds.csv",
    "cross_regression": OUTPUT_ROOT / "cross_dataset_summary_v1" / "cross_dataset_regression_tests.csv",
    "threshold_cell": OUTPUT_ROOT / "eol_threshold_robustness_v1" / "eol_threshold_cell_level.csv",
    "threshold_status": OUTPUT_ROOT / "eol_threshold_robustness_v1" / "eol_threshold_status_summary.csv",
    "threshold_transition": OUTPUT_ROOT / "eol_threshold_robustness_v1" / "eol_threshold_transition_summary.csv",
    "threshold_rul": OUTPUT_ROOT / "eol_threshold_robustness_v1" / "eol_threshold_rul_availability.csv",
    "threshold_effect": OUTPUT_ROOT / "eol_threshold_robustness_v1" / "eol_threshold_effect_bounds.csv",
    "threshold_cross": OUTPUT_ROOT / "eol_threshold_robustness_v1" / "eol_threshold_cross_threshold_summary.csv",
    "threshold_stopping": OUTPUT_ROOT / "eol_threshold_robustness_v1" / "eol_threshold_hust_stopping_rule_audit.csv",
    "threshold_regression": OUTPUT_ROOT / "eol_threshold_robustness_v1" / "eol_threshold_regression_tests.csv",
    "ml_primary_summary": OUTPUT_ROOT / "soh_primary_benchmark_v1" / "soh_primary_summary_metrics.csv",
    "ml_primary_rank": OUTPUT_ROOT / "soh_primary_benchmark_v1" / "soh_primary_model_rankings.csv",
    "ml_primary_regression": OUTPUT_ROOT / "soh_primary_benchmark_v1" / "soh_primary_regression_tests.csv",
    "ml_a8": OUTPUT_ROOT / "soh_ml_robustness_v1" / "soh_robustness_a8_summary.csv",
    "ml_b10": OUTPUT_ROOT / "soh_ml_robustness_v1" / "soh_robustness_b10_summary.csv",
    "ml_post5": OUTPUT_ROOT / "soh_ml_robustness_v1" / "soh_robustness_postcycle5_summary.csv",
    "ml_robust_rank": OUTPUT_ROOT / "soh_ml_robustness_v1" / "soh_robustness_rankings.csv",
    "ml_robust_regression": OUTPUT_ROOT / "soh_ml_robustness_v1" / "soh_robustness_regression_tests.csv",
    "a8_cell": OUTPUT_ROOT / "xjtu_a8_deterministic_reconstruction_v1" / "xjtu_a8_cell_metrics.csv",
    "a8_ci": OUTPUT_ROOT / "xjtu_a8_deterministic_reconstruction_v1" / "xjtu_a8_corrected_difference_of_medians_ci.csv",
    "a8_rank_frequency": OUTPUT_ROOT / "xjtu_a8_deterministic_reconstruction_v1" / "xjtu_a8_corrected_rank_frequency.csv",
    "a8_rankflip": OUTPUT_ROOT / "xjtu_a8_deterministic_reconstruction_v1" / "xjtu_a8_corrected_rankflip_frequency.csv",
    "a8_regression": OUTPUT_ROOT / "xjtu_a8_deterministic_reconstruction_v1" / "xjtu_a8_reconstruction_regression_tests.csv",
}


COMMON_QREFS = ["nominal", "first_comparable", "early_median5_comparable"]
QREF_LABELS = {
    "nominal": "Nominal",
    "first_comparable": "First comparable",
    "early_median5_comparable": "Early median-5",
    "initial_test": "Initial test",
}
MODEL_ORDER = [
    "Ridge Regression",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
]
MODEL_SHORT = {
    "Ridge Regression": "Ridge",
    "Random Forest Regressor": "Random Forest",
    "Gradient Boosting Regressor": "Gradient Boosting",
}
MODEL_COLORS = {
    "Ridge Regression": "#6B7280",
    "Random Forest Regressor": "#2563EB",
    "Gradient Boosting Regressor": "#E07A2D",
}
STATUS_ORDER = ["EXACT_OBSERVED", "INTERVAL_CENSORED", "RIGHT_CENSORED"]
STATUS_SHORT = {
    "EXACT_OBSERVED": "Exact",
    "INTERVAL_CENSORED": "Interval",
    "RIGHT_CENSORED": "Right",
}
STATUS_COLORS = {
    "EXACT_OBSERVED": "#2A9D8F",
    "INTERVAL_CENSORED": "#E9C46A",
    "RIGHT_CENSORED": "#8D99AE",
}


@dataclass
class Check:
    test_id: str
    status: str
    observed: str
    expected: str
    source_files: str
    notes: str = ""


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes() -> dict[str, dict[str, object]]:
    return {
        key: {"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for key, path in SOURCES.items()
    }


def load_sources() -> dict[str, pd.DataFrame]:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen source files: " + "; ".join(missing))
    data: dict[str, pd.DataFrame] = {}
    for key, path in SOURCES.items():
        if key == "hust_primary":
            data[key] = read_csv(path, usecols=["Cell_ID"])
        else:
            data[key] = read_csv(path)
    return data


def add_check(
    checks: list[Check],
    test_id: str,
    passed: bool,
    observed: object,
    expected: object,
    source_files: str,
    notes: str = "",
) -> None:
    checks.append(
        Check(
            test_id=test_id,
            status="PASS" if passed else "FAIL",
            observed=str(observed),
            expected=str(expected),
            source_files=source_files,
            notes=notes,
        )
    )


def status_col(df: pd.DataFrame) -> str:
    matches = [c for c in df.columns if c.lower() == "status"]
    if len(matches) != 1:
        raise ValueError(f"Could not identify unique Status column: {matches}")
    return matches[0]


def preflight(data: dict[str, pd.DataFrame]) -> list[Check]:
    checks: list[Check] = []

    regression_keys = [
        "xjtu_regression",
        "hust_regression",
        "cross_regression",
        "threshold_regression",
        "ml_primary_regression",
        "ml_robust_regression",
        "a8_regression",
    ]
    for key in regression_keys:
        d = data[key]
        col = status_col(d)
        passed = bool(d[col].eq("PASS").all())
        add_check(
            checks,
            f"SOURCE_REGRESSION_{key.upper()}",
            passed,
            d[col].value_counts(dropna=False).to_dict(),
            "all PASS",
            str(SOURCES[key].relative_to(ROOT)),
        )

    x_cells = int(data["xjtu_qref"]["Cell_ID"].nunique())
    h_cells = int(data["hust_qref"]["Cell_ID"].nunique())
    add_check(checks, "XJTU_CELL_COUNT_55", x_cells == 55, x_cells, 55, "xjtu_qref")
    add_check(checks, "HUST_CELL_COUNT_77", h_cells == 77, h_cells, 77, "hust_qref")
    add_check(checks, "COMBINED_CELL_COUNT_132", x_cells + h_cells == 132, x_cells + h_cells, 132, "xjtu_qref|hust_qref")

    cross = data["cross_cell"]
    observed_cross_cells = cross.groupby("Dataset")["Cell_ID"].nunique().to_dict()
    add_check(
        checks,
        "CROSS_DATASET_CELL_COUNTS_MATCH_FINALS",
        observed_cross_cells == {"HUST": 77, "XJTU": 55},
        observed_cross_cells,
        {"HUST": 77, "XJTU": 55},
        "cross_cell|xjtu_qref|hust_qref",
    )

    cross_qref = data["cross_qref"]
    th80 = data["threshold_status"].loc[
        data["threshold_status"]["Threshold_Percent"].eq(80)
        & data["threshold_status"]["Qref_Definition"].isin(COMMON_QREFS)
    ].copy()
    merged = th80.merge(
        cross_qref.loc[cross_qref["Qref_Definition"].isin(COMMON_QREFS)],
        on=["Dataset", "Qref_Definition"],
        suffixes=("_threshold", "_cross"),
        validate="one_to_one",
    )
    count_cols = ["EXACT_OBSERVED_Count", "INTERVAL_CENSORED_Count", "RIGHT_CENSORED_Count"]
    diffs = {
        c: int((merged[f"{c}_threshold"] != merged[f"{c}_cross"]).sum()) for c in count_cols
    }
    add_check(
        checks,
        "THRESHOLD_80_STATUS_EXACTLY_REPRODUCES_CROSS_DATASET",
        all(v == 0 for v in diffs.values()) and len(merged) == 6,
        {"rows": len(merged), "mismatches": diffs},
        {"rows": 6, "mismatches": {c: 0 for c in count_cols}},
        "threshold_status|cross_qref",
    )

    cross_trans = data["cross_transition"].rename(columns={"Qref_Definition": "Measured_Qref_Definition"})
    th_trans = data["threshold_transition"].loc[
        data["threshold_transition"]["Threshold_Percent"].eq(80)
    ]
    trans_merge = cross_trans.merge(
        th_trans,
        on=["Dataset", "Measured_Qref_Definition", "Transition_Category"],
        suffixes=("_cross", "_threshold"),
        validate="one_to_one",
    )
    trans_mismatch = int((trans_merge["Cell_Count_cross"] != trans_merge["Cell_Count_threshold"]).sum())
    add_check(
        checks,
        "THRESHOLD_80_TRANSITIONS_EXACTLY_REPRODUCE_CROSS_DATASET",
        len(trans_merge) == 36 and trans_mismatch == 0,
        {"rows": len(trans_merge), "count_mismatches": trans_mismatch},
        {"rows": 36, "count_mismatches": 0},
        "threshold_transition|cross_transition",
    )

    primary = data["ml_primary_summary"]
    ranks = data["ml_primary_rank"]
    rank_merge = primary.merge(
        ranks[["Dataset", "Qref_Definition", "Model", "Rank_Position", "Median_Cell_MAE_Q_Ah"]],
        on=["Dataset", "Qref_Definition", "Model"],
        validate="one_to_one",
    )
    max_metric_diff = float((rank_merge["Median_MAE_Q_Ah"] - rank_merge["Median_Cell_MAE_Q_Ah"]).abs().max())
    expected_rank = (
        primary.sort_values(["Dataset", "Qref_Definition", "Median_MAE_Q_Ah", "Model"])
        .groupby(["Dataset", "Qref_Definition"])
        .cumcount()
        .add(1)
    )
    rank_source = primary.assign(Expected_Rank=expected_rank).merge(
        ranks[["Dataset", "Qref_Definition", "Model", "Rank_Position"]],
        on=["Dataset", "Qref_Definition", "Model"],
        validate="one_to_one",
    )
    rank_mismatches = int((rank_source["Expected_Rank"] != rank_source["Rank_Position"]).sum())
    add_check(
        checks,
        "PRIMARY_ML_MEDIANS_MATCH_FROZEN_RANKINGS",
        max_metric_diff <= 1e-12,
        max_metric_diff,
        "<=1e-12",
        "ml_primary_summary|ml_primary_rank",
    )
    add_check(
        checks,
        "PRIMARY_ML_RANKINGS_MATCH_MEDIAN_CELL_MAE_Q",
        rank_mismatches == 0,
        rank_mismatches,
        0,
        "ml_primary_summary|ml_primary_rank",
    )

    a8_source = data["ml_a8"].loc[data["ml_a8"]["Dataset"].eq("XJTU")]
    a8_rebuilt = (
        data["a8_cell"].groupby(["Qref_Definition", "Model"], as_index=False)["MAE_Q_Ah"].median()
    )
    a8_compare = a8_source.merge(a8_rebuilt, on=["Qref_Definition", "Model"], validate="one_to_one")
    a8_max_diff = float((a8_compare["Median_MAE_Q_Ah"] - a8_compare["MAE_Q_Ah"]).abs().max())
    add_check(
        checks,
        "A8_CELL_METRICS_MATCH_FROZEN_ROBUSTNESS_MEDIANS",
        a8_max_diff <= 1e-12,
        a8_max_diff,
        "<=1e-12",
        "ml_a8|a8_cell",
    )
    a8_ci = data["a8_ci"]
    measured_ci = a8_ci.loc[a8_ci["Qref_Definition"].ne("nominal")]
    a8_interp_ok = bool(
        measured_ci["Interpretation_Classification"]
        .eq("DESCRIPTIVE_RANK_REVERSAL_NOT_BOOTSTRAP_ROBUST")
        .all()
        and measured_ci["CI_Includes_Zero"].eq("YES").all()
    )
    add_check(
        checks,
        "A8_CORRECTED_INTERPRETATION_FROZEN",
        a8_interp_ok,
        measured_ci[["Qref_Definition", "CI_Includes_Zero", "Interpretation_Classification"]].to_dict("records"),
        "measured Qrefs: CI includes zero and descriptive reversal not bootstrap robust",
        "a8_ci",
    )

    no_hust_initial = not bool(
        cross_qref.loc[cross_qref["Dataset"].eq("HUST"), "Qref_Definition"].eq("initial_test").any()
    )
    add_check(checks, "NO_HUST_INITIAL_TEST_INVENTED", no_hust_initial, no_hust_initial, True, "cross_qref")

    point_col = "RUL_Point_Label_Available"
    censored_point = cross.loc[cross["EOL_Status"].ne("EXACT_OBSERVED"), point_col].eq("YES").sum()
    add_check(
        checks,
        "NO_CENSORED_EOL_HAS_POINT_RUL",
        int(censored_point) == 0,
        int(censored_point),
        0,
        "cross_cell",
    )

    cross_effect = data["cross_effect"]
    cross_bad_exact = cross_effect.loc[
        cross_effect["Delta_EOL_Exact_Cycles"].notna()
        & ~(
            cross_effect["EOL_Status_A"].eq("EXACT_OBSERVED")
            & cross_effect["EOL_Status_B"].eq("EXACT_OBSERVED")
        )
    ]
    th_effect = data["threshold_effect"]
    th_bad_exact = th_effect.loc[
        th_effect["Delta_EOL_Cycles"].notna()
        & ~(
            th_effect["EOL_Status_A"].eq("EXACT_OBSERVED")
            & th_effect["EOL_Status_B"].eq("EXACT_OBSERVED")
        )
    ]
    add_check(
        checks,
        "NO_EXACT_EOL_DELTA_FROM_CENSORED_PAIRS",
        cross_bad_exact.empty and th_bad_exact.empty,
        {"cross_bad": len(cross_bad_exact), "threshold_bad": len(th_bad_exact)},
        {"cross_bad": 0, "threshold_bad": 0},
        "cross_effect|threshold_effect",
    )

    return checks


def ensure_preflight_passed(checks: list[Check]) -> None:
    failed = [c for c in checks if c.status != "PASS"]
    if failed:
        print("FINAL RESULTS SYNTHESIS NEEDS REVIEW -")
        print("FROZEN RESULT CONFLICT DETECTED")
        for c in failed:
            print(f"{c.test_id}: observed={c.observed}; expected={c.expected}; sources={c.source_files}")
        raise SystemExit(2)


def clean_zero(v: float | int | None) -> float | int | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    return 0.0 if abs(float(v)) < 1e-12 else v


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, na_rep="", float_format="%.10f", lineterminator="\n")


def dataset_overview(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    calce = data["calce_final"]
    x_cell = data["xjtu_cell"]
    h_cell = data["hust_cell"]
    rows = [
        {
            "Dataset": "CALCE",
            "Cell_Count": int(calce["Cell_ID"].nunique()),
            "Primary_Observation_Count": int(calce["Final_Include"].eq("YES").sum()),
            "Nominal_Qref_Ah": 1.1,
            "Available_Qref_Definitions": "nominal protocol value only; Qref sensitivity not evaluated",
            "Initial_Test_Available_Cell_Count": np.nan,
            "Primary_Evaluation_Context": "Cycle-level discharge-capacity extraction and protocol-integrity QC",
            "Observation_Density": "Cycle-level records with explicit file-boundary and protocol QC",
            "EOL_Censoring_Context": "NOT_EVALUATED_IN_FROZEN_CROSS_DATASET_EOL_ANALYSIS",
            "Dataset_Stopping_Rule_Context": "Discharge cutoff near 2.7 V; no frozen cross-dataset EOL endpoint analysis",
            "Primary_Role_In_Study": "Method-development, QC, and reproducibility support",
        },
        {
            "Dataset": "XJTU",
            "Cell_Count": int(x_cell["Cell_ID"].nunique()),
            "Primary_Observation_Count": int(x_cell["Primary_Capacity_Observation_Count"].sum()),
            "Nominal_Qref_Ah": 2.0,
            "Available_Qref_Definitions": "nominal|initial_test|first_comparable|early_median5_comparable",
            "Initial_Test_Available_Cell_Count": int(data["xjtu_qref"]["Qref_initial_test_Status"].eq("AVAILABLE").sum()),
            "Primary_Evaluation_Context": "Primary multi-protocol benchmark of Qref-induced label and censoring sensitivity",
            "Observation_Density": "Mixed density: dense Batch-1 cycles and periodic diagnostics in Batches 2-6",
            "EOL_Censoring_Context": "Exact, interval-censored, and right-censored EOL retained",
            "Dataset_Stopping_Rule_Context": "Frozen multi-batch observation windows with sparse diagnostic sampling in some batches",
            "Primary_Role_In_Study": "Primary benchmark",
        },
        {
            "Dataset": "HUST",
            "Cell_Count": int(h_cell["Cell_ID"].nunique()),
            "Primary_Observation_Count": int(h_cell["Primary_Capacity_Eligible_Count"].sum()),
            "Nominal_Qref_Ah": 1.1,
            "Available_Qref_Definitions": "nominal|first_comparable|early_median5_comparable",
            "Initial_Test_Available_Cell_Count": 0,
            "Primary_Evaluation_Context": "External mechanism-level replication under a different protocol and stopping-rule context",
            "Observation_Density": "Dense cycle-level capacity observations",
            "EOL_Censoring_Context": "Exact or right-censored EOL in the frozen threshold analyses",
            "Dataset_Stopping_Rule_Context": "Released observation window ends near the nominal 80% region",
            "Primary_Role_In_Study": "External mechanism-level replication; not an exact initial-test replication",
        },
    ]
    return pd.DataFrame(rows)


def qref_effect_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    src = data["cross_qref"].copy()

    def direction(row: pd.Series) -> str:
        available = int(row["Qref_Available_Count"])
        if int(row["QREF_ABOVE_NOMINAL_Count"]) == available:
            return "ABOVE_NOMINAL"
        if int(row["QREF_BELOW_NOMINAL_Count"]) == available:
            return "BELOW_NOMINAL"
        if int(row["QREF_EQUAL_NOMINAL_Count"]) == available:
            return "EQUAL"
        return "MIXED"

    out = pd.DataFrame(
        {
            "Dataset": src["Dataset"],
            "Qref_Definition": src["Qref_Definition"],
            "Comparison_Scope": src["Comparison_Scope"],
            "Cell_Count": src["Dataset_Cell_Count"].astype(int),
            "Qref_Available_Count": src["Qref_Available_Count"].astype(int),
            "Delta_Qref_vs_Nominal_Ah_Min": src["Delta_Qref_vs_Nominal_Ah_Min"].map(clean_zero),
            "Delta_Qref_vs_Nominal_Ah_Median": src["Delta_Qref_vs_Nominal_Ah_Median"].map(clean_zero),
            "Delta_Qref_vs_Nominal_Ah_Max": src["Delta_Qref_vs_Nominal_Ah_Max"].map(clean_zero),
            "Delta_Qref_vs_Nominal_Pct_Min": src["Delta_Qref_vs_Nominal_Pct_Min"].map(clean_zero),
            "Delta_Qref_vs_Nominal_Pct_Median": src["Delta_Qref_vs_Nominal_Pct_Median"].map(clean_zero),
            "Delta_Qref_vs_Nominal_Pct_Max": src["Delta_Qref_vs_Nominal_Pct_Max"].map(clean_zero),
            "Qref_Direction": src.apply(direction, axis=1),
            "80pct_Threshold_Shift_Ah_Min": src["Delta_Threshold_vs_Nominal_Ah_Min"].map(clean_zero),
            "80pct_Threshold_Shift_Ah_Median": src["Delta_Threshold_vs_Nominal_Ah_Median"].map(clean_zero),
            "80pct_Threshold_Shift_Ah_Max": src["Delta_Threshold_vs_Nominal_Ah_Max"].map(clean_zero),
        }
    )
    order = pd.Categorical(out["Dataset"], ["XJTU", "HUST"], ordered=True)
    qorder = pd.Categorical(out["Qref_Definition"], ["nominal", "initial_test", "first_comparable", "early_median5_comparable"], ordered=True)
    return out.assign(_d=order, _q=qorder).sort_values(["_d", "_q"]).drop(columns=["_d", "_q"]).reset_index(drop=True)


def eol_80_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    status = data["threshold_status"].loc[
        data["threshold_status"]["Threshold_Percent"].eq(80)
        & data["threshold_status"]["Qref_Definition"].isin(COMMON_QREFS)
    ].copy()
    effect = data["cross_effect"]
    rows: list[dict[str, object]] = []
    for _, row in status.iterrows():
        dataset = row["Dataset"]
        qref = row["Qref_Definition"]
        if qref == "nominal":
            status_changed = 0
            rul_changed = 0
            exact_valid = 0
            lower_valid = 0
            exact_median = np.nan
            earlier_median = np.nan
            comparison = "REFERENCE_BASELINE"
        else:
            g = effect.loc[(effect["Dataset"] == dataset) & (effect["Definition_B"] == qref)]
            status_changed = int((g["EOL_Status_A"] != g["EOL_Status_B"]).sum())
            rul_changed = int(
                ((g["EOL_Status_A"] == "EXACT_OBSERVED") != (g["EOL_Status_B"] == "EXACT_OBSERVED")).sum()
            )
            exact_series = pd.to_numeric(g["Delta_EOL_Exact_Cycles"], errors="coerce")
            lower_series = pd.to_numeric(g["Minimum_Observed_Earlier_By_Cycles"], errors="coerce")
            exact_valid = int(exact_series.notna().sum())
            lower_valid = int(lower_series.notna().sum())
            exact_median = exact_series.median() if exact_valid else np.nan
            earlier_median = lower_series.median() if lower_valid else np.nan
            comparison = "VERSUS_NOMINAL"
        rows.append(
            {
                "Dataset": dataset,
                "Qref_Definition": qref,
                "Cell_Count": int(row["Cell_Count"]),
                "EXACT_Count": int(row["EXACT_OBSERVED_Count"]),
                "INTERVAL_Count": int(row["INTERVAL_CENSORED_Count"]),
                "RIGHT_Count": int(row["RIGHT_CENSORED_Count"]),
                "Point_RUL_Available_Count": int(row["Point_RUL_Available_Count"]),
                "Point_RUL_Available_Fraction": float(row["Point_RUL_Available_Fraction"]),
                "Comparison_vs_Nominal": comparison,
                "Status_Changed_Count": status_changed,
                "RUL_Availability_Changed_Count": rul_changed,
                "Exact_Delta_Valid_Count": exact_valid,
                "Lower_Bound_Valid_Count": lower_valid,
                "Exact_Delta_EOL_Cycles_Median": exact_median,
                "Minimum_Observed_Earlier_By_Cycles_Median": earlier_median,
            }
        )
    out = pd.DataFrame(rows)
    out["_d"] = pd.Categorical(out["Dataset"], ["XJTU", "HUST"], ordered=True)
    out["_q"] = pd.Categorical(out["Qref_Definition"], COMMON_QREFS, ordered=True)
    return out.sort_values(["_d", "_q"]).drop(columns=["_d", "_q"]).reset_index(drop=True)


def observation_window_label(dataset: str, threshold: int) -> str:
    if dataset == "HUST":
        return {
            75: "IMPORTANT_FINITE_RELEASED_WINDOW",
            80: "IMPORTANT_NEAR_RELEASED_STOPPING_RULE",
            85: "REDUCED_ALL_COMMON_QREFS_EXACT",
        }[threshold]
    return {
        75: "MOSTLY_OBSERVATION_WINDOW_LIMITED",
        80: "MIXED_EXACT_INTERVAL_RIGHT_INFORMATION",
        85: "MORE_EXACT_OR_INTERVAL_INFORMATION_AVAILABLE",
    }[threshold]


def threshold_robustness_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    status = data["threshold_status"].copy()
    cross = data["threshold_cross"].copy()
    rows: list[dict[str, object]] = []
    for _, row in status.iterrows():
        dataset = row["Dataset"]
        threshold = int(row["Threshold_Percent"])
        qref = row["Qref_Definition"]
        if qref == "nominal":
            effect_count = status_changed = rul_changed = exact_shift = 0
            both_right = int(row["RIGHT_CENSORED_Count"])
            classification = "REFERENCE_BASELINE"
            comparison = "SELF_BASELINE"
        else:
            c = cross.loc[(cross["Dataset"] == dataset) & (cross["Measured_Qref_Definition"] == qref)]
            if len(c) != 1:
                raise ValueError(f"Missing unique cross-threshold row for {dataset} {qref}")
            c = c.iloc[0]
            effect_count = int(c[f"Qref_Induced_Label_Effect_Cell_Count_{threshold}pct"])
            status_changed = int(c[f"EOL_Status_Changed_Cell_Count_{threshold}pct"])
            rul_changed = int(c[f"RUL_Point_Availability_Changed_Cell_Count_{threshold}pct"])
            exact_shift = int(c[f"Exact_EOL_Shift_Cell_Count_{threshold}pct"])
            both_right = int(c[f"Both_Right_Censored_Cell_Count_{threshold}pct"])
            classification = c["Cross_Threshold_Classification"]
            comparison = "VERSUS_NOMINAL"
        rows.append(
            {
                "Dataset": dataset,
                "Threshold_Fraction": float(row["Threshold_Fraction"]),
                "Threshold_Percent": threshold,
                "Qref_Definition": qref,
                "Comparison_Scope": row["Comparison_Scope"],
                "Cell_Count": int(row["Cell_Count"]),
                "Qref_Available_Count": int(row["Qref_Available_Count"]),
                "EXACT_Count": int(row["EXACT_OBSERVED_Count"]),
                "INTERVAL_Count": int(row["INTERVAL_CENSORED_Count"]),
                "RIGHT_Count": int(row["RIGHT_CENSORED_Count"]),
                "Point_RUL_Available_Count": int(row["Point_RUL_Available_Count"]),
                "Point_RUL_Available_Fraction": float(row["Point_RUL_Available_Fraction"]),
                "Relative_Comparison": comparison,
                "Qref_Induced_Label_Effect_Cell_Count": effect_count,
                "EOL_Status_Changed_Cell_Count": status_changed,
                "RUL_Point_Availability_Changed_Cell_Count": rul_changed,
                "Exact_EOL_Shift_Cell_Count": exact_shift,
                "Both_Right_Censored_Cell_Count": both_right,
                "Observation_Window_Limited": observation_window_label(dataset, threshold),
                "Cross_Threshold_Classification": classification,
                "Dataset_Stopping_Rule_Context": row["Dataset_Stopping_Rule_Context"],
            }
        )
    out = pd.DataFrame(rows)
    out["_d"] = pd.Categorical(out["Dataset"], ["XJTU", "HUST"], ordered=True)
    out["_q"] = pd.Categorical(out["Qref_Definition"], ["nominal", "initial_test", "first_comparable", "early_median5_comparable"], ordered=True)
    return out.sort_values(["_d", "Threshold_Percent", "_q"]).drop(columns=["_d", "_q"]).reset_index(drop=True)


def ml_primary_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = data["ml_primary_summary"]
    rank = data["ml_primary_rank"]
    merged = summary.merge(
        rank[["Dataset", "Qref_Definition", "Model", "Rank_Position"]],
        on=["Dataset", "Qref_Definition", "Model"],
        validate="one_to_one",
    )
    cols = [
        "Dataset",
        "Qref_Definition",
        "Model",
        "Cell_Count",
        "Observation_Count",
        "Median_MAE_Q_Ah",
        "IQR_MAE_Q_Ah",
        "Median_RMSE_Q_Ah",
        "Median_R2_Q",
        "Median_MAE_SOH_percentage_points",
        "Median_RMSE_SOH_fraction",
        "Median_R2_SOH",
        "Rank_Position",
    ]
    out = merged[cols].rename(columns={"Rank_Position": "Rank_Position_Under_Median_Cell_MAE_Q"})
    out["Primary_Ranking_Metric"] = "median held-out-cell MAE_Q_Ah"
    out["Ranking_Terminology"] = "neutral rank position under predefined metric"
    out["_d"] = pd.Categorical(out["Dataset"], ["XJTU", "HUST"], ordered=True)
    out["_q"] = pd.Categorical(out["Qref_Definition"], COMMON_QREFS, ordered=True)
    return out.sort_values(["_d", "_q", "Rank_Position_Under_Median_Cell_MAE_Q"]).drop(columns=["_d", "_q"]).reset_index(drop=True)


def ranking_order(group: pd.DataFrame) -> str:
    ordered = group.sort_values(["Median_MAE_Q_Ah", "Model"])["Model"].tolist()
    return " > ".join(ordered)


def ml_robustness_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces = []
    primary = data["ml_primary_summary"].copy()
    primary["Experiment"] = "PRIMARY_A9"
    primary["Feature_Set"] = "A9_PRIMARY"
    pieces.append(primary)
    pieces.extend([data["ml_a8"].copy(), data["ml_b10"].copy(), data["ml_post5"].copy()])
    all_rows = pd.concat(pieces, ignore_index=True, sort=False)

    robust_rank = data["ml_robust_rank"].loc[data["ml_robust_rank"]["Row_Type"].eq("QREF_RANKING")]
    robust_map = {
        (r.Experiment, r.Dataset, r.Qref_Definition): (r.Ranking_Order, r.Ranking_Changed_vs_Primary)
        for r in robust_rank.itertuples(index=False)
    }
    primary_orders = {
        (dataset, qref): ranking_order(g)
        for (dataset, qref), g in primary.groupby(["Dataset", "Qref_Definition"])
    }
    a8_ci = data["a8_ci"].set_index("Qref_Definition")

    rows: list[dict[str, object]] = []
    for (experiment, dataset, qref), group in all_rows.groupby(["Experiment", "Dataset", "Qref_Definition"], sort=False):
        if experiment == "PRIMARY_A9":
            order = primary_orders[(dataset, qref)]
            changed = "REFERENCE"
        else:
            order, changed = robust_map[(experiment, dataset, qref)]
        ranks = {
            model: i + 1
            for i, model in enumerate(group.sort_values(["Median_MAE_Q_Ah", "Model"])["Model"].tolist())
        }
        for _, row in group.iterrows():
            if experiment == "ROBUSTNESS_A8" and dataset == "XJTU":
                ci = a8_ci.loc[qref]
                interp = (
                    "DESCRIPTIVE_RANK_REVERSAL_NOT_BOOTSTRAP_ROBUST"
                    if qref != "nominal"
                    else "REFERENCE_QREF_RF_GB_CI_INCLUDES_ZERO"
                )
                delta = float(ci["Observed_Difference_Of_Medians"])
                ci_lo = float(ci["Bootstrap_95CI_Lower"])
                ci_hi = float(ci["Bootstrap_95CI_Upper"])
            else:
                interp = "NOT_APPLICABLE"
                delta = ci_lo = ci_hi = np.nan
            rows.append(
                {
                    "Experiment": experiment,
                    "Feature_Set": row["Feature_Set"],
                    "Dataset": dataset,
                    "Qref_Definition": qref,
                    "Model": row["Model"],
                    "Cell_Count": int(row["Cell_Count"]),
                    "Observation_Count": int(row["Observation_Count"]),
                    "Median_MAE_Q_Ah": float(row["Median_MAE_Q_Ah"]),
                    "Rank_Position_Under_Median_Cell_MAE_Q": ranks[row["Model"]],
                    "Ranking_Order": order,
                    "Ranking_Changed_vs_Primary": changed,
                    "A8_Corrected_RF_GB_Observed_Difference_Of_Medians": delta,
                    "A8_Corrected_RF_GB_95CI_Lower": ci_lo,
                    "A8_Corrected_RF_GB_95CI_Upper": ci_hi,
                    "A8_Corrected_Bootstrap_Interpretation": interp,
                    "Ranking_Metric": "median held-out-cell MAE_Q_Ah ascending",
                }
            )
    out = pd.DataFrame(rows)
    exp_order = ["PRIMARY_A9", "ROBUSTNESS_A8", "ROBUSTNESS_B10", "ROBUSTNESS_POSTCYCLE5"]
    out["_e"] = pd.Categorical(out["Experiment"], exp_order, ordered=True)
    out["_d"] = pd.Categorical(out["Dataset"], ["XJTU", "HUST"], ordered=True)
    out["_q"] = pd.Categorical(out["Qref_Definition"], COMMON_QREFS, ordered=True)
    return out.sort_values(["_e", "_d", "_q", "Rank_Position_Under_Median_Cell_MAE_Q"]).drop(columns=["_e", "_d", "_q"]).reset_index(drop=True)


def supplementary_plan() -> pd.DataFrame:
    rows = [
        ("S1", "CALCE QC workflow and manual-review logic", "RECOMMENDED", "Documents method-development and reproducibility support without elevating CALCE to an independent biological replication."),
        ("S2", "XJTU batch protocol mapping and censoring rules", "RECOMMENDED", "Supports interpretation of sparse diagnostics, cumulative aging operation counts, and interval censoring."),
        ("S3", "HUST paper-versus-raw protocol discrepancy audit", "RECOMMENDED", "Documents why QC used the raw-observed protocol and why HUST is a mechanism-level replication."),
        ("S4", "Full cell-level Qref summary", "RECOMMENDED", "Provides traceability for every cell-level reference definition and Qref deviation."),
        ("S5", "Full 75%, 80%, and 85% transition tables", "RECOMMENDED", "Preserves all exact, interval, right-censored, effect-bound, and RUL-availability transitions."),
        ("S6", "ML feature-leakage audit", "RECOMMENDED", "Documents the frozen A9 predictor set and excluded protocol-shortcut features."),
        ("S7", "Feature-extraction QC and missingness", "RECOMMENDED", "Explains the single excluded HUST A9 row and verifies common physical populations across Qrefs."),
        ("S8", "XJTU A8 corrected bootstrap details", "RECOMMENDED", "Reports the difference-of-medians statistic, confidence intervals, rank frequencies, and descriptive rank-flip stability."),
        ("S9", "XJTU initial-test 47-cell secondary analysis", "RECOMMENDED", "Keeps the XJTU-only initial-test comparison separate from the common cross-dataset definitions."),
        ("S10", "Additional per-cell and per-batch ML metrics", "OPTIONAL_IF_SPACE", "Supports heterogeneity review without duplicating the primary cell-first summary."),
    ]
    return pd.DataFrame(rows, columns=["Supplement_ID", "Proposed_Item", "Recommendation", "Rationale"])


def fmt(value: float, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def transition_global(data: dict[str, pd.DataFrame], dataset: str, threshold: int, qref: str) -> pd.Series:
    g = data["threshold_transition"].loc[
        (data["threshold_transition"]["Dataset"] == dataset)
        & (data["threshold_transition"]["Threshold_Percent"] == threshold)
        & (data["threshold_transition"]["Measured_Qref_Definition"] == qref)
    ]
    if g.empty:
        raise ValueError(f"Missing transition summary for {dataset} {threshold} {qref}")
    return g.iloc[0]


def results_text(
    qref: pd.DataFrame,
    eol80: pd.DataFrame,
    threshold: pd.DataFrame,
    ml: pd.DataFrame,
    robust: pd.DataFrame,
    data: dict[str, pd.DataFrame],
) -> str:
    def qr(dataset: str, definition: str) -> pd.Series:
        return qref.loc[(qref.Dataset == dataset) & (qref.Qref_Definition == definition)].iloc[0]

    def er(dataset: str, definition: str) -> pd.Series:
        return eol80.loc[(eol80.Dataset == dataset) & (eol80.Qref_Definition == definition)].iloc[0]

    x_first, x_med, x_init = qr("XJTU", "first_comparable"), qr("XJTU", "early_median5_comparable"), qr("XJTU", "initial_test")
    h_first, h_med = qr("HUST", "first_comparable"), qr("HUST", "early_median5_comparable")
    x_nom_e, x_first_e, x_med_e = er("XJTU", "nominal"), er("XJTU", "first_comparable"), er("XJTU", "early_median5_comparable")
    h_nom_e, h_first_e, h_med_e = er("HUST", "nominal"), er("HUST", "first_comparable"), er("HUST", "early_median5_comparable")

    x_initial_row = data["cross_qref"].loc[
        (data["cross_qref"].Dataset == "XJTU") & (data["cross_qref"].Qref_Definition == "initial_test")
    ].iloc[0]

    x85_first = transition_global(data, "XJTU", 85, "first_comparable")
    x85_med = transition_global(data, "XJTU", 85, "early_median5_comparable")
    h85_first = transition_global(data, "HUST", 85, "first_comparable")
    h85_med = transition_global(data, "HUST", 85, "early_median5_comparable")
    h75_first = transition_global(data, "HUST", 75, "first_comparable")
    h75_med = transition_global(data, "HUST", 75, "early_median5_comparable")

    def status_triplet(dataset: str, threshold_value: int, qdef: str) -> tuple[int, int, int]:
        r = threshold.loc[
            (threshold.Dataset == dataset)
            & (threshold.Threshold_Percent == threshold_value)
            & (threshold.Qref_Definition == qdef)
        ].iloc[0]
        return int(r.EXACT_Count), int(r.INTERVAL_Count), int(r.RIGHT_Count)

    x75 = {q: status_triplet("XJTU", 75, q) for q in COMMON_QREFS}
    x85 = {q: status_triplet("XJTU", 85, q) for q in COMMON_QREFS}
    h75 = {q: status_triplet("HUST", 75, q) for q in COMMON_QREFS}
    h85 = {q: status_triplet("HUST", 85, q) for q in COMMON_QREFS}

    def metric_range(dataset: str, model: str, metric: str) -> tuple[float, float]:
        s = ml.loc[(ml.Dataset == dataset) & (ml.Model == model), metric]
        return float(s.min()), float(s.max())

    x_rf_q = metric_range("XJTU", "Random Forest Regressor", "Median_MAE_Q_Ah")
    x_gb_q = metric_range("XJTU", "Gradient Boosting Regressor", "Median_MAE_Q_Ah")
    x_rg_q = metric_range("XJTU", "Ridge Regression", "Median_MAE_Q_Ah")
    h_rf_q = metric_range("HUST", "Random Forest Regressor", "Median_MAE_Q_Ah")
    h_gb_q = metric_range("HUST", "Gradient Boosting Regressor", "Median_MAE_Q_Ah")
    h_rg_q = metric_range("HUST", "Ridge Regression", "Median_MAE_Q_Ah")
    x_rf_s = metric_range("XJTU", "Random Forest Regressor", "Median_MAE_SOH_percentage_points")
    x_gb_s = metric_range("XJTU", "Gradient Boosting Regressor", "Median_MAE_SOH_percentage_points")
    x_rg_s = metric_range("XJTU", "Ridge Regression", "Median_MAE_SOH_percentage_points")
    h_rf_s = metric_range("HUST", "Random Forest Regressor", "Median_MAE_SOH_percentage_points")
    h_gb_s = metric_range("HUST", "Gradient Boosting Regressor", "Median_MAE_SOH_percentage_points")
    h_rg_s = metric_range("HUST", "Ridge Regression", "Median_MAE_SOH_percentage_points")

    ci = data["a8_ci"].set_index("Qref_Definition")
    flip = data["a8_rankflip"].set_index("Measured_Qref_Definition")

    return f"""# 3. Results

## 3.1 Reference-capacity distributions

Across 55 XJTU cells, the first-comparable Qref was below the 2.000 Ah nominal reference in every cell. The median difference was {fmt(x_first.Delta_Qref_vs_Nominal_Ah_Median)} Ah ({fmt(x_first.Delta_Qref_vs_Nominal_Pct_Median)}%), with a cell-level range from {fmt(x_first.Delta_Qref_vs_Nominal_Ah_Min)} to {fmt(x_first.Delta_Qref_vs_Nominal_Ah_Max)} Ah ({fmt(x_first.Delta_Qref_vs_Nominal_Pct_Min)}% to {fmt(x_first.Delta_Qref_vs_Nominal_Pct_Max)}%). The early-median-5 comparable Qref was also below nominal in all 55 cells, with a median difference of {fmt(x_med.Delta_Qref_vs_Nominal_Ah_Median)} Ah ({fmt(x_med.Delta_Qref_vs_Nominal_Pct_Median)}%) and a range from {fmt(x_med.Delta_Qref_vs_Nominal_Ah_Min)} to {fmt(x_med.Delta_Qref_vs_Nominal_Ah_Max)} Ah. The XJTU-only initial-test reference was available for 47 cells and showed a mixed direction: {int(x_initial_row.QREF_BELOW_NOMINAL_Count)} values were below and {int(x_initial_row.QREF_ABOVE_NOMINAL_Count)} were above nominal; its median difference was {fmt(x_init.Delta_Qref_vs_Nominal_Ah_Median)} Ah ({fmt(x_init.Delta_Qref_vs_Nominal_Pct_Median)}%).

Across 77 HUST cells, both measured comparable definitions were above the 1.100 Ah nominal reference in every cell. The first-comparable difference had a median of +{fmt(h_first.Delta_Qref_vs_Nominal_Ah_Median)} Ah (+{fmt(h_first.Delta_Qref_vs_Nominal_Pct_Median)}%) and ranged from +{fmt(h_first.Delta_Qref_vs_Nominal_Ah_Min)} to +{fmt(h_first.Delta_Qref_vs_Nominal_Ah_Max)} Ah. The early-median-5 difference had a median of +{fmt(h_med.Delta_Qref_vs_Nominal_Ah_Median)} Ah (+{fmt(h_med.Delta_Qref_vs_Nominal_Pct_Median)}%) and ranged from +{fmt(h_med.Delta_Qref_vs_Nominal_Ah_Min)} to +{fmt(h_med.Delta_Qref_vs_Nominal_Ah_Max)} Ah. HUST had no separate standardized initial-test Qref, so these results represent mechanism-level replication with common comparable definitions rather than an exact nominal-versus-initial-test replication.

## 3.2 Qref-induced EOL and censoring changes at 80%

Under the predefined 80% labeling rule, XJTU nominal Qref produced {int(x_nom_e.EXACT_Count)} exact, {int(x_nom_e.INTERVAL_Count)} interval-censored, and {int(x_nom_e.RIGHT_Count)} right-censored EOL states. First-comparable Qref produced {int(x_first_e.EXACT_Count)}, {int(x_first_e.INTERVAL_Count)}, and {int(x_first_e.RIGHT_Count)}, respectively, while early-median-5 produced {int(x_med_e.EXACT_Count)}, {int(x_med_e.INTERVAL_Count)}, and {int(x_med_e.RIGHT_Count)}. Relative to nominal, EOL status changed in {int(x_first_e.Status_Changed_Count)} of 55 cells for first-comparable and {int(x_med_e.Status_Changed_Count)} of 55 for early-median-5. The median absolute 80% threshold shifts were {fmt(x_first['80pct_Threshold_Shift_Ah_Median'], 4)} and {fmt(x_med['80pct_Threshold_Shift_Ah_Median'], 4)} Ah, respectively. No exact EOL-cycle delta was created for these censored pairs.

In HUST, nominal Qref produced {int(h_nom_e.RIGHT_Count)} right-censored and no exact EOL states at 80%, whereas both measured comparable definitions produced {int(h_first_e.EXACT_Count)} exact states and no right-censored states. Thus, EOL status changed in all 77 cells for each measured definition. The measured references shifted the median 80% analytical threshold by +{fmt(h_first['80pct_Threshold_Shift_Ah_Median'], 4)} Ah for first-comparable and +{fmt(h_med['80pct_Threshold_Shift_Ah_Median'], 4)} Ah for early-median-5. Because the nominal endpoints were right-censored, these were not exact paired EOL differences; the minimum observed earlier-by-cycle bound had a median of {fmt(h_first_e.Minimum_Observed_Earlier_By_Cycles_Median, 0)} cycles for both comparisons. These results must be interpreted in the HUST released observation-window and stopping-rule context near the nominal 80% region.

## 3.3 RUL point-label availability

At 80%, point-RUL labels were available for {int(x_nom_e.Point_RUL_Available_Count)} of 55 XJTU cells ({100*x_nom_e.Point_RUL_Available_Fraction:.1f}%) under nominal Qref and for 0 of 55 cells under either measured comparable definition. The corresponding availability change involved {int(x_first_e.RUL_Availability_Changed_Count)} cells for each measured comparison. In HUST, availability changed from 0 of 77 cells under nominal Qref to 77 of 77 cells under both measured definitions. Censored EOL states retained missing point-RUL labels; no last-observed cycle was substituted as EOL.

## 3.4 Threshold robustness at 75%, 80%, and 85%

The Qref-related labeling effect was not unique to the 80% threshold. At 75%, XJTU nominal Qref yielded exact/interval/right counts of {x75['nominal'][0]}/{x75['nominal'][1]}/{x75['nominal'][2]}, compared with {x75['first_comparable'][0]}/{x75['first_comparable'][1]}/{x75['first_comparable'][2]} for first-comparable and {x75['early_median5_comparable'][0]}/{x75['early_median5_comparable'][1]}/{x75['early_median5_comparable'][2]} for early-median-5; this level was mostly observation-window limited. At 85%, the corresponding XJTU counts were {x85['nominal'][0]}/{x85['nominal'][1]}/{x85['nominal'][2]}, {x85['first_comparable'][0]}/{x85['first_comparable'][1]}/{x85['first_comparable'][2]}, and {x85['early_median5_comparable'][0]}/{x85['early_median5_comparable'][1]}/{x85['early_median5_comparable'][2]}. Among exact pairs at 85%, the median EOL shift was {fmt(x85_first.Exact_Delta_EOL_Cycles_Median, 1)} cycles for first-comparable (n={int(x85_first.Exact_Delta_Valid_Cell_Count_All_Transitions)}) and {fmt(x85_med.Exact_Delta_EOL_Cycles_Median, 1)} cycles for early-median-5 (n={int(x85_med.Exact_Delta_Valid_Cell_Count_All_Transitions)}).

In HUST at 75%, nominal Qref remained right-censored in all 77 cells, while first-comparable produced {h75['first_comparable'][0]} exact and {h75['first_comparable'][2]} right-censored states and early-median-5 produced {h75['early_median5_comparable'][0]} exact and {h75['early_median5_comparable'][2]} right-censored states. The minimum observed earlier-by-cycle bound had medians of {fmt(h75_first.Minimum_Observed_Earlier_By_Cycles_Median, 0)} and {fmt(h75_med.Minimum_Observed_Earlier_By_Cycles_Median, 0)} cycles, respectively. At 85%, all three HUST Qrefs produced 77 exact EOL states; measured-versus-nominal exact EOL shifts remained observable, with medians of {fmt(h85_first.Exact_Delta_EOL_Cycles_Median, 0)} cycles for first-comparable and {fmt(h85_med.Exact_Delta_EOL_Cycles_Median, 0)} cycles for early-median-5. The HUST 75% and 80% counts remain dependent on the finite released observation window, whereas the 85% threshold is encountered earlier in the recorded trajectories.

## 3.5 SOH benchmark metric sensitivity

The primary A9 benchmark used 55 XJTU cells with 10,427 observations and 77 HUST cells with 146,121 feature-complete observations. In physical-capacity space, XJTU median held-out-cell MAE_Q ranges across Qrefs were {fmt(x_rf_q[0], 4)}-{fmt(x_rf_q[1], 4)} Ah for Random Forest, {fmt(x_gb_q[0], 4)}-{fmt(x_gb_q[1], 4)} Ah for Gradient Boosting, and {fmt(x_rg_q[0], 4)}-{fmt(x_rg_q[1], 4)} Ah for Ridge. Their SOH-space MAE ranges were {fmt(x_rf_s[0], 3)}-{fmt(x_rf_s[1], 3)}, {fmt(x_gb_s[0], 3)}-{fmt(x_gb_s[1], 3)}, and {fmt(x_rg_s[0], 3)}-{fmt(x_rg_s[1], 3)} percentage points, respectively. In HUST, physical-space ranges were {fmt(h_rf_q[0], 4)}-{fmt(h_rf_q[1], 4)}, {fmt(h_gb_q[0], 4)}-{fmt(h_gb_q[1], 4)}, and {fmt(h_rg_q[0], 4)}-{fmt(h_rg_q[1], 4)} Ah, whereas SOH-space ranges were {fmt(h_rf_s[0], 3)}-{fmt(h_rf_s[1], 3)}, {fmt(h_gb_s[0], 3)}-{fmt(h_gb_s[1], 3)}, and {fmt(h_rg_s[0], 3)}-{fmt(h_rg_s[1], 3)} percentage points. Thus, Qref definition was associated with changes in reported error magnitudes, and physical-capacity-space reporting reduced ambiguity attributable solely to SOH label scaling without eliminating model or dataset effects.

## 3.6 Model-ranking robustness

Using median held-out-cell MAE_Q_Ah as the predefined ranking metric, the primary A9 order was Random Forest, Gradient Boosting, then Ridge for every XJTU and HUST common Qref. The same order was retained in the B10 A9-plus-cycle-index and A9 post-cycle-5 analyses, and in HUST after removing CV duration. In the XJTU A8 analysis, the observed order was Random Forest, Gradient Boosting, then Ridge under nominal Qref but Gradient Boosting, Random Forest, then Ridge under both measured definitions. The corrected RF-versus-GB difference of medians was {fmt(ci.loc['nominal','Observed_Difference_Of_Medians'], 6)} Ah under nominal Qref, {fmt(ci.loc['first_comparable','Observed_Difference_Of_Medians'], 6)} Ah under first-comparable, and {fmt(ci.loc['early_median5_comparable','Observed_Difference_Of_Medians'], 6)} Ah under early-median-5. All corrected 95% bootstrap intervals included zero: [{fmt(ci.loc['nominal','Bootstrap_95CI_Lower'], 6)}, {fmt(ci.loc['nominal','Bootstrap_95CI_Upper'], 6)}], [{fmt(ci.loc['first_comparable','Bootstrap_95CI_Lower'], 6)}, {fmt(ci.loc['first_comparable','Bootstrap_95CI_Upper'], 6)}], and [{fmt(ci.loc['early_median5_comparable','Bootstrap_95CI_Lower'], 6)}, {fmt(ci.loc['early_median5_comparable','Bootstrap_95CI_Upper'], 6)}] Ah. The same-draw cell bootstrap reproduced the nominal-RF-lower and measured-GB-lower pattern in {100*flip.loc['first_comparable','Observed_Sample_RankFlip_Stability']:.2f}% of replicates for first-comparable and {100*flip.loc['early_median5_comparable','Observed_Sample_RankFlip_Stability']:.2f}% for early-median-5. This was a descriptive rank reversal, not a bootstrap-robust model-ranking effect.
"""


def discussion_map(qref: pd.DataFrame, eol80: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        ("Qref is a benchmark-definition choice", "XJTU; HUST", "Frozen nominal and measured definitions were applied to the same retained physical capacity records.", "SUPPORTED_DIRECTLY", "The analysis does not establish a universally correct reference.", "Qref should be treated as an explicit benchmark-definition choice."),
        ("Qref alters SOH scale", "XJTU; HUST", "SOH was defined as Q/Qref, and cell-level Qref deviations had opposite directions across datasets.", "SUPPORTED_DIRECTLY", "The transformation changes labels, not physical degradation.", "Changing Qref rescaled SOH labels for an unchanged capacity observation."),
        ("Qref shifts the analytical EOL threshold", "XJTU; HUST", "The absolute threshold was alpha multiplied by Qref at 75%, 80%, and 85%.", "SUPPORTED_DIRECTLY", "The direction follows algebraically from Qref ordering.", "Under the predefined labeling rule, Qref shifted the absolute EOL threshold."),
        ("Qref can change censoring status", "XJTU; HUST", "At 80%, status changed in 51/55 and 49/55 XJTU cells and in 77/77 HUST cells for the two measured definitions.", "SUPPORTED_DIRECTLY", "Counts depend on observation density and released windows.", "Qref definition changed the observed EOL censoring state within the frozen observation windows."),
        ("Qref can determine whether point-RUL exists", "XJTU; HUST", "At 80%, point-RUL availability changed from 20/55 to 0/55 in XJTU and from 0/77 to 77/77 in HUST.", "SUPPORTED_DIRECTLY", "This concerns point-label availability, not true remaining life.", "Qref definition changed whether a point-RUL label was available under the censoring rules."),
        ("SOH-space errors can exaggerate label-scale differences", "XJTU; HUST", "SOH-space errors changed with Qref scaling while physical-space MAE_Q remained directly comparable in Ah.", "SUPPORTED_WITH_LIMITATION", "The pattern is descriptive and not identical for every model-dataset pair.", "SOH-space errors were generally more sensitive to label scaling than physical-capacity-space errors."),
        ("Physical-capacity-space evaluation reduces pure scaling ambiguity", "XJTU; HUST", "The primary ranking metric used median cell MAE_Q_Ah rather than an SOH-normalized error.", "SUPPORTED_DIRECTLY", "Physical-space metrics do not remove protocol, feature, or model effects.", "Reporting MAE_Q alongside SOH-space errors reduced ambiguity due solely to Qref scaling."),
        ("Model ranking was stable in the primary analysis", "XJTU; HUST", "Random Forest ranked first, Gradient Boosting second, and Ridge third for all six dataset-by-common-Qref combinations.", "SUPPORTED_DIRECTLY", "This ranking is specific to the frozen models, folds, features, and metric.", "Primary model rank positions were stable under the predefined median cell MAE_Q metric."),
        ("One A8 descriptive reversal was not bootstrap robust", "XJTU", "RF and GB exchanged observed rank positions for measured Qrefs, but all corrected difference-of-medians 95% CIs included zero.", "SUPPORTED_DIRECTLY", "Observed-sample bootstrap stability is not a universal superiority probability.", "The XJTU A8 RF/GB reversal was descriptive and not bootstrap robust."),
        ("Threshold effects were not unique to 80%", "XJTU; HUST", "Frozen 75%, 80%, and 85% analyses all contained Qref-related label effects, with threshold-specific observability.", "SUPPORTED_DIRECTLY", "The magnitude and estimability changed with threshold and observation window.", "Qref-related labeling effects occurred across all three predefined thresholds."),
        ("Observation-window dependence limits interpretation", "XJTU; HUST", "XJTU sparse diagnostics produced interval/right censoring; HUST 75% and nominal 80% were constrained by the finite released window and stopping-rule context.", "SUPPORTED_DIRECTLY", "Censoring states do not identify unobserved physical event times.", "EOL and point-RUL results should be interpreted conditional on observation density and stopping rules."),
    ]
    return pd.DataFrame(rows, columns=["Discussion_Point", "Supporting_Dataset", "Supporting_Result", "Strength", "Important_Limitation", "Recommended_Wording"])


def claim_audit() -> pd.DataFrame:
    rows = [
        ("A", "Qref changes SOH labels.", "SUPPORTED_DIRECTLY", "SOH equals Q divided by Qref for the frozen capacity observations.", "This is a label-scale statement, not physical degradation."),
        ("B", "Qref changes analytical EOL thresholds.", "SUPPORTED_DIRECTLY", "Every threshold was defined as alpha multiplied by Qref.", "The result is definition-dependent."),
        ("C", "Qref changes observed EOL timing.", "SUPPORTED_WITH_LIMITATION", "Exact timing shifts were observed where both events were exact; censored comparisons used bounds.", "Do not replace censored pairs with exact deltas."),
        ("D", "Qref can change EOL censoring state.", "SUPPORTED_DIRECTLY", "Frozen 80% transitions changed status in both datasets.", "Depends on observation window and sampling density."),
        ("E", "Qref can change point-RUL availability.", "SUPPORTED_DIRECTLY", "Point RUL was available only when exact EOL was observed and changed across Qrefs.", "This concerns analytical label availability."),
        ("F", "Qref always changes model ranking.", "NOT_SUPPORTED", "Primary rankings were unchanged across all common Qrefs.", "One A8 descriptive reversal was not bootstrap robust."),
        ("G", "Qref changes reported model error magnitudes.", "SUPPORTED_DIRECTLY", "Frozen primary metrics differed across Qref definitions in both label and capacity spaces.", "Magnitude and direction varied by dataset and model."),
        ("H", "SOH-space error is generally more sensitive to Qref scaling than physical-space error.", "SUPPORTED_WITH_LIMITATION", "Frozen medians showed larger relative scaling effects in many comparisons.", "Not universal across every model-dataset-Qref combination."),
        ("I", "The effect is unique to an 80% threshold.", "NOT_SUPPORTED", "Effects were retained at 75%, 80%, and 85%, although observability differed.", "Threshold-specific censoring must be retained."),
        ("J", "Measured Qref is universally superior to nominal.", "NOT_SUPPORTED", "The study compared consequences of definitions and did not adjudicate universal correctness.", "Avoid accuracy or superiority claims."),
    ]
    return pd.DataFrame(rows, columns=["Claim_ID", "Candidate_Claim", "Classification", "Evidence_Basis", "Important_Limitation"])


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, title: str) -> None:
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", metadata={"Title": title, "Author": "Frozen-results synthesis"})
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def figure1(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.3))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6.3)
    ax.axis("off")
    ax.text(0.2, 6.0, "Study design: reference definition to benchmark outcomes", fontsize=16, weight="bold", ha="left")
    steps = [
        ("Reference capacity\nQref", "#DCEAF7"),
        ("SOH label\nSOH = Q / Qref", "#E8F1FA"),
        ("Analytical threshold\nalpha x Qref", "#FFF1D6"),
        ("EOL timing and\ncensoring state", "#FDE2D8"),
        ("Point-RUL label\navailability", "#E9E2F4"),
        ("SOH benchmark\nmetrics and ranks", "#DDEFE8"),
    ]
    xs = np.linspace(0.35, 11.05, len(steps))
    for i, ((label, color), x) in enumerate(zip(steps, xs)):
        box = FancyBboxPatch((x, 3.25), 1.75, 1.25, boxstyle="round,pad=0.04,rounding_size=0.08", linewidth=1.2, edgecolor="#334155", facecolor=color)
        ax.add_patch(box)
        ax.text(x + 0.875, 3.875, label, ha="center", va="center", fontsize=9.5, weight="bold" if i in (0, 2) else "normal")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.78, 3.875), (xs[i + 1] - 0.05, 3.875), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#475569"))
    ax.text(xs[2] + 0.875, 2.88, "alpha = 0.75, 0.80, 0.85", ha="center", va="top", fontsize=9, color="#7C4A03")
    role_y = 1.25
    roles = [
        ("XJTU", "Primary multi-protocol benchmark", "#2563EB"),
        ("HUST", "External mechanism-level replication", "#E07A2D"),
        ("CALCE", "QC and reproducibility support", "#64748B"),
    ]
    for i, (name, desc, color) in enumerate(roles):
        x = 0.55 + i * 4.2
        ax.add_patch(FancyBboxPatch((x, role_y), 3.55, 0.9, boxstyle="round,pad=0.03,rounding_size=0.06", linewidth=1.0, edgecolor=color, facecolor="white"))
        ax.text(x + 0.2, role_y + 0.58, name, fontsize=10, weight="bold", color=color, ha="left")
        ax.text(x + 0.2, role_y + 0.25, desc, fontsize=8.5, color="#334155", ha="left")
    ax.text(0.55, 0.55, "Frozen outputs only: no raw-data reprocessing, model retraining, or threshold redefinition", fontsize=9, color="#475569")
    save_figure(fig, out_dir, "figure1_study_design", "Study design")


def deterministic_offsets(n: int, width: float = 0.18) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    pattern = np.linspace(-width, width, 11)
    return np.resize(pattern, n)


def figure2(out_dir: Path, data: dict[str, pd.DataFrame]) -> None:
    d = data["cross_cell"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4), sharey=True)
    panels = [
        ("XJTU", ["first_comparable", "early_median5_comparable", "initial_test"], 55),
        ("HUST", ["first_comparable", "early_median5_comparable"], 77),
    ]
    colors = {"first_comparable": "#2563EB", "early_median5_comparable": "#E07A2D", "initial_test": "#8B5CF6"}
    for ax, (dataset, qrefs, ncell) in zip(axes, panels):
        for x, qref in enumerate(qrefs):
            g = d.loc[(d.Dataset == dataset) & (d.Qref_Definition == qref) & d.Delta_Qref_vs_Nominal_Pct.notna()].copy()
            vals = g["Delta_Qref_vs_Nominal_Pct"].to_numpy(float)
            offsets = deterministic_offsets(len(vals))
            ax.scatter(x + offsets, vals, s=18, alpha=0.62, color=colors[qref], edgecolor="white", linewidth=0.25)
            ax.scatter([x], [np.median(vals)], marker="D", s=58, color="black", edgecolor="white", linewidth=0.7, zorder=5)
            ax.text(x, np.median(vals) + (0.8 if np.median(vals) <= 0 else 0.55), f"median {np.median(vals):.2f}%", ha="center", va="bottom", fontsize=8)
        ax.axhline(0, color="#111827", linewidth=0.9)
        ax.set_xticks(range(len(qrefs)))
        labels = [QREF_LABELS[q] + ("\nsecondary" if q == "initial_test" else "") for q in qrefs]
        ax.set_xticklabels(labels)
        ax.set_title(f"{dataset} (N={ncell} cells)", loc="left", weight="bold")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Delta Qref vs nominal (%)")
    axes[0].set_ylim(-12.5, 13.2)
    fig.suptitle("Cell-level reference-capacity deviation", fontsize=14, weight="bold", x=0.06, ha="left")
    fig.text(0.06, 0.02, "Points are individual cells; black diamonds are medians. Capacities are normalized within dataset.", fontsize=8.5, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save_figure(fig, out_dir, "figure2_qref_deviation", "Reference-capacity deviation")


def transition_matrix(data: dict[str, pd.DataFrame], dataset: str, qref: str) -> np.ndarray:
    t = data["threshold_transition"]
    g = t.loc[(t.Dataset == dataset) & t.Threshold_Percent.eq(80) & t.Measured_Qref_Definition.eq(qref)]
    matrix = np.zeros((3, 3), dtype=int)
    mapping = {
        "EXACT_TO_EXACT": (0, 0), "EXACT_TO_INTERVAL": (0, 1), "EXACT_TO_RIGHT": (0, 2),
        "INTERVAL_TO_EXACT": (1, 0), "INTERVAL_TO_INTERVAL": (1, 1), "INTERVAL_TO_RIGHT": (1, 2),
        "RIGHT_TO_EXACT": (2, 0), "RIGHT_TO_INTERVAL": (2, 1), "BOTH_RIGHT": (2, 2),
    }
    for r in g.itertuples(index=False):
        matrix[mapping[r.Transition_Category]] = int(r.Cell_Count)
    return matrix


def figure3(out_dir: Path, data: dict[str, pd.DataFrame], eol80: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.3))
    specs = [
        ("XJTU", "first_comparable", "A"),
        ("XJTU", "early_median5_comparable", "B"),
        ("HUST", "first_comparable", "C"),
        ("HUST", "early_median5_comparable", "D"),
    ]
    cmap = LinearSegmentedColormap.from_list("cleanblue", ["#FFFFFF", "#D9EAF7", "#2C7FB8"])
    for ax, (dataset, qref, panel) in zip(axes.flat, specs):
        matrix = transition_matrix(data, dataset, qref)
        n = matrix.sum()
        ax.imshow(matrix / n * 100, cmap=cmap, vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                count = matrix[i, j]
                pct = 100 * count / n if n else 0
                color = "white" if pct > 55 else "#111827"
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", fontsize=9, color=color, weight="bold" if count else "normal")
        nom = eol80.loc[(eol80.Dataset == dataset) & eol80.Qref_Definition.eq("nominal")].iloc[0]
        meas = eol80.loc[(eol80.Dataset == dataset) & eol80.Qref_Definition.eq(qref)].iloc[0]
        delta = int(meas.Point_RUL_Available_Count - nom.Point_RUL_Available_Count)
        ax.set_title(f"{panel}. {dataset}: nominal to {QREF_LABELS[qref]}\nPoint-RUL: {int(nom.Point_RUL_Available_Count)} to {int(meas.Point_RUL_Available_Count)} ({delta:+d})", loc="left", fontsize=10, weight="bold")
        ax.set_xticks(range(3), [STATUS_SHORT[s] for s in STATUS_ORDER])
        ax.set_yticks(range(3), [STATUS_SHORT[s] for s in STATUS_ORDER])
        ax.set_xlabel("Measured-Qref EOL status")
        ax.set_ylabel("Nominal-Qref EOL status")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("80% EOL censoring-state transitions", fontsize=14, weight="bold", x=0.07, ha="left")
    fig.text(0.07, 0.02, "Counts retain exact, interval-censored, and right-censored states; no transition is inferred for an unchanged cell.", fontsize=8.5, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, out_dir, "figure3_eol_transitions_80pct", "80% EOL censoring transitions")


def figure4(out_dir: Path, threshold: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharey=True)
    for ax, dataset in zip(axes, ["XJTU", "HUST"]):
        g = threshold.loc[(threshold.Dataset == dataset) & threshold.Qref_Definition.isin(COMMON_QREFS)].copy()
        x = np.arange(len(g))
        bottom = np.zeros(len(g))
        for status, col in [("EXACT_Count", "EXACT_OBSERVED"), ("INTERVAL_Count", "INTERVAL_CENSORED"), ("RIGHT_Count", "RIGHT_CENSORED")]:
            vals = g[status].to_numpy(float) / g["Cell_Count"].to_numpy(float) * 100
            ax.bar(x, vals, bottom=bottom, color=STATUS_COLORS[col], width=0.78, label=STATUS_SHORT[col], edgecolor="white", linewidth=0.5)
            bottom += vals
        labels = [f"{int(r.Threshold_Percent)}%\n{QREF_LABELS[r.Qref_Definition].replace(' comparable','')}" for r in g.itertuples(index=False)]
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Cells (%)")
        context = (
            "75% mostly window-limited; sparse diagnostics in some batches"
            if dataset == "XJTU"
            else "75% finite-window limited; 80% near stopping rule; 85% all exact"
        )
        ax.set_title(f"{dataset} (N={int(g.Cell_Count.max())})\n{context}", loc="left", weight="bold", fontsize=10.5, pad=8)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.96, 0.965), ncol=3, frameon=False)
    fig.suptitle("EOL-status distribution across predefined thresholds", fontsize=14, weight="bold", x=0.06, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    save_figure(fig, out_dir, "figure4_threshold_robustness", "Threshold robustness")


def figure5(out_dir: Path, threshold: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), sharey=True)
    qcolors = {"nominal": "#475569", "first_comparable": "#2563EB", "early_median5_comparable": "#E07A2D"}
    for ax, dataset in zip(axes, ["XJTU", "HUST"]):
        g = threshold.loc[(threshold.Dataset == dataset) & threshold.Qref_Definition.isin(COMMON_QREFS)]
        for qref in COMMON_QREFS:
            q = g.loc[g.Qref_Definition.eq(qref)].sort_values("Threshold_Percent")
            y = q.Point_RUL_Available_Fraction.to_numpy(float) * 100
            ax.plot(q.Threshold_Percent, y, marker="o", linewidth=2, markersize=6, color=qcolors[qref], label=QREF_LABELS[qref])
            for xv, yv in zip(q.Threshold_Percent, y):
                if qref == "early_median5_comparable" and yv >= 70:
                    offset, va = -4.0, "top"
                else:
                    offset, va = 3.0, "bottom"
                ax.text(xv, yv + offset, f"{yv:.0f}%", ha="center", va=va, fontsize=7.5, color=qcolors[qref])
        ax.set_xticks([75, 80, 85], ["75%", "80%", "85%"])
        ax.set_ylim(-2, 108)
        ax.set_xlabel("SOH threshold")
        ax.set_title(f"{dataset} (N={int(g.Cell_Count.max())})", loc="left", weight="bold")
        ax.grid(color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Cells with point-RUL label (%)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Point-RUL label availability depends on Qref, threshold, and observation window", fontsize=13.5, weight="bold", x=0.06, ha="left")
    fig.text(0.06, 0.015, "Only exact observed EOL supports a point-RUL label; censored states remain without artificial point labels.", fontsize=8.5, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save_figure(fig, out_dir, "figure5_rul_availability", "Point-RUL availability")


def figure6(out_dir: Path, ml: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    specs = [
        ("XJTU", "Median_MAE_Q_Ah", "Physical-capacity space: median cell MAE_Q (Ah)", "A"),
        ("XJTU", "Median_MAE_SOH_percentage_points", "SOH-label space: median cell MAE (percentage points)", "B"),
        ("HUST", "Median_MAE_Q_Ah", "Physical-capacity space: median cell MAE_Q (Ah)", "C"),
        ("HUST", "Median_MAE_SOH_percentage_points", "SOH-label space: median cell MAE (percentage points)", "D"),
    ]
    for ax, (dataset, metric, subtitle, panel) in zip(axes.flat, specs):
        g = ml.loc[ml.Dataset.eq(dataset)]
        x = np.arange(3)
        for model in MODEL_ORDER:
            m = g.loc[g.Model.eq(model)].set_index("Qref_Definition").loc[COMMON_QREFS]
            ax.plot(x, m[metric].to_numpy(float), marker="o", linewidth=2, markersize=5.5, color=MODEL_COLORS[model], label=MODEL_SHORT[model])
        ax.set_xticks(x, ["Nominal", "First", "Median-5"])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{panel}. {dataset} - {subtitle}", loc="left", fontsize=10, weight="bold")
        if metric == "Median_MAE_Q_Ah":
            ax.set_ylabel("MAE_Q (Ah)")
        else:
            ax.set_ylabel("MAE_SOH (percentage points)")
        n = int(g.Cell_Count.max())
        obs = int(g.Observation_Count.max())
        ax.text(
            0.99,
            0.96,
            f"N={n} cells; {obs:,} observations",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.8,
            color="#475569",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.5},
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.96, 0.97), ncol=3, frameon=False)
    fig.suptitle("Primary A9 SOH benchmark: physical-space and label-space error", fontsize=14, weight="bold", x=0.06, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    save_figure(fig, out_dir, "figure6_soh_ml_benchmark", "SOH model benchmark")


def figure7(out_dir: Path, robust: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.2), sharex=True)
    experiments = ["PRIMARY_A9", "ROBUSTNESS_A8", "ROBUSTNESS_B10", "ROBUSTNESS_POSTCYCLE5"]
    exp_short = {"PRIMARY_A9": "Primary A9", "ROBUSTNESS_A8": "A8 - CV duration", "ROBUSTNESS_B10": "B10 + cycle", "ROBUSTNESS_POSTCYCLE5": "A9 post-cycle-5"}
    qshort = {"nominal": "Nom.", "first_comparable": "First", "early_median5_comparable": "Median-5"}
    positions = {(e, q): i for i, (e, q) in enumerate((e, q) for e in experiments for q in COMMON_QREFS)}
    for ax, dataset in zip(axes, ["XJTU", "HUST"]):
        g = robust.loc[robust.Dataset.eq(dataset)]
        for model in MODEL_ORDER:
            xs, ys = [], []
            for e in experiments:
                for q in COMMON_QREFS:
                    r = g.loc[(g.Experiment == e) & (g.Qref_Definition == q) & (g.Model == model)].iloc[0]
                    xs.append(positions[(e, q)])
                    ys.append(int(r.Rank_Position_Under_Median_Cell_MAE_Q))
            ax.scatter(xs, ys, s=48, color=MODEL_COLORS[model], label=MODEL_SHORT[model], zorder=3)
            for e in experiments:
                ex = [positions[(e, q)] for q in COMMON_QREFS]
                ey = [ys[xs.index(v)] for v in ex]
                ax.plot(ex, ey, color=MODEL_COLORS[model], linewidth=1.2, alpha=0.75)
        for boundary in [2.5, 5.5, 8.5]:
            ax.axvline(boundary, color="#D1D5DB", linewidth=0.8)
        if dataset == "XJTU":
            ax.axvspan(3.45, 5.5, color="#FDECEC", alpha=0.8, zorder=0)
            ax.text(4.48, 3.26, "Descriptive RF/GB reversal\ncorrected 95% CIs include zero", ha="center", va="bottom", fontsize=8.2, color="#9B2C2C")
        ax.set_yticks([1, 2, 3], ["Rank 1", "Rank 2", "Rank 3"])
        ax.set_ylim(3.45, 0.55)
        ax.set_title(f"{dataset} (median cell MAE_Q rank)", loc="left", weight="bold")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    labels = [f"{exp_short[e]}\n{qshort[q]}" for e in experiments for q in COMMON_QREFS]
    axes[1].set_xticks(range(12), labels, rotation=35, ha="right")
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles[:3], labels_legend[:3], loc="upper right", bbox_to_anchor=(0.96, 0.97), ncol=3, frameon=False)
    fig.suptitle("Model-ranking robustness across frozen feature sensitivities", fontsize=14, weight="bold", x=0.06, ha="left")
    fig.text(0.06, 0.01, "Rank 1 is the lowest median held-out-cell MAE_Q. The highlighted XJTU A8 reversal is descriptive, not bootstrap robust.", fontsize=8.5, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, out_dir, "figure7_ml_ranking_robustness", "ML ranking robustness")


CAPTIONS = {
    "figure1_caption.txt": "Figure 1. Frozen study design linking Qref definition to SOH label scale, the absolute alpha x Qref threshold, EOL timing and censoring, point-RUL label availability, and SOH benchmark metrics. XJTU is the primary multi-protocol benchmark, HUST is an external mechanism-level replication under a different protocol and stopping-rule context, and CALCE supports QC and reproducibility. Alpha was evaluated at 0.75, 0.80, and 0.85.",
    "figure2_caption.txt": "Figure 2. Cell-level Qref deviation from the within-dataset nominal reference. Points show individual cells and black diamonds show medians. XJTU measured comparable Qrefs were below nominal in all 55 cells; HUST measured comparable Qrefs were above nominal in all 77 cells. The XJTU initial-test definition is shown as a secondary 47-cell analysis and has no HUST analogue.",
    "figure3_caption.txt": "Figure 3. Cell-level EOL censoring-state transitions at the 80% threshold for nominal versus first-comparable and early-median-5 Qrefs. Heatmaps show exact, interval-censored, and right-censored states without replacing censored states by point EOL values. Annotations report the corresponding change in cells with point-RUL labels.",
    "figure4_caption.txt": "Figure 4. Distribution of exact, interval-censored, and right-censored EOL states at 75%, 80%, and 85% for the common Qref definitions. Percentages are calculated within dataset and definition. HUST results are interpreted in the finite released observation-window and stopping-rule context; XJTU includes sparse periodic diagnostics in several batches.",
    "figure5_caption.txt": "Figure 5. Fraction of cells with an available point-RUL label across dataset, threshold, and Qref definition. Point RUL is reported only when EOL is exactly observed; interval- and right-censored EOL states retain no artificial point label.",
    "figure6_caption.txt": "Figure 6. Primary A9 SOH benchmark summarized by median held-out-cell error. Physical-capacity-space MAE_Q in Ah is contrasted with SOH-label-space MAE in percentage points for Ridge, Random Forest, and Gradient Boosting across common Qref definitions. Results are cell-first summaries using frozen folds and model fits.",
    "figure7_caption.txt": "Figure 7. Model rank positions under median held-out-cell MAE_Q for the primary A9 benchmark, A8 without CV duration, B10 with cycle index, and the post-cycle-5 analysis. The XJTU A8 RF/GB reversal under measured Qrefs is descriptive; corrected difference-of-medians bootstrap confidence intervals crossed zero.",
}


def build_consistency_tests(
    preflight_checks: list[Check],
    overview: pd.DataFrame,
    qref: pd.DataFrame,
    eol80: pd.DataFrame,
    threshold: pd.DataFrame,
    ml: pd.DataFrame,
    robust: pd.DataFrame,
    discussion: pd.DataFrame,
    claims: pd.DataFrame,
    results_md: str,
    before_hashes: dict[str, dict[str, object]],
    after_hashes: dict[str, dict[str, object]],
) -> pd.DataFrame:
    checks = list(preflight_checks)
    add_check(checks, "FROZEN_SOURCE_HASHES_UNCHANGED", before_hashes == after_hashes, before_hashes == after_hashes, True, "all frozen inputs")
    add_check(checks, "DATASET_OVERVIEW_UNIQUE", not overview.Dataset.duplicated().any(), int(overview.Dataset.duplicated().sum()), 0, "table_dataset_overview")
    add_check(checks, "QREF_TABLE_UNIQUE_KEYS", not qref.duplicated(["Dataset", "Qref_Definition"]).any(), int(qref.duplicated(["Dataset", "Qref_Definition"]).sum()), 0, "table_qref_effect_summary")
    add_check(checks, "EOL80_TABLE_UNIQUE_KEYS", not eol80.duplicated(["Dataset", "Qref_Definition"]).any(), int(eol80.duplicated(["Dataset", "Qref_Definition"]).sum()), 0, "table_eol_80pct_summary")
    add_check(checks, "THRESHOLD_TABLE_UNIQUE_KEYS", not threshold.duplicated(["Dataset", "Threshold_Percent", "Qref_Definition"]).any(), int(threshold.duplicated(["Dataset", "Threshold_Percent", "Qref_Definition"]).sum()), 0, "table_threshold_robustness")
    add_check(checks, "ML_PRIMARY_TABLE_UNIQUE_KEYS", not ml.duplicated(["Dataset", "Qref_Definition", "Model"]).any(), int(ml.duplicated(["Dataset", "Qref_Definition", "Model"]).sum()), 0, "table_soh_ml_primary")
    add_check(checks, "ML_ROBUSTNESS_TABLE_UNIQUE_KEYS", not robust.duplicated(["Experiment", "Dataset", "Qref_Definition", "Model"]).any(), int(robust.duplicated(["Experiment", "Dataset", "Qref_Definition", "Model"]).sum()), 0, "table_soh_ml_robustness")
    add_check(checks, "NO_HUST_INITIAL_IN_SYNTHESIS", qref.loc[qref.Dataset.eq("HUST"), "Qref_Definition"].ne("initial_test").all(), sorted(qref.loc[qref.Dataset.eq("HUST"), "Qref_Definition"].unique()), "no initial_test", "qref summary")
    a8_notes = robust.loc[(robust.Experiment == "ROBUSTNESS_A8") & (robust.Dataset == "XJTU") & robust.Qref_Definition.ne("nominal"), "A8_Corrected_Bootstrap_Interpretation"]
    add_check(checks, "A8_CORRECTED_INTERPRETATION_USED", a8_notes.eq("DESCRIPTIVE_RANK_REVERSAL_NOT_BOOTSTRAP_ROBUST").all(), a8_notes.unique().tolist(), ["DESCRIPTIVE_RANK_REVERSAL_NOT_BOOTSTRAP_ROBUST"], "table_soh_ml_robustness")
    add_check(checks, "NO_ROBUST_A8_REVERSAL_CLAIM", "not a bootstrap-robust model-ranking effect" in results_md.lower() and "bootstrap_supported_rank_reversal" not in results_md.lower(), "descriptive/not bootstrap robust language", "no robust reversal claim", "results_manuscript_ready")
    add_check(checks, "HUST_STOPPING_CAVEAT_IN_RESULTS", "HUST released observation-window and stopping-rule context" in results_md, "present", "present", "results_manuscript_ready")
    caveat_map = discussion.Important_Limitation.str.contains("observation", case=False, na=False).any()
    add_check(checks, "HUST_STOPPING_CAVEAT_IN_DISCUSSION_MAP", caveat_map, caveat_map, True, "discussion_evidence_map")
    not_pooled = not qref.Dataset.astype(str).str.contains("COMBINED", case=False).any() and not eol80.Dataset.astype(str).str.contains("COMBINED", case=False).any()
    add_check(checks, "XJTU_HUST_NOT_NUMERICALLY_POOLED", not_pooled, not_pooled, True, "qref and EOL master tables")
    calce_role = overview.loc[overview.Dataset.eq("CALCE"), "Primary_Role_In_Study"].iloc[0]
    add_check(checks, "CALCE_ROLE_IS_QC_SUPPORT", "support" in calce_role.lower() and "replication" not in calce_role.lower(), calce_role, "method/QC/reproducibility support", "table_dataset_overview")
    expected_not_supported = claims.set_index("Claim_ID").loc[["F", "I", "J"], "Classification"].eq("NOT_SUPPORTED").all()
    add_check(checks, "REQUIRED_UNSUPPORTED_CLAIMS_CLASSIFIED", expected_not_supported, claims.set_index("Claim_ID").loc[["F", "I", "J"], "Classification"].to_dict(), {"F": "NOT_SUPPORTED", "I": "NOT_SUPPORTED", "J": "NOT_SUPPORTED"}, "claim_strength_audit")
    sections = [f"## 3.{i}" for i in range(1, 7)]
    add_check(checks, "RESULTS_SECTIONS_3_1_TO_3_6_PRESENT", all(s in results_md for s in sections), [s for s in sections if s in results_md], sections, "results_manuscript_ready")
    return pd.DataFrame([c.__dict__ for c in checks])


def validate_outputs(staging: Path, consistency: pd.DataFrame) -> None:
    required = [
        "table_dataset_overview.csv", "table_qref_effect_summary.csv", "table_eol_80pct_summary.csv",
        "table_threshold_robustness.csv", "table_soh_ml_primary.csv", "table_soh_ml_robustness.csv",
        "supplementary_plan.csv", "results_manuscript_ready.md", "discussion_evidence_map.csv",
        "claim_strength_audit.csv", "final_results_consistency_tests.csv", "processing_log.txt",
    ]
    for i in range(1, 8):
        stem = {
            1: "figure1_study_design", 2: "figure2_qref_deviation", 3: "figure3_eol_transitions_80pct",
            4: "figure4_threshold_robustness", 5: "figure5_rul_availability", 6: "figure6_soh_ml_benchmark",
            7: "figure7_ml_ranking_robustness",
        }[i]
        required.extend([f"{stem}.pdf", f"{stem}.png", f"figure{i}_caption.txt"])
    missing = [name for name in required if not (staging / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing required outputs: {missing}")
    if not consistency["status"].eq("PASS").all():
        failed = consistency.loc[consistency.status.ne("PASS")]
        raise RuntimeError("Consistency tests failed: " + failed[["test_id", "observed", "expected"]].to_json(orient="records"))
    for pdf in staging.glob("*.pdf"):
        if pdf.stat().st_size < 5000:
            raise RuntimeError(f"Suspiciously small PDF: {pdf.name}")
    for png in staging.glob("*.png"):
        if png.stat().st_size < 10000:
            raise RuntimeError(f"Suspiciously small PNG: {png.name}")


def run_full() -> None:
    if FINAL_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {FINAL_DIR}")
    data = load_sources()
    checks = preflight(data)
    ensure_preflight_passed(checks)
    before = source_hashes()
    stamp = datetime.now(TZ).strftime("%Y%m%dT%H%M%S%z")
    staging = OUTPUT_ROOT / f".final_results_synthesis_v1_staging_{stamp}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)

    set_style()
    overview = dataset_overview(data)
    qref = qref_effect_table(data)
    eol80 = eol_80_table(data)
    threshold = threshold_robustness_table(data)
    ml = ml_primary_table(data)
    robust = ml_robustness_table(data)
    supplementary = supplementary_plan()
    results_md = results_text(qref, eol80, threshold, ml, robust, data)
    discussion = discussion_map(qref, eol80, data)
    claims = claim_audit()

    write_csv(overview, staging / "table_dataset_overview.csv")
    write_csv(qref, staging / "table_qref_effect_summary.csv")
    write_csv(eol80, staging / "table_eol_80pct_summary.csv")
    write_csv(threshold, staging / "table_threshold_robustness.csv")
    write_csv(ml, staging / "table_soh_ml_primary.csv")
    write_csv(robust, staging / "table_soh_ml_robustness.csv")
    write_csv(supplementary, staging / "supplementary_plan.csv")
    (staging / "results_manuscript_ready.md").write_text(results_md, encoding="utf-8", newline="\n")
    write_csv(discussion, staging / "discussion_evidence_map.csv")
    write_csv(claims, staging / "claim_strength_audit.csv")

    figure1(staging)
    figure2(staging, data)
    figure3(staging, data, eol80)
    figure4(staging, threshold)
    figure5(staging, threshold)
    figure6(staging, ml)
    figure7(staging, robust)
    for name, text in CAPTIONS.items():
        (staging / name).write_text(text.strip() + "\n", encoding="utf-8", newline="\n")

    after = source_hashes()
    consistency = build_consistency_tests(checks, overview, qref, eol80, threshold, ml, robust, discussion, claims, results_md, before, after)
    write_csv(consistency, staging / "final_results_consistency_tests.csv")

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    log_lines = [
        "FINAL RESULTS SYNTHESIS V1 PROCESSING LOG",
        f"Processing timestamp: {datetime.now(TZ).isoformat()}",
        f"Workspace: {ROOT}",
        f"Versions: {json.dumps(versions, sort_keys=True)}",
        "Scope: frozen-output joins, presentation-only percentages/differences, tables, figures, and descriptive text.",
        "No raw-data reprocessing, model retraining, threshold changes, or new scientific experiments were performed.",
        "",
        "FROZEN SOURCE MANIFEST",
    ]
    for key, meta in sorted(before.items()):
        log_lines.append(f"{key}: {meta['path']} | {meta['size_bytes']} bytes | sha256={meta['sha256']}")
    log_lines.extend(
        [
            "",
            "SOURCE REGRESSION STATUS",
            f"All source and synthesis consistency tests passed: {bool(consistency.status.eq('PASS').all())}",
            f"Consistency test count: {len(consistency)}",
            "Artifact-tool CSV validation: PENDING_EXTERNAL_VALIDATION",
            "PDF render and visual validation: PENDING_EXTERNAL_VALIDATION",
            "",
            "FINAL RESULTS SYNTHESIS V1 COMPLETE —",
            "NO NEW SCIENTIFIC RESULTS GENERATED —",
            "READY FOR MANUSCRIPT WRITING",
        ]
    )
    (staging / "processing_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8", newline="\n")

    validate_outputs(staging, consistency)
    staging.replace(FINAL_DIR)
    print("FINAL RESULTS SYNTHESIS V1 COMPLETE —")
    print("NO NEW SCIENTIFIC RESULTS GENERATED —")
    print("READY FOR MANUSCRIPT WRITING")


def record_external_validation(report_path: Path) -> None:
    if not FINAL_DIR.is_dir():
        raise FileNotFoundError(FINAL_DIR)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError(f"External validation did not pass: {report}")
    log_path = FINAL_DIR / "processing_log.txt"
    text = log_path.read_text(encoding="utf-8")
    text = text.replace(
        "Artifact-tool CSV validation: PENDING_EXTERNAL_VALIDATION",
        f"Artifact-tool CSV validation: PASS ({report.get('validated_csv_count')} CSV files)",
    )
    text = text.replace(
        "PDF render and visual validation: PENDING_EXTERNAL_VALIDATION",
        "PDF render and visual validation: PASS (7 one-page PDFs rendered with Poppler and visually inspected)",
    )
    log_path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--record-external-validation", type=Path)
    args = parser.parse_args()
    selected = sum([args.preflight, args.run, args.record_external_validation is not None])
    if selected != 1:
        parser.error("Choose exactly one operation")
    if args.preflight:
        data = load_sources()
        checks = preflight(data)
        ensure_preflight_passed(checks)
        print(f"PREFLIGHT PASSED: {len(checks)} checks")
    elif args.run:
        run_full()
    else:
        record_external_validation(args.record_external_validation.resolve())


if __name__ == "__main__":
    main()
