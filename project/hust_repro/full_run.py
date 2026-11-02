from __future__ import annotations

import json
import platform
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import make_77cell_full_config
from .pilot import (
    EOL_EXACT,
    EOL_RIGHT_CENSORED,
    _directory_hashes,
    _merge_eol_into_qref_summary,
    _prepare_cycle_inventory,
    _sha256,
    _write_csv,
    build_eol_pairwise,
    build_eol_sensitivity,
    build_native_endpoint_audit,
    build_qc_summary,
    build_rul_labels,
    build_soh_labels,
    build_stopping_rule_audit,
    compute_qref_summary,
)


@dataclass(frozen=True)
class FullRunResult:
    output_dir: Path
    canonical_cell_count: int
    raw_cycle_count: int
    primary_observation_count: int
    soh_row_count: int
    rul_row_count: int
    plot_count: int
    first_qref_stats: dict[str, float]
    first_minus_nominal_pct_stats: dict[str, float]
    median5_qref_stats: dict[str, float]
    eol_status_counts: dict[str, dict[str, int]]
    nominal_first_transitions: dict[str, int]
    nominal_median5_transitions: dict[str, int]
    first_median5_delta_stats: dict[str, float | None]
    lower_bound_stats: dict[str, float | None]
    check_count: int
    invalid_count: int
    unresolved_review_count: int
    regression_tests_passed: bool


def _natural_cell_key(value: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", str(value))
    return tuple(int(number) for number in numbers)


def _canonical_manifest(scan_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(scan_dir / "hust_file_manifest.csv")
    canonical = manifest[
        manifest["File_Role"].eq("CANONICAL")
        & manifest["Included_In_Analysis"].astype(str).str.upper().eq("YES")
    ].copy()
    canonical["_Natural_Order"] = canonical["Cell_ID"].map(_natural_cell_key)
    canonical = canonical.sort_values("_Natural_Order", kind="stable").drop(
        columns="_Natural_Order"
    )
    canonical["Cell_Selection_Order"] = np.arange(1, len(canonical) + 1)
    return canonical.reset_index(drop=True)


def build_protocol_discrepancy_audit(
    fingerprint: pd.DataFrame,
    cell_inventory: pd.DataFrame,
    ordered_cells: Sequence[str],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    limits = config["validated_raw_protocol_limits"]
    fp_lookup = fingerprint.set_index("Cell_ID")
    cell_lookup = cell_inventory.set_index("Cell_ID")
    rows: list[dict[str, Any]] = []
    for order, cell_id in enumerate(ordered_cells, start=1):
        if cell_id not in fp_lookup.index or cell_id not in cell_lookup.index:
            raise RuntimeError(f"MISSING_PROTOCOL_OR_CELL_INVENTORY:{cell_id}")
        fp = fp_lookup.loc[cell_id]
        cell = cell_lookup.loc[cell_id]
        terminal_current = float(fp["Discharge_Step4_Median_Current_A"])
        terminal_crate = float(fp["Estimated_C_Rate_4"])
        current_ok = limits["terminal_stage_abs_current_min_A"] <= abs(
            terminal_current
        ) <= limits["terminal_stage_abs_current_max_A"]
        crate_ok = limits["terminal_stage_CRate_min"] <= abs(
            terminal_crate
        ) <= limits["terminal_stage_CRate_max"]
        stable = str(fp["Protocol_Fingerprint_Status"]) == "STABLE_SCAN_CANDIDATE"
        no_change = str(cell["Protocol_Change_Candidate"]).upper() == "NO"
        consistent = current_ok and crate_ok and stable and no_change
        rows.append(
            {
                "Cell_Selection_Order": order,
                "Cell_ID": cell_id,
                "Native_Cell_ID": cell["Native_Cell_ID"],
                "Source_File": cell["Source_File"],
                "Paper_Final_Stage_CRate": float(
                    config["paper_reported_terminal_discharge_CRate"]
                ),
                "Raw_Observed_Final_Stage_Current_A": terminal_current,
                "Raw_Observed_Final_Stage_CRate_Approx": terminal_crate,
                "Raw_Terminal_Current_Within_Validated_Range": (
                    "YES" if current_ok else "NO"
                ),
                "Raw_Terminal_CRate_Within_Validated_Range": (
                    "YES" if crate_ok else "NO"
                ),
                "Protocol_Fingerprint_ID": fp["Protocol_Fingerprint_ID"],
                "Protocol_Fingerprint_Status": fp["Protocol_Fingerprint_Status"],
                "Protocol_Change_Candidate": cell["Protocol_Change_Candidate"],
                "Paper_vs_Raw_Protocol_Discrepancy": "YES",
                "Protocol_QC_Basis": "RAW_OBSERVED_PROTOCOL",
                "Protocol_Consistency_Status": (
                    "CONSISTENT_WITH_VALIDATED_RAW_OBSERVED_PROTOCOL"
                    if consistent
                    else "PROTOCOL_REVIEW_REQUIRED"
                ),
                "Protocol_Review_Required": "NO" if consistent else "YES",
                "Included_In_Final_Scientific_Aggregation": (
                    "YES" if consistent else "NO"
                ),
            }
        )
    return pd.DataFrame(rows)


def _augment_cycle_inventory(
    cycles: pd.DataFrame, protocol_audit: pd.DataFrame
) -> pd.DataFrame:
    protocol_fields = protocol_audit[
        [
            "Cell_ID",
            "Raw_Observed_Final_Stage_CRate_Approx",
            "Paper_vs_Raw_Protocol_Discrepancy",
            "Protocol_QC_Basis",
            "Protocol_Consistency_Status",
            "Protocol_Review_Required",
            "Included_In_Final_Scientific_Aggregation",
        ]
    ]
    result = cycles.merge(
        protocol_fields, on="Cell_ID", how="left", validate="many_to_one"
    )
    result["Paper_Final_Stage_CRate"] = 1.0
    review = result["Protocol_Review_Required"].eq("YES")
    result.loc[review, "Primary_Capacity_Eligible"] = "NO"
    result["Full_Run_Exclusion_Reason"] = np.where(
        review, "PROTOCOL_REVIEW_REQUIRED", ""
    )
    return result


def add_qref_scientific_fields(qref: pd.DataFrame) -> pd.DataFrame:
    result = qref.copy()
    result["Qref_First_Minus_Nominal_Ah"] = result["First_Minus_Nominal_Ah"]
    result["Qref_First_Minus_Nominal_Pct"] = result["First_Minus_Nominal_Pct"]
    result["Qref_Median5_Minus_Nominal_Ah"] = result[
        "Median5_Minus_Nominal_Ah"
    ]
    result["Qref_Median5_Minus_Nominal_Pct"] = result[
        "Median5_Minus_Nominal_Pct"
    ]
    result["Threshold_Shift_First_Vs_Nominal_Ah"] = (
        result["Threshold_First_Comparable_Ah"] - result["Threshold_Nominal_Ah"]
    )
    result["Threshold_Shift_Median5_Vs_Nominal_Ah"] = (
        result["Threshold_Early_Median5_Ah"] - result["Threshold_Nominal_Ah"]
    )
    return result


def add_pairwise_lower_bound(
    pairwise: pd.DataFrame,
    eol_sensitivity: pd.DataFrame,
    definition_text: str,
) -> pd.DataFrame:
    result = pairwise.copy()
    last_valid = (
        eol_sensitivity.groupby("Cell_ID", sort=False)["Last_Valid_Raw_Cycle"]
        .first()
        .to_dict()
    )
    lower_bounds: list[float] = []
    statuses: list[str] = []
    for _, row in result.iterrows():
        eligible_case = (
            row["Definition_A"] == "nominal"
            and row["EOL_Status_A"] == EOL_RIGHT_CENSORED
            and row["EOL_Status_B"] == EOL_EXACT
        )
        if eligible_case:
            lower_bounds.append(
                float(last_valid[row["Cell_ID"]]) - float(row["EOL_B"])
            )
            statuses.append("AVAILABLE_LOWER_BOUND__NOT_EXACT_SHIFT")
        else:
            lower_bounds.append(np.nan)
            statuses.append("NOT_APPLICABLE")
    result["Minimum_Observed_Earlier_By_Cycles"] = lower_bounds
    result["Minimum_Observed_Earlier_By_Cycles_Status"] = statuses
    result["Minimum_Observed_Earlier_By_Cycles_Definition"] = definition_text
    return result


def build_censoring_summary(
    eol: pd.DataFrame, pairwise: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for definition, group in eol.groupby("Qref_Definition", sort=False):
        rc = group[group["EOL_Status"].eq(EOL_RIGHT_CENSORED)]
        rows.append(
            {
                "Summary_Record_Type": "EOL_DEFINITION",
                "Qref_Definition": definition,
                "Definition_A": "",
                "Definition_B": "",
                "Total_Cell_Count": int(group["Cell_ID"].nunique()),
                "EXACT_OBSERVED_Count": int(group["EOL_Status"].eq(EOL_EXACT).sum()),
                "RIGHT_CENSORED_Count": int(len(rc)),
                "BOTH_EXACT_OBSERVED_Count": np.nan,
                "BOTH_RIGHT_CENSORED_Count": np.nan,
                "RIGHT_CENSORED_TO_EXACT_OBSERVED_Count": np.nan,
                "EXACT_OBSERVED_TO_RIGHT_CENSORED_Count": np.nan,
                "Exact_Delta_Available_Count": np.nan,
                "Lower_Bound_Available_Count": np.nan,
                "Right_Censored_Point_Value_Violation_Count": int(
                    rc["EOL_Raw_Cycle"].notna().sum()
                    + rc["EOL_Capacity_Ah"].notna().sum()
                ),
            }
        )
    for (definition_a, definition_b), group in pairwise.groupby(
        ["Definition_A", "Definition_B"], sort=False
    ):
        transitions = group["Censoring_Status_Change"].value_counts()
        rows.append(
            {
                "Summary_Record_Type": "PAIRWISE_TRANSITION",
                "Qref_Definition": "",
                "Definition_A": definition_a,
                "Definition_B": definition_b,
                "Total_Cell_Count": int(group["Cell_ID"].nunique()),
                "EXACT_OBSERVED_Count": np.nan,
                "RIGHT_CENSORED_Count": np.nan,
                "BOTH_EXACT_OBSERVED_Count": int(
                    transitions.get("BOTH_EXACT_OBSERVED", 0)
                ),
                "BOTH_RIGHT_CENSORED_Count": int(
                    transitions.get("BOTH_RIGHT_CENSORED", 0)
                ),
                "RIGHT_CENSORED_TO_EXACT_OBSERVED_Count": int(
                    transitions.get("RIGHT_CENSORED_TO_EXACT_OBSERVED", 0)
                ),
                "EXACT_OBSERVED_TO_RIGHT_CENSORED_Count": int(
                    transitions.get("EXACT_OBSERVED_TO_RIGHT_CENSORED", 0)
                ),
                "Exact_Delta_Available_Count": int(
                    group["Delta_EOL_Cycles"].notna().sum()
                ),
                "Lower_Bound_Available_Count": int(
                    group["Minimum_Observed_Earlier_By_Cycles"].notna().sum()
                ),
                "Right_Censored_Point_Value_Violation_Count": 0,
            }
        )
    return pd.DataFrame(rows)


def build_full_stopping_rule_audit(
    cycles: pd.DataFrame,
    eol: pd.DataFrame,
    fixed_threshold_Ah: float,
    near_tolerance_Ah: float,
) -> pd.DataFrame:
    result = build_stopping_rule_audit(cycles, eol, fixed_threshold_Ah)
    result["Native_End_Near_Nominal_80pct"] = np.where(
        result["Last_Valid_Capacity_Minus_0p880000_Ah"].abs()
        <= near_tolerance_Ah,
        "YES",
        "NO",
    )
    result["Native_End_Near_Nominal_80pct_Tolerance_Ah"] = near_tolerance_Ah
    return result


def build_qc_review_candidates(
    cycles: pd.DataFrame, eol: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "Cell_Selection_Order",
        "Cell_ID",
        "Native_Cell_ID",
        "Source_File",
        "Raw_Cycle_Index",
        "Primary_Capacity_Ah",
        "Auto_QC_Flag",
        "Auto_QC_Reason_Codes",
        "Capacity_Jump_Flag",
        "Primary_Capacity_Eligible",
        "Protocol_Consistency_Status",
        "Potential_EOL_Impact",
        "EOL_Definitions_Affected",
        "Manual_Review_Required",
        "Review_Reason",
    ]
    candidates = cycles[
        cycles["Auto_QC_Flag"].ne("PASS")
        | cycles["Protocol_Review_Required"].eq("YES")
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    exact = eol[eol["EOL_Status"].eq(EOL_EXACT)].copy()
    affected_lookup = (
        exact.groupby(["Cell_ID", "EOL_Raw_Cycle"])["Qref_Definition"]
        .apply(lambda values: "|".join(values.astype(str)))
        .to_dict()
    )
    affected: list[str] = []
    definitions: list[str] = []
    review_reason: list[str] = []
    for _, row in candidates.iterrows():
        key = (row["Cell_ID"], float(row["Raw_Cycle_Index"]))
        definitions_value = affected_lookup.get(key, "")
        protocol_review = row["Protocol_Review_Required"] == "YES"
        affects_eol = bool(definitions_value)
        affected.append("YES" if affects_eol else "NO")
        definitions.append(definitions_value)
        reasons = []
        if protocol_review:
            reasons.append("PROTOCOL_REVIEW_REQUIRED")
        if affects_eol:
            reasons.append("CHECK_OR_INVALID_AFFECTS_EOL_OUTCOME")
        if not reasons:
            reasons.append("NONPASS_QC_REVIEW")
        review_reason.append("|".join(reasons))
    candidates["Potential_EOL_Impact"] = affected
    candidates["EOL_Definitions_Affected"] = definitions
    candidates["Manual_Review_Required"] = "YES"
    candidates["Review_Reason"] = review_reason
    return candidates[columns].sort_values(
        ["Cell_Selection_Order", "Raw_Cycle_Index"], kind="stable"
    )


def build_full_file_manifest(
    canonical_manifest: pd.DataFrame,
    cell_inventory: pd.DataFrame,
    protocol_audit: pd.DataFrame,
    raw_hashes_before: Mapping[str, str],
    raw_hashes_after: Mapping[str, str],
) -> pd.DataFrame:
    manifest = canonical_manifest.copy()
    cells = cell_inventory[
        ["Cell_ID", "Cycle_Count", "First_Raw_Cycle_Index", "Last_Raw_Cycle_Index"]
    ]
    protocol = protocol_audit[
        [
            "Cell_ID",
            "Protocol_Consistency_Status",
            "Protocol_Review_Required",
            "Included_In_Final_Scientific_Aggregation",
        ]
    ]
    manifest = manifest.merge(cells, on="Cell_ID", how="left", validate="one_to_one")
    manifest = manifest.merge(
        protocol, on="Cell_ID", how="left", validate="one_to_one"
    )
    manifest["Verified_Raw_SHA256_Before"] = manifest["Cell_ID"].map(
        raw_hashes_before
    )
    manifest["Verified_Raw_SHA256_After"] = manifest["Cell_ID"].map(
        raw_hashes_after
    )
    manifest["Raw_File_Unchanged_During_Run"] = np.where(
        manifest["Verified_Raw_SHA256_Before"].eq(
            manifest["Verified_Raw_SHA256_After"]
        )
        & manifest["SHA256"].eq(manifest["Verified_Raw_SHA256_Before"]),
        "YES",
        "NO",
    )
    return manifest.sort_values("Cell_Selection_Order", kind="stable")


def build_cell_summary(
    qref: pd.DataFrame,
    qc: pd.DataFrame,
    native_endpoint: pd.DataFrame,
    stopping: pd.DataFrame,
    protocol_audit: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    qc_fields = qc[
        [
            "Cell_ID",
            "Total_Raw_Cycles",
            "PASS_Count",
            "CHECK_Count",
            "INVALID_CANDIDATE_Count",
            "Primary_Capacity_Eligible_Count",
            "Primary_Capacity_Ineligible_Count",
            "Capacity_Jump_CHECK_Count",
            "Pilot_QC_Status",
            "Pilot_QC_Issue_Codes",
        ]
    ].rename(
        columns={
            "Pilot_QC_Status": "Full_Run_QC_Status",
            "Pilot_QC_Issue_Codes": "Full_Run_QC_Issue_Codes",
        }
    )
    native_fields = native_endpoint[
        [
            "Cell_ID",
            "Native_Dataset_Endpoint_Cycle",
            "Native_Dataset_Endpoint_Status",
            "Native_RUL_At_Final_Cycle",
            "Last_Valid_Raw_Cycle",
            "Last_Valid_Capacity_Ah",
        ]
    ]
    stopping_fields = stopping[
        [
            "Cell_ID",
            "Last_Valid_Capacity_Minus_0p880000_Ah",
            "Strict_Nominal_Threshold_Crossed",
            "Strict_Fixed_0p88_Threshold_Crossed",
            "Native_End_Near_Nominal_80pct",
        ]
    ]
    protocol_fields = protocol_audit[
        [
            "Cell_ID",
            "Protocol_Consistency_Status",
            "Protocol_Review_Required",
            "Included_In_Final_Scientific_Aggregation",
        ]
    ]
    lower = pairwise[
        pairwise["Definition_A"].eq("nominal")
        & pairwise["Definition_B"].isin(
            ["first_comparable", "early_median5_comparable"]
        )
    ][
        [
            "Cell_ID",
            "Definition_B",
            "Minimum_Observed_Earlier_By_Cycles",
        ]
    ].copy()
    lower["Lower_Bound_Field"] = lower["Definition_B"].map(
        {
            "first_comparable": "Minimum_Observed_Earlier_By_Cycles_First_Comparable",
            "early_median5_comparable": "Minimum_Observed_Earlier_By_Cycles_Early_Median5",
        }
    )
    lower_wide = lower.pivot(
        index="Cell_ID",
        columns="Lower_Bound_Field",
        values="Minimum_Observed_Earlier_By_Cycles",
    ).reset_index()
    result = qref.merge(qc_fields, on="Cell_ID", how="left", validate="one_to_one")
    result = result.merge(
        native_fields, on="Cell_ID", how="left", validate="one_to_one"
    )
    result = result.merge(
        stopping_fields, on="Cell_ID", how="left", validate="one_to_one"
    )
    result = result.merge(
        protocol_fields, on="Cell_ID", how="left", validate="one_to_one"
    )
    result = result.merge(
        lower_wide, on="Cell_ID", how="left", validate="one_to_one"
    )
    return result.sort_values("Cell_Selection_Order", kind="stable")


def _plot_full_capacity(
    path: Path,
    cell_id: str,
    cycles: pd.DataFrame,
    qref: pd.Series,
    eol: pd.DataFrame,
) -> None:
    ordered = cycles.sort_values("Raw_Cycle_Index", kind="stable")
    eligible = ordered[ordered["Primary_Capacity_Eligible"].eq("YES")]
    check = ordered[ordered["Auto_QC_Flag"].eq("CHECK")]
    invalid = ordered[ordered["Auto_QC_Flag"].eq("INVALID_CANDIDATE")]
    eol_lookup = eol.set_index("Qref_Definition")

    fig, axis = plt.subplots(figsize=(11.5, 6.6))
    axis.plot(
        eligible["Raw_Cycle_Index"],
        eligible["Primary_Capacity_Ah"],
        color="#1F2937",
        linewidth=0.85,
        label="Eligible measured capacity",
    )
    if not check.empty:
        axis.scatter(
            check["Raw_Cycle_Index"],
            check["Primary_Capacity_Ah"],
            color="#D97706",
            s=15,
            label="CHECK",
            zorder=4,
        )
    if not invalid.empty:
        axis.scatter(
            invalid["Raw_Cycle_Index"],
            invalid["Primary_Capacity_Ah"],
            color="#DC2626",
            s=18,
            marker="x",
            label="INVALID_CANDIDATE",
            zorder=5,
        )

    thresholds = [
        (0.880000, "Nominal and fixed threshold: 0.880000 Ah", "#111827", "--"),
        (
            float(qref["Threshold_First_Comparable_Ah"]),
            "0.8 × first-comparable Qref",
            "#2563EB",
            "-.",
        ),
        (
            float(qref["Threshold_Early_Median5_Ah"]),
            "0.8 × early-median5 Qref",
            "#7C3AED",
            ":",
        ),
    ]
    for value, label, color, style in thresholds:
        axis.axhline(value, color=color, linestyle=style, linewidth=1.35, label=label)

    vertical_specs = [
        ("nominal", "#111827", "Nominal/fixed EOL"),
        ("first_comparable", "#2563EB", "First-comparable EOL"),
        ("early_median5_comparable", "#7C3AED", "Early-median5 EOL"),
    ]
    for definition, color, label in vertical_specs:
        row = eol_lookup.loc[definition]
        if row["EOL_Status"] == EOL_EXACT:
            axis.axvline(
                int(row["EOL_Raw_Cycle"]),
                color=color,
                linestyle="--",
                linewidth=1.05,
                alpha=0.85,
                label=f"{label}: cycle {int(row['EOL_Raw_Cycle'])}",
            )

    last = eligible.iloc[-1]
    nominal_status = str(eol_lookup.loc["nominal", "EOL_Status"])
    fixed_status = str(eol_lookup.loc["fixed_absolute_0p88Ah", "EOL_Status"])
    axis.scatter(
        [last["Raw_Cycle_Index"]],
        [last["Primary_Capacity_Ah"]],
        color="#D97706",
        s=28,
        zorder=6,
    )
    axis.annotate(
        f"Last valid: cycle {int(last['Raw_Cycle_Index'])}, "
        f"{float(last['Primary_Capacity_Ah']):.6f} Ah\n"
        f"Nominal/fixed EOL: {nominal_status}/{fixed_status}",
        xy=(last["Raw_Cycle_Index"], last["Primary_Capacity_Ah"]),
        xytext=(0.56, 0.14),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#92400E"},
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#F59E0B"},
    )
    axis.set_title(
        f"{cell_id} capacity and Qref-dependent EOL thresholds\n"
        "Measured native dq / 1000; no smoothing or interpolation",
        fontweight="bold",
    )
    axis.set_xlabel("Raw Cycle Index")
    axis.set_ylabel("Primary Capacity (Ah)")
    axis.grid(True, color="#D1D5DB", linewidth=0.5, alpha=0.6)
    axis.legend(loc="upper right", fontsize=7.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _stats(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="raise")
    return {
        "min": float(numeric.min()),
        "median": float(numeric.median()),
        "max": float(numeric.max()),
    }


def _nullable_stats(series: pd.Series, include_abs_median: bool = False) -> dict[str, float | None]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    result: dict[str, float | None] = {
        "min": float(numeric.min()) if not numeric.empty else None,
        "median": float(numeric.median()) if not numeric.empty else None,
        "max": float(numeric.max()) if not numeric.empty else None,
    }
    if include_abs_median:
        result["median_absolute"] = (
            float(numeric.abs().median()) if not numeric.empty else None
        )
    return result


def _summary_metrics(
    cycles: pd.DataFrame,
    qref: pd.DataFrame,
    eol: pd.DataFrame,
    pairwise: pd.DataFrame,
    protocol_audit: pd.DataFrame,
) -> dict[str, Any]:
    status_counts: dict[str, dict[str, int]] = {}
    for definition, group in eol.groupby("Qref_Definition", sort=False):
        status_counts[definition] = {
            EOL_EXACT: int(group["EOL_Status"].eq(EOL_EXACT).sum()),
            EOL_RIGHT_CENSORED: int(
                group["EOL_Status"].eq(EOL_RIGHT_CENSORED).sum()
            ),
        }

    def transition_counts(definition_b: str) -> dict[str, int]:
        subset = pairwise[
            pairwise["Definition_A"].eq("nominal")
            & pairwise["Definition_B"].eq(definition_b)
        ]
        return {
            str(key): int(value)
            for key, value in subset["Censoring_Status_Change"]
            .value_counts()
            .items()
        }

    first_median = pairwise[
        pairwise["Definition_A"].eq("first_comparable")
        & pairwise["Definition_B"].eq("early_median5_comparable")
    ]
    return {
        "first_qref_stats": _stats(qref["Qref_First_Comparable_Ah"]),
        "first_minus_nominal_pct_stats": _stats(
            qref["Qref_First_Minus_Nominal_Pct"]
        ),
        "median5_qref_stats": _stats(
            qref["Qref_Early_Median5_Comparable_Ah"]
        ),
        "eol_status_counts": status_counts,
        "nominal_first_transitions": transition_counts("first_comparable"),
        "nominal_median5_transitions": transition_counts(
            "early_median5_comparable"
        ),
        "first_median5_delta_stats": _nullable_stats(
            first_median["Delta_EOL_Cycles"], include_abs_median=True
        ),
        "lower_bound_stats": _nullable_stats(
            pairwise["Minimum_Observed_Earlier_By_Cycles"]
        ),
        "check_count": int(cycles["Auto_QC_Flag"].eq("CHECK").sum()),
        "invalid_count": int(
            cycles["Auto_QC_Flag"].eq("INVALID_CANDIDATE").sum()
        ),
        "unresolved_review_count": int(
            protocol_audit["Protocol_Review_Required"].eq("YES").sum()
        ),
    }


def _series_equal_exact(left: pd.Series, right: pd.Series) -> bool:
    left_values = left.reset_index(drop=True)
    right_values = right.reset_index(drop=True)
    if len(left_values) != len(right_values):
        return False
    return bool(
        (
            left_values.eq(right_values)
            | (left_values.isna() & right_values.isna())
        ).all()
    )


def _series_equal_csv12(left: pd.Series, right: pd.Series) -> bool:
    """Compare numeric values exactly as delivered by the frozen %.12g CSV format."""

    left_values = pd.to_numeric(left.reset_index(drop=True), errors="raise")
    right_values = pd.to_numeric(right.reset_index(drop=True), errors="raise")
    if len(left_values) != len(right_values):
        return False
    for left_value, right_value in zip(left_values, right_values):
        if pd.isna(left_value) and pd.isna(right_value):
            continue
        if pd.isna(left_value) or pd.isna(right_value):
            return False
        if format(float(left_value), ".12g") != format(float(right_value), ".12g"):
            return False
    return True


def build_full_regression_tests(
    *,
    config: Mapping[str, Any],
    canonical_manifest: pd.DataFrame,
    cell_inventory: pd.DataFrame,
    cycles: pd.DataFrame,
    qref: pd.DataFrame,
    soh: pd.DataFrame,
    eol: pd.DataFrame,
    rul: pd.DataFrame,
    protocol_audit: pd.DataFrame,
    pilot_qref: pd.DataFrame,
    pilot_eol: pd.DataFrame,
    raw_files_unchanged: bool,
    scan_unchanged: bool,
    validation_unchanged: bool,
    pilot_unchanged: bool,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(
        test_id: str,
        expected: str,
        observed: str,
        passed: bool,
        details: str = "",
    ) -> None:
        rows.append(
            {
                "Test_ID": test_id,
                "Scope": "HUST_77CELL_FINAL_V1",
                "Expected": expected,
                "Observed": observed,
                "Status": "PASS" if passed else "FAIL",
                "Details": details,
            }
        )

    expected_cells = int(config["expected_canonical_cell_count"])
    observed_cells = int(canonical_manifest["Cell_ID"].nunique())
    add(
        "CANONICAL_CELL_COUNT",
        str(expected_cells),
        str(observed_cells),
        observed_cells == expected_cells and len(canonical_manifest) == expected_cells,
    )
    expected_cycles = int(config["expected_raw_cycle_count"])
    add(
        "RAW_CYCLE_COUNT",
        str(expected_cycles),
        str(len(cycles)),
        len(cycles) == expected_cycles,
    )
    duplicate_keys = int(cycles.duplicated(["Cell_ID", "Raw_Cycle_Index"]).sum())
    add(
        "RAW_CYCLE_KEYS_UNIQUE",
        "0 duplicate Cell_ID + Raw_Cycle_Index keys",
        f"duplicates={duplicate_keys}",
        duplicate_keys == 0,
    )

    no_renumber = True
    renumber_details: list[str] = []
    cell_lookup = cell_inventory.set_index("Cell_ID")
    for cell_id, group in cycles.groupby("Cell_ID", sort=False):
        source = cell_lookup.loc[cell_id]
        observed = np.sort(
            pd.to_numeric(group["Raw_Cycle_Index"], errors="raise")
            .astype(int)
            .to_numpy()
        )
        expected = np.arange(
            int(source["First_Raw_Cycle_Index"]),
            int(source["Last_Raw_Cycle_Index"]) + 1,
        )
        cell_ok = (
            len(observed) == int(source["Cycle_Count"])
            and len(np.unique(observed)) == int(source["Unique_Cycle_Count"])
            and np.array_equal(observed, expected)
        )
        if not cell_ok:
            no_renumber = False
            renumber_details.append(cell_id)
    add(
        "NO_CYCLE_RENUMBERING",
        "raw numeric cycle identities preserved exactly",
        "all 77 match frozen cell inventory" if no_renumber else "mismatch",
        no_renumber,
        "|".join(renumber_details),
    )

    nominal = pd.to_numeric(qref["Qref_Nominal_Ah"], errors="raise")
    add(
        "QREF_NOMINAL_1P1_ALL_CELLS",
        "1.100000 Ah for all 77 cells",
        f"exact={int(nominal.eq(1.1).sum())}/{len(nominal)}",
        len(nominal) == expected_cells and nominal.eq(1.1).all(),
    )
    initial_unavailable = qref["Qref_Initial_Test_Ah"].isna() & qref[
        "Qref_Initial_Test_Status"
    ].eq("UNAVAILABLE_NO_SEPARATE_STANDARDIZED_BOL_TEST")
    add(
        "INITIAL_TEST_QREF_UNAVAILABLE_ALL_CELLS",
        "77/77 unavailable with NA capacity",
        f"{int(initial_unavailable.sum())}/{len(initial_unavailable)}",
        len(initial_unavailable) == expected_cells and initial_unavailable.all(),
    )

    first_source_ok = True
    median_source_ok = True
    source_failures: list[str] = []
    for _, row in qref.iterrows():
        eligible = cycles[
            cycles["Cell_ID"].eq(row["Cell_ID"])
            & cycles["Primary_Capacity_Eligible"].eq("YES")
        ].sort_values("Raw_Cycle_Index", kind="stable")
        expected_first = int(eligible.iloc[0]["Raw_Cycle_Index"])
        expected_five = "|".join(
            str(int(value)) for value in eligible.head(5)["Raw_Cycle_Index"]
        )
        cell_first_ok = int(row["Qref_First_Comparable_Source_Cycle"]) == expected_first
        cell_median_ok = (
            str(row["Qref_Early_Median5_Comparable_Source_Cycles"])
            == expected_five
        )
        first_source_ok &= cell_first_ok
        median_source_ok &= cell_median_ok
        if not (cell_first_ok and cell_median_ok):
            source_failures.append(str(row["Cell_ID"]))
    add(
        "FIRST_COMPARABLE_SOURCE_NUMERIC",
        "numeric first valid comparable cycle",
        "all 77 checked",
        first_source_ok,
        "|".join(source_failures),
    )
    add(
        "EARLY_MEDIAN5_SOURCE_NUMERIC",
        "first five valid cycles in numeric Raw_Cycle_Index order",
        "all 77 checked",
        median_source_ok,
        "|".join(source_failures),
    )

    direct_soh = soh["Capacity_Ah"] / soh["Qref_Ah"]
    soh_error = float((soh["SOH_Fraction"] - direct_soh).abs().max())
    above_one = int(soh["SOH_Fraction"].gt(1.0).sum())
    add(
        "NO_SOH_CLIPPING",
        "direct capacity/Qref ratio retained, including values > 1",
        f"max_error={soh_error:.12g}; rows_above_1={above_one}",
        soh_error <= 1e-12
        and above_one > 0
        and soh["SOH_Clipped"].eq("NO").all(),
    )
    source_match = cycles["Primary_Capacity_Source_Match"].eq("YES")
    add(
        "CAPACITY_EQUALS_NATIVE_DQ_DIV_1000",
        "all 146122 raw cycles",
        f"{int(source_match.sum())}/{len(source_match)}",
        source_match.all() and len(source_match) == expected_cycles,
    )
    exact_rul = rul[rul["EOL_Status"].eq(EOL_EXACT)]
    exact_rul_error = (
        exact_rul["RUL_Cycles"]
        - (exact_rul["EOL_Raw_Cycle"] - exact_rul["Raw_Cycle_Index"])
    ).abs().max()
    add(
        "NO_NATIVE_RUL_CONTAMINATION",
        "native RUL never used; exact derived RUL follows EOL-cycle minus current-cycle",
        f"native_used={int(rul['Native_RUL_Used'].ne('NO').sum())}; "
        f"max_formula_error={exact_rul_error:.12g}",
        rul["Native_RUL_Used"].eq("NO").all()
        and rul["RUL_Source"].eq("STUDY_QREF_OR_FIXED_THRESHOLD_DERIVED").all()
        and exact_rul_error == 0,
    )

    censored_eol = eol[eol["EOL_Status"].eq(EOL_RIGHT_CENSORED)]
    add(
        "RIGHT_CENSORED_EOL_HAS_NO_POINT",
        "right-censored EOL cycle and capacity are NA",
        f"rows={len(censored_eol)}; point_values="
        f"{int(censored_eol['EOL_Raw_Cycle'].notna().sum() + censored_eol['EOL_Capacity_Ah'].notna().sum())}",
        censored_eol["EOL_Raw_Cycle"].isna().all()
        and censored_eol["EOL_Capacity_Ah"].isna().all(),
    )
    censored_rul = rul[rul["EOL_Status"].eq(EOL_RIGHT_CENSORED)]
    add(
        "RIGHT_CENSORED_RUL_HAS_NO_POINT",
        "all censored RUL point labels are NA",
        f"rows={len(censored_rul)}; numeric={int(censored_rul['RUL_Cycles'].notna().sum())}",
        not censored_rul.empty
        and censored_rul["RUL_Cycles"].isna().all()
        and censored_rul["RUL_Status"].eq("RIGHT_CENSORED_EOL").all(),
    )

    fixed = eol[eol["Qref_Definition"].eq("fixed_absolute_0p88Ah")]
    add(
        "FIXED_THRESHOLD_EXACTLY_0P88",
        "0.880000 Ah for all 77 cells",
        f"min={fixed['EOThreshold_Ah'].min():.12g}; max={fixed['EOThreshold_Ah'].max():.12g}",
        len(fixed) == expected_cells and fixed["EOThreshold_Ah"].eq(0.88).all(),
    )
    nominal_eol = eol[eol["Qref_Definition"].eq("nominal")].sort_values(
        "Cell_ID", kind="stable"
    )
    fixed_eol = fixed.sort_values("Cell_ID", kind="stable")
    nominal_fixed_equal = (
        nominal_eol["Cell_ID"].reset_index(drop=True).eq(
            fixed_eol["Cell_ID"].reset_index(drop=True)
        ).all()
        and nominal_eol["EOL_Status"].reset_index(drop=True).eq(
            fixed_eol["EOL_Status"].reset_index(drop=True)
        ).all()
        and _series_equal_exact(
            nominal_eol["EOL_Raw_Cycle"], fixed_eol["EOL_Raw_Cycle"]
        )
    )
    add(
        "NOMINAL_EQUALS_FIXED_CONTROL_EOL",
        "identical strict status and EOL cycle for every cell",
        f"matching={int(nominal_fixed_equal)}/{1}",
        nominal_fixed_equal,
    )

    discrepancy_ok = (
        protocol_audit["Paper_vs_Raw_Protocol_Discrepancy"].eq("YES").all()
        and protocol_audit["Protocol_QC_Basis"].eq("RAW_OBSERVED_PROTOCOL").all()
        and protocol_audit["Paper_Final_Stage_CRate"].eq(1.0).all()
    )
    add(
        "PAPER_RAW_PROTOCOL_DISCREPANCY_RETAINED",
        "paper 1C, raw approximately 2C, RAW_OBSERVED_PROTOCOL for all 77",
        f"{int(discrepancy_ok)}/{1}",
        discrepancy_ok and len(protocol_audit) == expected_cells,
    )
    invalid_primary = cycles[
        cycles["Auto_QC_Flag"].eq("INVALID_CANDIDATE")
        & cycles["Primary_Capacity_Eligible"].eq("YES")
    ]
    add(
        "NO_INVALID_IN_PRIMARY",
        "0 INVALID_CANDIDATE rows eligible",
        f"count={len(invalid_primary)}",
        invalid_primary.empty,
    )
    check_rows = cycles[cycles["Auto_QC_Flag"].eq("CHECK")]
    explicit_exclusion = (
        check_rows["Protocol_Review_Required"].eq("YES")
        | check_rows["Primary_Capacity_Candidate"].ne("YES")
        | check_rows["Capacity_Semantics_Match"].ne("YES")
        | check_rows["Charge_Protocol_Traceable"].ne("YES")
        | check_rows["Charge_Complete_Scan_Candidate"].ne("YES")
        | check_rows["Full_Discharge_Traceable"].ne("YES")
        | check_rows["Full_Discharge_Complete_Scan_Candidate"].ne("YES")
        | check_rows["Severe_Voltage_Anomaly"].eq("YES")
    )
    unexplained_check_exclusion = check_rows[
        check_rows["Primary_Capacity_Eligible"].eq("NO") & ~explicit_exclusion
    ]
    add(
        "CHECK_NOT_AUTOMATICALLY_EXCLUDED_WITHOUT_RULE",
        "no CHECK excluded without a documented protocol/data rule",
        f"CHECK={len(check_rows)}; unexplained_exclusions={len(unexplained_check_exclusion)}",
        unexplained_check_exclusion.empty,
    )

    review_protocol = protocol_audit[
        protocol_audit["Protocol_Review_Required"].eq("YES")
    ]
    protocol_review_handled = review_protocol[
        "Included_In_Final_Scientific_Aggregation"
    ].eq("NO").all()
    add(
        "PROTOCOL_CONSISTENCY_AND_REVIEW_HANDLING",
        "stable raw-observed fingerprints, or explicit review exclusion",
        f"review_required={len(review_protocol)}",
        protocol_review_handled,
    )
    add(
        "DENSE_EOL_STATUS_ONLY",
        "EXACT_OBSERVED or RIGHT_CENSORED",
        "|".join(sorted(eol["EOL_Status"].unique())),
        set(eol["EOL_Status"]).issubset({EOL_EXACT, EOL_RIGHT_CENSORED}),
    )

    pilot_cells = list(config["frozen_pilot_cells"])
    full_pilot_qref = qref[qref["Cell_ID"].isin(pilot_cells)].sort_values(
        "Cell_ID", kind="stable"
    )
    frozen_qref = pilot_qref[pilot_qref["Cell_ID"].isin(pilot_cells)].sort_values(
        "Cell_ID", kind="stable"
    )
    qref_numeric_fields = [
        "Qref_Nominal_Ah",
        "Qref_First_Comparable_Ah",
        "Qref_Early_Median5_Comparable_Ah",
        "Threshold_Nominal_Ah",
        "Threshold_First_Comparable_Ah",
        "Threshold_Early_Median5_Ah",
    ]
    qref_exact = len(full_pilot_qref) == len(frozen_qref) == 8
    if qref_exact:
        qref_exact = full_pilot_qref["Cell_ID"].reset_index(drop=True).eq(
            frozen_qref["Cell_ID"].reset_index(drop=True)
        ).all()
        for field in qref_numeric_fields:
            qref_exact &= _series_equal_csv12(
                full_pilot_qref[field], frozen_qref[field]
            )
    add(
        "FROZEN_8CELL_PILOT_QREFS_EXACT",
        "all frozen Qrefs and thresholds reproduce the exact delivered %.12g CSV values",
        f"matched={int(qref_exact)}/1",
        bool(qref_exact),
    )
    source_exact = bool(
        len(full_pilot_qref) == 8
        and full_pilot_qref["Qref_First_Comparable_Source_Cycle"]
        .reset_index(drop=True)
        .eq(frozen_qref["Qref_First_Comparable_Source_Cycle"].reset_index(drop=True))
        .all()
        and full_pilot_qref["Qref_Early_Median5_Comparable_Source_Cycles"]
        .astype(str)
        .reset_index(drop=True)
        .eq(
            frozen_qref["Qref_Early_Median5_Comparable_Source_Cycles"]
            .astype(str)
            .reset_index(drop=True)
        )
        .all()
    )
    add(
        "FROZEN_8CELL_PILOT_SOURCE_CYCLES_EXACT",
        "first and median5 source cycles unchanged",
        f"matched={int(source_exact)}/1",
        source_exact,
    )

    full_pilot_eol = eol[eol["Cell_ID"].isin(pilot_cells)][
        ["Cell_ID", "Qref_Definition", "EOL_Status", "EOL_Raw_Cycle"]
    ].sort_values(["Cell_ID", "Qref_Definition"], kind="stable")
    frozen_eol = pilot_eol[pilot_eol["Cell_ID"].isin(pilot_cells)][
        ["Cell_ID", "Qref_Definition", "EOL_Status", "EOL_Raw_Cycle"]
    ].sort_values(["Cell_ID", "Qref_Definition"], kind="stable")
    eol_status_exact = bool(
        len(full_pilot_eol) == len(frozen_eol) == 32
        and full_pilot_eol[["Cell_ID", "Qref_Definition", "EOL_Status"]]
        .reset_index(drop=True)
        .eq(
            frozen_eol[["Cell_ID", "Qref_Definition", "EOL_Status"]]
            .reset_index(drop=True)
        )
        .all()
        .all()
    )
    eol_cycle_exact = eol_status_exact and _series_equal_exact(
        full_pilot_eol["EOL_Raw_Cycle"], frozen_eol["EOL_Raw_Cycle"]
    )
    add(
        "FROZEN_8CELL_PILOT_EOL_STATUS_EXACT",
        "all four definition statuses unchanged for all 8 cells",
        f"matched={int(eol_status_exact)}/1",
        eol_status_exact,
    )
    add(
        "FROZEN_8CELL_PILOT_EOL_CYCLES_EXACT",
        "all exact measured-Qref EOL cycles unchanged",
        f"matched={int(eol_cycle_exact)}/1",
        bool(eol_cycle_exact),
    )
    add(
        "RAW_FILES_UNCHANGED",
        "all 77 SHA-256 values unchanged and match frozen manifest",
        "YES" if raw_files_unchanged else "NO",
        raw_files_unchanged,
    )
    add(
        "FROZEN_INPUT_OUTPUTS_UNCHANGED",
        "scan, protocol validation, and 8-cell pilot unchanged",
        f"scan={scan_unchanged}; validation={validation_unchanged}; pilot={pilot_unchanged}",
        scan_unchanged and validation_unchanged and pilot_unchanged,
    )
    return pd.DataFrame(rows)


def _format_stats(values: Mapping[str, float | None]) -> str:
    if values.get("median") is None:
        return "NA / NA / NA"
    return (
        f"{values['min']:.12g} / {values['median']:.12g} / "
        f"{values['max']:.12g}"
    )


def _processing_log(
    *,
    config: Mapping[str, Any],
    file_manifest: pd.DataFrame,
    cycles: pd.DataFrame,
    primary: pd.DataFrame,
    qref: pd.DataFrame,
    eol: pd.DataFrame,
    pairwise: pd.DataFrame,
    tests: pd.DataFrame,
    summary: Mapping[str, Any],
    plot_count: int,
    scan_unchanged: bool,
    validation_unchanged: bool,
    pilot_unchanged: bool,
) -> str:
    lines = [
        "HUST 77-cell Qref / EOL final replication v1",
        f"Processing timestamp UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Python version: {platform.python_version()}",
        f"numpy version: {np.__version__}",
        f"pandas version: {pd.__version__}",
        f"matplotlib version: {matplotlib.__version__}",
        "",
        "Frozen scientific rules:",
        "- Primary_Capacity_Ah = native dq / 1000",
        "- Protocol_QC_Basis = RAW_OBSERVED_PROTOCOL",
        "- Paper terminal stage = 1C; public raw terminal stage approximately 2C",
        "- Strict EOL comparison uses unrounded measured capacity <= threshold",
        "- EOL statuses are EXACT_OBSERVED or RIGHT_CENSORED",
        "- No smoothing, interpolation, capacity correction, cycle renumbering, or model training",
        "- Native RUL and native endpoint remain audit-only",
        "",
        "Config:",
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "Raw source integrity:",
    ]
    for _, row in file_manifest.iterrows():
        lines.append(
            f"- {row['Cell_ID']} {row['Relative_Path']}: "
            f"sha256={row['Verified_Raw_SHA256_Before']} "
            f"unchanged={row['Raw_File_Unchanged_During_Run']}"
        )
    lines.extend(
        [
            "",
            f"Frozen scan output unchanged: {'YES' if scan_unchanged else 'NO'}",
            f"Frozen 4-cell protocol validation unchanged: {'YES' if validation_unchanged else 'NO'}",
            f"Frozen 8-cell pilot unchanged: {'YES' if pilot_unchanged else 'NO'}",
            "",
            f"Canonical cells: {file_manifest['Cell_ID'].nunique()}",
            f"Total raw cycles: {len(cycles)}",
            f"Total primary observations: {len(primary)}",
            f"Review plots generated: {plot_count}",
            "",
            "Qref summary (min / median / max):",
            f"- first-comparable Ah: {_format_stats(summary['first_qref_stats'])}",
            f"- first-comparable minus nominal percent: {_format_stats(summary['first_minus_nominal_pct_stats'])}",
            f"- early-median5 Ah: {_format_stats(summary['median5_qref_stats'])}",
            "",
            "EOL status counts:",
        ]
    )
    for definition, counts in summary["eol_status_counts"].items():
        lines.append(
            f"- {definition}: EXACT_OBSERVED={counts[EOL_EXACT]}, "
            f"RIGHT_CENSORED={counts[EOL_RIGHT_CENSORED]}"
        )
    lines.extend(
        [
            "",
            "Nominal to first-comparable transitions:",
            json.dumps(summary["nominal_first_transitions"], sort_keys=True),
            "Nominal to early-median5 transitions:",
            json.dumps(summary["nominal_median5_transitions"], sort_keys=True),
            "First-comparable vs early-median5 exact EOL delta (min / median / max):",
            _format_stats(summary["first_median5_delta_stats"]),
            "First-comparable vs early-median5 median absolute delta:",
            str(summary["first_median5_delta_stats"].get("median_absolute")),
            "Minimum observed earlier-by lower bound (min / median / max):",
            _format_stats(summary["lower_bound_stats"]),
            "",
            f"QC CHECK count: {summary['check_count']}",
            f"INVALID_CANDIDATE count: {summary['invalid_count']}",
            f"Unresolved protocol review count: {summary['unresolved_review_count']}",
            "",
            "Regression tests:",
        ]
    )
    for _, row in tests.iterrows():
        lines.append(
            f"- {row['Test_ID']}: {row['Status']} "
            f"(expected={row['Expected']}; observed={row['Observed']})"
        )
    passed = tests["Status"].eq("PASS").all()
    lines.extend(
        [
            "",
            "ALL REGRESSION TESTS PASSED" if passed else "HUST 77-CELL FULL RUN FAILED",
            "",
            "HUST 77-CELL FINAL V1 COMPLETE" if passed else "HUST 77-CELL FULL RUN FAILED",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_delivered_csvs(
    temp_dir: Path,
    expected_cycle_count: int,
    expected_cell_count: int,
    expected_primary_count: int,
    expected_plot_count: int,
) -> None:
    manifest = pd.read_csv(temp_dir / "hust_77_file_manifest.csv")
    if len(manifest) != expected_cell_count or manifest["Cell_ID"].isna().any():
        raise RuntimeError("OUTPUT_REREAD_MANIFEST_FAILURE")
    if manifest.duplicated(["Cell_ID"]).any():
        raise RuntimeError("OUTPUT_REREAD_MANIFEST_DUPLICATE_CELL")

    cycle = pd.read_csv(
        temp_dir / "hust_77_cycle_qc.csv",
        usecols=["Cell_ID", "Raw_Cycle_Index", "Auto_QC_Flag", "Primary_Capacity_Eligible"],
    )
    if len(cycle) != expected_cycle_count or cycle["Cell_ID"].isna().any():
        raise RuntimeError("OUTPUT_REREAD_CYCLE_COUNT_OR_ID_FAILURE")
    if cycle.duplicated(["Cell_ID", "Raw_Cycle_Index"]).any():
        raise RuntimeError("OUTPUT_REREAD_CYCLE_KEY_DUPLICATE")
    if (
        cycle["Auto_QC_Flag"].eq("INVALID_CANDIDATE")
        & cycle["Primary_Capacity_Eligible"].eq("YES")
    ).any():
        raise RuntimeError("OUTPUT_REREAD_INVALID_IN_PRIMARY")

    primary = pd.read_csv(
        temp_dir / "hust_77_primary_capacity.csv",
        usecols=["Cell_ID", "Raw_Cycle_Index", "Primary_Capacity_Ah"],
    )
    if len(primary) != expected_primary_count:
        raise RuntimeError("OUTPUT_REREAD_PRIMARY_COUNT_FAILURE")
    if primary.duplicated(["Cell_ID", "Raw_Cycle_Index"]).any():
        raise RuntimeError("OUTPUT_REREAD_PRIMARY_KEY_DUPLICATE")

    qref = pd.read_csv(temp_dir / "hust_77_qref_summary.csv")
    if len(qref) != expected_cell_count or qref.duplicated(["Cell_ID"]).any():
        raise RuntimeError("OUTPUT_REREAD_QREF_FAILURE")
    if not qref["Qref_Nominal_Ah"].eq(1.1).all():
        raise RuntimeError("OUTPUT_REREAD_QREF_NOMINAL_FAILURE")

    soh = pd.read_csv(
        temp_dir / "hust_77_soh_labels_long.csv",
        usecols=["Cell_ID", "Raw_Cycle_Index", "Qref_Definition", "SOH_Fraction"],
    )
    if soh.duplicated(["Cell_ID", "Raw_Cycle_Index", "Qref_Definition"]).any():
        raise RuntimeError("OUTPUT_REREAD_SOH_KEY_DUPLICATE")
    if float(soh["SOH_Fraction"].max()) <= 1.0:
        raise RuntimeError("OUTPUT_REREAD_SOH_CLIPPING_SUSPECTED")

    eol = pd.read_csv(temp_dir / "hust_77_eol_sensitivity.csv")
    if eol.duplicated(["Cell_ID", "Qref_Definition"]).any():
        raise RuntimeError("OUTPUT_REREAD_EOL_KEY_DUPLICATE")
    rc_eol = eol[eol["EOL_Status"].eq(EOL_RIGHT_CENSORED)]
    if rc_eol["EOL_Raw_Cycle"].notna().any() or rc_eol[
        "EOL_Capacity_Ah"
    ].notna().any():
        raise RuntimeError("OUTPUT_REREAD_CENSORED_EOL_HAS_POINT")
    nominal = eol[eol["Qref_Definition"].eq("nominal")].sort_values("Cell_ID")
    fixed = eol[eol["Qref_Definition"].eq("fixed_absolute_0p88Ah")].sort_values(
        "Cell_ID"
    )
    if not (
        nominal["EOL_Status"].reset_index(drop=True).eq(
            fixed["EOL_Status"].reset_index(drop=True)
        ).all()
        and _series_equal_exact(
            nominal["EOL_Raw_Cycle"], fixed["EOL_Raw_Cycle"]
        )
    ):
        raise RuntimeError("OUTPUT_REREAD_NOMINAL_FIXED_EOL_MISMATCH")

    pairwise = pd.read_csv(temp_dir / "hust_77_eol_pairwise.csv")
    if pairwise.duplicated(["Cell_ID", "Definition_A", "Definition_B"]).any():
        raise RuntimeError("OUTPUT_REREAD_PAIRWISE_KEY_DUPLICATE")

    rul = pd.read_csv(
        temp_dir / "hust_77_rul_labels.csv",
        usecols=[
            "Cell_ID",
            "Raw_Cycle_Index",
            "Qref_Definition",
            "EOL_Status",
            "EOL_Raw_Cycle",
            "RUL_Cycles",
            "Native_RUL_Used",
        ],
    )
    if rul.duplicated(["Cell_ID", "Raw_Cycle_Index", "Qref_Definition"]).any():
        raise RuntimeError("OUTPUT_REREAD_RUL_KEY_DUPLICATE")
    rc_rul = rul[rul["EOL_Status"].eq(EOL_RIGHT_CENSORED)]
    if rc_rul["RUL_Cycles"].notna().any():
        raise RuntimeError("OUTPUT_REREAD_CENSORED_RUL_HAS_POINT")
    exact_rul = rul[rul["EOL_Status"].eq(EOL_EXACT)]
    if not exact_rul["RUL_Cycles"].eq(
        exact_rul["EOL_Raw_Cycle"] - exact_rul["Raw_Cycle_Index"]
    ).all():
        raise RuntimeError("OUTPUT_REREAD_EXACT_RUL_FORMULA_FAILURE")
    if rul["Native_RUL_Used"].ne("NO").any():
        raise RuntimeError("OUTPUT_REREAD_NATIVE_RUL_CONTAMINATION")

    tests = pd.read_csv(temp_dir / "hust_77_run_regression_tests.csv")
    if tests.duplicated(["Test_ID"]).any() or tests["Status"].ne("PASS").any():
        raise RuntimeError("OUTPUT_REREAD_REGRESSION_FAILURE")
    plot_count = len(list((temp_dir / "figures").glob("*.png")))
    if plot_count != expected_plot_count:
        raise RuntimeError(
            f"OUTPUT_REREAD_PLOT_COUNT_FAILURE:{plot_count}!={expected_plot_count}"
        )


def run_full_77(
    workspace: Path,
    input_dir: Path,
    scan_dir: Path,
    protocol_validation_dir: Path,
    pilot_dir: Path,
    output_dir: Path,
) -> FullRunResult:
    workspace = workspace.resolve()
    input_dir = input_dir.resolve()
    scan_dir = scan_dir.resolve()
    protocol_validation_dir = protocol_validation_dir.resolve()
    pilot_dir = pilot_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    for required_dir in [input_dir, scan_dir, protocol_validation_dir, pilot_dir]:
        if not required_dir.is_dir():
            raise FileNotFoundError(required_dir)
    required_files = [
        scan_dir / "hust_cycle_inventory.csv",
        scan_dir / "hust_cell_inventory.csv",
        scan_dir / "hust_file_manifest.csv",
        scan_dir / "hust_protocol_fingerprint.csv",
        protocol_validation_dir / "hust_paper_vs_raw_protocol_audit.csv",
        pilot_dir / "hust_8cell_qref_summary.csv",
        pilot_dir / "hust_8cell_eol_sensitivity.csv",
        pilot_dir / "hust_8cell_regression_tests.csv",
    ]
    for required_file in required_files:
        if not required_file.is_file():
            raise FileNotFoundError(required_file)

    config = make_77cell_full_config(
        workspace,
        input_dir,
        scan_dir,
        protocol_validation_dir,
        pilot_dir,
        output_dir,
    )
    canonical = _canonical_manifest(scan_dir)
    expected_cells = int(config["expected_canonical_cell_count"])
    if len(canonical) != expected_cells or canonical["Cell_ID"].nunique() != expected_cells:
        raise RuntimeError("HUST_77CELL_FULL_RUN_FAILED:CANONICAL_CELL_COUNT")
    ordered_cells = canonical["Cell_ID"].tolist()

    cell_inventory = pd.read_csv(scan_dir / "hust_cell_inventory.csv")
    if (
        len(cell_inventory) != expected_cells
        or cell_inventory["Cell_ID"].nunique() != expected_cells
        or int(cell_inventory["Cycle_Count"].sum())
        != int(config["expected_raw_cycle_count"])
    ):
        raise RuntimeError("HUST_77CELL_FULL_RUN_FAILED:FROZEN_SOURCE_INVENTORY")

    validation_audit = pd.read_csv(
        protocol_validation_dir / "hust_paper_vs_raw_protocol_audit.csv"
    )
    if not (
        len(validation_audit) == 4
        and validation_audit["Final_Assessment"]
        .eq("RAW_DATA_PROTOCOL_DIFFERS_FROM_PAPER")
        .all()
        and validation_audit["Paper_vs_Raw_Protocol_Discrepancy"].eq("YES").all()
        and validation_audit["Stage_Mapping_Validated"].eq("YES").all()
    ):
        raise RuntimeError("HUST_77CELL_FULL_RUN_FAILED:PROTOCOL_RESOLUTION_NOT_FROZEN")
    pilot_tests = pd.read_csv(pilot_dir / "hust_8cell_regression_tests.csv")
    if pilot_tests["Status"].ne("PASS").any():
        raise RuntimeError("HUST_77CELL_FULL_RUN_FAILED:FROZEN_PILOT_NOT_PASSED")

    scan_hashes_before = _directory_hashes(scan_dir)
    validation_hashes_before = _directory_hashes(protocol_validation_dir)
    pilot_hashes_before = _directory_hashes(pilot_dir)

    raw_paths: dict[str, Path] = {}
    raw_hashes_before: dict[str, str] = {}
    for _, row in canonical.iterrows():
        cell_id = str(row["Cell_ID"])
        raw_path = input_dir / str(row["Relative_Path"])
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        actual_hash = _sha256(raw_path)
        if actual_hash != str(row["SHA256"]):
            raise RuntimeError(f"HUST_77CELL_FULL_RUN_FAILED:RAW_HASH_MISMATCH:{cell_id}")
        raw_paths[cell_id] = raw_path
        raw_hashes_before[cell_id] = actual_hash

    fingerprint = pd.read_csv(scan_dir / "hust_protocol_fingerprint.csv")
    protocol_audit = build_protocol_discrepancy_audit(
        fingerprint, cell_inventory, ordered_cells, config
    )
    cycles = _prepare_cycle_inventory(scan_dir, ordered_cells)
    cycles = _augment_cycle_inventory(cycles, protocol_audit)
    if len(cycles) != int(config["expected_raw_cycle_count"]):
        raise RuntimeError("HUST_77CELL_FULL_RUN_FAILED:RAW_CYCLE_COUNT")
    if cycles["Capacity_Semantics_Match"].ne("YES").any():
        raise RuntimeError("HUST_77CELL_FULL_RUN_FAILED:CAPACITY_SEMANTICS")
    if cycles["Primary_Capacity_Source_Match"].ne("YES").any():
        raise RuntimeError("HUST_77CELL_FULL_RUN_FAILED:CAPACITY_SOURCE")
    if protocol_audit["Protocol_Review_Required"].eq("YES").any():
        review_cells = "|".join(
            protocol_audit.loc[
                protocol_audit["Protocol_Review_Required"].eq("YES"), "Cell_ID"
            ]
        )
        raise RuntimeError(
            "HUST_77CELL_FULL_RUN_FAILED:PROTOCOL_REVIEW_REQUIRED:" + review_cells
        )

    qref_base = compute_qref_summary(
        cycles,
        ordered_cells,
        nominal_capacity_Ah=float(config["nominal_capacity_Ah"]),
        eol_soh_threshold=float(config["eol_soh_threshold"]),
    )
    qref_base = add_qref_scientific_fields(qref_base)
    soh = build_soh_labels(cycles, qref_base, config["qref_definitions"])
    eol = build_eol_sensitivity(
        cycles,
        qref_base,
        config["eol_definitions"],
        float(config["eol_soh_threshold"]),
        float(config["fixed_absolute_eol_threshold_Ah"]),
    )
    pairwise = build_eol_pairwise(
        eol,
        config["pairwise_comparisons"],
        str(config["pairwise_delta_sign_convention"]),
    )
    pairwise = add_pairwise_lower_bound(
        pairwise,
        eol,
        str(config["minimum_observed_earlier_by_cycles_definition"]),
    )
    rul = build_rul_labels(cycles, eol)
    native_endpoint = build_native_endpoint_audit(cycles)
    stopping = build_full_stopping_rule_audit(
        cycles,
        eol,
        float(config["fixed_absolute_eol_threshold_Ah"]),
        float(config["stopping_rule_near_0p88_abs_tolerance_Ah"]),
    )
    qc = build_qc_summary(cycles, fingerprint, qref_base)
    qref = _merge_eol_into_qref_summary(qref_base, eol)
    censoring = build_censoring_summary(eol, pairwise)
    review_candidates = build_qc_review_candidates(cycles, eol)
    primary_columns = [
        "Cell_Selection_Order",
        "Cell_ID",
        "Native_Cell_ID",
        "Source_File",
        "Raw_Cycle_Index",
        "Primary_Capacity_Ah",
        "Primary_Capacity_Source",
        "Primary_Capacity_Source_Match",
        "Primary_Capacity_Eligible",
        "Auto_QC_Flag",
        "Auto_QC_Reason_Codes",
        "Paper_Final_Stage_CRate",
        "Raw_Observed_Final_Stage_CRate_Approx",
        "Paper_vs_Raw_Protocol_Discrepancy",
        "Protocol_QC_Basis",
        "Protocol_Consistency_Status",
    ]
    primary = cycles[cycles["Primary_Capacity_Eligible"].eq("YES")][
        primary_columns
    ].copy()
    cycle_qc = cycles.sort_values(
        ["Cell_Selection_Order", "Raw_Cycle_Index"], kind="stable"
    )

    raw_hashes_after = {cell: _sha256(path) for cell, path in raw_paths.items()}
    raw_files_unchanged = raw_hashes_before == raw_hashes_after
    scan_unchanged = _directory_hashes(scan_dir) == scan_hashes_before
    validation_unchanged = (
        _directory_hashes(protocol_validation_dir) == validation_hashes_before
    )
    pilot_unchanged = _directory_hashes(pilot_dir) == pilot_hashes_before
    file_manifest = build_full_file_manifest(
        canonical,
        cell_inventory,
        protocol_audit,
        raw_hashes_before,
        raw_hashes_after,
    )
    pilot_qref = pd.read_csv(pilot_dir / "hust_8cell_qref_summary.csv")
    pilot_eol = pd.read_csv(pilot_dir / "hust_8cell_eol_sensitivity.csv")
    tests = build_full_regression_tests(
        config=config,
        canonical_manifest=canonical,
        cell_inventory=cell_inventory,
        cycles=cycles,
        qref=qref,
        soh=soh,
        eol=eol,
        rul=rul,
        protocol_audit=protocol_audit,
        pilot_qref=pilot_qref,
        pilot_eol=pilot_eol,
        raw_files_unchanged=raw_files_unchanged,
        scan_unchanged=scan_unchanged,
        validation_unchanged=validation_unchanged,
        pilot_unchanged=pilot_unchanged,
    )
    if tests["Status"].ne("PASS").any():
        failed = "|".join(tests.loc[tests["Status"].ne("PASS"), "Test_ID"])
        raise RuntimeError(f"HUST_77CELL_FULL_RUN_FAILED:{failed}")

    cell_summary = build_cell_summary(
        qref, qc, native_endpoint, stopping, protocol_audit, pairwise
    )
    summary = _summary_metrics(cycles, qref, eol, pairwise, protocol_audit)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    figure_dir = temp_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    try:
        qref_lookup = qref.set_index("Cell_ID")
        plot_count = 0
        for cell_id in ordered_cells:
            _plot_full_capacity(
                figure_dir / f"capacity_{cell_id}.png",
                cell_id,
                cycles[cycles["Cell_ID"].eq(cell_id)],
                qref_lookup.loc[cell_id],
                eol[eol["Cell_ID"].eq(cell_id)],
            )
            plot_count += 1

        outputs = {
            "hust_77_file_manifest.csv": file_manifest,
            "hust_77_cycle_qc.csv": cycle_qc,
            "hust_77_primary_capacity.csv": primary,
            "hust_77_qref_summary.csv": qref,
            "hust_77_soh_labels_long.csv": soh,
            "hust_77_eol_sensitivity.csv": eol,
            "hust_77_eol_pairwise.csv": pairwise,
            "hust_77_rul_labels.csv": rul,
            "hust_77_censoring_summary.csv": censoring,
            "hust_77_cell_summary.csv": cell_summary,
            "hust_77_protocol_discrepancy_audit.csv": protocol_audit,
            "hust_77_stopping_rule_audit.csv": stopping,
            "hust_77_qc_review_candidates.csv": review_candidates,
            "hust_77_run_regression_tests.csv": tests,
        }
        for name, frame in outputs.items():
            _write_csv(frame, temp_dir / name)

        _validate_delivered_csvs(
            temp_dir,
            expected_cycle_count=int(config["expected_raw_cycle_count"]),
            expected_cell_count=expected_cells,
            expected_primary_count=len(primary),
            expected_plot_count=expected_cells,
        )
        log = _processing_log(
            config=config,
            file_manifest=file_manifest,
            cycles=cycles,
            primary=primary,
            qref=qref,
            eol=eol,
            pairwise=pairwise,
            tests=tests,
            summary=summary,
            plot_count=plot_count,
            scan_unchanged=scan_unchanged,
            validation_unchanged=validation_unchanged,
            pilot_unchanged=pilot_unchanged,
        )
        (temp_dir / "processing_log.txt").write_text(log, encoding="utf-8")
        temp_dir.replace(output_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    return FullRunResult(
        output_dir=output_dir,
        canonical_cell_count=expected_cells,
        raw_cycle_count=len(cycles),
        primary_observation_count=len(primary),
        soh_row_count=len(soh),
        rul_row_count=len(rul),
        plot_count=plot_count,
        first_qref_stats=summary["first_qref_stats"],
        first_minus_nominal_pct_stats=summary[
            "first_minus_nominal_pct_stats"
        ],
        median5_qref_stats=summary["median5_qref_stats"],
        eol_status_counts=summary["eol_status_counts"],
        nominal_first_transitions=summary["nominal_first_transitions"],
        nominal_median5_transitions=summary["nominal_median5_transitions"],
        first_median5_delta_stats=summary["first_median5_delta_stats"],
        lower_bound_stats=summary["lower_bound_stats"],
        check_count=summary["check_count"],
        invalid_count=summary["invalid_count"],
        unresolved_review_count=summary["unresolved_review_count"],
        regression_tests_passed=True,
    )


__all__ = [
    "FullRunResult",
    "add_pairwise_lower_bound",
    "build_censoring_summary",
    "build_full_regression_tests",
    "build_protocol_discrepancy_audit",
    "run_full_77",
]
