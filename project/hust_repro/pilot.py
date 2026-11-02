from __future__ import annotations

import hashlib
import json
import platform
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

from .config import make_8cell_pilot_config


EOL_EXACT = "EXACT_OBSERVED"
EOL_RIGHT_CENSORED = "RIGHT_CENSORED"


@dataclass(frozen=True)
class PilotResult:
    output_dir: Path
    pilot_cell_count: int
    eligible_cycle_count: int
    soh_row_count: int
    rul_row_count: int
    plot_count: int
    first_qref_stats: dict[str, float]
    median5_qref_stats: dict[str, float]
    eol_status_counts: dict[str, dict[str, int]]
    nominal_first_status_changes: int
    nominal_first_exact_pair_count: int
    nominal_first_shift_stats: dict[str, float | None]
    regression_tests_passed: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_hashes(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        return {}
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        float_format="%.12g",
    )


def _pipe_cycles(values: Sequence[Any]) -> str:
    return "|".join(str(int(value)) for value in values)


def _prepare_cycle_inventory(
    scan_dir: Path, selected_cells: Sequence[str]
) -> pd.DataFrame:
    columns = [
        "Cell_ID",
        "Native_Cell_ID",
        "Source_File",
        "Raw_Cycle_Index",
        "Native_Discharge_Capacity_mAh",
        "Native_RUL_Cycles",
        "Raw_Full_Discharge_Capacity_mAh",
        "Capacity_Difference_mAh",
        "Capacity_Difference_Pct",
        "Capacity_Semantics_Match",
        "Primary_Capacity_Ah",
        "Charge_Protocol_Traceable",
        "Charge_Complete_Scan_Candidate",
        "Full_Discharge_Traceable",
        "Full_Discharge_Complete_Scan_Candidate",
        "Severe_Voltage_Anomaly",
        "Protocol_Current_Deviation",
        "Protocol_Change_Candidate",
        "Capacity_Jump_Flag",
        "Auto_QC_Flag",
        "Auto_QC_Reason_Codes",
        "Primary_Capacity_Candidate",
    ]
    frame = pd.read_csv(scan_dir / "hust_cycle_inventory.csv", usecols=columns)
    frame = frame[frame["Cell_ID"].isin(selected_cells)].copy()
    numeric_columns = [
        "Raw_Cycle_Index",
        "Native_Discharge_Capacity_mAh",
        "Native_RUL_Cycles",
        "Raw_Full_Discharge_Capacity_mAh",
        "Capacity_Difference_mAh",
        "Capacity_Difference_Pct",
        "Primary_Capacity_Ah",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    raw_cycle_values = frame["Raw_Cycle_Index"].to_numpy(dtype=float)
    if not np.equal(raw_cycle_values, np.floor(raw_cycle_values)).all():
        raise RuntimeError("NONINTEGER_RAW_CYCLE_INDEX_IN_PILOT")
    frame["Raw_Cycle_Index"] = frame["Raw_Cycle_Index"].astype(int)
    order = {cell: index for index, cell in enumerate(selected_cells, start=1)}
    frame["Cell_Selection_Order"] = frame["Cell_ID"].map(order)
    frame = frame.sort_values(
        ["Cell_Selection_Order", "Raw_Cycle_Index"], kind="stable"
    ).reset_index(drop=True)

    capacity = frame["Primary_Capacity_Ah"].to_numpy(dtype=float)
    dq_capacity = (
        frame["Native_Discharge_Capacity_mAh"].to_numpy(dtype=float) / 1000.0
    )
    frame["Primary_Capacity_Source"] = "NATIVE_DQ_MAH_DIVIDED_BY_1000"
    frame["Primary_Capacity_Source_Match"] = np.where(
        np.isfinite(capacity)
        & np.isfinite(dq_capacity)
        & (np.abs(capacity - dq_capacity) <= 1e-12),
        "YES",
        "NO",
    )
    frame["Primary_Capacity_Eligible"] = np.where(
        frame["Primary_Capacity_Candidate"].eq("YES")
        & frame["Capacity_Semantics_Match"].eq("YES")
        & frame["Charge_Protocol_Traceable"].eq("YES")
        & frame["Charge_Complete_Scan_Candidate"].eq("YES")
        & frame["Full_Discharge_Traceable"].eq("YES")
        & frame["Full_Discharge_Complete_Scan_Candidate"].eq("YES")
        & frame["Severe_Voltage_Anomaly"].eq("NO")
        & frame["Auto_QC_Flag"].ne("INVALID_CANDIDATE")
        & np.isfinite(capacity)
        & (capacity > 0),
        "YES",
        "NO",
    )
    return frame


def compute_qref_summary(
    cycle_inventory: pd.DataFrame,
    selected_cells: Sequence[str],
    nominal_capacity_Ah: float = 1.1,
    eol_soh_threshold: float = 0.80,
) -> pd.DataFrame:
    """Compute numeric-order Qref definitions from eligible observations."""

    rows: list[dict[str, Any]] = []
    for cell_order, cell_id in enumerate(selected_cells, start=1):
        cell = cycle_inventory[cycle_inventory["Cell_ID"].eq(cell_id)].copy()
        if cell.empty:
            raise RuntimeError(f"MISSING_PILOT_CELL:{cell_id}")
        cell["Raw_Cycle_Index"] = pd.to_numeric(
            cell["Raw_Cycle_Index"], errors="raise"
        ).astype(int)
        eligible = cell[cell["Primary_Capacity_Eligible"].eq("YES")].sort_values(
            "Raw_Cycle_Index", kind="stable"
        )
        if len(eligible) < 5:
            raise RuntimeError(f"INSUFFICIENT_ELIGIBLE_QREF_SOURCE_CYCLES:{cell_id}")
        first = eligible.iloc[0]
        early = eligible.head(5)
        first_qref = float(first["Primary_Capacity_Ah"])
        median5_qref = float(np.median(early["Primary_Capacity_Ah"].to_numpy(float)))
        rows.append(
            {
                "Cell_Selection_Order": cell_order,
                "Cell_ID": cell_id,
                "Source_File": str(first["Source_File"]),
                "Qref_Nominal_Ah": float(nominal_capacity_Ah),
                "Qref_Nominal_Status": "AVAILABLE",
                "Qref_Nominal_Source": "DATASET_NOMINAL_CAPACITY",
                "Qref_Initial_Test_Ah": np.nan,
                "Qref_Initial_Test_Status": "UNAVAILABLE_NO_SEPARATE_STANDARDIZED_BOL_TEST",
                "Qref_Initial_Test_Source_Cycle": np.nan,
                "Qref_First_Comparable_Ah": first_qref,
                "Qref_First_Comparable_Status": "AVAILABLE",
                "Qref_First_Comparable_Source_Cycle": int(
                    first["Raw_Cycle_Index"]
                ),
                "Qref_Early_Median5_Comparable_Ah": median5_qref,
                "Qref_Early_Median5_Comparable_Status": "AVAILABLE",
                "Qref_Early_Median5_Comparable_Source_Cycles": _pipe_cycles(
                    early["Raw_Cycle_Index"].tolist()
                ),
                "First_Minus_Nominal_Ah": first_qref - nominal_capacity_Ah,
                "First_Minus_Nominal_Pct": 100.0
                * (first_qref - nominal_capacity_Ah)
                / nominal_capacity_Ah,
                "Median5_Minus_Nominal_Ah": median5_qref
                - nominal_capacity_Ah,
                "Median5_Minus_Nominal_Pct": 100.0
                * (median5_qref - nominal_capacity_Ah)
                / nominal_capacity_Ah,
                "Qref_First_Comparable_Greater_Than_Nominal": "YES"
                if first_qref > nominal_capacity_Ah
                else "NO",
                "Qref_Early_Median5_Greater_Than_Nominal": "YES"
                if median5_qref > nominal_capacity_Ah
                else "NO",
                "Threshold_Nominal_Ah": eol_soh_threshold
                * nominal_capacity_Ah,
                "Threshold_First_Comparable_Ah": eol_soh_threshold
                * first_qref,
                "Threshold_Early_Median5_Ah": eol_soh_threshold
                * median5_qref,
                "Threshold_Fixed_Absolute_0p88Ah": 0.880000,
                "Paper_Final_Stage_CRate": 1.0,
                "Raw_Observed_Final_Stage_CRate_Approx": 2.0,
                "Paper_vs_Raw_Protocol_Discrepancy": "YES",
                "Protocol_QC_Basis": "RAW_OBSERVED_PROTOCOL",
            }
        )
    return pd.DataFrame(rows)


def _qref_map(row: pd.Series) -> dict[str, float]:
    return {
        "nominal": float(row["Qref_Nominal_Ah"]),
        "first_comparable": float(row["Qref_First_Comparable_Ah"]),
        "early_median5_comparable": float(
            row["Qref_Early_Median5_Comparable_Ah"]
        ),
    }


def build_soh_labels(
    cycle_inventory: pd.DataFrame,
    qref_summary: pd.DataFrame,
    qref_definitions: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    qref_lookup = qref_summary.set_index("Cell_ID")
    definition_order = {
        definition: index for index, definition in enumerate(qref_definitions, start=1)
    }
    for _, cycle in cycle_inventory[
        cycle_inventory["Primary_Capacity_Eligible"].eq("YES")
    ].iterrows():
        qrefs = _qref_map(qref_lookup.loc[cycle["Cell_ID"]])
        capacity = float(cycle["Primary_Capacity_Ah"])
        for definition in qref_definitions:
            qref = qrefs[definition]
            fraction = capacity / qref
            rows.append(
                {
                    "Cell_Selection_Order": int(cycle["Cell_Selection_Order"]),
                    "Cell_ID": cycle["Cell_ID"],
                    "Source_File": cycle["Source_File"],
                    "Raw_Cycle_Index": int(cycle["Raw_Cycle_Index"]),
                    "Capacity_Ah": capacity,
                    "Auto_QC_Flag": cycle["Auto_QC_Flag"],
                    "Auto_QC_Reason_Codes": cycle["Auto_QC_Reason_Codes"],
                    "Primary_Capacity_Eligible": "YES",
                    "Qref_Definition_Order": definition_order[definition],
                    "Qref_Definition": definition,
                    "Qref_Status": "AVAILABLE",
                    "Qref_Ah": qref,
                    "SOH_Fraction": fraction,
                    "SOH_Percent": 100.0 * fraction,
                    "SOH_Clipped": "NO",
                    "Paper_vs_Raw_Protocol_Discrepancy": "YES",
                    "Protocol_QC_Basis": "RAW_OBSERVED_PROTOCOL",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["Cell_Selection_Order", "Raw_Cycle_Index", "Qref_Definition_Order"],
        kind="stable",
    )


def detect_eol(
    eligible: pd.DataFrame, threshold_Ah: float
) -> dict[str, Any]:
    """Find the first unrounded capacity <= threshold in numeric cycle order."""

    ordered = eligible.copy()
    ordered["Raw_Cycle_Index"] = pd.to_numeric(
        ordered["Raw_Cycle_Index"], errors="raise"
    ).astype(int)
    ordered["Primary_Capacity_Ah"] = pd.to_numeric(
        ordered["Primary_Capacity_Ah"], errors="raise"
    )
    ordered = ordered.sort_values("Raw_Cycle_Index", kind="stable")
    if ordered.empty:
        raise ValueError("EOL detection requires at least one eligible observation")
    crossed = ordered[ordered["Primary_Capacity_Ah"] <= float(threshold_Ah)]
    last = ordered.iloc[-1]
    if crossed.empty:
        return {
            "EOL_Status": EOL_RIGHT_CENSORED,
            "EOL_Raw_Cycle": np.nan,
            "EOL_Capacity_Ah": np.nan,
            "Previous_Valid_Raw_Cycle": np.nan,
            "Previous_Valid_Capacity_Ah": np.nan,
            "Last_Valid_Raw_Cycle": int(last["Raw_Cycle_Index"]),
            "Last_Valid_Capacity_Ah": float(last["Primary_Capacity_Ah"]),
            "Censoring_Reason": "THRESHOLD_NOT_CROSSED_BY_LAST_VALID_OBSERVATION",
        }
    first = crossed.iloc[0]
    prior = ordered[ordered["Raw_Cycle_Index"] < int(first["Raw_Cycle_Index"])]
    previous = prior.iloc[-1] if not prior.empty else None
    return {
        "EOL_Status": EOL_EXACT,
        "EOL_Raw_Cycle": int(first["Raw_Cycle_Index"]),
        "EOL_Capacity_Ah": float(first["Primary_Capacity_Ah"]),
        "Previous_Valid_Raw_Cycle": int(previous["Raw_Cycle_Index"])
        if previous is not None
        else np.nan,
        "Previous_Valid_Capacity_Ah": float(previous["Primary_Capacity_Ah"])
        if previous is not None
        else np.nan,
        "Last_Valid_Raw_Cycle": int(last["Raw_Cycle_Index"]),
        "Last_Valid_Capacity_Ah": float(last["Primary_Capacity_Ah"]),
        "Censoring_Reason": "",
    }


def build_eol_sensitivity(
    cycle_inventory: pd.DataFrame,
    qref_summary: pd.DataFrame,
    eol_definitions: Sequence[str],
    eol_soh_threshold: float,
    fixed_threshold_Ah: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    qref_lookup = qref_summary.set_index("Cell_ID")
    definition_order = {
        definition: index for index, definition in enumerate(eol_definitions, start=1)
    }
    for cell_id, group in cycle_inventory.groupby("Cell_ID", sort=False):
        eligible = group[group["Primary_Capacity_Eligible"].eq("YES")]
        qrefs = _qref_map(qref_lookup.loc[cell_id])
        for definition in eol_definitions:
            if definition == "fixed_absolute_0p88Ah":
                qref_Ah = np.nan
                threshold = float(fixed_threshold_Ah)
                definition_type = "FIXED_ABSOLUTE_CONTROL"
            else:
                qref_Ah = qrefs[definition]
                threshold = float(eol_soh_threshold) * qref_Ah
                definition_type = "QREF_DERIVED"
            result = detect_eol(eligible, threshold)
            rows.append(
                {
                    "Cell_Selection_Order": int(group["Cell_Selection_Order"].iloc[0]),
                    "Cell_ID": cell_id,
                    "Definition_Order": definition_order[definition],
                    "Qref_Definition": definition,
                    "Definition_Type": definition_type,
                    "Qref_Ah": qref_Ah,
                    "EOThreshold_Ah": threshold,
                    "Threshold_Comparison_Rule": "UNROUNDED_CAPACITY_AH_LE_THRESHOLD_AH",
                    "EOL_Status": result["EOL_Status"],
                    "EOL_Observed": "YES"
                    if result["EOL_Status"] == EOL_EXACT
                    else "NO",
                    "EOL_Raw_Cycle": result["EOL_Raw_Cycle"],
                    "EOL_Capacity_Ah": result["EOL_Capacity_Ah"],
                    "Previous_Valid_Raw_Cycle": result[
                        "Previous_Valid_Raw_Cycle"
                    ],
                    "Previous_Valid_Capacity_Ah": result[
                        "Previous_Valid_Capacity_Ah"
                    ],
                    "Last_Valid_Raw_Cycle": result["Last_Valid_Raw_Cycle"],
                    "Last_Valid_Capacity_Ah": result["Last_Valid_Capacity_Ah"],
                    "Censoring_Reason": result["Censoring_Reason"],
                    "Primary_Observation_Count": int(len(eligible)),
                    "Paper_vs_Raw_Protocol_Discrepancy": "YES",
                    "Protocol_QC_Basis": "RAW_OBSERVED_PROTOCOL",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["Cell_Selection_Order", "Definition_Order"], kind="stable"
    )


def build_eol_pairwise(
    eol_sensitivity: pd.DataFrame,
    comparisons: Sequence[Sequence[str]],
    sign_convention: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell_id, group in eol_sensitivity.groupby("Cell_ID", sort=False):
        lookup = group.set_index("Qref_Definition")
        for pair_order, (definition_a, definition_b) in enumerate(
            comparisons, start=1
        ):
            a = lookup.loc[definition_a]
            b = lookup.loc[definition_b]
            status_a = str(a["EOL_Status"])
            status_b = str(b["EOL_Status"])
            if status_a == EOL_EXACT and status_b == EOL_EXACT:
                delta = int(b["EOL_Raw_Cycle"]) - int(a["EOL_Raw_Cycle"])
                transition = "BOTH_EXACT_OBSERVED"
            else:
                delta = np.nan
                if status_a == EOL_RIGHT_CENSORED and status_b == EOL_EXACT:
                    transition = "RIGHT_CENSORED_TO_EXACT_OBSERVED"
                elif status_a == EOL_EXACT and status_b == EOL_RIGHT_CENSORED:
                    transition = "EXACT_OBSERVED_TO_RIGHT_CENSORED"
                elif (
                    status_a == EOL_RIGHT_CENSORED
                    and status_b == EOL_RIGHT_CENSORED
                ):
                    transition = "BOTH_RIGHT_CENSORED"
                else:
                    transition = "UNRESOLVED_STATUS_COMBINATION"
            rows.append(
                {
                    "Cell_Selection_Order": int(a["Cell_Selection_Order"]),
                    "Cell_ID": cell_id,
                    "Pair_Order": pair_order,
                    "Definition_A": definition_a,
                    "Definition_B": definition_b,
                    "Qref_A_Ah": a["Qref_Ah"],
                    "Qref_B_Ah": b["Qref_Ah"],
                    "EOThreshold_A_Ah": a["EOThreshold_Ah"],
                    "EOThreshold_B_Ah": b["EOThreshold_Ah"],
                    "EOL_Status_A": status_a,
                    "EOL_Status_B": status_b,
                    "EOL_A": a["EOL_Raw_Cycle"],
                    "EOL_B": b["EOL_Raw_Cycle"],
                    "Delta_EOL_Cycles": delta,
                    "Delta_EOL_Sign_Convention": sign_convention,
                    "Censoring_Status_Change": transition,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["Cell_Selection_Order", "Pair_Order"], kind="stable"
    )


def build_rul_labels(
    cycle_inventory: pd.DataFrame, eol_sensitivity: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, eol in eol_sensitivity.iterrows():
        cell_id = str(eol["Cell_ID"])
        eligible = cycle_inventory[
            cycle_inventory["Cell_ID"].eq(cell_id)
            & cycle_inventory["Primary_Capacity_Eligible"].eq("YES")
        ].sort_values("Raw_Cycle_Index", kind="stable")
        if eol["EOL_Status"] == EOL_EXACT:
            eol_cycle = int(eol["EOL_Raw_Cycle"])
            label_cycles = eligible[eligible["Raw_Cycle_Index"] <= eol_cycle]
        else:
            eol_cycle = None
            label_cycles = eligible
        for _, cycle in label_cycles.iterrows():
            raw_cycle = int(cycle["Raw_Cycle_Index"])
            if eol_cycle is None:
                rul = np.nan
                rul_status = "RIGHT_CENSORED_EOL"
                point_available = "NO"
            else:
                rul = eol_cycle - raw_cycle
                rul_status = "EXACT_OBSERVED_EOL"
                point_available = "YES"
            rows.append(
                {
                    "Cell_Selection_Order": int(cycle["Cell_Selection_Order"]),
                    "Cell_ID": cell_id,
                    "Source_File": cycle["Source_File"],
                    "Raw_Cycle_Index": raw_cycle,
                    "Capacity_Ah": float(cycle["Primary_Capacity_Ah"]),
                    "Primary_Capacity_Eligible": "YES",
                    "Qref_Definition": eol["Qref_Definition"],
                    "Definition_Order": int(eol["Definition_Order"]),
                    "Definition_Type": eol["Definition_Type"],
                    "Qref_Ah": eol["Qref_Ah"],
                    "EOThreshold_Ah": eol["EOThreshold_Ah"],
                    "EOL_Status": eol["EOL_Status"],
                    "EOL_Raw_Cycle": eol["EOL_Raw_Cycle"],
                    "RUL_Cycles": rul,
                    "RUL_Status": rul_status,
                    "RUL_Point_Label_Available": point_available,
                    "RUL_Source": "STUDY_QREF_OR_FIXED_THRESHOLD_DERIVED",
                    "Native_RUL_Used": "NO",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["Cell_Selection_Order", "Raw_Cycle_Index", "Definition_Order"],
        kind="stable",
    )


def build_native_endpoint_audit(
    cycle_inventory: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell_id, group in cycle_inventory.groupby("Cell_ID", sort=False):
        ordered = group.sort_values("Raw_Cycle_Index", kind="stable")
        eligible = ordered[ordered["Primary_Capacity_Eligible"].eq("YES")]
        last_raw = ordered.iloc[-1]
        last_valid = eligible.iloc[-1]
        native_rul = pd.to_numeric(ordered["Native_RUL_Cycles"], errors="coerce")
        zero_cycles = ordered.loc[native_rul.eq(0), "Raw_Cycle_Index"].astype(int)
        reaches_zero_at_final = bool(
            np.isfinite(last_raw["Native_RUL_Cycles"])
            and float(last_raw["Native_RUL_Cycles"]) == 0
        )
        status = (
            "NATIVE_DATASET_FAILURE_ENDPOINT"
            if reaches_zero_at_final
            else "NATIVE_DATASET_ENDPOINT_RECORDED__FINAL_NATIVE_RUL_NONZERO"
        )
        rows.append(
            {
                "Cell_Selection_Order": int(ordered["Cell_Selection_Order"].iloc[0]),
                "Cell_ID": cell_id,
                "Source_File": last_raw["Source_File"],
                "Native_Dataset_Endpoint_Cycle": int(last_raw["Raw_Cycle_Index"]),
                "Native_Dataset_Endpoint_Status": status,
                "Native_RUL_Min": float(native_rul.min()),
                "Native_RUL_Max": float(native_rul.max()),
                "Native_RUL_At_Final_Cycle": float(last_raw["Native_RUL_Cycles"]),
                "Native_RUL_Zero_Cycle_Count": int(len(zero_cycles)),
                "Native_RUL_Zero_Cycles": _pipe_cycles(zero_cycles.tolist()),
                "Native_RUL_Reaches_Zero_At_Final": "YES"
                if reaches_zero_at_final
                else "NO",
                "Last_Valid_Raw_Cycle": int(last_valid["Raw_Cycle_Index"]),
                "Last_Valid_Capacity_Ah": float(last_valid["Primary_Capacity_Ah"]),
                "Native_Endpoint_Used_As_Strict_EOL": "NO",
                "Native_RUL_Used_As_Qref_Derived_RUL": "NO",
            }
        )
    return pd.DataFrame(rows)


def build_stopping_rule_audit(
    cycle_inventory: pd.DataFrame,
    eol_sensitivity: pd.DataFrame,
    fixed_threshold_Ah: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell_id, group in cycle_inventory.groupby("Cell_ID", sort=False):
        eligible = group[group["Primary_Capacity_Eligible"].eq("YES")].sort_values(
            "Raw_Cycle_Index", kind="stable"
        )
        last = eligible.iloc[-1]
        eol = eol_sensitivity[eol_sensitivity["Cell_ID"].eq(cell_id)].set_index(
            "Qref_Definition"
        )
        nominal = eol.loc["nominal"]
        fixed = eol.loc["fixed_absolute_0p88Ah"]
        rows.append(
            {
                "Cell_Selection_Order": int(group["Cell_Selection_Order"].iloc[0]),
                "Cell_ID": cell_id,
                "Last_Valid_Raw_Cycle": int(last["Raw_Cycle_Index"]),
                "Last_Valid_Capacity_Ah": float(last["Primary_Capacity_Ah"]),
                "Last_Valid_Capacity_Minus_0p880000_Ah": float(
                    last["Primary_Capacity_Ah"]
                )
                - fixed_threshold_Ah,
                "Strict_Nominal_Threshold_Ah": 0.880000,
                "Strict_Nominal_Threshold_Crossed": "YES"
                if nominal["EOL_Status"] == EOL_EXACT
                else "NO",
                "Strict_Nominal_EOL_Status": nominal["EOL_Status"],
                "Strict_Nominal_EOL_Raw_Cycle": nominal["EOL_Raw_Cycle"],
                "Strict_Fixed_0p88_Threshold_Ah": fixed_threshold_Ah,
                "Strict_Fixed_0p88_Threshold_Crossed": "YES"
                if fixed["EOL_Status"] == EOL_EXACT
                else "NO",
                "Strict_Fixed_0p88_EOL_Status": fixed["EOL_Status"],
                "Strict_Fixed_0p88_EOL_Raw_Cycle": fixed["EOL_Raw_Cycle"],
                "Threshold_Comparison_Uses_Rounding": "NO",
                "Threshold_Comparison_Tolerance_Ah": 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_qc_summary(
    cycle_inventory: pd.DataFrame,
    fingerprint: pd.DataFrame,
    qref_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fp_lookup = fingerprint.set_index("Cell_ID")
    qref_lookup = qref_summary.set_index("Cell_ID")
    for cell_id, group in cycle_inventory.groupby("Cell_ID", sort=False):
        fp = fp_lookup.loc[cell_id]
        qref = qref_lookup.loc[cell_id]
        invalid_count = int(group["Auto_QC_Flag"].eq("INVALID_CANDIDATE").sum())
        issue_codes: list[str] = []
        if invalid_count:
            issue_codes.append("INVALID_CANDIDATE_PRESENT")
        if group["Capacity_Semantics_Match"].ne("YES").any():
            issue_codes.append("CAPACITY_SEMANTICS_MISMATCH")
        if group["Primary_Capacity_Source_Match"].ne("YES").any():
            issue_codes.append("PRIMARY_CAPACITY_SOURCE_MISMATCH")
        rows.append(
            {
                "Cell_Selection_Order": int(group["Cell_Selection_Order"].iloc[0]),
                "Cell_ID": cell_id,
                "Source_File": group["Source_File"].iloc[0],
                "Total_Raw_Cycles": int(len(group)),
                "PASS_Count": int(group["Auto_QC_Flag"].eq("PASS").sum()),
                "CHECK_Count": int(group["Auto_QC_Flag"].eq("CHECK").sum()),
                "INVALID_CANDIDATE_Count": invalid_count,
                "Primary_Capacity_Eligible_Count": int(
                    group["Primary_Capacity_Eligible"].eq("YES").sum()
                ),
                "Primary_Capacity_Ineligible_Count": int(
                    group["Primary_Capacity_Eligible"].eq("NO").sum()
                ),
                "Capacity_Semantics_Match_Count": int(
                    group["Capacity_Semantics_Match"].eq("YES").sum()
                ),
                "Incomplete_Terminal_Discharge_Count": int(
                    group["Full_Discharge_Complete_Scan_Candidate"].ne("YES").sum()
                ),
                "Severe_Voltage_Anomaly_Count": int(
                    group["Severe_Voltage_Anomaly"].eq("YES").sum()
                ),
                "Capacity_Jump_CHECK_Count": int(
                    group["Capacity_Jump_Flag"].eq("YES").sum()
                ),
                "Primary_Capacity_Source": "NATIVE_DQ_MAH_DIVIDED_BY_1000",
                "Protocol_Fingerprint_ID": fp["Protocol_Fingerprint_ID"],
                "Raw_Observed_Final_Stage_Current_A": fp[
                    "Discharge_Step4_Median_Current_A"
                ],
                "Raw_Observed_Final_Stage_CRate": fp["Estimated_C_Rate_4"],
                "Paper_Final_Stage_CRate": 1.0,
                "Paper_vs_Raw_Protocol_Discrepancy": "YES",
                "Protocol_QC_Basis": "RAW_OBSERVED_PROTOCOL",
                "Qref_First_Comparable_Source_Cycle": qref[
                    "Qref_First_Comparable_Source_Cycle"
                ],
                "Qref_Early_Median5_Source_Cycles": qref[
                    "Qref_Early_Median5_Comparable_Source_Cycles"
                ],
                "Pilot_QC_Status": "PASS" if not issue_codes else "CHECK",
                "Pilot_QC_Issue_Codes": "PASS"
                if not issue_codes
                else "|".join(issue_codes),
            }
        )
    return pd.DataFrame(rows)


def _merge_eol_into_qref_summary(
    qref_summary: pd.DataFrame, eol_sensitivity: pd.DataFrame
) -> pd.DataFrame:
    result = qref_summary.copy()
    name_map = {
        "nominal": "Nominal",
        "first_comparable": "First_Comparable",
        "early_median5_comparable": "Early_Median5",
        "fixed_absolute_0p88Ah": "Fixed_Absolute_0p88Ah",
    }
    for definition, suffix in name_map.items():
        subset = eol_sensitivity[
            eol_sensitivity["Qref_Definition"].eq(definition)
        ][["Cell_ID", "EOL_Status", "EOL_Raw_Cycle", "EOL_Capacity_Ah"]].rename(
            columns={
                "EOL_Status": f"EOL_{suffix}_Status",
                "EOL_Raw_Cycle": f"EOL_{suffix}_Raw_Cycle",
                "EOL_Capacity_Ah": f"EOL_{suffix}_Capacity_Ah",
            }
        )
        result = result.merge(subset, on="Cell_ID", how="left", validate="one_to_one")
    return result.sort_values("Cell_Selection_Order", kind="stable")


def _regression_tests(
    *,
    selected_cells: Sequence[str],
    cycle_inventory: pd.DataFrame,
    qref_summary: pd.DataFrame,
    soh: pd.DataFrame,
    eol: pd.DataFrame,
    rul: pd.DataFrame,
    qc: pd.DataFrame,
    fixed_threshold_Ah: float,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(test_id: str, expected: str, observed: str, passed: bool, details: str = "") -> None:
        rows.append(
            {
                "Test_ID": test_id,
                "Scope": "HUST_8CELL_QREF_EOL_PILOT",
                "Expected": expected,
                "Observed": observed,
                "Status": "PASS" if passed else "FAIL",
                "Details": details,
            }
        )

    observed_cells = qref_summary.sort_values("Cell_Selection_Order")["Cell_ID"].tolist()
    add(
        "PILOT_CELL_COUNT",
        "8 fixed cells with exact identity and order",
        f"count={len(observed_cells)}; cells={'|'.join(observed_cells)}",
        observed_cells == list(selected_cells),
    )
    nominal = pd.to_numeric(qref_summary["Qref_Nominal_Ah"], errors="coerce")
    add(
        "QREF_NOMINAL",
        "1.1 Ah for all 8",
        f"{int(nominal.eq(1.1).sum())}/8 exact",
        bool(len(nominal) == 8 and nominal.eq(1.1).all()),
    )
    unavailable = qref_summary["Qref_Initial_Test_Status"].eq(
        "UNAVAILABLE_NO_SEPARATE_STANDARDIZED_BOL_TEST"
    ) & qref_summary["Qref_Initial_Test_Ah"].isna()
    add(
        "INITIAL_TEST_QREF_UNAVAILABLE",
        "8/8 unavailable; capacity NA",
        f"{int(unavailable.sum())}/8",
        bool(unavailable.all() and len(unavailable) == 8),
    )

    first_source_pass = True
    median_source_pass = True
    source_details: list[str] = []
    for _, qref in qref_summary.iterrows():
        eligible = cycle_inventory[
            cycle_inventory["Cell_ID"].eq(qref["Cell_ID"])
            & cycle_inventory["Primary_Capacity_Eligible"].eq("YES")
        ].sort_values("Raw_Cycle_Index", kind="stable")
        expected_first = int(eligible.iloc[0]["Raw_Cycle_Index"])
        expected_five = _pipe_cycles(eligible.head(5)["Raw_Cycle_Index"].tolist())
        first_source_pass &= int(qref["Qref_First_Comparable_Source_Cycle"]) == expected_first
        median_source_pass &= (
            str(qref["Qref_Early_Median5_Comparable_Source_Cycles"])
            == expected_five
        )
        source_details.append(
            f"{qref['Cell_ID']}:{expected_first};{expected_five}"
        )
    add(
        "FIRST_COMPARABLE_SOURCE",
        "numeric first valid cycle",
        "all sources checked",
        first_source_pass,
        "|".join(source_details),
    )
    add(
        "EARLY_MEDIAN5_SOURCE",
        "first five numeric valid cycles",
        "all 8 use 1|2|3|4|5",
        median_source_pass
        and qref_summary["Qref_Early_Median5_Comparable_Source_Cycles"]
        .astype(str)
        .eq("1|2|3|4|5")
        .all(),
    )

    direct_soh = soh["Capacity_Ah"] / soh["Qref_Ah"]
    soh_error = (soh["SOH_Fraction"] - direct_soh).abs().max()
    above_one = int(soh["SOH_Fraction"].gt(1.0).sum())
    no_clip = bool(
        soh_error <= 1e-12
        and above_one > 0
        and soh["SOH_Clipped"].eq("NO").all()
    )
    add(
        "NO_SOH_CLIPPING",
        "direct ratio retained, including SOH > 100%",
        f"max_formula_error={soh_error:.12g}; rows_above_100pct={above_one}",
        no_clip,
    )

    exact_rul = rul[rul["EOL_Status"].eq(EOL_EXACT)]
    exact_error = (
        exact_rul["RUL_Cycles"]
        - (exact_rul["EOL_Raw_Cycle"] - exact_rul["Raw_Cycle_Index"])
    ).abs().max()
    no_native = bool(
        rul["Native_RUL_Used"].eq("NO").all()
        and rul["RUL_Source"].eq("STUDY_QREF_OR_FIXED_THRESHOLD_DERIVED").all()
        and (pd.isna(exact_error) or exact_error == 0)
    )
    add(
        "NO_NATIVE_RUL_AS_QREF_RUL",
        "native RUL never used; exact labels equal EOL cycle minus current cycle",
        f"max_exact_formula_error={exact_error if np.isfinite(exact_error) else 'NA'}",
        no_native,
    )

    censored_eol = eol[eol["EOL_Status"].eq(EOL_RIGHT_CENSORED)]
    no_censored_eol_point = bool(
        censored_eol["EOL_Raw_Cycle"].isna().all()
        and censored_eol["EOL_Capacity_Ah"].isna().all()
    )
    add(
        "RIGHT_CENSORED_EOL_HAS_NO_POINT_VALUE",
        "all right-censored EOL cycle/capacity values are NA",
        f"right_censored_rows={len(censored_eol)}",
        no_censored_eol_point,
    )
    censored_rul = rul[rul["EOL_Status"].eq(EOL_RIGHT_CENSORED)]
    add(
        "RIGHT_CENSORED_RUL_HAS_NO_POINT_LABEL",
        "all censored RUL labels are NA",
        f"censored_rul_rows={len(censored_rul)}; numeric={int(censored_rul['RUL_Cycles'].notna().sum())}",
        bool(
            not censored_rul.empty
            and censored_rul["RUL_Cycles"].isna().all()
            and censored_rul["RUL_Status"].eq("RIGHT_CENSORED_EOL").all()
        ),
    )
    fixed = eol[eol["Qref_Definition"].eq("fixed_absolute_0p88Ah")]
    add(
        "FIXED_CONTROL_THRESHOLD",
        "exactly 0.880000 Ah for all 8",
        f"min={fixed['EOThreshold_Ah'].min():.12g}; max={fixed['EOThreshold_Ah'].max():.12g}",
        bool(
            len(fixed) == 8
            and fixed["EOThreshold_Ah"].eq(float(fixed_threshold_Ah)).all()
            and float(fixed_threshold_Ah) == 0.88
        ),
    )
    discrepancy = qref_summary["Paper_vs_Raw_Protocol_Discrepancy"].eq(
        "YES"
    ) & qref_summary["Protocol_QC_Basis"].eq("RAW_OBSERVED_PROTOCOL")
    discrepancy &= qc["Paper_vs_Raw_Protocol_Discrepancy"].eq("YES").to_numpy()
    add(
        "PAPER_RAW_PROTOCOL_DISCREPANCY_RETAINED",
        "YES and RAW_OBSERVED_PROTOCOL for all 8",
        f"{int(discrepancy.sum())}/8",
        bool(discrepancy.all() and len(discrepancy) == 8),
    )

    semantics = cycle_inventory["Capacity_Semantics_Match"].eq("YES")
    semantics_diff = cycle_inventory["Capacity_Difference_mAh"].abs().max()
    add(
        "CAPACITY_SEMANTICS_REMAINS_VALID",
        "all selected cycles match; max abs difference <= 0.001 mAh",
        f"{int(semantics.sum())}/{len(semantics)}; max_abs_diff={semantics_diff:.12g} mAh",
        bool(semantics.all() and semantics_diff <= 0.001),
    )
    source_match = cycle_inventory["Primary_Capacity_Source_Match"].eq("YES")
    add(
        "PRIMARY_CAPACITY_EQUALS_NATIVE_DQ_DIV_1000",
        "all selected cycles",
        f"{int(source_match.sum())}/{len(source_match)}",
        bool(source_match.all()),
    )
    add(
        "EOL_STATUS_DENSE_OBSERVATION_ONLY",
        "EXACT_OBSERVED or RIGHT_CENSORED only",
        "|".join(sorted(eol["EOL_Status"].unique())),
        set(eol["EOL_Status"]).issubset({EOL_EXACT, EOL_RIGHT_CENSORED}),
    )
    return pd.DataFrame(rows)


def _plot_capacity(
    path: Path,
    cell_id: str,
    cycles: pd.DataFrame,
    qref: pd.Series,
    eol: pd.DataFrame,
) -> None:
    ordered = cycles.sort_values("Raw_Cycle_Index", kind="stable")
    eligible = ordered[ordered["Primary_Capacity_Eligible"].eq("YES")]
    ineligible = ordered[ordered["Primary_Capacity_Eligible"].eq("NO")]
    eol_lookup = eol.set_index("Qref_Definition")

    fig, axis = plt.subplots(figsize=(11.5, 6.6))
    axis.plot(
        eligible["Raw_Cycle_Index"],
        eligible["Primary_Capacity_Ah"],
        color="#1F2937",
        linewidth=1.0,
        label="Eligible measured capacity",
    )
    if not ineligible.empty:
        axis.scatter(
            ineligible["Raw_Cycle_Index"],
            ineligible["Primary_Capacity_Ah"],
            color="#DC2626",
            s=18,
            marker="x",
            label="Ineligible observation",
            zorder=4,
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
        axis.axhline(value, color=color, linestyle=style, linewidth=1.4, label=label)

    vertical_specs = [
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
                linewidth=1.1,
                alpha=0.85,
                label=f"{label}: cycle {int(row['EOL_Raw_Cycle'])}",
            )

    last = eligible.iloc[-1]
    axis.scatter(
        [last["Raw_Cycle_Index"]],
        [last["Primary_Capacity_Ah"]],
        color="#D97706",
        s=28,
        zorder=5,
    )
    axis.annotate(
        f"Last valid: cycle {int(last['Raw_Cycle_Index'])}, {float(last['Primary_Capacity_Ah']):.6f} Ah\n"
        "Nominal/fixed EOL: RIGHT_CENSORED",
        xy=(last["Raw_Cycle_Index"], last["Primary_Capacity_Ah"]),
        xytext=(0.58, 0.15),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#92400E"},
        fontsize=9,
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
    axis.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _summary_metrics(
    qref_summary: pd.DataFrame,
    eol: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> dict[str, Any]:
    def stats(series: pd.Series) -> dict[str, float]:
        numeric = pd.to_numeric(series, errors="raise")
        return {
            "min": float(numeric.min()),
            "median": float(numeric.median()),
            "max": float(numeric.max()),
        }

    status_counts: dict[str, dict[str, int]] = {}
    for definition, group in eol.groupby("Qref_Definition", sort=False):
        status_counts[definition] = {
            EOL_EXACT: int(group["EOL_Status"].eq(EOL_EXACT).sum()),
            EOL_RIGHT_CENSORED: int(
                group["EOL_Status"].eq(EOL_RIGHT_CENSORED).sum()
            ),
        }
    nominal_first = pairwise[
        pairwise["Definition_A"].eq("nominal")
        & pairwise["Definition_B"].eq("first_comparable")
    ]
    status_changes = int(
        nominal_first["Censoring_Status_Change"]
        .isin(
            {
                "RIGHT_CENSORED_TO_EXACT_OBSERVED",
                "EXACT_OBSERVED_TO_RIGHT_CENSORED",
            }
        )
        .sum()
    )
    exact_shifts = pd.to_numeric(
        nominal_first["Delta_EOL_Cycles"], errors="coerce"
    ).dropna()
    shift_stats: dict[str, float | None] = {
        "min": float(exact_shifts.min()) if not exact_shifts.empty else None,
        "median": float(exact_shifts.median()) if not exact_shifts.empty else None,
        "max": float(exact_shifts.max()) if not exact_shifts.empty else None,
    }
    return {
        "first_qref_stats": stats(qref_summary["Qref_First_Comparable_Ah"]),
        "median5_qref_stats": stats(
            qref_summary["Qref_Early_Median5_Comparable_Ah"]
        ),
        "eol_status_counts": status_counts,
        "nominal_first_status_changes": status_changes,
        "nominal_first_exact_pair_count": int(len(exact_shifts)),
        "nominal_first_shift_stats": shift_stats,
    }


def _processing_log(
    *,
    config: Mapping[str, Any],
    raw_hashes_before: Mapping[str, str],
    raw_hashes_after: Mapping[str, str],
    scan_unchanged: bool,
    validation_unchanged: bool,
    cycle_inventory: pd.DataFrame,
    qref_summary: pd.DataFrame,
    eol: pd.DataFrame,
    pairwise: pd.DataFrame,
    tests: pd.DataFrame,
    summary: Mapping[str, Any],
    plot_count: int,
) -> str:
    lines = [
        "HUST 8-cell Qref / EOL sensitivity pilot",
        f"Processing timestamp UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Python version: {platform.python_version()}",
        f"numpy version: {np.__version__}",
        f"pandas version: {pd.__version__}",
        f"matplotlib version: {matplotlib.__version__}",
        "",
        "Scope controls:",
        "- Exactly 8 deterministic representative cells",
        "- Capacity source: verified native dq / 1000",
        "- No full 77-cell analysis",
        "- No machine-learning model training",
        "- No smoothing, interpolation, capacity correction, or raw-file writes",
        "- Protocol QC follows raw-observed terminal stage near 2C",
        "- Paper-reported 1C versus raw-observed 2C discrepancy retained without speculation",
        "",
        "Config:",
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "Selected raw-file SHA-256 verification:",
    ]
    for name in sorted(raw_hashes_before):
        lines.append(
            f"- {name}: before={raw_hashes_before[name]} after={raw_hashes_after.get(name, '')} "
            f"unchanged={'YES' if raw_hashes_after.get(name) == raw_hashes_before[name] else 'NO'}"
        )
    lines.extend(
        [
            f"Frozen scan output unchanged: {'YES' if scan_unchanged else 'NO'}",
            f"Frozen 4-cell protocol validation output unchanged: {'YES' if validation_unchanged else 'NO'}",
            "",
            f"Pilot cells: {qref_summary['Cell_ID'].nunique()}",
            f"Raw cycles read from frozen scan: {len(cycle_inventory)}",
            f"Primary eligible observations: {int(cycle_inventory['Primary_Capacity_Eligible'].eq('YES').sum())}",
            f"Review plots generated: {plot_count}",
            "",
            "Qref summary:",
            f"- first-comparable min/median/max Ah: {summary['first_qref_stats']['min']:.12g} / {summary['first_qref_stats']['median']:.12g} / {summary['first_qref_stats']['max']:.12g}",
            f"- early-median5 min/median/max Ah: {summary['median5_qref_stats']['min']:.12g} / {summary['median5_qref_stats']['median']:.12g} / {summary['median5_qref_stats']['max']:.12g}",
            "",
            "EOL status counts:",
        ]
    )
    for definition, counts in summary["eol_status_counts"].items():
        lines.append(
            f"- {definition}: EXACT_OBSERVED={counts[EOL_EXACT]}, RIGHT_CENSORED={counts[EOL_RIGHT_CENSORED]}"
        )
    shift_stats = summary["nominal_first_shift_stats"]
    lines.extend(
        [
            "",
            "Nominal vs first-comparable:",
            f"- Censoring-status changes: {summary['nominal_first_status_changes']}",
            f"- Exact comparable EOL shifts: {summary['nominal_first_exact_pair_count']}",
            "- Exact shift min/median/max cycles: "
            + (
                "NA / NA / NA"
                if shift_stats["median"] is None
                else f"{shift_stats['min']:.12g} / {shift_stats['median']:.12g} / {shift_stats['max']:.12g}"
            ),
            "",
            "Regression tests:",
        ]
    )
    for _, row in tests.iterrows():
        lines.append(
            f"- {row['Test_ID']}: {row['Status']} (expected={row['Expected']}; observed={row['Observed']})"
        )
    lines.extend(
        [
            "",
            "ALL REGRESSION TESTS PASSED"
            if tests["Status"].eq("PASS").all()
            else "HUST 8-CELL PILOT FAILED",
            "",
            "HUST 8-CELL QREF/EOL PILOT COMPLETE —"
            if tests["Status"].eq("PASS").all()
            else "HUST 8-CELL PILOT FAILED",
            "WAITING FOR HUMAN REVIEW" if tests["Status"].eq("PASS").all() else "STOP",
            "",
            "DO NOT RUN ALL 77 CELLS",
            "",
        ]
    )
    return "\n".join(lines)


def run_8cell_pilot(
    workspace: Path,
    input_dir: Path,
    scan_dir: Path,
    protocol_validation_dir: Path,
    output_dir: Path,
) -> PilotResult:
    workspace = workspace.resolve()
    input_dir = input_dir.resolve()
    scan_dir = scan_dir.resolve()
    protocol_validation_dir = protocol_validation_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    for required in [
        input_dir,
        scan_dir,
        protocol_validation_dir,
    ]:
        if not required.is_dir():
            raise FileNotFoundError(required)
    for required in [
        scan_dir / "hust_cycle_inventory.csv",
        scan_dir / "hust_cell_inventory.csv",
        scan_dir / "hust_file_manifest.csv",
        scan_dir / "hust_protocol_fingerprint.csv",
        protocol_validation_dir / "hust_paper_vs_raw_protocol_audit.csv",
    ]:
        if not required.is_file():
            raise FileNotFoundError(required)

    config = make_8cell_pilot_config(
        workspace, input_dir, scan_dir, protocol_validation_dir, output_dir
    )
    selected_cells = list(config["selected_cells"])

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
        raise RuntimeError("FOUR_CELL_PROTOCOL_RESOLUTION_NOT_FROZEN_AS_CASE_B")

    scan_hashes_before = _directory_hashes(scan_dir)
    validation_hashes_before = _directory_hashes(protocol_validation_dir)
    manifest = pd.read_csv(scan_dir / "hust_file_manifest.csv")
    canonical = manifest[
        manifest["File_Role"].eq("CANONICAL")
        & manifest["Included_In_Analysis"].astype(str).str.upper().eq("YES")
    ]
    selected_manifest = canonical[canonical["Cell_ID"].isin(selected_cells)].copy()
    if (
        len(selected_manifest) != 8
        or set(selected_manifest["Cell_ID"]) != set(selected_cells)
    ):
        raise RuntimeError("PILOT_CELL_IDENTITY_MISMATCH_IN_MANIFEST")

    raw_paths: dict[str, Path] = {}
    raw_hashes_before: dict[str, str] = {}
    for cell_id in selected_cells:
        row = selected_manifest[selected_manifest["Cell_ID"].eq(cell_id)].iloc[0]
        raw_path = input_dir / str(row["Relative_Path"])
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        actual_hash = _sha256(raw_path)
        if actual_hash != str(row["SHA256"]):
            raise RuntimeError(f"RAW_HASH_MISMATCH:{cell_id}:{raw_path.name}")
        raw_paths[cell_id] = raw_path
        raw_hashes_before[raw_path.name] = actual_hash

    fingerprint_all = pd.read_csv(scan_dir / "hust_protocol_fingerprint.csv")
    fingerprint = fingerprint_all[
        fingerprint_all["Cell_ID"].isin(selected_cells)
    ].copy()
    fingerprint["Discharge_Step4_Median_Current_A"] = pd.to_numeric(
        fingerprint["Discharge_Step4_Median_Current_A"], errors="raise"
    )
    if (
        len(fingerprint) != 8
        or set(fingerprint["Cell_ID"]) != set(selected_cells)
        or not fingerprint["Discharge_Step4_Median_Current_A"]
        .abs()
        .between(2.15, 2.25)
        .all()
    ):
        raise RuntimeError("RAW_OBSERVED_2C_PROTOCOL_RECORD_MISSING_OR_INCONSISTENT")

    cycles = _prepare_cycle_inventory(scan_dir, selected_cells)
    if set(cycles["Cell_ID"]) != set(selected_cells):
        raise RuntimeError("PILOT_CELL_IDENTITY_MISMATCH_IN_CYCLE_INVENTORY")
    if cycles["Capacity_Semantics_Match"].ne("YES").any():
        raise RuntimeError("CAPACITY_SEMANTICS_NO_LONGER_MATCHES")
    if cycles["Primary_Capacity_Source_Match"].ne("YES").any():
        raise RuntimeError("PRIMARY_CAPACITY_SOURCE_NO_LONGER_EQUALS_NATIVE_DQ_DIV_1000")

    qref = compute_qref_summary(
        cycles,
        selected_cells,
        nominal_capacity_Ah=float(config["nominal_capacity_Ah"]),
        eol_soh_threshold=float(config["eol_soh_threshold"]),
    )
    soh = build_soh_labels(cycles, qref, config["qref_definitions"])
    eol = build_eol_sensitivity(
        cycles,
        qref,
        config["eol_definitions"],
        float(config["eol_soh_threshold"]),
        float(config["fixed_absolute_eol_threshold_Ah"]),
    )
    pairwise = build_eol_pairwise(
        eol,
        config["pairwise_comparisons"],
        str(config["pairwise_delta_sign_convention"]),
    )
    rul = build_rul_labels(cycles, eol)
    native_endpoint = build_native_endpoint_audit(cycles)
    stopping = build_stopping_rule_audit(
        cycles, eol, float(config["fixed_absolute_eol_threshold_Ah"])
    )
    qc = build_qc_summary(cycles, fingerprint, qref)
    qref = _merge_eol_into_qref_summary(qref, eol)
    tests = _regression_tests(
        selected_cells=selected_cells,
        cycle_inventory=cycles,
        qref_summary=qref,
        soh=soh,
        eol=eol,
        rul=rul,
        qc=qc,
        fixed_threshold_Ah=float(config["fixed_absolute_eol_threshold_Ah"]),
    )
    if not tests["Status"].eq("PASS").all():
        failed = "|".join(tests.loc[tests["Status"].ne("PASS"), "Test_ID"])
        raise RuntimeError(f"HUST_8CELL_PILOT_REGRESSION_FAILURE:{failed}")

    summary = _summary_metrics(qref, eol, pairwise)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    plot_dir = temp_dir / "review_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        plot_count = 0
        qref_lookup = qref.set_index("Cell_ID")
        for cell_id in selected_cells:
            _plot_capacity(
                plot_dir / f"capacity_{cell_id}.png",
                cell_id,
                cycles[cycles["Cell_ID"].eq(cell_id)],
                qref_lookup.loc[cell_id],
                eol[eol["Cell_ID"].eq(cell_id)],
            )
            plot_count += 1

        outputs = {
            "hust_8cell_qref_summary.csv": qref,
            "hust_8cell_soh_labels_long.csv": soh,
            "hust_8cell_eol_sensitivity.csv": eol,
            "hust_8cell_eol_pairwise.csv": pairwise,
            "hust_8cell_rul_labels.csv": rul,
            "hust_8cell_native_endpoint_audit.csv": native_endpoint,
            "hust_8cell_stopping_rule_audit.csv": stopping,
            "hust_8cell_qc_summary.csv": qc,
            "hust_8cell_regression_tests.csv": tests,
        }
        for name, frame in outputs.items():
            _write_csv(frame, temp_dir / name)

        # Re-read delivered CSVs before committing the atomic output directory.
        reread = {name: pd.read_csv(temp_dir / name) for name in outputs}
        if len(reread["hust_8cell_qref_summary.csv"]) != 8:
            raise RuntimeError("OUTPUT_REREAD_QREF_ROW_COUNT_FAILURE")
        if not reread["hust_8cell_regression_tests.csv"]["Status"].eq("PASS").all():
            raise RuntimeError("OUTPUT_REREAD_REGRESSION_FAILURE")
        rc_eol = reread["hust_8cell_eol_sensitivity.csv"]
        rc_eol = rc_eol[rc_eol["EOL_Status"].eq(EOL_RIGHT_CENSORED)]
        if rc_eol["EOL_Raw_Cycle"].notna().any():
            raise RuntimeError("OUTPUT_REREAD_CENSORED_EOL_HAS_POINT")
        rc_rul = reread["hust_8cell_rul_labels.csv"]
        rc_rul = rc_rul[rc_rul["EOL_Status"].eq(EOL_RIGHT_CENSORED)]
        if rc_rul["RUL_Cycles"].notna().any():
            raise RuntimeError("OUTPUT_REREAD_CENSORED_RUL_HAS_POINT")

        raw_hashes_after = {path.name: _sha256(path) for path in raw_paths.values()}
        scan_unchanged = _directory_hashes(scan_dir) == scan_hashes_before
        validation_unchanged = (
            _directory_hashes(protocol_validation_dir) == validation_hashes_before
        )
        if raw_hashes_after != raw_hashes_before:
            raise RuntimeError("RAW_FILE_HASH_CHANGED_DURING_PILOT")
        if not scan_unchanged:
            raise RuntimeError("FROZEN_SCAN_OUTPUT_CHANGED_DURING_PILOT")
        if not validation_unchanged:
            raise RuntimeError("FROZEN_PROTOCOL_VALIDATION_OUTPUT_CHANGED_DURING_PILOT")

        log = _processing_log(
            config=config,
            raw_hashes_before=raw_hashes_before,
            raw_hashes_after=raw_hashes_after,
            scan_unchanged=scan_unchanged,
            validation_unchanged=validation_unchanged,
            cycle_inventory=cycles,
            qref_summary=qref,
            eol=eol,
            pairwise=pairwise,
            tests=tests,
            summary=summary,
            plot_count=plot_count,
        )
        (temp_dir / "processing_log.txt").write_text(log, encoding="utf-8")
        temp_dir.replace(output_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    return PilotResult(
        output_dir=output_dir,
        pilot_cell_count=8,
        eligible_cycle_count=int(cycles["Primary_Capacity_Eligible"].eq("YES").sum()),
        soh_row_count=len(soh),
        rul_row_count=len(rul),
        plot_count=plot_count,
        first_qref_stats=summary["first_qref_stats"],
        median5_qref_stats=summary["median5_qref_stats"],
        eol_status_counts=summary["eol_status_counts"],
        nominal_first_status_changes=summary["nominal_first_status_changes"],
        nominal_first_exact_pair_count=summary["nominal_first_exact_pair_count"],
        nominal_first_shift_stats=summary["nominal_first_shift_stats"],
        regression_tests_passed=True,
    )


__all__ = [
    "EOL_EXACT",
    "EOL_RIGHT_CENSORED",
    "PilotResult",
    "build_eol_pairwise",
    "build_eol_sensitivity",
    "build_rul_labels",
    "build_soh_labels",
    "compute_qref_summary",
    "detect_eol",
    "run_8cell_pilot",
]
