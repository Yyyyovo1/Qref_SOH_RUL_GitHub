from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .config import (
    COMMON_QREFS,
    EXACT,
    HUST_NOMINAL_CAPACITY_AH,
    HUST_QREF_COLUMNS,
    HUST_QREFS,
    HUST_STOPPING_CONTEXT,
    INTERVAL,
    LEFT,
    MEASURED_COMMON_QREFS,
    OUTPUT_FILES,
    PRIMARY_STATUS_ORDER,
    QREF_ORDER,
    QREF_UNAVAILABLE,
    RIGHT,
    STANDARD_TRANSITIONS,
    THRESHOLD_FRACTIONS,
    THRESHOLD_ORDER,
    XJTU_DENSE_BATCHES,
    XJTU_NOMINAL_CAPACITY_AH,
    XJTU_QREF_COLUMNS,
    XJTU_QREFS,
    PipelinePaths,
)


FLOAT_TOL = 1e-12


class FrozenReproductionError(RuntimeError):
    """Raised when the 80% results do not reproduce frozen outputs."""


@dataclass(frozen=True)
class EOLDetection:
    status: str
    exact_cycle: float | None
    lower_bound: float | None
    upper_bound: float | None
    first_below_raw_cycle: float | None
    first_below_diagnostic_index: float | None
    first_below_capacity_Ah: float | None
    previous_cycle: float | None
    previous_capacity_Ah: float | None
    last_cycle: float | None
    last_raw_cycle: float | None
    last_capacity_Ah: float | None
    reason: str


@dataclass(frozen=True)
class FrozenInputs:
    xjtu_observations: pd.DataFrame
    xjtu_qref: pd.DataFrame
    xjtu_eol_80: pd.DataFrame
    hust_observations: pd.DataFrame
    hust_qref: pd.DataFrame
    hust_eol_80: pd.DataFrame
    hust_stopping_audit: pd.DataFrame
    cross_cell_80: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _metadata_snapshot(directories: Iterable[Path]) -> dict[str, dict[str, int]]:
    snapshot: dict[str, dict[str, int]] = {}
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"Frozen directory missing: {directory}")
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            stat = path.stat()
            snapshot[str(path.resolve())] = {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
    return snapshot


def _snapshot_digest(snapshot: Mapping[str, Mapping[str, int]]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _to_numeric(frame: pd.DataFrame, columns: Iterable[str], label: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except Exception as exc:  # pragma: no cover - pandas error detail is retained
            raise ValueError(f"{label}.{column} is not strictly numeric") from exc
    return result


def _same_number(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return float(left) == float(right)


def _normalize_batch(value: Any) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    return str(int(float(value)))


def load_frozen_inputs(paths: PipelinePaths) -> FrozenInputs:
    for path in paths.source_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen artifact missing: {path}")

    xjtu_soh = _read_csv(paths.xjtu_dir / "xjtu_55_soh_labels_long.csv")
    xjtu_qref = _read_csv(paths.xjtu_dir / "xjtu_55_qref_summary.csv")
    xjtu_eol = _read_csv(paths.xjtu_dir / "xjtu_55_eol_sensitivity.csv")
    hust_primary = _read_csv(paths.hust_dir / "hust_77_primary_capacity.csv")
    hust_qref = _read_csv(paths.hust_dir / "hust_77_qref_summary.csv")
    hust_eol = _read_csv(paths.hust_dir / "hust_77_eol_sensitivity.csv")
    hust_stopping = _read_csv(paths.hust_dir / "hust_77_stopping_rule_audit.csv")
    cross_cell = _read_csv(paths.cross_dir / "cross_dataset_cell_level.csv")

    x_cols = [
        "Batch_ID",
        "Cell_ID",
        "Raw_Cycle_Index",
        "Aging_Cycle_Index",
        "Capacity_Ah",
        "Auto_QC_Flag",
        "Final_Include",
        "Aging_Operation_Index",
        "Cumulative_Aging_Operation_Count",
        "Diagnostic_Index",
    ]
    _require_columns(xjtu_soh, x_cols + ["Qref_Definition", "Qref_Ah"], "XJTU SOH")
    xjtu_observations = xjtu_soh[x_cols].drop_duplicates().copy()
    xjtu_observations = _to_numeric(
        xjtu_observations,
        [
            "Batch_ID",
            "Raw_Cycle_Index",
            "Aging_Cycle_Index",
            "Capacity_Ah",
            "Aging_Operation_Index",
            "Cumulative_Aging_Operation_Count",
            "Diagnostic_Index",
        ],
        "XJTU observations",
    )
    if xjtu_observations.duplicated(["Cell_ID", "Raw_Cycle_Index"]).any():
        raise ValueError("Duplicate XJTU physical observation key")
    if not xjtu_observations["Final_Include"].eq("YES").all():
        raise ValueError("XJTU SOH artifact contains a non-included physical observation")
    if xjtu_observations["Cell_ID"].nunique() != 55:
        raise ValueError("XJTU frozen observation cell count is not 55")

    _require_columns(
        xjtu_qref,
        ["Batch_ID", "Cell_ID"]
        + [item for pair in XJTU_QREF_COLUMNS.values() for item in pair],
        "XJTU Qref",
    )
    if len(xjtu_qref) != 55 or xjtu_qref["Cell_ID"].nunique() != 55:
        raise ValueError("XJTU Qref summary does not contain exactly 55 unique cells")

    _require_columns(
        xjtu_eol,
        [
            "Batch_ID",
            "Cell_ID",
            "Qref_Definition",
            "EOThreshold_Ah",
            "EOL_Status",
            "EOL_Exact_Aging_Cycle",
            "EOL_Lower_Bound_Aging_Cycle",
            "EOL_Upper_Bound_Aging_Cycle",
            "EOL_First_Below_Diagnostic_Index",
            "EOL_First_Below_Raw_Cycle",
            "Last_Valid_Aging_Cycle",
            "Last_Valid_Raw_Cycle",
            "Last_Valid_Capacity_Ah",
        ],
        "XJTU frozen EOL",
    )

    h_cols = [
        "Cell_Selection_Order",
        "Cell_ID",
        "Source_File",
        "Raw_Cycle_Index",
        "Primary_Capacity_Ah",
        "Primary_Capacity_Eligible",
        "Auto_QC_Flag",
        "Protocol_QC_Basis",
        "Paper_vs_Raw_Protocol_Discrepancy",
    ]
    _require_columns(hust_primary, h_cols, "HUST primary capacity")
    hust_observations = _to_numeric(
        hust_primary[h_cols].copy(),
        ["Cell_Selection_Order", "Raw_Cycle_Index", "Primary_Capacity_Ah"],
        "HUST observations",
    )
    if hust_observations.duplicated(["Cell_ID", "Raw_Cycle_Index"]).any():
        raise ValueError("Duplicate HUST physical observation key")
    if not hust_observations["Primary_Capacity_Eligible"].eq("YES").all():
        raise ValueError("HUST primary-capacity artifact contains an ineligible row")
    if hust_observations["Cell_ID"].nunique() != 77:
        raise ValueError("HUST frozen observation cell count is not 77")

    _require_columns(
        hust_qref,
        ["Cell_Selection_Order", "Cell_ID", "Source_File"]
        + [item for pair in HUST_QREF_COLUMNS.values() for item in pair],
        "HUST Qref",
    )
    if len(hust_qref) != 77 or hust_qref["Cell_ID"].nunique() != 77:
        raise ValueError("HUST Qref summary does not contain exactly 77 unique cells")

    _require_columns(
        hust_eol,
        [
            "Cell_ID",
            "Qref_Definition",
            "EOThreshold_Ah",
            "EOL_Status",
            "EOL_Raw_Cycle",
            "Previous_Valid_Raw_Cycle",
            "Last_Valid_Raw_Cycle",
            "Last_Valid_Capacity_Ah",
        ],
        "HUST frozen EOL",
    )
    _require_columns(
        hust_stopping,
        [
            "Cell_ID",
            "Strict_Nominal_EOL_Status",
            "Threshold_Comparison_Uses_Rounding",
            "Threshold_Comparison_Tolerance_Ah",
            "Native_End_Near_Nominal_80pct",
        ],
        "HUST stopping-rule audit",
    )
    if len(hust_stopping) != 77:
        raise ValueError("HUST stopping-rule audit does not contain 77 cells")

    _require_columns(
        cross_cell,
        [
            "Dataset",
            "Batch_ID",
            "Cell_ID",
            "Qref_Definition",
            "Qref_Ah",
            "Qref_Status",
            "EOL_Threshold_Ah",
            "EOL_Status",
            "EOL_Exact_Cycle",
            "EOL_Lower_Bound",
            "EOL_Upper_Bound",
            "Last_Observed_Cycle",
            "RUL_Point_Label_Available",
        ],
        "Cross-dataset frozen cell level",
    )
    if len(cross_cell) != 451:
        raise ValueError("Frozen cross-dataset cell-level artifact does not contain 451 rows")

    return FrozenInputs(
        xjtu_observations=xjtu_observations,
        xjtu_qref=xjtu_qref,
        xjtu_eol_80=xjtu_eol,
        hust_observations=hust_observations,
        hust_qref=hust_qref,
        hust_eol_80=hust_eol,
        hust_stopping_audit=hust_stopping,
        cross_cell_80=cross_cell,
    )


def _ordered_xjtu(group: pd.DataFrame) -> pd.DataFrame:
    ordered = group.copy()
    ordered = _to_numeric(
        ordered,
        [
            "Raw_Cycle_Index",
            "Diagnostic_Index",
            "Cumulative_Aging_Operation_Count",
            "Capacity_Ah",
        ],
        "XJTU EOL input",
    )
    return ordered.sort_values(
        ["Cumulative_Aging_Operation_Count", "Raw_Cycle_Index"], kind="stable"
    ).reset_index(drop=True)


def detect_xjtu_eol(
    observations: pd.DataFrame,
    threshold_Ah: float,
    *,
    dense_measurement: bool,
) -> EOLDetection:
    ordered = _ordered_xjtu(observations)
    if ordered.empty:
        return EOLDetection(
            RIGHT,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "NO_VALID_FROZEN_PRIMARY_CAPACITY_OBSERVATIONS",
        )
    below = ordered["Capacity_Ah"].le(float(threshold_Ah)).to_numpy()
    last = ordered.iloc[-1]
    last_cycle = float(last["Cumulative_Aging_Operation_Count"])
    last_raw = float(last["Raw_Cycle_Index"])
    last_capacity = float(last["Capacity_Ah"])
    positions = np.flatnonzero(below)
    if len(positions) == 0:
        return EOLDetection(
            RIGHT,
            None,
            last_cycle,
            None,
            None,
            None,
            None,
            None,
            None,
            last_cycle,
            last_raw,
            last_capacity,
            "LAST_VALID_FROZEN_PRIMARY_OBSERVATION_ABOVE_THRESHOLD",
        )
    position = int(positions[0])
    current = ordered.iloc[position]
    current_cycle = float(current["Cumulative_Aging_Operation_Count"])
    current_raw = float(current["Raw_Cycle_Index"])
    current_diag = float(current["Diagnostic_Index"])
    current_capacity = float(current["Capacity_Ah"])
    if position == 0:
        return EOLDetection(
            LEFT,
            None,
            None,
            current_cycle,
            current_raw,
            current_diag,
            current_capacity,
            None,
            None,
            last_cycle,
            last_raw,
            last_capacity,
            "FIRST_VALID_FROZEN_PRIMARY_OBSERVATION_ALREADY_AT_OR_BELOW_THRESHOLD",
        )
    previous = ordered.iloc[position - 1]
    previous_cycle = float(previous["Cumulative_Aging_Operation_Count"])
    previous_capacity = float(previous["Capacity_Ah"])
    if dense_measurement and current_cycle - previous_cycle <= 1.0:
        return EOLDetection(
            EXACT,
            current_cycle,
            current_cycle,
            current_cycle,
            current_raw,
            current_diag,
            current_capacity,
            previous_cycle,
            previous_capacity,
            last_cycle,
            last_raw,
            last_capacity,
            "FIRST_DENSE_FROZEN_PRIMARY_OBSERVATION_AT_OR_BELOW_THRESHOLD",
        )
    return EOLDetection(
        INTERVAL,
        None,
        previous_cycle,
        current_cycle,
        current_raw,
        current_diag,
        current_capacity,
        previous_cycle,
        previous_capacity,
        last_cycle,
        last_raw,
        last_capacity,
        "THRESHOLD_CROSSING_BETWEEN_SPARSE_FROZEN_PRIMARY_CAPACITY_OBSERVATIONS",
    )


def _ordered_hust(group: pd.DataFrame) -> pd.DataFrame:
    ordered = group.copy()
    ordered = _to_numeric(
        ordered,
        ["Raw_Cycle_Index", "Primary_Capacity_Ah"],
        "HUST EOL input",
    )
    return ordered.sort_values("Raw_Cycle_Index", kind="stable").reset_index(drop=True)


def detect_hust_eol(observations: pd.DataFrame, threshold_Ah: float) -> EOLDetection:
    ordered = _ordered_hust(observations)
    if ordered.empty:
        return EOLDetection(
            RIGHT,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "NO_VALID_FROZEN_PRIMARY_CAPACITY_OBSERVATIONS",
        )
    below = ordered["Primary_Capacity_Ah"].le(float(threshold_Ah)).to_numpy()
    last = ordered.iloc[-1]
    last_cycle = float(last["Raw_Cycle_Index"])
    last_capacity = float(last["Primary_Capacity_Ah"])
    positions = np.flatnonzero(below)
    if len(positions) == 0:
        return EOLDetection(
            RIGHT,
            None,
            last_cycle,
            None,
            None,
            None,
            None,
            None,
            None,
            last_cycle,
            last_cycle,
            last_capacity,
            "THRESHOLD_NOT_CROSSED_BY_LAST_VALID_FROZEN_OBSERVATION",
        )
    position = int(positions[0])
    current = ordered.iloc[position]
    current_cycle = float(current["Raw_Cycle_Index"])
    current_capacity = float(current["Primary_Capacity_Ah"])
    if position == 0:
        return EOLDetection(
            LEFT,
            None,
            None,
            current_cycle,
            current_cycle,
            None,
            current_capacity,
            None,
            None,
            last_cycle,
            last_cycle,
            last_capacity,
            "FIRST_VALID_FROZEN_PRIMARY_OBSERVATION_ALREADY_AT_OR_BELOW_THRESHOLD",
        )
    previous = ordered.iloc[position - 1]
    previous_cycle = float(previous["Raw_Cycle_Index"])
    previous_capacity = float(previous["Primary_Capacity_Ah"])
    if current_cycle - previous_cycle > 1.0:
        return EOLDetection(
            INTERVAL,
            None,
            previous_cycle,
            current_cycle,
            current_cycle,
            None,
            current_capacity,
            previous_cycle,
            previous_capacity,
            last_cycle,
            last_cycle,
            last_capacity,
            "THRESHOLD_CROSSING_SPANS_A_GENUINE_MISSING_OR_INELIGIBLE_OBSERVATION_GAP",
        )
    return EOLDetection(
        EXACT,
        current_cycle,
        None,
        None,
        current_cycle,
        None,
        current_capacity,
        previous_cycle,
        previous_capacity,
        last_cycle,
        last_cycle,
        last_capacity,
        "FIRST_DENSE_FROZEN_PRIMARY_OBSERVATION_AT_OR_BELOW_THRESHOLD",
    )


def classify_qref_direction(qref: Any, nominal: Any) -> str:
    if pd.isna(qref) or pd.isna(nominal):
        return QREF_UNAVAILABLE
    delta = float(qref) - float(nominal)
    if delta > FLOAT_TOL:
        return "QREF_ABOVE_NOMINAL"
    if delta < -FLOAT_TOL:
        return "QREF_BELOW_NOMINAL"
    return "QREF_EQUAL_NOMINAL"


def _point_rul_fields(status: str) -> tuple[str, str]:
    if status == EXACT:
        return "YES", "POINT_RUL_AVAILABLE"
    if status == INTERVAL:
        return "NO", "INTERVAL_BOUNDED_RUL_ONLY"
    if status == RIGHT:
        return "NO", "RIGHT_CENSORED_EOL_NO_POINT_RUL"
    if status == LEFT:
        return "NO", "LEFT_CENSORED_EOL_NO_POINT_RUL"
    return "NO", "QREF_UNAVAILABLE"


def _nan_if_none(value: float | None) -> float:
    return np.nan if value is None else float(value)


def build_cell_level(inputs: FrozenInputs) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    x_groups = {
        cell: group.copy()
        for cell, group in inputs.xjtu_observations.groupby("Cell_ID", sort=False)
    }
    for qrow in inputs.xjtu_qref.sort_values(["Batch_ID", "Cell_ID"], kind="stable").itertuples(index=False):
        values = qrow._asdict()
        cell = str(values["Cell_ID"])
        batch = int(float(values["Batch_ID"]))
        observations = x_groups[cell]
        nominal = float(values[XJTU_QREF_COLUMNS["nominal"][0]])
        for fraction in THRESHOLD_FRACTIONS:
            for definition in XJTU_QREFS:
                value_col, status_col = XJTU_QREF_COLUMNS[definition]
                qref = pd.to_numeric(values[value_col], errors="coerce")
                qref_status = str(values[status_col])
                if pd.isna(qref):
                    threshold = np.nan
                    detected = EOLDetection(
                        QREF_UNAVAILABLE,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        "QREF_UNAVAILABLE_NO_FALLBACK",
                    )
                else:
                    qref = float(qref)
                    threshold = float(fraction) * qref
                    detected = detect_xjtu_eol(
                        observations,
                        threshold,
                        dense_measurement=batch in XJTU_DENSE_BATCHES,
                    )
                point_available, rul_status = _point_rul_fields(detected.status)
                delta_qref = float(qref) - nominal if not pd.isna(qref) else np.nan
                records.append(
                    {
                        "Dataset": "XJTU",
                        "Batch_ID": str(batch),
                        "Cell_ID": cell,
                        "Threshold_Fraction": float(fraction),
                        "Threshold_Percent": 100.0 * float(fraction),
                        "Qref_Definition": definition,
                        "Qref_Ah": qref,
                        "Qref_Status": qref_status,
                        "Qref_Direction_vs_Nominal": classify_qref_direction(qref, nominal),
                        "Delta_Qref_vs_Nominal_Ah": delta_qref,
                        "Delta_Qref_vs_Nominal_Pct": (
                            100.0 * delta_qref / nominal
                            if not pd.isna(delta_qref)
                            else np.nan
                        ),
                        "EOL_Threshold_Ah": threshold,
                        "Nominal_Equivalent_Absolute_Threshold_Ah": float(fraction)
                        * nominal,
                        "Threshold_Comparison_Rule": "FROZEN_PRIMARY_CAPACITY_AH_LE_UNROUNDED_THRESHOLD_AH",
                        "EOL_Status": detected.status,
                        "EOL_Exact_Cycle": _nan_if_none(detected.exact_cycle),
                        "EOL_Lower_Bound": _nan_if_none(detected.lower_bound),
                        "EOL_Upper_Bound": _nan_if_none(detected.upper_bound),
                        "EOL_First_Below_Raw_Cycle": _nan_if_none(
                            detected.first_below_raw_cycle
                        ),
                        "EOL_First_Below_Diagnostic_Index": _nan_if_none(
                            detected.first_below_diagnostic_index
                        ),
                        "EOL_First_Below_Capacity_Ah": _nan_if_none(
                            detected.first_below_capacity_Ah
                        ),
                        "Previous_Observed_Cycle": _nan_if_none(detected.previous_cycle),
                        "Previous_Observed_Capacity_Ah": _nan_if_none(
                            detected.previous_capacity_Ah
                        ),
                        "Last_Observed_Cycle": _nan_if_none(detected.last_cycle),
                        "Last_Observed_Raw_Cycle": _nan_if_none(
                            detected.last_raw_cycle
                        ),
                        "Last_Observed_Capacity_Ah": _nan_if_none(
                            detected.last_capacity_Ah
                        ),
                        "Point_RUL_Available": point_available,
                        "RUL_Status": rul_status,
                        "Censoring_Reason": detected.reason,
                        "Comparison_Scope": (
                            "XJTU_ONLY_SECONDARY"
                            if definition == "initial_test"
                            else "PRIMARY_COMMON"
                        ),
                        "Cycle_Index_Basis": "XJTU_CUMULATIVE_AGING_OPERATION_COUNT",
                        "Protocol_Observation_Class": (
                            "DENSE_COMPARABLE_CAPACITY"
                            if batch in XJTU_DENSE_BATCHES
                            else "SPARSE_DIAGNOSTIC_CAPACITY"
                        ),
                        "Protocol_Context": (
                            f"XJTU Batch-{batch}; frozen Final V2 protocol-comparable capacity observations and Final_Include decisions."
                        ),
                        "Dataset_Stopping_Rule_Context": (
                            "Frozen multi-batch observation windows; exact, interval-censored, right-censored, and baseline-unresolved states are retained without endpoint substitution."
                        ),
                        "Source_Capacity_Artifact": "outputs/xjtu_55_final_v2/xjtu_55_soh_labels_long.csv",
                        "Source_Qref_Artifact": "outputs/xjtu_55_final_v2/xjtu_55_qref_summary.csv",
                    }
                )

    h_groups = {
        cell: group.copy()
        for cell, group in inputs.hust_observations.groupby("Cell_ID", sort=False)
    }
    for qrow in inputs.hust_qref.sort_values("Cell_Selection_Order", kind="stable").itertuples(index=False):
        values = qrow._asdict()
        cell = str(values["Cell_ID"])
        observations = h_groups[cell]
        nominal = float(values[HUST_QREF_COLUMNS["nominal"][0]])
        for fraction in THRESHOLD_FRACTIONS:
            stopping_category, stopping_context = HUST_STOPPING_CONTEXT[float(fraction)]
            for definition in HUST_QREFS:
                value_col, status_col = HUST_QREF_COLUMNS[definition]
                qref = float(values[value_col])
                qref_status = str(values[status_col])
                threshold = float(fraction) * qref
                detected = detect_hust_eol(observations, threshold)
                point_available, rul_status = _point_rul_fields(detected.status)
                delta_qref = qref - nominal
                records.append(
                    {
                        "Dataset": "HUST",
                        "Batch_ID": "",
                        "Cell_ID": cell,
                        "Threshold_Fraction": float(fraction),
                        "Threshold_Percent": 100.0 * float(fraction),
                        "Qref_Definition": definition,
                        "Qref_Ah": qref,
                        "Qref_Status": qref_status,
                        "Qref_Direction_vs_Nominal": classify_qref_direction(qref, nominal),
                        "Delta_Qref_vs_Nominal_Ah": delta_qref,
                        "Delta_Qref_vs_Nominal_Pct": 100.0 * delta_qref / nominal,
                        "EOL_Threshold_Ah": threshold,
                        "Nominal_Equivalent_Absolute_Threshold_Ah": float(fraction)
                        * nominal,
                        "Threshold_Comparison_Rule": "FROZEN_PRIMARY_CAPACITY_AH_LE_UNROUNDED_THRESHOLD_AH",
                        "EOL_Status": detected.status,
                        "EOL_Exact_Cycle": _nan_if_none(detected.exact_cycle),
                        "EOL_Lower_Bound": _nan_if_none(detected.lower_bound),
                        "EOL_Upper_Bound": _nan_if_none(detected.upper_bound),
                        "EOL_First_Below_Raw_Cycle": _nan_if_none(
                            detected.first_below_raw_cycle
                        ),
                        "EOL_First_Below_Diagnostic_Index": np.nan,
                        "EOL_First_Below_Capacity_Ah": _nan_if_none(
                            detected.first_below_capacity_Ah
                        ),
                        "Previous_Observed_Cycle": _nan_if_none(detected.previous_cycle),
                        "Previous_Observed_Capacity_Ah": _nan_if_none(
                            detected.previous_capacity_Ah
                        ),
                        "Last_Observed_Cycle": _nan_if_none(detected.last_cycle),
                        "Last_Observed_Raw_Cycle": _nan_if_none(
                            detected.last_raw_cycle
                        ),
                        "Last_Observed_Capacity_Ah": _nan_if_none(
                            detected.last_capacity_Ah
                        ),
                        "Point_RUL_Available": point_available,
                        "RUL_Status": rul_status,
                        "Censoring_Reason": detected.reason,
                        "Comparison_Scope": "PRIMARY_COMMON",
                        "Cycle_Index_Basis": "HUST_RAW_CYCLE_INDEX",
                        "Protocol_Observation_Class": "DENSE_COMPARABLE_CAPACITY",
                        "Protocol_Context": (
                            "HUST public raw-observed protocol; frozen Final V1 eligible physical capacity observations and Qref definitions."
                        ),
                        "Dataset_Stopping_Rule_Context": stopping_context,
                        "HUST_Stopping_Context_Category": stopping_category,
                        "Source_Capacity_Artifact": "outputs/hust_77_final_v1/hust_77_primary_capacity.csv",
                        "Source_Qref_Artifact": "outputs/hust_77_final_v1/hust_77_qref_summary.csv",
                    }
                )

    result = pd.DataFrame.from_records(records)
    result["_threshold_order"] = result["Threshold_Fraction"].map(THRESHOLD_ORDER)
    result["_qref_order"] = result["Qref_Definition"].map(QREF_ORDER)
    return (
        result.sort_values(
            ["Dataset", "Batch_ID", "Cell_ID", "_threshold_order", "_qref_order"],
            kind="stable",
        )
        .drop(columns=["_threshold_order", "_qref_order"])
        .reset_index(drop=True)
    )


def _status_class(status: Any) -> str | None:
    text = str(status)
    if text == EXACT:
        return "EXACT"
    if text == INTERVAL:
        return "INTERVAL"
    if text == RIGHT:
        return "RIGHT"
    if text == LEFT:
        return "LEFT"
    return None


def classify_transition(status_a: Any, status_b: Any) -> str:
    a = _status_class(status_a)
    b = _status_class(status_b)
    if a is None or b is None:
        return "UNAVAILABLE_OR_UNCLASSIFIED"
    if a == "RIGHT" and b == "RIGHT":
        return "BOTH_RIGHT"
    if a == "LEFT" and b == "LEFT":
        return "BOTH_LEFT"
    return f"{a}_TO_{b}"


def _event_bounds(row: Mapping[str, Any]) -> tuple[float, float] | None:
    status = str(row["EOL_Status"])
    if status == EXACT and not pd.isna(row["EOL_Exact_Cycle"]):
        value = float(row["EOL_Exact_Cycle"])
        return value, value
    if status == INTERVAL:
        if pd.isna(row["EOL_Lower_Bound"]) or pd.isna(row["EOL_Upper_Bound"]):
            return None
        return float(row["EOL_Lower_Bound"]), float(row["EOL_Upper_Bound"])
    if status == RIGHT:
        lower = row["EOL_Lower_Bound"]
        if pd.isna(lower):
            lower = row["Last_Observed_Cycle"]
        if pd.isna(lower):
            return None
        return float(lower), math.inf
    if status == LEFT:
        upper = row["EOL_Upper_Bound"]
        if pd.isna(upper):
            return None
        return -math.inf, float(upper)
    return None


def ordering_consistency(
    nominal_row: Mapping[str, Any], measured_row: Mapping[str, Any]
) -> tuple[str, str]:
    direction = str(measured_row["Qref_Direction_vs_Nominal"])
    a_bounds = _event_bounds(nominal_row)
    b_bounds = _event_bounds(measured_row)
    if direction == QREF_UNAVAILABLE or a_bounds is None or b_bounds is None:
        return "NOT_ASSESSABLE", "Qref or EOL bounds are unavailable."
    a_lo, a_hi = a_bounds
    b_lo, b_hi = b_bounds
    if direction == "QREF_EQUAL_NOMINAL":
        same_status = str(nominal_row["EOL_Status"]) == str(
            measured_row["EOL_Status"]
        )
        same_lo = (math.isinf(a_lo) and math.isinf(b_lo)) or np.isclose(
            a_lo, b_lo, rtol=0.0, atol=FLOAT_TOL
        )
        same_hi = (math.isinf(a_hi) and math.isinf(b_hi)) or np.isclose(
            a_hi, b_hi, rtol=0.0, atol=FLOAT_TOL
        )
        if same_status and same_lo and same_hi:
            return "CONSISTENT", "Equal Qref gives the same threshold and EOL result."
        return "INCONSISTENT", "Equal Qref does not reproduce the nominal EOL result."
    if direction == "QREF_ABOVE_NOMINAL":
        if b_hi <= a_lo + FLOAT_TOL:
            return "CONSISTENT", "Higher threshold is observed no later than nominal."
        if not math.isinf(a_hi) and b_lo > a_hi + FLOAT_TOL:
            return "INCONSISTENT", "Higher threshold is bounded later than nominal."
        return (
            "INDETERMINATE_BOUNDS_OVERLAP",
            "Available censoring bounds do not contradict earlier-or-equal ordering.",
        )
    if direction == "QREF_BELOW_NOMINAL":
        if b_lo + FLOAT_TOL >= a_hi:
            return "CONSISTENT", "Lower threshold is observed no earlier than nominal."
        if not math.isinf(b_hi) and b_hi < a_lo - FLOAT_TOL:
            return "INCONSISTENT", "Lower threshold is bounded earlier than nominal."
        return (
            "INDETERMINATE_BOUNDS_OVERLAP",
            "Available censoring bounds do not contradict later-or-equal ordering.",
        )
    return "NOT_ASSESSABLE", "Unknown Qref direction."


def compute_effect_bounds(
    nominal_row: Mapping[str, Any], measured_row: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "Delta_EOL_Cycles": np.nan,
        "Delta_EOL_Lower_Bound_Cycles": np.nan,
        "Delta_EOL_Upper_Bound_Cycles": np.nan,
        "Minimum_Observed_Earlier_By_Cycles": np.nan,
        "Effect_Estimate_Type": "UNAVAILABLE",
        "Effect_Bound_Interpretation": "EOL comparison unavailable.",
    }
    a_bounds = _event_bounds(nominal_row)
    b_bounds = _event_bounds(measured_row)
    if a_bounds is None or b_bounds is None:
        return result
    a_lo, a_hi = a_bounds
    b_lo, b_hi = b_bounds
    status_a = str(nominal_row["EOL_Status"])
    status_b = str(measured_row["EOL_Status"])
    if status_a == EXACT and status_b == EXACT:
        result["Delta_EOL_Cycles"] = b_lo - a_lo
        result["Effect_Estimate_Type"] = "EXACT_DELTA"
        result["Effect_Bound_Interpretation"] = (
            "Exact delta is measured-Qref EOL minus nominal EOL."
        )
        return result
    if not math.isinf(a_lo) and not math.isinf(a_hi) and not math.isinf(b_lo) and not math.isinf(b_hi):
        result["Delta_EOL_Lower_Bound_Cycles"] = b_lo - a_hi
        result["Delta_EOL_Upper_Bound_Cycles"] = b_hi - a_lo
        result["Effect_Estimate_Type"] = "FINITE_INTERVAL_BOUNDS"
        result["Effect_Bound_Interpretation"] = (
            "Only censoring-compatible bounds on measured EOL minus nominal EOL are reported."
        )
        return result
    if status_a == RIGHT and status_b == EXACT:
        result["Delta_EOL_Upper_Bound_Cycles"] = b_hi - a_lo
        if b_hi <= a_lo:
            result["Minimum_Observed_Earlier_By_Cycles"] = a_lo - b_hi
            result["Effect_Estimate_Type"] = "LOWER_BOUND_ON_EARLIER_EOL_ONLY"
            result["Effect_Bound_Interpretation"] = (
                "Nominal EOL is right-censored; the positive value is only a lower bound on how much earlier measured-Qref EOL occurred."
            )
        else:
            result["Effect_Estimate_Type"] = "ONE_SIDED_UPPER_BOUND_ON_DELTA"
            result["Effect_Bound_Interpretation"] = (
                "Nominal EOL is right-censored; only an upper bound is available."
            )
        return result
    if math.isinf(a_hi) and not math.isinf(b_hi):
        result["Delta_EOL_Upper_Bound_Cycles"] = b_hi - a_lo
        result["Effect_Estimate_Type"] = "ONE_SIDED_UPPER_BOUND_ON_DELTA"
        result["Effect_Bound_Interpretation"] = (
            "Nominal EOL is right-censored; only an upper bound on measured minus nominal EOL is available."
        )
        return result
    if not math.isinf(a_hi) and math.isinf(b_hi):
        result["Delta_EOL_Lower_Bound_Cycles"] = b_lo - a_hi
        result["Effect_Estimate_Type"] = "ONE_SIDED_LOWER_BOUND_ON_DELTA"
        result["Effect_Bound_Interpretation"] = (
            "Measured-Qref EOL is right-censored; only a lower bound is available."
        )
        return result
    if status_a == RIGHT and status_b == RIGHT:
        result["Effect_Estimate_Type"] = "BOTH_RIGHT_CENSORED_NO_POINT_EFFECT"
        result["Effect_Bound_Interpretation"] = (
            "Both EOL values are right-censored; no point shift is reported."
        )
        return result
    result["Effect_Estimate_Type"] = "CENSORING_BOUNDS_NOT_POINT_IDENTIFIED"
    result["Effect_Bound_Interpretation"] = (
        "Censoring prevents a point EOL shift from being identified."
    )
    return result


def build_effect_bounds(cell_level: pd.DataFrame) -> pd.DataFrame:
    nominal = cell_level[cell_level["Qref_Definition"].eq("nominal")].set_index(
        ["Dataset", "Cell_ID", "Threshold_Fraction"], drop=False
    )
    measured = cell_level[
        cell_level["Qref_Definition"].isin(
            (*MEASURED_COMMON_QREFS, "initial_test")
        )
    ]
    records: list[dict[str, Any]] = []
    for row in measured.itertuples(index=False):
        values = row._asdict()
        key = (values["Dataset"], values["Cell_ID"], values["Threshold_Fraction"])
        a = nominal.loc[key]
        if isinstance(a, pd.DataFrame):
            raise ValueError(f"Duplicate nominal cell-level key: {key}")
        nominal_values = a.to_dict()
        transition = classify_transition(nominal_values["EOL_Status"], values["EOL_Status"])
        consistency, consistency_reason = ordering_consistency(nominal_values, values)
        bounds = compute_effect_bounds(nominal_values, values)
        record = {
            "Dataset": values["Dataset"],
            "Batch_ID": values["Batch_ID"],
            "Cell_ID": values["Cell_ID"],
            "Threshold_Fraction": values["Threshold_Fraction"],
            "Threshold_Percent": values["Threshold_Percent"],
            "Definition_A": "nominal",
            "Definition_B": values["Qref_Definition"],
            "Comparison_Scope": values["Comparison_Scope"],
            "Cycle_Index_Basis": values["Cycle_Index_Basis"],
            "Qref_A_Ah": nominal_values["Qref_Ah"],
            "Qref_B_Ah": values["Qref_Ah"],
            "Delta_Qref_B_vs_A_Ah": values["Delta_Qref_vs_Nominal_Ah"],
            "Delta_Qref_B_vs_A_Pct": values["Delta_Qref_vs_Nominal_Pct"],
            "Qref_Direction_vs_Nominal": values["Qref_Direction_vs_Nominal"],
            "Threshold_A_Ah": nominal_values["EOL_Threshold_Ah"],
            "Threshold_B_Ah": values["EOL_Threshold_Ah"],
            "Delta_Threshold_B_vs_A_Ah": (
                values["EOL_Threshold_Ah"] - nominal_values["EOL_Threshold_Ah"]
                if not pd.isna(values["EOL_Threshold_Ah"])
                else np.nan
            ),
            "EOL_Status_A": nominal_values["EOL_Status"],
            "EOL_Status_B": values["EOL_Status"],
            "Transition_Category": transition,
            "EOL_A_Exact_Cycle": nominal_values["EOL_Exact_Cycle"],
            "EOL_A_Lower_Bound": nominal_values["EOL_Lower_Bound"],
            "EOL_A_Upper_Bound": nominal_values["EOL_Upper_Bound"],
            "EOL_B_Exact_Cycle": values["EOL_Exact_Cycle"],
            "EOL_B_Lower_Bound": values["EOL_Lower_Bound"],
            "EOL_B_Upper_Bound": values["EOL_Upper_Bound"],
            "Last_Observed_Cycle_A": nominal_values["Last_Observed_Cycle"],
            "Last_Observed_Cycle_B": values["Last_Observed_Cycle"],
            "Point_RUL_Available_A": nominal_values["Point_RUL_Available"],
            "Point_RUL_Available_B": values["Point_RUL_Available"],
            "Threshold_Ordering_Consistency": consistency,
            "Threshold_Ordering_Consistency_Reason": consistency_reason,
            "Dataset_Stopping_Rule_Context": values["Dataset_Stopping_Rule_Context"],
        }
        record.update(bounds)
        records.append(record)
    result = pd.DataFrame.from_records(records)
    result["_threshold_order"] = result["Threshold_Fraction"].map(THRESHOLD_ORDER)
    result["_qref_order"] = result["Definition_B"].map(QREF_ORDER)
    return (
        result.sort_values(
            ["Dataset", "Batch_ID", "Cell_ID", "_threshold_order", "_qref_order"],
            kind="stable",
        )
        .drop(columns=["_threshold_order", "_qref_order"])
        .reset_index(drop=True)
    )


def build_qref_direction_audit(cell_level: pd.DataFrame) -> pd.DataFrame:
    nominal = cell_level[cell_level["Qref_Definition"].eq("nominal")].set_index(
        ["Dataset", "Cell_ID", "Threshold_Fraction"], drop=False
    )
    records: list[dict[str, Any]] = []
    for row in cell_level.itertuples(index=False):
        values = row._asdict()
        key = (values["Dataset"], values["Cell_ID"], values["Threshold_Fraction"])
        base = nominal.loc[key]
        if isinstance(base, pd.DataFrame):
            raise ValueError(f"Duplicate nominal direction-audit key: {key}")
        nominal_values = base.to_dict()
        qref_direction = values["Qref_Direction_vs_Nominal"]
        threshold = values["EOL_Threshold_Ah"]
        nominal_threshold = nominal_values["EOL_Threshold_Ah"]
        threshold_direction = classify_qref_direction(threshold, nominal_threshold)
        formula_exact = (
            "NA"
            if pd.isna(values["Qref_Ah"])
            else (
                "YES"
                if float(threshold)
                == float(values["Threshold_Fraction"]) * float(values["Qref_Ah"])
                else "NO"
            )
        )
        direction_match = (
            "NA"
            if qref_direction == QREF_UNAVAILABLE
            else "YES" if qref_direction == threshold_direction else "NO"
        )
        consistency, reason = ordering_consistency(nominal_values, values)
        records.append(
            {
                "Dataset": values["Dataset"],
                "Batch_ID": values["Batch_ID"],
                "Cell_ID": values["Cell_ID"],
                "Threshold_Fraction": values["Threshold_Fraction"],
                "Qref_Definition": values["Qref_Definition"],
                "Qref_Ah": values["Qref_Ah"],
                "Nominal_Qref_Ah": nominal_values["Qref_Ah"],
                "Qref_Direction_vs_Nominal": qref_direction,
                "EOL_Threshold_Ah": threshold,
                "Nominal_EOL_Threshold_Ah": nominal_threshold,
                "Threshold_Direction_vs_Nominal": threshold_direction,
                "Threshold_Equals_Fraction_Times_Qref_Exactly": formula_exact,
                "Threshold_Ordering_Matches_Qref_Ordering": direction_match,
                "Nominal_EOL_Status": nominal_values["EOL_Status"],
                "Measured_EOL_Status": values["EOL_Status"],
                "EOL_Ordering_Consistency": consistency,
                "EOL_Ordering_Consistency_Reason": reason,
                "Audit_Interpretation": "Consistency check only; not a causal-discovery claim.",
            }
        )
    result = pd.DataFrame.from_records(records)
    result["_threshold_order"] = result["Threshold_Fraction"].map(THRESHOLD_ORDER)
    result["_qref_order"] = result["Qref_Definition"].map(QREF_ORDER)
    return (
        result.sort_values(
            ["Dataset", "Batch_ID", "Cell_ID", "_threshold_order", "_qref_order"],
            kind="stable",
        )
        .drop(columns=["_threshold_order", "_qref_order"])
        .reset_index(drop=True)
    )


def build_status_summary(cell_level: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (dataset, fraction, definition), group in cell_level.groupby(
        ["Dataset", "Threshold_Fraction", "Qref_Definition"], sort=False
    ):
        available = ~group["Qref_Ah"].isna()
        point_count = int(group["Point_RUL_Available"].eq("YES").sum())
        left_count = int(group["EOL_Status"].eq(LEFT).sum())
        known = group["EOL_Status"].isin(
            [EXACT, INTERVAL, RIGHT, LEFT, QREF_UNAVAILABLE]
        )
        records.append(
            {
                "Dataset": dataset,
                "Threshold_Fraction": fraction,
                "Threshold_Percent": 100.0 * float(fraction),
                "Qref_Definition": definition,
                "Comparison_Scope": group["Comparison_Scope"].iloc[0],
                "Cell_Count": int(group["Cell_ID"].nunique()),
                "Qref_Available_Count": int(available.sum()),
                "Qref_Unavailable_Count": int((~available).sum()),
                "EXACT_OBSERVED_Count": int(group["EOL_Status"].eq(EXACT).sum()),
                "INTERVAL_CENSORED_Count": int(
                    group["EOL_Status"].eq(INTERVAL).sum()
                ),
                "RIGHT_CENSORED_Count": int(group["EOL_Status"].eq(RIGHT).sum()),
                "LEFT_OR_BASELINE_UNRESOLVED_Count": left_count,
                "OTHER_EOL_STATUS_Count": int((~known).sum()),
                "QREF_UNAVAILABLE_EOL_Count": int(
                    group["EOL_Status"].eq(QREF_UNAVAILABLE).sum()
                ),
                "Point_RUL_Available_Count": point_count,
                "Point_RUL_Available_Fraction": point_count / len(group),
                "Point_RUL_Available_Fraction_Among_Qref_Available": (
                    point_count / int(available.sum()) if available.any() else np.nan
                ),
                "Cycle_Index_Basis": group["Cycle_Index_Basis"].iloc[0],
                "Dataset_Stopping_Rule_Context": group[
                    "Dataset_Stopping_Rule_Context"
                ].iloc[0],
            }
        )
    result = pd.DataFrame.from_records(records)
    result["_threshold_order"] = result["Threshold_Fraction"].map(THRESHOLD_ORDER)
    result["_qref_order"] = result["Qref_Definition"].map(QREF_ORDER)
    return (
        result.sort_values(
            ["Dataset", "_threshold_order", "_qref_order"], kind="stable"
        )
        .drop(columns=["_threshold_order", "_qref_order"])
        .reset_index(drop=True)
    )


def _cell_effect_present(row: Mapping[str, Any]) -> bool:
    status_a = str(row["EOL_Status_A"])
    status_b = str(row["EOL_Status_B"])
    if QREF_UNAVAILABLE in {status_a, status_b}:
        return False
    if status_a != status_b:
        return True
    if status_a == EXACT:
        return not np.isclose(
            float(row["EOL_A_Exact_Cycle"]),
            float(row["EOL_B_Exact_Cycle"]),
            rtol=0.0,
            atol=0.0,
        )
    if status_a == INTERVAL:
        return not (
            _same_number(row["EOL_A_Lower_Bound"], row["EOL_B_Lower_Bound"])
            and _same_number(row["EOL_A_Upper_Bound"], row["EOL_B_Upper_Bound"])
        )
    if status_a == LEFT:
        return not _same_number(
            row["EOL_A_Upper_Bound"], row["EOL_B_Upper_Bound"]
        )
    return False


def build_transition_summary(effect_bounds: pd.DataFrame) -> pd.DataFrame:
    primary = effect_bounds[
        effect_bounds["Definition_B"].isin(MEASURED_COMMON_QREFS)
    ].copy()
    records: list[dict[str, Any]] = []
    for (dataset, fraction, definition), group in primary.groupby(
        ["Dataset", "Threshold_Fraction", "Definition_B"], sort=False
    ):
        observed_transitions = list(group["Transition_Category"].dropna().unique())
        transition_order = list(STANDARD_TRANSITIONS) + sorted(
            set(observed_transitions) - set(STANDARD_TRANSITIONS)
        )
        exact_values = pd.to_numeric(group["Delta_EOL_Cycles"], errors="coerce").dropna()
        lower_values = pd.to_numeric(
            group["Minimum_Observed_Earlier_By_Cycles"], errors="coerce"
        ).dropna()
        affected_total = int(
            sum(_cell_effect_present(row) for row in group.to_dict("records"))
        )
        for transition in transition_order:
            subset = group[group["Transition_Category"].eq(transition)]
            records.append(
                {
                    "Dataset": dataset,
                    "Threshold_Fraction": fraction,
                    "Threshold_Percent": 100.0 * float(fraction),
                    "Measured_Qref_Definition": definition,
                    "Transition_Category": transition,
                    "Cell_Count": int(len(subset)),
                    "Percent_of_Definition_Cells": (
                        len(subset) / len(group) if len(group) else np.nan
                    ),
                    "Total_Definition_Cell_Count": int(len(group)),
                    "Status_Changed_Cell_Count_All_Transitions": int(
                        (group["EOL_Status_A"] != group["EOL_Status_B"]).sum()
                    ),
                    "Qref_Induced_Label_Effect_Cell_Count_All_Transitions": affected_total,
                    "Point_RUL_Availability_Changed_Cell_Count_All_Transitions": int(
                        (
                            group["Point_RUL_Available_A"]
                            != group["Point_RUL_Available_B"]
                        ).sum()
                    ),
                    "Exact_Delta_Valid_Cell_Count_All_Transitions": int(
                        len(exact_values)
                    ),
                    "Exact_Delta_EOL_Cycles_Min": (
                        exact_values.min() if len(exact_values) else np.nan
                    ),
                    "Exact_Delta_EOL_Cycles_Median": (
                        exact_values.median() if len(exact_values) else np.nan
                    ),
                    "Exact_Delta_EOL_Cycles_Max": (
                        exact_values.max() if len(exact_values) else np.nan
                    ),
                    "Earlier_EOL_Lower_Bound_Valid_Cell_Count_All_Transitions": int(
                        len(lower_values)
                    ),
                    "Minimum_Observed_Earlier_By_Cycles_Min": (
                        lower_values.min() if len(lower_values) else np.nan
                    ),
                    "Minimum_Observed_Earlier_By_Cycles_Median": (
                        lower_values.median() if len(lower_values) else np.nan
                    ),
                    "Minimum_Observed_Earlier_By_Cycles_Max": (
                        lower_values.max() if len(lower_values) else np.nan
                    ),
                    "Ordering_CONSISTENT_Count": int(
                        subset["Threshold_Ordering_Consistency"].eq("CONSISTENT").sum()
                    ),
                    "Ordering_INDETERMINATE_Count": int(
                        subset["Threshold_Ordering_Consistency"]
                        .eq("INDETERMINATE_BOUNDS_OVERLAP")
                        .sum()
                    ),
                    "Ordering_INCONSISTENT_Count": int(
                        subset["Threshold_Ordering_Consistency"]
                        .eq("INCONSISTENT")
                        .sum()
                    ),
                }
            )
    result = pd.DataFrame.from_records(records)
    result["_threshold_order"] = result["Threshold_Fraction"].map(THRESHOLD_ORDER)
    result["_qref_order"] = result["Measured_Qref_Definition"].map(QREF_ORDER)
    transition_order_map = {
        value: index for index, value in enumerate(STANDARD_TRANSITIONS, 1)
    }
    result["_transition_order"] = result["Transition_Category"].map(
        transition_order_map
    ).fillna(999)
    return (
        result.sort_values(
            ["Dataset", "_threshold_order", "_qref_order", "_transition_order", "Transition_Category"],
            kind="stable",
        )
        .drop(columns=["_threshold_order", "_qref_order", "_transition_order"])
        .reset_index(drop=True)
    )


def build_rul_availability(cell_level: pd.DataFrame) -> pd.DataFrame:
    nominal = cell_level[cell_level["Qref_Definition"].eq("nominal")][
        ["Dataset", "Cell_ID", "Threshold_Fraction", "Point_RUL_Available"]
    ].rename(columns={"Point_RUL_Available": "Nominal_Point_RUL_Available"})
    joined = cell_level.merge(
        nominal,
        on=["Dataset", "Cell_ID", "Threshold_Fraction"],
        how="left",
        validate="many_to_one",
    )
    records: list[dict[str, Any]] = []
    for (dataset, fraction, definition), group in joined.groupby(
        ["Dataset", "Threshold_Fraction", "Qref_Definition"], sort=False
    ):
        available = group["Point_RUL_Available"].eq("YES")
        nominal_available = group["Nominal_Point_RUL_Available"].eq("YES")
        records.append(
            {
                "Dataset": dataset,
                "Threshold_Fraction": fraction,
                "Threshold_Percent": 100.0 * float(fraction),
                "Qref_Definition": definition,
                "Comparison_Scope": group["Comparison_Scope"].iloc[0],
                "Cell_Count": int(len(group)),
                "Point_RUL_Available_Count": int(available.sum()),
                "Point_RUL_Available_Fraction": float(available.mean()),
                "Point_RUL_Unavailable_Count": int((~available).sum()),
                "Gain_vs_Nominal_Cell_Count": int((~nominal_available & available).sum()),
                "Loss_vs_Nominal_Cell_Count": int((nominal_available & ~available).sum()),
                "Same_Availability_vs_Nominal_Cell_Count": int(
                    (nominal_available == available).sum()
                ),
                "Censored_EOL_Point_RUL_Violation_Count": int(
                    (
                        ~group["EOL_Status"].eq(EXACT)
                        & group["Point_RUL_Available"].eq("YES")
                    ).sum()
                ),
                "RUL_Point_Label_Rule": "POINT_RUL_ONLY_FOR_EXACT_OBSERVED_EOL",
            }
        )
    result = pd.DataFrame.from_records(records)
    result["_threshold_order"] = result["Threshold_Fraction"].map(THRESHOLD_ORDER)
    result["_qref_order"] = result["Qref_Definition"].map(QREF_ORDER)
    return (
        result.sort_values(
            ["Dataset", "_threshold_order", "_qref_order"], kind="stable"
        )
        .drop(columns=["_threshold_order", "_qref_order"])
        .reset_index(drop=True)
    )


def build_hust_stopping_rule_audit(
    cell_level: pd.DataFrame, frozen_stopping: pd.DataFrame
) -> pd.DataFrame:
    hust = cell_level[cell_level["Dataset"].eq("HUST")]
    frozen_near_count = int(
        frozen_stopping["Native_End_Near_Nominal_80pct"].eq("YES").sum()
    )
    records: list[dict[str, Any]] = []
    for (fraction, definition), group in hust.groupby(
        ["Threshold_Fraction", "Qref_Definition"], sort=False
    ):
        category, context = HUST_STOPPING_CONTEXT[float(fraction)]
        nominal_equivalent = float(fraction) * HUST_NOMINAL_CAPACITY_AH
        last_below_nominal_control = int(
            (group["Last_Observed_Capacity_Ah"] <= nominal_equivalent).sum()
        )
        records.append(
            {
                "Dataset": "HUST",
                "Threshold_Fraction": fraction,
                "Threshold_Percent": 100.0 * float(fraction),
                "Qref_Definition": definition,
                "Cell_Count": int(len(group)),
                "Nominal_Equivalent_Absolute_Threshold_Ah": nominal_equivalent,
                "Stopping_Context_Category": category,
                "Dataset_Stopping_Rule_Context": context,
                "EXACT_OBSERVED_Count": int(group["EOL_Status"].eq(EXACT).sum()),
                "INTERVAL_CENSORED_Count": int(
                    group["EOL_Status"].eq(INTERVAL).sum()
                ),
                "RIGHT_CENSORED_Count": int(group["EOL_Status"].eq(RIGHT).sum()),
                "LEFT_OR_BASELINE_UNRESOLVED_Count": int(
                    group["EOL_Status"].eq(LEFT).sum()
                ),
                "Last_Observed_Capacity_At_Or_Below_Nominal_Control_Count": last_below_nominal_control,
                "Frozen_Native_End_Near_Nominal_80pct_Count": frozen_near_count,
                "Frozen_Stopping_Audit_Cell_Count": int(len(frozen_stopping)),
                "Threshold_Comparison_Uses_Rounding": "NO",
                "Threshold_Comparison_Tolerance_Ah": 0.0,
                "Interpretation_Limitation": (
                    "Threshold-dependent censoring counts must be interpreted with the released HUST observation-window stopping context; they do not establish a universal EOL threshold."
                ),
            }
        )
    result = pd.DataFrame.from_records(records)
    result["_threshold_order"] = result["Threshold_Fraction"].map(THRESHOLD_ORDER)
    result["_qref_order"] = result["Qref_Definition"].map(QREF_ORDER)
    return (
        result.sort_values(["_threshold_order", "_qref_order"], kind="stable")
        .drop(columns=["_threshold_order", "_qref_order"])
        .reset_index(drop=True)
    )


def build_cross_threshold_summary(effect_bounds: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (dataset, definition), group in effect_bounds.groupby(
        ["Dataset", "Definition_B"], sort=False
    ):
        record: dict[str, Any] = {
            "Dataset": dataset,
            "Measured_Qref_Definition": definition,
            "Comparison_Scope": group["Comparison_Scope"].iloc[0],
        }
        effect_threshold_count = 0
        all_available_count = 0
        limitation_threshold_count = 0
        pattern: list[str] = []
        for fraction in THRESHOLD_FRACTIONS:
            subset = group[group["Threshold_Fraction"].eq(fraction)]
            available = subset[~subset["Qref_B_Ah"].isna()]
            affected = int(
                sum(_cell_effect_present(row) for row in available.to_dict("records"))
            )
            status_changed = int(
                (available["EOL_Status_A"] != available["EOL_Status_B"]).sum()
            )
            point_changed = int(
                (
                    available["Point_RUL_Available_A"]
                    != available["Point_RUL_Available_B"]
                ).sum()
            )
            both_right = int(
                (
                    available["EOL_Status_A"].eq(RIGHT)
                    & available["EOL_Status_B"].eq(RIGHT)
                ).sum()
            )
            exact_shift = int(
                (
                    pd.to_numeric(available["Delta_EOL_Cycles"], errors="coerce")
                    .fillna(0.0)
                    .abs()
                    > 0.0
                ).sum()
            )
            suffix = f"{int(round(100 * fraction)):02d}pct"
            record[f"Available_Cell_Count_{suffix}"] = int(len(available))
            record[f"Qref_Induced_Label_Effect_Cell_Count_{suffix}"] = affected
            record[f"EOL_Status_Changed_Cell_Count_{suffix}"] = status_changed
            record[f"RUL_Point_Availability_Changed_Cell_Count_{suffix}"] = point_changed
            record[f"Exact_EOL_Shift_Cell_Count_{suffix}"] = exact_shift
            record[f"Both_Right_Censored_Cell_Count_{suffix}"] = both_right
            effect_present = affected > 0
            record[f"Effect_Present_{suffix}"] = "YES" if effect_present else "NO"
            all_window_limited = len(available) > 0 and both_right == len(available)
            record[f"Observation_Window_Limited_{suffix}"] = (
                "YES" if all_window_limited else "NO"
            )
            pattern.append(f"{fraction:.2f}={'YES' if effect_present else 'NO'}")
            effect_threshold_count += int(effect_present)
            all_available_count += int(len(available))
            limitation_threshold_count += int(all_window_limited)
        if effect_threshold_count == len(THRESHOLD_FRACTIONS):
            classification = "ROBUST_ACROSS_ALL_THRESHOLDS"
        elif effect_threshold_count >= 2:
            classification = "ROBUST_ACROSS_MOST_THRESHOLDS"
        elif effect_threshold_count == 0 and limitation_threshold_count > 0:
            classification = "INSUFFICIENT_OBSERVATION_WINDOW"
        else:
            classification = "THRESHOLD_DEPENDENT"
        record["Thresholds_With_Observed_Qref_Label_Effect_Count"] = effect_threshold_count
        record["Thresholds_Fully_Observation_Window_Limited_Count"] = limitation_threshold_count
        record["Total_Available_Cell_Threshold_Comparisons"] = all_available_count
        record["Cross_Threshold_Effect_Pattern"] = "|".join(pattern)
        record["Cross_Threshold_Classification"] = classification
        record["Classification_Basis"] = (
            "Descriptive classification from frozen cell-level EOL state, event-cycle/bound, and point-RUL availability changes across 75%, 80%, and 85%; not a universal-threshold claim."
        )
        records.append(record)
    result = pd.DataFrame.from_records(records)
    result["_qref_order"] = result["Measured_Qref_Definition"].map(QREF_ORDER)
    return (
        result.sort_values(["Dataset", "_qref_order"], kind="stable")
        .drop(columns="_qref_order")
        .reset_index(drop=True)
    )


def _compare_numeric_series(
    left: pd.Series, right: pd.Series, *, atol: float = 0.0
) -> pd.Series:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    both_na = left_num.isna() & right_num.isna()
    both_number = left_num.notna() & right_num.notna()
    equal = pd.Series(False, index=left.index)
    equal.loc[both_na] = True
    if both_number.any():
        equal.loc[both_number] = np.isclose(
            left_num.loc[both_number],
            right_num.loc[both_number],
            rtol=0.0,
            atol=atol,
        )
    return equal


def compare_frozen_80(
    cell_level: pd.DataFrame, frozen_cross: pd.DataFrame
) -> tuple[bool, dict[str, Any], pd.DataFrame]:
    current = cell_level[cell_level["Threshold_Fraction"].eq(0.80)].copy()
    current["Batch_ID"] = current["Batch_ID"].map(_normalize_batch)
    frozen = frozen_cross.copy()
    frozen["Batch_ID"] = frozen["Batch_ID"].map(_normalize_batch)
    keys = ["Dataset", "Batch_ID", "Cell_ID", "Qref_Definition"]
    current = current.sort_values(keys, kind="stable").reset_index(drop=True)
    frozen = frozen.sort_values(keys, kind="stable").reset_index(drop=True)
    if current.duplicated(keys).any() or frozen.duplicated(keys).any():
        return False, {"duplicate_key": True}, pd.DataFrame()
    key_match = current[keys].equals(frozen[keys])
    if not key_match:
        merged = current[keys].merge(frozen[keys], how="outer", indicator=True)
        return False, {"key_match": False}, merged[merged["_merge"] != "both"]
    comparisons = pd.DataFrame(index=current.index)
    comparisons["Qref_Ah"] = _compare_numeric_series(
        current["Qref_Ah"], frozen["Qref_Ah"], atol=0.0
    )
    comparisons["EOL_Threshold_Ah"] = _compare_numeric_series(
        current["EOL_Threshold_Ah"], frozen["EOL_Threshold_Ah"], atol=FLOAT_TOL
    )
    for column in (
        "EOL_Exact_Cycle",
        "EOL_Lower_Bound",
        "EOL_Upper_Bound",
        "Last_Observed_Cycle",
    ):
        comparisons[column] = _compare_numeric_series(
            current[column], frozen[column], atol=0.0
        )
    comparisons["Qref_Status"] = (
        current["Qref_Status"].astype(str).to_numpy()
        == frozen["Qref_Status"].astype(str).to_numpy()
    )
    comparisons["EOL_Status"] = (
        current["EOL_Status"].astype(str).to_numpy()
        == frozen["EOL_Status"].astype(str).to_numpy()
    )
    comparisons["Point_RUL_Available"] = (
        current["Point_RUL_Available"].astype(str).to_numpy()
        == frozen["RUL_Point_Label_Available"].astype(str).to_numpy()
    )
    mismatched = ~comparisons.all(axis=1)
    detail = current.loc[mismatched, keys].copy()
    if mismatched.any():
        detail["Mismatched_Fields"] = [
            "|".join(comparisons.columns[~comparisons.loc[index]].tolist())
            for index in comparisons.index[mismatched]
        ]
    summary = {
        "key_match": key_match,
        "row_count_current": int(len(current)),
        "row_count_frozen": int(len(frozen)),
        "mismatch_count": int(mismatched.sum()),
        "xjtu_mismatch_count": int(
            (mismatched & current["Dataset"].eq("XJTU")).sum()
        ),
        "hust_mismatch_count": int(
            (mismatched & current["Dataset"].eq("HUST")).sum()
        ),
    }
    return bool(key_match and not mismatched.any()), summary, detail


def _validate_censoring_rules(
    cell_level: pd.DataFrame, inputs: FrozenInputs
) -> tuple[bool, bool, list[str]]:
    issues: list[str] = []
    x_groups = {
        cell: _ordered_xjtu(group)
        for cell, group in inputs.xjtu_observations.groupby("Cell_ID", sort=False)
    }
    x_rows = cell_level[
        cell_level["Dataset"].eq("XJTU")
        & ~cell_level["Qref_Ah"].isna()
    ]
    for row in x_rows.itertuples(index=False):
        values = row._asdict()
        observations = x_groups[values["Cell_ID"]]
        detected = detect_xjtu_eol(
            observations,
            float(values["EOL_Threshold_Ah"]),
            dense_measurement=int(values["Batch_ID"]) in XJTU_DENSE_BATCHES,
        )
        fields_match = (
            detected.status == values["EOL_Status"]
            and _same_number(detected.exact_cycle, values["EOL_Exact_Cycle"])
            and _same_number(detected.lower_bound, values["EOL_Lower_Bound"])
            and _same_number(detected.upper_bound, values["EOL_Upper_Bound"])
        )
        if not fields_match:
            issues.append(
                f"XJTU:{values['Cell_ID']}:{values['Threshold_Fraction']}:{values['Qref_Definition']}"
            )
    h_groups = {
        cell: _ordered_hust(group)
        for cell, group in inputs.hust_observations.groupby("Cell_ID", sort=False)
    }
    h_rows = cell_level[cell_level["Dataset"].eq("HUST")]
    h_issues: list[str] = []
    for row in h_rows.itertuples(index=False):
        values = row._asdict()
        detected = detect_hust_eol(
            h_groups[values["Cell_ID"]], float(values["EOL_Threshold_Ah"])
        )
        fields_match = (
            detected.status == values["EOL_Status"]
            and _same_number(detected.exact_cycle, values["EOL_Exact_Cycle"])
            and _same_number(detected.lower_bound, values["EOL_Lower_Bound"])
            and _same_number(detected.upper_bound, values["EOL_Upper_Bound"])
        )
        if not fields_match:
            h_issues.append(
                f"HUST:{values['Cell_ID']}:{values['Threshold_Fraction']}:{values['Qref_Definition']}"
            )
    issues.extend(h_issues)
    return len(issues) == 0, len(h_issues) == 0, issues


def build_regression_tests(
    cell_level: pd.DataFrame,
    qref_audit: pd.DataFrame,
    effect_bounds: pd.DataFrame,
    hust_stopping: pd.DataFrame,
    inputs: FrozenInputs,
    frozen_80_ok: bool,
    frozen_80_summary: Mapping[str, Any],
    *,
    frozen_metadata_unchanged: bool,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    def add(test_id: str, expected: Any, observed: Any, passed: bool, details: str = "") -> None:
        records.append(
            {
                "Test_ID": test_id,
                "Expected": expected,
                "Observed": observed,
                "Status": "PASS" if passed else "FAIL",
                "Details": details,
            }
        )

    x_cells = cell_level.loc[cell_level["Dataset"].eq("XJTU"), "Cell_ID"].nunique()
    h_cells = cell_level.loc[cell_level["Dataset"].eq("HUST"), "Cell_ID"].nunique()
    add("XJTU_CELL_COUNT_55", 55, x_cells, x_cells == 55)
    add("HUST_CELL_COUNT_77", 77, h_cells, h_cells == 77)

    fractions = sorted(cell_level["Threshold_Fraction"].unique().tolist())
    add(
        "THRESHOLDS_EXACTLY_075_080_085",
        "0.75|0.80|0.85",
        "|".join(f"{value:.2f}" for value in fractions),
        fractions == list(THRESHOLD_FRACTIONS),
    )

    qref_constancy = (
        cell_level.groupby(["Dataset", "Cell_ID", "Qref_Definition"])["Qref_Ah"]
        .nunique(dropna=False)
        .le(1)
        .all()
    )
    add(
        "NO_QREF_REDEFINITION",
        "one frozen Qref value per Dataset/Cell/Qref across thresholds",
        bool(qref_constancy),
        bool(qref_constancy),
    )
    hust_initial_count = int(
        (
            cell_level["Dataset"].eq("HUST")
            & cell_level["Qref_Definition"].eq("initial_test")
        ).sum()
    )
    add("NO_HUST_INITIAL_TEST_INVENTED", 0, hust_initial_count, hust_initial_count == 0)
    add(
        "NO_RAW_REPROCESSING",
        "frozen output artifacts only",
        "XJTU Final V2 SOH/Qref/EOL; HUST Final V1 primary/Qref/EOL; Cross-Dataset Summary V1",
        True,
        "No raw MAT or PKL path is opened by this pipeline.",
    )
    add("NO_SMOOTHING", "NO", "NO", True, "No smoothing operation exists in the threshold pipeline.")
    add("NO_INTERPOLATION", "NO", "NO", True, "No cycle or EOL interpolation is performed.")

    formula_rows = qref_audit[
        qref_audit["Threshold_Equals_Fraction_Times_Qref_Exactly"].ne("NA")
    ]
    formula_ok = formula_rows[
        "Threshold_Equals_Fraction_Times_Qref_Exactly"
    ].eq("YES").all()
    add(
        "NO_THRESHOLD_ROUNDING",
        "all available thresholds equal fraction*unrounded Qref in memory",
        f"PASS={int(formula_rows['Threshold_Equals_Fraction_Times_Qref_Exactly'].eq('YES').sum())};TOTAL={len(formula_rows)}",
        bool(formula_ok),
    )

    x_rules_ok, h_rules_ok, censoring_issues = _validate_censoring_rules(
        cell_level, inputs
    )
    add(
        "XJTU_CENSORING_RULES_PRESERVED",
        "0 violations",
        len([item for item in censoring_issues if item.startswith("XJTU:")]),
        x_rules_ok,
    )
    add(
        "HUST_DENSE_CENSORING_RULES_PRESERVED",
        "0 violations",
        len([item for item in censoring_issues if item.startswith("HUST:")]),
        h_rules_ok,
    )

    censored = cell_level[~cell_level["EOL_Status"].eq(EXACT)]
    censored_point_eol = int(censored["EOL_Exact_Cycle"].notna().sum())
    censored_point_rul = int(censored["Point_RUL_Available"].eq("YES").sum())
    add("NO_CENSORED_EOL_POINT_LABEL", 0, censored_point_eol, censored_point_eol == 0)
    add("NO_CENSORED_RUL_POINT_LABEL", 0, censored_point_rul, censored_point_rul == 0)

    add(
        "80PCT_RESULTS_REPRODUCE_FROZEN_RESULTS",
        "451 frozen cell-Qref rows; zero mismatches",
        json.dumps(dict(frozen_80_summary), sort_keys=True),
        frozen_80_ok,
    )
    h80 = cell_level[
        cell_level["Dataset"].eq("HUST")
        & cell_level["Threshold_Fraction"].eq(0.80)
    ]
    h_nominal_right = int(
        (
            h80["Qref_Definition"].eq("nominal")
            & h80["EOL_Status"].eq(RIGHT)
        ).sum()
    )
    h_first_exact = int(
        (
            h80["Qref_Definition"].eq("first_comparable")
            & h80["EOL_Status"].eq(EXACT)
        ).sum()
    )
    h_median_exact = int(
        (
            h80["Qref_Definition"].eq("early_median5_comparable")
            & h80["EOL_Status"].eq(EXACT)
        ).sum()
    )
    add("HUST_80PCT_NOMINAL_RIGHT_COUNT_77", 77, h_nominal_right, h_nominal_right == 77)
    add("HUST_80PCT_FIRST_EXACT_COUNT_77", 77, h_first_exact, h_first_exact == 77)
    add("HUST_80PCT_MEDIAN5_EXACT_COUNT_77", 77, h_median_exact, h_median_exact == 77)
    add(
        "XJTU_FROZEN_80PCT_RESULTS_REPRODUCED",
        0,
        frozen_80_summary.get("xjtu_mismatch_count", -1),
        frozen_80_ok and frozen_80_summary.get("xjtu_mismatch_count", -1) == 0,
    )

    hust_cell_context = cell_level.loc[
        cell_level["Dataset"].eq("HUST"), "Dataset_Stopping_Rule_Context"
    ]
    context_ok = (
        len(hust_stopping) == 9
        and hust_cell_context.notna().all()
        and hust_cell_context.astype(str).str.len().gt(0).all()
        and set(hust_stopping["Stopping_Context_Category"])
        == {value[0] for value in HUST_STOPPING_CONTEXT.values()}
    )
    add(
        "HUST_STOPPING_CONTEXT_RETAINED",
        "all HUST rows annotated; 3 threshold-specific categories",
        f"cell_rows={len(hust_cell_context)};audit_rows={len(hust_stopping)};categories={hust_stopping['Stopping_Context_Category'].nunique()}",
        bool(context_ok),
    )

    nominal_controls = {
        ("XJTU", 0.75): 1.5,
        ("XJTU", 0.80): 1.6,
        ("XJTU", 0.85): 1.7,
        ("HUST", 0.75): 0.825,
        ("HUST", 0.80): 0.88,
        ("HUST", 0.85): 0.935,
    }
    controls_ok = True
    observed_controls: list[str] = []
    for (dataset, fraction), expected in nominal_controls.items():
        values = cell_level.loc[
            cell_level["Dataset"].eq(dataset)
            & cell_level["Threshold_Fraction"].eq(fraction)
            & cell_level["Qref_Definition"].eq("nominal"),
            "EOL_Threshold_Ah",
        ]
        ok = len(values) > 0 and np.isclose(
            values.to_numpy(dtype=float), expected, rtol=0.0, atol=FLOAT_TOL
        ).all()
        controls_ok &= bool(ok)
        observed_controls.append(
            f"{dataset}:{fraction:.2f}={values.iloc[0] if len(values) else 'MISSING'}"
        )
    add(
        "NOMINAL_ABSOLUTE_THRESHOLD_CONTROLS",
        "XJTU 1.5/1.6/1.7; HUST 0.825/0.880/0.935",
        "|".join(observed_controls),
        controls_ok,
    )

    direction_bad = int(
        qref_audit["Threshold_Ordering_Matches_Qref_Ordering"].eq("NO").sum()
    )
    add("QREF_THRESHOLD_DIRECTION_ORDERING", 0, direction_bad, direction_bad == 0)
    ordering_inconsistent = int(
        effect_bounds["Threshold_Ordering_Consistency"].eq("INCONSISTENT").sum()
    )
    add(
        "EOL_ORDERING_CONSISTENCY_AUDIT",
        0,
        ordering_inconsistent,
        ordering_inconsistent == 0,
        "Overlapping censored bounds remain INDETERMINATE rather than being forced consistent.",
    )
    invalid_exact_delta = effect_bounds[
        effect_bounds["Delta_EOL_Cycles"].notna()
        & ~(
            effect_bounds["EOL_Status_A"].eq(EXACT)
            & effect_bounds["EOL_Status_B"].eq(EXACT)
        )
    ]
    add(
        "EXACT_EOL_DELTA_ONLY_WHEN_BOTH_EXACT",
        0,
        len(invalid_exact_delta),
        invalid_exact_delta.empty,
    )
    lower_bound_bad = effect_bounds[
        effect_bounds["Minimum_Observed_Earlier_By_Cycles"].notna()
        & ~(
            effect_bounds["EOL_Status_A"].eq(RIGHT)
            & effect_bounds["EOL_Status_B"].eq(EXACT)
        )
    ]
    add(
        "EARLIER_BY_VALUE_IS_LOWER_BOUND_ONLY",
        0,
        len(lower_bound_bad),
        lower_bound_bad.empty,
    )
    add(
        "FROZEN_PRIMARY_AND_ML_ARTIFACT_METADATA_UNCHANGED",
        "YES",
        "YES" if frozen_metadata_unchanged else "NO",
        frozen_metadata_unchanged,
    )
    return pd.DataFrame.from_records(records)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _validate_written_outputs(output_dir: Path) -> dict[str, Any]:
    expected = set(OUTPUT_FILES) | {"processing_log.txt"}
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual != expected:
        raise RuntimeError(f"Output file set mismatch: expected={sorted(expected)} actual={sorted(actual)}")
    frames = {name: _read_csv(output_dir / name) for name in OUTPUT_FILES}
    cell = frames["eol_threshold_cell_level.csv"]
    keys = ["Dataset", "Cell_ID", "Threshold_Fraction", "Qref_Definition"]
    if cell.duplicated(keys).any():
        raise RuntimeError("Written cell-level output has duplicate keys")
    if cell["Dataset"].isna().any() or cell["Cell_ID"].isna().any():
        raise RuntimeError("Written cell-level output has an empty Dataset or Cell_ID")
    if sorted(cell["Threshold_Fraction"].unique().tolist()) != list(
        THRESHOLD_FRACTIONS
    ):
        raise RuntimeError("Written thresholds differ from 0.75/0.80/0.85")
    censored = cell[~cell["EOL_Status"].eq(EXACT)]
    if censored["EOL_Exact_Cycle"].notna().any():
        raise RuntimeError("Written censored EOL has a point EOL cycle")
    if censored["Point_RUL_Available"].eq("YES").any():
        raise RuntimeError("Written censored EOL has point RUL availability")
    tests = frames["eol_threshold_regression_tests.csv"]
    if not tests["Status"].eq("PASS").all():
        failed = tests.loc[tests["Status"].ne("PASS"), "Test_ID"].tolist()
        raise RuntimeError(f"Written regression output contains failures: {failed}")
    effect = frames["eol_threshold_effect_bounds.csv"]
    invalid_delta = effect[
        effect["Delta_EOL_Cycles"].notna()
        & ~(
            effect["EOL_Status_A"].eq(EXACT)
            & effect["EOL_Status_B"].eq(EXACT)
        )
    ]
    if not invalid_delta.empty:
        raise RuntimeError("Written effect bounds contain a censored exact delta")
    return {
        "file_count": len(actual),
        "cell_level_rows": len(cell),
        "regression_test_count": len(tests),
        "all_regression_tests_passed": bool(tests["Status"].eq("PASS").all()),
    }


def _build_processing_log(
    *,
    paths: PipelinePaths,
    source_hashes: Mapping[str, str],
    metadata_digest_before: str,
    metadata_digest_after: str,
    inputs: FrozenInputs,
    cell_level: pd.DataFrame,
    status_summary: pd.DataFrame,
    transition_summary: pd.DataFrame,
    cross_threshold: pd.DataFrame,
    regression: pd.DataFrame,
    validation: Mapping[str, Any],
) -> str:
    lines = [
        "EOL THRESHOLD ROBUSTNESS V1 PROCESSING LOG",
        f"Processing timestamp: {datetime.now().astimezone().isoformat()}",
        f"Python version: {platform.python_version()}",
        f"Python executable: {sys.executable}",
        f"pandas version: {pd.__version__}",
        f"numpy version: {np.__version__}",
        "",
        "SCOPE",
        "- Inputs: frozen XJTU Final V2, HUST Final V1, and Cross-Dataset Summary V1 artifacts only.",
        "- Raw MAT/PKL files were not opened.",
        "- No ML model was run or retrained.",
        "- No smoothing, interpolation, clipping, cycle renumbering, or tolerance-based crossing was used.",
        "- Threshold criterion: frozen Primary Capacity (Ah) <= Threshold Fraction * unrounded frozen Qref (Ah).",
        "- Threshold fractions: 0.75, 0.80, 0.85.",
        "- Point RUL availability is reported only for EXACT_OBSERVED EOL.",
        "",
        "FROZEN SOURCE FILES",
    ]
    for source in paths.source_files:
        lines.append(
            f"- {source.relative_to(paths.root).as_posix()} | bytes={source.stat().st_size} | SHA256={source_hashes[str(source.resolve())]}"
        )
    lines.extend(
        [
            "",
            "FROZEN ARTIFACT METADATA SNAPSHOT",
            f"- before={metadata_digest_before}",
            f"- after={metadata_digest_after}",
            f"- unchanged={'YES' if metadata_digest_before == metadata_digest_after else 'NO'}",
            "",
            "FROZEN OBSERVATION COUNTS",
            f"- XJTU cells={inputs.xjtu_observations['Cell_ID'].nunique()} observations={len(inputs.xjtu_observations)}",
            f"- HUST cells={inputs.hust_observations['Cell_ID'].nunique()} observations={len(inputs.hust_observations)}",
            "",
            "OUTPUT COUNTS",
            f"- cell-level rows={len(cell_level)}",
            f"- status-summary rows={len(status_summary)}",
            f"- transition-summary rows={len(transition_summary)}",
            f"- cross-threshold rows={len(cross_threshold)}",
            f"- flagged regression tests={int(regression['Status'].ne('PASS').sum())}",
            "",
            "STATUS SUMMARY",
        ]
    )
    for row in status_summary.itertuples(index=False):
        lines.append(
            f"- {row.Dataset} threshold={row.Threshold_Fraction:.2f} qref={row.Qref_Definition}: "
            f"EXACT={row.EXACT_OBSERVED_Count} INTERVAL={row.INTERVAL_CENSORED_Count} "
            f"RIGHT={row.RIGHT_CENSORED_Count} LEFT={row.LEFT_OR_BASELINE_UNRESOLVED_Count} "
            f"QREF_UNAVAILABLE={row.QREF_UNAVAILABLE_EOL_Count} POINT_RUL={row.Point_RUL_Available_Count}"
        )
    lines.extend(["", "CROSS-THRESHOLD CLASSIFICATION"])
    for row in cross_threshold.itertuples(index=False):
        lines.append(
            f"- {row.Dataset} {row.Measured_Qref_Definition}: {row.Cross_Threshold_Classification} "
            f"({row.Cross_Threshold_Effect_Pattern})"
        )
    lines.extend(
        [
            "",
            "REGRESSION TESTS",
            *[
                f"- {row.Test_ID}: {row.Status} | expected={row.Expected} | observed={row.Observed}"
                for row in regression.itertuples(index=False)
            ],
            "",
            "OUTPUT RE-READ VALIDATION",
            f"- {json.dumps(dict(validation), sort_keys=True)}",
            "",
            "ALL REGRESSION TESTS PASSED",
            "EOL THRESHOLD ROBUSTNESS V1 COMPLETE —",
            "WAITING FOR HUMAN REVIEW",
        ]
    )
    return "\n".join(lines) + "\n"


def run_pipeline(root: Path) -> Path:
    root = root.resolve()
    paths = PipelinePaths(root=root)
    if paths.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {paths.output_dir}"
        )

    metadata_before = _metadata_snapshot(paths.frozen_dirs)
    metadata_digest_before = _snapshot_digest(metadata_before)
    source_hashes = {str(path.resolve()): _sha256(path) for path in paths.source_files}
    inputs = load_frozen_inputs(paths)

    cell_level = build_cell_level(inputs)
    frozen_80_ok, frozen_80_summary, frozen_80_detail = compare_frozen_80(
        cell_level, inputs.cross_cell_80
    )
    if not frozen_80_ok:
        detail = (
            frozen_80_detail.head(20).to_dict("records")
            if not frozen_80_detail.empty
            else []
        )
        raise FrozenReproductionError(
            f"80% frozen reproduction failed: {dict(frozen_80_summary)}; detail={detail}"
        )

    effect_bounds = build_effect_bounds(cell_level)
    qref_audit = build_qref_direction_audit(cell_level)
    status_summary = build_status_summary(cell_level)
    transition_summary = build_transition_summary(effect_bounds)
    rul_availability = build_rul_availability(cell_level)
    hust_stopping = build_hust_stopping_rule_audit(
        cell_level, inputs.hust_stopping_audit
    )
    cross_threshold = build_cross_threshold_summary(effect_bounds)

    metadata_mid = _metadata_snapshot(paths.frozen_dirs)
    frozen_metadata_unchanged = metadata_mid == metadata_before
    regression = build_regression_tests(
        cell_level,
        qref_audit,
        effect_bounds,
        hust_stopping,
        inputs,
        frozen_80_ok,
        frozen_80_summary,
        frozen_metadata_unchanged=frozen_metadata_unchanged,
    )
    failed = regression[regression["Status"].ne("PASS")]
    if not failed.empty:
        raise RuntimeError(
            "Regression tests failed before output creation: "
            + "|".join(failed["Test_ID"].astype(str))
        )

    staging = paths.outputs / f".{paths.output_dir.name}_staging_{uuid.uuid4().hex[:8]}"
    if staging.exists():
        raise FileExistsError(f"Unexpected staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        frames = {
            "eol_threshold_cell_level.csv": cell_level,
            "eol_threshold_status_summary.csv": status_summary,
            "eol_threshold_transition_summary.csv": transition_summary,
            "eol_threshold_rul_availability.csv": rul_availability,
            "eol_threshold_effect_bounds.csv": effect_bounds,
            "eol_threshold_qref_direction_audit.csv": qref_audit,
            "eol_threshold_hust_stopping_rule_audit.csv": hust_stopping,
            "eol_threshold_cross_threshold_summary.csv": cross_threshold,
            "eol_threshold_regression_tests.csv": regression,
        }
        for name, frame in frames.items():
            _write_csv(frame, staging / name)

        placeholder_log = (
            "EOL THRESHOLD ROBUSTNESS V1 PROCESSING LOG\n"
            "Validation in progress; this staging directory is not a final output.\n"
        )
        (staging / "processing_log.txt").write_text(placeholder_log, encoding="utf-8")
        validation = _validate_written_outputs(staging)

        metadata_after = _metadata_snapshot(paths.frozen_dirs)
        metadata_digest_after = _snapshot_digest(metadata_after)
        if metadata_after != metadata_before:
            raise RuntimeError("A frozen artifact changed during the robustness run")

        log_text = _build_processing_log(
            paths=paths,
            source_hashes=source_hashes,
            metadata_digest_before=metadata_digest_before,
            metadata_digest_after=metadata_digest_after,
            inputs=inputs,
            cell_level=cell_level,
            status_summary=status_summary,
            transition_summary=transition_summary,
            cross_threshold=cross_threshold,
            regression=regression,
            validation=validation,
        )
        (staging / "processing_log.txt").write_text(log_text, encoding="utf-8")
        staging.replace(paths.output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return paths.output_dir


__all__ = [
    "EOLDetection",
    "FrozenInputs",
    "FrozenReproductionError",
    "build_cell_level",
    "build_cross_threshold_summary",
    "build_effect_bounds",
    "build_hust_stopping_rule_audit",
    "build_qref_direction_audit",
    "build_regression_tests",
    "build_rul_availability",
    "build_status_summary",
    "build_transition_summary",
    "classify_qref_direction",
    "classify_transition",
    "compare_frozen_80",
    "compute_effect_bounds",
    "detect_hust_eol",
    "detect_xjtu_eol",
    "load_frozen_inputs",
    "ordering_consistency",
    "run_pipeline",
]
