from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import PIPELINE_VERSION
from .config import make_scan_config
from .safe_pickle import ALLOWED_PICKLE_GLOBALS, load_hust_pickle


@dataclass(frozen=True)
class ScanResult:
    output_dir: Path
    scan_status: str
    canonical_cell_count: int
    total_cycle_count: int
    file_type_counts: dict[str, int]
    schema_variant_count: int
    capacity_field_candidates: tuple[str, ...]
    cells_with_raw_curves: int
    cells_with_native_soh: int
    cells_with_native_rul: int
    protocol_fingerprint_count: int
    protocol_change_cell_count: int
    incomplete_discharge_candidate_count: int
    capacity_jump_check_count: int
    first_capacity_stats: dict[str, float]
    last_capacity_stats: dict[str, float]
    final_capacity_le_0p88_count: int
    final_capacity_gt_0p88_count: int
    representative_cells: tuple[str, ...]
    hard_stop_reasons: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hashes(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        return {}
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _natural_cell_key(value: str) -> tuple[Any, ...]:
    raw = str(value).removeprefix("HUST_")
    parts = re.split(r"(\d+)", raw)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _normal_duplicate_stem(path: Path) -> str:
    stem = path.stem.lower().strip()
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    stem = re.sub(r"(?:[_\- ](?:copy|duplicate|dup))\d*$", "", stem)
    return stem


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        float_format="%.12g",
    )


def _atomic_directory(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))


def _finite_number(value: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if np.isfinite(number) else np.nan


def _numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _safe_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else np.nan


def _safe_min(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else np.nan


def _safe_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else np.nan


def _safe_first(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[0]) if finite.size else np.nan


def _safe_last(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else np.nan


def _serialize_values(values: Iterable[Any], limit: int = 40) -> str:
    items = [str(value) for value in values]
    if len(items) > limit:
        return "|".join(items[:limit]) + f"|...(+{len(items) - limit})"
    return "|".join(items)


def _inventory_files(input_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    files = sorted(
        (path for path in input_dir.rglob("*") if path.is_file()),
        key=lambda path: _natural_cell_key(path.relative_to(input_dir).as_posix()),
    )
    rows: list[dict[str, Any]] = []
    stat_before: dict[Path, tuple[int, int]] = {}
    for index, path in enumerate(files, start=1):
        stat = path.stat()
        stat_before[path] = (stat.st_size, stat.st_mtime_ns)
        rows.append(
            {
                "File_Index": index,
                "Relative_Path": path.relative_to(input_dir).as_posix(),
                "File_Name": path.name,
                "Extension": path.suffix.lower(),
                "File_Size_Bytes": stat.st_size,
                "SHA256": _sha256(path),
                "File_Role": "CANONICAL_CANDIDATE",
                "Duplicate_Of": "",
                "Duplicate_SHA256": "",
                "Hash_Match": "",
                "Included_In_Analysis": "YES",
                "Native_Cell_ID": "",
                "Cell_ID": "",
                "Manifest_Issues": "",
            }
        )
    frame = pd.DataFrame(rows)
    issues: list[str] = []
    if frame.empty:
        return frame, ["NO_INPUT_FILES_FOUND"]

    for _, indices in frame.groupby("SHA256", sort=False).groups.items():
        ordered = sorted(indices, key=lambda idx: _natural_cell_key(frame.at[idx, "File_Name"]))
        canonical = ordered[0]
        frame.at[canonical, "File_Role"] = "CANONICAL"
        frame.at[canonical, "Hash_Match"] = "TRUE"
        for duplicate in ordered[1:]:
            frame.at[duplicate, "File_Role"] = "DUPLICATE_CONFIRMED"
            frame.at[duplicate, "Duplicate_Of"] = frame.at[canonical, "File_Name"]
            frame.at[duplicate, "Duplicate_SHA256"] = frame.at[canonical, "SHA256"]
            frame.at[duplicate, "Hash_Match"] = "TRUE"
            frame.at[duplicate, "Included_In_Analysis"] = "NO"

    for _, indices in frame.groupby(
        frame["File_Name"].map(lambda value: _normal_duplicate_stem(Path(value))),
        sort=False,
    ).groups.items():
        if len(indices) <= 1:
            continue
        hashes = set(frame.loc[list(indices), "SHA256"])
        if len(hashes) > 1:
            for index in indices:
                frame.at[index, "File_Role"] = "DUPLICATE_CONTENT_MISMATCH"
                frame.at[index, "Included_In_Analysis"] = "NO"
                frame.at[index, "Manifest_Issues"] = "DUPLICATE_CONTENT_MISMATCH"
            issues.append(
                "DUPLICATE_CONTENT_MISMATCH:"
                + _serialize_values(frame.loc[list(indices), "File_Name"])
            )

    frame.loc[frame["File_Role"].eq("CANONICAL_CANDIDATE"), "File_Role"] = "CANONICAL"
    frame.loc[frame["File_Role"].eq("CANONICAL"), "Hash_Match"] = "TRUE"

    for path, expected in stat_before.items():
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != expected:
            issues.append(f"RAW_FILE_STAT_CHANGED_DURING_HASH:{path.name}")
    return frame, issues


def _cycle_index_status(keys: Sequence[int], original_order: Sequence[int]) -> dict[str, Any]:
    numeric = sorted(set(keys))
    duplicates = len(keys) - len(numeric)
    missing: list[int] = []
    if numeric:
        missing = sorted(set(range(numeric[0], numeric[-1] + 1)).difference(numeric))
    reset = any(current <= previous for previous, current in zip(original_order, original_order[1:]))
    if duplicates:
        status = "DUPLICATE_CYCLE_INDEX"
    elif reset:
        status = "NONMONOTONIC_OR_RESET"
    elif missing:
        status = "MISSING_CYCLE_INDEX"
    elif numeric and numeric[0] == 1:
        status = "CONTINUOUS_UNIQUE_FROM_1"
    else:
        status = "CONTINUOUS_UNIQUE_NONUNIT_START"
    return {
        "Cycle_Index_Status": status,
        "First_Raw_Cycle_Index": numeric[0] if numeric else np.nan,
        "Last_Raw_Cycle_Index": numeric[-1] if numeric else np.nan,
        "Unique_Cycle_Count": len(numeric),
        "Duplicate_Cycle_Index_Count": duplicates,
        "Missing_Cycle_Count": len(missing),
        "Missing_Cycle_Indexes": _serialize_values(missing),
        "Cycle_Index_Reset_Suspected": "YES" if reset else "NO",
    }


def _empty_cycle_row(
    cell_id: str,
    native_cell_id: str,
    source_file: str,
    raw_cycle: int,
    native_dq: Any,
    native_rul: Any,
    reason: str,
) -> dict[str, Any]:
    return {
        "Cell_ID": cell_id,
        "Native_Cell_ID": native_cell_id,
        "Source_File": source_file,
        "Raw_Cycle_Index": raw_cycle,
        "Cycle_Number_Field_Status": "UNAVAILABLE",
        "Raw_Record_Count": 0,
        "Raw_Status_Count": 0,
        "Raw_Status_Sequence": "",
        "Native_Discharge_Capacity_mAh": _finite_number(native_dq),
        "Native_RUL_Cycles": _finite_number(native_rul),
        "Raw_Full_Discharge_Capacity_mAh": np.nan,
        "Capacity_Difference_mAh": np.nan,
        "Capacity_Difference_Pct": np.nan,
        "Capacity_Semantics_Match": "NO",
        "Primary_Capacity_Ah": np.nan,
        "Charge_Stage1_Median_Current_A": np.nan,
        "Charge_Stage2_Plateau_Median_Current_A": np.nan,
        "Charge_Max_V": np.nan,
        "CV_End_Current_A": np.nan,
        "Charge_Protocol_Traceable": "NO",
        "Charge_Complete_Scan_Candidate": "NO",
        "Discharge_Step1_Median_Current_A": np.nan,
        "Discharge_Step2_Median_Current_A": np.nan,
        "Discharge_Step3_Median_Current_A": np.nan,
        "Discharge_Step4_Median_Current_A": np.nan,
        "Discharge_Step1_Capacity_mAh": np.nan,
        "Discharge_Step2_Capacity_mAh": np.nan,
        "Discharge_Step3_Capacity_mAh": np.nan,
        "Discharge_Step4_Capacity_mAh": np.nan,
        "Discharge_Min_V": np.nan,
        "Discharge_End_V": np.nan,
        "Final_Discharge_Current_A": np.nan,
        "Full_Discharge_Traceable": "NO",
        "Full_Discharge_Complete_Scan_Candidate": "NO",
        "Severe_Voltage_Anomaly": "NO",
        "Protocol_Current_Deviation": "NO",
        "Protocol_Change_Candidate": "NO",
        "Capacity_Jump_Flag": "NO",
        "Jump_Reference_Raw_Cycle": np.nan,
        "Jump_Reference_Capacity_Ah": np.nan,
        "Capacity_Jump_Ah": np.nan,
        "Capacity_Jump_Pct": np.nan,
        "Auto_QC_Flag": "INVALID_CANDIDATE",
        "Auto_QC_Reason_Codes": reason,
        "Primary_Capacity_Candidate": "NO",
    }


def _cycle_metrics(
    cell_id: str,
    native_cell_id: str,
    source_file: str,
    raw_cycle: int,
    frame: Any,
    native_dq: Any,
    native_rul: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    required = list(config["raw_column_map"].values())
    if not isinstance(frame, pd.DataFrame):
        return _empty_cycle_row(
            cell_id,
            native_cell_id,
            source_file,
            raw_cycle,
            native_dq,
            native_rul,
            "RAW_CYCLE_NOT_DATAFRAME",
        )
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns or frame.empty:
        reason = (
            "EMPTY_RAW_CYCLE"
            if frame.empty
            else "MISSING_RAW_COLUMNS:" + _serialize_values(missing_columns)
        )
        return _empty_cycle_row(
            cell_id,
            native_cell_id,
            source_file,
            raw_cycle,
            native_dq,
            native_rul,
            reason,
        )

    columns = config["raw_column_map"]
    status = frame[columns["status"]].astype(str).to_numpy()
    current_A = _numeric_array(frame, columns["current_mA"]) / 1000.0
    voltage_V = _numeric_array(frame, columns["voltage_V"])
    capacity_mAh = _numeric_array(frame, columns["capacity_mAh"])
    time_s = _numeric_array(frame, columns["time_s"])
    cycle_field = pd.to_numeric(frame[columns["cycle_index"]], errors="coerce")

    boundaries = np.flatnonzero(status[1:] != status[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(frame)]
    segments: dict[str, list[tuple[int, int]]] = {}
    status_sequence: list[str] = []
    for start, end in zip(starts, ends):
        name = str(status[start])
        status_sequence.append(name)
        segments.setdefault(name, []).append((int(start), int(end)))

    def indices_for(name: str) -> np.ndarray:
        pieces = [np.arange(start, end) for start, end in segments.get(name, [])]
        return np.concatenate(pieces) if pieces else np.asarray([], dtype=int)

    discharge_statuses = list(config["required_discharge_statuses"])
    discharge_indices = [indices_for(name) for name in discharge_statuses]
    full_traceable = all(indices.size for indices in discharge_indices)
    all_discharge = (
        np.concatenate(discharge_indices)
        if any(indices.size for indices in discharge_indices)
        else np.asarray([], dtype=int)
    )
    first_discharge_index = (
        int(discharge_indices[0][0]) if discharge_indices[0].size else None
    )
    final_discharge_index = (
        int(discharge_indices[-1][-1]) if discharge_indices[-1].size else None
    )
    raw_discharge_mAh = (
        float(capacity_mAh[first_discharge_index] - capacity_mAh[final_discharge_index])
        if first_discharge_index is not None
        and final_discharge_index is not None
        and np.isfinite(capacity_mAh[first_discharge_index])
        and np.isfinite(capacity_mAh[final_discharge_index])
        else np.nan
    )
    native_dq_mAh = _finite_number(native_dq)
    difference_mAh = (
        raw_discharge_mAh - native_dq_mAh
        if np.isfinite(raw_discharge_mAh) and np.isfinite(native_dq_mAh)
        else np.nan
    )
    difference_pct = (
        100.0 * difference_mAh / native_dq_mAh
        if np.isfinite(difference_mAh) and native_dq_mAh != 0
        else np.nan
    )
    threshold = config["scan_thresholds"]
    semantics_limit = max(
        float(threshold["capacity_semantics_abs_tolerance_mAh"]),
        float(threshold["capacity_semantics_relative_tolerance"])
        * abs(native_dq_mAh),
    ) if np.isfinite(native_dq_mAh) else np.nan
    semantics_match = bool(
        np.isfinite(difference_mAh)
        and np.isfinite(semantics_limit)
        and abs(difference_mAh) <= semantics_limit
    )

    stage1_indices = indices_for(str(config["charge_stage1_status"]))
    cccv_indices = indices_for(str(config["charge_cccv_status"]))
    all_charge = (
        np.concatenate([value for value in (stage1_indices, cccv_indices) if value.size])
        if stage1_indices.size or cccv_indices.size
        else np.asarray([], dtype=int)
    )
    plateau_indices = (
        cccv_indices[current_A[cccv_indices] >= 0.5]
        if cccv_indices.size
        else np.asarray([], dtype=int)
    )
    cv_end_current = (
        _safe_last(current_A[cccv_indices]) if cccv_indices.size else np.nan
    )
    charge_max = _safe_max(voltage_V[all_charge]) if all_charge.size else np.nan
    charge_traceable = bool(stage1_indices.size and cccv_indices.size)
    charge_complete = bool(
        charge_traceable
        and np.isfinite(charge_max)
        and charge_max >= float(threshold["charge_complete_min_max_voltage_V"])
        and np.isfinite(cv_end_current)
        and cv_end_current <= float(threshold["cv_end_current_upper_A"])
    )

    discharge_min = _safe_min(voltage_V[all_discharge]) if all_discharge.size else np.nan
    discharge_end = (
        _safe_last(voltage_V[discharge_indices[-1]])
        if discharge_indices[-1].size
        else np.nan
    )
    final_current = (
        _safe_median(current_A[discharge_indices[-1]])
        if discharge_indices[-1].size
        else np.nan
    )
    discharge_complete = bool(
        full_traceable
        and np.isfinite(discharge_min)
        and discharge_min
        <= float(threshold["terminal_discharge_complete_max_min_voltage_V"])
        and np.isfinite(discharge_end)
        and discharge_end
        <= float(threshold["terminal_discharge_end_max_voltage_V"])
    )
    severe_voltage = bool(
        all_discharge.size
        and (
            _safe_max(voltage_V[all_discharge])
            > float(threshold["severe_discharge_voltage_high_V"])
            or discharge_min < float(threshold["severe_discharge_voltage_low_V"])
        )
    )

    cycle_values = sorted(
        set(int(value) for value in cycle_field.dropna().astype(int).tolist())
    )
    cycle_field_status = (
        "MATCH"
        if cycle_values == [raw_cycle]
        else (
            "MULTIPLE_VALUES:" + _serialize_values(cycle_values)
            if len(cycle_values) > 1
            else "MISMATCH:" + _serialize_values(cycle_values)
        )
    )

    reasons: list[str] = []
    if not full_traceable:
        reasons.append("MISSING_FULL_FOUR_STAGE_DISCHARGE")
    elif not discharge_complete:
        reasons.append("INCOMPLETE_TERMINAL_DISCHARGE_CANDIDATE")
    if not np.isfinite(native_dq_mAh) or native_dq_mAh <= 0:
        reasons.append("NONPOSITIVE_OR_MISSING_CAPACITY")
    if not semantics_match:
        reasons.append("CAPACITY_SEMANTICS_MISMATCH")
    if severe_voltage:
        reasons.append("SEVERE_VOLTAGE_ANOMALY")
    if cycle_field_status != "MATCH":
        reasons.append("RAW_CYCLE_INDEX_FIELD_MISMATCH")
    invalid = bool(reasons)

    stage_medians: list[float] = []
    stage_capacities: list[float] = []
    for indices in discharge_indices:
        stage_medians.append(
            _safe_median(current_A[indices]) if indices.size else np.nan
        )
        if indices.size:
            start_capacity = _safe_first(capacity_mAh[indices])
            end_capacity = _safe_last(capacity_mAh[indices])
            stage_capacities.append(
                start_capacity - end_capacity
                if np.isfinite(start_capacity) and np.isfinite(end_capacity)
                else np.nan
            )
        else:
            stage_capacities.append(np.nan)

    return {
        "Cell_ID": cell_id,
        "Native_Cell_ID": native_cell_id,
        "Source_File": source_file,
        "Raw_Cycle_Index": raw_cycle,
        "Cycle_Number_Field_Status": cycle_field_status,
        "Raw_Record_Count": len(frame),
        "Raw_Status_Count": len(status_sequence),
        "Raw_Status_Sequence": "|".join(status_sequence),
        "Native_Discharge_Capacity_mAh": native_dq_mAh,
        "Native_RUL_Cycles": _finite_number(native_rul),
        "Raw_Full_Discharge_Capacity_mAh": raw_discharge_mAh,
        "Capacity_Difference_mAh": difference_mAh,
        "Capacity_Difference_Pct": difference_pct,
        "Capacity_Semantics_Match": "YES" if semantics_match else "NO",
        "Primary_Capacity_Ah": native_dq_mAh / 1000.0 if np.isfinite(native_dq_mAh) else np.nan,
        "Charge_Stage1_Median_Current_A": (
            _safe_median(current_A[stage1_indices]) if stage1_indices.size else np.nan
        ),
        "Charge_Stage2_Plateau_Median_Current_A": (
            _safe_median(current_A[plateau_indices]) if plateau_indices.size else np.nan
        ),
        "Charge_Max_V": charge_max,
        "CV_End_Current_A": cv_end_current,
        "Charge_Protocol_Traceable": "YES" if charge_traceable else "NO",
        "Charge_Complete_Scan_Candidate": "YES" if charge_complete else "NO",
        "Discharge_Step1_Median_Current_A": stage_medians[0],
        "Discharge_Step2_Median_Current_A": stage_medians[1],
        "Discharge_Step3_Median_Current_A": stage_medians[2],
        "Discharge_Step4_Median_Current_A": stage_medians[3],
        "Discharge_Step1_Capacity_mAh": stage_capacities[0],
        "Discharge_Step2_Capacity_mAh": stage_capacities[1],
        "Discharge_Step3_Capacity_mAh": stage_capacities[2],
        "Discharge_Step4_Capacity_mAh": stage_capacities[3],
        "Discharge_Min_V": discharge_min,
        "Discharge_End_V": discharge_end,
        "Final_Discharge_Current_A": final_current,
        "Full_Discharge_Traceable": "YES" if full_traceable else "NO",
        "Full_Discharge_Complete_Scan_Candidate": "YES" if discharge_complete else "NO",
        "Severe_Voltage_Anomaly": "YES" if severe_voltage else "NO",
        "Protocol_Current_Deviation": "NO",
        "Protocol_Change_Candidate": "NO",
        "Capacity_Jump_Flag": "NO",
        "Jump_Reference_Raw_Cycle": np.nan,
        "Jump_Reference_Capacity_Ah": np.nan,
        "Capacity_Jump_Ah": np.nan,
        "Capacity_Jump_Pct": np.nan,
        "Auto_QC_Flag": "INVALID_CANDIDATE" if invalid else "PASS",
        "Auto_QC_Reason_Codes": "|".join(reasons) if reasons else "PASS",
        "Primary_Capacity_Candidate": "NO" if invalid else "YES",
    }


def _longest_true_run(mask: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    best = 0
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            end = index + 1 if value and index == len(mask) - 1 else index
            runs.append((start, end))
            best = max(best, end - start)
            start = None
    return best, runs


def _apply_protocol_and_jump_qc(
    cycles: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = cycles.copy()
    result["Raw_Cycle_Index"] = pd.to_numeric(
        result["Raw_Cycle_Index"], errors="raise"
    )
    result = result.sort_values("Raw_Cycle_Index", kind="stable").reset_index(drop=True)
    thresholds = config["scan_thresholds"]
    current_columns = [
        f"Discharge_Step{step}_Median_Current_A" for step in range(1, 5)
    ]
    complete = result["Full_Discharge_Traceable"].eq("YES")
    baseline_rows = result.loc[complete].head(
        int(thresholds["protocol_baseline_cycle_count"])
    )
    baselines = {
        column: float(pd.to_numeric(baseline_rows[column], errors="coerce").median())
        for column in current_columns
    }
    deviation = np.zeros(len(result), dtype=bool)
    for column in current_columns:
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(float)
        baseline = baselines[column]
        tolerance = max(
            float(thresholds["protocol_current_abs_tolerance_A"]),
            float(thresholds["protocol_current_relative_tolerance"]) * abs(baseline),
        ) if np.isfinite(baseline) else np.nan
        if np.isfinite(tolerance):
            deviation |= np.isfinite(values) & (np.abs(values - baseline) > tolerance)
    result["Protocol_Current_Deviation"] = np.where(deviation, "YES", "NO")
    minimum_run = int(thresholds["protocol_change_min_consecutive_cycles"])
    longest_run, runs = _longest_true_run(deviation)
    systematic = np.zeros(len(result), dtype=bool)
    for start, end in runs:
        if end - start >= minimum_run:
            systematic[start:end] = True
    result["Protocol_Change_Candidate"] = np.where(systematic, "YES", "NO")

    for index in np.flatnonzero(systematic):
        reasons = str(result.at[index, "Auto_QC_Reason_Codes"])
        reason_list = [] if reasons == "PASS" else reasons.split("|")
        reason_list.append("PROTOCOL_CHANGE_CANDIDATE")
        result.at[index, "Auto_QC_Reason_Codes"] = "|".join(dict.fromkeys(reason_list))
        if result.at[index, "Auto_QC_Flag"] == "PASS":
            result.at[index, "Auto_QC_Flag"] = "CHECK"

    jump_abs = float(thresholds["capacity_jump_abs_threshold_Ah"])
    jump_rel = float(thresholds["capacity_jump_relative_threshold"])
    reference_index: int | None = None
    for index, row in result.iterrows():
        comparable = (
            row["Primary_Capacity_Candidate"] == "YES"
            and row["Protocol_Change_Candidate"] == "NO"
            and np.isfinite(row["Primary_Capacity_Ah"])
            and float(row["Primary_Capacity_Ah"]) > 0
        )
        if comparable and reference_index is not None:
            reference_capacity = float(result.at[reference_index, "Primary_Capacity_Ah"])
            current_capacity = float(row["Primary_Capacity_Ah"])
            delta = current_capacity - reference_capacity
            result.at[index, "Jump_Reference_Raw_Cycle"] = result.at[
                reference_index, "Raw_Cycle_Index"
            ]
            result.at[index, "Jump_Reference_Capacity_Ah"] = reference_capacity
            result.at[index, "Capacity_Jump_Ah"] = delta
            result.at[index, "Capacity_Jump_Pct"] = 100.0 * delta / reference_capacity
            if abs(delta) > max(jump_abs, jump_rel * abs(reference_capacity)):
                result.at[index, "Capacity_Jump_Flag"] = "YES"
                reasons = str(result.at[index, "Auto_QC_Reason_Codes"])
                reason_list = [] if reasons == "PASS" else reasons.split("|")
                reason_list.append("CAPACITY_JUMP")
                result.at[index, "Auto_QC_Reason_Codes"] = "|".join(
                    dict.fromkeys(reason_list)
                )
                if result.at[index, "Auto_QC_Flag"] == "PASS":
                    result.at[index, "Auto_QC_Flag"] = "CHECK"
        if comparable:
            reference_index = index

    result["Primary_Capacity_Candidate"] = np.where(
        result["Auto_QC_Flag"].eq("INVALID_CANDIDATE"), "NO", "YES"
    )

    valid_fingerprint = result.loc[result["Full_Discharge_Traceable"].eq("YES")]
    fingerprint: dict[str, Any] = {
        "Protocol_Baseline_Cycle_Count": len(baseline_rows),
        "Protocol_Current_Deviation_Cycle_Count": int(deviation.sum()),
        "Protocol_Change_Candidate_Cycle_Count": int(systematic.sum()),
        "Longest_Protocol_Deviation_Run": longest_run,
    }
    rounded_rates: list[float] = []
    for step, column in enumerate(current_columns, start=1):
        values = pd.to_numeric(valid_fingerprint[column], errors="coerce")
        median = float(values.median())
        q25 = float(values.quantile(0.25))
        q75 = float(values.quantile(0.75))
        fingerprint[f"Discharge_Step{step}_Median_Current_A"] = median
        fingerprint[f"Discharge_Step{step}_Current_IQR_A"] = q75 - q25
        c_rate = abs(median) / float(config["nominal_capacity_Ah"])
        fingerprint[f"Estimated_C_Rate_{step}"] = c_rate
        rounded_rates.append(round(c_rate / 0.05) * 0.05)
    cutoff = float(
        pd.to_numeric(valid_fingerprint["Discharge_End_V"], errors="coerce").median()
    )
    fingerprint["Final_Discharge_Cutoff_V"] = cutoff
    fingerprint["Protocol_Fingerprint_ID"] = (
        "CRATE_"
        + "_".join(f"{value:.2f}" for value in rounded_rates)
        + f"_CUT{round(cutoff / 0.05) * 0.05:.2f}V"
    )
    fingerprint["Protocol_Fingerprint_Status"] = (
        "REVIEW_PROTOCOL_CHANGE_CANDIDATE" if systematic.any() else "STABLE_SCAN_CANDIDATE"
    )
    return result, fingerprint


def _schema_signature(
    cell_keys: Sequence[str],
    columns: Sequence[str],
    dtypes: Sequence[str],
    statuses: Sequence[str],
) -> str:
    payload = json.dumps(
        {
            "cell_keys": sorted(cell_keys),
            "columns": list(columns),
            "dtypes": list(dtypes),
            "statuses": sorted(statuses),
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return "SCHEMA_" + hashlib.sha256(payload).hexdigest()[:12].upper()


def _life_band(value: float, series: pd.Series) -> str:
    rank = float(series.rank(method="average", pct=True).loc[value]) if value in series.index else np.nan
    if not np.isfinite(rank):
        return "UNAVAILABLE"
    if rank <= 0.20:
        return "LOW"
    if rank <= 0.40:
        return "LOWER_MIDDLE"
    if rank <= 0.60:
        return "MEDIAN"
    if rank <= 0.80:
        return "UPPER_MIDDLE"
    return "HIGH"


def _quantile_band(rank: float) -> str:
    if rank <= 1 / 3:
        return "LOW"
    if rank <= 2 / 3:
        return "MEDIUM"
    return "HIGH"


def _select_representatives(
    cell_inventory: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    candidates = cell_inventory.loc[
        cell_inventory["Processing_Status"].eq("SCAN_PROCESSED")
        & pd.to_numeric(cell_inventory["First_Valid_Capacity_Ah"], errors="coerce").notna()
    ].copy()
    candidates["Cycle_Life_Rank"] = pd.to_numeric(
        candidates["Cycle_Count"], errors="raise"
    ).rank(method="average", pct=True)
    candidates["First_Capacity_Rank"] = pd.to_numeric(
        candidates["First_Valid_Capacity_Ah"], errors="raise"
    ).rank(method="average", pct=True)
    targets = list(config["representative_selection"]["life_capacity_quantile_targets"])
    penalty = float(config["representative_selection"]["seen_fingerprint_penalty"])
    selected: list[dict[str, Any]] = []
    selected_cells: set[str] = set()
    selected_fingerprints: set[str] = set()
    ordered = candidates.sort_values("Cell_ID", key=lambda values: values.map(_natural_cell_key))
    for slot, (target_life, target_capacity) in enumerate(targets, start=1):
        best: tuple[float, tuple[Any, ...], pd.Series] | None = None
        for _, row in ordered.iterrows():
            cell = str(row["Cell_ID"])
            if cell in selected_cells:
                continue
            fingerprint = str(row["Protocol_Fingerprint_ID"])
            score = (
                (float(row["Cycle_Life_Rank"]) - float(target_life)) ** 2
                + (float(row["First_Capacity_Rank"]) - float(target_capacity)) ** 2
                + (penalty if fingerprint in selected_fingerprints else 0.0)
            )
            candidate = (score, _natural_cell_key(cell), row)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            break
        score, _, row = best
        cell = str(row["Cell_ID"])
        fingerprint = str(row["Protocol_Fingerprint_ID"])
        selected_cells.add(cell)
        selected_fingerprints.add(fingerprint)
        selected.append(
            {
                "Selection_Order": slot,
                "Cell_ID": cell,
                "Native_Cell_ID": row["Native_Cell_ID"],
                "Source_File": row["Source_File"],
                "Cycle_Count": int(row["Cycle_Count"]),
                "Cycle_Life_Band": (
                    "LOW"
                    if row["Cycle_Life_Rank"] <= 0.20
                    else "LOWER_MIDDLE"
                    if row["Cycle_Life_Rank"] <= 0.40
                    else "MEDIAN"
                    if row["Cycle_Life_Rank"] <= 0.60
                    else "UPPER_MIDDLE"
                    if row["Cycle_Life_Rank"] <= 0.80
                    else "HIGH"
                ),
                "First_Valid_Capacity_Ah": row["First_Valid_Capacity_Ah"],
                "First_Capacity_Band": _quantile_band(float(row["First_Capacity_Rank"])),
                "Protocol_Fingerprint_ID": fingerprint,
                "Target_Life_Quantile": target_life,
                "Target_First_Capacity_Quantile": target_capacity,
                "Selection_Score": score,
                "Selection_Method": config["representative_selection"]["method"],
                "Manual_Decision": "PENDING_HUMAN_REVIEW",
                "Manual_Reason": "",
            }
        )
    return pd.DataFrame(selected)


def _summary_stats(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "min": float(numeric.min()) if not numeric.empty else np.nan,
        "median": float(numeric.median()) if not numeric.empty else np.nan,
        "max": float(numeric.max()) if not numeric.empty else np.nan,
    }


def run_scan(workspace: Path, input_dir: Path, output_dir: Path) -> ScanResult:
    workspace = workspace.resolve()
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"HUST input directory not found: {input_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    config = make_scan_config(workspace, input_dir, output_dir)
    xjtu_final_v2 = workspace / "outputs" / "xjtu_55_final_v2"
    xjtu_hashes_before = _tree_hashes(xjtu_final_v2)
    raw_stat_before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in input_dir.rglob("*")
        if path.is_file()
    }

    print("HUST scan: hashing and classifying input files", flush=True)
    manifest, manifest_issues = _inventory_files(input_dir)
    hard_stop_reasons = list(manifest_issues)
    canonical_rows = manifest.loc[manifest["Included_In_Analysis"].eq("YES")].copy()

    all_cycles: list[pd.DataFrame] = []
    schema_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, Any]] = []
    semantics_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    identity_to_file: dict[str, str] = {}

    total_files = len(canonical_rows)
    for position, (manifest_index, manifest_row) in enumerate(canonical_rows.iterrows(), start=1):
        path = input_dir / str(manifest_row["Relative_Path"])
        print(f"HUST scan: loading {position}/{total_files} {path.name}", flush=True)
        schema_issues: list[str] = []
        processing_status = "SCAN_PROCESSED"
        try:
            payload = load_hust_pickle(path)
        except Exception as exc:
            schema_issues.append(f"RESTRICTED_PICKLE_LOAD_FAILED:{type(exc).__name__}:{exc}")
            processing_status = "STOPPED_SCHEMA_ERROR"
            payload = None

        native_cell_id = ""
        cell_id = ""
        cell_payload: Mapping[str, Any] = {}
        if not isinstance(payload, dict) or len(payload) != 1:
            schema_issues.append("TOP_LEVEL_NOT_SINGLE_CELL_DICT")
            processing_status = "STOPPED_SCHEMA_ERROR"
        else:
            native_cell_id = str(next(iter(payload)))
            cell_id = f"HUST_{native_cell_id}"
            candidate_payload = payload[native_cell_id]
            if isinstance(candidate_payload, dict):
                cell_payload = candidate_payload
            else:
                schema_issues.append("CELL_PAYLOAD_NOT_DICT")
                processing_status = "STOPPED_SCHEMA_ERROR"

        manifest.at[manifest_index, "Native_Cell_ID"] = native_cell_id
        manifest.at[manifest_index, "Cell_ID"] = cell_id
        if native_cell_id:
            previous_file = identity_to_file.get(native_cell_id)
            if previous_file is not None and previous_file != path.name:
                issue = f"CANONICAL_CELL_IDENTITY_CONFLICT:{native_cell_id}:{previous_file}:{path.name}"
                schema_issues.append(issue)
                hard_stop_reasons.append(issue)
            identity_to_file[native_cell_id] = path.name
        if native_cell_id and native_cell_id != path.stem:
            schema_issues.append(
                f"FILENAME_CELL_ID_MISMATCH:{path.stem}:{native_cell_id}"
            )

        required_keys = set(config["required_cell_keys"])
        missing_keys = sorted(required_keys.difference(cell_payload))
        if missing_keys:
            schema_issues.append("MISSING_CELL_KEYS:" + _serialize_values(missing_keys))
            processing_status = "STOPPED_SCHEMA_ERROR"

        data = cell_payload.get("data", {}) if cell_payload else {}
        dq = cell_payload.get("dq", {}) if cell_payload else {}
        rul = cell_payload.get("rul", {}) if cell_payload else {}
        if not isinstance(data, dict) or not isinstance(dq, dict) or not isinstance(rul, dict):
            schema_issues.append("DATA_DQ_RUL_NOT_DICT")
            processing_status = "STOPPED_SCHEMA_ERROR"
            data = data if isinstance(data, dict) else {}
            dq = dq if isinstance(dq, dict) else {}
            rul = rul if isinstance(rul, dict) else {}

        def numeric_keys(mapping: Mapping[Any, Any], label: str) -> list[int]:
            result: list[int] = []
            for key in mapping:
                number = pd.to_numeric(key, errors="coerce")
                if not np.isfinite(number) or float(number) != int(number):
                    schema_issues.append(f"NONINTEGER_{label}_CYCLE_KEY:{key}")
                    continue
                result.append(int(number))
            return result

        data_keys = numeric_keys(data, "DATA")
        dq_keys = numeric_keys(dq, "DQ")
        rul_keys = numeric_keys(rul, "RUL")
        union_keys = sorted(set(data_keys) | set(dq_keys) | set(rul_keys))
        cycle_status = _cycle_index_status(union_keys, data_keys)
        if set(data_keys) != set(dq_keys) or set(data_keys) != set(rul_keys):
            schema_issues.append("DATA_DQ_RUL_CYCLE_KEY_MISMATCH")
            processing_status = "STOPPED_SCHEMA_ERROR"

        cycle_records: list[dict[str, Any]] = []
        raw_columns: list[str] = []
        raw_dtypes: list[str] = []
        status_union: set[str] = set()
        per_cycle_column_variants: set[tuple[str, ...]] = set()
        for raw_cycle in union_keys:
            frame = data.get(raw_cycle)
            if frame is None:
                frame = data.get(np.int64(raw_cycle))
            row = _cycle_metrics(
                cell_id,
                native_cell_id,
                path.name,
                raw_cycle,
                frame,
                dq.get(raw_cycle),
                rul.get(raw_cycle),
                config,
            )
            cycle_records.append(row)
            if isinstance(frame, pd.DataFrame):
                columns = tuple(str(column) for column in frame.columns)
                per_cycle_column_variants.add(columns)
                if not raw_columns:
                    raw_columns = list(columns)
                    raw_dtypes = [str(dtype) for dtype in frame.dtypes]
                if config["raw_column_map"]["status"] in frame.columns:
                    status_union.update(
                        frame[config["raw_column_map"]["status"]]
                        .astype(str)
                        .unique()
                        .tolist()
                    )

        cycles = pd.DataFrame(cycle_records)
        if not cycles.empty:
            cycles, fingerprint = _apply_protocol_and_jump_qc(cycles, config)
            fingerprint.update(
                {
                    "Cell_ID": cell_id,
                    "Native_Cell_ID": native_cell_id,
                    "Source_File": path.name,
                }
            )
            fingerprint_rows.append(fingerprint)
            all_cycles.append(cycles)
        else:
            fingerprint = {
                "Cell_ID": cell_id,
                "Native_Cell_ID": native_cell_id,
                "Source_File": path.name,
                "Protocol_Fingerprint_ID": "UNAVAILABLE",
                "Protocol_Fingerprint_Status": "UNAVAILABLE_NO_CYCLES",
                "Protocol_Change_Candidate_Cycle_Count": 0,
            }
            fingerprint_rows.append(fingerprint)

        if len(per_cycle_column_variants) > 1:
            schema_issues.append("WITHIN_FILE_RAW_COLUMN_VARIANTS")
        schema_variant = _schema_signature(
            list(cell_payload.keys()), raw_columns, raw_dtypes, sorted(status_union)
        )
        if processing_status == "STOPPED_SCHEMA_ERROR":
            hard_stop_reasons.append(f"SCHEMA_ERROR:{path.name}")

        schema_rows.append(
            {
                "Cell_ID": cell_id,
                "Native_Cell_ID": native_cell_id,
                "Source_File": path.name,
                "Top_Level_Type": type(payload).__name__ if payload is not None else "LOAD_FAILED",
                "Top_Level_Key_Count": len(payload) if isinstance(payload, dict) else np.nan,
                "Top_Level_Keys": _serialize_values(payload.keys()) if isinstance(payload, dict) else "",
                "Cell_Payload_Type": type(cell_payload).__name__,
                "Cell_Payload_Keys": _serialize_values(cell_payload.keys()),
                "Data_Type": type(data).__name__,
                "DQ_Type": type(dq).__name__,
                "RUL_Type": type(rul).__name__,
                "Data_Cycle_Count": len(data),
                "DQ_Label_Count": len(dq),
                "RUL_Label_Count": len(rul),
                "Raw_Columns": _serialize_values(raw_columns),
                "Raw_Dtypes": _serialize_values(raw_dtypes),
                "Observed_Status_Values": _serialize_values(sorted(status_union)),
                "Raw_Column_Variant_Count": len(per_cycle_column_variants),
                "Schema_Variant_ID": schema_variant,
                "Schema_Status": "ERROR" if processing_status == "STOPPED_SCHEMA_ERROR" else ("REVIEW" if schema_issues else "PASS"),
                "Schema_Issues": "|".join(schema_issues),
                "Processing_Status": processing_status,
            }
        )

        valid = cycles.loc[
            cycles["Primary_Capacity_Candidate"].eq("YES")
            & pd.to_numeric(cycles["Primary_Capacity_Ah"], errors="coerce").notna()
        ] if not cycles.empty else cycles
        first_valid = valid.iloc[0] if not valid.empty else None
        last_valid = valid.iloc[-1] if not valid.empty else None
        semantics_fraction = (
            float(cycles["Capacity_Semantics_Match"].eq("YES").mean())
            if not cycles.empty
            else np.nan
        )
        traceable_fraction = (
            float(cycles["Full_Discharge_Traceable"].eq("YES").mean())
            if not cycles.empty
            else np.nan
        )
        cell_rows.append(
            {
                "Cell_ID": cell_id,
                "Native_Cell_ID": native_cell_id,
                "Source_File": path.name,
                "Cycle_Count": len(union_keys),
                **cycle_status,
                "Cycle_Life_Expected_Range_Status": (
                    "WITHIN_EXPECTED_RANGE"
                    if int(config["expected_cycle_life_min"]) <= len(union_keys) <= int(config["expected_cycle_life_max"])
                    else "OUTSIDE_EXPECTED_RANGE"
                ),
                "Has_Raw_Curve_Data": "YES" if bool(data) else "NO",
                "Has_Native_Discharge_Capacity_Label": "YES" if bool(dq) else "NO",
                "Has_Native_RUL_Label": "YES" if bool(rul) else "NO",
                "Has_Native_SOH_Label": "NO",
                "Capacity_Semantics_Match_Fraction": semantics_fraction,
                "Full_Discharge_Traceable_Fraction": traceable_fraction,
                "First_Valid_Raw_Cycle": first_valid["Raw_Cycle_Index"] if first_valid is not None else np.nan,
                "First_Valid_Capacity_Ah": first_valid["Primary_Capacity_Ah"] if first_valid is not None else np.nan,
                "Last_Valid_Raw_Cycle": last_valid["Raw_Cycle_Index"] if last_valid is not None else np.nan,
                "Last_Valid_Capacity_Ah": last_valid["Primary_Capacity_Ah"] if last_valid is not None else np.nan,
                "Protocol_Fingerprint_ID": fingerprint.get("Protocol_Fingerprint_ID", "UNAVAILABLE"),
                "Protocol_Fingerprint_Status": fingerprint.get("Protocol_Fingerprint_Status", "UNAVAILABLE"),
                "Protocol_Change_Candidate": "YES" if int(fingerprint.get("Protocol_Change_Candidate_Cycle_Count", 0)) > 0 else "NO",
                "Capacity_Semantics_Status": (
                    "VERIFIED_FULL_CYCLE_DISCHARGE_LABEL"
                    if np.isfinite(semantics_fraction)
                    and semantics_fraction >= float(config["scan_thresholds"]["capacity_semantics_required_match_fraction"])
                    else "CAPACITY_SEMANTICS_UNRESOLVED"
                ),
                "Initial_Test_Qref_Feasibility": "UNAVAILABLE_NO_SEPARATE_STANDARDIZED_BOL_TEST",
                "First_Comparable_Qref_Feasibility": "FEASIBLE" if len(valid) >= 1 else "UNAVAILABLE",
                "Early_Median5_Qref_Feasibility": "FEASIBLE" if len(valid) >= 5 else "UNAVAILABLE_FEWER_THAN_5",
                "Schema_Variant_ID": schema_variant,
                "Schema_Status": "ERROR" if processing_status == "STOPPED_SCHEMA_ERROR" else ("REVIEW" if schema_issues else "PASS"),
                "Processing_Status": processing_status,
            }
        )

        if not cycles.empty:
            selected_positions = sorted(set([0, len(cycles) // 2, len(cycles) - 1]))
            for audit_order, cycle_position in enumerate(selected_positions, start=1):
                row = cycles.iloc[cycle_position]
                semantics_rows.append(
                    {
                        "Cell_ID": cell_id,
                        "Source_File": path.name,
                        "Audit_Order": audit_order,
                        "Representative_Position": (
                            "FIRST" if cycle_position == 0 else "LAST" if cycle_position == len(cycles) - 1 else "MIDDLE"
                        ),
                        "Raw_Cycle_Index": row["Raw_Cycle_Index"],
                        "Capacity_Field_Candidate": "dq",
                        "Capacity_Field_Unit": "mAh",
                        "Raw_Capacity_Channel": "Capacity (mAh)",
                        "Raw_Capacity_Channel_Semantics": "WITHIN_CYCLE_CUMULATIVE_CHARGE_THEN_DECREASING_DISCHARGE_COUNTER",
                        "Raw_Crosscheck_Method": "capacity_at_discharge_0_start_minus_capacity_at_discharge_3_end",
                        "Native_Discharge_Capacity_mAh": row["Native_Discharge_Capacity_mAh"],
                        "Raw_Full_Discharge_Capacity_mAh": row["Raw_Full_Discharge_Capacity_mAh"],
                        "Capacity_Difference_mAh": row["Capacity_Difference_mAh"],
                        "Capacity_Difference_Pct": row["Capacity_Difference_Pct"],
                        "Discharge_Step1_Capacity_mAh": row["Discharge_Step1_Capacity_mAh"],
                        "Discharge_Step2_Capacity_mAh": row["Discharge_Step2_Capacity_mAh"],
                        "Discharge_Step3_Capacity_mAh": row["Discharge_Step3_Capacity_mAh"],
                        "Discharge_Step4_Capacity_mAh": row["Discharge_Step4_Capacity_mAh"],
                        "Full_Discharge_Traceable": row["Full_Discharge_Traceable"],
                        "Terminal_Discharge_End_V": row["Discharge_End_V"],
                        "Capacity_Semantics_Match": row["Capacity_Semantics_Match"],
                        "Capacity_Semantics_Conclusion": (
                            "FULL_CYCLE_DISCHARGE_CAPACITY_LABEL_VERIFIED"
                            if row["Capacity_Semantics_Match"] == "YES"
                            else "CAPACITY_SEMANTICS_UNRESOLVED"
                        ),
                        "Primary_Capacity_Ah_Scan_Candidate": row["Primary_Capacity_Ah"],
                    }
                )

        rul_values = pd.Series([_finite_number(value) for value in rul.values()]).dropna()
        dq_values = pd.Series([_finite_number(value) for value in dq.values()]).dropna()
        expected_rul = {
            cycle: len(union_keys) - cycle + 1 for cycle in union_keys
        }
        rul_formula_match = bool(
            rul_keys
            and all(
                np.isclose(_finite_number(rul.get(cycle)), expected_rul[cycle])
                for cycle in rul_keys
                if cycle in expected_rul
            )
        )
        native_rows.append(
            {
                "Cell_ID": cell_id,
                "Source_File": path.name,
                "Native_Discharge_Capacity_Field": "dq",
                "Native_Discharge_Capacity_Present": "YES" if bool(dq) else "NO",
                "Native_Discharge_Capacity_Count": len(dq_values),
                "Native_Discharge_Capacity_Min_mAh": dq_values.min() if not dq_values.empty else np.nan,
                "Native_Discharge_Capacity_Max_mAh": dq_values.max() if not dq_values.empty else np.nan,
                "Native_RUL_Field": "rul",
                "Native_RUL_Present": "YES" if bool(rul) else "NO",
                "Native_RUL_Count": len(rul_values),
                "Native_RUL_Min": rul_values.min() if not rul_values.empty else np.nan,
                "Native_RUL_Max": rul_values.max() if not rul_values.empty else np.nan,
                "Native_RUL_Equals_CycleLife_Minus_Cycle_Plus_One": "YES" if rul_formula_match else "NO",
                "Native_SOH_Field": "",
                "Native_SOH_Present": "NO",
                "Native_Cycle_Life_Field": "IMPLIED_BY_RUL_AND_MAX_RAW_CYCLE",
                "Native_Label_Use": "AUDIT_ONLY_NOT_GROUND_TRUTH",
                "Native_Labels_Used_For_Study_EOL": "NO",
                "Native_Labels_Used_For_Study_RUL": "NO",
            }
        )

        del payload, cell_payload, data, dq, rul, cycles
        gc.collect()

    cycle_inventory = (
        pd.concat(all_cycles, ignore_index=True)
        if all_cycles
        else pd.DataFrame()
    )
    if not cycle_inventory.empty:
        cycle_inventory = cycle_inventory.sort_values(
            ["Cell_ID", "Raw_Cycle_Index"],
            key=lambda column: column.map(_natural_cell_key) if column.name == "Cell_ID" else column,
            kind="stable",
        ).reset_index(drop=True)
    schema_inventory = pd.DataFrame(schema_rows).sort_values(
        "Cell_ID", key=lambda values: values.map(_natural_cell_key), kind="stable"
    ).reset_index(drop=True)
    cell_inventory = pd.DataFrame(cell_rows).sort_values(
        "Cell_ID", key=lambda values: values.map(_natural_cell_key), kind="stable"
    ).reset_index(drop=True)
    fingerprint_inventory = pd.DataFrame(fingerprint_rows).sort_values(
        "Cell_ID", key=lambda values: values.map(_natural_cell_key), kind="stable"
    ).reset_index(drop=True)
    semantics_audit = pd.DataFrame(semantics_rows).sort_values(
        ["Cell_ID", "Audit_Order"],
        key=lambda column: column.map(_natural_cell_key) if column.name == "Cell_ID" else column,
        kind="stable",
    ).reset_index(drop=True)
    native_audit = pd.DataFrame(native_rows).sort_values(
        "Cell_ID", key=lambda values: values.map(_natural_cell_key), kind="stable"
    ).reset_index(drop=True)

    canonical_cell_count = int(cell_inventory["Cell_ID"].replace("", np.nan).nunique())
    if canonical_cell_count != int(config["expected_canonical_cell_count"]):
        cell_count_issue = (
            f"CELL_COUNT_MISMATCH:expected={config['expected_canonical_cell_count']}:observed={canonical_cell_count}"
        )
        manifest_issues.append(cell_count_issue)
    semantics_global = float(
        cycle_inventory["Capacity_Semantics_Match"].eq("YES").mean()
    ) if not cycle_inventory.empty else 0.0
    traceable_global = float(
        cycle_inventory["Full_Discharge_Traceable"].eq("YES").mean()
    ) if not cycle_inventory.empty else 0.0
    protocol_change_cells = int(
        cell_inventory["Protocol_Change_Candidate"].eq("YES").sum()
    )
    if semantics_global < float(config["scan_thresholds"]["capacity_semantics_required_match_fraction"]):
        hard_stop_reasons.append(
            f"CAPACITY_SEMANTICS_UNRESOLVED:match_fraction={semantics_global:.6f}"
        )
    if traceable_global < float(config["scan_thresholds"]["full_discharge_required_traceable_fraction"]):
        hard_stop_reasons.append(
            f"FULL_DISCHARGE_NOT_TRACEABLE:traceable_fraction={traceable_global:.6f}"
        )
    if canonical_cell_count and (
        protocol_change_cells / canonical_cell_count
        > float(config["scan_thresholds"]["protocol_change_hard_stop_cell_fraction"])
    ):
        hard_stop_reasons.append(
            f"LARGE_UNEXPLAINED_PROTOCOL_CHANGE_FRACTION:{protocol_change_cells}/{canonical_cell_count}"
        )

    anomalies: list[dict[str, Any]] = []
    for row in cycle_inventory.loc[
        cycle_inventory["Auto_QC_Flag"].ne("PASS")
        | cycle_inventory["Protocol_Current_Deviation"].eq("YES")
    ].itertuples(index=False):
        codes = str(row.Auto_QC_Reason_Codes)
        if codes == "PASS" and row.Protocol_Current_Deviation == "YES":
            codes = "ISOLATED_PROTOCOL_CURRENT_DEVIATION_INFO"
        anomalies.append(
            {
                "Cell_ID": row.Cell_ID,
                "Source_File": row.Source_File,
                "Raw_Cycle_Index": row.Raw_Cycle_Index,
                "Anomaly_Level": "CYCLE",
                "Auto_QC_Flag": row.Auto_QC_Flag,
                "Anomaly_Codes": codes,
                "Requires_Human_Review": "YES" if row.Auto_QC_Flag != "PASS" else "NO",
                "Primary_Capacity_Candidate": row.Primary_Capacity_Candidate,
                "Details": (
                    f"capacity_Ah={row.Primary_Capacity_Ah};"
                    f"discharge_end_V={row.Discharge_End_V};"
                    f"jump_Ah={row.Capacity_Jump_Ah}"
                ),
            }
        )
    for row in schema_inventory.loc[schema_inventory["Schema_Status"].ne("PASS")].itertuples(index=False):
        anomalies.append(
            {
                "Cell_ID": row.Cell_ID,
                "Source_File": row.Source_File,
                "Raw_Cycle_Index": np.nan,
                "Anomaly_Level": "FILE_SCHEMA",
                "Auto_QC_Flag": "INVALID_CANDIDATE" if row.Schema_Status == "ERROR" else "CHECK",
                "Anomaly_Codes": row.Schema_Issues,
                "Requires_Human_Review": "YES",
                "Primary_Capacity_Candidate": "NO" if row.Schema_Status == "ERROR" else "REVIEW",
                "Details": f"schema_variant={row.Schema_Variant_ID}",
            }
        )
    protocol_anomalies = pd.DataFrame(
        anomalies,
        columns=[
            "Cell_ID",
            "Source_File",
            "Raw_Cycle_Index",
            "Anomaly_Level",
            "Auto_QC_Flag",
            "Anomaly_Codes",
            "Requires_Human_Review",
            "Primary_Capacity_Candidate",
            "Details",
        ],
    )
    if not protocol_anomalies.empty:
        protocol_anomalies = protocol_anomalies.sort_values(
            ["Cell_ID", "Raw_Cycle_Index"],
            key=lambda column: column.map(_natural_cell_key) if column.name == "Cell_ID" else column,
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)

    stopping_rows: list[dict[str, Any]] = []
    fixed_threshold = float(config["fixed_absolute_EOL_threshold_Ah"])
    near_tolerance = float(config["scan_thresholds"]["stopping_rule_near_0p88_abs_tolerance_Ah"])
    for cell, rows in cycle_inventory.groupby("Cell_ID", sort=False):
        valid = rows.loc[
            rows["Primary_Capacity_Candidate"].eq("YES")
            & pd.to_numeric(rows["Primary_Capacity_Ah"], errors="coerce").notna()
        ].sort_values("Raw_Cycle_Index", kind="stable")
        if valid.empty:
            last = None
            capacity = np.nan
        else:
            last = valid.iloc[-1]
            capacity = float(last["Primary_Capacity_Ah"])
        stopping_rows.append(
            {
                "Cell_ID": cell,
                "Source_File": rows["Source_File"].iloc[0],
                "Last_Valid_Raw_Cycle": last["Raw_Cycle_Index"] if last is not None else np.nan,
                "Last_Valid_Capacity_Ah": capacity,
                "Fixed_Absolute_Threshold_Ah": fixed_threshold,
                "Last_Capacity_vs_0p88_Ah": capacity - fixed_threshold if np.isfinite(capacity) else np.nan,
                "Last_Capacity_LE_0p88_Ah": "YES" if np.isfinite(capacity) and capacity <= fixed_threshold else "NO",
                "Last_Capacity_GT_0p88_Ah": "YES" if np.isfinite(capacity) and capacity > fixed_threshold else "NO",
                "Native_End_Near_Nominal_80pct": "YES" if np.isfinite(capacity) and abs(capacity - fixed_threshold) <= near_tolerance else "NO",
                "Dataset_Stopping_Rule_Audit_Status": "TRACEABLE" if last is not None else "UNAVAILABLE_NO_VALID_CAPACITY",
                "Interpretation_Note": "Dataset design may stop near 80% nominal; do not present nominal EOL as an independent discovery.",
            }
        )
    stopping_audit = pd.DataFrame(stopping_rows).sort_values(
        "Cell_ID", key=lambda values: values.map(_natural_cell_key), kind="stable"
    ).reset_index(drop=True)
    representative_plan = _select_representatives(cell_inventory, config)

    hard_stop_reasons = sorted(set(hard_stop_reasons))
    scan_status = (
        "HUST_SCAN_NEEDS_REVIEW"
        if hard_stop_reasons
        else "HUST_SCAN_COMPLETE_WAITING_FOR_HUMAN_REVIEW"
    )
    config["scan_result"] = {
        "scan_status": scan_status,
        "hard_stop_reasons": hard_stop_reasons,
        "cell_count_mismatch": canonical_cell_count != int(config["expected_canonical_cell_count"]),
        "run_gate": "BLOCKED_PENDING_HUMAN_REVIEW",
        "validate_executed": False,
        "run_executed": False,
        "model_training_executed": False,
    }

    raw_stat_after = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in raw_stat_before
    }
    raw_stats_unchanged = raw_stat_before == raw_stat_after
    xjtu_unchanged = _tree_hashes(xjtu_final_v2) == xjtu_hashes_before
    if not raw_stats_unchanged:
        raise RuntimeError("RAW_HUST_FILE_STAT_CHANGED_DURING_SCAN")
    if not xjtu_unchanged:
        raise RuntimeError("FROZEN_XJTU_FINAL_V2_CHANGED_DURING_HUST_SCAN")

    temp_dir = _atomic_directory(output_dir)
    try:
        outputs = {
            "hust_file_manifest.csv": manifest,
            "hust_schema_inventory.csv": schema_inventory,
            "hust_cell_inventory.csv": cell_inventory,
            "hust_cycle_inventory.csv": cycle_inventory,
            "hust_protocol_fingerprint.csv": fingerprint_inventory,
            "hust_capacity_semantics_audit.csv": semantics_audit,
            "hust_native_label_audit.csv": native_audit,
            "hust_protocol_anomalies.csv": protocol_anomalies,
            "hust_dataset_stopping_rule_audit.csv": stopping_audit,
            "representative_validation_plan.csv": representative_plan,
        }
        for name, frame in outputs.items():
            _write_csv(frame, temp_dir / name)
        (temp_dir / "hust_scan_config_draft.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        first_stats = _summary_stats(cell_inventory["First_Valid_Capacity_Ah"])
        last_stats = _summary_stats(cell_inventory["Last_Valid_Capacity_Ah"])
        file_type_counts = manifest["Extension"].value_counts().sort_index().to_dict()
        log_lines = [
            f"Pipeline_Version={PIPELINE_VERSION}",
            f"Python_Version={sys.version.replace(chr(10), ' ')}",
            f"Platform={platform.platform()}",
            f"NumPy_Version={np.__version__}",
            f"pandas_Version={pd.__version__}",
            f"Scan_Timestamp_UTC={datetime.now(timezone.utc).isoformat()}",
            f"Workspace={workspace}",
            f"Input_Directory={input_dir}",
            f"Output_Directory={output_dir}",
            f"Input_File_Count={len(manifest)}",
            f"Input_Total_Bytes={int(pd.to_numeric(manifest['File_Size_Bytes']).sum())}",
            "Input_File_Types=" + "|".join(f"{key}:{value}" for key, value in file_type_counts.items()),
            f"Canonical_Cell_Count={canonical_cell_count}",
            f"Total_Cycle_Count={len(cycle_inventory)}",
            f"Schema_Variant_Count={schema_inventory['Schema_Variant_ID'].nunique()}",
            "Capacity_Field_Candidates=dq|Capacity (mAh)",
            f"Capacity_Semantics_Global_Match_Fraction={semantics_global:.12g}",
            f"Full_Discharge_Traceable_Global_Fraction={traceable_global:.12g}",
            f"Cells_With_Raw_Curve_Data={int(cell_inventory['Has_Raw_Curve_Data'].eq('YES').sum())}",
            f"Cells_With_Native_SOH={int(cell_inventory['Has_Native_SOH_Label'].eq('YES').sum())}",
            f"Cells_With_Native_RUL={int(cell_inventory['Has_Native_RUL_Label'].eq('YES').sum())}",
            f"Protocol_Fingerprint_Count={fingerprint_inventory['Protocol_Fingerprint_ID'].nunique()}",
            f"Cells_With_Protocol_Change_Candidate={protocol_change_cells}",
            f"Incomplete_Discharge_Candidate_Count={int(cycle_inventory['Auto_QC_Reason_Codes'].str.contains('INCOMPLETE_TERMINAL_DISCHARGE_CANDIDATE', regex=False).sum())}",
            f"Capacity_Jump_CHECK_Candidate_Count={int(cycle_inventory['Capacity_Jump_Flag'].eq('YES').sum())}",
            f"First_Capacity_Min_Ah={first_stats['min']:.12g}",
            f"First_Capacity_Median_Ah={first_stats['median']:.12g}",
            f"First_Capacity_Max_Ah={first_stats['max']:.12g}",
            f"Last_Capacity_Min_Ah={last_stats['min']:.12g}",
            f"Last_Capacity_Median_Ah={last_stats['median']:.12g}",
            f"Last_Capacity_Max_Ah={last_stats['max']:.12g}",
            f"Final_Capacity_LE_0p88_Count={int(stopping_audit['Last_Capacity_LE_0p88_Ah'].eq('YES').sum())}",
            f"Final_Capacity_GT_0p88_Count={int(stopping_audit['Last_Capacity_GT_0p88_Ah'].eq('YES').sum())}",
            "Representative_Cells=" + "|".join(representative_plan["Cell_ID"].astype(str)),
            "Restricted_Pickle_Global_Whitelist=" + "|".join(
                f"{module}.{name}" for module, name in sorted(ALLOWED_PICKLE_GLOBALS)
            ),
            "Raw_HUST_Files_Modified=NO",
            f"Raw_HUST_File_Stats_Unchanged={'YES' if raw_stats_unchanged else 'NO'}",
            "XJTU_Final_V2_Modified=NO",
            f"XJTU_Final_V2_Hashes_Unchanged={'YES' if xjtu_unchanged else 'NO'}",
            "Smoothing_Performed=NO",
            "Interpolation_Performed=NO",
            "Capacity_Correction_Performed=NO",
            "Cycle_Renumbering_Performed=NO",
            "Validate_Executed=NO",
            "Run_Executed=NO",
            "Model_Training_Executed=NO",
            f"Hard_Stop_Reason_Count={len(hard_stop_reasons)}",
            *[f"Hard_Stop_Reason={reason}" for reason in hard_stop_reasons],
            f"Scan_Status={scan_status}",
            "Run_Gate=BLOCKED_PENDING_HUMAN_REVIEW",
            (
                "HUST SCAN NEEDS REVIEW"
                if scan_status == "HUST_SCAN_NEEDS_REVIEW"
                else "HUST SCAN COMPLETE — WAITING FOR HUMAN REVIEW"
            ),
        ]
        (temp_dir / "processing_log_scan.txt").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8"
        )

        expected_names = set(outputs) | {
            "hust_scan_config_draft.json",
            "processing_log_scan.txt",
        }
        actual_names = {path.name for path in temp_dir.iterdir() if path.is_file()}
        if actual_names != expected_names:
            raise RuntimeError(
                f"HUST_SCAN_OUTPUT_FILE_SET_MISMATCH:{sorted(actual_names)}"
            )

        # Read-back validation before committing the atomic output directory.
        read_cycle = pd.read_csv(
            temp_dir / "hust_cycle_inventory.csv",
            keep_default_na=False,
            low_memory=False,
        )
        read_cell = pd.read_csv(
            temp_dir / "hust_cell_inventory.csv", keep_default_na=False
        )
        read_manifest = pd.read_csv(
            temp_dir / "hust_file_manifest.csv", keep_default_na=False
        )
        if read_cycle.duplicated(["Cell_ID", "Raw_Cycle_Index"]).any():
            raise RuntimeError("HUST_SCAN_DUPLICATE_CYCLE_KEY_AFTER_WRITE")
        if read_cell["Cell_ID"].astype(str).str.strip().eq("").any():
            raise RuntimeError("HUST_SCAN_BLANK_CELL_ID_AFTER_WRITE")
        if int(read_manifest["Included_In_Analysis"].eq("YES").sum()) != total_files:
            raise RuntimeError("HUST_SCAN_CANONICAL_FILE_COUNT_CHANGED_AFTER_WRITE")
        temp_dir.rename(output_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    if _tree_hashes(xjtu_final_v2) != xjtu_hashes_before:
        raise RuntimeError("FROZEN_XJTU_FINAL_V2_CHANGED_AFTER_HUST_SCAN")
    for path, expected in raw_stat_before.items():
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != expected:
            raise RuntimeError(f"RAW_HUST_FILE_CHANGED_AFTER_SCAN:{path.name}")

    first_stats = _summary_stats(cell_inventory["First_Valid_Capacity_Ah"])
    last_stats = _summary_stats(cell_inventory["Last_Valid_Capacity_Ah"])
    return ScanResult(
        output_dir=output_dir,
        scan_status=scan_status,
        canonical_cell_count=canonical_cell_count,
        total_cycle_count=len(cycle_inventory),
        file_type_counts={
            str(key): int(value)
            for key, value in manifest["Extension"].value_counts().sort_index().items()
        },
        schema_variant_count=int(schema_inventory["Schema_Variant_ID"].nunique()),
        capacity_field_candidates=("dq", "Capacity (mAh)"),
        cells_with_raw_curves=int(cell_inventory["Has_Raw_Curve_Data"].eq("YES").sum()),
        cells_with_native_soh=int(cell_inventory["Has_Native_SOH_Label"].eq("YES").sum()),
        cells_with_native_rul=int(cell_inventory["Has_Native_RUL_Label"].eq("YES").sum()),
        protocol_fingerprint_count=int(fingerprint_inventory["Protocol_Fingerprint_ID"].nunique()),
        protocol_change_cell_count=protocol_change_cells,
        incomplete_discharge_candidate_count=int(
            cycle_inventory["Auto_QC_Reason_Codes"]
            .str.contains("INCOMPLETE_TERMINAL_DISCHARGE_CANDIDATE", regex=False)
            .sum()
        ),
        capacity_jump_check_count=int(cycle_inventory["Capacity_Jump_Flag"].eq("YES").sum()),
        first_capacity_stats=first_stats,
        last_capacity_stats=last_stats,
        final_capacity_le_0p88_count=int(stopping_audit["Last_Capacity_LE_0p88_Ah"].eq("YES").sum()),
        final_capacity_gt_0p88_count=int(stopping_audit["Last_Capacity_GT_0p88_Ah"].eq("YES").sum()),
        representative_cells=tuple(representative_plan["Cell_ID"].astype(str)),
        hard_stop_reasons=tuple(hard_stop_reasons),
    )


__all__ = [
    "ScanResult",
    "_apply_protocol_and_jump_qc",
    "_cycle_index_status",
    "_cycle_metrics",
    "_select_representatives",
    "run_scan",
]
