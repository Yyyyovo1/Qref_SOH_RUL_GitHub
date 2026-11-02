from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .config import make_protocol_validation_config
from .safe_pickle import load_hust_pickle


FINAL_ASSESSMENT_PARSER_BUG = "PARSER_STAGE_MAPPING_BUG"
FINAL_ASSESSMENT_RAW_DIFFERS = "RAW_DATA_PROTOCOL_DIFFERS_FROM_PAPER"
FINAL_ASSESSMENT_UNRESOLVED = "PROTOCOL_DISCREPANCY_UNRESOLVED"

ALLOWED_PHYSICAL_FUNCTIONS = {
    "CHARGE_CC_HIGH_RATE",
    "CHARGE_CC_LOW_RATE",
    "CHARGE_CV",
    "REST",
    "DISCHARGE_STAGE_1",
    "DISCHARGE_STAGE_2",
    "DISCHARGE_STAGE_3",
    "DISCHARGE_STAGE_4",
    "OTHER",
    "UNKNOWN",
}


@dataclass(frozen=True)
class ProtocolValidationResult:
    output_dir: Path
    selected_cycle_count: int
    raw_step_count: int
    plot_count: int
    final_assessment: str
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


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _safe_stat(values: np.ndarray, operation: str) -> float:
    clean = _finite(values)
    if not clean.size:
        return np.nan
    operations = {
        "first": lambda x: x[0],
        "last": lambda x: x[-1],
        "median": np.median,
        "mean": np.mean,
        "min": np.min,
        "max": np.max,
    }
    return float(operations[operation](clean))


def _lookup(mapping: Mapping[Any, Any], cycle: int) -> Any:
    if cycle in mapping:
        return mapping[cycle]
    text = str(cycle)
    if text in mapping:
        return mapping[text]
    for key, value in mapping.items():
        try:
            if int(key) == cycle:
                return value
        except (TypeError, ValueError):
            continue
    raise KeyError(cycle)


def _integer_cycle_keys(mapping: Mapping[Any, Any]) -> list[int]:
    result: list[int] = []
    for key in mapping:
        number = pd.to_numeric(key, errors="coerce")
        if np.isfinite(number) and float(number).is_integer():
            result.append(int(number))
    return sorted(set(result))


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def select_validation_cycles(cycle_keys: Sequence[int]) -> list[dict[str, Any]]:
    """Select fixed initial, second, half-up middle, and final raw cycles."""

    cycles = sorted(set(int(value) for value in cycle_keys))
    if len(cycles) < 4:
        raise ValueError("At least four unique raw cycles are required")
    first = cycles[0]
    last = cycles[-1]
    target_middle = _round_half_up((1 + last) / 2.0)

    def nearest(target: int) -> int:
        return min(cycles, key=lambda cycle: (abs(cycle - target), cycle))

    choices = [
        ("INITIAL", first, 1, "RAW_CYCLE_1"),
        ("SECOND", nearest(2), 2, "RAW_CYCLE_2"),
        (
            "MIDDLE",
            nearest(target_middle),
            target_middle,
            f"ROUND_HALF_UP_OF_(1+LAST)/2_TARGET_{target_middle}",
        ),
        ("FINAL", last, last, "LAST_RAW_CYCLE_INDEX"),
    ]
    selected = [choice[1] for choice in choices]
    if len(set(selected)) != 4:
        raise ValueError(f"Selected cycles are not unique: {selected}")
    return [
        {
            "Cycle_Position": position,
            "Raw_Cycle_Index": cycle,
            "Target_Raw_Cycle_Index": target,
            "Reason_Selected": reason
            if cycle == target
            else f"{reason};TARGET_ABSENT_NEAREST_ACTUAL_{cycle}",
        }
        for position, cycle, target, reason in choices
    ]


def _base_segments(statuses: np.ndarray) -> list[dict[str, Any]]:
    if not len(statuses):
        return []
    boundaries = np.flatnonzero(statuses[1:] != statuses[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(statuses)]
    return [
        {
            "status": str(statuses[int(start)]),
            "indices": np.arange(int(start), int(end), dtype=int),
        }
        for start, end in zip(starts, ends)
    ]


def _split_cccv(
    indices: np.ndarray,
    current_A: np.ndarray,
    voltage_V: np.ndarray,
    thresholds: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if indices.size < 2:
        return indices, np.asarray([], dtype=int)
    segment_current = current_A[indices]
    segment_voltage = voltage_V[indices]
    plateau_values = segment_current[
        np.isfinite(segment_current) & (segment_current >= 0.5)
    ]
    plateau = float(np.median(plateau_values)) if plateau_values.size else np.nan
    maximum_voltage = _safe_stat(segment_voltage, "max")
    if not np.isfinite(plateau) or not np.isfinite(maximum_voltage):
        return indices, np.asarray([], dtype=int)
    candidates = np.flatnonzero(
        np.isfinite(segment_current)
        & np.isfinite(segment_voltage)
        & (
            segment_current
            <= plateau * float(thresholds["cv_transition_current_fraction"])
        )
        & (
            segment_voltage
            >= maximum_voltage
            - float(thresholds["cv_transition_voltage_below_max_V"])
        )
    )
    if not candidates.size or int(candidates[0]) == 0:
        return indices, np.asarray([], dtype=int)
    split = int(candidates[0])
    return indices[:split], indices[split:]


def _step_mode(physical_function: str) -> str:
    if physical_function in {"CHARGE_CC_HIGH_RATE", "CHARGE_CC_LOW_RATE"}:
        return "CC_CHARGE"
    if physical_function == "CHARGE_CV":
        return "CV_CHARGE"
    if physical_function.startswith("DISCHARGE_STAGE_"):
        return "CC_DISCHARGE"
    if physical_function == "REST":
        return "REST"
    return "OTHER_OR_UNKNOWN"


def _segment_row(
    cell_id: str,
    raw_cycle: int,
    cycle_position: str,
    source_file: str,
    order: int,
    status: str,
    raw_step_label: str,
    physical_function: str,
    indices: np.ndarray,
    time_s: np.ndarray,
    current_A: np.ndarray,
    voltage_V: np.ndarray,
    capacity_mAh: np.ndarray,
    nominal_capacity_Ah: float,
) -> dict[str, Any]:
    assert physical_function in ALLOWED_PHYSICAL_FUNCTIONS
    current = current_A[indices]
    voltage = voltage_V[indices]
    capacity = capacity_mAh[indices]
    times = time_s[indices]
    median_current = _safe_stat(current, "median")
    start_capacity = _safe_stat(capacity, "first")
    end_capacity = _safe_stat(capacity, "last")
    return {
        "Cell_ID": cell_id,
        "Source_File": source_file,
        "Raw_Cycle_Index": raw_cycle,
        "Cycle_Position": cycle_position,
        "Physical_Step_Order": order,
        "Raw_Status_Label": status,
        "Raw_Step_Label": raw_step_label,
        "Raw_Mode_Label": _step_mode(physical_function),
        "Record_Count": int(indices.size),
        "Start_Time": _safe_stat(times, "first"),
        "End_Time": _safe_stat(times, "last"),
        "Duration_s": _safe_stat(times, "last") - _safe_stat(times, "first"),
        "Median_Current_A": median_current,
        "Mean_Current_A": _safe_stat(current, "mean"),
        "Min_Current_A": _safe_stat(current, "min"),
        "Max_Current_A": _safe_stat(current, "max"),
        "Estimated_C_Rate": abs(median_current) / nominal_capacity_Ah
        if np.isfinite(median_current)
        else np.nan,
        "Start_V": _safe_stat(voltage, "first"),
        "End_V": _safe_stat(voltage, "last"),
        "Min_V": _safe_stat(voltage, "min"),
        "Max_V": _safe_stat(voltage, "max"),
        "Capacity_Start_mAh": start_capacity,
        "Capacity_End_mAh": end_capacity,
        "Capacity_Change_mAh": end_capacity - start_capacity,
        "Physical_Function": physical_function,
    }


def _classify_terminal_current(
    median_current_A: float, thresholds: Mapping[str, Any]
) -> str:
    if not np.isfinite(median_current_A):
        return "UNRESOLVED"
    magnitude = abs(median_current_A)
    if (
        float(thresholds["terminal_1C_current_abs_min_A"])
        <= magnitude
        <= float(thresholds["terminal_1C_current_abs_max_A"])
    ):
        return "APPROX_1C"
    if (
        float(thresholds["terminal_2C_current_abs_min_A"])
        <= magnitude
        <= float(thresholds["terminal_2C_current_abs_max_A"])
    ):
        return "APPROX_2C"
    return "OTHER"


def analyze_raw_cycle(
    *,
    cell_id: str,
    source_file: str,
    raw_cycle: int,
    cycle_position: str,
    reason_selected: str,
    frame: pd.DataFrame,
    native_dq_mAh: Any,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reconstruct one cycle from stored row order and raw time-series values."""

    columns = config["raw_column_map"]
    required = list(columns.values())
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{cell_id} cycle {raw_cycle}: missing raw columns {missing}"
        )
    if frame.empty:
        raise ValueError(f"{cell_id} cycle {raw_cycle}: empty raw cycle")

    thresholds = config["validation_thresholds"]
    nominal = float(config["nominal_capacity_Ah"])
    statuses = frame[columns["status"]].astype(str).to_numpy()
    time_s = _numeric(frame, columns["time_s"])
    current_A = _numeric(frame, columns["current_mA"]) / 1000.0
    voltage_V = _numeric(frame, columns["voltage_V"])
    capacity_mAh = _numeric(frame, columns["capacity_mAh"])
    cycle_field = pd.to_numeric(frame[columns["cycle_index"]], errors="coerce")

    time_order_monotonic = bool(
        np.isfinite(time_s).all() and np.all(np.diff(time_s) >= 0)
    )
    raw_cycle_field_matches = bool(
        cycle_field.notna().all()
        and set(cycle_field.astype(int).tolist()) == {int(raw_cycle)}
    )

    descriptors: list[dict[str, Any]] = []
    for base in _base_segments(statuses):
        status = str(base["status"])
        indices = base["indices"]
        median_current = _safe_stat(current_A[indices], "median")
        if status == str(config["charge_stage1_status"]):
            descriptors.append(
                {
                    "status": status,
                    "label": status,
                    "function": "CHARGE_CC_HIGH_RATE",
                    "indices": indices,
                }
            )
        elif status == str(config["charge_cccv_status"]):
            cc_indices, cv_indices = _split_cccv(
                indices, current_A, voltage_V, thresholds
            )
            if cc_indices.size:
                descriptors.append(
                    {
                        "status": status,
                        "label": status + "::LOW_RATE_CC_PORTION",
                        "function": "CHARGE_CC_LOW_RATE",
                        "indices": cc_indices,
                    }
                )
            if cv_indices.size:
                descriptors.append(
                    {
                        "status": status,
                        "label": status + "::CV_PORTION",
                        "function": "CHARGE_CV",
                        "indices": cv_indices,
                    }
                )
        elif np.isfinite(median_current) and median_current < float(
            thresholds["discharge_current_detection_threshold_A"]
        ):
            descriptors.append(
                {
                    "status": status,
                    "label": status,
                    "function": "DISCHARGE_PENDING",
                    "indices": indices,
                }
            )
        elif np.isfinite(median_current) and abs(median_current) <= 0.05:
            descriptors.append(
                {
                    "status": status,
                    "label": status,
                    "function": "REST",
                    "indices": indices,
                }
            )
        else:
            descriptors.append(
                {
                    "status": status,
                    "label": status,
                    "function": "OTHER",
                    "indices": indices,
                }
            )

    discharge_descriptors = [
        descriptor
        for descriptor in descriptors
        if descriptor["function"] == "DISCHARGE_PENDING"
    ]
    for stage, descriptor in enumerate(discharge_descriptors, start=1):
        descriptor["stage"] = stage
        descriptor["function"] = (
            f"DISCHARGE_STAGE_{stage}" if stage <= 4 else "UNKNOWN"
        )

    step_rows = [
        _segment_row(
            cell_id,
            raw_cycle,
            cycle_position,
            source_file,
            order,
            str(descriptor["status"]),
            str(descriptor["label"]),
            str(descriptor["function"]),
            descriptor["indices"],
            time_s,
            current_A,
            voltage_V,
            capacity_mAh,
            nominal,
        )
        for order, descriptor in enumerate(descriptors, start=1)
    ]

    discharge_rows = [
        row for row in step_rows if str(row["Physical_Function"]).startswith("DISCHARGE_STAGE_")
    ]
    four_stages = len(discharge_rows) == 4
    terminal = discharge_rows[-1] if discharge_rows else None
    terminal_identified = bool(
        four_stages
        and terminal is not None
        and int(str(terminal["Physical_Function"]).rsplit("_", 1)[1]) == 4
    )
    terminal_end_voltage = float(terminal["End_V"]) if terminal is not None else np.nan
    terminal_reaches_low_voltage = bool(
        terminal_identified
        and np.isfinite(terminal_end_voltage)
        and float(thresholds["terminal_low_voltage_min_V"])
        <= terminal_end_voltage
        <= float(thresholds["terminal_low_voltage_max_V"])
    )

    suffixes: list[int | None] = []
    for descriptor in discharge_descriptors:
        match = re.search(r"_(\d+)\s*$", str(descriptor["status"]))
        suffixes.append(int(match.group(1)) if match else None)
    mapping_valid = suffixes == [0, 1, 2, 3]
    mapping_status = (
        "RAW_STATUS_INDEX_MATCHES_PHYSICAL_TIME_ORDER"
        if mapping_valid
        else "STATUS_INDEX_NOT_EQUAL_TO_PHYSICAL_STAGE_ORDER"
    )

    native = pd.to_numeric(native_dq_mAh, errors="coerce")
    native = float(native) if np.isfinite(native) else np.nan
    if four_stages:
        discharge_start = float(discharge_rows[0]["Capacity_Start_mAh"])
        discharge_end = float(discharge_rows[-1]["Capacity_End_mAh"])
        reconstructed = discharge_start - discharge_end
        stage_contributions = [
            float(row["Capacity_Start_mAh"] - row["Capacity_End_mAh"])
            for row in discharge_rows
        ]
        stage_sum = float(sum(stage_contributions))
    else:
        reconstructed = np.nan
        stage_contributions = [np.nan] * len(discharge_rows)
        stage_sum = np.nan
    difference = reconstructed - native if np.isfinite(reconstructed) and np.isfinite(native) else np.nan
    difference_pct = (
        100.0 * difference / native
        if np.isfinite(difference) and native != 0
        else np.nan
    )
    stage_sum_difference = (
        reconstructed - stage_sum
        if np.isfinite(reconstructed) and np.isfinite(stage_sum)
        else np.nan
    )
    native_match = bool(
        np.isfinite(difference)
        and abs(difference)
        <= float(thresholds["capacity_match_abs_tolerance_mAh"])
    )
    stage_sum_match = bool(
        np.isfinite(stage_sum_difference)
        and abs(stage_sum_difference)
        <= float(thresholds["stage_sum_match_abs_tolerance_mAh"])
    )

    terminal_current = (
        float(terminal["Median_Current_A"]) if terminal is not None else np.nan
    )
    terminal_crate = (
        abs(terminal_current) / nominal if np.isfinite(terminal_current) else np.nan
    )
    terminal_class = _classify_terminal_current(terminal_current, thresholds)

    charge_rows = [
        row
        for row in step_rows
        if row["Physical_Function"]
        in {"CHARGE_CC_HIGH_RATE", "CHARGE_CC_LOW_RATE", "CHARGE_CV"}
    ]
    high_rate_rows = [
        row for row in step_rows if row["Physical_Function"] == "CHARGE_CC_HIGH_RATE"
    ]
    low_rate_rows = [
        row for row in step_rows if row["Physical_Function"] == "CHARGE_CC_LOW_RATE"
    ]
    cv_rows = [row for row in step_rows if row["Physical_Function"] == "CHARGE_CV"]
    positive_current = current_A[np.isfinite(current_A) & (current_A > 0)]

    cycle_reasons: list[str] = []
    if not time_order_monotonic:
        cycle_reasons.append("RAW_TIME_ORDER_NOT_MONOTONIC")
    if not raw_cycle_field_matches:
        cycle_reasons.append("RAW_CYCLE_INDEX_FIELD_MISMATCH")
    if not four_stages:
        cycle_reasons.append("FOUR_DISCHARGE_STAGES_NOT_IDENTIFIED")
    if not terminal_identified:
        cycle_reasons.append("TERMINAL_STAGE_NOT_IDENTIFIED")
    elif not terminal_reaches_low_voltage:
        cycle_reasons.append("TERMINAL_STAGE_DID_NOT_REACH_LOW_VOLTAGE")
    if not native_match:
        cycle_reasons.append("NATIVE_DQ_RAW_RECONSTRUCTION_MISMATCH")
    if not stage_sum_match:
        cycle_reasons.append("FULL_DISCHARGE_STAGE_SUM_MISMATCH")
    if terminal_class not in {"APPROX_1C", "APPROX_2C"}:
        cycle_reasons.append("TERMINAL_CURRENT_CLASS_UNRESOLVED")
    protocol_flag = "PASS" if not cycle_reasons else "UNRESOLVED"

    for row in step_rows:
        row["Protocol_Validation_Flag"] = protocol_flag
        row["Protocol_Validation_Reason_Codes"] = (
            "PASS" if not cycle_reasons else "|".join(cycle_reasons)
        )

    terminal_row: dict[str, Any] = {
        "Cell_ID": cell_id,
        "Source_File": source_file,
        "Raw_Cycle_Index": raw_cycle,
        "Cycle_Position": cycle_position,
        "Reason_Selected": reason_selected,
        "Raw_Record_Count": len(frame),
        "Raw_Time_Order_Monotonic": "YES" if time_order_monotonic else "NO",
        "Raw_Cycle_Field_Matches": "YES" if raw_cycle_field_matches else "NO",
        "Discharge_Stage_Count": len(discharge_rows),
        "Four_Discharge_Stages_Identified": "YES" if four_stages else "NO",
        "Terminal_Stage_Identified": "YES" if terminal_identified else "NO",
        "Terminal_Stage_Order": 4 if terminal_identified else np.nan,
        "Terminal_Current_A": terminal_current,
        "Terminal_CRate": terminal_crate,
        "Terminal_Current_Class": terminal_class,
        "Terminal_Start_V": float(terminal["Start_V"]) if terminal else np.nan,
        "Terminal_End_V": terminal_end_voltage,
        "Terminal_Min_V": float(terminal["Min_V"]) if terminal else np.nan,
        "Terminal_Capacity_mAh": stage_contributions[-1]
        if stage_contributions
        else np.nan,
        "Terminal_Reaches_Low_Voltage": "YES"
        if terminal_reaches_low_voltage
        else "NO",
        "Paper_Expected_CRate": float(
            config["paper_reported_terminal_discharge_CRate"]
        ),
        "Raw_Observed_CRate": terminal_crate,
        "Mapping_Status": mapping_status,
        "Raw_Discharge_Status_Suffix_Order": "|".join(
            "NA" if suffix is None else str(suffix) for suffix in suffixes
        ),
        "Protocol_Discrepancy_Status": (
            "RAW_TERMINAL_APPROX_2C__PAPER_REPORTS_1C"
            if terminal_class == "APPROX_2C"
            else (
                "RAW_TERMINAL_APPROX_1C__AGREES_WITH_PAPER"
                if terminal_class == "APPROX_1C"
                else "UNRESOLVED"
            )
        ),
        "Charge_Max_Current_A": _safe_stat(positive_current, "max"),
        "Charge_High_Rate_CC_Median_A": float(high_rate_rows[0]["Median_Current_A"])
        if high_rate_rows
        else np.nan,
        "Charge_1C_CC_Median_A": float(low_rate_rows[0]["Median_Current_A"])
        if low_rate_rows
        else np.nan,
        "Charge_Max_V": max((float(row["Max_V"]) for row in charge_rows), default=np.nan),
        "CV_End_Current_A": _safe_stat(
            current_A[
                np.flatnonzero(statuses == str(config["charge_cccv_status"]))
            ],
            "last",
        )
        if np.any(statuses == str(config["charge_cccv_status"]))
        else np.nan,
        "Protocol_Validation_Flag": protocol_flag,
        "Protocol_Validation_Reason_Codes": (
            "PASS" if not cycle_reasons else "|".join(cycle_reasons)
        ),
    }
    for stage in range(1, 5):
        if stage <= len(discharge_rows):
            row = discharge_rows[stage - 1]
            contribution = stage_contributions[stage - 1]
            terminal_row.update(
                {
                    f"Discharge_Stage{stage}_Raw_Status_Label": row[
                        "Raw_Status_Label"
                    ],
                    f"Discharge_Stage{stage}_Median_Current_A": row[
                        "Median_Current_A"
                    ],
                    f"Discharge_Stage{stage}_Estimated_C_Rate": row[
                        "Estimated_C_Rate"
                    ],
                    f"Discharge_Stage{stage}_Start_V": row["Start_V"],
                    f"Discharge_Stage{stage}_End_V": row["End_V"],
                    f"Discharge_Stage{stage}_Capacity_Contribution_mAh": contribution,
                    f"Discharge_Stage{stage}_Is_Terminal": "YES"
                    if stage == 4
                    else "NO",
                }
            )
        else:
            terminal_row.update(
                {
                    f"Discharge_Stage{stage}_Raw_Status_Label": "",
                    f"Discharge_Stage{stage}_Median_Current_A": np.nan,
                    f"Discharge_Stage{stage}_Estimated_C_Rate": np.nan,
                    f"Discharge_Stage{stage}_Start_V": np.nan,
                    f"Discharge_Stage{stage}_End_V": np.nan,
                    f"Discharge_Stage{stage}_Capacity_Contribution_mAh": np.nan,
                    f"Discharge_Stage{stage}_Is_Terminal": "NO",
                }
            )

    capacity_row = {
        "Cell_ID": cell_id,
        "Source_File": source_file,
        "Raw_Cycle_Index": raw_cycle,
        "Cycle_Position": cycle_position,
        "Native_dq_mAh": native,
        "Raw_Capacity_Channel_Semantics": "WITHIN_CYCLE_CUMULATIVE_CHARGE_THEN_DECREASING_DURING_DISCHARGE",
        "Reconstruction_Method": "CAPACITY_AT_STAGE1_START_MINUS_CAPACITY_AT_TERMINAL_STAGE_END",
        "Raw_Reconstructed_Discharge_mAh": reconstructed,
        "Stage_Contribution_Sum_mAh": stage_sum,
        "Full_Minus_Stage_Sum_mAh": stage_sum_difference,
        "Capacity_Difference_mAh": difference,
        "Capacity_Difference_Pct": difference_pct,
        "Native_dq_Abs_Tolerance_mAh": float(
            thresholds["capacity_match_abs_tolerance_mAh"]
        ),
        "Native_dq_Match": "YES" if native_match else "NO",
        "Stage_Sum_Match": "YES" if stage_sum_match else "NO",
        "Validation_Status": "PASS"
        if native_match and stage_sum_match and four_stages
        else "UNRESOLVED",
    }

    plot_context = {
        "time_s": time_s,
        "current_A": current_A,
        "voltage_V": voltage_V,
        "discharge_rows": discharge_rows,
        "terminal_current_A": terminal_current,
        "terminal_end_V": terminal_end_voltage,
    }
    return step_rows, terminal_row, capacity_row, plot_context


def determine_final_assessment(terminal_audit: pd.DataFrame) -> str:
    if terminal_audit.empty:
        return FINAL_ASSESSMENT_UNRESOLVED
    all_core_pass = terminal_audit["Protocol_Validation_Flag"].eq("PASS").all()
    current_classes = set(terminal_audit["Terminal_Current_Class"].astype(str))
    mapping_valid = terminal_audit["Mapping_Status"].eq(
        "RAW_STATUS_INDEX_MATCHES_PHYSICAL_TIME_ORDER"
    ).all()
    if all_core_pass and current_classes == {"APPROX_1C"}:
        return FINAL_ASSESSMENT_PARSER_BUG
    if all_core_pass and current_classes == {"APPROX_2C"} and mapping_valid:
        return FINAL_ASSESSMENT_RAW_DIFFERS
    return FINAL_ASSESSMENT_UNRESOLVED


def _plot_cycle(
    output_path: Path,
    cell_id: str,
    raw_cycle: int,
    cycle_position: str,
    context: Mapping[str, Any],
) -> None:
    time_s = np.asarray(context["time_s"], dtype=float)
    current_A = np.asarray(context["current_A"], dtype=float)
    voltage_V = np.asarray(context["voltage_V"], dtype=float)
    discharge_rows = list(context["discharge_rows"])
    colors = ["#2563EB", "#F59E0B", "#7C3AED", "#DC2626"]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True)
    axes[0].plot(time_s, current_A, color="#111827", linewidth=1.0)
    axes[1].plot(time_s, voltage_V, color="#0F766E", linewidth=1.0)
    for stage, row in enumerate(discharge_rows, start=1):
        start = float(row["Start_Time"])
        end = float(row["End_Time"])
        for axis in axes:
            axis.axvspan(start, end, color=colors[min(stage - 1, 3)], alpha=0.16)
        midpoint = (start + end) / 2.0
        axes[0].text(
            midpoint,
            0.96,
            f"D{stage}",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=colors[min(stage - 1, 3)],
            fontweight="bold",
        )

    axes[0].axhline(0, color="#6B7280", linewidth=0.7)
    axes[0].set_ylabel("Current (A)")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].set_xlabel("Raw Time (s)")
    for axis in axes:
        axis.grid(True, color="#D1D5DB", linewidth=0.5, alpha=0.6)

    terminal_current = float(context["terminal_current_A"])
    terminal_end_voltage = float(context["terminal_end_V"])
    axes[1].annotate(
        f"Terminal D4: median I={terminal_current:.4f} A, end V={terminal_end_voltage:.4f} V",
        xy=(float(discharge_rows[-1]["End_Time"]), terminal_end_voltage),
        xytext=(0.52, 0.18),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#991B1B"},
        fontsize=9,
        color="#991B1B",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#FCA5A5"},
    )
    handles = [
        Patch(facecolor=colors[index], alpha=0.25, label=f"Discharge Stage {index + 1}")
        for index in range(4)
    ]
    axes[0].legend(handles=handles, ncol=4, loc="lower center", fontsize=8)
    fig.suptitle(
        f"{cell_id} — Raw Cycle {raw_cycle} ({cycle_position})\n"
        "Unsmoothed raw time series; discharge stages follow physical time order",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _cell_protocol_audit(
    terminal_audit: pd.DataFrame, final_assessment: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell_id, group in terminal_audit.groupby("Cell_ID", sort=False):
        currents = pd.to_numeric(group["Terminal_Current_A"], errors="coerce")
        crates = pd.to_numeric(group["Terminal_CRate"], errors="coerce")
        mapping_valid = group["Mapping_Status"].eq(
            "RAW_STATUS_INDEX_MATCHES_PHYSICAL_TIME_ORDER"
        ).all()
        current_classes = set(group["Terminal_Current_Class"].astype(str))
        consistent = bool(
            len(current_classes) == 1
            and currents.max() - currents.min() <= 0.05
            and group["Protocol_Validation_Flag"].eq("PASS").all()
        )
        rows.append(
            {
                "Cell_ID": cell_id,
                "Cycles_Checked": "|".join(
                    str(int(value)) for value in group["Raw_Cycle_Index"]
                ),
                "Paper_Final_Stage_CRate": 1.0,
                "Raw_Final_Stage_CRate_Summary": (
                    f"{next(iter(current_classes))};MEDIAN={crates.median():.6f}C"
                    if len(current_classes) == 1
                    else "MIXED_OR_UNRESOLVED"
                ),
                "Raw_Final_Current_A_Median": currents.median(),
                "Raw_Final_Current_A_Min": currents.min(),
                "Raw_Final_Current_A_Max": currents.max(),
                "Stage_Mapping_Validated": "YES" if mapping_valid else "NO",
                "Protocol_Consistent_Across_Life": "YES" if consistent else "NO",
                "Paper_vs_Raw_Protocol_Discrepancy": "YES"
                if current_classes == {"APPROX_2C"}
                else "NO_OR_UNRESOLVED",
                "Final_Assessment": final_assessment,
                "Protocol_Validation_Flag": "PASS"
                if consistent and mapping_valid
                else "UNRESOLVED",
            }
        )
    return pd.DataFrame(rows)


def _regression_tests(
    selected: pd.DataFrame,
    steps: pd.DataFrame,
    terminal: pd.DataFrame,
    capacity: pd.DataFrame,
    cell_audit: pd.DataFrame,
    expected_cells: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(test_id: str, expected: str, observed: str, passed: bool, details: str = "") -> None:
        rows.append(
            {
                "Test_ID": test_id,
                "Scope": "HUST_4CELL_PROTOCOL_VALIDATION",
                "Expected": expected,
                "Observed": observed,
                "Status": "PASS" if passed else "FAIL",
                "Details": details,
            }
        )

    cell_count = selected["Cell_ID"].nunique()
    add("CELL_COUNT_SELECTED", "4", str(cell_count), cell_count == 4)
    counts = selected.groupby("Cell_ID")["Raw_Cycle_Index"].nunique()
    counts_text = "|".join(f"{cell}={int(counts.get(cell, 0))}" for cell in expected_cells)
    add(
        "CYCLES_PER_CELL_SELECTED",
        "4 for each fixed cell",
        counts_text,
        all(int(counts.get(cell, 0)) == 4 for cell in expected_cells),
    )
    total = len(selected)
    add("TOTAL_VALIDATION_CYCLES", "16", str(total), total == 16)

    time_pass = terminal["Raw_Time_Order_Monotonic"].eq("YES")
    add(
        "RAW_TIME_ORDER_MONOTONIC",
        "PASS for all 16",
        f"{int(time_pass.sum())}/{len(time_pass)} PASS",
        bool(time_pass.all() and len(time_pass) == 16),
    )
    stage_pass = terminal["Four_Discharge_Stages_Identified"].eq("YES")
    add(
        "FOUR_DISCHARGE_STAGES_IDENTIFIED",
        "PASS for all 16",
        f"{int(stage_pass.sum())}/{len(stage_pass)} PASS",
        bool(stage_pass.all() and len(stage_pass) == 16),
    )
    low_v = terminal["Terminal_Reaches_Low_Voltage"].eq("YES")
    add(
        "TERMINAL_STAGE_REACHES_LOW_VOLTAGE",
        "PASS for all 16",
        f"{int(low_v.sum())}/{len(low_v)} PASS",
        bool(low_v.all() and len(low_v) == 16),
        "Measured terminal end voltages are retained without clipping.",
    )
    native_match = capacity["Native_dq_Match"].eq("YES")
    add(
        "NATIVE_DQ_MATCHES_RAW_RECONSTRUCTION",
        "PASS for all 16 at abs difference <= 0.001 mAh",
        f"{int(native_match.sum())}/{len(native_match)} PASS; max_abs_diff="
        f"{pd.to_numeric(capacity['Capacity_Difference_mAh']).abs().max():.12g} mAh",
        bool(native_match.all() and len(native_match) == 16),
    )
    semantics = capacity["Validation_Status"].eq("PASS")
    add(
        "CAPACITY_SEMANTICS_STILL_VALID",
        "PASS",
        f"{int(semantics.sum())}/{len(semantics)} PASS",
        bool(semantics.all() and len(semantics) == 16),
        "Full discharge uses stage-1 start minus terminal-stage end; stage contributions are not double-counted.",
    )
    mapping = terminal["Mapping_Status"].eq(
        "RAW_STATUS_INDEX_MATCHES_PHYSICAL_TIME_ORDER"
    )
    add(
        "RAW_STATUS_STAGE_MAPPING_VALIDATED",
        "PASS for all 16",
        f"{int(mapping.sum())}/{len(mapping)} PASS",
        bool(mapping.all() and len(mapping) == 16),
    )
    class_1c = terminal["Terminal_Current_Class"].eq("APPROX_1C")
    class_2c = terminal["Terminal_Current_Class"].eq("APPROX_2C")
    uniform_resolved_class = bool(
        len(terminal) == 16 and (class_1c.all() or class_2c.all())
    )
    add(
        "TERMINAL_STAGE_CURRENT_CLASS",
        "All 16 approximately 2C or all 16 approximately 1C",
        f"APPROX_1C={int(class_1c.sum())}; APPROX_2C={int(class_2c.sum())}; total={len(class_2c)}",
        uniform_resolved_class,
        "A uniform 1C result supports parser correction; a uniform 2C result with valid mapping supports a raw-data-versus-paper discrepancy.",
    )
    life_consistent = cell_audit["Protocol_Consistent_Across_Life"].eq("YES")
    add(
        "PROTOCOL_CONSISTENT_ACROSS_SELECTED_LIFE_POSITIONS",
        "PASS for all 4 cells",
        f"{int(life_consistent.sum())}/{len(life_consistent)} PASS",
        bool(life_consistent.all() and len(life_consistent) == 4),
    )
    allowed = set(steps["Physical_Function"].astype(str)).issubset(
        ALLOWED_PHYSICAL_FUNCTIONS
    )
    add(
        "PHYSICAL_FUNCTION_VALUES_ALLOWED",
        "PASS",
        "PASS" if allowed else "FAIL",
        allowed,
    )
    return pd.DataFrame(rows)


def _additional_validation_recommendation(
    terminal: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Recommendation_ID": "HUST_PROTOCOL_ADDITIONAL_4CELL",
                "Reason": "The fixed 16-cycle validation did not yield one internally consistent terminal-stage interpretation.",
                "Recommended_Additional_Cell_Count": 4,
                "Selection_Principle": "Select four new cells spanning cycle life, first capacity, and discharge fingerprints; do not run all 77 cells.",
                "Observed_Current_Classes": "|".join(
                    sorted(set(terminal["Terminal_Current_Class"].astype(str)))
                ),
                "Action": "STOP_QREF_SOH_EOL_RUL_AND_AWAIT_HUMAN_REVIEW",
            }
        ]
    )


def _final_console_block(final_assessment: str) -> str:
    if final_assessment == FINAL_ASSESSMENT_PARSER_BUG:
        return "\n".join(
            [
                "HUST 4-CELL PROTOCOL VALIDATION COMPLETE",
                "",
                "FINAL ASSESSMENT:",
                FINAL_ASSESSMENT_PARSER_BUG,
                "",
                "ACTION REQUIRED:",
                "FIX PARSER AND REBUILD PROTOCOL FINGERPRINTS",
                "",
                "DO NOT RUN QREF PILOT YET",
            ]
        )
    if final_assessment == FINAL_ASSESSMENT_RAW_DIFFERS:
        return "\n".join(
            [
                "HUST 4-CELL PROTOCOL VALIDATION COMPLETE",
                "",
                "FINAL ASSESSMENT:",
                FINAL_ASSESSMENT_RAW_DIFFERS,
                "",
                "RAW PROTOCOL IS INTERNALLY CONSISTENT",
                "",
                "READY FOR HUMAN REVIEW BEFORE QREF PILOT",
            ]
        )
    return "\n".join(
        [
            "HUST 4-CELL PROTOCOL VALIDATION NEEDS REVIEW",
            "",
            "FINAL ASSESSMENT:",
            FINAL_ASSESSMENT_UNRESOLVED,
            "",
            "DO NOT RUN QREF PILOT",
            "DO NOT RUN 77 CELLS",
        ]
    )


def _processing_log(
    *,
    config: Mapping[str, Any],
    raw_hashes_before: Mapping[str, str],
    raw_hashes_after: Mapping[str, str],
    scan_unchanged: bool,
    selected: pd.DataFrame,
    terminal: pd.DataFrame,
    capacity: pd.DataFrame,
    tests: pd.DataFrame,
    final_assessment: str,
    plot_count: int,
) -> str:
    lines = [
        "HUST 4-cell raw protocol validation",
        f"Processing timestamp UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Python version: {platform.python_version()}",
        f"numpy version: {np.__version__}",
        f"pandas version: {pd.__version__}",
        f"matplotlib version: {matplotlib.__version__}",
        "",
        "Scope controls:",
        "- Fixed cells only: HUST_2-5, HUST_9-3, HUST_4-7, HUST_10-3",
        "- Four raw cycles per cell: cycle 1, cycle 2, half-up middle, final",
        "- No Qref/SOH/EOL/RUL calculation",
        "- No model training",
        "- No smoothing, interpolation, capacity correction, or raw-file writes",
        "",
        "Validation config:",
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
            "",
            "Selected cycles:",
        ]
    )
    for _, row in selected.iterrows():
        lines.append(
            f"- {row['Cell_ID']} cycle {int(row['Raw_Cycle_Index'])} "
            f"({row['Cycle_Position']}): {row['Reason_Selected']}"
        )
    lines.extend(
        [
            "",
            "Observed terminal-stage summary:",
            f"- Validation cycles: {len(terminal)}",
            f"- Approximate 1C terminal stages: {int(terminal['Terminal_Current_Class'].eq('APPROX_1C').sum())}",
            f"- Approximate 2C terminal stages: {int(terminal['Terminal_Current_Class'].eq('APPROX_2C').sum())}",
            f"- Terminal current median (A): {pd.to_numeric(terminal['Terminal_Current_A']).median():.12g}",
            f"- Terminal end voltage range (V): {pd.to_numeric(terminal['Terminal_End_V']).min():.12g} to {pd.to_numeric(terminal['Terminal_End_V']).max():.12g}",
            f"- Stage mapping validated: {int(terminal['Mapping_Status'].eq('RAW_STATUS_INDEX_MATCHES_PHYSICAL_TIME_ORDER').sum())}/{len(terminal)}",
            "",
            "Capacity semantics regression:",
            f"- Native dq matches: {int(capacity['Native_dq_Match'].eq('YES').sum())}/{len(capacity)}",
            f"- Maximum absolute native/raw difference (mAh): {pd.to_numeric(capacity['Capacity_Difference_mAh']).abs().max():.12g}",
            f"- Maximum absolute full/stage-sum difference (mAh): {pd.to_numeric(capacity['Full_Minus_Stage_Sum_mAh']).abs().max():.12g}",
            "",
            f"Review plots generated: {plot_count}",
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
            else "REGRESSION TEST FAILURE — HARD STOP",
            "",
            _final_console_block(final_assessment),
            "",
        ]
    )
    return "\n".join(lines)


def run_protocol_validation(
    workspace: Path, input_dir: Path, scan_dir: Path, output_dir: Path
) -> ProtocolValidationResult:
    workspace = workspace.resolve()
    input_dir = input_dir.resolve()
    scan_dir = scan_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    required_scan_files = [
        scan_dir / "hust_file_manifest.csv",
        scan_dir / "hust_cell_inventory.csv",
    ]
    for path in required_scan_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    config = make_protocol_validation_config(
        workspace, input_dir, scan_dir, output_dir
    )
    expected_cells = list(config["selected_cells"])
    file_manifest = pd.read_csv(scan_dir / "hust_file_manifest.csv")
    cell_inventory = pd.read_csv(scan_dir / "hust_cell_inventory.csv")
    canonical = file_manifest[
        file_manifest["File_Role"].eq("CANONICAL")
        & file_manifest["Included_In_Analysis"].astype(str).str.upper().eq("YES")
    ].copy()
    selected_manifest = canonical[canonical["Cell_ID"].isin(expected_cells)].copy()
    if set(selected_manifest["Cell_ID"]) != set(expected_cells) or len(selected_manifest) != 4:
        raise RuntimeError(
            "The frozen scan manifest does not contain exactly the four required canonical cells"
        )

    scan_hashes_before = _directory_hashes(scan_dir)
    raw_paths: dict[str, Path] = {}
    raw_hashes_before: dict[str, str] = {}
    for cell_id in expected_cells:
        row = selected_manifest[selected_manifest["Cell_ID"].eq(cell_id)].iloc[0]
        path = input_dir / str(row["Relative_Path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = _sha256(path)
        expected_hash = str(row["SHA256"])
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"RAW_HASH_MISMATCH:{cell_id}:{path.name}:expected={expected_hash}:actual={actual_hash}"
            )
        raw_paths[cell_id] = path
        raw_hashes_before[path.name] = actual_hash

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    plot_dir = temp_dir / "review_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []
    plot_count = 0
    try:
        for cell_order, cell_id in enumerate(expected_cells, start=1):
            manifest_row = selected_manifest[
                selected_manifest["Cell_ID"].eq(cell_id)
            ].iloc[0]
            source_file = str(manifest_row["File_Name"])
            payload = load_hust_pickle(raw_paths[cell_id])
            native_id = str(manifest_row["Native_Cell_ID"])
            if native_id not in payload:
                if len(payload) != 1:
                    raise RuntimeError(
                        f"{cell_id}: expected native cell {native_id}, found {list(payload)}"
                    )
                native_key = next(iter(payload))
            else:
                native_key = native_id
            cell_payload = payload[native_key]
            data = cell_payload["data"]
            dq = cell_payload["dq"]
            cycle_keys = _integer_cycle_keys(data)
            choices = select_validation_cycles(cycle_keys)

            inventory_rows = cell_inventory[cell_inventory["Cell_ID"].eq(cell_id)]
            if len(inventory_rows) != 1:
                raise RuntimeError(f"{cell_id}: missing or duplicate frozen cell inventory row")
            frozen_last = int(inventory_rows.iloc[0]["Last_Raw_Cycle_Index"])
            if max(cycle_keys) != frozen_last:
                raise RuntimeError(
                    f"{cell_id}: current last cycle {max(cycle_keys)} differs from frozen scan {frozen_last}"
                )

            for position_order, choice in enumerate(choices, start=1):
                raw_cycle = int(choice["Raw_Cycle_Index"])
                selected_rows.append(
                    {
                        "Cell_Selection_Order": cell_order,
                        "Cycle_Position_Order": position_order,
                        "Cell_ID": cell_id,
                        "Source_File": source_file,
                        "Raw_Cycle_Index": raw_cycle,
                        "Cycle_Position": choice["Cycle_Position"],
                        "Target_Raw_Cycle_Index": choice[
                            "Target_Raw_Cycle_Index"
                        ],
                        "Last_Raw_Cycle_Index": max(cycle_keys),
                        "Reason_Selected": choice["Reason_Selected"],
                    }
                )
                frame = _lookup(data, raw_cycle)
                native_dq = _lookup(dq, raw_cycle)
                steps, terminal, capacity, plot_context = analyze_raw_cycle(
                    cell_id=cell_id,
                    source_file=source_file,
                    raw_cycle=raw_cycle,
                    cycle_position=str(choice["Cycle_Position"]),
                    reason_selected=str(choice["Reason_Selected"]),
                    frame=frame,
                    native_dq_mAh=native_dq,
                    config=config,
                )
                all_step_rows.extend(steps)
                terminal_rows.append(terminal)
                capacity_rows.append(capacity)
                if choice["Cycle_Position"] in {"INITIAL", "FINAL"}:
                    plot_name = (
                        f"{cell_id}_cycle_{raw_cycle}_{str(choice['Cycle_Position']).lower()}_protocol.png"
                    )
                    _plot_cycle(
                        plot_dir / plot_name,
                        cell_id,
                        raw_cycle,
                        str(choice["Cycle_Position"]),
                        plot_context,
                    )
                    plot_count += 1
            del payload

        selected = pd.DataFrame(selected_rows).sort_values(
            ["Cell_Selection_Order", "Cycle_Position_Order"], kind="stable"
        )
        steps = pd.DataFrame(all_step_rows)
        steps["Cell_Selection_Order"] = steps["Cell_ID"].map(
            {cell: index for index, cell in enumerate(expected_cells, start=1)}
        )
        steps["Cycle_Position_Order"] = steps["Cycle_Position"].map(
            {"INITIAL": 1, "SECOND": 2, "MIDDLE": 3, "FINAL": 4}
        )
        steps = steps.sort_values(
            ["Cell_Selection_Order", "Cycle_Position_Order", "Physical_Step_Order"],
            kind="stable",
        )
        terminal = pd.DataFrame(terminal_rows)
        terminal["Cell_Selection_Order"] = terminal["Cell_ID"].map(
            {cell: index for index, cell in enumerate(expected_cells, start=1)}
        )
        terminal["Cycle_Position_Order"] = terminal["Cycle_Position"].map(
            {"INITIAL": 1, "SECOND": 2, "MIDDLE": 3, "FINAL": 4}
        )
        terminal = terminal.sort_values(
            ["Cell_Selection_Order", "Cycle_Position_Order"], kind="stable"
        )
        capacity = pd.DataFrame(capacity_rows)
        capacity["Cell_Selection_Order"] = capacity["Cell_ID"].map(
            {cell: index for index, cell in enumerate(expected_cells, start=1)}
        )
        capacity["Cycle_Position_Order"] = capacity["Cycle_Position"].map(
            {"INITIAL": 1, "SECOND": 2, "MIDDLE": 3, "FINAL": 4}
        )
        capacity = capacity.sort_values(
            ["Cell_Selection_Order", "Cycle_Position_Order"], kind="stable"
        )

        final_assessment = determine_final_assessment(terminal)
        cell_audit = _cell_protocol_audit(terminal, final_assessment)
        tests = _regression_tests(
            selected, steps, terminal, capacity, cell_audit, expected_cells
        )
        tests_passed = bool(tests["Status"].eq("PASS").all())
        if not tests_passed:
            final_assessment = FINAL_ASSESSMENT_UNRESOLVED
            cell_audit["Final_Assessment"] = final_assessment
            cell_audit["Protocol_Validation_Flag"] = "UNRESOLVED"

        raw_hashes_after = {
            path.name: _sha256(path) for path in raw_paths.values()
        }
        if raw_hashes_after != raw_hashes_before:
            raise RuntimeError("RAW_FILE_HASH_CHANGED_DURING_VALIDATION")
        scan_hashes_after = _directory_hashes(scan_dir)
        scan_unchanged = scan_hashes_after == scan_hashes_before
        if not scan_unchanged:
            raise RuntimeError("FROZEN_HUST_SCAN_OUTPUT_CHANGED_DURING_VALIDATION")

        _write_csv(selected, temp_dir / "hust_4cell_selected_cycles.csv")
        _write_csv(steps, temp_dir / "hust_4cell_raw_step_audit.csv")
        _write_csv(
            terminal, temp_dir / "hust_4cell_terminal_discharge_audit.csv"
        )
        _write_csv(
            capacity,
            temp_dir / "hust_4cell_capacity_semantics_validation.csv",
        )
        _write_csv(cell_audit, temp_dir / "hust_paper_vs_raw_protocol_audit.csv")
        _write_csv(
            tests, temp_dir / "hust_4cell_protocol_regression_tests.csv"
        )
        if final_assessment == FINAL_ASSESSMENT_UNRESOLVED:
            _write_csv(
                _additional_validation_recommendation(terminal),
                temp_dir / "additional_validation_recommendation.csv",
            )
        log_text = _processing_log(
            config=config,
            raw_hashes_before=raw_hashes_before,
            raw_hashes_after=raw_hashes_after,
            scan_unchanged=scan_unchanged,
            selected=selected,
            terminal=terminal,
            capacity=capacity,
            tests=tests,
            final_assessment=final_assessment,
            plot_count=plot_count,
        )
        (temp_dir / "processing_log_validate.txt").write_text(
            log_text, encoding="utf-8"
        )

        temp_dir.replace(output_dir)
        return ProtocolValidationResult(
            output_dir=output_dir,
            selected_cycle_count=len(selected),
            raw_step_count=len(steps),
            plot_count=plot_count,
            final_assessment=final_assessment,
            regression_tests_passed=tests_passed,
        )
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


__all__ = [
    "FINAL_ASSESSMENT_PARSER_BUG",
    "FINAL_ASSESSMENT_RAW_DIFFERS",
    "FINAL_ASSESSMENT_UNRESOLVED",
    "ProtocolValidationResult",
    "analyze_raw_cycle",
    "determine_final_assessment",
    "run_protocol_validation",
    "select_validation_cycles",
]
