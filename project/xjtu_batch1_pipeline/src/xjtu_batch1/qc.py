from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .mat_io import cycle_sequence, get_field, load_mat_payload, numeric_1d, text_value


INITIAL_PROTOCOL = "INITIAL_CAPACITY_0P2C_DISCHARGE"
STANDARD_PROTOCOL = "STANDARD_AGING_1C_DISCHARGE"


@dataclass(frozen=True)
class Segment:
    start_index: int
    end_index: int
    record_count: int
    median_current_A: float
    mean_current_A: float
    start_voltage_V: float
    end_voltage_V: float
    min_voltage_V: float
    max_voltage_V: float
    duration_s: float
    capacity_start_Ah: float
    capacity_end_Ah: float
    capacity_max_Ah: float
    capacity_delta_Ah: float


def _contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask.astype(bool), False]))
    return [(int(start), int(stop)) for start, stop in boundaries.reshape(-1, 2)]


def _duration_seconds(relative_time_min: np.ndarray) -> float:
    if relative_time_min.size < 2:
        return 0.0
    direct = float(relative_time_min[-1] - relative_time_min[0])
    if np.isfinite(direct) and direct >= 0:
        return direct * 60.0
    differences = np.diff(relative_time_min)
    positive = differences[np.isfinite(differences) & (differences > 0)]
    return float(positive.sum() * 60.0)


def _segment(
    start: int,
    stop: int,
    current: np.ndarray,
    voltage: np.ndarray,
    capacity: np.ndarray,
    relative_time_min: np.ndarray,
) -> Segment:
    current_values = current[start:stop]
    voltage_values = voltage[start:stop]
    capacity_values = capacity[start:stop]
    time_values = relative_time_min[start:stop]
    return Segment(
        start_index=start,
        end_index=stop - 1,
        record_count=stop - start,
        median_current_A=float(np.median(current_values)),
        mean_current_A=float(np.mean(current_values)),
        start_voltage_V=float(voltage_values[0]),
        end_voltage_V=float(voltage_values[-1]),
        min_voltage_V=float(np.min(voltage_values)),
        max_voltage_V=float(np.max(voltage_values)),
        duration_s=_duration_seconds(time_values),
        capacity_start_Ah=float(capacity_values[0]),
        capacity_end_Ah=float(capacity_values[-1]),
        capacity_max_Ah=float(np.max(capacity_values)),
        capacity_delta_Ah=float(np.max(capacity_values) - np.min(capacity_values)),
    )


def _select_main_segment(
    mask: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    capacity: np.ndarray,
    relative_time_min: np.ndarray,
) -> Segment | None:
    candidates = [
        _segment(start, stop, current, voltage, capacity, relative_time_min)
        for start, stop in _contiguous_spans(mask)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.capacity_delta_Ah, item.record_count))


def _issue(row: dict[str, Any], severity: str, code: str, reason: str) -> None:
    issues: list[tuple[str, str, str]] = row["_Issues"]
    if code not in {existing[1] for existing in issues}:
        issues.append((severity, code, reason))


def _flag_from_issues(issues: list[tuple[str, str, str]]) -> str:
    severities = {severity for severity, _, _ in issues}
    if "INVALID_CANDIDATE" in severities:
        return "INVALID_CANDIDATE"
    if "CHECK" in severities:
        return "CHECK"
    return "PASS"


def _finalize_row(row: dict[str, Any]) -> None:
    issues: list[tuple[str, str, str]] = row["_Issues"]
    flag = _flag_from_issues(issues)
    codes = [code for _, code, _ in issues]
    row["Auto_QC_Flag"] = flag
    row["Auto_QC_Reason_Codes"] = "|".join(codes) if codes else "PASS"
    row["Auto_QC_Reason"] = (
        "; ".join(reason for _, _, reason in issues)
        if issues
        else "No automatic protocol or capacity QC issue detected"
    )
    row["Primary_Include_Status"] = {
        "PASS": "YES",
        "CHECK": "REVIEW",
        "INVALID_CANDIDATE": "NO",
    }[flag]
    jump_only = codes == ["CAPACITY_JUMP"]
    row["Provisional_Include"] = (
        "YES"
        if flag == "PASS"
        or (
            flag == "CHECK"
            and jump_only
            and bool(row["Charge_Complete"])
            and bool(row["Discharge_Complete"])
            and float(row["Summary_Discharge_Capacity_Ah"]) > 0
        )
        else "NO"
    )


def _valid_jump_reference(row: dict[str, Any]) -> bool:
    return bool(
        row["Schema_Valid"]
        and row["Charge_Complete"]
        and row["Discharge_Complete"]
        and float(row["Summary_Discharge_Capacity_Ah"]) > 0
        and _flag_from_issues(row["_Issues"]) != "INVALID_CANDIDATE"
    )


def _apply_capacity_jump(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    last_valid_by_protocol: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: int(item["Raw_Cycle_Index"])):
        protocol = str(row["Protocol_Class"])
        reference = last_valid_by_protocol.get(protocol)
        row["Capacity_Jump_Flag"] = False
        row["Jump_Reference_Raw_Cycle"] = pd.NA
        row["Jump_Reference_Capacity_Ah"] = np.nan
        row["Jump_Delta_Ah"] = np.nan
        row["Jump_Delta_Pct"] = np.nan
        if reference is None:
            row["Jump_Reference_Status"] = "NO_VALID_JUMP_REFERENCE"
        else:
            reference_capacity = float(reference["Summary_Discharge_Capacity_Ah"])
            current_capacity = float(row["Summary_Discharge_Capacity_Ah"])
            delta = current_capacity - reference_capacity
            delta_pct = 100.0 * delta / reference_capacity
            row["Jump_Reference_Status"] = "VALID_JUMP_REFERENCE"
            row["Jump_Reference_Raw_Cycle"] = int(reference["Raw_Cycle_Index"])
            row["Jump_Reference_Capacity_Ah"] = reference_capacity
            row["Jump_Delta_Ah"] = delta
            row["Jump_Delta_Pct"] = delta_pct
            threshold = max(
                float(config["capacity_jump_abs_threshold_Ah"]),
                float(config["capacity_jump_relative_threshold"]) * reference_capacity,
            )
            if abs(delta) > threshold:
                row["Capacity_Jump_Flag"] = True
                _issue(
                    row,
                    "CHECK",
                    "CAPACITY_JUMP",
                    "Capacity changed by "
                    f"{delta:.6f} Ah ({delta_pct:.3f}%) versus valid comparable Raw Cycle "
                    f"{int(reference['Raw_Cycle_Index'])}; trigger>{threshold:.6f} Ah",
                )

        _finalize_row(row)
        if _valid_jump_reference(row):
            last_valid_by_protocol[protocol] = row


def _process_cycle(
    cell_id: str,
    source_file: str,
    raw_index: int,
    cycle: Any,
    summary_capacity_Ah: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    protocol = INITIAL_PROTOCOL if raw_index == 1 else STANDARD_PROTOCOL
    aging_index: int | pd._libs.missing.NAType = pd.NA if raw_index == 1 else raw_index - 1
    description = text_value(get_field(cycle, "description"))
    relative_time = numeric_1d(get_field(cycle, "relative_time_min"))
    voltage = numeric_1d(get_field(cycle, "voltage_V"))
    current = numeric_1d(get_field(cycle, "current_A"))
    capacity = numeric_1d(get_field(cycle, "capacity_Ah"))

    charge = _select_main_segment(
        current > float(config["charge_current_detection_threshold_A"]),
        current,
        voltage,
        capacity,
        relative_time,
    )
    discharge = _select_main_segment(
        current < float(config["discharge_current_detection_threshold_A"]),
        current,
        voltage,
        capacity,
        relative_time,
    )

    charge_cutoff_upper = float(
        config["cycle1_cv_cutoff_upper_A"]
        if raw_index == 1
        else config["standard_cv_cutoff_upper_A"]
    )
    charge_complete = bool(
        charge is not None
        and charge.max_voltage_V >= float(config["charge_complete_min_V"])
        and 0 <= charge.median_current_A
        and abs(charge.capacity_end_Ah - charge.capacity_start_Ah) >= 0
        and 0 < float(current[charge.end_index]) <= charge_cutoff_upper
    )
    discharge_complete = bool(
        discharge is not None
        and discharge.min_voltage_V <= float(config["discharge_complete_max_V"])
        and discharge.end_voltage_V <= float(config["discharge_end_voltage_max_V"])
    )

    raw_discharge_capacity = discharge.capacity_delta_Ah if discharge is not None else np.nan
    difference = raw_discharge_capacity - summary_capacity_Ah
    difference_pct = (
        100.0 * difference / summary_capacity_Ah
        if np.isfinite(summary_capacity_Ah) and summary_capacity_Ah != 0 and np.isfinite(difference)
        else np.nan
    )

    voltage_spike = False
    abnormal_voltage_jump = False
    max_voltage_jump = np.nan
    if discharge is not None:
        start = discharge.start_index
        stop = discharge.end_index + 1
        discharge_voltage = voltage[start:stop]
        discharge_current = current[start:stop]
        voltage_spike = bool(
            np.any(
                (discharge_current < float(config["voltage_spike_current_threshold_A"]))
                & (discharge_voltage > float(config["voltage_spike_threshold_V"]))
            )
        )
        differences = np.abs(np.diff(discharge_voltage))
        max_voltage_jump = float(differences.max()) if differences.size else 0.0
        abnormal_voltage_jump = bool(
            max_voltage_jump > float(config["voltage_jump_threshold_V"])
        )

    row: dict[str, Any] = {
        "Cell_ID": cell_id,
        "Source_File": source_file,
        "Raw_Cycle_Index": raw_index,
        "Aging_Cycle_Index": aging_index,
        "Protocol_Class": protocol,
        "Description": description,
        "Schema_Valid": True,
        "Summary_Discharge_Capacity_Ah": float(summary_capacity_Ah),
        "Raw_Discharge_Capacity_Ah": raw_discharge_capacity,
        "Capacity_Difference_Ah": difference,
        "Capacity_Difference_Pct": difference_pct,
        "Charge_Found": charge is not None,
        "Charge_Complete": charge_complete,
        "Charge_Record_Count": charge.record_count if charge else 0,
        "Charge_Max_V": charge.max_voltage_V if charge else np.nan,
        "Charge_End_Current_A": float(current[charge.end_index]) if charge else np.nan,
        "Charge_Duration_s": charge.duration_s if charge else np.nan,
        "Main_Discharge_Found": discharge is not None,
        "Discharge_Complete": discharge_complete,
        "Discharge_Record_Count": discharge.record_count if discharge else 0,
        "Discharge_Median_Current_A": discharge.median_current_A if discharge else np.nan,
        "Discharge_Mean_Current_A": discharge.mean_current_A if discharge else np.nan,
        "Discharge_Start_V": discharge.start_voltage_V if discharge else np.nan,
        "Discharge_End_V": discharge.end_voltage_V if discharge else np.nan,
        "Discharge_Min_V": discharge.min_voltage_V if discharge else np.nan,
        "Discharge_Duration_s": discharge.duration_s if discharge else np.nan,
        "Voltage_Spike_During_Discharge": voltage_spike,
        "Abnormal_Voltage_Jump": abnormal_voltage_jump,
        "Max_Abs_Voltage_Jump_V": max_voltage_jump,
        "_Issues": [],
    }

    if summary_capacity_Ah <= 0:
        _issue(
            row,
            "INVALID_CANDIDATE",
            "NONPOSITIVE_CAPACITY",
            f"Summary discharge capacity is nonpositive: {summary_capacity_Ah:.6f} Ah",
        )
    if charge is None:
        _issue(
            row,
            "INVALID_CANDIDATE",
            "INCOMPLETE_CHARGE_PROTOCOL",
            "No main positive-current charge segment was found",
        )
    elif not charge_complete:
        _issue(
            row,
            "INVALID_CANDIDATE",
            "INCOMPLETE_CHARGE_PROTOCOL",
            f"Charge max voltage={charge.max_voltage_V:.4f} V and end current="
            f"{float(current[charge.end_index]):.4f} A; required max voltage>="
            f"{float(config['charge_complete_min_V']):.4f} V and end current<="
            f"{charge_cutoff_upper:.4f} A",
        )

    if discharge is None:
        _issue(
            row,
            "INVALID_CANDIDATE",
            "MISSING_MAIN_DISCHARGE",
            "No continuous negative-current main discharge segment was found",
        )
    else:
        if not discharge_complete:
            _issue(
                row,
                "INVALID_CANDIDATE",
                "INCOMPLETE_DISCHARGE_PROTOCOL",
                f"Main discharge min/end voltage={discharge.min_voltage_V:.4f}/"
                f"{discharge.end_voltage_V:.4f} V; required min<="
                f"{float(config['discharge_complete_max_V']):.4f} V and end<="
                f"{float(config['discharge_end_voltage_max_V']):.4f} V",
            )
        expected_current = float(
            config["cycle1_discharge_current_A"]
            if raw_index == 1
            else config["standard_discharge_current_A"]
        )
        current_tolerance = float(
            config["cycle1_discharge_current_abs_tolerance_A"]
            if raw_index == 1
            else config["standard_discharge_current_abs_tolerance_A"]
        )
        if abs(discharge.median_current_A - expected_current) > current_tolerance:
            _issue(
                row,
                "CHECK",
                "DISCHARGE_CURRENT_PROTOCOL_DEVIATION",
                f"Median discharge current={discharge.median_current_A:.4f} A; expected "
                f"{expected_current:.4f}±{current_tolerance:.4f} A",
            )
        if voltage_spike:
            _issue(
                row,
                "INVALID_CANDIDATE",
                "VOLTAGE_SPIKE_DURING_DISCHARGE",
                f"Voltage exceeded {float(config['voltage_spike_threshold_V']):.3f} V while "
                f"current was below {float(config['voltage_spike_current_threshold_A']):.3f} A",
            )
        if abnormal_voltage_jump:
            _issue(
                row,
                "CHECK",
                "ABNORMAL_VOLTAGE_JUMP",
                f"Maximum adjacent voltage jump in main discharge={max_voltage_jump:.4f} V; "
                f"threshold={float(config['voltage_jump_threshold_V']):.4f} V",
            )

    if np.isfinite(difference) and summary_capacity_Ah > 0:
        mismatch_threshold = max(
            float(config["capacity_match_abs_tolerance_Ah"]),
            float(config["capacity_match_relative_tolerance"]) * summary_capacity_Ah,
        )
        if abs(difference) > mismatch_threshold:
            _issue(
                row,
                "CHECK",
                "CAPACITY_SUMMARY_RAW_MISMATCH",
                f"Raw-summary discharge capacity difference={difference:.6f} Ah "
                f"({difference_pct:.3f}%); trigger>{mismatch_threshold:.6f} Ah",
            )
    return row


def process_battery(path: Path, battery_id: int, config: dict[str, Any]) -> pd.DataFrame:
    """Extract summary capacity and raw-curve QC for every original cycle."""

    payload = load_mat_payload(path)
    cycles = cycle_sequence(payload["data"])
    summary = payload["summary"]
    summary_capacities = numeric_1d(get_field(summary, "discharge_capacity_Ah"))
    if len(cycles) != summary_capacities.size:
        raise ValueError(
            f"{path.name}: data cycles={len(cycles)} but summary capacities={summary_capacities.size}"
        )

    cell_id = f"2C_battery-{battery_id}"
    rows = [
        _process_cycle(
            cell_id=cell_id,
            source_file=path.name,
            raw_index=raw_index,
            cycle=cycle,
            summary_capacity_Ah=float(summary_capacities[raw_index - 1]),
            config=config,
        )
        for raw_index, cycle in enumerate(cycles, start=1)
    ]
    _apply_capacity_jump(rows, config)
    for row in rows:
        row.pop("_Issues", None)

    frame = pd.DataFrame(rows)
    frame["Raw_Cycle_Index"] = frame["Raw_Cycle_Index"].astype("Int64")
    frame["Aging_Cycle_Index"] = pd.array(frame["Aging_Cycle_Index"], dtype="Int64")
    frame["Jump_Reference_Raw_Cycle"] = pd.array(
        frame["Jump_Reference_Raw_Cycle"], dtype="Int64"
    )
    return frame

