"""Read CALCE workbooks and extract per-cycle raw capacity and protocol metrics."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from openpyxl import load_workbook

from .config import CELL_ID, QC, QCThresholds


CHANNEL_COLUMNS = (
    "Date_Time",
    "Step_Time(s)",
    "Step_Index",
    "Cycle_Index",
    "Current(A)",
    "Voltage(V)",
    "Charge_Capacity(Ah)",
    "Discharge_Capacity(Ah)",
)


def normalize_header(value: Any) -> str:
    return "" if value is None else str(value).strip()


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def iso_time(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    return None


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_sheet(workbook: Any, prefix: str) -> Any | None:
    prefix = prefix.lower()
    return next((sheet for sheet in workbook.worksheets if sheet.title.lower().startswith(prefix)), None)


def _read_statistics(sheet: Any | None) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if sheet is None:
        return {}, {"present": False, "headers": [], "duplicate_cycles": []}

    iterator = sheet.iter_rows(values_only=True)
    headers = [normalize_header(value) for value in next(iterator)]
    index = {name: position for position, name in enumerate(headers) if name}
    required = ("Cycle_Index", "Discharge_Capacity(Ah)")
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"Statistics sheet missing required columns: {missing}")

    rows: dict[int, dict[str, Any]] = {}
    counts: Counter[int] = Counter()
    for values in iterator:
        cycle = as_int(values[index["Cycle_Index"]])
        if cycle is None:
            continue
        counts[cycle] += 1
        rows[cycle] = {
            "cycle": cycle,
            "date_time": values[index["Date_Time"]] if "Date_Time" in index else None,
            "charge_capacity_ah": as_float(values[index["Charge_Capacity(Ah)"]])
            if "Charge_Capacity(Ah)" in index
            else None,
            "discharge_capacity_ah": as_float(values[index["Discharge_Capacity(Ah)"]]),
        }

    duplicate_cycles = sorted(cycle for cycle, count in counts.items() if count > 1)
    return rows, {
        "present": True,
        "headers": headers,
        "duplicate_cycles": duplicate_cycles,
    }


def _read_channel(sheet: Any) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    iterator = sheet.iter_rows(values_only=True)
    headers = [normalize_header(value) for value in next(iterator)]
    index = {name: position for position, name in enumerate(headers) if name}
    missing = [name for name in CHANNEL_COLUMNS if name not in index]
    if missing:
        raise ValueError(f"Channel sheet missing required columns: {missing}")

    cycles: dict[int, list[dict[str, Any]]] = {}
    cycle_transitions: list[int] = []
    last_cycle: int | None = None
    for excel_row, values in enumerate(iterator, start=2):
        cycle = as_int(values[index["Cycle_Index"]])
        if cycle is None:
            continue
        record = {
            "excel_row": excel_row,
            "date_time": values[index["Date_Time"]],
            "step_time_s": as_float(values[index["Step_Time(s)"]]),
            "step_index": as_int(values[index["Step_Index"]]),
            "cycle_index": cycle,
            "current_a": as_float(values[index["Current(A)"]]),
            "voltage_v": as_float(values[index["Voltage(V)"]]),
            "charge_capacity_ah": as_float(values[index["Charge_Capacity(Ah)"]]),
            "discharge_capacity_ah": as_float(values[index["Discharge_Capacity(Ah)"]]),
        }
        cycles.setdefault(cycle, []).append(record)
        if cycle != last_cycle:
            cycle_transitions.append(cycle)
            last_cycle = cycle

    repeated_transitions = sorted(
        cycle for cycle, count in Counter(cycle_transitions).items() if count > 1
    )
    return cycles, {
        "headers": headers,
        "cycle_transitions": cycle_transitions,
        "repeated_cycle_transitions": repeated_transitions,
    }


def _valid_values(records: Iterable[dict[str, Any]], field: str) -> list[float]:
    return [float(record[field]) for record in records if record.get(field) is not None]


def _span(values: list[float]) -> float | None:
    return max(values) - min(values) if values else None


def _duration(records: list[dict[str, Any]]) -> float | None:
    step_times = _valid_values(records, "step_time_s")
    if step_times:
        return max(step_times) - min(step_times)
    times = [record["date_time"] for record in records if isinstance(record.get("date_time"), datetime)]
    if len(times) >= 2:
        return (max(times) - min(times)).total_seconds()
    return 0.0 if len(times) == 1 else None


def _contiguous_step_blocks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[tuple[int | None, list[dict[str, Any]]]] = []
    current_step: int | None = None
    current_rows: list[dict[str, Any]] = []
    for record in records:
        step = record.get("step_index")
        if current_rows and step != current_step:
            grouped.append((current_step, current_rows))
            current_rows = []
        current_step = step
        current_rows.append(record)
    if current_rows:
        grouped.append((current_step, current_rows))

    blocks: list[dict[str, Any]] = []
    for order, (step, rows) in enumerate(grouped):
        currents = _valid_values(rows, "current_a")
        voltages = _valid_values(rows, "voltage_v")
        charge = _valid_values(rows, "charge_capacity_ah")
        discharge = _valid_values(rows, "discharge_capacity_ah")
        blocks.append(
            {
                "order": order,
                "step_index": step,
                "rows": rows,
                "record_count": len(rows),
                "duration_s": _duration(rows),
                "current_start_a": currents[0] if currents else None,
                "current_end_a": currents[-1] if currents else None,
                "current_mean_a": float(np.mean(currents)) if currents else None,
                "current_std_a": float(np.std(currents)) if currents else None,
                "current_min_a": min(currents) if currents else None,
                "current_max_a": max(currents) if currents else None,
                "voltage_start_v": voltages[0] if voltages else None,
                "voltage_end_v": voltages[-1] if voltages else None,
                "voltage_mean_v": float(np.mean(voltages)) if voltages else None,
                "voltage_min_v": min(voltages) if voltages else None,
                "voltage_max_v": max(voltages) if voltages else None,
                "charge_delta_ah": _span(charge),
                "discharge_delta_ah": _span(discharge),
            }
        )
    return blocks


def _choose_cc(blocks: list[dict[str, Any]], thresholds: QCThresholds) -> dict[str, Any] | None:
    candidates = []
    for block in blocks:
        mean = block["current_mean_a"]
        std = block["current_std_a"]
        if (
            mean is not None
            and thresholds.cc_current_min_a <= mean <= thresholds.cc_current_max_a
            and std is not None
            and std <= thresholds.cc_max_current_std_a
        ):
            candidates.append(block)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda block: (
            block["record_count"],
            block["charge_delta_ah"] or 0.0,
            block["duration_s"] or 0.0,
        ),
    )


def _choose_discharge(blocks: list[dict[str, Any]], thresholds: QCThresholds) -> dict[str, Any] | None:
    candidates = [
        block
        for block in blocks
        if block["current_mean_a"] is not None
        and block["current_mean_a"] <= thresholds.discharge_candidate_current_max_a
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda block: (
            block["discharge_delta_ah"] or 0.0,
            block["duration_s"] or 0.0,
            block["record_count"],
        ),
    )


def _choose_cv(
    blocks: list[dict[str, Any]],
    cc_block: dict[str, Any] | None,
    discharge_block: dict[str, Any] | None,
    thresholds: QCThresholds,
) -> tuple[dict[str, Any] | None, str]:
    cc_order = cc_block["order"] if cc_block is not None else -1
    discharge_order = discharge_block["order"] if discharge_block is not None else 10**9
    between = [block for block in blocks if cc_order < block["order"] < discharge_order]

    primary = []
    for block in between:
        start_current = block["current_start_a"]
        end_current = block["current_end_a"]
        mean_voltage = block["voltage_mean_v"]
        if (
            mean_voltage is not None
            and thresholds.cv_voltage_min_v <= mean_voltage <= thresholds.cv_voltage_max_v
            and start_current is not None
            and end_current is not None
            and start_current > end_current
            and (block["charge_delta_ah"] or 0.0) > 0.0
        ):
            primary.append(block)
    if primary:
        return max(
            primary,
            key=lambda block: (
                block["charge_delta_ah"] or 0.0,
                block["duration_s"] or 0.0,
                block["record_count"],
            ),
        ), "electrochemical criteria"

    # Auxiliary fallback: Step_Index does not prove that CV occurred.  It is
    # used only to retain the suspicious block's raw evidence for QC reporting.
    step_four = [block for block in between if block["step_index"] == 4]
    if step_four:
        return max(step_four, key=lambda block: block["record_count"]), "Step_Index 4 auxiliary fallback"

    high_voltage = [
        block for block in between if (block["voltage_max_v"] or -np.inf) >= 4.0
    ]
    if high_voltage:
        return max(
            high_voltage,
            key=lambda block: (
                block["voltage_max_v"] or 0.0,
                block["charge_delta_ah"] or 0.0,
                block["duration_s"] or 0.0,
            ),
        ), "high-voltage auxiliary fallback"
    return None, "no candidate block"


def _block_value(block: dict[str, Any] | None, key: str) -> Any:
    return None if block is None else block.get(key)


def analyze_cycle(
    records: list[dict[str, Any]],
    stats_row: dict[str, Any] | None,
    stats_previous_cumulative_ah: float | None,
    thresholds: QCThresholds = QC,
) -> dict[str, Any]:
    """Extract one cycle without changing, smoothing, or replacing any value."""

    if not records:
        raise ValueError("cycle has no Channel records")

    blocks = _contiguous_step_blocks(records)
    cc_block = _choose_cc(blocks, thresholds)
    discharge_block = _choose_discharge(blocks, thresholds)
    cv_block, cv_selection_basis = _choose_cv(blocks, cc_block, discharge_block, thresholds)

    qd_values = _valid_values(records, "discharge_capacity_ah")
    qd_start = qd_values[0] if qd_values else None
    qd_end = qd_values[-1] if qd_values else None
    channel_capacity = qd_end - qd_start if qd_start is not None and qd_end is not None else None
    counter_decreases = 0
    if len(qd_values) > 1:
        counter_decreases = sum(
            1
            for previous, current in zip(qd_values, qd_values[1:])
            if current < previous - thresholds.counter_decrease_tolerance_ah
        )

    stats_cumulative = stats_row.get("discharge_capacity_ah") if stats_row else None
    stats_capacity = None
    capacity_source = "CHANNEL_CYCLE_DELTA"
    if stats_cumulative is not None:
        baseline = stats_previous_cumulative_ah
        if baseline is None:
            baseline = qd_start
            capacity_source = "STATISTICS_DELTA_FROM_CHANNEL_BASELINE"
        else:
            capacity_source = "STATISTICS_ADJACENT_DELTA"
        if baseline is not None:
            stats_capacity = stats_cumulative - baseline

    selected_capacity = stats_capacity if stats_capacity is not None else channel_capacity
    agreement_delta = (
        stats_capacity - channel_capacity
        if stats_capacity is not None and channel_capacity is not None
        else None
    )

    cc_present = cc_block is not None
    cc_complete = bool(
        cc_block is not None
        and cc_block["record_count"] >= thresholds.cc_complete_min_records
        and (cc_block["duration_s"] or 0.0) >= thresholds.cc_complete_min_duration_s
        and (cc_block["charge_delta_ah"] or 0.0) >= thresholds.cc_complete_min_charge_delta_ah
        and (cc_block["voltage_end_v"] or -np.inf) >= thresholds.cc_expected_end_voltage_min_v
    )

    cv_stage_present = bool(
        cv_block is not None
        and cv_block["record_count"] >= thresholds.cv_min_records
        and (cv_block["duration_s"] or 0.0) >= thresholds.cv_min_duration_s
        and (cv_block["charge_delta_ah"] or 0.0) >= thresholds.cv_min_charge_delta_ah
        and cv_block["voltage_mean_v"] is not None
        and thresholds.cv_voltage_min_v
        <= cv_block["voltage_mean_v"]
        <= thresholds.cv_voltage_max_v
    )
    cv_protocol_complete = bool(
        cv_stage_present
        and (cv_block["current_start_a"] or -np.inf) >= thresholds.cv_min_start_current_a
        and (cv_block["current_end_a"] or np.inf) <= thresholds.cv_max_end_current_a
        and (cv_block["current_start_a"] - cv_block["current_end_a"])
        >= thresholds.cv_min_current_drop_a
    )

    discharge_present = bool(
        discharge_block is not None
        and discharge_block["record_count"] >= thresholds.discharge_min_records
        and (discharge_block["duration_s"] or 0.0) >= thresholds.discharge_min_duration_s
        and (discharge_block["discharge_delta_ah"] or 0.0)
        >= thresholds.discharge_min_capacity_delta_ah
    )
    discharge_cutoff = _block_value(discharge_block, "voltage_end_v")
    discharge_protocol_complete = bool(
        discharge_present
        and discharge_cutoff is not None
        and thresholds.discharge_cutoff_voltage_min_v
        <= discharge_cutoff
        <= thresholds.discharge_cutoff_voltage_max_v
    )

    times = [record["date_time"] for record in records if isinstance(record.get("date_time"), datetime)]
    return {
        "Cycle_Start_Time": iso_time(min(times)) if times else None,
        "Cycle_End_Time": iso_time(max(times)) if times else None,
        "Channel_Record_Count": len(records),
        "Observed_Step_Indices": "|".join(
            str(block["step_index"]) for block in blocks if block["step_index"] is not None
        ),
        "Statistics_Cumulative_Discharge_Capacity_Ah": stats_cumulative,
        "Statistics_Previous_Cumulative_Discharge_Capacity_Ah": stats_previous_cumulative_ah,
        "Statistics_Single_Cycle_Discharge_Capacity_Ah": stats_capacity,
        "Channel_Discharge_Capacity_Start_Ah": qd_start,
        "Channel_Discharge_Capacity_End_Ah": qd_end,
        "Channel_Single_Cycle_Discharge_Capacity_Ah": channel_capacity,
        "Capacity_Agreement_Delta_Ah": agreement_delta,
        "Capacity_Source": capacity_source,
        "Single_Cycle_Discharge_Capacity_Ah": selected_capacity,
        "Single_Cycle_Discharge_Capacity_mAh": selected_capacity * 1000.0
        if selected_capacity is not None
        else None,
        "Discharge_Counter_Decrease_Count": counter_decreases,
        "CC_Stage_Present": cc_present,
        "CC_Protocol_Complete": cc_complete,
        "CC_Step_Index": _block_value(cc_block, "step_index"),
        "CC_Record_Count": _block_value(cc_block, "record_count") or 0,
        "CC_Duration_s": _block_value(cc_block, "duration_s"),
        "CC_Mean_Current_A": _block_value(cc_block, "current_mean_a"),
        "CC_Start_Voltage_V": _block_value(cc_block, "voltage_start_v"),
        "CC_End_Voltage_V": _block_value(cc_block, "voltage_end_v"),
        "CC_Charge_Delta_Ah": _block_value(cc_block, "charge_delta_ah"),
        "CV_Stage_Present": cv_stage_present,
        "CV_Protocol_Complete": cv_protocol_complete,
        "CV_Selection_Basis": cv_selection_basis,
        "CV_Step_Index": _block_value(cv_block, "step_index"),
        "CV_Record_Count": _block_value(cv_block, "record_count") or 0,
        "CV_Duration_s": _block_value(cv_block, "duration_s"),
        "CV_Average_Voltage_V": _block_value(cv_block, "voltage_mean_v"),
        "CV_Start_Voltage_V": _block_value(cv_block, "voltage_start_v"),
        "CV_End_Voltage_V": _block_value(cv_block, "voltage_end_v"),
        "CV_Start_Current_A": _block_value(cv_block, "current_start_a"),
        "CV_End_Current_A": _block_value(cv_block, "current_end_a"),
        "CV_Charge_Delta_Ah": _block_value(cv_block, "charge_delta_ah"),
        "Main_Discharge_Stage_Present": discharge_present,
        "Main_Discharge_Protocol_Complete": discharge_protocol_complete,
        "Discharge_Step_Index": _block_value(discharge_block, "step_index"),
        "Discharge_Record_Count": _block_value(discharge_block, "record_count") or 0,
        "Main_Discharge_Mean_Current_A": _block_value(discharge_block, "current_mean_a"),
        "Discharge_Start_Voltage_V": _block_value(discharge_block, "voltage_start_v"),
        "Discharge_Cutoff_Voltage_V": discharge_cutoff,
        "Discharge_Duration_s": _block_value(discharge_block, "duration_s"),
        "Main_Discharge_Segment_Delta_Ah": _block_value(discharge_block, "discharge_delta_ah"),
    }


def extract_workbook(
    path: Path,
    thresholds: QCThresholds = QC,
    cell_id: str = CELL_ID,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract one workbook and return its manifest row and cycle rows."""

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        channel_sheet = find_sheet(workbook, "channel")
        if channel_sheet is None:
            raise ValueError("Channel sheet not found")
        statistics_sheet = find_sheet(workbook, "statistics")
        statistics, statistics_meta = _read_statistics(statistics_sheet)
        channel_cycles, channel_meta = _read_channel(channel_sheet)

        cycle_indices = sorted(channel_cycles)
        if not cycle_indices:
            raise ValueError("Channel sheet contains no valid Cycle_Index rows")

        cycle_rows: list[dict[str, Any]] = []
        previous_stats_cycle: int | None = None
        previous_stats_cumulative: float | None = None
        for cycle in cycle_indices:
            stats_row = statistics.get(cycle)
            baseline = None
            if stats_row is not None and previous_stats_cycle is not None and cycle == previous_stats_cycle + 1:
                baseline = previous_stats_cumulative
            metrics = analyze_cycle(channel_cycles[cycle], stats_row, baseline, thresholds)
            row = {
                "Cell_ID": cell_id,
                "Source_File": path.name,
                "Local_Cycle_Index": cycle,
                **metrics,
            }
            cycle_rows.append(row)
            if stats_row is not None:
                previous_stats_cycle = cycle
                previous_stats_cumulative = stats_row.get("discharge_capacity_ah")

        all_records = [record for cycle in cycle_indices for record in channel_cycles[cycle]]
        times = [record["date_time"] for record in all_records if isinstance(record.get("date_time"), datetime)]
        discharge_values = _valid_values(all_records, "discharge_capacity_ah")
        statistics_cycles = sorted(statistics)
        manifest = {
            "Cell_ID": cell_id,
            "File_Name": path.name,
            "Start_Time": iso_time(min(times)) if times else None,
            "End_Time": iso_time(max(times)) if times else None,
            "First_Cycle_Index": cycle_indices[0],
            "Last_Cycle_Index": cycle_indices[-1],
            "Channel_Cycle_Count": len(cycle_indices),
            "Statistics_Cycle_Count": len(statistics_cycles),
            "First_Cumulative_Discharge_Capacity": discharge_values[0] if discharge_values else None,
            "Last_Cumulative_Discharge_Capacity": discharge_values[-1] if discharge_values else None,
            "First_Statistics_Cumulative_Discharge_Capacity_Ah": statistics[statistics_cycles[0]]["discharge_capacity_ah"]
            if statistics_cycles
            else None,
            "Last_Statistics_Cumulative_Discharge_Capacity_Ah": statistics[statistics_cycles[-1]]["discharge_capacity_ah"]
            if statistics_cycles
            else None,
            "Statistics_First_Cycle_Index": statistics_cycles[0] if statistics_cycles else None,
            "Statistics_Last_Cycle_Index": statistics_cycles[-1] if statistics_cycles else None,
            "Statistics_Sheet_Present": statistics_meta["present"],
            "Sheet_Names": "|".join(workbook.sheetnames),
            "Sheet_Shapes": "|".join(
                f"{sheet.title}:{sheet.max_row}x{sheet.max_column}" for sheet in workbook.worksheets
            ),
            "Channel_Repeated_Cycle_Transitions": "|".join(
                map(str, channel_meta["repeated_cycle_transitions"])
            ),
            "Statistics_Duplicate_Cycles": "|".join(map(str, statistics_meta["duplicate_cycles"])),
            "Source_File_SHA256": file_sha256(path),
        }
        return manifest, cycle_rows
    finally:
        workbook.close()


def benchmark_capacities(
    path: Path,
    cycle: int = 2,
    cell_id: str = CELL_ID,
) -> tuple[float, float]:
    """Return independent Statistics and Channel capacities for one cycle."""

    manifest, rows = extract_workbook(path, cell_id=cell_id)
    _ = manifest
    row = next(item for item in rows if item["Local_Cycle_Index"] == cycle)
    stats = row["Statistics_Single_Cycle_Discharge_Capacity_Ah"]
    channel = row["Channel_Single_Cycle_Discharge_Capacity_Ah"]
    if stats is None or channel is None:
        raise ValueError(
            f"Cycle {cycle} benchmark lacks Statistics or Channel capacity"
        )
    return float(stats), float(channel)
