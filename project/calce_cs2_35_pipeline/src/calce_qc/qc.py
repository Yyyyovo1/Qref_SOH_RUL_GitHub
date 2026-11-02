"""Protocol-first QC rules for CALCE CS2-35 cycle records."""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

import pandas as pd

from .config import KNOWN_BOUNDARY_TAILS, QC, QCThresholds


SEVERITY = {"PASS": 0, "CHECK": 1, "INVALID_CANDIDATE": 2}


def _finite(value: Any) -> bool:
    try:
        return value is not None and isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _format(value: Any, digits: int = 4) -> str:
    return "n.a." if not _finite(value) else f"{float(value):.{digits}f}"


def add_issue(row: dict[str, Any], severity: str, code: str, reason: str) -> None:
    row.setdefault("_QC_Issues", []).append((severity, code, reason))


def add_protocol_issues(
    row: dict[str, Any],
    thresholds: QCThresholds = QC,
    known_boundary_tails: Mapping[tuple[str, int], str] = KNOWN_BOUNDARY_TAILS,
) -> None:
    """Apply protocol and counter checks to one cycle; never remove a row."""

    row["Incomplete_CV_Flag"] = False
    row["Incomplete_Discharge_Flag"] = False
    row["Nonpositive_Capacity_Flag"] = False
    row["Capacity_Jump_Flag"] = False

    if not bool(row.get("CC_Stage_Present")):
        add_issue(
            row,
            "INVALID_CANDIDATE",
            "MISSING_CC_CHARGE",
            "missing CC charging stage based on current, voltage, time, and charge-capacity evidence",
        )
    elif not bool(row.get("CC_Protocol_Complete")):
        add_issue(
            row,
            "CHECK",
            "SHORT_OR_INCOMPLETE_CC_CHARGE",
            "CC charging stage unusually short or incomplete "
            f"(records={row.get('CC_Record_Count')}, duration={_format(row.get('CC_Duration_s'), 1)} s, "
            f"delta_Q={_format(row.get('CC_Charge_Delta_Ah'), 6)} Ah, "
            f"end_voltage={_format(row.get('CC_End_Voltage_V'))} V)",
        )

    if not bool(row.get("CV_Stage_Present")):
        row["Incomplete_CV_Flag"] = True
        add_issue(
            row,
            "INVALID_CANDIDATE",
            "INCOMPLETE_CV_CHARGING_PROTOCOL",
            "incomplete CV charging protocol "
            f"(records={row.get('CV_Record_Count')}, duration={_format(row.get('CV_Duration_s'), 1)} s, "
            f"mean_voltage={_format(row.get('CV_Average_Voltage_V'))} V, "
            f"delta_Q={_format(row.get('CV_Charge_Delta_Ah'), 6)} Ah; "
            f"selection={row.get('CV_Selection_Basis')})",
        )
    elif not bool(row.get("CV_Protocol_Complete")):
        row["Incomplete_CV_Flag"] = True
        add_issue(
            row,
            "INVALID_CANDIDATE",
            "INCOMPLETE_CV_CHARGING_PROTOCOL",
            "incomplete CV charging protocol: current did not taper to the configured cutoff "
            f"(start={_format(row.get('CV_Start_Current_A'))} A, "
            f"end={_format(row.get('CV_End_Current_A'))} A)",
        )

    if not bool(row.get("Main_Discharge_Stage_Present")):
        row["Incomplete_Discharge_Flag"] = True
        add_issue(
            row,
            "INVALID_CANDIDATE",
            "MISSING_MAIN_DISCHARGE",
            "main discharge stage missing or too short "
            f"(records={row.get('Discharge_Record_Count')}, "
            f"duration={_format(row.get('Discharge_Duration_s'), 1)} s)",
        )
    else:
        mean_current = row.get("Main_Discharge_Mean_Current_A")
        if _finite(mean_current) and not (
            thresholds.discharge_mean_current_min_a
            <= float(mean_current)
            <= thresholds.discharge_mean_current_max_a
        ):
            add_issue(
                row,
                "CHECK",
                "UNUSUAL_DISCHARGE_CURRENT",
                f"main discharge mean current is {_format(mean_current)} A; expected approximately -1.1 A",
            )

        start_voltage = row.get("Discharge_Start_Voltage_V")
        if _finite(start_voltage) and not (
            thresholds.discharge_start_voltage_min_v
            <= float(start_voltage)
            <= thresholds.discharge_start_voltage_max_v
        ):
            add_issue(
                row,
                "CHECK",
                "UNUSUAL_DISCHARGE_START_VOLTAGE",
                f"discharge start voltage is {_format(start_voltage)} V",
            )

        cutoff = row.get("Discharge_Cutoff_Voltage_V")
        if not _finite(cutoff):
            row["Incomplete_Discharge_Flag"] = True
            add_issue(
                row,
                "INVALID_CANDIDATE",
                "MISSING_DISCHARGE_CUTOFF",
                "discharge cutoff voltage is unavailable",
            )
        elif float(cutoff) > thresholds.discharge_cutoff_voltage_max_v:
            row["Incomplete_Discharge_Flag"] = True
            add_issue(
                row,
                "INVALID_CANDIDATE",
                "INCOMPLETE_DISCHARGE_PROTOCOL",
                "incomplete discharge protocol: discharge ended at "
                f"{_format(cutoff)} V instead of approximately 2.7 V",
            )
        elif float(cutoff) < thresholds.discharge_cutoff_voltage_min_v:
            add_issue(
                row,
                "CHECK",
                "LOW_DISCHARGE_CUTOFF_VOLTAGE",
                f"discharge cutoff voltage is unusually low ({_format(cutoff)} V)",
            )

    capacity = row.get("Single_Cycle_Discharge_Capacity_Ah")
    if not _finite(capacity):
        add_issue(
            row,
            "CHECK",
            "MISSING_CAPACITY",
            "single-cycle discharge capacity is unavailable",
        )
    elif float(capacity) <= 0.0:
        row["Nonpositive_Capacity_Flag"] = True
        add_issue(
            row,
            "CHECK",
            "NONPOSITIVE_CAPACITY",
            f"single-cycle discharge capacity is nonpositive ({_format(capacity, 8)} Ah)",
        )

    if int(row.get("Discharge_Counter_Decrease_Count") or 0) > 0:
        add_issue(
            row,
            "CHECK",
            "WITHIN_CYCLE_COUNTER_DECREASE",
            "Discharge_Capacity counter decreases within the cycle "
            f"({row.get('Discharge_Counter_Decrease_Count')} occurrence(s))",
        )

    agreement = row.get("Capacity_Agreement_Delta_Ah")
    if _finite(agreement) and abs(float(agreement)) > thresholds.stats_channel_agreement_tolerance_ah:
        add_issue(
            row,
            "CHECK",
            "STATISTICS_CHANNEL_CAPACITY_MISMATCH",
            "Statistics and Channel single-cycle capacities differ by "
            f"{_format(agreement, 8)} Ah",
        )

    key = (str(row.get("Source_File")), int(row.get("Local_Cycle_Index")))
    if key in known_boundary_tails:
        add_issue(
            row,
            "CHECK",
            "NEEDS_BOUNDARY_CHECK",
            f"NEEDS_BOUNDARY_CHECK: {known_boundary_tails[key]}; cycle retained and not stitched",
        )

    if str(row.get("File_Role")) != "CANONICAL":
        add_issue(
            row,
            "CHECK",
            "DUPLICATE_FILE_CANDIDATE",
            "duplicate/reference workbook retained; Global_Cycle_Index intentionally left blank",
        )


def _add_local_cycle_sequence_issues(rows: list[dict[str, Any]]) -> None:
    by_file: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_file.setdefault(str(row["Source_File"]), []).append(row)
    for file_rows in by_file.values():
        file_rows.sort(key=lambda row: int(row["Local_Cycle_Index"]))
        previous: int | None = None
        for row in file_rows:
            current = int(row["Local_Cycle_Index"])
            if previous is not None and current > previous + 1:
                add_issue(
                    row,
                    "CHECK",
                    "LOCAL_CYCLE_INDEX_GAP",
                    f"Cycle_Index gap before this cycle: previous={previous}, current={current}",
                )
            elif previous is not None and current <= previous:
                add_issue(
                    row,
                    "CHECK",
                    "LOCAL_CYCLE_INDEX_DUPLICATE_OR_REORDERED",
                    f"Cycle_Index duplicate or reversal: previous={previous}, current={current}",
                )
            previous = current


def _has_invalid_candidate_issue(row: dict[str, Any]) -> bool:
    return any(
        severity == "INVALID_CANDIDATE"
        for severity, _code, _reason in row.get("_QC_Issues", [])
    )


def _is_valid_jump_reference(row: dict[str, Any]) -> bool:
    """Return whether a cycle can anchor a later capacity-jump comparison."""

    capacity = row.get("Single_Cycle_Discharge_Capacity_Ah")
    return bool(
        str(row.get("File_Role")) == "CANONICAL"
        and str(row.get("Boundary_Status")) == "CONFIRMED"
        and _finite(capacity)
        and float(capacity) > 0.0
        and not bool(row.get("Incomplete_CV_Flag"))
        and not bool(row.get("Incomplete_Discharge_Flag"))
        and not bool(row.get("Nonpositive_Capacity_Flag"))
        and bool(row.get("Main_Discharge_Protocol_Complete"))
        and not _has_invalid_candidate_issue(row)
    )


def _add_capacity_jump_issues(rows: list[dict[str, Any]], thresholds: QCThresholds) -> None:
    """Compare each cycle with the nearest preceding valid reference cycle."""

    for row in rows:
        row["Jump_Reference_Status"] = "NOT_EVALUATED_NONCANONICAL"
        row["Jump_Reference_Global_Cycle"] = None
        row["Jump_Reference_Capacity_Ah"] = None
        row["Jump_Delta_Ah"] = None
        row["Jump_Reference_Reason"] = (
            "File_Role is not CANONICAL or Global_Cycle_Index is unavailable"
        )

    canonical = [
        row
        for row in rows
        if str(row.get("File_Role")) == "CANONICAL"
        and _finite(row.get("Global_Cycle_Index"))
    ]
    canonical.sort(key=lambda row: int(row["Global_Cycle_Index"]))

    last_valid_reference: dict[str, Any] | None = None
    for current in canonical:
        current_capacity = current.get("Single_Cycle_Discharge_Capacity_Ah")
        if last_valid_reference is None:
            current["Jump_Reference_Status"] = "NO_VALID_JUMP_REFERENCE"
            current["Jump_Reference_Reason"] = (
                "no earlier canonical cycle satisfies all jump-reference validity criteria"
            )
        else:
            reference_global = int(last_valid_reference["Global_Cycle_Index"])
            reference_capacity = float(
                last_valid_reference["Single_Cycle_Discharge_Capacity_Ah"]
            )
            current["Jump_Reference_Status"] = "VALID_JUMP_REFERENCE"
            current["Jump_Reference_Global_Cycle"] = reference_global
            current["Jump_Reference_Capacity_Ah"] = reference_capacity
            current["Jump_Reference_Reason"] = (
                "nearest preceding cycle satisfying all jump-reference validity criteria"
            )

            if _finite(current_capacity):
                delta = float(current_capacity) - reference_capacity
                current["Jump_Delta_Ah"] = delta
                relative = abs(delta) / max(abs(reference_capacity), 1.0e-12)
                if (
                    abs(delta) >= thresholds.capacity_jump_abs_ah
                    and relative >= thresholds.capacity_jump_relative
                ):
                    current["Capacity_Jump_Flag"] = True
                    add_issue(
                        current,
                        "CHECK",
                        "CAPACITY_JUMP",
                        "capacity jump relative to nearest preceding valid comparable cycle "
                        f"(reference_global={reference_global}, "
                        f"reference={reference_capacity:.6f} Ah, "
                        f"current_global={int(current['Global_Cycle_Index'])}, "
                        f"current={float(current_capacity):.6f} Ah, "
                        f"delta={delta:+.6f} Ah); "
                        "capacity change alone does not invalidate a cycle",
                    )
            else:
                current["Jump_Reference_Reason"] += (
                    "; current cycle capacity is unavailable, so no jump was evaluated"
                )

        if _is_valid_jump_reference(current):
            last_valid_reference = current


def finalize_qc(
    rows: list[dict[str, Any]],
    thresholds: QCThresholds = QC,
    known_boundary_tails: Mapping[tuple[str, int], str] = KNOWN_BOUNDARY_TAILS,
) -> pd.DataFrame:
    """Apply all QC rules and return a stable, flat output table."""

    for row in rows:
        add_protocol_issues(row, thresholds, known_boundary_tails)
    _add_local_cycle_sequence_issues(rows)
    _add_capacity_jump_issues(rows, thresholds)

    for row in rows:
        issues = row.pop("_QC_Issues", [])
        if issues:
            highest = max(issues, key=lambda issue: SEVERITY[issue[0]])[0]
            row["QC_Flag"] = highest
            row["QC_Reason_Codes"] = "|".join(dict.fromkeys(issue[1] for issue in issues))
            row["QC_Reason"] = "; ".join(dict.fromkeys(issue[2] for issue in issues))
        else:
            row["QC_Flag"] = "PASS"
            row["QC_Reason_Codes"] = "PASS"
            row["QC_Reason"] = "protocol checks passed"

    frame = pd.DataFrame(rows)
    if "Global_Cycle_Index" in frame:
        frame["Global_Cycle_Index"] = pd.array(frame["Global_Cycle_Index"], dtype="Int64")
    if "Jump_Reference_Global_Cycle" in frame:
        frame["Jump_Reference_Global_Cycle"] = pd.array(
            frame["Jump_Reference_Global_Cycle"], dtype="Int64"
        )
    return frame
