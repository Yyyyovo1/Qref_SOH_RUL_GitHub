from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .matio import MatFileScan, get_field, numeric_1d, text_value, time_to_seconds


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
    capacity_delta_Ah: float


def _contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask.astype(bool), False]))
    return [(int(start), int(stop)) for start, stop in boundaries.reshape(-1, 2)]


def _duration_seconds(time_seconds: np.ndarray) -> float:
    if time_seconds.size < 2:
        return 0.0
    direct = float(time_seconds[-1] - time_seconds[0])
    if np.isfinite(direct) and direct >= 0:
        return direct
    differences = np.diff(time_seconds)
    return float(differences[np.isfinite(differences) & (differences > 0)].sum())


def _segment(
    start: int,
    stop: int,
    current: np.ndarray,
    voltage: np.ndarray,
    capacity: np.ndarray,
    time_seconds: np.ndarray,
) -> Segment:
    current_values = current[start:stop]
    voltage_values = voltage[start:stop]
    capacity_values = capacity[start:stop]
    time_values = time_seconds[start:stop]
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
        capacity_delta_Ah=float(np.max(capacity_values) - np.min(capacity_values)),
    )


def _select_main_segment(
    mask: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    capacity: np.ndarray,
    time_seconds: np.ndarray,
) -> Segment | None:
    candidates = [
        _segment(start, stop, current, voltage, capacity, time_seconds)
        for start, stop in _contiguous_spans(mask)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.capacity_delta_Ah, item.duration_s, item.record_count))


def _aggregate_charge_segment(
    mask: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    capacity: np.ndarray,
    time_seconds: np.ndarray,
    quantile: float,
) -> Segment | None:
    """Summarize the complete charge process while estimating its CC plateau.

    XJTU charge traces contain a long CV tail.  A median of all positive-current
    records therefore severely understates the nominal CC current.  The upper
    current quantile is a deterministic, configuration-controlled estimate of
    the CC plateau; voltage and cutoff current still come from the full positive
    current process.
    """
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    current_values = current[indices]
    voltage_values = voltage[indices]
    capacity_values = capacity[indices]
    cutoff = float(np.quantile(current_values, quantile))
    plateau_values = current_values[current_values >= cutoff]
    cc_estimate = float(np.median(plateau_values))
    start = int(indices[0])
    end = int(indices[-1])
    return Segment(
        start_index=start,
        end_index=end,
        record_count=int(indices.size),
        median_current_A=cc_estimate,
        mean_current_A=float(np.mean(current_values)),
        start_voltage_V=float(voltage_values[0]),
        end_voltage_V=float(voltage_values[-1]),
        min_voltage_V=float(np.min(voltage_values)),
        max_voltage_V=float(np.max(voltage_values)),
        duration_s=_duration_seconds(time_seconds[start : end + 1]),
        capacity_delta_Ah=float(np.max(capacity_values) - np.min(capacity_values)),
    )


def _rest_metrics(
    current: np.ndarray,
    time_seconds: np.ndarray,
    threshold_A: float,
) -> tuple[int, float]:
    spans = _contiguous_spans(np.abs(current) <= threshold_A)
    durations = [
        _duration_seconds(time_seconds[start:stop])
        for start, stop in spans
        if stop - start >= 2
    ]
    material = [duration for duration in durations if duration >= 30.0]
    return len(material), float(sum(material))


def _protocol_lookup(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["Protocol_Class"]): item for item in config["protocols"]}


def _close(value: float, expected: float | None, tolerance: float | None) -> bool:
    if expected is None or tolerance is None:
        return True
    return bool(np.isfinite(value) and abs(value - expected) <= tolerance)


def _description_support(description: str, markers: list[str]) -> bool:
    lowered = description.lower()
    return any(marker.lower() in lowered for marker in markers)


def _mode_match(
    mode: str,
    full_discharge: bool,
    discharge: Segment | None,
    config: Mapping[str, Any],
) -> bool:
    if discharge is None:
        return False
    if mode == "FULL_TO_2P5V":
        return full_discharge
    if mode == "PARTIAL_TO_3P0V_OR_TIME":
        return not full_discharge
    if mode == "PARTIAL_BY_PROTOCOL_DURATION":
        # A duration-controlled GEO aging operation can eventually encounter the
        # safety cutoff as the cell fades.  It remains an aging operation, not a
        # capacity diagnostic.
        return True
    return False


def _special_protocol_match(
    batch_id: int,
    protocol_class: str,
    discharge: Segment | None,
    config: Mapping[str, Any],
) -> tuple[bool, str]:
    if discharge is None:
        return False, "MISSING_MAIN_DISCHARGE"
    median = discharge.median_current_A
    if batch_id == 4 and protocol_class == "AGING_PARTIAL_DISCHARGE":
        allowed = [float(value) for value in config["allowed_partial_discharge_currents_A"]]
        relative = float(config["allowed_partial_current_relative_tolerance"])
        matched = any(abs(median - target) <= max(0.15, abs(target) * relative) for target in allowed)
        return matched, "" if matched else "UNEXPECTED_BATCH4_PARTIAL_DISCHARGE_RATE"
    if batch_id == 5 and protocol_class == "RANDOM_WALK_AGING":
        low, high = [float(value) for value in config["random_walk_discharge_current_abs_range_A"]]
        magnitude = abs(median)
        matched = low - 0.35 <= magnitude <= high + 0.50
        return matched, "" if matched else "UNEXPECTED_RANDOM_WALK_CURRENT"
    return True, ""


def classify_protocol(
    batch_id: int,
    raw_cycle_index: int,
    description: str,
    charge: Segment | None,
    charge_end_current_A: float,
    discharge: Segment | None,
    full_discharge: bool,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    candidates: list[tuple[float, Mapping[str, Any], list[str]]] = []
    for protocol in config["protocols"]:
        protocol_class = str(protocol["Protocol_Class"])
        is_initial = bool(protocol["Initial_Test"])
        if is_initial and raw_cycle_index != 1:
            continue
        if not is_initial and protocol_class != "CAPACITY_DIAGNOSTIC_1C" and raw_cycle_index == 1:
            # Batch-5 may legitimately begin with random walking, so retain its partial class.
            if not (batch_id == 5 and protocol_class == "RANDOM_WALK_AGING"):
                continue

        mode_ok = _mode_match(str(protocol["Discharge_Mode"]), full_discharge, discharge, config)
        discharge_median = discharge.median_current_A if discharge else np.nan
        discharge_ok = _close(
            discharge_median,
            protocol["Expected_Discharge_Current_A"],
            protocol["Discharge_Current_Tolerance_A"],
        )
        charge_median = charge.median_current_A if charge else np.nan
        charge_ok = _close(
            charge_median,
            protocol["Expected_Charge_Current_A"],
            protocol["Charge_Current_Tolerance_A"],
        )
        special_ok, special_issue = _special_protocol_match(
            batch_id, protocol_class, discharge, config
        )
        desc_ok = _description_support(
            description, list(protocol.get("Description_Markers", []))
        )
        max_voltage_ok = bool(
            charge is not None
            and charge.max_voltage_V >= float(config["charge_complete_min_V"]) - 0.05
        )
        expected_cv = protocol["Expected_CV_Cutoff_Upper_A"]
        cv_ok = bool(
            charge is not None
            and (
                expected_cv is None
                or (
                    np.isfinite(charge_end_current_A)
                    and charge_end_current_A <= float(expected_cv)
                )
            )
        )
        weights = config["protocol_classification_weights"]
        score = (
            float(weights["description"]) * float(desc_ok)
            + float(weights["discharge_current"]) * float(discharge_ok)
            + float(weights["discharge_mode"]) * float(mode_ok)
            + float(weights["charge_current"]) * float(charge_ok)
            + float(weights["charge_voltage"]) * float(max_voltage_ok)
            + float(weights["cv_cutoff"]) * float(cv_ok)
        )
        issue_codes: list[str] = []
        if not mode_ok:
            issue_codes.append("DISCHARGE_MODE_MISMATCH")
        if not discharge_ok:
            issue_codes.append("DISCHARGE_CURRENT_PROTOCOL_DEVIATION")
        if not charge_ok:
            issue_codes.append("CHARGE_CURRENT_PROTOCOL_DEVIATION")
        if not max_voltage_ok:
            issue_codes.append("CHARGE_VOLTAGE_PROTOCOL_DEVIATION")
        if not cv_ok:
            issue_codes.append("CHARGE_CV_CUTOFF_DEVIATION")
        if not special_ok and special_issue:
            issue_codes.append(special_issue)
            score -= 0.15
        candidates.append((score, protocol, issue_codes))

    if not candidates:
        return {
            "Protocol_Class": "UNKNOWN_PROTOCOL",
            "Primary_Capacity_Eligible": "REVIEW",
            "Primary_Capacity_Reason": "NO_PROTOCOL_CANDIDATE",
            "Protocol_Confidence": "LOW",
            "Protocol_Confidence_Score": 0.0,
            "Expected_Protocol_Match": "REVIEW",
            "Classification_Issues": ["NO_PROTOCOL_CANDIDATE"],
            "Protocol_Config": None,
        }
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best, best_issues = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else -1.0
    ambiguity = best_score - second_score < 0.08
    threshold = float(config["minimum_protocol_confidence_for_auto_match"])
    if best_score >= threshold and not ambiguity:
        confidence = "HIGH" if best_score >= 0.90 else "MEDIUM"
        critical_issues = {
            "DISCHARGE_MODE_MISMATCH",
            "DISCHARGE_CURRENT_PROTOCOL_DEVIATION",
            "CHARGE_CURRENT_PROTOCOL_DEVIATION",
            "CHARGE_VOLTAGE_PROTOCOL_DEVIATION",
            "CHARGE_CV_CUTOFF_DEVIATION",
            "MISSING_MAIN_DISCHARGE",
        }
        expected_match = (
            "REVIEW" if critical_issues.intersection(best_issues) else "YES"
        )
        protocol_class = str(best["Protocol_Class"])
        configured_eligibility = str(best["Primary_Capacity_Eligible"])
        if expected_match == "YES" or configured_eligibility == "NO":
            eligibility = configured_eligibility
            reason = str(best["Primary_Capacity_Reason"])
        else:
            eligibility = "REVIEW"
            reason = "PROTOCOL_DEVIATION_REQUIRES_REVIEW"
    elif best_score >= 0.55 and not ambiguity:
        confidence = "LOW"
        expected_match = "REVIEW"
        protocol_class = str(best["Protocol_Class"])
        eligibility = "REVIEW"
        reason = "LOW_CONFIDENCE_PROTOCOL_CLASSIFICATION"
        best_issues = [*best_issues, "LOW_CONFIDENCE_PROTOCOL_CLASSIFICATION"]
    else:
        confidence = "LOW"
        expected_match = "REVIEW"
        protocol_class = "UNKNOWN_PROTOCOL"
        eligibility = "REVIEW"
        reason = "AMBIGUOUS_OR_UNMATCHED_PROTOCOL"
        best_issues = [*best_issues, "AMBIGUOUS_PROTOCOL_CLASSIFICATION" if ambiguity else "UNMATCHED_PROTOCOL"]
        best = None

    # A GEO cycle can start at high SOC after a shallow prior discharge and
    # therefore enter CV before a sustained 4 A plateau is observable.  Once
    # the configured charge voltage and CV terminal-current criteria are met,
    # the missing plateau is retained as information rather than review.
    if (
        batch_id == 6
        and protocol_class == "GEO_AGING_PARTIAL"
        and best is not None
        and charge is not None
        and "CHARGE_CURRENT_PROTOCOL_DEVIATION" in best_issues
    ):
        other_charge_faults = {
            "CHARGE_VOLTAGE_PROTOCOL_DEVIATION",
            "CHARGE_CV_CUTOFF_DEVIATION",
            "MISSING_MAIN_CHARGE",
        }.intersection(best_issues)
        expected_cv = best.get("Expected_CV_Cutoff_Upper_A")
        charge_complete = charge.max_voltage_V >= float(
            config["charge_complete_min_V"]
        )
        cv_complete = bool(
            expected_cv is not None
            and np.isfinite(charge_end_current_A)
            and abs(charge_end_current_A) <= float(expected_cv)
        )
        if charge_complete and cv_complete and not other_charge_faults:
            best_issues = [
                "CC_PLATEAU_NOT_OBSERVED_HIGH_SOC_START"
                if code == "CHARGE_CURRENT_PROTOCOL_DEVIATION"
                else code
                for code in best_issues
            ]
            remaining_critical = {
                "DISCHARGE_MODE_MISMATCH",
                "DISCHARGE_CURRENT_PROTOCOL_DEVIATION",
                "CHARGE_VOLTAGE_PROTOCOL_DEVIATION",
                "CHARGE_CV_CUTOFF_DEVIATION",
                "MISSING_MAIN_DISCHARGE",
            }.intersection(best_issues)
            if not remaining_critical:
                expected_match = "YES"
                eligibility = "NO"
                reason = str(best["Primary_Capacity_Reason"])
    return {
        "Protocol_Class": protocol_class,
        "Primary_Capacity_Eligible": eligibility,
        "Primary_Capacity_Reason": reason,
        "Protocol_Confidence": confidence,
        "Protocol_Confidence_Score": max(0.0, float(best_score)),
        "Expected_Protocol_Match": expected_match,
        "Classification_Issues": list(dict.fromkeys(best_issues)),
        "Protocol_Config": best,
    }


def scan_protocol_cycles(
    file_scan: MatFileScan,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if file_scan.adapter is None or file_scan.summary_capacities_Ah is None:
        return []
    adapter = file_scan.adapter
    rows: list[dict[str, Any]] = []
    protocols = _protocol_lookup(config)
    for raw_index, cycle in enumerate(file_scan.cycles, start=1):
        description = text_value(get_field(cycle, adapter.description_field))
        voltage = numeric_1d(get_field(cycle, adapter.voltage_field))
        current = numeric_1d(get_field(cycle, adapter.current_field))
        capacity = numeric_1d(get_field(cycle, adapter.capacity_field))
        time_raw = numeric_1d(get_field(cycle, adapter.time_field))
        time_seconds = time_to_seconds(time_raw, adapter.time_unit)
        positive_charge_mask = current > float(
            config["charge_current_detection_threshold_A"]
        )
        charge = _aggregate_charge_segment(
            positive_charge_mask,
            current,
            voltage,
            capacity,
            time_seconds,
            float(config["charge_cc_estimation_quantile"]),
        )
        discharge = _select_main_segment(
            current < float(config["discharge_current_detection_threshold_A"]),
            current,
            voltage,
            capacity,
            time_seconds,
        )
        charge_end_current = float(current[charge.end_index]) if charge is not None else np.nan
        full_discharge = bool(
            discharge is not None
            and discharge.min_voltage_V
            <= float(config["full_discharge_min_voltage_max_V"])
            and discharge.end_voltage_V
            <= float(config["full_discharge_end_voltage_max_V"])
        )
        classification = classify_protocol(
            file_scan.batch_id,
            raw_index,
            description,
            charge,
            charge_end_current,
            discharge,
            full_discharge,
            config,
        )
        protocol_config = classification["Protocol_Config"]
        issue_codes = list(classification["Classification_Issues"])

        # Batch-5 has two random-walk charge regimes.  The expected current is
        # selected from description text, but the classification still relies
        # on raw discharge mode/current and other signals as well.
        description_expected_charge = None
        description_charge_tolerance = None
        for expectation in config.get("description_charge_expectations", []):
            if str(expectation["description_marker"]).lower() in description.lower():
                description_expected_charge = float(expectation["expected_current_A"])
                description_charge_tolerance = float(expectation["tolerance_A"])
                break
        if (
            description_expected_charge is not None
            and charge is not None
            and abs(charge.median_current_A - description_expected_charge)
            > float(description_charge_tolerance)
        ):
            issue_codes.append("DESCRIPTION_CHARGE_CURRENT_MISMATCH")
            known_batch5_metadata_mismatch = bool(
                file_scan.batch_id == 5
                and classification["Protocol_Class"] == "RANDOM_WALK_AGING"
                and classification["Primary_Capacity_Eligible"] == "NO"
            )
            if not known_batch5_metadata_mismatch:
                classification["Expected_Protocol_Match"] = "REVIEW"
            if (
                not known_batch5_metadata_mismatch
                and classification["Primary_Capacity_Eligible"] == "YES"
            ):
                classification["Primary_Capacity_Eligible"] = "REVIEW"
                classification["Primary_Capacity_Reason"] = (
                    "DESCRIPTION_CHARGE_CURRENT_MISMATCH_REQUIRES_REVIEW"
                )
        rest_count, rest_duration = _rest_metrics(
            current, time_seconds, float(config["rest_current_abs_threshold_A"])
        )
        voltage_spike = False
        abnormal_jump = False
        max_jump = np.nan
        if discharge is not None:
            start, stop = discharge.start_index, discharge.end_index + 1
            segment_voltage = voltage[start:stop]
            segment_current = current[start:stop]
            voltage_spike = bool(
                np.any(
                    (segment_current < float(config["voltage_spike_current_threshold_A"]))
                    & (segment_voltage > float(config["voltage_spike_threshold_V"]))
                )
            )
            differences = np.abs(np.diff(segment_voltage))
            max_jump = float(differences.max()) if differences.size else 0.0
            abnormal_jump = bool(max_jump > float(config["voltage_jump_threshold_V"]))
        if charge is None:
            issue_codes.append("MISSING_MAIN_CHARGE")
        if discharge is None:
            issue_codes.append("MISSING_MAIN_DISCHARGE")
        if voltage_spike:
            issue_codes.append("VOLTAGE_SPIKE_DURING_DISCHARGE")
        if abnormal_jump:
            issue_codes.append("ABNORMAL_VOLTAGE_JUMP")
        if (
            file_scan.batch_id == 6
            and classification["Protocol_Class"] == "GEO_AGING_PARTIAL"
            and full_discharge
        ):
            issue_codes.append("GEO_AGING_REACHED_FULL_CUTOFF")
        if not description:
            issue_codes.append("EMPTY_DESCRIPTION")

        summary_capacity = float(file_scan.summary_capacities_Ah[raw_index - 1])
        raw_capacity = discharge.capacity_delta_Ah if discharge else np.nan
        capacity_difference = raw_capacity - summary_capacity
        capacity_difference_pct = (
            100.0 * capacity_difference / summary_capacity
            if np.isfinite(capacity_difference) and summary_capacity != 0
            else np.nan
        )
        expected_charge = (
            protocol_config["Expected_Charge_Current_A"] if protocol_config else np.nan
        )
        if description_expected_charge is not None:
            expected_charge = description_expected_charge
        expected_cv = (
            protocol_config["Expected_CV_Cutoff_Upper_A"] if protocol_config else np.nan
        )
        expected_discharge = (
            protocol_config["Expected_Discharge_Current_A"] if protocol_config else np.nan
        )
        expected_cutoff = (
            2.5
            if protocol_config and protocol_config["Discharge_Mode"] == "FULL_TO_2P5V"
            else (
                3.0
                if protocol_config
                and protocol_config["Discharge_Mode"] == "PARTIAL_TO_3P0V_OR_TIME"
                else np.nan
            )
        )
        rows.append(
            {
                "Batch_ID": file_scan.batch_id,
                "Cell_ID": file_scan.cell_id,
                "Source_File": file_scan.source_file,
                "Schema_Variant_ID": adapter.variant_id,
                "Raw_Cycle_Index": raw_index,
                "Description": description,
                "Summary_Discharge_Capacity_Ah": summary_capacity,
                "Raw_Discharge_Capacity_Ah": raw_capacity,
                "Capacity_Difference_Ah": capacity_difference,
                "Capacity_Difference_Pct": capacity_difference_pct,
                "Observed_Charge_Current_A": charge.median_current_A if charge else np.nan,
                "Observed_Charge_Max_V": charge.max_voltage_V if charge else np.nan,
                "Observed_Charge_End_Current_A": charge_end_current,
                "Observed_Charge_Duration_s": charge.duration_s if charge else np.nan,
                "Observed_Discharge_Median_Current_A": discharge.median_current_A if discharge else np.nan,
                "Observed_Discharge_Min_V": discharge.min_voltage_V if discharge else np.nan,
                "Observed_Discharge_End_V": discharge.end_voltage_V if discharge else np.nan,
                "Observed_Discharge_Duration_s": discharge.duration_s if discharge else np.nan,
                "Observed_Rest_Segment_Count": rest_count,
                "Observed_Total_Rest_Duration_s": rest_duration,
                "Full_Discharge_Observed": full_discharge,
                "Voltage_Spike_During_Discharge": voltage_spike,
                "Abnormal_Voltage_Jump": abnormal_jump,
                "Max_Abs_Voltage_Jump_V": max_jump,
                "Expected_Charge_Current_A": expected_charge,
                "Expected_CV_Cutoff_A": expected_cv,
                "Expected_Discharge_Current_A": expected_discharge,
                "Expected_Discharge_Cutoff_V": expected_cutoff,
                "Protocol_Class": classification["Protocol_Class"],
                "Primary_Capacity_Eligible": classification["Primary_Capacity_Eligible"],
                "Primary_Capacity_Reason": classification["Primary_Capacity_Reason"],
                "Protocol_Confidence": classification["Protocol_Confidence"],
                "Protocol_Confidence_Score": classification["Protocol_Confidence_Score"],
                "Expected_Protocol_Match": classification["Expected_Protocol_Match"],
                "Issues": "|".join(dict.fromkeys(issue_codes)),
            }
        )
    return assign_protocol_indices(rows, config)


def assign_protocol_indices(
    rows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    protocol_lookup = _protocol_lookup(config)
    aging_count = 0
    diagnostic_count = 0
    for row in sorted(rows, key=lambda item: int(item["Raw_Cycle_Index"])):
        row["Aging_Operations_Before_Cycle"] = aging_count
        protocol = protocol_lookup.get(str(row["Protocol_Class"]))
        counts_as_aging = bool(protocol and protocol["Counts_As_Aging_Operation"])
        recurring = bool(protocol and protocol["Recurring_Comparable"])
        if counts_as_aging:
            aging_count += 1
            row["Aging_Operation_Index"] = aging_count
        else:
            row["Aging_Operation_Index"] = pd.NA
        row["Cumulative_Aging_Operation_Count"] = aging_count
        if recurring and row["Primary_Capacity_Eligible"] == "YES":
            diagnostic_count += 1
            row["Diagnostic_Index"] = diagnostic_count
        else:
            row["Diagnostic_Index"] = pd.NA
    return rows


def protocol_rows_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "Batch_ID",
        "Cell_ID",
        "Source_File",
        "Schema_Variant_ID",
        "Raw_Cycle_Index",
        "Aging_Operation_Index",
        "Diagnostic_Index",
        "Aging_Operations_Before_Cycle",
        "Cumulative_Aging_Operation_Count",
        "Description",
        "Summary_Discharge_Capacity_Ah",
        "Raw_Discharge_Capacity_Ah",
        "Capacity_Difference_Ah",
        "Capacity_Difference_Pct",
        "Observed_Charge_Current_A",
        "Observed_Charge_Max_V",
        "Observed_Charge_End_Current_A",
        "Observed_Charge_Duration_s",
        "Observed_Discharge_Median_Current_A",
        "Observed_Discharge_Min_V",
        "Observed_Discharge_End_V",
        "Observed_Discharge_Duration_s",
        "Observed_Rest_Segment_Count",
        "Observed_Total_Rest_Duration_s",
        "Full_Discharge_Observed",
        "Voltage_Spike_During_Discharge",
        "Abnormal_Voltage_Jump",
        "Max_Abs_Voltage_Jump_V",
        "Expected_Charge_Current_A",
        "Expected_CV_Cutoff_A",
        "Expected_Discharge_Current_A",
        "Expected_Discharge_Cutoff_V",
        "Protocol_Class",
        "Primary_Capacity_Eligible",
        "Primary_Capacity_Reason",
        "Protocol_Confidence",
        "Protocol_Confidence_Score",
        "Expected_Protocol_Match",
        "Issues",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        for column in (
            "Batch_ID",
            "Raw_Cycle_Index",
            "Aging_Operation_Index",
            "Diagnostic_Index",
            "Aging_Operations_Before_Cycle",
            "Cumulative_Aging_Operation_Count",
        ):
            frame[column] = pd.array(frame[column], dtype="Int64")
        frame = frame.sort_values(["Batch_ID", "Cell_ID", "Raw_Cycle_Index"]).reset_index(drop=True)
    return frame


def description_catalog(protocol_inventory: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Batch_ID",
        "Description",
        "Occurrence_Count",
        "Cell_Count",
        "Sample_Cell_ID",
        "Sample_Raw_Cycles",
        "Detected_Protocol_Classes",
        "Median_Discharge_Current_A",
        "Median_Discharge_Min_V",
    ]
    if protocol_inventory.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (batch_id, description), group in protocol_inventory.groupby(
        ["Batch_ID", "Description"], dropna=False, sort=True
    ):
        sample = group.iloc[0]
        rows.append(
            {
                "Batch_ID": batch_id,
                "Description": description,
                "Occurrence_Count": len(group),
                "Cell_Count": group["Cell_ID"].nunique(),
                "Sample_Cell_ID": sample["Cell_ID"],
                "Sample_Raw_Cycles": "|".join(
                    str(int(value)) for value in group["Raw_Cycle_Index"].head(10)
                ),
                "Detected_Protocol_Classes": "|".join(
                    sorted(set(group["Protocol_Class"].astype(str)))
                ),
                "Median_Discharge_Current_A": pd.to_numeric(
                    group["Observed_Discharge_Median_Current_A"], errors="coerce"
                ).median(),
                "Median_Discharge_Min_V": pd.to_numeric(
                    group["Observed_Discharge_Min_V"], errors="coerce"
                ).median(),
            }
        )
    return pd.DataFrame(rows, columns=columns)
