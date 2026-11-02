from __future__ import annotations

import json
import platform
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import scipy

from . import PIPELINE_VERSION
from .analysis import _atomic_directory, _load_json, _sha256, _write_csv, detect_eol
from .config_templates import EXPECTED_CANONICAL_COUNTS
from .scan import output_hash_manifest


VALIDATION_STATUS_PASSED = "VALIDATION_PASSED_AWAITING_HUMAN_RUN_APPROVAL"
VALIDATION_STATUS_FAILED = "VALIDATION_FAILED_DO_NOT_RUN"


@dataclass(frozen=True)
class ValidationResult:
    output_dir: Path
    passed: bool
    regression_test_count: int
    failed_test_count: int
    info_anomaly_count: int
    review_anomaly_count: int


def _split_codes(value: object) -> list[str]:
    return [code for code in str(value or "").split("|") if code]


def _join_codes(values: list[str]) -> str:
    return "|".join(dict.fromkeys(value for value in values if value))


def _number(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _is_yes(value: object) -> bool:
    return str(value).strip().upper() == "YES"


def resolve_known_protocol_information(
    inventory: pd.DataFrame,
    configs: Mapping[int, Mapping[str, Any]],
) -> pd.DataFrame:
    """Apply only the three manually approved informational anomaly rules.

    Original scan columns are never changed.  Validated columns show the
    deterministic resolution and leave every unknown/unexplained condition at
    REVIEW.
    """
    result = inventory.copy()
    result["Validated_Protocol_Class"] = result["Protocol_Class"]
    result["Validated_Primary_Capacity_Eligible"] = result[
        "Primary_Capacity_Eligible"
    ]
    result["Validated_Expected_Protocol_Match"] = result[
        "Expected_Protocol_Match"
    ]
    result["Validated_Issue_Codes"] = result["Issues"].fillna("")
    result["Validation_Resolution_Codes"] = ""
    result["Validation_Severity"] = "NONE"
    result["Validation_Requires_Human_Review"] = "NO"
    result["Validation_Status"] = "AUTO_MATCH"
    result["Validation_Notes"] = ""

    for index, row in result.iterrows():
        batch_id = int(row["Batch_ID"])
        protocol_class = str(row["Protocol_Class"])
        eligibility = str(row["Primary_Capacity_Eligible"])
        original_codes = _split_codes(row.get("Issues", ""))
        unresolved = list(original_codes)
        resolved: list[str] = []
        notes: list[str] = []

        if (
            batch_id == 5
            and protocol_class == "RANDOM_WALK_AGING"
            and eligibility == "NO"
            and "DESCRIPTION_CHARGE_CURRENT_MISMATCH" in unresolved
        ):
            unresolved.remove("DESCRIPTION_CHARGE_CURRENT_MISMATCH")
            resolved.append("DESCRIPTION_CHARGE_CURRENT_MISMATCH")
            notes.append(
                "Known Batch-5 description/current metadata mismatch; random-walk "
                "cycle remains nonprimary and counts as an aging operation."
            )

        if (
            batch_id == 6
            and protocol_class == "GEO_AGING_PARTIAL"
            and eligibility == "NO"
            and (
                "CHARGE_CURRENT_PROTOCOL_DEVIATION" in unresolved
                or "CC_PLATEAU_NOT_OBSERVED_HIGH_SOC_START" in unresolved
            )
        ):
            config = configs[6]
            charge_max = _number(row.get("Observed_Charge_Max_V", np.nan))
            charge_end = abs(_number(row.get("Observed_Charge_End_Current_A", np.nan)))
            max_voltage_ok = bool(
                np.isfinite(charge_max)
                and charge_max >= float(config["charge_complete_min_V"])
            )
            cv_limit = float(
                config.get("geo_high_soc_cv_terminal_current_upper_A", 0.13)
            )
            cv_ok = bool(np.isfinite(charge_end) and charge_end <= cv_limit)
            other_charge_faults = {
                "CHARGE_VOLTAGE_PROTOCOL_DEVIATION",
                "CHARGE_CV_CUTOFF_DEVIATION",
                "MISSING_MAIN_CHARGE",
            }.intersection(unresolved)
            if max_voltage_ok and cv_ok and not other_charge_faults:
                if "CHARGE_CURRENT_PROTOCOL_DEVIATION" in unresolved:
                    unresolved.remove("CHARGE_CURRENT_PROTOCOL_DEVIATION")
                if "CC_PLATEAU_NOT_OBSERVED_HIGH_SOC_START" in unresolved:
                    unresolved.remove("CC_PLATEAU_NOT_OBSERVED_HIGH_SOC_START")
                resolved.append("CC_PLATEAU_NOT_OBSERVED_HIGH_SOC_START")
                notes.append(
                    "GEO charge reached the configured voltage and CV terminal-current "
                    "criteria; an unobserved 4 A plateau is informational at high SOC."
                )

        if (
            batch_id == 6
            and protocol_class == "GEO_AGING_PARTIAL"
            and eligibility == "NO"
            and "GEO_AGING_REACHED_FULL_CUTOFF" in unresolved
        ):
            other_true_faults = set(unresolved) - {
                "GEO_AGING_REACHED_FULL_CUTOFF",
                "CHARGE_CURRENT_PROTOCOL_DEVIATION",
            }
            if not other_true_faults:
                unresolved.remove("GEO_AGING_REACHED_FULL_CUTOFF")
                resolved.append("GEO_AGING_REACHED_FULL_CUTOFF")
                notes.append(
                    "GEO aging reached the low-voltage cutoff but remains a nonprimary "
                    "GEO cycle; no reclassification to a capacity diagnostic."
                )

        if str(row["Protocol_Class"]) == "UNKNOWN_PROTOCOL":
            unresolved.append("UNKNOWN_PROTOCOL")
        if eligibility == "REVIEW":
            unresolved.append("PRIMARY_CAPACITY_ELIGIBILITY_UNRESOLVED")
        if str(row["Expected_Protocol_Match"]) != "YES" and not unresolved and not resolved:
            unresolved.append("UNEXPLAINED_PROTOCOL_REVIEW")

        if unresolved:
            severity = "REVIEW"
            requires_review = "YES"
            status = "REQUIRES_HUMAN_REVIEW"
            validated_match = "REVIEW"
        elif resolved:
            severity = "INFO"
            requires_review = "NO"
            status = "RESOLVED_KNOWN_PROTOCOL_INFORMATION"
            validated_match = "YES"
        else:
            severity = "NONE"
            requires_review = "NO"
            status = "AUTO_MATCH"
            validated_match = "YES"

        result.at[index, "Validated_Expected_Protocol_Match"] = validated_match
        result.at[index, "Validated_Issue_Codes"] = _join_codes(resolved + unresolved)
        result.at[index, "Validation_Resolution_Codes"] = _join_codes(resolved)
        result.at[index, "Validation_Severity"] = severity
        result.at[index, "Validation_Requires_Human_Review"] = requires_review
        result.at[index, "Validation_Status"] = status
        result.at[index, "Validation_Notes"] = " ".join(notes)
    return result


def _anomaly_table(validated: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Anomaly_Scope",
        "Batch_ID",
        "Cell_ID",
        "Source_File",
        "Raw_Cycle_Index",
        "Protocol_Class",
        "Primary_Capacity_Eligible",
        "Original_Anomaly_Codes",
        "Resolved_Anomaly_Codes",
        "Remaining_Review_Codes",
        "Severity",
        "Requires_Human_Review",
        "Original_Expected_Protocol_Match",
        "Validated_Expected_Protocol_Match",
        "Observed_Charge_Current_A",
        "Observed_Charge_Max_V",
        "Observed_Charge_End_Current_A",
        "Observed_Discharge_Median_Current_A",
        "Observed_Discharge_Min_V",
        "Resolution_Basis",
    ]
    records: list[dict[str, Any]] = []
    for row in validated.loc[validated["Validation_Severity"].ne("NONE")].itertuples(
        index=False
    ):
        values = row._asdict()
        original = _split_codes(values.get("Issues", ""))
        resolved = _split_codes(values.get("Validation_Resolution_Codes", ""))
        validated_codes = _split_codes(values.get("Validated_Issue_Codes", ""))
        remaining = [code for code in validated_codes if code not in resolved]
        records.append(
            {
                "Anomaly_Scope": "CYCLE_PROTOCOL",
                "Batch_ID": values["Batch_ID"],
                "Cell_ID": values["Cell_ID"],
                "Source_File": values["Source_File"],
                "Raw_Cycle_Index": values["Raw_Cycle_Index"],
                "Protocol_Class": values["Validated_Protocol_Class"],
                "Primary_Capacity_Eligible": values[
                    "Validated_Primary_Capacity_Eligible"
                ],
                "Original_Anomaly_Codes": _join_codes(original),
                "Resolved_Anomaly_Codes": _join_codes(resolved),
                "Remaining_Review_Codes": _join_codes(remaining),
                "Severity": values["Validation_Severity"],
                "Requires_Human_Review": values[
                    "Validation_Requires_Human_Review"
                ],
                "Original_Expected_Protocol_Match": values[
                    "Expected_Protocol_Match"
                ],
                "Validated_Expected_Protocol_Match": values[
                    "Validated_Expected_Protocol_Match"
                ],
                "Observed_Charge_Current_A": values[
                    "Observed_Charge_Current_A"
                ],
                "Observed_Charge_Max_V": values["Observed_Charge_Max_V"],
                "Observed_Charge_End_Current_A": values[
                    "Observed_Charge_End_Current_A"
                ],
                "Observed_Discharge_Median_Current_A": values[
                    "Observed_Discharge_Median_Current_A"
                ],
                "Observed_Discharge_Min_V": values["Observed_Discharge_Min_V"],
                "Resolution_Basis": values["Validation_Notes"],
            }
        )
    return pd.DataFrame(records, columns=columns).sort_values(
        ["Batch_ID", "Cell_ID", "Raw_Cycle_Index"], kind="stable"
    ).reset_index(drop=True)


def _regression_tests(validated: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    def add(
        test_id: str,
        batch_id: object,
        cell_id: str,
        cycles: str,
        expected: object,
        observed: object,
        passed: bool,
        tolerance: str = "EXACT",
        failure_message: str = "",
    ) -> None:
        records.append(
            {
                "Test_ID": test_id,
                "Batch_ID": batch_id,
                "Cell_ID": cell_id,
                "Raw_Cycle_or_Range": cycles,
                "Expected": expected,
                "Observed": observed,
                "Tolerance": tolerance,
                "Status": "PASS" if bool(passed) else "FAIL",
                "Failure_Message": "" if passed else failure_message,
            }
        )

    def one(cell_id: str, cycle: int) -> pd.Series:
        rows = validated.loc[
            validated["Cell_ID"].eq(cell_id)
            & pd.to_numeric(validated["Raw_Cycle_Index"], errors="coerce").eq(cycle)
        ]
        if len(rows) != 1:
            raise RuntimeError(f"REGRESSION_KEY_NOT_UNIQUE:{cell_id}:{cycle}:{len(rows)}")
        return rows.iloc[0]

    def exact(
        test_id: str,
        batch: int,
        cell: str,
        cycle: int,
        field: str,
        expected: object,
    ) -> None:
        observed = one(cell, cycle)[field]
        add(
            test_id,
            batch,
            cell,
            str(cycle),
            expected,
            observed,
            str(observed) == str(expected),
            failure_message=f"{field} did not match the approved protocol decision",
        )

    def approx(
        test_id: str,
        batch: int,
        cell: str,
        cycle: int,
        field: str,
        expected: float,
        tolerance: float,
    ) -> None:
        observed = _number(one(cell, cycle)[field])
        passed = bool(np.isfinite(observed) and abs(observed - expected) <= tolerance)
        add(
            test_id,
            batch,
            cell,
            str(cycle),
            expected,
            observed,
            passed,
            f"±{tolerance:g}",
            f"{field} outside tolerance",
        )

    b2 = "B2_3C_battery-3"
    exact("B2_C1_INITIAL_CLASS", 2, b2, 1, "Validated_Protocol_Class", "INITIAL_CAPACITY_TEST")
    exact("B2_C2_STANDARD_CLASS", 2, b2, 2, "Validated_Protocol_Class", "STANDARD_FULL_DISCHARGE_1C")
    exact("B2_C2_PRIMARY", 2, b2, 2, "Validated_Primary_Capacity_Eligible", "YES")
    approx("B2_C2_CAPACITY", 2, b2, 2, "Summary_Discharge_Capacity_Ah", 1.878, 0.002)
    approx("B2_C2_CHARGE_CURRENT", 2, b2, 2, "Observed_Charge_Current_A", 6.0, 0.05)
    approx("B2_C2_DISCHARGE_CURRENT", 2, b2, 2, "Observed_Discharge_Median_Current_A", -2.0, 0.05)
    approx("B2_C2_CUTOFF", 2, b2, 2, "Observed_Discharge_Min_V", 2.5, 0.01)
    exact("B2_C286_STANDARD_CLASS", 2, b2, 286, "Validated_Protocol_Class", "STANDARD_FULL_DISCHARGE_1C")

    b3 = "B3_R2.5_battery-4"
    b3_cases = [
        (2, "VARIABLE_FULL_DISCHARGE_0P5C", -1.0, "NO"),
        (3, "VARIABLE_FULL_DISCHARGE_1C", -2.0, "YES"),
        (4, "VARIABLE_FULL_DISCHARGE_2C", -4.0, "NO"),
        (5, "VARIABLE_FULL_DISCHARGE_3C", -6.0, "NO"),
        (6, "VARIABLE_FULL_DISCHARGE_5C", -10.0, "NO"),
    ]
    for cycle, protocol_class, current, eligible in b3_cases:
        exact(f"B3_C{cycle}_CLASS", 3, b3, cycle, "Validated_Protocol_Class", protocol_class)
        exact(f"B3_C{cycle}_PRIMARY", 3, b3, cycle, "Validated_Primary_Capacity_Eligible", eligible)
        approx(f"B3_C{cycle}_CURRENT", 3, b3, cycle, "Observed_Discharge_Median_Current_A", current, 0.05)
        approx(f"B3_C{cycle}_CUTOFF", 3, b3, cycle, "Observed_Discharge_Min_V", 2.5, 0.01)

    b4 = "B4_R3_battery-4"
    partial = validated.loc[
        validated["Cell_ID"].eq(b4)
        & pd.to_numeric(validated["Raw_Cycle_Index"], errors="coerce").between(2, 6)
    ]
    add(
        "B4_C2_6_PARTIAL_NONPRIMARY",
        4,
        b4,
        "2-6",
        "5 AGING_PARTIAL_DISCHARGE rows; all primary=NO",
        f"rows={len(partial)};classes={sorted(partial['Validated_Protocol_Class'].unique())};primary={sorted(partial['Validated_Primary_Capacity_Eligible'].unique())}",
        len(partial) == 5
        and partial["Validated_Protocol_Class"].eq("AGING_PARTIAL_DISCHARGE").all()
        and partial["Validated_Primary_Capacity_Eligible"].eq("NO").all(),
        failure_message="Batch-4 partial aging classification changed",
    )
    add(
        "B4_C2_6_CUTOFF_3V",
        4,
        b4,
        "2-6",
        "all discharge minima ≈3.0 V",
        "|".join(f"{value:.4f}" for value in pd.to_numeric(partial["Observed_Discharge_Min_V"])),
        len(partial) == 5
        and np.allclose(pd.to_numeric(partial["Observed_Discharge_Min_V"]), 3.0, atol=0.01),
        "±0.01 V",
        "Batch-4 partial-discharge cutoff mismatch",
    )
    exact("B4_C7_DIAGNOSTIC_CLASS", 4, b4, 7, "Validated_Protocol_Class", "CAPACITY_DIAGNOSTIC_1C")
    exact("B4_C7_PRIMARY", 4, b4, 7, "Validated_Primary_Capacity_Eligible", "YES")
    approx("B4_C7_CAPACITY", 4, b4, 7, "Summary_Discharge_Capacity_Ah", 1.909, 0.002)

    b5_4 = "B5_RW_battery-4"
    for cycle in (1, 20, 22, 31):
        exact(f"B5_B4_C{cycle}_RANDOM", 5, b5_4, cycle, "Validated_Protocol_Class", "RANDOM_WALK_AGING")
        exact(f"B5_B4_C{cycle}_NONPRIMARY", 5, b5_4, cycle, "Validated_Primary_Capacity_Eligible", "NO")
    for cycle in (21, 32):
        exact(f"B5_B4_C{cycle}_DIAGNOSTIC", 5, b5_4, cycle, "Validated_Protocol_Class", "CAPACITY_DIAGNOSTIC_1C")
        exact(f"B5_B4_C{cycle}_PRIMARY", 5, b5_4, cycle, "Validated_Primary_Capacity_Eligible", "YES")

    b5_1 = "B5_RW_battery-1"
    b5_info = validated.loc[
        validated["Batch_ID"].eq(5)
        & validated["Validation_Resolution_Codes"].str.contains(
            "DESCRIPTION_CHARGE_CURRENT_MISMATCH", regex=False
        )
    ]
    add(
        "B5_METADATA_MISMATCH_INFO_COUNT",
        5,
        b5_1,
        "1-10",
        10,
        len(b5_info),
        len(b5_info) == 10
        and b5_info["Validation_Severity"].eq("INFO").all()
        and b5_info["Validation_Requires_Human_Review"].eq("NO").all()
        and b5_info["Validated_Primary_Capacity_Eligible"].eq("NO").all(),
        failure_message="Known Batch-5 metadata mismatch was not retained as INFO-only",
    )
    b5_cycles = validated.loc[validated["Cell_ID"].eq(b5_1)].copy()
    initial_count = int(b5_cycles["Validated_Protocol_Class"].eq("INITIAL_CAPACITY_TEST").sum())
    add(
        "B5_NO_BOL_INITIAL_TEST",
        5,
        b5_1,
        "all",
        "Qref_initial_test unavailable; no fallback",
        f"initial_class_rows={initial_count}",
        initial_count == 0,
        failure_message="Batch-5 must not substitute random-walk Cycle 1 for a BOL Qref",
    )
    b5_diag = b5_cycles.loc[
        b5_cycles["Validated_Protocol_Class"].eq("CAPACITY_DIAGNOSTIC_1C")
        & b5_cycles["Validated_Primary_Capacity_Eligible"].eq("YES")
        & b5_cycles["Validated_Expected_Protocol_Match"].eq("YES")
    ].sort_values("Raw_Cycle_Index")
    first5 = b5_diag.head(5)
    source_cycles = "|".join(str(int(value)) for value in first5["Raw_Cycle_Index"])
    median5 = float(pd.to_numeric(first5["Summary_Discharge_Capacity_Ah"]).median())
    add("B5_FIRST_COMPARABLE_SOURCE", 5, b5_1, "diagnostics", 21, int(b5_diag.iloc[0]["Raw_Cycle_Index"]), int(b5_diag.iloc[0]["Raw_Cycle_Index"]) == 21, failure_message="Wrong first comparable diagnostic")
    add("B5_MEDIAN5_SOURCE_CYCLES", 5, b5_1, "diagnostics", "21|32|43|54|65", source_cycles, source_cycles == "21|32|43|54|65", failure_message="Wrong early comparable source cycles")
    add("B5_MEDIAN5_CAPACITY", 5, b5_1, "21|32|43|54|65", 1.923, median5, abs(median5 - 1.923) <= 0.002, "±0.002 Ah", "Wrong early median-5 comparable capacity")

    b6_1 = "B6_Sim_satellite_battery-1"
    exact("B6_B1_C7_DIAGNOSTIC", 6, b6_1, 7, "Validated_Protocol_Class", "CAPACITY_DIAGNOSTIC_0P5C")
    exact("B6_B1_C7_PRIMARY", 6, b6_1, 7, "Validated_Primary_Capacity_Eligible", "YES")
    approx("B6_B1_C7_CAPACITY", 6, b6_1, 7, "Summary_Discharge_Capacity_Ah", 1.922, 0.002)
    exact("B6_B1_C419_GEO", 6, b6_1, 419, "Validated_Protocol_Class", "GEO_AGING_PARTIAL")
    exact("B6_B1_C419_NONPRIMARY", 6, b6_1, 419, "Validated_Primary_Capacity_Eligible", "NO")
    exact("B6_B1_C419_EXPECTED_MATCH", 6, b6_1, 419, "Validated_Expected_Protocol_Match", "YES")
    exact("B6_B1_C419_INFO_CODE", 6, b6_1, 419, "Validation_Resolution_Codes", "CC_PLATEAU_NOT_OBSERVED_HIGH_SOC_START")

    cc_info = validated.loc[
        validated["Validation_Resolution_Codes"].str.contains(
            "CC_PLATEAU_NOT_OBSERVED_HIGH_SOC_START", regex=False
        )
    ]
    add("B6_HIGH_SOC_CC_INFO_COUNT", 6, "ALL_BATCH6", "all", 63, len(cc_info), len(cc_info) == 63 and cc_info["Validation_Severity"].eq("INFO").all() and cc_info["Validation_Requires_Human_Review"].eq("NO").all(), failure_message="Batch-6 high-SOC CC plateau rule did not resolve exactly 63 known rows")
    cutoff_info = validated.loc[
        validated["Validation_Resolution_Codes"].str.contains(
            "GEO_AGING_REACHED_FULL_CUTOFF", regex=False
        )
    ]
    add("B6_GEO_CUTOFF_INFO_COUNT", 6, "ALL_BATCH6", "all", 21, len(cutoff_info), len(cutoff_info) == 21 and cutoff_info["Validated_Protocol_Class"].eq("GEO_AGING_PARTIAL").all() and cutoff_info["Validated_Primary_Capacity_Eligible"].eq("NO").all(), failure_message="GEO full-cutoff information was reclassified or left for review")

    b6_2 = "B6_Sim_satellite_battery-2"
    approx("B6_B2_C1220_CAPACITY", 6, b6_2, 1220, "Summary_Discharge_Capacity_Ah", 1.605, 0.002)
    exact("B6_B2_C1220_PRIMARY", 6, b6_2, 1220, "Validated_Primary_Capacity_Eligible", "YES")
    middle = validated.loc[
        validated["Cell_ID"].eq(b6_2)
        & pd.to_numeric(validated["Raw_Cycle_Index"], errors="coerce").between(1221, 1225)
    ]
    add("B6_B2_C1221_1225_GEO_NONPRIMARY", 6, b6_2, "1221-1225", "5 GEO rows; all primary=NO", f"rows={len(middle)};classes={sorted(middle['Validated_Protocol_Class'].unique())};primary={sorted(middle['Validated_Primary_Capacity_Eligible'].unique())}", len(middle) == 5 and middle["Validated_Protocol_Class"].eq("GEO_AGING_PARTIAL").all() and middle["Validated_Primary_Capacity_Eligible"].eq("NO").all(), failure_message="A GEO row could incorrectly trigger primary EOL")
    approx("B6_B2_C1226_CAPACITY", 6, b6_2, 1226, "Summary_Discharge_Capacity_Ah", 1.596, 0.002)
    exact("B6_B2_C1226_PRIMARY", 6, b6_2, 1226, "Validated_Primary_Capacity_Eligible", "YES")

    primary = validated.loc[
        validated["Cell_ID"].eq(b6_2)
        & validated["Validated_Primary_Capacity_Eligible"].eq("YES")
        & validated["Validated_Expected_Protocol_Match"].eq("YES")
        & validated["Validated_Protocol_Class"].eq("CAPACITY_DIAGNOSTIC_0P5C")
    ].copy()
    observations = primary.rename(
        columns={"Summary_Discharge_Capacity_Ah": "Capacity_Ah"}
    )
    eol = detect_eol(observations, 1.600, dense_measurement=False)
    add("B6_B2_NOMINAL_EOL_STATUS", 6, b6_2, "primary diagnostics", "INTERVAL_CENSORED", eol.status, eol.status == "INTERVAL_CENSORED", failure_message="Nominal EOL must be interval-censored")
    add("B6_B2_NOMINAL_EOL_AGING_BRACKET", 6, b6_2, "1220< EOL <=1226", "1008 < EOL <= 1013", f"{eol.lower_bound_aging_cycle:g} < EOL <= {eol.upper_bound_aging_cycle:g}" if eol.lower_bound_aging_cycle is not None and eol.upper_bound_aging_cycle is not None else "NA", eol.lower_bound_aging_cycle == 1008.0 and eol.upper_bound_aging_cycle == 1013.0, failure_message="Wrong interval-censored aging bracket")
    add("B6_B2_NOMINAL_EOL_FIRST_BELOW_RAW", 6, b6_2, "primary diagnostics", 1226, eol.first_below_raw_cycle, eol.first_below_raw_cycle == 1226.0, failure_message="A nonprimary GEO cycle was used as EOL")

    return pd.DataFrame(records)


def _validation_report(
    validated: pd.DataFrame,
    anomalies: pd.DataFrame,
    tests: pd.DataFrame,
    review: Mapping[str, Any],
) -> pd.DataFrame:
    representatives = {
        2: "B2_3C_battery-3",
        3: "B3_R2.5_battery-4",
        4: "B4_R3_battery-4",
        5: "B5_RW_battery-1|B5_RW_battery-4",
        6: "B6_Sim_satellite_battery-1|B6_Sim_satellite_battery-2",
    }
    rows: list[dict[str, Any]] = []
    decisions = review.get("adapter_decisions", {})
    for batch_id in range(2, 7):
        group = validated.loc[validated["Batch_ID"].eq(batch_id)]
        group_anomalies = anomalies.loc[anomalies["Batch_ID"].eq(batch_id)]
        group_tests = tests.loc[pd.to_numeric(tests["Batch_ID"], errors="coerce").eq(batch_id)]
        rows.append(
            {
                "Batch_ID": batch_id,
                "Adapter_Manual_Decision": decisions.get(str(batch_id), "MISSING"),
                "Validation_Status": "PASS" if not group_tests["Status"].eq("FAIL").any() and not group_anomalies["Requires_Human_Review"].eq("YES").any() else "FAIL",
                "Canonical_File_Count": int(group["Cell_ID"].nunique()),
                "Cycle_Count": len(group),
                "Unknown_Protocol_Count": int(group["Validated_Protocol_Class"].eq("UNKNOWN_PROTOCOL").sum()),
                "Primary_Capacity_Candidate_Count": int(group["Validated_Primary_Capacity_Eligible"].eq("YES").sum()),
                "Info_Anomaly_Count": int(group_anomalies["Severity"].eq("INFO").sum()),
                "Review_Anomaly_Count": int(group_anomalies["Severity"].eq("REVIEW").sum()),
                "Regression_Test_Count": len(group_tests),
                "Regression_Failure_Count": int(group_tests["Status"].eq("FAIL").sum()),
                "Representative_Cell_ID": representatives[batch_id],
                "Full_55_Cell_Run_Executed": "NO",
                "Run_Authorization_Status": "WAITING_FOR_HUMAN_APPROVAL",
            }
        )
    return pd.DataFrame(rows)


def _candidate_config(
    draft: Mapping[str, Any],
    review_file: Path,
    scan_hashes: Mapping[str, str],
) -> dict[str, Any]:
    config = deepcopy(dict(draft))
    batch_id = int(config["batch_id"])
    config["config_status"] = "VALIDATED_FINAL_CANDIDATE_AWAITING_RUN_APPROVAL"
    config["manual_protocol_review"] = {
        "status": "REPRESENTATIVE_PROTOCOL_ADAPTER_APPROVED",
        "review_file": review_file.name,
        "review_file_sha256": _sha256(review_file),
        "full_55_cell_run_authorized": False,
    }
    config["known_information_anomaly_policy"] = {
        "unknown_or_unexplained_anomalies_require_human_review": True,
        "DESCRIPTION_CHARGE_CURRENT_MISMATCH": {
            "applicable_when": "Batch-5 RANDOM_WALK_AGING otherwise valid",
            "severity": "INFO",
            "requires_human_review": False,
        },
        "CC_PLATEAU_NOT_OBSERVED_HIGH_SOC_START": {
            "applicable_when": "Batch-6 GEO charge reaches >=4.18 V and CV terminal current <=0.13 A with no other charge truncation evidence",
            "severity": "INFO",
            "requires_human_review": False,
        },
        "GEO_AGING_REACHED_FULL_CUTOFF": {
            "applicable_when": "Batch-6 GEO_AGING_PARTIAL otherwise valid",
            "severity": "INFO",
            "requires_human_review": False,
            "reclassify_as_capacity_diagnostic": False,
        },
    }
    config["approval_gate"] = {
        "validation_status_required": VALIDATION_STATUS_PASSED,
        "separate_run_approval_file": "run_approval.json",
        "separate_run_approval_status_required": "APPROVED",
        "run_is_not_authorized_by_this_candidate_config": True,
    }
    config["locked_scan_hashes"] = dict(scan_hashes)
    if batch_id == 5:
        config["qref_initial_test_policy"] = {
            "value": None,
            "status": "UNAVAILABLE_NO_BOL_INITIAL_TEST",
            "fallback_to_random_walk_cycle_1": False,
        }
        config["first_comparable_policy"] = "FIRST_VALID_1C_FULL_CAPACITY_DIAGNOSTIC"
        config["early_median5_comparable_policy"] = "MEDIAN_OF_FIRST_5_VALID_1C_FULL_CAPACITY_DIAGNOSTICS"
    if batch_id == 6:
        config["geo_high_soc_cv_terminal_current_upper_A"] = 0.13
        config["geo_eol_policy"] = "GEO_AGING_PARTIAL_NEVER_DIRECTLY_TRIGGERS_PRIMARY_EOL"
        config["sparse_diagnostic_eol_policy"] = "INTERVAL_CENSORED_NO_INTERPOLATION"
    return config


def validate_scan(
    scan_dir: Path,
    output_dir: Path,
    frozen_batch1_dir: Path,
    review_file: Path,
) -> ValidationResult:
    """Validate approved representative protocol decisions; never run analysis."""
    scan_dir = scan_dir.resolve()
    output_dir = output_dir.resolve()
    frozen_batch1_dir = frozen_batch1_dir.resolve()
    review_file = review_file.resolve()
    if not review_file.is_file():
        raise RuntimeError(f"MANUAL_PROTOCOL_REVIEW_FILE_MISSING:{review_file}")
    review = _load_json(review_file)
    if review.get("review_status") != "REPRESENTATIVE_PROTOCOL_ADAPTERS_VALIDATED":
        raise RuntimeError("MANUAL_PROTOCOL_REVIEW_STATUS_INVALID")
    if {int(value) for value in review.get("reviewed_batches", [])} != set(range(2, 7)):
        raise RuntimeError("MANUAL_PROTOCOL_REVIEW_BATCH_SET_MUST_EQUAL_2_THROUGH_6")
    if review.get("full_55_cell_run_authorized") is not False:
        raise RuntimeError("VALIDATE_STAGE_MUST_NOT_AUTHORIZE_FULL_RUN")

    locked_scan_hashes = output_hash_manifest(scan_dir)
    manifest = pd.read_csv(
        scan_dir / "xjtu_file_manifest.csv", keep_default_na=False, low_memory=False
    )
    canonical = manifest.loc[
        manifest["File_Role"].eq("CANONICAL")
        & manifest["Included_In_Analysis"].eq("YES")
    ]
    counts = canonical.groupby("Batch_ID").size().to_dict()
    for batch_id, expected in EXPECTED_CANONICAL_COUNTS.items():
        if int(counts.get(batch_id, 0)) != expected:
            raise RuntimeError(
                f"CANONICAL_COUNT_MISMATCH:Batch-{batch_id}:"
                f"found={counts.get(batch_id, 0)};expected={expected}"
            )
    schema = pd.read_csv(
        scan_dir / "xjtu_schema_inventory.csv", keep_default_na=False, low_memory=False
    )
    if schema["Schema_Status"].eq("ERROR").any():
        raise RuntimeError("SCHEMA_ERROR_PRESENT; validation failed")
    protocol = pd.read_csv(
        scan_dir / "xjtu_protocol_inventory.csv",
        keep_default_na=False,
        low_memory=False,
    )
    if protocol.duplicated(["Cell_ID", "Raw_Cycle_Index"]).any():
        raise RuntimeError("DUPLICATE_PROTOCOL_CYCLE_KEY")
    expected_rows = int(
        pd.to_numeric(
            schema.loc[schema["Batch_ID"].between(2, 6), "Data_Cycle_Count"],
            errors="raise",
        ).sum()
    )
    if len(protocol) != expected_rows:
        raise RuntimeError(
            f"PROTOCOL_ROW_COUNT_MISMATCH:found={len(protocol)};expected={expected_rows}"
        )

    configs = {
        batch_id: _load_json(scan_dir / f"batch{batch_id}_config_draft.json")
        for batch_id in range(2, 7)
    }
    validated = resolve_known_protocol_information(protocol, configs)
    anomalies = _anomaly_table(validated)
    tests = _regression_tests(validated)

    integrity_records = [
        {
            "Test_ID": "GLOBAL_PROTOCOL_KEYS_UNIQUE",
            "Batch_ID": "ALL",
            "Cell_ID": "ALL_NEW_BATCHES",
            "Raw_Cycle_or_Range": "all",
            "Expected": "unique Cell_ID + Raw_Cycle_Index",
            "Observed": f"rows={len(validated)};unique={len(validated.drop_duplicates(['Cell_ID', 'Raw_Cycle_Index']))}",
            "Tolerance": "EXACT",
            "Status": "PASS",
            "Failure_Message": "",
        },
        {
            "Test_ID": "GLOBAL_UNKNOWN_PROTOCOL_COUNT",
            "Batch_ID": "ALL",
            "Cell_ID": "ALL_NEW_BATCHES",
            "Raw_Cycle_or_Range": "all",
            "Expected": 0,
            "Observed": int(validated["Validated_Protocol_Class"].eq("UNKNOWN_PROTOCOL").sum()),
            "Tolerance": "EXACT",
            "Status": "PASS" if not validated["Validated_Protocol_Class"].eq("UNKNOWN_PROTOCOL").any() else "FAIL",
            "Failure_Message": "" if not validated["Validated_Protocol_Class"].eq("UNKNOWN_PROTOCOL").any() else "Unknown protocols remain",
        },
        {
            "Test_ID": "GLOBAL_UNRESOLVED_REVIEW_COUNT",
            "Batch_ID": "ALL",
            "Cell_ID": "ALL_NEW_BATCHES",
            "Raw_Cycle_or_Range": "all",
            "Expected": 0,
            "Observed": int(validated["Validation_Requires_Human_Review"].eq("YES").sum()),
            "Tolerance": "EXACT",
            "Status": "PASS" if not validated["Validation_Requires_Human_Review"].eq("YES").any() else "FAIL",
            "Failure_Message": "" if not validated["Validation_Requires_Human_Review"].eq("YES").any() else "Unknown or unexplained protocol anomalies remain",
        },
    ]
    tests = pd.concat([pd.DataFrame(integrity_records), tests], ignore_index=True)
    failed_count = int(tests["Status"].eq("FAIL").sum())
    review_count = int(anomalies["Requires_Human_Review"].eq("YES").sum())
    passed = failed_count == 0 and review_count == 0
    report = _validation_report(validated, anomalies, tests, review)

    temp_dir = _atomic_directory(output_dir)
    _write_csv(report, temp_dir / "protocol_validation_report.csv")
    _write_csv(tests, temp_dir / "regression_test_results.csv")
    _write_csv(anomalies, temp_dir / "protocol_anomalies_v2.csv")
    _write_csv(validated, temp_dir / "xjtu_protocol_inventory_validated.csv")
    for batch_id in range(2, 7):
        candidate = _candidate_config(
            configs[batch_id], review_file, locked_scan_hashes
        )
        (temp_dir / f"batch{batch_id}_config_final_candidate.json").write_text(
            json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    status = VALIDATION_STATUS_PASSED if passed else VALIDATION_STATUS_FAILED
    status_message = (
        "VALIDATION PASSED — WAITING FOR HUMAN APPROVAL TO RUN"
        if passed
        else "VALIDATION FAILED — DO NOT RUN"
    )
    log_lines = [
        f"Pipeline_Version={PIPELINE_VERSION}",
        f"Python_Version={sys.version.replace(chr(10), ' ')}",
        f"Platform={platform.platform()}",
        f"NumPy_Version={np.__version__}",
        f"pandas_Version={pd.__version__}",
        f"SciPy_Version={scipy.__version__}",
        f"Validation_Timestamp_UTC={datetime.now(timezone.utc).isoformat()}",
        f"Scan_Directory={scan_dir}",
        f"Frozen_Batch1_Directory={frozen_batch1_dir}",
        f"Manual_Review_File={review_file}",
        f"Manual_Review_SHA256={_sha256(review_file)}",
        f"Validated_Protocol_Rows={len(validated)}",
        f"Info_Anomaly_Count={int(anomalies['Severity'].eq('INFO').sum())}",
        f"Review_Anomaly_Count={review_count}",
        f"Regression_Test_Count={len(tests)}",
        f"Regression_Test_Failures={failed_count}",
        f"Validation_Status={status}",
        "FULL_55_CELL_RUN_EXECUTED=NO",
        "RUN_APPROVAL_FILE_CREATED=NO",
        status_message,
    ]
    (temp_dir / "validation_log.txt").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )

    output_hashes = {
        path.name: _sha256(path)
        for path in sorted(temp_dir.iterdir())
        if path.is_file()
    }
    receipt = {
        "validation_status": status,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scan_directory": str(scan_dir),
        "frozen_batch1_directory": str(frozen_batch1_dir),
        "manual_review_file": str(review_file),
        "manual_review_file_sha256": _sha256(review_file),
        "locked_scan_hashes": locked_scan_hashes,
        "locked_scan_artifact_hashes": {
            path.name: _sha256(path)
            for path in sorted(scan_dir.iterdir())
            if path.is_file()
        },
        "frozen_batch1_hashes": {
            path.name: _sha256(path)
            for path in sorted(frozen_batch1_dir.glob("*.csv"))
        },
        "validated_protocol_inventory_sha256": _sha256(
            temp_dir / "xjtu_protocol_inventory_validated.csv"
        ),
        "candidate_config_hashes": {
            f"batch{batch_id}_config_final_candidate.json": _sha256(
                temp_dir / f"batch{batch_id}_config_final_candidate.json"
            )
            for batch_id in range(2, 7)
        },
        "validation_output_hashes_excluding_receipt": output_hashes,
        "regression_test_count": len(tests),
        "failed_regression_test_count": failed_count,
        "info_anomaly_count": int(anomalies["Severity"].eq("INFO").sum()),
        "review_anomaly_count": review_count,
        "full_55_cell_run_executed": False,
        "full_55_cell_run_authorized": False,
        "next_required_artifact": "run_approval.json",
    }
    (temp_dir / "validation_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_dir.rename(output_dir)
    return ValidationResult(
        output_dir=output_dir,
        passed=passed,
        regression_test_count=len(tests),
        failed_test_count=failed_count,
        info_anomaly_count=int(anomalies["Severity"].eq("INFO").sum()),
        review_anomaly_count=review_count,
    )
