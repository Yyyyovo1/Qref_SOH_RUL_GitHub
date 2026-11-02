from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from . import PIPELINE_VERSION
from .config_templates import EXPECTED_CANONICAL_COUNTS, NOMINAL_CAPACITY_AH
from .scan import output_hash_manifest


QREF_DEFINITIONS = (
    "nominal",
    "initial_test",
    "first_comparable",
    "early_median5_comparable",
)

NUMERIC_ORDER_COLUMNS = (
    "Diagnostic_Index",
    "Raw_Cycle_Index",
    "Cumulative_Aging_Operation_Count",
)


@dataclass(frozen=True)
class EOLResult:
    status: str
    lower_bound_aging_cycle: float | None
    upper_bound_aging_cycle: float | None
    exact_aging_cycle: float | None
    first_below_diagnostic_index: float | None
    first_below_raw_cycle: float | None
    last_valid_aging_cycle: float | None
    last_valid_raw_cycle: float | None
    last_valid_capacity_Ah: float | None
    censoring_reason: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_directory(target: Path) -> Path:
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / f".{target.name}.tmp-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    temp.mkdir(parents=False, exist_ok=False)
    return temp


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def _coerce_numeric_order_columns(
    frame: pd.DataFrame,
    columns: Iterable[str] = NUMERIC_ORDER_COLUMNS,
) -> pd.DataFrame:
    """Return a copy with ordering keys converted strictly to numeric values.

    Research selections such as first-comparable and early-median-5 must never
    inherit lexical ordering from CSV object columns (for example 1, 10, 100,
    2).  Missing required columns and non-numeric values are hard errors rather
    than silently coerced exclusions.
    """
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            raise ValueError(f"Missing numeric ordering column: {column}")
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_approval(scan_dir: Path, approval: Mapping[str, Any]) -> dict[str, str]:
    if approval.get("status") != "APPROVED":
        raise RuntimeError("PROTOCOL_REVIEW_NOT_APPROVED")
    if not approval.get("approved_by"):
        raise RuntimeError("PROTOCOL_REVIEW_APPROVER_MISSING")
    expected_hashes = output_hash_manifest(scan_dir)
    supplied_hashes = approval.get("file_hashes")
    if not isinstance(supplied_hashes, dict):
        raise RuntimeError("PROTOCOL_REVIEW_HASHES_MISSING")
    differences = [
        name
        for name, expected in expected_hashes.items()
        if supplied_hashes.get(name) != expected
    ]
    if differences:
        raise RuntimeError(f"PROTOCOL_REVIEW_HASH_MISMATCH:{'|'.join(differences)}")
    if approval.get("acknowledge_all_scan_anomalies") is not True:
        raise RuntimeError("SCAN_ANOMALIES_NOT_EXPLICITLY_ACKNOWLEDGED")
    approved_batches = {int(value) for value in approval.get("approved_batches", [])}
    if approved_batches != set(range(2, 7)):
        raise RuntimeError("APPROVED_BATCH_SET_MUST_EQUAL_2_THROUGH_6")
    return expected_hashes


def _apply_protocol_overrides(
    inventory: pd.DataFrame,
    approval: Mapping[str, Any],
) -> pd.DataFrame:
    result = inventory.copy()
    result["Validated_Protocol_Class"] = result["Protocol_Class"]
    result["Validated_Primary_Capacity_Eligible"] = result[
        "Primary_Capacity_Eligible"
    ]
    result["Protocol_Review_Status"] = np.where(
        result["Expected_Protocol_Match"].eq("YES"), "AUTO_MATCH_APPROVED", "UNRESOLVED"
    )
    result["Protocol_Review_Notes"] = ""

    seen_keys: set[tuple[str, int]] = set()
    for override in approval.get("protocol_overrides", []):
        if not isinstance(override, dict):
            raise RuntimeError("INVALID_PROTOCOL_OVERRIDE_RECORD")
        key = (str(override.get("Cell_ID", "")), int(override.get("Raw_Cycle_Index", -1)))
        if key in seen_keys:
            raise RuntimeError(f"DUPLICATE_PROTOCOL_OVERRIDE:{key[0]}:{key[1]}")
        seen_keys.add(key)
        mask = result["Cell_ID"].eq(key[0]) & result["Raw_Cycle_Index"].eq(key[1])
        if int(mask.sum()) != 1:
            raise RuntimeError(f"PROTOCOL_OVERRIDE_KEY_NOT_UNIQUE:{key[0]}:{key[1]}")
        protocol_class = str(override.get("Protocol_Class", ""))
        eligibility = str(override.get("Primary_Capacity_Eligible", ""))
        if not protocol_class or eligibility not in {"YES", "NO"}:
            raise RuntimeError(f"INVALID_PROTOCOL_OVERRIDE_VALUES:{key[0]}:{key[1]}")
        result.loc[mask, "Validated_Protocol_Class"] = protocol_class
        result.loc[mask, "Validated_Primary_Capacity_Eligible"] = eligibility
        result.loc[mask, "Protocol_Review_Status"] = "MANUAL_OVERRIDE_APPROVED"
        result.loc[mask, "Protocol_Review_Notes"] = str(override.get("Reason", ""))

    unresolved_unknown = result["Validated_Protocol_Class"].eq("UNKNOWN_PROTOCOL")
    if unresolved_unknown.any():
        keys = result.loc[unresolved_unknown, ["Cell_ID", "Raw_Cycle_Index"]].head(20)
        raise RuntimeError(
            "UNRESOLVED_UNKNOWN_PROTOCOL:" + "|".join(
                f"{row.Cell_ID}:{int(row.Raw_Cycle_Index)}" for row in keys.itertuples(index=False)
            )
        )

    unresolved = result["Protocol_Review_Status"].eq("UNRESOLVED")
    if unresolved.any():
        nonprimary = unresolved & result["Primary_Capacity_Eligible"].eq("NO")
        primary = unresolved & ~result["Primary_Capacity_Eligible"].eq("NO")
        if nonprimary.any() and approval.get("accept_nonprimary_review_rows") is True:
            result.loc[nonprimary, "Protocol_Review_Status"] = (
                "MANUAL_ACCEPTED_NONPRIMARY_REVIEW"
            )
            result.loc[nonprimary, "Validated_Primary_Capacity_Eligible"] = "NO"
        if primary.any() and approval.get("exclude_unresolved_primary_review_rows") is True:
            result.loc[primary, "Protocol_Review_Status"] = (
                "MANUAL_EXCLUDED_UNRESOLVED_PRIMARY_REVIEW"
            )
            result.loc[primary, "Validated_Primary_Capacity_Eligible"] = "NO"
        unresolved = result["Protocol_Review_Status"].eq("UNRESOLVED")
        if unresolved.any():
            raise RuntimeError(
                f"UNRESOLVED_PROTOCOL_REVIEW_ROWS:{int(unresolved.sum())}"
            )
    return result


def detect_eol(
    observations: pd.DataFrame,
    threshold_Ah: float,
    *,
    dense_measurement: bool,
) -> EOLResult:
    """Detect exact/interval/left/right EOL without interpolation."""
    required = {
        "Raw_Cycle_Index",
        "Diagnostic_Index",
        "Cumulative_Aging_Operation_Count",
        "Capacity_Ah",
    }
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"EOL observations missing columns: {sorted(missing)}")
    data = observations.copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Raw_Cycle_Index", "Cumulative_Aging_Operation_Count", "Capacity_Ah"])
    data = data.sort_values(
        ["Cumulative_Aging_Operation_Count", "Raw_Cycle_Index"], kind="stable"
    ).reset_index(drop=True)
    if data.empty:
        return EOLResult(
            "RIGHT_CENSORED",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "NO_VALID_PRIMARY_CAPACITY_OBSERVATIONS",
        )
    below = data["Capacity_Ah"].le(float(threshold_Ah))
    last = data.iloc[-1]
    if not below.any():
        return EOLResult(
            "RIGHT_CENSORED",
            float(last["Cumulative_Aging_Operation_Count"]),
            None,
            None,
            None,
            None,
            float(last["Cumulative_Aging_Operation_Count"]),
            float(last["Raw_Cycle_Index"]),
            float(last["Capacity_Ah"]),
            "LAST_VALID_PRIMARY_OBSERVATION_ABOVE_THRESHOLD",
        )
    position = int(np.flatnonzero(below.to_numpy())[0])
    current = data.iloc[position]
    current_age = float(current["Cumulative_Aging_Operation_Count"])
    diag = (
        float(current["Diagnostic_Index"])
        if np.isfinite(current["Diagnostic_Index"])
        else None
    )
    if position == 0:
        return EOLResult(
            "LEFT_CENSORED_OR_BASELINE_UNRESOLVED",
            None,
            current_age,
            None,
            diag,
            float(current["Raw_Cycle_Index"]),
            float(last["Cumulative_Aging_Operation_Count"]),
            float(last["Raw_Cycle_Index"]),
            float(last["Capacity_Ah"]),
            "FIRST_VALID_PRIMARY_OBSERVATION_ALREADY_AT_OR_BELOW_THRESHOLD",
        )
    previous = data.iloc[position - 1]
    previous_age = float(previous["Cumulative_Aging_Operation_Count"])
    if dense_measurement and current_age - previous_age <= 1.0:
        return EOLResult(
            "EXACT_OBSERVED",
            current_age,
            current_age,
            current_age,
            diag,
            float(current["Raw_Cycle_Index"]),
            float(last["Cumulative_Aging_Operation_Count"]),
            float(last["Raw_Cycle_Index"]),
            float(last["Capacity_Ah"]),
            "FIRST_DENSE_PRIMARY_OBSERVATION_AT_OR_BELOW_THRESHOLD",
        )
    return EOLResult(
        "INTERVAL_CENSORED",
        previous_age,
        current_age,
        None,
        diag,
        float(current["Raw_Cycle_Index"]),
        float(last["Cumulative_Aging_Operation_Count"]),
        float(last["Raw_Cycle_Index"]),
        float(last["Capacity_Ah"]),
        "THRESHOLD_CROSSING_BETWEEN_SPARSE_PRIMARY_CAPACITY_OBSERVATIONS",
    )


def rul_from_eol(current_aging_cycle: float, eol: EOLResult) -> dict[str, Any]:
    if eol.status == "QREF_UNAVAILABLE":
        return {
            "RUL_Cycles": np.nan,
            "RUL_Lower_Bound": np.nan,
            "RUL_Upper_Bound": np.nan,
            "RUL_Status": "QREF_UNAVAILABLE",
        }
    if eol.status == "EXACT_OBSERVED" and eol.exact_aging_cycle is not None:
        value = max(0.0, eol.exact_aging_cycle - current_aging_cycle)
        return {
            "RUL_Cycles": value,
            "RUL_Lower_Bound": value,
            "RUL_Upper_Bound": value,
            "RUL_Status": "EXACT_OBSERVED_EOL",
        }
    if (
        eol.status == "INTERVAL_CENSORED"
        and eol.lower_bound_aging_cycle is not None
        and eol.upper_bound_aging_cycle is not None
    ):
        return {
            "RUL_Cycles": np.nan,
            "RUL_Lower_Bound": max(
                0.0, eol.lower_bound_aging_cycle - current_aging_cycle
            ),
            "RUL_Upper_Bound": max(
                0.0, eol.upper_bound_aging_cycle - current_aging_cycle
            ),
            "RUL_Status": "INTERVAL_CENSORED_EOL",
        }
    if eol.status == "RIGHT_CENSORED":
        return {
            "RUL_Cycles": np.nan,
            "RUL_Lower_Bound": np.nan,
            "RUL_Upper_Bound": np.nan,
            "RUL_Status": "RIGHT_CENSORED_EOL",
        }
    return {
        "RUL_Cycles": np.nan,
        "RUL_Lower_Bound": np.nan,
        "RUL_Upper_Bound": np.nan,
        "RUL_Status": "LEFT_CENSORED_OR_BASELINE_UNRESOLVED_EOL",
    }


def _load_configs(scan_dir: Path) -> dict[int, dict[str, Any]]:
    return {
        batch_id: _load_json(scan_dir / f"batch{batch_id}_config_draft.json")
        for batch_id in range(2, 7)
    }


def _new_cycle_qc(
    validated: pd.DataFrame,
    configs: Mapping[int, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in validated.itertuples(index=False):
        row = item._asdict()
        batch_id = int(row["Batch_ID"])
        config = configs[batch_id]
        protocol_class = str(row["Validated_Protocol_Class"])
        eligible = str(row["Validated_Primary_Capacity_Eligible"])
        capacity = float(row["Summary_Discharge_Capacity_Ah"])
        full_protocol = protocol_class in {
            "INITIAL_CAPACITY_TEST",
            "STANDARD_FULL_DISCHARGE_1C",
            "VARIABLE_FULL_DISCHARGE_0P5C",
            "VARIABLE_FULL_DISCHARGE_1C",
            "VARIABLE_FULL_DISCHARGE_2C",
            "VARIABLE_FULL_DISCHARGE_3C",
            "VARIABLE_FULL_DISCHARGE_5C",
            "CAPACITY_DIAGNOSTIC_1C",
            "CAPACITY_DIAGNOSTIC_0P5C",
        }
        invalid: list[str] = []
        checks: list[str] = []
        if not np.isfinite(capacity) or capacity <= 0:
            invalid.append("NONPOSITIVE_CAPACITY")
        raw_capacity = pd.to_numeric(row["Raw_Discharge_Capacity_Ah"], errors="coerce")
        if not np.isfinite(raw_capacity):
            invalid.append("MISSING_MAIN_DISCHARGE")
        if full_protocol:
            charge_max = pd.to_numeric(row["Observed_Charge_Max_V"], errors="coerce")
            charge_end = pd.to_numeric(row["Observed_Charge_End_Current_A"], errors="coerce")
            expected_cv = pd.to_numeric(row["Expected_CV_Cutoff_A"], errors="coerce")
            if not np.isfinite(charge_max) or charge_max < float(config["charge_complete_min_V"]):
                invalid.append("INCOMPLETE_CHARGE_PROTOCOL")
            if np.isfinite(expected_cv) and (
                not np.isfinite(charge_end) or charge_end > expected_cv
            ):
                invalid.append("INCOMPLETE_CHARGE_PROTOCOL")
            if not _as_bool(row["Full_Discharge_Observed"]):
                invalid.append("INCOMPLETE_DISCHARGE_PROTOCOL")
        if _as_bool(row["Voltage_Spike_During_Discharge"]):
            invalid.append("VOLTAGE_SPIKE_DURING_DISCHARGE")
        if _as_bool(row["Abnormal_Voltage_Jump"]):
            invalid.append("ABNORMAL_VOLTAGE_JUMP")
        difference = pd.to_numeric(row["Capacity_Difference_Ah"], errors="coerce")
        tolerance = max(
            float(config["capacity_match_abs_tolerance_Ah"]),
            float(config["capacity_match_relative_tolerance"]) * abs(capacity),
        )
        if np.isfinite(difference) and abs(difference) > tolerance:
            checks.append("CAPACITY_SUMMARY_RAW_MISMATCH")
        invalid = list(dict.fromkeys(invalid))
        checks = list(dict.fromkeys(checks))
        auto_flag = "INVALID_CANDIDATE" if invalid else ("CHECK" if checks else "PASS")
        if protocol_class == "INITIAL_CAPACITY_TEST":
            include = "NO" if invalid else ("REVIEW" if checks else "YES")
            basis = "INITIAL_TEST_QREF_ONLY" if include == "YES" else "QC_REVIEW_OR_EXCLUSION"
        elif eligible != "YES":
            include = "NO"
            basis = "NONPRIMARY_PROTOCOL"
        elif auto_flag == "PASS":
            include = "YES"
            basis = "AUTO_QC_PASS_PRIMARY_COMPARABLE"
        elif auto_flag == "CHECK":
            include = "REVIEW"
            basis = "AUTO_QC_CHECK_REQUIRES_MANUAL_REVIEW"
        else:
            include = "NO"
            basis = "AUTO_QC_INVALID_CANDIDATE"
        rows.append(
            {
                **row,
                "Protocol_Class": protocol_class,
                "Primary_Capacity_Eligible": eligible,
                "Capacity_Ah": capacity,
                "Auto_QC_Flag": auto_flag,
                "Auto_QC_Reason_Codes": "|".join(invalid + checks),
                "Final_Include": include,
                "Final_Include_Basis": basis,
                "Provisional_Include": "NO",
                "Capacity_Jump_Flag": False,
                "Jump_Reference_Raw_Cycle": pd.NA,
                "Jump_Reference_Capacity_Ah": np.nan,
                "Jump_Delta_Ah": np.nan,
                "Jump_Delta_Pct": np.nan,
                "Jump_Reference_Status": "NOT_EVALUATED",
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Jump reference is the nearest preceding protocol-comparable, protocol-complete cycle.
    for (_, protocol_class), indices in result.loc[
        result["Primary_Capacity_Eligible"].eq("YES")
    ].groupby(["Cell_ID", "Protocol_Class"], sort=False).groups.items():
        reference_index: int | None = None
        ordered = _coerce_numeric_order_columns(result.loc[list(indices)]).sort_values(
            ["Raw_Cycle_Index", "Diagnostic_Index"], kind="stable"
        )
        for index in ordered.index:
            current = result.loc[index]
            reference_valid = (
                current["Auto_QC_Flag"] != "INVALID_CANDIDATE"
                and current["Final_Include"] in {"YES", "REVIEW"}
                and float(current["Capacity_Ah"]) > 0
            )
            if reference_index is None:
                result.at[index, "Jump_Reference_Status"] = "NO_VALID_JUMP_REFERENCE"
            else:
                reference = result.loc[reference_index]
                delta = float(current["Capacity_Ah"]) - float(reference["Capacity_Ah"])
                threshold = max(
                    float(configs[int(current["Batch_ID"])]["capacity_jump_abs_threshold_Ah"]),
                    float(configs[int(current["Batch_ID"])]["capacity_jump_relative_threshold"])
                    * float(reference["Capacity_Ah"]),
                )
                result.at[index, "Jump_Reference_Raw_Cycle"] = int(
                    reference["Raw_Cycle_Index"]
                )
                result.at[index, "Jump_Reference_Capacity_Ah"] = float(
                    reference["Capacity_Ah"]
                )
                result.at[index, "Jump_Delta_Ah"] = delta
                result.at[index, "Jump_Delta_Pct"] = (
                    100.0 * delta / float(reference["Capacity_Ah"])
                )
                result.at[index, "Jump_Reference_Status"] = "VALID_COMPARABLE_REFERENCE"
                if reference_valid and abs(delta) > threshold:
                    result.at[index, "Capacity_Jump_Flag"] = True
                    codes = [
                        code
                        for code in str(result.at[index, "Auto_QC_Reason_Codes"]).split("|")
                        if code
                    ]
                    codes.append("CAPACITY_JUMP")
                    result.at[index, "Auto_QC_Reason_Codes"] = "|".join(dict.fromkeys(codes))
                    if result.at[index, "Auto_QC_Flag"] == "PASS":
                        result.at[index, "Auto_QC_Flag"] = "CHECK"
                        result.at[index, "Final_Include"] = "REVIEW"
                        result.at[index, "Final_Include_Basis"] = (
                            "CAPACITY_JUMP_ONLY_REQUIRES_MANUAL_REVIEW"
                        )
                        result.at[index, "Provisional_Include"] = "YES"
            if reference_valid:
                reference_index = int(index)
    return result


def _frozen_cycle_qc(frozen_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(frozen_dir / "xjtu_cycle_capacity_final.csv")
    source.insert(0, "Batch_ID", 1)
    source["Capacity_Ah"] = source["Summary_Discharge_Capacity_Ah"]
    source["Primary_Capacity_Eligible"] = np.where(
        source["Raw_Cycle_Index"].eq(1), "NO", "YES"
    )
    source["Aging_Operation_Index"] = source["Aging_Cycle_Index"]
    source["Cumulative_Aging_Operation_Count"] = source["Aging_Cycle_Index"].fillna(0)
    source["Aging_Operations_Before_Cycle"] = np.where(
        source["Raw_Cycle_Index"].eq(1), 0, source["Aging_Cycle_Index"] - 1
    )
    source["Diagnostic_Index"] = source["Aging_Cycle_Index"]
    source["Reference_Status"] = "FROZEN_REFERENCE"
    return source


def _qref_row_from_observations(cell: str, batch_id: int, cell_rows: pd.DataFrame) -> dict[str, Any]:
    valid = cell_rows.loc[cell_rows["Final_Include"].eq("YES")].copy()
    initial = valid.loc[valid["Protocol_Class"].eq("INITIAL_CAPACITY_TEST")]
    if not initial.empty:
        initial = _coerce_numeric_order_columns(
            initial, ("Raw_Cycle_Index", "Cumulative_Aging_Operation_Count")
        ).sort_values("Raw_Cycle_Index", kind="stable")
    recurring = valid.loc[
        valid["Primary_Capacity_Eligible"].eq("YES")
        & ~valid["Protocol_Class"].eq("INITIAL_CAPACITY_TEST")
    ]
    recurring = _coerce_numeric_order_columns(recurring).sort_values(
        ["Diagnostic_Index", "Raw_Cycle_Index"],
        kind="stable",
        na_position="last",
    )
    result: dict[str, Any] = {
        "Batch_ID": batch_id,
        "Cell_ID": cell,
        "Qref_nominal_Ah": NOMINAL_CAPACITY_AH,
        "Qref_nominal_Status": "AVAILABLE",
        "Qref_Nominal_Source": "DATASET_NOMINAL_CAPACITY",
    }
    if initial.empty:
        result.update(
            {
                "Qref_initial_test_Ah": np.nan,
                "Qref_initial_test_Status": "UNAVAILABLE_NO_BOL_INITIAL_TEST",
                "Qref_Initial_Test_Source_Cycle": pd.NA,
            }
        )
    else:
        first = initial.iloc[0]
        result.update(
            {
                "Qref_initial_test_Ah": float(first["Capacity_Ah"]),
                "Qref_initial_test_Status": "AVAILABLE",
                "Qref_Initial_Test_Source_Cycle": int(first["Raw_Cycle_Index"]),
            }
        )
    if recurring.empty:
        result.update(
            {
                "Qref_first_comparable_Ah": np.nan,
                "Qref_first_comparable_Status": "UNAVAILABLE_NO_VALID_COMPARABLE_CAPACITY",
                "Qref_First_Comparable_Source_Cycle": pd.NA,
            }
        )
    else:
        first = recurring.iloc[0]
        result.update(
            {
                "Qref_first_comparable_Ah": float(first["Capacity_Ah"]),
                "Qref_first_comparable_Status": "AVAILABLE",
                "Qref_First_Comparable_Source_Cycle": int(first["Raw_Cycle_Index"]),
            }
        )
    first_five = recurring.head(5)
    if len(first_five) < 5:
        result.update(
            {
                "Qref_early_median5_comparable_Ah": np.nan,
                "Qref_early_median5_comparable_Status": "UNAVAILABLE_FEWER_THAN_5_VALID_COMPARABLE_CAPACITIES",
                "Qref_Early_Median5_Comparable_Source_Cycles": "",
            }
        )
    else:
        result.update(
            {
                "Qref_early_median5_comparable_Ah": float(first_five["Capacity_Ah"].median()),
                "Qref_early_median5_comparable_Status": "AVAILABLE",
                "Qref_Early_Median5_Comparable_Source_Cycles": "|".join(
                    str(int(value)) for value in first_five["Raw_Cycle_Index"]
                ),
            }
        )
    return result


def _frozen_qref(frozen_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(frozen_dir / "xjtu_qref_summary_final.csv")
    source.insert(0, "Batch_ID", 1)
    return source.rename(
        columns={
            "Qref_first_standard_Ah": "Qref_first_comparable_Ah",
            "Qref_first_standard_Status": "Qref_first_comparable_Status",
            "Qref_First_Standard_Source_Cycle": "Qref_First_Comparable_Source_Cycle",
            "Qref_early_median5_Ah": "Qref_early_median5_comparable_Ah",
            "Qref_early_median5_Status": "Qref_early_median5_comparable_Status",
            "Qref_Early_Median5_Source_Cycles": "Qref_Early_Median5_Comparable_Source_Cycles",
        }
    )


def _finalize_qref_summary(qref: pd.DataFrame) -> pd.DataFrame:
    """Normalize sensitivity fields after Batch-1 semantic name mapping."""
    result = qref.copy()
    numeric_columns = [
        "Qref_nominal_Ah",
        "Qref_initial_test_Ah",
        "Qref_first_comparable_Ah",
        "Qref_early_median5_comparable_Ah",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    def difference(
        output_prefix: str, left: str, right: str
    ) -> None:
        delta = result[left] - result[right]
        result[f"{output_prefix}_Ah"] = delta
        result[f"{output_prefix}_pct"] = np.where(
            result[right].notna() & result[right].ne(0),
            100.0 * delta / result[right],
            np.nan,
        )

    difference(
        "Qref_initial_minus_nominal",
        "Qref_initial_test_Ah",
        "Qref_nominal_Ah",
    )
    difference(
        "Qref_first_comparable_minus_initial",
        "Qref_first_comparable_Ah",
        "Qref_initial_test_Ah",
    )
    difference(
        "Qref_early_median5_comparable_minus_initial",
        "Qref_early_median5_comparable_Ah",
        "Qref_initial_test_Ah",
    )
    legacy_columns = [
        "Qref_firststandard_minus_initial_Ah",
        "Qref_firststandard_minus_initial_pct",
        "Qref_median5_minus_initial_Ah",
        "Qref_median5_minus_initial_pct",
    ]
    result = result.drop(columns=[c for c in legacy_columns if c in result.columns])
    result["Analysis_Inclusion_Field"] = "Final_Include"
    return result


def _qref_long_values(row: pd.Series) -> dict[str, float]:
    return {
        "nominal": float(row["Qref_nominal_Ah"]),
        "initial_test": pd.to_numeric(row["Qref_initial_test_Ah"], errors="coerce"),
        "first_comparable": pd.to_numeric(
            row["Qref_first_comparable_Ah"], errors="coerce"
        ),
        "early_median5_comparable": pd.to_numeric(
            row["Qref_early_median5_comparable_Ah"], errors="coerce"
        ),
    }


def _new_soh(primary: pd.DataFrame, qref: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for qref_row in qref.itertuples(index=False):
        cell = qref_row.Cell_ID
        observations = primary.loc[primary["Cell_ID"].eq(cell)]
        values = _qref_long_values(pd.Series(qref_row._asdict()))
        for definition, reference in values.items():
            if not np.isfinite(reference):
                continue
            for item in observations.itertuples(index=False):
                fraction = float(item.Capacity_Ah) / float(reference)
                rows.append(
                    {
                        "Batch_ID": int(item.Batch_ID),
                        "Cell_ID": cell,
                        "Raw_Cycle_Index": item.Raw_Cycle_Index,
                        "Aging_Operation_Index": item.Aging_Operation_Index,
                        "Diagnostic_Index": item.Diagnostic_Index,
                        "Cumulative_Aging_Operation_Count": item.Cumulative_Aging_Operation_Count,
                        "Capacity_Ah": item.Capacity_Ah,
                        "Auto_QC_Flag": item.Auto_QC_Flag,
                        "Final_Include": item.Final_Include,
                        "Qref_Definition": definition,
                        "Qref_Ah": reference,
                        "SOH_Fraction": fraction,
                        "SOH_Percent": 100.0 * fraction,
                    }
                )
    return pd.DataFrame(rows)


def _frozen_soh(frozen_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(frozen_dir / "xjtu_soh_labels_long_final.csv")
    source.insert(0, "Batch_ID", 1)
    source["Qref_Definition"] = source["Qref_Definition"].replace(
        {
            "first_standard": "first_comparable",
            "early_median5": "early_median5_comparable",
        }
    )
    source["Aging_Operation_Index"] = source["Aging_Cycle_Index"]
    source["Cumulative_Aging_Operation_Count"] = source["Aging_Cycle_Index"]
    source["Diagnostic_Index"] = source["Aging_Cycle_Index"]
    return source


def _new_eol(primary: pd.DataFrame, qref: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], EOLResult]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], EOLResult] = {}
    for qref_row in qref.itertuples(index=False):
        row_series = pd.Series(qref_row._asdict())
        batch_id = int(row_series["Batch_ID"])
        cell = str(row_series["Cell_ID"])
        observations = primary.loc[primary["Cell_ID"].eq(cell), [
            "Raw_Cycle_Index",
            "Diagnostic_Index",
            "Cumulative_Aging_Operation_Count",
            "Capacity_Ah",
        ]]
        dense = batch_id == 2
        qrefs = _qref_long_values(row_series)
        thresholds = [(definition, reference, 0.8 * reference) for definition, reference in qrefs.items()]
        thresholds.append(("fixed_absolute_1p6Ah", np.nan, 1.6))
        for definition, reference, threshold in thresholds:
            threshold_type = (
                "FIXED_ABSOLUTE" if definition == "fixed_absolute_1p6Ah" else "QREF_RELATIVE"
            )
            if definition != "fixed_absolute_1p6Ah" and not np.isfinite(reference):
                eol = EOLResult(
                    "QREF_UNAVAILABLE",
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
                threshold = np.nan
            else:
                eol = detect_eol(observations, threshold, dense_measurement=dense)
            lookup[(cell, definition)] = eol
            exact_observed = eol.status == "EXACT_OBSERVED"
            rows.append(
                {
                    "Batch_ID": batch_id,
                    "Cell_ID": cell,
                    "Qref_Definition": definition,
                    "Threshold_Type": threshold_type,
                    "Qref_Ah": reference,
                    "EOThreshold_Ah": threshold,
                    "EOL_Observed": (
                        "YES"
                        if exact_observed
                        else ("NA" if eol.status == "QREF_UNAVAILABLE" else "NO")
                    ),
                    "EOL_Status": eol.status,
                    "EOL_Raw_Cycle": (
                        eol.first_below_raw_cycle if exact_observed else np.nan
                    ),
                    "EOL_Aging_Cycle": (
                        eol.exact_aging_cycle if exact_observed else np.nan
                    ),
                    "EOL_Exact_Aging_Cycle": eol.exact_aging_cycle,
                    "EOL_Lower_Bound_Aging_Cycle": eol.lower_bound_aging_cycle,
                    "EOL_Upper_Bound_Aging_Cycle": eol.upper_bound_aging_cycle,
                    "EOL_First_Below_Diagnostic_Index": eol.first_below_diagnostic_index,
                    "EOL_First_Below_Raw_Cycle": eol.first_below_raw_cycle,
                    "Last_Valid_Aging_Cycle": eol.last_valid_aging_cycle,
                    "Last_Valid_Raw_Cycle": eol.last_valid_raw_cycle,
                    "Last_Valid_Capacity_Ah": eol.last_valid_capacity_Ah,
                    "Censoring_Reason": eol.censoring_reason,
                    "Analysis_Inclusion_Field": "Final_Include",
                }
            )
    return pd.DataFrame(rows), lookup


def _frozen_eol(frozen_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(frozen_dir / "xjtu_eol_sensitivity_final.csv")
    source.insert(0, "Batch_ID", 1)
    source["Qref_Definition"] = source["Qref_Definition"].replace(
        {
            "first_standard": "first_comparable",
            "early_median5": "early_median5_comparable",
        }
    )
    source["EOL_Status"] = source["EOL_Status"].replace({"OBSERVED": "EXACT_OBSERVED"})
    source["EOL_Exact_Aging_Cycle"] = np.where(
        source["EOL_Status"].eq("EXACT_OBSERVED"), source["EOL_Aging_Cycle"], np.nan
    )
    source["EOL_Lower_Bound_Aging_Cycle"] = np.where(
        source["EOL_Status"].eq("EXACT_OBSERVED"),
        source["EOL_Aging_Cycle"],
        source["Last_Valid_Raw_Cycle"] - 1,
    )
    source["EOL_Upper_Bound_Aging_Cycle"] = np.where(
        source["EOL_Status"].eq("EXACT_OBSERVED"), source["EOL_Aging_Cycle"], np.nan
    )
    source["EOL_First_Below_Diagnostic_Index"] = np.where(
        source["EOL_Status"].eq("EXACT_OBSERVED"), source["EOL_Aging_Cycle"], np.nan
    )
    source["EOL_First_Below_Raw_Cycle"] = source["EOL_Raw_Cycle"]
    source["Last_Valid_Aging_Cycle"] = source["Last_Valid_Raw_Cycle"] - 1
    return source


def _eol_lookup_from_frame(eol: pd.DataFrame) -> dict[tuple[str, str], EOLResult]:
    result: dict[tuple[str, str], EOLResult] = {}
    for item in eol.itertuples(index=False):
        result[(str(item.Cell_ID), str(item.Qref_Definition))] = EOLResult(
            status=str(item.EOL_Status),
            lower_bound_aging_cycle=_finite_or_none(item.EOL_Lower_Bound_Aging_Cycle),
            upper_bound_aging_cycle=_finite_or_none(item.EOL_Upper_Bound_Aging_Cycle),
            exact_aging_cycle=_finite_or_none(item.EOL_Exact_Aging_Cycle),
            first_below_diagnostic_index=_finite_or_none(item.EOL_First_Below_Diagnostic_Index),
            first_below_raw_cycle=_finite_or_none(item.EOL_First_Below_Raw_Cycle),
            last_valid_aging_cycle=_finite_or_none(item.Last_Valid_Aging_Cycle),
            last_valid_raw_cycle=_finite_or_none(item.Last_Valid_Raw_Cycle),
            last_valid_capacity_Ah=_finite_or_none(item.Last_Valid_Capacity_Ah),
            censoring_reason=str(item.Censoring_Reason),
        )
    return result


def _finite_or_none(value: Any) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if np.isfinite(number) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _pairwise(eol: pd.DataFrame, qref: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    qref_by_cell = qref.set_index("Cell_ID")
    status_labels = {
        "EXACT_OBSERVED": "EXACT",
        "INTERVAL_CENSORED": "INTERVAL",
        "RIGHT_CENSORED": "RIGHT_CENSORED",
        "LEFT_CENSORED_OR_BASELINE_UNRESOLVED": "LEFT_CENSORED",
        "QREF_UNAVAILABLE": "QREF_UNAVAILABLE",
    }
    for cell, qrow in qref_by_cell.iterrows():
        cell_eol = eol.loc[
            eol["Cell_ID"].eq(cell)
            & eol["Qref_Definition"].isin(QREF_DEFINITIONS)
        ]
        eol_by_definition = cell_eol.set_index("Qref_Definition")
        for definition_a, definition_b in combinations(QREF_DEFINITIONS, 2):
            a = (
                eol_by_definition.loc[definition_a]
                if definition_a in eol_by_definition.index
                else pd.Series(dtype=object)
            )
            b = (
                eol_by_definition.loc[definition_b]
                if definition_b in eol_by_definition.index
                else pd.Series(dtype=object)
            )
            status_a = str(a.get("EOL_Status", "QREF_UNAVAILABLE"))
            status_b = str(b.get("EOL_Status", "QREF_UNAVAILABLE"))
            both_exact = status_a == status_b == "EXACT_OBSERVED"
            exact_a = pd.to_numeric(a.get("EOL_Exact_Aging_Cycle", np.nan), errors="coerce")
            exact_b = pd.to_numeric(b.get("EOL_Exact_Aging_Cycle", np.nan), errors="coerce")
            delta = (
                float(exact_b) - float(exact_a)
                if both_exact
                else np.nan
            )
            lower_a = pd.to_numeric(
                a.get("EOL_Lower_Bound_Aging_Cycle", np.nan), errors="coerce"
            )
            upper_a = pd.to_numeric(
                a.get("EOL_Upper_Bound_Aging_Cycle", np.nan), errors="coerce"
            )
            lower_b = pd.to_numeric(
                b.get("EOL_Lower_Bound_Aging_Cycle", np.nan), errors="coerce"
            )
            upper_b = pd.to_numeric(
                b.get("EOL_Upper_Bound_Aging_Cycle", np.nan), errors="coerce"
            )
            delta_lower = (
                float(lower_b) - float(upper_a)
                if np.isfinite(lower_b) and np.isfinite(upper_a)
                else np.nan
            )
            delta_upper = (
                float(upper_b) - float(lower_a)
                if np.isfinite(upper_b) and np.isfinite(lower_a)
                else np.nan
            )
            if status_a == status_b == "RIGHT_CENSORED":
                transition = "BOTH_RIGHT_CENSORED"
            elif status_a == status_b == "QREF_UNAVAILABLE":
                transition = "BOTH_QREF_UNAVAILABLE"
            else:
                transition = (
                    f"{status_labels.get(status_a, status_a)}_TO_"
                    f"{status_labels.get(status_b, status_b)}"
                )
            rows.append(
                {
                    "Batch_ID": int(qrow["Batch_ID"]),
                    "Cell_ID": cell,
                    "Definition_A": definition_a,
                    "Definition_B": definition_b,
                    "Qref_A_Ah": _qref_long_values(qrow)[definition_a],
                    "Qref_B_Ah": _qref_long_values(qrow)[definition_b],
                    "EOL_Status_A": status_a,
                    "EOL_Status_B": status_b,
                    "EOL_Exact_A": exact_a,
                    "EOL_Exact_B": exact_b,
                    "EOL_Lower_Bound_A": lower_a,
                    "EOL_Upper_Bound_A": upper_a,
                    "EOL_Lower_Bound_B": lower_b,
                    "EOL_Upper_Bound_B": upper_b,
                    "Delta_EOL_Cycles": delta,
                    "Delta_EOL_Lower_Bound_Cycles": delta_lower,
                    "Delta_EOL_Upper_Bound_Cycles": delta_upper,
                    "Censoring_Status_Change": transition,
                    "Analysis_Inclusion_Field": "Final_Include",
                }
            )
    return pd.DataFrame(rows)


def _frozen_pairwise(frozen_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(frozen_dir / "xjtu_eol_pairwise_shift_final.csv")
    source.insert(0, "Batch_ID", 1)
    mapping = {
        "first_standard": "first_comparable",
        "early_median5": "early_median5_comparable",
    }
    source["Definition_A"] = source["Definition_A"].replace(mapping)
    source["Definition_B"] = source["Definition_B"].replace(mapping)
    source["EOL_Status_A"] = source["EOL_Status_A"].replace(
        {"OBSERVED": "EXACT_OBSERVED"}
    )
    source["EOL_Status_B"] = source["EOL_Status_B"].replace(
        {"OBSERVED": "EXACT_OBSERVED"}
    )
    source["EOL_Exact_A"] = np.where(
        source["EOL_Status_A"].eq("EXACT_OBSERVED"), source["EOL_A"], np.nan
    )
    source["EOL_Exact_B"] = np.where(
        source["EOL_Status_B"].eq("EXACT_OBSERVED"), source["EOL_B"], np.nan
    )
    source["Censoring_Status_Change"] = [
        (
            "BOTH_EXACT_OBSERVED"
            if status_a == status_b == "EXACT_OBSERVED"
            else (
                "SAME_CENSORING_STATUS"
                if status_a == status_b
                else f"{status_a}_TO_{status_b}"
            )
        )
        for status_a, status_b in zip(
            source["EOL_Status_A"], source["EOL_Status_B"]
        )
    ]
    return source


def _rul(primary: pd.DataFrame, eol_lookup: Mapping[tuple[str, str], EOLResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in primary.itertuples(index=False):
        age = float(item.Cumulative_Aging_Operation_Count)
        for definition in QREF_DEFINITIONS:
            eol = eol_lookup.get((str(item.Cell_ID), definition))
            if eol is None:
                continue
            rul = rul_from_eol(age, eol)
            rows.append(
                {
                    "Batch_ID": int(item.Batch_ID),
                    "Cell_ID": item.Cell_ID,
                    "Raw_Cycle_Index": item.Raw_Cycle_Index,
                    "Aging_Cycle_Index": getattr(item, "Aging_Cycle_Index", np.nan),
                    "Final_Include": item.Final_Include,
                    "Aging_Operation_Index": item.Aging_Operation_Index,
                    "Diagnostic_Index": item.Diagnostic_Index,
                    "Cumulative_Aging_Operation_Count": age,
                    "Qref_Definition": definition,
                    "EOL_Status": eol.status,
                    "EOL_Aging_Cycle": eol.exact_aging_cycle,
                    "EOL_Exact_Aging_Cycle": eol.exact_aging_cycle,
                    "EOL_Lower_Bound_Aging_Cycle": eol.lower_bound_aging_cycle,
                    "EOL_Upper_Bound_Aging_Cycle": eol.upper_bound_aging_cycle,
                    **rul,
                }
            )
    return pd.DataFrame(rows)


def _frozen_rul(frozen_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(frozen_dir / "xjtu_rul_sensitivity_final.csv")
    source.insert(0, "Batch_ID", 1)
    source["Qref_Definition"] = source["Qref_Definition"].replace(
        {
            "first_standard": "first_comparable",
            "early_median5": "early_median5_comparable",
        }
    )
    source["EOL_Status"] = source["EOL_Status"].replace(
        {"OBSERVED": "EXACT_OBSERVED"}
    )
    source["Aging_Operation_Index"] = source["Aging_Cycle_Index"]
    source["Diagnostic_Index"] = source["Aging_Cycle_Index"]
    source["Cumulative_Aging_Operation_Count"] = source["Aging_Cycle_Index"]
    exact = source["EOL_Status"].eq("EXACT_OBSERVED")
    source["EOL_Exact_Aging_Cycle"] = np.where(
        exact, source["EOL_Aging_Cycle"], np.nan
    )
    source["EOL_Lower_Bound_Aging_Cycle"] = np.where(
        exact, source["EOL_Aging_Cycle"], np.nan
    )
    source["EOL_Upper_Bound_Aging_Cycle"] = np.where(
        exact, source["EOL_Aging_Cycle"], np.nan
    )
    source["RUL_Lower_Bound"] = np.where(
        source["RUL_Status"].eq("DEFINED"), source["RUL_Cycles"], np.nan
    )
    source["RUL_Upper_Bound"] = np.where(
        source["RUL_Status"].eq("DEFINED"), source["RUL_Cycles"], np.nan
    )
    source["RUL_Status"] = source["RUL_Status"].replace(
        {
            "DEFINED": "EXACT_OBSERVED_EOL",
            "CENSORED_EOL": "RIGHT_CENSORED_EOL",
        }
    )
    return source


def _known_data_quality_notes(
    validation_dir: Path,
    frozen_dir: Path,
    cycle_qc: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "Batch_ID",
        "Cell_ID",
        "Source_File",
        "Raw_Cycle_Index",
        "Note_Source",
        "Data_Quality_Code",
        "Severity",
        "Requires_Human_Review",
        "Auto_QC_Flag",
        "Final_Include",
        "Primary_Capacity_Eligible",
        "Details",
    ]
    cycle_lookup = cycle_qc.set_index(["Cell_ID", "Raw_Cycle_Index"])
    rows: list[dict[str, Any]] = []
    protocol_notes = pd.read_csv(
        validation_dir / "protocol_anomalies_v2.csv",
        keep_default_na=False,
        low_memory=False,
    )
    for item in protocol_notes.itertuples(index=False):
        key = (str(item.Cell_ID), int(item.Raw_Cycle_Index))
        cycle = cycle_lookup.loc[key]
        rows.append(
            {
                "Batch_ID": int(item.Batch_ID),
                "Cell_ID": item.Cell_ID,
                "Source_File": item.Source_File,
                "Raw_Cycle_Index": int(item.Raw_Cycle_Index),
                "Note_Source": "VALIDATED_PROTOCOL_INFORMATION",
                "Data_Quality_Code": item.Resolved_Anomaly_Codes,
                "Severity": item.Severity,
                "Requires_Human_Review": item.Requires_Human_Review,
                "Auto_QC_Flag": cycle["Auto_QC_Flag"],
                "Final_Include": cycle["Final_Include"],
                "Primary_Capacity_Eligible": cycle[
                    "Primary_Capacity_Eligible"
                ],
                "Details": item.Resolution_Basis,
            }
        )

    manual = pd.read_csv(
        frozen_dir / "xjtu_manual_qc_review.csv",
        keep_default_na=False,
        low_memory=False,
    )
    for item in manual.itertuples(index=False):
        key = (str(item.Cell_ID), int(item.Raw_Cycle_Index))
        cycle = cycle_lookup.loc[key]
        rows.append(
            {
                "Batch_ID": 1,
                "Cell_ID": item.Cell_ID,
                "Source_File": cycle["Source_File"],
                "Raw_Cycle_Index": int(item.Raw_Cycle_Index),
                "Note_Source": "FROZEN_MANUAL_QC_DECISION",
                "Data_Quality_Code": item.Auto_QC_Reason_Codes,
                "Severity": "EXCLUDE_CONFIRMED",
                "Requires_Human_Review": "NO",
                "Auto_QC_Flag": item.Auto_QC_Flag,
                "Final_Include": item.Final_Include,
                "Primary_Capacity_Eligible": cycle[
                    "Primary_Capacity_Eligible"
                ],
                "Details": item.Manual_Reason,
            }
        )

    documented = {
        (str(row["Cell_ID"]), int(row["Raw_Cycle_Index"])) for row in rows
    }
    flagged = cycle_qc.loc[cycle_qc["Auto_QC_Flag"].ne("PASS")]
    for item in flagged.itertuples(index=False):
        key = (str(item.Cell_ID), int(item.Raw_Cycle_Index))
        if key in documented:
            continue
        rows.append(
            {
                "Batch_ID": int(item.Batch_ID),
                "Cell_ID": item.Cell_ID,
                "Source_File": item.Source_File,
                "Raw_Cycle_Index": int(item.Raw_Cycle_Index),
                "Note_Source": "FULL_RUN_AUTO_QC",
                "Data_Quality_Code": item.Auto_QC_Reason_Codes,
                "Severity": item.Auto_QC_Flag,
                "Requires_Human_Review": (
                    "YES" if item.Auto_QC_Flag == "CHECK" else "NO"
                ),
                "Auto_QC_Flag": item.Auto_QC_Flag,
                "Final_Include": item.Final_Include,
                "Primary_Capacity_Eligible": item.Primary_Capacity_Eligible,
                "Details": item.Final_Include_Basis,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["Batch_ID", "Cell_ID", "Raw_Cycle_Index", "Note_Source"], kind="stable"
    ).reset_index(drop=True)


def _run_regression_tests(
    manifest: pd.DataFrame,
    validated: pd.DataFrame,
    frozen_dir: Path,
    cycle_qc: pd.DataFrame,
    primary: pd.DataFrame,
    qref: pd.DataFrame,
    soh: pd.DataFrame,
    eol: pd.DataFrame,
    pairwise: pd.DataFrame,
    rul: pd.DataFrame,
    known_notes: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    def add(
        test_id: str,
        scope: str,
        expected: object,
        observed: object,
        passed: bool,
        details: str = "",
    ) -> None:
        records.append(
            {
                "Test_ID": test_id,
                "Scope": scope,
                "Expected": expected,
                "Observed": observed,
                "Status": "PASS" if bool(passed) else "FAIL",
                "Details": "" if passed else details,
            }
        )

    canonical = manifest.loc[
        manifest["File_Role"].eq("CANONICAL")
        & manifest["Included_In_Analysis"].eq("YES")
    ]
    batch_counts = {
        int(key): int(value)
        for key, value in canonical.groupby("Batch_ID").size().to_dict().items()
    }
    add(
        "CANONICAL_CELL_COUNT_55",
        "manifest",
        55,
        int(canonical["Cell_ID"].nunique()),
        int(canonical["Cell_ID"].nunique()) == 55,
        "Canonical cell count changed",
    )
    add(
        "CANONICAL_BATCH_COUNTS",
        "manifest",
        "1:8|2:15|3:8|4:8|5:8|6:8",
        "|".join(f"{key}:{batch_counts.get(key, 0)}" for key in range(1, 7)),
        batch_counts == {1: 8, 2: 15, 3: 8, 4: 8, 5: 8, 6: 8},
        "Canonical batch distribution changed",
    )
    duplicate_mismatch = manifest["File_Role"].astype(str).str.contains(
        "DUPLICATE_CONTENT_MISMATCH", regex=False
    )
    if "Hash_Match" in manifest.columns:
        duplicate_rows = manifest["File_Role"].eq("DUPLICATE_CONFIRMED")
        duplicate_hash_bad = duplicate_rows & ~manifest["Hash_Match"].astype(str).str.upper().isin(
            {"TRUE", "YES", "1"}
        )
    else:
        duplicate_hash_bad = pd.Series(False, index=manifest.index)
    add(
        "NO_DUPLICATE_CONTENT_MISMATCH",
        "manifest",
        0,
        int((duplicate_mismatch | duplicate_hash_bad).sum()),
        not (duplicate_mismatch | duplicate_hash_bad).any(),
        "Duplicate content mismatch detected",
    )

    frozen_cycle = pd.read_csv(
        frozen_dir / "xjtu_cycle_capacity_final.csv",
        keep_default_na=False,
        low_memory=False,
    )
    expected_keys = pd.concat(
        [
            frozen_cycle[["Cell_ID", "Raw_Cycle_Index"]],
            validated[["Cell_ID", "Raw_Cycle_Index"]],
        ],
        ignore_index=True,
    ).astype({"Cell_ID": str, "Raw_Cycle_Index": int})
    actual_keys = cycle_qc[["Cell_ID", "Raw_Cycle_Index"]].astype(
        {"Cell_ID": str, "Raw_Cycle_Index": int}
    )
    expected_keys = expected_keys.sort_values(
        ["Cell_ID", "Raw_Cycle_Index"], kind="stable"
    ).reset_index(drop=True)
    actual_keys = actual_keys.sort_values(
        ["Cell_ID", "Raw_Cycle_Index"], kind="stable"
    ).reset_index(drop=True)
    add(
        "RAW_CYCLE_KEYS_PRESERVED",
        "all cycles",
        f"{len(expected_keys)} exact source keys",
        f"{len(actual_keys)} output keys",
        expected_keys.equals(actual_keys),
        "Raw cycles were dropped, duplicated, or renumbered",
    )
    add(
        "CYCLE_KEYS_UNIQUE",
        "all cycles",
        0,
        int(cycle_qc.duplicated(["Cell_ID", "Raw_Cycle_Index"]).sum()),
        not cycle_qc.duplicated(["Cell_ID", "Raw_Cycle_Index"]).any(),
        "Duplicate cycle keys in final QC table",
    )

    unknown_primary = primary["Protocol_Class"].eq("UNKNOWN_PROTOCOL")
    noneligible_primary = ~primary["Primary_Capacity_Eligible"].eq("YES")
    invalid_primary = primary["Auto_QC_Flag"].eq("INVALID_CANDIDATE")
    nonincluded_primary = ~primary["Final_Include"].eq("YES")
    add("NO_UNKNOWN_PROTOCOL_IN_PRIMARY", "primary", 0, int(unknown_primary.sum()), not unknown_primary.any(), "Unknown protocol entered primary capacity")
    add("NO_NONELIGIBLE_PROTOCOL_IN_PRIMARY", "primary", 0, int(noneligible_primary.sum()), not noneligible_primary.any(), "Noneligible protocol entered primary capacity")
    add("NO_INVALID_IN_PRIMARY", "primary", 0, int(invalid_primary.sum()), not invalid_primary.any(), "Invalid cycle entered primary capacity")
    add("PRIMARY_FINAL_INCLUDE_ONLY", "primary", 0, int(nonincluded_primary.sum()), not nonincluded_primary.any(), "Nonincluded cycle entered primary capacity")

    b5 = qref.loc[qref["Batch_ID"].eq(5)]
    b5_initial_values = pd.to_numeric(b5["Qref_initial_test_Ah"], errors="coerce")
    b5_initial_ok = (
        len(b5) == 8
        and b5_initial_values.isna().all()
        and b5["Qref_initial_test_Status"].eq(
            "UNAVAILABLE_NO_BOL_INITIAL_TEST"
        ).all()
        and pd.to_numeric(
            b5["Qref_Initial_Test_Source_Cycle"], errors="coerce"
        ).isna().all()
    )
    add("BATCH5_INITIAL_QREF_UNAVAILABLE", "Batch-5", "8 unavailable rows; no source cycle", f"rows={len(b5)};available_values={int(b5_initial_values.notna().sum())}", b5_initial_ok, "Batch-5 initial-test Qref fallback was introduced")

    traceable = True
    trace_failures: list[str] = []
    primary_keys = {
        (str(item.Cell_ID), int(item.Raw_Cycle_Index))
        for item in primary.itertuples(index=False)
    }
    cycle_lookup = cycle_qc.set_index(["Cell_ID", "Raw_Cycle_Index"])
    for item in qref.itertuples(index=False):
        values = item._asdict()
        cell = str(values["Cell_ID"])
        if values.get("Qref_initial_test_Status") == "AVAILABLE":
            source = pd.to_numeric(
                values.get("Qref_Initial_Test_Source_Cycle"), errors="coerce"
            )
            key = (cell, int(source)) if np.isfinite(source) else None
            if key is None or key not in cycle_lookup.index:
                traceable = False
                trace_failures.append(f"{cell}:initial")
            else:
                source_row = cycle_lookup.loc[key]
                if source_row["Final_Include"] != "YES" or source_row["Protocol_Class"] != "INITIAL_CAPACITY_TEST" and int(values["Batch_ID"]) != 1:
                    traceable = False
                    trace_failures.append(f"{cell}:initial_invalid")
        if values.get("Qref_first_comparable_Status") == "AVAILABLE":
            source = pd.to_numeric(
                values.get("Qref_First_Comparable_Source_Cycle"), errors="coerce"
            )
            if not np.isfinite(source) or (cell, int(source)) not in primary_keys:
                traceable = False
                trace_failures.append(f"{cell}:first")
        if values.get("Qref_early_median5_comparable_Status") == "AVAILABLE":
            sources = [
                value
                for value in str(
                    values.get("Qref_Early_Median5_Comparable_Source_Cycles", "")
                ).split("|")
                if value
            ]
            if len(sources) != 5 or any(
                (cell, int(value)) not in primary_keys for value in sources
            ):
                traceable = False
                trace_failures.append(f"{cell}:median5")
    add("QREF_SOURCE_CYCLES_TRACEABLE", "Qref", "all available Qrefs trace to included source cycles", "PASS" if traceable else "|".join(trace_failures[:20]), traceable, "Qref source-cycle traceability failure")

    interval = eol["EOL_Status"].eq("INTERVAL_CENSORED")
    right = eol["EOL_Status"].eq("RIGHT_CENSORED")
    point_columns = ["EOL_Raw_Cycle", "EOL_Aging_Cycle", "EOL_Exact_Aging_Cycle"]
    interval_point_count = int(
        eol.loc[interval, point_columns].apply(pd.to_numeric, errors="coerce").notna().sum().sum()
    )
    right_point_count = int(
        eol.loc[right, point_columns].apply(pd.to_numeric, errors="coerce").notna().sum().sum()
    )
    add("INTERVAL_EOL_HAS_NO_POINT_ESTIMATE", "EOL", 0, interval_point_count, interval_point_count == 0, "Interval-censored EOL contains a point estimate")
    add("RIGHT_CENSORED_EOL_HAS_NO_POINT_ESTIMATE", "EOL", 0, right_point_count, right_point_count == 0, "Right-censored EOL contains a point estimate")
    censored_rul = rul["EOL_Status"].isin(["INTERVAL_CENSORED", "RIGHT_CENSORED"])
    censored_rul_points = int(pd.to_numeric(rul.loc[censored_rul, "RUL_Cycles"], errors="coerce").notna().sum())
    add("CENSORED_RUL_HAS_NO_POINT_LABEL", "RUL", 0, censored_rul_points, censored_rul_points == 0, "Censored EOL generated a point RUL label")

    b6_interval = eol.loc[
        eol["Cell_ID"].eq("B6_Sim_satellite_battery-2")
        & eol["Qref_Definition"].eq("nominal")
    ]
    if len(b6_interval) == 1:
        interval_row = b6_interval.iloc[0]
        interval_ok = (
            interval_row["EOL_Status"] == "INTERVAL_CENSORED"
            and pd.to_numeric(interval_row["EOL_Lower_Bound_Aging_Cycle"], errors="coerce") == 1008
            and pd.to_numeric(interval_row["EOL_Upper_Bound_Aging_Cycle"], errors="coerce") == 1013
            and pd.to_numeric(interval_row["EOL_First_Below_Raw_Cycle"], errors="coerce") == 1226
        )
        interval_observed = (
            f"status={interval_row['EOL_Status']};"
            f"aging={interval_row['EOL_Lower_Bound_Aging_Cycle']}<EOL<="
            f"{interval_row['EOL_Upper_Bound_Aging_Cycle']};"
            f"first_below_raw={interval_row['EOL_First_Below_Raw_Cycle']}"
        )
    else:
        interval_ok = False
        interval_observed = f"rows={len(b6_interval)}"
    add("B6_BATTERY2_NOMINAL_INTERVAL_REGRESSION", "B6_Sim_satellite_battery-2", "INTERVAL_CENSORED;1008<EOL<=1013;first_below_raw=1226", interval_observed, interval_ok, "Critical interval-censoring regression failed")

    b1_cycle_lookup = cycle_qc.set_index(["Cell_ID", "Raw_Cycle_Index"])
    b1_390_ok = b1_cycle_lookup.loc[("2C_battery-1", 390), "Final_Include"] == "NO"
    b3_393_ok = b1_cycle_lookup.loc[("2C_battery-3", 393), "Final_Include"] == "NO"
    add("BATCH1_BATTERY1_C390_EXCLUDED", "Batch-1 frozen", "NO", b1_cycle_lookup.loc[("2C_battery-1", 390), "Final_Include"], b1_390_ok, "Frozen manual exclusion changed")
    add("BATCH1_BATTERY3_C393_EXCLUDED", "Batch-1 frozen", "NO", b1_cycle_lookup.loc[("2C_battery-3", 393), "Final_Include"], b3_393_ok, "Frozen manual exclusion changed")
    for cell in ("2C_battery-1", "2C_battery-3"):
        rows = eol.loc[eol["Cell_ID"].eq(cell) & eol["Qref_Definition"].eq("nominal")]
        observed = rows.iloc[0]["EOL_Status"] if len(rows) == 1 else f"rows={len(rows)}"
        add(f"BATCH1_{cell.upper()}_NOMINAL_RIGHT_CENSORED", "Batch-1 frozen", "RIGHT_CENSORED", observed, len(rows) == 1 and observed == "RIGHT_CENSORED", "Frozen Batch-1 nominal EOL changed")

    frozen_actual = cycle_qc.loc[cycle_qc["Batch_ID"].eq(1)].sort_values(
        ["Cell_ID", "Raw_Cycle_Index"], kind="stable"
    )
    frozen_expected = frozen_cycle.sort_values(
        ["Cell_ID", "Raw_Cycle_Index"], kind="stable"
    )
    frozen_equal = (
        len(frozen_actual) == len(frozen_expected)
        and np.array_equal(
            pd.to_numeric(frozen_actual["Summary_Discharge_Capacity_Ah"], errors="coerce").to_numpy(),
            pd.to_numeric(frozen_expected["Summary_Discharge_Capacity_Ah"], errors="coerce").to_numpy(),
            equal_nan=True,
        )
        and frozen_actual["Auto_QC_Flag"].astype(str).tolist()
        == frozen_expected["Auto_QC_Flag"].astype(str).tolist()
        and frozen_actual["Final_Include"].astype(str).tolist()
        == frozen_expected["Final_Include"].astype(str).tolist()
    )
    add("BATCH1_FROZEN_CYCLE_RESULTS_UNCHANGED", "Batch-1 frozen", f"{len(frozen_expected)} rows with identical capacity/QC/include", f"{len(frozen_actual)} rows", frozen_equal, "Frozen Batch-1 cycle results changed")

    info_notes = known_notes.loc[
        known_notes["Note_Source"].eq("VALIDATED_PROTOCOL_INFORMATION")
    ]
    info_keys = {
        (str(item.Cell_ID), int(item.Raw_Cycle_Index))
        for item in info_notes.itertuples(index=False)
    }
    qc_keys = {
        (str(item.Cell_ID), int(item.Raw_Cycle_Index))
        for item in cycle_qc.itertuples(index=False)
    }
    known_info_ok = (
        len(info_notes) == 94
        and info_notes["Severity"].eq("INFO").all()
        and info_notes["Requires_Human_Review"].eq("NO").all()
        and info_keys.issubset(qc_keys)
    )
    add("KNOWN_INFO_ANOMALIES_RETAINED", "validated protocol information", "94 INFO rows retained; no review", f"rows={len(info_notes)};missing_keys={len(info_keys - qc_keys)}", known_info_ok, "Known INFO anomaly was deleted or escalated")

    available_qrefs = {
        "nominal": "Qref_nominal_Ah",
        "initial_test": "Qref_initial_test_Ah",
        "first_comparable": "Qref_first_comparable_Ah",
        "early_median5_comparable": "Qref_early_median5_comparable_Ah",
    }
    expected_soh_rows = 0
    primary_counts = primary.groupby("Cell_ID").size().to_dict()
    for item in qref.itertuples(index=False):
        values = item._asdict()
        available = sum(
            np.isfinite(pd.to_numeric(values[column], errors="coerce"))
            for column in available_qrefs.values()
        )
        expected_soh_rows += int(primary_counts.get(item.Cell_ID, 0)) * available
    add("SOH_USES_OBSERVED_PRIMARY_ONLY", "SOH", expected_soh_rows, len(soh), len(soh) == expected_soh_rows and soh["Final_Include"].eq("YES").all(), "SOH row count does not match primary observations × available Qrefs")
    soh_recomputed = pd.to_numeric(soh["Capacity_Ah"], errors="coerce") / pd.to_numeric(soh["Qref_Ah"], errors="coerce")
    soh_diff = np.abs(soh_recomputed - pd.to_numeric(soh["SOH_Fraction"], errors="coerce"))
    # Frozen Batch-1 CSV values were serialized to 12 significant digits, so
    # the independent Q/Qref recomputation can differ by about 5e-12 solely
    # from decimal serialization.  The 1e-10 audit tolerance remains many
    # orders below the source capacity precision and does not alter any label.
    add("SOH_FORMULA_AND_NO_CLIPPING", "SOH", "SOH=Capacity/Qref without clipping", f"max_abs_error={float(soh_diff.max()):.3g};tolerance=1e-10;max_soh={float(pd.to_numeric(soh['SOH_Fraction']).max()):.6g}", bool((soh_diff.fillna(0) <= 1e-10).all()), "SOH formula mismatch or clipping detected")

    nonexact_pair_delta = pairwise.loc[
        ~(
            pairwise["EOL_Status_A"].eq("EXACT_OBSERVED")
            & pairwise["EOL_Status_B"].eq("EXACT_OBSERVED")
        ),
        "Delta_EOL_Cycles",
    ]
    pairwise_ok = len(pairwise) == 55 * 6 and pd.to_numeric(
        nonexact_pair_delta, errors="coerce"
    ).isna().all()
    add("PAIRWISE_ALL_DEFINITIONS_AND_NO_CENSORED_POINT_DELTA", "EOL pairwise", 330, len(pairwise), pairwise_ok, "Pairwise matrix incomplete or censored comparison has point delta")

    return pd.DataFrame(records)


def _figures(output_dir: Path, primary: pd.DataFrame, qref: pd.DataFrame, eol: pd.DataFrame) -> None:
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(qref))
    for definition, column in (
        ("nominal", "Qref_nominal_Ah"),
        ("initial_test", "Qref_initial_test_Ah"),
        ("first_comparable", "Qref_first_comparable_Ah"),
        ("early_median5_comparable", "Qref_early_median5_comparable_Ah"),
    ):
        ax.scatter(positions, pd.to_numeric(qref[column], errors="coerce"), s=12, label=definition)
    ax.set_xlabel("Canonical cell (sorted by Batch_ID, Cell_ID)")
    ax.set_ylabel("Qref (Ah)")
    ax.set_title("XJTU 55-cell reference-capacity definitions")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "xjtu_55_qref_comparison.png", dpi=180)
    plt.close(fig)

    status_counts = eol.groupby(["Batch_ID", "EOL_Status"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    status_counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Cell × threshold-definition rows")
    ax.set_title("EOL observation and censoring status")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "xjtu_55_eol_censoring_overview.png", dpi=180)
    plt.close(fig)

    for batch_id, group in primary.groupby("Batch_ID"):
        fig, ax = plt.subplots(figsize=(10, 5))
        for cell, cell_rows in group.groupby("Cell_ID"):
            ax.plot(
                cell_rows["Cumulative_Aging_Operation_Count"],
                cell_rows["Capacity_Ah"],
                marker=".",
                linewidth=0.8,
                markersize=2,
                label=cell,
            )
        ax.axhline(1.6, color="black", linestyle="--", linewidth=0.8, label="1.600 Ah")
        ax.set_xlabel("Cumulative aging-operation count")
        ax.set_ylabel("Observed comparable capacity (Ah)")
        ax.set_title(f"Batch-{int(batch_id)} primary observed capacity")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(figures / f"xjtu_batch{int(batch_id)}_primary_capacity.png", dpi=180)
        plt.close(fig)


def run_full_analysis(validation_dir: Path, output_dir: Path) -> Path:
    """Run the approved 55-cell analysis. This function is not called by scan."""
    validation_dir = validation_dir.resolve()
    receipt_path = validation_dir / "validation_receipt.json"
    receipt = _load_json(receipt_path)
    if (
        receipt.get("validation_status")
        != "VALIDATION_PASSED_AWAITING_HUMAN_RUN_APPROVAL"
    ):
        raise RuntimeError("VALIDATION_RECEIPT_NOT_PASSED")
    run_approval_path = validation_dir / "run_approval.json"
    if not run_approval_path.is_file():
        raise RuntimeError(
            "FULL_RUN_APPROVAL_MISSING: validation passed, but run remains blocked"
        )
    run_approval = _load_json(run_approval_path)
    if run_approval.get("status") != "APPROVED":
        raise RuntimeError("FULL_RUN_APPROVAL_STATUS_NOT_APPROVED")
    if not run_approval.get("approved_by"):
        raise RuntimeError("FULL_RUN_APPROVER_MISSING")
    if run_approval.get("validation_receipt_sha256") != _sha256(receipt_path):
        raise RuntimeError("FULL_RUN_APPROVAL_VALIDATION_RECEIPT_HASH_MISMATCH")
    validated_path = validation_dir / "xjtu_protocol_inventory_validated.csv"
    if _sha256(validated_path) != receipt.get("validated_protocol_inventory_sha256"):
        raise RuntimeError("VALIDATED_PROTOCOL_INVENTORY_HASH_MISMATCH")
    if (
        run_approval.get("validated_protocol_inventory_sha256")
        != receipt.get("validated_protocol_inventory_sha256")
    ):
        raise RuntimeError("FULL_RUN_APPROVAL_INVENTORY_HASH_MISMATCH")
    for name, expected in receipt.get(
        "validation_output_hashes_excluding_receipt", {}
    ).items():
        if _sha256(validation_dir / name) != expected:
            raise RuntimeError(f"VALIDATION_ARTIFACT_CHANGED_AFTER_REVIEW:{name}")
    scan_dir = Path(str(receipt["scan_directory"]))
    current_scan_hashes = output_hash_manifest(scan_dir)
    if current_scan_hashes != receipt.get("locked_scan_hashes"):
        raise RuntimeError("SCAN_ARTIFACT_CHANGED_AFTER_VALIDATION")
    frozen_dir = Path(str(receipt["frozen_batch1_directory"]))
    for name, expected in receipt.get("frozen_batch1_hashes", {}).items():
        if _sha256(frozen_dir / name) != expected:
            raise RuntimeError(f"FROZEN_BATCH1_ARTIFACT_CHANGED:{name}")
    manifest = pd.read_csv(scan_dir / "xjtu_file_manifest.csv", keep_default_na=False)
    for item in manifest.loc[
        manifest["Batch_ID"].ne(1)
        & manifest["File_Role"].eq("CANONICAL")
        & manifest["Included_In_Analysis"].eq("YES")
    ].itertuples(index=False):
        path = Path(str(item.Source_Directory)) / str(item.Source_File)
        if _sha256(path) != str(item.SHA256):
            raise RuntimeError(f"RAW_INPUT_HASH_CHANGED:{path}")

    configs = {
        batch_id: _load_json(
            validation_dir / f"batch{batch_id}_config_final_candidate.json"
        )
        for batch_id in range(2, 7)
    }
    validated = pd.read_csv(
        validated_path, keep_default_na=False, low_memory=False
    )
    new_qc = _new_cycle_qc(validated, configs)
    frozen_qc = _frozen_cycle_qc(frozen_dir)
    cycle_qc = pd.concat([frozen_qc, new_qc], ignore_index=True, sort=False)
    cycle_qc = cycle_qc.sort_values(["Batch_ID", "Cell_ID", "Raw_Cycle_Index"]).reset_index(drop=True)

    new_primary = new_qc.loc[
        new_qc["Primary_Capacity_Eligible"].eq("YES")
        & new_qc["Final_Include"].eq("YES")
    ].copy()
    frozen_primary = frozen_qc.loc[
        frozen_qc["Primary_Capacity_Eligible"].eq("YES")
        & frozen_qc["Final_Include"].eq("YES")
    ].copy()
    primary = pd.concat([frozen_primary, new_primary], ignore_index=True, sort=False)
    primary = primary.sort_values(
        ["Batch_ID", "Cell_ID", "Cumulative_Aging_Operation_Count", "Raw_Cycle_Index"]
    ).reset_index(drop=True)

    new_qref_rows = [
        _qref_row_from_observations(cell, int(rows["Batch_ID"].iloc[0]), rows)
        for cell, rows in new_qc.groupby("Cell_ID", sort=True)
    ]
    new_qref = pd.DataFrame(new_qref_rows)
    qref = pd.concat([_frozen_qref(frozen_dir), new_qref], ignore_index=True, sort=False)
    qref = _finalize_qref_summary(
        qref.sort_values(["Batch_ID", "Cell_ID"]).reset_index(drop=True)
    )

    new_soh = _new_soh(new_primary, new_qref)
    soh = pd.concat([_frozen_soh(frozen_dir), new_soh], ignore_index=True, sort=False)
    new_eol, new_eol_lookup = _new_eol(new_primary, new_qref)
    frozen_eol = _frozen_eol(frozen_dir)
    eol = pd.concat([frozen_eol, new_eol], ignore_index=True, sort=False)
    eol = eol.sort_values(["Batch_ID", "Cell_ID", "Qref_Definition"]).reset_index(drop=True)
    pairwise = _pairwise(eol, qref)
    rul = pd.concat(
        [_frozen_rul(frozen_dir), _rul(new_primary, new_eol_lookup)],
        ignore_index=True,
        sort=False,
    )

    censoring = (
        eol.groupby(["Batch_ID", "Qref_Definition", "EOL_Status"])
        .size()
        .rename("Cell_Count")
        .reset_index()
    )
    known_notes = _known_data_quality_notes(
        validation_dir, frozen_dir, cycle_qc
    )

    batch_summary = (
        cycle_qc.groupby("Batch_ID")
        .agg(
            Canonical_Cell_Count=("Cell_ID", "nunique"),
            Total_Raw_Cycles=("Raw_Cycle_Index", "size"),
            PASS_Count=("Auto_QC_Flag", lambda values: int((values == "PASS").sum())),
            Invalid_Candidate_Count=("Auto_QC_Flag", lambda values: int((values == "INVALID_CANDIDATE").sum())),
            Check_Count=("Auto_QC_Flag", lambda values: int((values == "CHECK").sum())),
        )
        .reset_index()
    )
    primary_by_batch = (
        primary.groupby("Batch_ID")
        .size()
        .rename("Primary_Capacity_Observation_Count")
        .reset_index()
    )
    info_by_batch = (
        known_notes.loc[known_notes["Severity"].eq("INFO")]
        .groupby("Batch_ID")
        .size()
        .rename("Known_INFO_Count")
        .reset_index()
    )
    batch_summary = (
        batch_summary.merge(primary_by_batch, on="Batch_ID", how="left")
        .merge(info_by_batch, on="Batch_ID", how="left")
        .fillna(
            {
                "Primary_Capacity_Observation_Count": 0,
                "Known_INFO_Count": 0,
            }
        )
    )
    for definition, column in (
        ("nominal", "Qref_nominal_Ah"),
        ("initial_test", "Qref_initial_test_Ah"),
        ("first_comparable", "Qref_first_comparable_Ah"),
        (
            "early_median5_comparable",
            "Qref_early_median5_comparable_Ah",
        ),
    ):
        counts = (
            qref.loc[pd.to_numeric(qref[column], errors="coerce").notna()]
            .groupby("Batch_ID")
            .size()
        )
        batch_summary[f"Qref_{definition}_Available_Cell_Count"] = (
            batch_summary["Batch_ID"].map(counts).fillna(0).astype(int)
        )

    cell_summary = (
        cycle_qc.groupby(["Batch_ID", "Cell_ID"])
        .agg(
            Total_Raw_Cycles=("Raw_Cycle_Index", "size"),
            PASS_Count=("Auto_QC_Flag", lambda values: int((values == "PASS").sum())),
            Invalid_Candidate_Count=("Auto_QC_Flag", lambda values: int((values == "INVALID_CANDIDATE").sum())),
            Check_Count=("Auto_QC_Flag", lambda values: int((values == "CHECK").sum())),
        )
        .reset_index()
    )
    primary_by_cell = (
        primary.groupby(["Batch_ID", "Cell_ID"])
        .size()
        .rename("Primary_Capacity_Observation_Count")
        .reset_index()
    )
    qref_cell = qref[
        [
            "Batch_ID",
            "Cell_ID",
            "Qref_nominal_Ah",
            "Qref_initial_test_Ah",
            "Qref_first_comparable_Ah",
            "Qref_early_median5_comparable_Ah",
        ]
    ]
    eol_cell = (
        eol.loc[
            eol["Qref_Definition"].isin(
                [*QREF_DEFINITIONS, "fixed_absolute_1p6Ah"]
            )
        ]
        .pivot(index=["Batch_ID", "Cell_ID"], columns="Qref_Definition", values="EOL_Status")
        .add_prefix("EOL_Status_")
        .reset_index()
    )
    cell_summary = (
        cell_summary.merge(primary_by_cell, on=["Batch_ID", "Cell_ID"], how="left")
        .merge(qref_cell, on=["Batch_ID", "Cell_ID"], how="left")
        .merge(eol_cell, on=["Batch_ID", "Cell_ID"], how="left")
    )
    cell_summary["Primary_Capacity_Observation_Count"] = (
        cell_summary["Primary_Capacity_Observation_Count"].fillna(0).astype(int)
    )

    run_tests = _run_regression_tests(
        manifest,
        validated,
        frozen_dir,
        cycle_qc,
        primary,
        qref,
        soh,
        eol,
        pairwise,
        rul,
        known_notes,
    )
    failures = run_tests.loc[run_tests["Status"].eq("FAIL")]
    if not failures.empty:
        raise RuntimeError(
            "FULL_RUN_REGRESSION_FAILED:"
            + "|".join(failures["Test_ID"].astype(str).tolist())
        )

    temp_dir = _atomic_directory(output_dir)
    _write_csv(manifest, temp_dir / "xjtu_55_file_manifest.csv")
    _write_csv(cycle_qc, temp_dir / "xjtu_55_cycle_qc.csv")
    _write_csv(primary, temp_dir / "xjtu_55_primary_capacity.csv")
    _write_csv(qref, temp_dir / "xjtu_55_qref_summary.csv")
    _write_csv(soh, temp_dir / "xjtu_55_soh_labels_long.csv")
    _write_csv(eol, temp_dir / "xjtu_55_eol_sensitivity.csv")
    _write_csv(pairwise, temp_dir / "xjtu_55_eol_pairwise.csv")
    _write_csv(rul, temp_dir / "xjtu_55_rul_labels.csv")
    _write_csv(censoring, temp_dir / "xjtu_55_censoring_summary.csv")
    _write_csv(batch_summary, temp_dir / "xjtu_55_batch_summary.csv")
    _write_csv(cell_summary, temp_dir / "xjtu_55_cell_summary.csv")
    _write_csv(known_notes, temp_dir / "xjtu_55_known_data_quality_notes.csv")
    _write_csv(run_tests, temp_dir / "xjtu_55_run_regression_tests.csv")
    _figures(temp_dir, primary, qref, eol)

    qref_availability = {
        definition: int(pd.to_numeric(qref[column], errors="coerce").notna().sum())
        for definition, column in (
            ("nominal", "Qref_nominal_Ah"),
            ("initial_test", "Qref_initial_test_Ah"),
            ("first_comparable", "Qref_first_comparable_Ah"),
            (
                "early_median5_comparable",
                "Qref_early_median5_comparable_Ah",
            ),
        )
    }
    eol_counts = (
        eol.groupby(["Qref_Definition", "EOL_Status"])
        .size()
        .to_dict()
    )
    log_lines = [
        f"Pipeline_Version={PIPELINE_VERSION}",
        f"Python_Version={sys.version.replace(chr(10), ' ')}",
        f"Platform={platform.platform()}",
        f"NumPy_Version={np.__version__}",
        f"pandas_Version={pd.__version__}",
        f"SciPy_Version={scipy.__version__}",
        f"Run_Timestamp_UTC={datetime.now(timezone.utc).isoformat()}",
        f"Validation_Directory={validation_dir}",
        f"Validation_Receipt_SHA256={_sha256(receipt_path)}",
        f"Run_Approval_SHA256={_sha256(run_approval_path)}",
        f"Frozen_Batch1_Directory={frozen_dir}",
        "BATCH1_SOURCE=FROZEN_REFERENCE",
        "BATCH1_RAW_MAT_REPROCESSED=NO",
        "NO_INTERPOLATION=TRUE",
        "NO_SMOOTHING=TRUE",
        "RAW_CYCLE_RENUMBERING=FALSE",
        f"Total_Canonical_Cells={qref['Cell_ID'].nunique()}",
        f"Total_Raw_Cycles={len(cycle_qc)}",
        f"Total_Primary_Capacity_Observations={len(primary)}",
        "Primary_Observations_By_Batch="
        + "|".join(
            f"{int(key)}:{int(value)}"
            for key, value in primary.groupby("Batch_ID").size().items()
        ),
        "Qref_Available_Cell_Counts="
        + "|".join(f"{key}:{value}" for key, value in qref_availability.items()),
        "EOL_Status_Counts="
        + "|".join(
            f"{definition}:{status}:{int(count)}"
            for (definition, status), count in sorted(eol_counts.items())
        ),
        f"Known_INFO_Count={int(known_notes['Severity'].eq('INFO').sum())}",
        f"Run_Regression_Test_Count={len(run_tests)}",
        f"Run_Regression_Test_Failures={int(run_tests['Status'].eq('FAIL').sum())}",
        "MACHINE_LEARNING_TRAINED=NO",
        "RUN_STATUS=COMPLETE",
        "XJTU 55-CELL RUN COMPLETE",
    ]
    (temp_dir / "processing_log.txt").write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )

    output_hashes = {
        str(path.relative_to(temp_dir)).replace("\\", "/"): _sha256(path)
        for path in sorted(temp_dir.rglob("*"))
        if path.is_file()
    }
    run_receipt = {
        "run_status": "COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_receipt_sha256": _sha256(receipt_path),
        "run_approval_sha256": _sha256(run_approval_path),
        "validated_protocol_inventory_sha256": _sha256(validated_path),
        "frozen_batch1_hashes": receipt.get("frozen_batch1_hashes", {}),
        "canonical_raw_inputs_verified_against_manifest": True,
        "output_hashes_excluding_receipt": output_hashes,
        "total_cells": int(qref["Cell_ID"].nunique()),
        "total_raw_cycles": int(len(cycle_qc)),
        "total_primary_capacity_observations": int(len(primary)),
        "regression_test_count": int(len(run_tests)),
        "regression_test_failures": int(run_tests["Status"].eq("FAIL").sum()),
        "machine_learning_trained": False,
    }
    (temp_dir / "run_receipt.json").write_text(
        json.dumps(run_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_dir.rename(output_dir.resolve())
    return output_dir.resolve()
