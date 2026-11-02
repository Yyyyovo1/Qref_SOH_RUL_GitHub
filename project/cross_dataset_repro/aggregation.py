from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR_NAME = "cross_dataset_summary_v1"
COMMON_QREFS = ("nominal", "first_comparable", "early_median5_comparable")
MEASURED_COMMON_QREFS = ("first_comparable", "early_median5_comparable")
XJTU_SECONDARY_QREFS = ("initial_test",)
QREF_ORDER = {
    "nominal": 1,
    "first_comparable": 2,
    "early_median5_comparable": 3,
    "initial_test": 4,
}
STATUS_ORDER = {
    "EXACT_OBSERVED": 1,
    "INTERVAL_CENSORED": 2,
    "RIGHT_CENSORED": 3,
    "QREF_UNAVAILABLE": 4,
}
ALLOWED_TRANSITIONS = (
    "EXACT_TO_EXACT",
    "EXACT_TO_INTERVAL",
    "EXACT_TO_RIGHT",
    "INTERVAL_TO_EXACT",
    "INTERVAL_TO_INTERVAL",
    "INTERVAL_TO_RIGHT",
    "RIGHT_TO_EXACT",
    "RIGHT_TO_INTERVAL",
    "BOTH_RIGHT",
)
FLOAT_TOL = 1e-12
FIXED_CONTROL_TOL_AH = 1e-12


def _is_number(value: Any) -> bool:
    try:
        return not pd.isna(float(value))
    except (TypeError, ValueError):
        return False


def _num(value: Any) -> float:
    if not _is_number(value):
        return math.nan
    return float(value)


def classify_qref_direction(qref: Any, nominal: Any, tol: float = FLOAT_TOL) -> str:
    q = _num(qref)
    n = _num(nominal)
    if math.isnan(q) or math.isnan(n):
        return "QREF_UNAVAILABLE"
    delta = q - n
    if delta > tol:
        return "QREF_ABOVE_NOMINAL"
    if delta < -tol:
        return "QREF_BELOW_NOMINAL"
    return "QREF_EQUAL_NOMINAL"


def _status_class(status: Any) -> str | None:
    text = str(status)
    if text == "EXACT_OBSERVED":
        return "EXACT"
    if text == "INTERVAL_CENSORED":
        return "INTERVAL"
    if text == "RIGHT_CENSORED":
        return "RIGHT"
    return None


def classify_transition(status_a: Any, status_b: Any) -> str:
    a = _status_class(status_a)
    b = _status_class(status_b)
    if a is None or b is None:
        return ""
    if a == "RIGHT" and b == "RIGHT":
        return "BOTH_RIGHT"
    return f"{a}_TO_{b}"


def _event_bounds(
    status: Any,
    exact: Any,
    lower: Any,
    upper: Any,
    last: Any,
) -> tuple[float, float] | None:
    cls = _status_class(status)
    if cls == "EXACT":
        value = _num(exact)
        return None if math.isnan(value) else (value, value)
    if cls == "INTERVAL":
        lo = _num(lower)
        hi = _num(upper)
        return None if math.isnan(lo) or math.isnan(hi) else (lo, hi)
    if cls == "RIGHT":
        lo = _num(lower)
        if math.isnan(lo):
            lo = _num(last)
        return None if math.isnan(lo) else (lo, math.inf)
    return None


def ordering_consistency(
    *,
    direction: str,
    status_a: Any,
    exact_a: Any,
    lower_a: Any,
    upper_a: Any,
    last_a: Any,
    status_b: Any,
    exact_b: Any,
    lower_b: Any,
    upper_b: Any,
    last_b: Any,
) -> tuple[str, str]:
    bounds_a = _event_bounds(status_a, exact_a, lower_a, upper_a, last_a)
    bounds_b = _event_bounds(status_b, exact_b, lower_b, upper_b, last_b)
    if direction == "QREF_UNAVAILABLE" or bounds_a is None or bounds_b is None:
        return "NOT_ASSESSABLE", "Qref or EOL bounds unavailable."

    a_lo, a_hi = bounds_a
    b_lo, b_hi = bounds_b
    if direction == "QREF_EQUAL_NOMINAL":
        same = (
            str(status_a) == str(status_b)
            and np.isclose(a_lo, b_lo, atol=FLOAT_TOL, rtol=0)
            and (
                (math.isinf(a_hi) and math.isinf(b_hi))
                or np.isclose(a_hi, b_hi, atol=FLOAT_TOL, rtol=0)
            )
        )
        if same:
            return "CONSISTENT", "Equal Qref gives the same threshold and frozen EOL result."
        return "INCONSISTENT", "Equal Qref does not reproduce the nominal EOL result."

    if direction == "QREF_ABOVE_NOMINAL":
        if b_hi <= a_lo + FLOAT_TOL:
            return "CONSISTENT", "Higher measured threshold is observed no later than nominal."
        if not math.isinf(a_hi) and b_lo > a_hi + FLOAT_TOL:
            return "INCONSISTENT", "Higher measured threshold is bounded later than nominal."
        return "INDETERMINATE_BOUNDS_OVERLAP", "Available censoring bounds do not contradict the expected earlier-or-equal ordering."

    if direction == "QREF_BELOW_NOMINAL":
        if b_lo + FLOAT_TOL >= a_hi:
            return "CONSISTENT", "Lower measured threshold is observed no earlier than nominal."
        if not math.isinf(b_hi) and b_hi < a_lo - FLOAT_TOL:
            return "INCONSISTENT", "Lower measured threshold is bounded earlier than nominal."
        return "INDETERMINATE_BOUNDS_OVERLAP", "Available censoring bounds do not contradict the expected later-or-equal ordering."

    return "NOT_ASSESSABLE", "Unknown Qref direction."


def compute_effect_bounds(
    *,
    status_a: Any,
    exact_a: Any,
    lower_a: Any,
    upper_a: Any,
    last_a: Any,
    status_b: Any,
    exact_b: Any,
    lower_b: Any,
    upper_b: Any,
    last_b: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "Delta_EOL_Exact_Cycles": math.nan,
        "Delta_EOL_Lower_Bound_Cycles": math.nan,
        "Delta_EOL_Upper_Bound_Cycles": math.nan,
        "Minimum_Observed_Earlier_By_Cycles": math.nan,
        "Effect_Estimate_Type": "UNAVAILABLE",
        "Effect_Bound_Interpretation": "EOL comparison unavailable.",
    }
    bounds_a = _event_bounds(status_a, exact_a, lower_a, upper_a, last_a)
    bounds_b = _event_bounds(status_b, exact_b, lower_b, upper_b, last_b)
    if bounds_a is None or bounds_b is None:
        return result

    a_lo, a_hi = bounds_a
    b_lo, b_hi = bounds_b
    a_cls = _status_class(status_a)
    b_cls = _status_class(status_b)
    if a_cls == "EXACT" and b_cls == "EXACT":
        result["Delta_EOL_Exact_Cycles"] = b_lo - a_lo
        result["Effect_Estimate_Type"] = "EXACT_DELTA"
        result["Effect_Bound_Interpretation"] = "Exact delta uses measured-definition EOL minus nominal EOL."
        return result

    if not math.isinf(a_hi) and not math.isinf(b_hi):
        result["Delta_EOL_Lower_Bound_Cycles"] = b_lo - a_hi
        result["Delta_EOL_Upper_Bound_Cycles"] = b_hi - a_lo
        result["Effect_Estimate_Type"] = "FINITE_INTERVAL_BOUNDS"
        result["Effect_Bound_Interpretation"] = "Only censoring-compatible bounds on measured EOL minus nominal EOL are reported."
        return result

    if math.isinf(a_hi) and not math.isinf(b_hi):
        result["Delta_EOL_Upper_Bound_Cycles"] = b_hi - a_lo
        if b_hi <= a_lo:
            result["Minimum_Observed_Earlier_By_Cycles"] = a_lo - b_hi
            result["Effect_Estimate_Type"] = "LOWER_BOUND_ON_EARLIER_EOL_ONLY"
            result["Effect_Bound_Interpretation"] = "Nominal EOL is right-censored; the positive value is only a lower bound on how much earlier measured-Qref EOL occurred."
        else:
            result["Effect_Estimate_Type"] = "ONE_SIDED_UPPER_BOUND_ON_DELTA"
            result["Effect_Bound_Interpretation"] = "Nominal EOL is right-censored; only an upper bound on measured EOL minus nominal EOL is available."
        return result

    if not math.isinf(a_hi) and math.isinf(b_hi):
        result["Delta_EOL_Lower_Bound_Cycles"] = b_lo - a_hi
        result["Effect_Estimate_Type"] = "ONE_SIDED_LOWER_BOUND_ON_DELTA"
        result["Effect_Bound_Interpretation"] = "Measured-definition EOL is right-censored; only a lower bound on measured EOL minus nominal EOL is available."
        return result

    result["Effect_Estimate_Type"] = "BOTH_RIGHT_CENSORED_NO_POINT_EFFECT"
    result["Effect_Bound_Interpretation"] = "Both EOL values are right-censored; no point shift is reported."
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        count = sum(1 for _ in reader)
    return max(count - 1, 0)


def _snapshot(directory: Path) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".txt"}):
        rel = path.relative_to(directory).as_posix()
        snapshot[rel] = {
            "Size_Bytes": path.stat().st_size,
            "SHA256": _sha256(path),
            "Data_Row_Count": _csv_row_count(path) if path.suffix.lower() == ".csv" else None,
        }
    return snapshot


def _snapshot_digest(snapshot: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _rul_cell_status(eol_status: str) -> tuple[str, str]:
    if eol_status == "EXACT_OBSERVED":
        return "POINT_RUL_AVAILABLE", "YES"
    if eol_status == "INTERVAL_CENSORED":
        return "INTERVAL_BOUNDED_RUL_ONLY", "NO"
    if eol_status == "RIGHT_CENSORED":
        return "CENSORED_EOL_NO_POINT_RUL", "NO"
    return "QREF_UNAVAILABLE", "NO"


def _build_xjtu_cell_level(x_qref: pd.DataFrame, x_eol: pd.DataFrame) -> pd.DataFrame:
    qref_columns = {
        "nominal": ("Qref_nominal_Ah", "Qref_nominal_Status"),
        "initial_test": ("Qref_initial_test_Ah", "Qref_initial_test_Status"),
        "first_comparable": ("Qref_first_comparable_Ah", "Qref_first_comparable_Status"),
        "early_median5_comparable": ("Qref_early_median5_comparable_Ah", "Qref_early_median5_comparable_Status"),
    }
    records: list[dict[str, Any]] = []
    eol_index = x_eol.set_index(["Batch_ID", "Cell_ID", "Qref_Definition"], drop=False)
    for row in x_qref.itertuples(index=False):
        batch = str(int(float(row.Batch_ID)))
        cell = str(row.Cell_ID)
        nominal = _num(getattr(row, "Qref_nominal_Ah"))
        for definition, (value_col, status_col) in qref_columns.items():
            source_key = (float(row.Batch_ID), cell, definition)
            if source_key not in eol_index.index:
                source_key = (int(float(row.Batch_ID)), cell, definition)
            eol_row = eol_index.loc[source_key]
            if isinstance(eol_row, pd.DataFrame):
                raise ValueError(f"Duplicate XJTU EOL key: {source_key}")
            qref = _num(getattr(row, value_col))
            qref_status = str(getattr(row, status_col))
            threshold = _num(eol_row["EOThreshold_Ah"])
            exact = _num(eol_row["EOL_Exact_Aging_Cycle"])
            lower = _num(eol_row["EOL_Lower_Bound_Aging_Cycle"])
            upper = _num(eol_row["EOL_Upper_Bound_Aging_Cycle"])
            last = _num(eol_row["Last_Valid_Aging_Cycle"])
            eol_status = str(eol_row["EOL_Status"])
            if eol_status == "RIGHT_CENSORED" and math.isnan(lower):
                lower = last
            delta = qref - nominal if not math.isnan(qref) else math.nan
            pct = 100.0 * delta / nominal if not math.isnan(delta) and nominal != 0 else math.nan
            rul_status, point_available = _rul_cell_status(eol_status)
            records.append(
                {
                    "Dataset": "XJTU",
                    "Batch_ID": batch,
                    "Cell_ID": cell,
                    "Qref_Definition": definition,
                    "Qref_Ah": qref,
                    "Qref_Status": qref_status,
                    "Delta_Qref_vs_Nominal_Ah": delta,
                    "Delta_Qref_vs_Nominal_Pct": pct,
                    "EOL_Threshold_Ah": threshold,
                    "Delta_Threshold_vs_Nominal_Ah": threshold - 0.8 * nominal if not math.isnan(threshold) else math.nan,
                    "EOL_Status": eol_status,
                    "EOL_Exact_Cycle": exact,
                    "EOL_Lower_Bound": lower,
                    "EOL_Upper_Bound": upper,
                    "Last_Observed_Cycle": last,
                    "RUL_Status": rul_status,
                    "Protocol_Context": f"XJTU Batch-{batch}; protocol-comparable capacity diagnostics and inclusion follow frozen Final V2.",
                    "Dataset_Stopping_Rule_Context": "Frozen multi-batch observation windows; exact, interval-censored, and right-censored EOL states are retained without endpoint substitution.",
                    "Comparison_Scope": "XJTU_ONLY_SECONDARY" if definition == "initial_test" else "PRIMARY_COMMON",
                    "Cycle_Index_Basis": "XJTU_CUMULATIVE_AGING_OPERATION_COUNT",
                    "Qref_Direction_vs_Nominal": classify_qref_direction(qref, nominal),
                    "RUL_Point_Label_Available": point_available,
                    "Source_Qref_Definition": definition,
                }
            )
    return pd.DataFrame.from_records(records)


def _build_hust_cell_level(h_qref: pd.DataFrame, h_eol: pd.DataFrame) -> pd.DataFrame:
    qref_columns = {
        "nominal": ("Qref_Nominal_Ah", "Qref_Nominal_Status"),
        "first_comparable": ("Qref_First_Comparable_Ah", "Qref_First_Comparable_Status"),
        "early_median5_comparable": ("Qref_Early_Median5_Comparable_Ah", "Qref_Early_Median5_Comparable_Status"),
    }
    eol_index = h_eol.set_index(["Cell_ID", "Qref_Definition"], drop=False)
    records: list[dict[str, Any]] = []
    for row in h_qref.itertuples(index=False):
        cell = str(row.Cell_ID)
        nominal = _num(getattr(row, "Qref_Nominal_Ah"))
        for definition, (value_col, status_col) in qref_columns.items():
            eol_row = eol_index.loc[(cell, definition)]
            if isinstance(eol_row, pd.DataFrame):
                raise ValueError(f"Duplicate HUST EOL key: {(cell, definition)}")
            qref = _num(getattr(row, value_col))
            qref_status = str(getattr(row, status_col))
            threshold = _num(eol_row["EOThreshold_Ah"])
            eol_status = str(eol_row["EOL_Status"])
            exact = _num(eol_row["EOL_Raw_Cycle"])
            last = _num(eol_row["Last_Valid_Raw_Cycle"])
            lower = last if eol_status == "RIGHT_CENSORED" else math.nan
            delta = qref - nominal if not math.isnan(qref) else math.nan
            pct = 100.0 * delta / nominal if not math.isnan(delta) and nominal != 0 else math.nan
            rul_status, point_available = _rul_cell_status(eol_status)
            records.append(
                {
                    "Dataset": "HUST",
                    "Batch_ID": "",
                    "Cell_ID": cell,
                    "Qref_Definition": definition,
                    "Qref_Ah": qref,
                    "Qref_Status": qref_status,
                    "Delta_Qref_vs_Nominal_Ah": delta,
                    "Delta_Qref_vs_Nominal_Pct": pct,
                    "EOL_Threshold_Ah": threshold,
                    "Delta_Threshold_vs_Nominal_Ah": threshold - 0.8 * nominal if not math.isnan(threshold) else math.nan,
                    "EOL_Status": eol_status,
                    "EOL_Exact_Cycle": exact,
                    "EOL_Lower_Bound": lower,
                    "EOL_Upper_Bound": math.nan,
                    "Last_Observed_Cycle": last,
                    "RUL_Status": rul_status,
                    "Protocol_Context": "HUST public raw-observed protocol; comparable measured Qrefs use frozen Final V1 capacity rules. Paper terminal 1C versus public-raw approximately 2C discrepancy remains explicit.",
                    "Dataset_Stopping_Rule_Context": "Native series terminate near nominal 80% capacity but remain above the strict 0.880 Ah threshold; native endpoint and native RUL are audit-only.",
                    "Comparison_Scope": "PRIMARY_COMMON",
                    "Cycle_Index_Basis": "HUST_RAW_CYCLE_INDEX",
                    "Qref_Direction_vs_Nominal": classify_qref_direction(qref, nominal),
                    "RUL_Point_Label_Available": point_available,
                    "Source_Qref_Definition": definition,
                }
            )
    return pd.DataFrame.from_records(records)


def _build_effect_bounds(cell_level: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    key_cols = ["Dataset", "Cell_ID"]
    nominal = cell_level[cell_level["Qref_Definition"] == "nominal"].set_index(key_cols)
    measured = cell_level[cell_level["Qref_Definition"].isin((*MEASURED_COMMON_QREFS, *XJTU_SECONDARY_QREFS))]
    for row in measured.itertuples(index=False):
        a = nominal.loc[(row.Dataset, row.Cell_ID)]
        if isinstance(a, pd.DataFrame):
            raise ValueError(f"Duplicate nominal row for {(row.Dataset, row.Cell_ID)}")
        transition = classify_transition(a["EOL_Status"], row.EOL_Status)
        consistency, consistency_reason = ordering_consistency(
            direction=row.Qref_Direction_vs_Nominal,
            status_a=a["EOL_Status"],
            exact_a=a["EOL_Exact_Cycle"],
            lower_a=a["EOL_Lower_Bound"],
            upper_a=a["EOL_Upper_Bound"],
            last_a=a["Last_Observed_Cycle"],
            status_b=row.EOL_Status,
            exact_b=row.EOL_Exact_Cycle,
            lower_b=row.EOL_Lower_Bound,
            upper_b=row.EOL_Upper_Bound,
            last_b=row.Last_Observed_Cycle,
        )
        bounds = compute_effect_bounds(
            status_a=a["EOL_Status"],
            exact_a=a["EOL_Exact_Cycle"],
            lower_a=a["EOL_Lower_Bound"],
            upper_a=a["EOL_Upper_Bound"],
            last_a=a["Last_Observed_Cycle"],
            status_b=row.EOL_Status,
            exact_b=row.EOL_Exact_Cycle,
            lower_b=row.EOL_Lower_Bound,
            upper_b=row.EOL_Upper_Bound,
            last_b=row.Last_Observed_Cycle,
        )
        record = {
            "Dataset": row.Dataset,
            "Batch_ID": row.Batch_ID,
            "Cell_ID": row.Cell_ID,
            "Definition_A": "nominal",
            "Definition_B": row.Qref_Definition,
            "Comparison_Scope": row.Comparison_Scope,
            "Cycle_Index_Basis": row.Cycle_Index_Basis,
            "Qref_A_Ah": a["Qref_Ah"],
            "Qref_B_Ah": row.Qref_Ah,
            "Delta_Qref_B_vs_A_Ah": row.Delta_Qref_vs_Nominal_Ah,
            "Delta_Qref_B_vs_A_Pct": row.Delta_Qref_vs_Nominal_Pct,
            "Qref_Direction_vs_Nominal": row.Qref_Direction_vs_Nominal,
            "Threshold_A_Ah": a["EOL_Threshold_Ah"],
            "Threshold_B_Ah": row.EOL_Threshold_Ah,
            "Delta_Threshold_B_vs_A_Ah": row.Delta_Threshold_vs_Nominal_Ah,
            "EOL_Status_A": a["EOL_Status"],
            "EOL_Status_B": row.EOL_Status,
            "Transition_Category": transition,
            "EOL_A_Exact_Cycle": a["EOL_Exact_Cycle"],
            "EOL_A_Lower_Bound": a["EOL_Lower_Bound"],
            "EOL_A_Upper_Bound": a["EOL_Upper_Bound"],
            "EOL_B_Exact_Cycle": row.EOL_Exact_Cycle,
            "EOL_B_Lower_Bound": row.EOL_Lower_Bound,
            "EOL_B_Upper_Bound": row.EOL_Upper_Bound,
            "Threshold_Ordering_Consistency": consistency,
            "Threshold_Ordering_Consistency_Reason": consistency_reason,
        }
        record.update(bounds)
        records.append(record)
    result = pd.DataFrame.from_records(records)
    return result.sort_values(
        ["Dataset", "Batch_ID", "Cell_ID", "Definition_B"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _source_rul_summary(path: Path, dataset: str, definitions: set[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = ["Qref_Definition", "EOL_Status", "RUL_Cycles", "RUL_Status"]
    if "RUL_Point_Label_Available" in header:
        wanted.append("RUL_Point_Label_Available")
    if "RUL_Lower_Bound" in header:
        wanted.append("RUL_Lower_Bound")
    if "RUL_Upper_Bound" in header:
        wanted.append("RUL_Upper_Bound")
    counters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for chunk in pd.read_csv(path, usecols=wanted, chunksize=200_000, low_memory=False):
        chunk = chunk[chunk["Qref_Definition"].isin(definitions)].copy()
        if chunk.empty:
            continue
        rul_numeric = pd.to_numeric(chunk["RUL_Cycles"], errors="coerce").notna()
        lower_numeric = pd.to_numeric(chunk.get("RUL_Lower_Bound", pd.Series(index=chunk.index, dtype=float)), errors="coerce").notna()
        upper_numeric = pd.to_numeric(chunk.get("RUL_Upper_Bound", pd.Series(index=chunk.index, dtype=float)), errors="coerce").notna()
        for definition, group_idx in chunk.groupby("Qref_Definition", sort=False).groups.items():
            idx = list(group_idx)
            statuses = chunk.loc[idx, "EOL_Status"]
            rul_statuses = chunk.loc[idx, "RUL_Status"]
            c = counters[str(definition)]
            c["Source_RUL_Row_Count"] += len(idx)
            c["Source_Point_RUL_Row_Count"] += int(rul_numeric.loc[idx].sum())
            c["Source_Censored_No_Point_RUL_Row_Count"] += int(((statuses == "RIGHT_CENSORED") & ~rul_numeric.loc[idx]).sum())
            c["Source_Interval_Bound_RUL_Row_Count"] += int(((statuses == "INTERVAL_CENSORED") & (lower_numeric.loc[idx] | upper_numeric.loc[idx])).sum())
            c["Source_Qref_Unavailable_Row_Count"] += int((statuses == "QREF_UNAVAILABLE").sum())
            c["Source_Post_EOL_Excluded_Row_Count"] += int((rul_statuses == "POST_EOL_EXCLUDED").sum())
            c["Source_Censored_Numeric_RUL_Violation_Count"] += int(((statuses != "EXACT_OBSERVED") & rul_numeric.loc[idx]).sum())
    records = []
    for definition in sorted(definitions, key=lambda d: QREF_ORDER.get(d, 99)):
        record = {"Dataset": dataset, "Qref_Definition": definition}
        record.update(counters.get(definition, {}))
        records.append(record)
    return pd.DataFrame.from_records(records).fillna(0)


def _first_vs_median_delta(cell_level: pd.DataFrame) -> pd.DataFrame:
    first = cell_level[cell_level["Qref_Definition"] == "first_comparable"].copy()
    median = cell_level[cell_level["Qref_Definition"] == "early_median5_comparable"].copy()
    merged = first.merge(
        median,
        on=["Dataset", "Batch_ID", "Cell_ID"],
        suffixes=("_First", "_Median5"),
        validate="one_to_one",
    )
    both_exact = (merged["EOL_Status_First"] == "EXACT_OBSERVED") & (merged["EOL_Status_Median5"] == "EXACT_OBSERVED")
    merged["Both_Exact"] = both_exact
    merged["Delta_EOL_Median5_Minus_First_Cycles"] = np.where(
        both_exact,
        pd.to_numeric(merged["EOL_Exact_Cycle_Median5"]) - pd.to_numeric(merged["EOL_Exact_Cycle_First"]),
        np.nan,
    )
    return merged


def _scientific_context(dataset: str, definition: str) -> tuple[str, str, str]:
    if definition == "nominal":
        direction = "Nominal is the within-dataset reference baseline."
    elif dataset == "XJTU":
        direction = "Measured Qref may be below nominal; a lower Qref mathematically lowers the 80% absolute threshold and can delay or censor EOL."
    else:
        direction = "Measured comparable Qref is above nominal for all 77 HUST cells; the higher Qref mathematically raises the 80% threshold and permits earlier observed EOL."
    if dataset == "HUST":
        comparability = "Broader measured-reference sensitivity replication; not an exact nominal-versus-initial-test replication."
    elif definition == "initial_test":
        comparability = "XJTU-only secondary initial-test analysis; no HUST analogue is created."
    else:
        comparability = "Primary definition shared by XJTU Final V2 and HUST Final V1."
    focus = "Interpret magnitude, censoring-state changes, RUL-label availability, and benchmark comparability; threshold direction itself follows mathematically from Qref ordering."
    return direction, comparability, focus


def _build_qref_summary(cell_level: pd.DataFrame, first_median: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (dataset, definition), group in cell_level.groupby(["Dataset", "Qref_Definition"], sort=False):
        available = group[group["Qref_Status"].astype(str).str.startswith("AVAILABLE")]
        direction, comparability, focus = _scientific_context(dataset, definition)
        record: dict[str, Any] = {
            "Dataset": dataset,
            "Qref_Definition": definition,
            "Comparison_Scope": "XJTU_ONLY_SECONDARY" if definition == "initial_test" else "PRIMARY_COMMON",
            "Dataset_Cell_Count": group["Cell_ID"].nunique(),
            "Qref_Available_Count": len(available),
            "Qref_Unavailable_Count": len(group) - len(available),
            "Qref_Ah_Min": available["Qref_Ah"].min(),
            "Qref_Ah_Median": available["Qref_Ah"].median(),
            "Qref_Ah_Max": available["Qref_Ah"].max(),
            "Delta_Qref_vs_Nominal_Ah_Min": available["Delta_Qref_vs_Nominal_Ah"].min(),
            "Delta_Qref_vs_Nominal_Ah_Median": available["Delta_Qref_vs_Nominal_Ah"].median(),
            "Delta_Qref_vs_Nominal_Ah_Max": available["Delta_Qref_vs_Nominal_Ah"].max(),
            "Delta_Qref_vs_Nominal_Pct_Min": available["Delta_Qref_vs_Nominal_Pct"].min(),
            "Delta_Qref_vs_Nominal_Pct_Median": available["Delta_Qref_vs_Nominal_Pct"].median(),
            "Delta_Qref_vs_Nominal_Pct_Max": available["Delta_Qref_vs_Nominal_Pct"].max(),
            "Delta_Threshold_vs_Nominal_Ah_Min": available["Delta_Threshold_vs_Nominal_Ah"].min(),
            "Delta_Threshold_vs_Nominal_Ah_Median": available["Delta_Threshold_vs_Nominal_Ah"].median(),
            "Delta_Threshold_vs_Nominal_Ah_Max": available["Delta_Threshold_vs_Nominal_Ah"].max(),
            "QREF_ABOVE_NOMINAL_Count": int((group["Qref_Direction_vs_Nominal"] == "QREF_ABOVE_NOMINAL").sum()),
            "QREF_EQUAL_NOMINAL_Count": int((group["Qref_Direction_vs_Nominal"] == "QREF_EQUAL_NOMINAL").sum()),
            "QREF_BELOW_NOMINAL_Count": int((group["Qref_Direction_vs_Nominal"] == "QREF_BELOW_NOMINAL").sum()),
            "EXACT_OBSERVED_Count": int((group["EOL_Status"] == "EXACT_OBSERVED").sum()),
            "INTERVAL_CENSORED_Count": int((group["EOL_Status"] == "INTERVAL_CENSORED").sum()),
            "RIGHT_CENSORED_Count": int((group["EOL_Status"] == "RIGHT_CENSORED").sum()),
            "QREF_UNAVAILABLE_EOL_Count": int((group["EOL_Status"] == "QREF_UNAVAILABLE").sum()),
            "RUL_Point_Label_Available_Cell_Count": int((group["RUL_Point_Label_Available"] == "YES").sum()),
            "Scientific_Direction_Context": direction,
            "Cross_Dataset_Comparability_Context": comparability,
            "Primary_Interpretive_Focus": focus,
            "First_vs_Median_Both_Exact_Cell_Count": math.nan,
            "First_vs_Median_Delta_EOL_Cycles_Min": math.nan,
            "First_vs_Median_Delta_EOL_Cycles_Median": math.nan,
            "First_vs_Median_Delta_EOL_Cycles_Max": math.nan,
            "First_vs_Median_Absolute_Delta_EOL_Cycles_Median": math.nan,
        }
        if definition == "early_median5_comparable":
            values = first_median.loc[
                (first_median["Dataset"] == dataset) & first_median["Both_Exact"],
                "Delta_EOL_Median5_Minus_First_Cycles",
            ].dropna()
            record["First_vs_Median_Both_Exact_Cell_Count"] = len(values)
            if len(values):
                record["First_vs_Median_Delta_EOL_Cycles_Min"] = values.min()
                record["First_vs_Median_Delta_EOL_Cycles_Median"] = values.median()
                record["First_vs_Median_Delta_EOL_Cycles_Max"] = values.max()
                record["First_vs_Median_Absolute_Delta_EOL_Cycles_Median"] = values.abs().median()
        records.append(record)
    result = pd.DataFrame.from_records(records)
    result["_order"] = result["Qref_Definition"].map(QREF_ORDER)
    return result.sort_values(["Dataset", "_order"], kind="mergesort").drop(columns="_order").reset_index(drop=True)


def _build_transition_summary(effect_bounds: pd.DataFrame) -> pd.DataFrame:
    primary = effect_bounds[effect_bounds["Definition_B"].isin(MEASURED_COMMON_QREFS)].copy()
    records: list[dict[str, Any]] = []
    for dataset in ("XJTU", "HUST"):
        for definition in MEASURED_COMMON_QREFS:
            subset = primary[(primary["Dataset"] == dataset) & (primary["Definition_B"] == definition)]
            total = len(subset)
            for transition in ALLOWED_TRANSITIONS:
                group = subset[subset["Transition_Category"] == transition]
                records.append(
                    {
                        "Dataset": dataset,
                        "Qref_Definition": definition,
                        "Transition_Category": transition,
                        "Cell_Count": len(group),
                        "Percent_of_Definition_Cells": 100.0 * len(group) / total if total else math.nan,
                        "QREF_ABOVE_NOMINAL_Count": int((group["Qref_Direction_vs_Nominal"] == "QREF_ABOVE_NOMINAL").sum()),
                        "QREF_EQUAL_NOMINAL_Count": int((group["Qref_Direction_vs_Nominal"] == "QREF_EQUAL_NOMINAL").sum()),
                        "QREF_BELOW_NOMINAL_Count": int((group["Qref_Direction_vs_Nominal"] == "QREF_BELOW_NOMINAL").sum()),
                        "Ordering_CONSISTENT_Count": int((group["Threshold_Ordering_Consistency"] == "CONSISTENT").sum()),
                        "Ordering_INDETERMINATE_Count": int((group["Threshold_Ordering_Consistency"] == "INDETERMINATE_BOUNDS_OVERLAP").sum()),
                        "Ordering_INCONSISTENT_Count": int((group["Threshold_Ordering_Consistency"] == "INCONSISTENT").sum()),
                    }
                )
    return pd.DataFrame.from_records(records)


def _build_censoring_summary(cell_level: pd.DataFrame) -> pd.DataFrame:
    statuses = ("EXACT_OBSERVED", "INTERVAL_CENSORED", "RIGHT_CENSORED", "QREF_UNAVAILABLE")
    records: list[dict[str, Any]] = []
    for (dataset, definition), group in cell_level.groupby(["Dataset", "Qref_Definition"], sort=False):
        total = len(group)
        for status in statuses:
            count = int((group["EOL_Status"] == status).sum())
            records.append(
                {
                    "Dataset": dataset,
                    "Qref_Definition": definition,
                    "EOL_Status": status,
                    "Cell_Count": count,
                    "Percent_of_Definition_Cells": 100.0 * count / total if total else math.nan,
                }
            )
    result = pd.DataFrame.from_records(records)
    result["_q"] = result["Qref_Definition"].map(QREF_ORDER)
    result["_s"] = result["EOL_Status"].map(STATUS_ORDER)
    return result.sort_values(["Dataset", "_q", "_s"], kind="mergesort").drop(columns=["_q", "_s"]).reset_index(drop=True)


def _build_rul_summary(cell_level: pd.DataFrame, source_rul: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    source_idx = source_rul.set_index(["Dataset", "Qref_Definition"], drop=False)
    for (dataset, definition), group in cell_level.groupby(["Dataset", "Qref_Definition"], sort=False):
        key = (dataset, definition)
        source = source_idx.loc[key] if key in source_idx.index else pd.Series(dtype=object)
        if isinstance(source, pd.DataFrame):
            raise ValueError(f"Duplicate source RUL summary key: {key}")
        record = {
            "Dataset": dataset,
            "Qref_Definition": definition,
            "Dataset_Cell_Count": len(group),
            "Point_RUL_Available_Cell_Count": int((group["RUL_Point_Label_Available"] == "YES").sum()),
            "Interval_Bounded_RUL_Only_Cell_Count": int((group["RUL_Status"] == "INTERVAL_BOUNDED_RUL_ONLY").sum()),
            "Right_Censored_No_Point_RUL_Cell_Count": int((group["RUL_Status"] == "CENSORED_EOL_NO_POINT_RUL").sum()),
            "Qref_Unavailable_RUL_Cell_Count": int((group["RUL_Status"] == "QREF_UNAVAILABLE").sum()),
            "Point_RUL_Availability_Pct": 100.0 * (group["RUL_Point_Label_Available"] == "YES").sum() / len(group),
        }
        for column in [
            "Source_RUL_Row_Count",
            "Source_Point_RUL_Row_Count",
            "Source_Censored_No_Point_RUL_Row_Count",
            "Source_Interval_Bound_RUL_Row_Count",
            "Source_Qref_Unavailable_Row_Count",
            "Source_Post_EOL_Excluded_Row_Count",
            "Source_Censored_Numeric_RUL_Violation_Count",
        ]:
            record[column] = int(source.get(column, 0)) if len(source) else 0
        records.append(record)
    result = pd.DataFrame.from_records(records)
    result["_order"] = result["Qref_Definition"].map(QREF_ORDER)
    return result.sort_values(["Dataset", "_order"], kind="mergesort").drop(columns="_order").reset_index(drop=True)


def _series_equal_with_nan(left: pd.Series, right: pd.Series) -> bool:
    l = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return len(l) == len(r) and bool(np.allclose(l, r, atol=0, rtol=0, equal_nan=True))


def _build_fixed_control_audit(x_eol: pd.DataFrame, h_eol: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    specs = [
        ("XJTU", x_eol, "fixed_absolute_1p6Ah", 2.0, 1.6, ["Batch_ID", "Cell_ID"], "EOL_Exact_Aging_Cycle", "EOL_Lower_Bound_Aging_Cycle", "EOL_Upper_Bound_Aging_Cycle"),
        ("HUST", h_eol, "fixed_absolute_0p88Ah", 1.1, 0.88, ["Cell_ID"], "EOL_Raw_Cycle", None, None),
    ]
    for dataset, frame, fixed_def, nominal_qref, fixed_value, keys, exact_col, lower_col, upper_col in specs:
        nominal = frame[frame["Qref_Definition"] == "nominal"].copy()
        fixed = frame[frame["Qref_Definition"] == fixed_def].copy()
        merged = nominal.merge(fixed, on=keys, suffixes=("_Nominal", "_Fixed"), validate="one_to_one")
        expected = 0.8 * nominal_qref
        status_match = bool((merged["EOL_Status_Nominal"] == merged["EOL_Status_Fixed"]).all())
        exact_match = _series_equal_with_nan(merged[f"{exact_col}_Nominal"], merged[f"{exact_col}_Fixed"])
        bounds_match = True
        if lower_col:
            bounds_match &= _series_equal_with_nan(merged[f"{lower_col}_Nominal"], merged[f"{lower_col}_Fixed"])
        if upper_col:
            bounds_match &= _series_equal_with_nan(merged[f"{upper_col}_Nominal"], merged[f"{upper_col}_Fixed"])
        records.append(
            {
                "Dataset": dataset,
                "Nominal_Qref_Ah": nominal_qref,
                "EOL_SOH_Threshold": 0.8,
                "Computed_Nominal_80pct_Threshold_Ah": expected,
                "Frozen_Fixed_Absolute_Control_Ah": fixed_value,
                "Absolute_Difference_Ah": fixed_value - expected,
                "Match_Within_Tolerance": "YES" if abs(fixed_value - expected) <= FIXED_CONTROL_TOL_AH else "NO",
                "Frozen_Nominal_Cell_Count": len(nominal),
                "Frozen_Fixed_Control_Cell_Count": len(fixed),
                "EOL_Status_Matches_Nominal_All_Cells": "YES" if status_match else "NO",
                "EOL_Cycle_And_Bounds_Match_Nominal_All_Cells": "YES" if exact_match and bounds_match else "NO",
                "Cross_Chemistry_Numeric_Comparison": "NOT_PERFORMED_DIFFERENT_CHEMISTRY_NOMINALS",
            }
        )
    return pd.DataFrame.from_records(records)


def _xjtu_summary_reproduction(
    x_qref: pd.DataFrame,
    x_eol: pd.DataFrame,
    x_cell: pd.DataFrame,
    x_censor: pd.DataFrame,
    x_batch: pd.DataFrame,
) -> tuple[bool, str]:
    qref_pairs = {
        "Qref_nominal_Ah": "Qref_nominal_Ah",
        "Qref_initial_test_Ah": "Qref_initial_test_Ah",
        "Qref_first_comparable_Ah": "Qref_first_comparable_Ah",
        "Qref_early_median5_comparable_Ah": "Qref_early_median5_comparable_Ah",
    }
    merged_q = x_qref.merge(x_cell, on=["Batch_ID", "Cell_ID"], suffixes=("_qref", "_cell"), validate="one_to_one")
    qref_ok = all(_series_equal_with_nan(merged_q[f"{left}_qref"], merged_q[f"{right}_cell"]) for left, right in qref_pairs.items())

    pivot = x_eol.pivot(index=["Batch_ID", "Cell_ID"], columns="Qref_Definition", values="EOL_Status").reset_index()
    merged_e = x_cell.merge(pivot, on=["Batch_ID", "Cell_ID"], validate="one_to_one")
    status_defs = ("nominal", "initial_test", "first_comparable", "early_median5_comparable", "fixed_absolute_1p6Ah")
    eol_ok = all((merged_e[f"EOL_Status_{definition}"] == merged_e[definition]).all() for definition in status_defs)

    recomputed = (
        x_eol.groupby(["Batch_ID", "Qref_Definition", "EOL_Status"], dropna=False)
        .size()
        .reset_index(name="Cell_Count")
    )
    frozen = x_censor[["Batch_ID", "Qref_Definition", "EOL_Status", "Cell_Count"]].copy()
    for frame in (recomputed, frozen):
        frame["Batch_ID"] = pd.to_numeric(frame["Batch_ID"], errors="raise").astype(int)
        frame["Cell_Count"] = pd.to_numeric(frame["Cell_Count"], errors="raise").astype(int)
    sort_cols = ["Batch_ID", "Qref_Definition", "EOL_Status"]
    censor_ok = recomputed.sort_values(sort_cols).reset_index(drop=True).equals(frozen.sort_values(sort_cols).reset_index(drop=True))

    counts = x_qref.groupby("Batch_ID").size().rename("Observed").reset_index()
    expected_counts = x_batch[["Batch_ID", "Canonical_Cell_Count"]].copy()
    merged_b = counts.merge(expected_counts, on="Batch_ID", validate="one_to_one")
    batch_ok = bool((pd.to_numeric(merged_b["Observed"]) == pd.to_numeric(merged_b["Canonical_Cell_Count"])).all())
    details = f"qref_values={qref_ok}; eol_statuses={eol_ok}; censoring_counts={censor_ok}; batch_cell_counts={batch_ok}"
    return qref_ok and eol_ok and censor_ok and batch_ok, details


def _plot_qref_deviation(cell_level: pd.DataFrame, path: Path) -> None:
    categories = [
        ("XJTU", "initial_test", "XJTU\ninitial test"),
        ("XJTU", "first_comparable", "XJTU\nfirst comparable"),
        ("XJTU", "early_median5_comparable", "XJTU\nearly median-5"),
        ("HUST", "first_comparable", "HUST\nfirst comparable"),
        ("HUST", "early_median5_comparable", "HUST\nearly median-5"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    rng = np.random.default_rng(20260928)
    colors = {"XJTU": "#2867B2", "HUST": "#D97706"}
    all_values: list[np.ndarray] = []
    for idx, (dataset, definition, label) in enumerate(categories, start=1):
        values = cell_level.loc[
            (cell_level["Dataset"] == dataset) & (cell_level["Qref_Definition"] == definition),
            "Delta_Qref_vs_Nominal_Pct",
        ].dropna().to_numpy(dtype=float)
        all_values.append(values)
        if len(values):
            parts = ax.boxplot(
                [values],
                positions=[idx],
                widths=0.52,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#111827", "linewidth": 1.5},
                whiskerprops={"color": colors[dataset]},
                capprops={"color": colors[dataset]},
                boxprops={"facecolor": colors[dataset], "alpha": 0.25, "edgecolor": colors[dataset]},
            )
            _ = parts
            jitter = rng.normal(0, 0.055, size=len(values))
            ax.scatter(np.full(len(values), idx) + jitter, values, s=15, alpha=0.58, color=colors[dataset], edgecolors="none")
    ax.axhline(0, color="#4B5563", linewidth=1, linestyle="--")
    ax.set_xticks(range(1, len(categories) + 1), [item[2] for item in categories])
    ax.set_ylabel("Qref deviation from nominal (%)")
    ax.set_title("Qref deviation from nominal by dataset")
    ax.text(0.01, 0.01, "HUST has no initial-test Qref; XJTU initial-test is secondary only.", transform=ax.transAxes, fontsize=9, color="#4B5563")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_censoring(censoring: pd.DataFrame, path: Path) -> None:
    colors = {
        "EXACT_OBSERVED": "#2E8B57",
        "INTERVAL_CENSORED": "#E0A100",
        "RIGHT_CENSORED": "#6B7280",
        "QREF_UNAVAILABLE": "#C2415D",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), sharey=True)
    for ax, dataset in zip(axes, ("XJTU", "HUST")):
        subset = censoring[censoring["Dataset"] == dataset].copy()
        definitions = [d for d in ("nominal", "first_comparable", "early_median5_comparable", "initial_test") if d in set(subset["Qref_Definition"])]
        x = np.arange(len(definitions))
        bottom = np.zeros(len(definitions))
        for status in ("EXACT_OBSERVED", "INTERVAL_CENSORED", "RIGHT_CENSORED", "QREF_UNAVAILABLE"):
            vals = []
            for definition in definitions:
                row = subset[(subset["Qref_Definition"] == definition) & (subset["EOL_Status"] == status)]
                vals.append(float(row["Cell_Count"].iloc[0]) if len(row) else 0.0)
            ax.bar(x, vals, bottom=bottom, color=colors[status], label=status.replace("_", " "), width=0.68)
            bottom += np.asarray(vals)
        ax.set_xticks(x, [d.replace("early_median5_comparable", "early median-5").replace("first_comparable", "first comparable").replace("initial_test", "initial test") for d in definitions], rotation=22, ha="right")
        ax.set_title(dataset)
        ax.set_ylabel("Cell count")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=4, frameon=False)
    fig.suptitle("EOL censoring status by dataset and Qref definition", y=0.985)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.20, top=0.77, wspace=0.10)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_transition_matrix(transition: pd.DataFrame, path: Path) -> None:
    columns = [
        ("XJTU", "first_comparable"),
        ("XJTU", "early_median5_comparable"),
        ("HUST", "first_comparable"),
        ("HUST", "early_median5_comparable"),
    ]
    matrix = np.zeros((len(ALLOWED_TRANSITIONS), len(columns)), dtype=int)
    for j, (dataset, definition) in enumerate(columns):
        subset = transition[(transition["Dataset"] == dataset) & (transition["Qref_Definition"] == definition)].set_index("Transition_Category")
        for i, category in enumerate(ALLOWED_TRANSITIONS):
            matrix[i, j] = int(subset.loc[category, "Cell_Count"]) if category in subset.index else 0
    fig, ax = plt.subplots(figsize=(10.4, 7.1), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="#111827" if matrix[i, j] < matrix.max() * 0.65 else "white", fontsize=9)
    ax.set_xticks(range(len(columns)), ["XJTU\nfirst", "XJTU\nmedian-5", "HUST\nfirst", "HUST\nmedian-5"])
    ax.set_yticks(range(len(ALLOWED_TRANSITIONS)), [x.replace("_", " ") for x in ALLOWED_TRANSITIONS])
    ax.set_title("EOL status transitions relative to nominal")
    fig.colorbar(image, ax=ax, label="Cell count", shrink=0.82)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_first_median_delta(first_median: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True, sharey=False)
    for ax, dataset, color in zip(axes, ("XJTU", "HUST"), ("#2867B2", "#D97706")):
        values = first_median.loc[
            (first_median["Dataset"] == dataset) & first_median["Both_Exact"],
            "Delta_EOL_Median5_Minus_First_Cycles",
        ].dropna().to_numpy(dtype=float)
        if len(values):
            bins = np.arange(math.floor(values.min()) - 0.5, math.ceil(values.max()) + 1.5, 1.0)
            ax.hist(values, bins=bins, color=color, alpha=0.82, edgecolor="white")
            ax.axvline(np.median(values), color="#111827", linestyle="--", linewidth=1.4, label=f"Median = {np.median(values):.0f}")
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.55, "No exact/exact cell pairs", transform=ax.transAxes, ha="center", va="center", fontsize=12)
            ax.text(0.5, 0.43, "Censored EOL values are not converted to points.", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#4B5563")
        ax.axvline(0, color="#6B7280", linewidth=1)
        ax.set_title(dataset)
        ax.set_xlabel("EOL(early median-5) − EOL(first comparable), cycles")
        ax.set_ylabel("Cell count")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("First-comparable versus early-median5 exact EOL difference")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _add_test(tests: list[dict[str, Any]], test_id: str, scope: str, expected: Any, observed: Any, passed: bool, details: str = "") -> None:
    tests.append(
        {
            "Test_ID": test_id,
            "Scope": scope,
            "Expected": expected,
            "Observed": observed,
            "Status": "PASS" if passed else "FAIL",
            "Details": details,
        }
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g", na_rep="")


def run_cross_dataset_summary(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[1]
    xdir = root / "outputs" / "xjtu_55_final_v2"
    hdir = root / "outputs" / "hust_77_final_v1"
    target = root / "outputs" / OUTPUT_DIR_NAME
    temp = root / "outputs" / f".{OUTPUT_DIR_NAME}.tmp-{os.getpid()}"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {target}")
    if temp.exists():
        raise FileExistsError(f"Temporary output directory already exists: {temp}")
    if not xdir.is_dir() or not hdir.is_dir():
        raise FileNotFoundError("Frozen XJTU Final V2 or HUST Final V1 directory is missing.")

    xfiles = {
        "qref": xdir / "xjtu_55_qref_summary.csv",
        "eol": xdir / "xjtu_55_eol_sensitivity.csv",
        "pairwise": xdir / "xjtu_55_eol_pairwise.csv",
        "rul": xdir / "xjtu_55_rul_labels.csv",
        "censor": xdir / "xjtu_55_censoring_summary.csv",
        "cell": xdir / "xjtu_55_cell_summary.csv",
        "batch": xdir / "xjtu_55_batch_summary.csv",
        "regression": xdir / "xjtu_55_run_regression_tests.csv",
    }
    hfiles = {
        "qref": hdir / "hust_77_qref_summary.csv",
        "eol": hdir / "hust_77_eol_sensitivity.csv",
        "pairwise": hdir / "hust_77_eol_pairwise.csv",
        "rul": hdir / "hust_77_rul_labels.csv",
        "censor": hdir / "hust_77_censoring_summary.csv",
        "cell": hdir / "hust_77_cell_summary.csv",
        "stop": hdir / "hust_77_stopping_rule_audit.csv",
        "regression": hdir / "hust_77_run_regression_tests.csv",
    }
    for path in (*xfiles.values(), *hfiles.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    snapshot_before = {"XJTU": _snapshot(xdir), "HUST": _snapshot(hdir)}
    snapshot_digest_before = {key: _snapshot_digest(value) for key, value in snapshot_before.items()}

    x_qref = _read_csv(xfiles["qref"])
    x_eol = _read_csv(xfiles["eol"])
    x_pair = _read_csv(xfiles["pairwise"])
    x_censor = _read_csv(xfiles["censor"])
    x_cell = _read_csv(xfiles["cell"])
    x_batch = _read_csv(xfiles["batch"])
    x_reg = _read_csv(xfiles["regression"])
    h_qref = _read_csv(hfiles["qref"])
    h_eol = _read_csv(hfiles["eol"])
    h_pair = _read_csv(hfiles["pairwise"])
    h_censor = _read_csv(hfiles["censor"])
    h_cell = _read_csv(hfiles["cell"])
    h_stop = _read_csv(hfiles["stop"])
    h_reg = _read_csv(hfiles["regression"])

    _require_columns(x_qref, ["Batch_ID", "Cell_ID", "Qref_nominal_Ah", "Qref_first_comparable_Ah", "Qref_early_median5_comparable_Ah"], "XJTU qref")
    _require_columns(x_eol, ["Batch_ID", "Cell_ID", "Qref_Definition", "EOThreshold_Ah", "EOL_Status", "EOL_Exact_Aging_Cycle", "EOL_Lower_Bound_Aging_Cycle", "EOL_Upper_Bound_Aging_Cycle", "Last_Valid_Aging_Cycle"], "XJTU eol")
    _require_columns(h_qref, ["Cell_ID", "Qref_Nominal_Ah", "Qref_First_Comparable_Ah", "Qref_Early_Median5_Comparable_Ah", "Qref_Initial_Test_Ah"], "HUST qref")
    _require_columns(h_eol, ["Cell_ID", "Qref_Definition", "EOThreshold_Ah", "EOL_Status", "EOL_Raw_Cycle", "Last_Valid_Raw_Cycle"], "HUST eol")

    x_level = _build_xjtu_cell_level(x_qref, x_eol)
    h_level = _build_hust_cell_level(h_qref, h_eol)
    cell_level = pd.concat([x_level, h_level], ignore_index=True)
    cell_level["_q"] = cell_level["Qref_Definition"].map(QREF_ORDER)
    cell_level = cell_level.sort_values(["Dataset", "Batch_ID", "Cell_ID", "_q"], kind="mergesort", na_position="last").drop(columns="_q").reset_index(drop=True)

    effect_bounds = _build_effect_bounds(cell_level)
    first_median = _first_vs_median_delta(cell_level)
    qref_summary = _build_qref_summary(cell_level, first_median)
    transition_summary = _build_transition_summary(effect_bounds)
    censoring_summary = _build_censoring_summary(cell_level)
    x_source_rul = _source_rul_summary(xfiles["rul"], "XJTU", set(COMMON_QREFS) | set(XJTU_SECONDARY_QREFS))
    h_source_rul = _source_rul_summary(hfiles["rul"], "HUST", set(COMMON_QREFS))
    source_rul = pd.concat([x_source_rul, h_source_rul], ignore_index=True)
    rul_summary = _build_rul_summary(cell_level, source_rul)
    fixed_audit = _build_fixed_control_audit(x_eol, h_eol)

    tests: list[dict[str, Any]] = []
    x_cells = x_qref["Cell_ID"].nunique()
    h_cells = h_qref["Cell_ID"].nunique()
    combined = cell_level[["Dataset", "Cell_ID"]].drop_duplicates().shape[0]
    _add_test(tests, "XJTU_CELL_COUNT_55", "frozen cell inventory", 55, x_cells, x_cells == 55)
    _add_test(tests, "HUST_CELL_COUNT_77", "frozen cell inventory", 77, h_cells, h_cells == 77)
    _add_test(tests, "COMBINED_UNIQUE_CELL_COUNT_132", "harmonized inventory", 132, combined, combined == 132)
    _add_test(tests, "CELL_LEVEL_UNIQUE_KEY", "harmonized cell level", "no duplicate Dataset|Cell_ID|Qref_Definition", int(cell_level.duplicated(["Dataset", "Cell_ID", "Qref_Definition"]).sum()), not cell_level.duplicated(["Dataset", "Cell_ID", "Qref_Definition"]).any())

    hust_initial_source_nonblank = int(pd.to_numeric(h_qref["Qref_Initial_Test_Ah"], errors="coerce").notna().sum())
    hust_initial_output_rows = int(((cell_level["Dataset"] == "HUST") & (cell_level["Qref_Definition"] == "initial_test")).sum())
    _add_test(tests, "NO_HUST_INITIAL_TEST_INVENTED", "Qref harmonization", "0 source values and 0 output rows", f"source={hust_initial_source_nonblank}; output={hust_initial_output_rows}", hust_initial_source_nonblank == 0 and hust_initial_output_rows == 0)

    expected_rows = 55 * 4 + 77 * 3
    _add_test(tests, "CELL_LEVEL_ROW_COUNT", "harmonized cell level", expected_rows, len(cell_level), len(cell_level) == expected_rows)
    common_counts = cell_level[cell_level["Qref_Definition"].isin(COMMON_QREFS)].groupby(["Dataset", "Qref_Definition"])["Cell_ID"].nunique()
    common_ok = all(common_counts.get((dataset, definition), 0) == count for dataset, count in (("XJTU", 55), ("HUST", 77)) for definition in COMMON_QREFS)
    _add_test(tests, "COMMON_QREFS_COMPLETE", "primary cross-dataset definitions", "XJTU=55 and HUST=77 for each common definition", common_counts.to_dict(), common_ok)

    threshold_error = (cell_level.loc[cell_level["Qref_Ah"].notna(), "EOL_Threshold_Ah"] - 0.8 * cell_level.loc[cell_level["Qref_Ah"].notna(), "Qref_Ah"]).abs().max()
    _add_test(tests, "QREF_THRESHOLDS_RETAIN_80_PERCENT_RULE", "harmonized cell level", f"max error <= {FLOAT_TOL}", threshold_error, bool(threshold_error <= FLOAT_TOL))
    censored_point_violation = cell_level[(cell_level["EOL_Status"] != "EXACT_OBSERVED") & cell_level["EOL_Exact_Cycle"].notna()]
    _add_test(tests, "NO_CENSORED_EOL_CONVERTED_TO_EXACT_POINT", "harmonized cell level", 0, len(censored_point_violation), len(censored_point_violation) == 0)
    cell_rul_violation = cell_level[(cell_level["EOL_Status"] != "EXACT_OBSERVED") & (cell_level["RUL_Point_Label_Available"] == "YES")]
    _add_test(tests, "NO_CENSORED_CELL_RUL_POINT_AVAILABILITY", "harmonized cell level", 0, len(cell_rul_violation), len(cell_rul_violation) == 0)
    source_rul_violations = int(source_rul["Source_Censored_Numeric_RUL_Violation_Count"].sum())
    _add_test(tests, "NO_FROZEN_CENSORED_RUL_POINT_LABEL", "frozen RUL rows", 0, source_rul_violations, source_rul_violations == 0)

    fixed_value_ok = bool((fixed_audit["Match_Within_Tolerance"] == "YES").all())
    fixed_eol_ok = bool((fixed_audit["EOL_Status_Matches_Nominal_All_Cells"] == "YES").all() and (fixed_audit["EOL_Cycle_And_Bounds_Match_Nominal_All_Cells"] == "YES").all())
    _add_test(tests, "FIXED_CONTROLS_MATCH_NOMINAL_80_PERCENT", "dataset-specific physical controls", "XJTU=1.600 Ah; HUST=0.880 Ah", fixed_audit[["Dataset", "Frozen_Fixed_Absolute_Control_Ah", "Computed_Nominal_80pct_Threshold_Ah"]].to_dict("records"), fixed_value_ok)
    _add_test(tests, "FIXED_CONTROL_EOL_RESULTS_MATCH_NOMINAL", "frozen EOL rows", "all cells", fixed_audit[["Dataset", "EOL_Status_Matches_Nominal_All_Cells", "EOL_Cycle_And_Bounds_Match_Nominal_All_Cells"]].to_dict("records"), fixed_eol_ok)

    hust_status = h_eol[h_eol["Qref_Definition"].isin(COMMON_QREFS)].groupby(["Qref_Definition", "EOL_Status"]).size()
    hust_status_expectations = {
        ("nominal", "RIGHT_CENSORED"): 77,
        ("first_comparable", "EXACT_OBSERVED"): 77,
        ("early_median5_comparable", "EXACT_OBSERVED"): 77,
    }
    hust_status_ok = all(int(hust_status.get(key, 0)) == value for key, value in hust_status_expectations.items()) and int(hust_status.sum()) == 231
    _add_test(tests, "HUST_FROZEN_EOL_STATUS_COUNTS", "HUST common Qrefs", hust_status_expectations, hust_status.to_dict(), hust_status_ok)

    hust_above = cell_level[(cell_level["Dataset"] == "HUST") & cell_level["Qref_Definition"].isin(MEASURED_COMMON_QREFS)]
    _add_test(tests, "HUST_MEASURED_QREFS_ALL_ABOVE_NOMINAL", "direction audit", 154, int((hust_above["Qref_Direction_vs_Nominal"] == "QREF_ABOVE_NOMINAL").sum()), bool((hust_above["Qref_Direction_vs_Nominal"] == "QREF_ABOVE_NOMINAL").all()))
    transition_values = set(effect_bounds.loc[effect_bounds["Comparison_Scope"] == "PRIMARY_COMMON", "Transition_Category"])
    transition_ok = transition_values.issubset(set(ALLOWED_TRANSITIONS)) and "" not in transition_values
    _add_test(tests, "TRANSITIONS_USE_ALLOWED_CATEGORIES", "direction audit", sorted(ALLOWED_TRANSITIONS), sorted(transition_values), transition_ok)
    inconsistent_count = int((effect_bounds.loc[effect_bounds["Comparison_Scope"] == "PRIMARY_COMMON", "Threshold_Ordering_Consistency"] == "INCONSISTENT").sum())
    _add_test(tests, "NO_THRESHOLD_ORDERING_CONTRADICTION", "direction audit", 0, inconsistent_count, inconsistent_count == 0)

    x_reproduced, x_reproduction_details = _xjtu_summary_reproduction(x_qref, x_eol, x_cell, x_censor, x_batch)
    _add_test(tests, "XJTU_FROZEN_SUMMARY_RESULTS_REPRODUCED_EXACTLY", "XJTU Final V2", "all frozen Qref/EOL/censoring/batch summaries", x_reproduction_details, x_reproduced)
    x_reg_ok = bool((x_reg["Status"] == "PASS").all())
    h_reg_ok = bool((h_reg["Status"] == "PASS").all())
    _add_test(tests, "XJTU_FROZEN_REGRESSION_TABLE_ALL_PASS", "XJTU Final V2", "all PASS", x_reg["Status"].value_counts().to_dict(), x_reg_ok)
    _add_test(tests, "HUST_FROZEN_REGRESSION_TABLE_ALL_PASS", "HUST Final V1", "all PASS", h_reg["Status"].value_counts().to_dict(), h_reg_ok)

    hust_effect = effect_bounds[(effect_bounds["Dataset"] == "HUST") & effect_bounds["Definition_B"].isin(MEASURED_COMMON_QREFS)].copy()
    hust_pair = h_pair[(h_pair["Definition_A"] == "nominal") & h_pair["Definition_B"].isin(MEASURED_COMMON_QREFS)][["Cell_ID", "Definition_B", "Minimum_Observed_Earlier_By_Cycles"]].copy()
    hust_effect_compare = hust_effect.merge(hust_pair, left_on=["Cell_ID", "Definition_B"], right_on=["Cell_ID", "Definition_B"], suffixes=("_Computed", "_Frozen"), validate="one_to_one")
    bound_match = _series_equal_with_nan(hust_effect_compare["Minimum_Observed_Earlier_By_Cycles_Computed"], hust_effect_compare["Minimum_Observed_Earlier_By_Cycles_Frozen"])
    exact_shift_blank = hust_effect["Delta_EOL_Exact_Cycles"].isna().all()
    _add_test(tests, "HUST_RIGHT_TO_EXACT_EFFECT_REMAINS_LOWER_BOUND", "effect bounds", "154 frozen lower bounds; 0 exact deltas", f"bounds_match={bound_match}; exact_delta_nonblank={int(hust_effect['Delta_EOL_Exact_Cycles'].notna().sum())}", bool(bound_match and exact_shift_blank))

    temp.mkdir(parents=False)
    figures = temp / "figures"
    figures.mkdir()
    try:
        output_frames = {
            "cross_dataset_cell_level.csv": cell_level,
            "cross_dataset_qref_summary.csv": qref_summary,
            "cross_dataset_eol_transition_summary.csv": transition_summary,
            "cross_dataset_censoring_summary.csv": censoring_summary,
            "cross_dataset_rul_availability_summary.csv": rul_summary,
            "cross_dataset_effect_bounds.csv": effect_bounds,
            "cross_dataset_fixed_control_audit.csv": fixed_audit,
        }
        for name, frame in output_frames.items():
            _write_csv(frame, temp / name)

        _plot_qref_deviation(cell_level, figures / "qref_deviation_by_dataset.png")
        _plot_censoring(censoring_summary, figures / "eol_censoring_distribution.png")
        _plot_transition_matrix(transition_summary, figures / "eol_transition_matrix.png")
        _plot_first_median_delta(first_median, figures / "first_vs_median5_eol_difference.png")

        readback_ok = True
        readback_details: list[str] = []
        for name, frame in output_frames.items():
            loaded = pd.read_csv(temp / name, low_memory=False)
            same_rows = len(loaded) == len(frame)
            same_cols = loaded.columns.tolist() == frame.columns.tolist()
            readback_ok &= same_rows and same_cols
            readback_details.append(f"{name}:rows={len(loaded)}/{len(frame)},columns={same_cols}")
        _add_test(tests, "OUTPUT_CSV_READBACK_SCHEMA_AND_ROW_COUNTS", "seven analysis CSVs", "rows and columns preserved", " | ".join(readback_details), readback_ok)

        snapshot_after = {"XJTU": _snapshot(xdir), "HUST": _snapshot(hdir)}
        snapshot_digest_after = {key: _snapshot_digest(value) for key, value in snapshot_after.items()}
        hashes_unchanged = snapshot_digest_before == snapshot_digest_after
        rows_unchanged = all(
            snapshot_before[dataset][name]["Data_Row_Count"] == snapshot_after[dataset][name]["Data_Row_Count"]
            for dataset in snapshot_before
            for name in snapshot_before[dataset]
            if name.endswith(".csv")
        )
        _add_test(tests, "FROZEN_SOURCE_HASHES_UNCHANGED", "XJTU Final V2 and HUST Final V1", snapshot_digest_before, snapshot_digest_after, hashes_unchanged)
        _add_test(tests, "FROZEN_SOURCE_ROWS_UNCHANGED", "XJTU Final V2 and HUST Final V1", "all CSV row counts unchanged", "all unchanged" if rows_unchanged else "row-count difference detected", rows_unchanged)

        regression = pd.DataFrame.from_records(tests)
        _write_csv(regression, temp / "cross_dataset_regression_tests.csv")
        regression_readback = pd.read_csv(temp / "cross_dataset_regression_tests.csv")
        if len(regression_readback) != len(regression) or regression_readback["Test_ID"].duplicated().any():
            raise RuntimeError("Regression CSV readback validation failed.")
        if not (regression["Status"] == "PASS").all():
            failures = regression.loc[regression["Status"] != "PASS", ["Test_ID", "Observed", "Details"]].to_dict("records")
            raise RuntimeError(f"CROSS-DATASET AGGREGATION FAILED: {failures}")

        x_below = cell_level[(cell_level["Dataset"] == "XJTU") & cell_level["Qref_Definition"].isin(MEASURED_COMMON_QREFS) & (cell_level["Qref_Direction_vs_Nominal"] == "QREF_BELOW_NOMINAL")]
        first_median_stats: list[str] = []
        for dataset in ("XJTU", "HUST"):
            vals = first_median.loc[(first_median["Dataset"] == dataset) & first_median["Both_Exact"], "Delta_EOL_Median5_Minus_First_Cycles"].dropna()
            if len(vals):
                first_median_stats.append(f"{dataset}: n={len(vals)}, min={vals.min():.12g}, median={vals.median():.12g}, max={vals.max():.12g}, median_abs={vals.abs().median():.12g}")
            else:
                first_median_stats.append(f"{dataset}: n=0 exact/exact pairs; censored EOL values were not converted to points")

        log_lines = [
            "Cross-dataset aggregation: XJTU Final V2 + HUST Final V1",
            f"Processing timestamp UTC: {datetime.now(timezone.utc).isoformat()}",
            f"Processing timestamp local: {datetime.now().astimezone().isoformat()}",
            f"Python: {platform.python_version()}",
            f"pandas: {pd.__version__}",
            f"numpy: {np.__version__}",
            f"matplotlib: {matplotlib.__version__}",
            "",
            "Frozen input policy",
            f"- XJTU: {xdir}",
            f"- HUST: {hdir}",
            "- Only frozen CSV/TXT outputs were read. No raw MAT/PKL files were read or reprocessed.",
            f"- XJTU snapshot digest before/after: {snapshot_digest_before['XJTU']} / {snapshot_digest_after['XJTU']}",
            f"- HUST snapshot digest before/after: {snapshot_digest_before['HUST']} / {snapshot_digest_after['HUST']}",
            "",
            "Cell inventory",
            f"- XJTU cells: {x_cells}",
            f"- HUST cells: {h_cells}",
            f"- Combined unique Dataset|Cell_ID keys: {combined}",
            f"- Harmonized cell-Qref rows: {len(cell_level)}",
            "",
            "Scientific scope",
            "- Primary common Qrefs: nominal, first_comparable, early_median5_comparable.",
            "- XJTU initial_test remains XJTU-only secondary analysis; no HUST initial_test was created.",
            "- HUST is a broader protocol-comparable measured-Qref sensitivity replication, not an exact nominal-versus-initial-test replication.",
            "- Qref ordering mathematically determines threshold ordering; interpretation focuses on magnitude, censoring transitions, RUL-label availability, and benchmark comparability.",
            f"- XJTU common measured-Qref rows below nominal: {len(x_below)} of {55 * 2}.",
            "- HUST common measured-Qref rows above nominal: 154 of 154.",
            "",
            "EOL and RUL handling",
            "- Interval- and right-censored EOL values remain censored; no artificial exact EOL point was created.",
            "- Censored RUL remains without a point label.",
            "- HUST nominal right-censored to measured exact comparisons retain Minimum_Observed_Earlier_By_Cycles as a lower bound only.",
            "- Dataset-specific fixed controls: XJTU 1.600 Ah and HUST 0.880 Ah; no cross-chemistry numeric comparison was performed.",
            "",
            "First-comparable versus early-median5 exact EOL deltas",
            *[f"- {line}" for line in first_median_stats],
            "",
            "Regression tests",
            *[f"- {row.Test_ID}: {row.Status} (expected={row.Expected}; observed={row.Observed})" for row in regression.itertuples(index=False)],
            "",
            "CROSS-DATASET SUMMARY V1 COMPLETE",
        ]
        (temp / "processing_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        expected_files = set(output_frames) | {"cross_dataset_regression_tests.csv", "processing_log.txt"}
        observed_files = {p.name for p in temp.iterdir() if p.is_file()}
        if observed_files != expected_files:
            raise RuntimeError(f"Unexpected top-level output files: expected={sorted(expected_files)}, observed={sorted(observed_files)}")
        expected_figures = {
            "qref_deviation_by_dataset.png",
            "eol_censoring_distribution.png",
            "eol_transition_matrix.png",
            "first_vs_median5_eol_difference.png",
        }
        observed_figures = {p.name for p in figures.iterdir() if p.is_file()}
        if observed_figures != expected_figures:
            raise RuntimeError(f"Figure set mismatch: expected={sorted(expected_figures)}, observed={sorted(observed_figures)}")

        temp.rename(target)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        print("CROSS-DATASET AGGREGATION FAILED")
        raise

    print("CROSS-DATASET SUMMARY V1 COMPLETE")
    return target
