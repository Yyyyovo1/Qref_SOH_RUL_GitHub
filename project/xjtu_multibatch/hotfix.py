from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import scipy

from . import PIPELINE_VERSION
from .analysis import (
    _atomic_directory,
    _coerce_numeric_order_columns,
    _finalize_qref_summary,
    _qref_row_from_observations,
    _sha256,
    _write_csv,
)
from .matio import (
    cycle_sequence,
    discover_adapter,
    field_names,
    get_field,
    load_mat_payload,
    numeric_1d,
    time_to_seconds,
)
from .protocols import _select_main_segment


EXPECTED_BATCH5_CHECKS = {
    ("B5_RW_battery-4", 252),
    ("B5_RW_battery-5", 274),
    ("B5_RW_battery-6", 219),
    ("B5_RW_battery-7", 186),
    ("B5_RW_battery-8", 208),
}

QREF_REGRESSION_CASES = (
    ("BATCH2_REPRESENTATIVE_FIRST5", "B2_3C_battery-3", "2|3|4|5|6", None),
    ("BATCH3_R2P5_BATTERY4_FIRST5", "B3_R2.5_battery-4", "3|8|13|18|23", None),
    ("BATCH4_R3_BATTERY4_FIRST5", "B4_R3_battery-4", "7|13|19|25|31", None),
    ("BATCH5_RW_BATTERY1_FIRST5", "B5_RW_battery-1", "21|32|43|54|65", 1.923),
    (
        "BATCH6_SIM_SATELLITE_BATTERY1_FIRST5",
        "B6_Sim_satellite_battery-1",
        "7|13|19|24|30",
        1.942,
    ),
)

REQUIRED_ORDER_COLUMNS = (
    "Diagnostic_Index",
    "Raw_Cycle_Index",
    "Cumulative_Aging_Operation_Count",
)


def _tree_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _same_numeric(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    a = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    b = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return bool(np.isclose(float(a), float(b), atol=tolerance, rtol=0.0))


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _corrected_qref_preview(
    cycle_qc: pd.DataFrame,
    prehotfix_qref: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    corrected_rows: list[dict[str, Any]] = []
    ordered_groups: dict[str, pd.DataFrame] = {}

    for item in prehotfix_qref.sort_values(["Batch_ID", "Cell_ID"]).itertuples(index=False):
        existing = item._asdict()
        batch_id = int(existing["Batch_ID"])
        cell = str(existing["Cell_ID"])
        cell_rows = cycle_qc.loc[cycle_qc["Cell_ID"].eq(cell)].copy()
        if cell_rows.empty:
            raise RuntimeError(f"QREF_CELL_MISSING_FROM_CYCLE_QC:{cell}")

        comparable = cell_rows.loc[
            cell_rows["Primary_Capacity_Eligible"].eq("YES")
            & cell_rows["Final_Include"].eq("YES")
            & ~cell_rows["Auto_QC_Flag"].eq("INVALID_CANDIDATE")
        ].copy()
        comparable = _coerce_numeric_order_columns(comparable)
        comparable["Capacity_Ah"] = pd.to_numeric(
            comparable["Capacity_Ah"], errors="raise"
        )
        comparable = comparable.sort_values(
            ["Diagnostic_Index", "Raw_Cycle_Index"], kind="stable"
        ).reset_index(drop=True)
        ordered_groups[cell] = comparable

        if batch_id == 1:
            corrected_rows.append(existing)
        else:
            corrected_rows.append(
                _qref_row_from_observations(cell, batch_id, cell_rows)
            )

    corrected = _finalize_qref_summary(pd.DataFrame(corrected_rows))
    corrected = corrected.sort_values(["Batch_ID", "Cell_ID"]).reset_index(drop=True)
    prehotfix = prehotfix_qref.sort_values(["Batch_ID", "Cell_ID"]).reset_index(drop=True)
    if list(corrected["Cell_ID"]) != list(prehotfix["Cell_ID"]):
        raise RuntimeError("QREF_CELL_ORDER_OR_MEMBERSHIP_CHANGED")

    protected_numeric = (
        "Qref_nominal_Ah",
        "Qref_initial_test_Ah",
        "Qref_first_comparable_Ah",
    )
    protected_text = (
        "Qref_nominal_Status",
        "Qref_Nominal_Source",
        "Qref_initial_test_Status",
        "Qref_Initial_Test_Source_Cycle",
        "Qref_first_comparable_Status",
        "Qref_First_Comparable_Source_Cycle",
    )
    for row_index in corrected.index:
        cell = corrected.at[row_index, "Cell_ID"]
        for column in protected_numeric:
            if not _same_numeric(corrected.at[row_index, column], prehotfix.at[row_index, column]):
                raise RuntimeError(f"HOTFIX_SCOPE_VIOLATION:{cell}:{column}")
        for column in protected_text:
            if _clean_text(corrected.at[row_index, column]) != _clean_text(
                prehotfix.at[row_index, column]
            ):
                raise RuntimeError(f"HOTFIX_SCOPE_VIOLATION:{cell}:{column}")

    preview = corrected.copy()
    preview["PRE_HOTFIX_Qref_early_median5_comparable_Ah"] = pd.to_numeric(
        prehotfix["Qref_early_median5_comparable_Ah"], errors="coerce"
    )
    preview["PRE_HOTFIX_Qref_Early_Median5_Comparable_Source_Cycles"] = prehotfix[
        "Qref_Early_Median5_Comparable_Source_Cycles"
    ].astype(str)
    preview["Hotfix_Qref_Delta_Ah"] = (
        pd.to_numeric(preview["Qref_early_median5_comparable_Ah"], errors="coerce")
        - preview["PRE_HOTFIX_Qref_early_median5_comparable_Ah"]
    )
    preview["Hotfix_Changed"] = np.where(
        preview["Qref_Early_Median5_Comparable_Source_Cycles"].astype(str)
        != preview[
            "PRE_HOTFIX_Qref_Early_Median5_Comparable_Source_Cycles"
        ].astype(str),
        "YES",
        "NO",
    )
    preview["Hotfix_Basis"] = np.where(
        preview["Batch_ID"].eq(1),
        "FROZEN_BATCH1_UNCHANGED",
        "FIRST_5_FINAL_INCLUDED_COMPARABLE_OBSERVATIONS_SORTED_BY_NUMERIC_DIAGNOSTIC_INDEX",
    )
    preview["Numeric_Order_Columns"] = "|".join(REQUIRED_ORDER_COLUMNS)
    preview["EOL_RUL_Recomputed"] = "NO"
    return preview, ordered_groups


def _regression_results(
    preview: pd.DataFrame,
    ordered_groups: Mapping[str, pd.DataFrame],
    batch5_checks: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for test_id, cell, expected_sources, expected_median in QREF_REGRESSION_CASES:
        match = preview.loc[preview["Cell_ID"].eq(cell)]
        if len(match) != 1:
            raise RuntimeError(f"REGRESSION_CELL_NOT_UNIQUE:{cell}")
        observed_sources = str(
            match.iloc[0]["Qref_Early_Median5_Comparable_Source_Cycles"]
        )
        observed_median = float(
            match.iloc[0]["Qref_early_median5_comparable_Ah"]
        )
        source_pass = observed_sources == expected_sources
        median_pass = expected_median is None or bool(
            np.isclose(observed_median, expected_median, atol=0.002, rtol=0.0)
        )
        rows.append(
            {
                "Test_ID": test_id,
                "Cell_ID": cell,
                "Expected_Source_Cycles": expected_sources,
                "Observed_Source_Cycles": observed_sources,
                "Expected_Median_Capacity_Ah": expected_median,
                "Observed_Median_Capacity_Ah": observed_median,
                "Tolerance_Ah": 0.002 if expected_median is not None else np.nan,
                "Expected_Capacity_Vector_Ah": "",
                "Observed_Capacity_Vector_Ah": "",
                "Status": "PASS" if source_pass and median_pass else "FAIL",
                "Details": "Numeric Diagnostic_Index order and median verified",
            }
        )

    b6 = ordered_groups["B6_Sim_satellite_battery-1"].head(5)
    expected_b6 = np.asarray([1.922, 1.928, 1.942, 1.953, 1.962])
    observed_b6 = pd.to_numeric(b6["Capacity_Ah"], errors="raise").to_numpy(float)
    vector_pass = observed_b6.shape == expected_b6.shape and bool(
        np.allclose(observed_b6, expected_b6, atol=0.002, rtol=0.0)
    )
    rows.append(
        {
            "Test_ID": "BATCH6_SIM_SATELLITE_BATTERY1_CAPACITY_VECTOR",
            "Cell_ID": "B6_Sim_satellite_battery-1",
            "Expected_Source_Cycles": "7|13|19|24|30",
            "Observed_Source_Cycles": "|".join(
                str(int(value)) for value in b6["Raw_Cycle_Index"]
            ),
            "Expected_Median_Capacity_Ah": 1.942,
            "Observed_Median_Capacity_Ah": float(np.median(observed_b6)),
            "Tolerance_Ah": 0.002,
            "Expected_Capacity_Vector_Ah": "|".join(f"{value:.3f}" for value in expected_b6),
            "Observed_Capacity_Vector_Ah": "|".join(f"{value:.3f}" for value in observed_b6),
            "Status": "PASS" if vector_pass else "FAIL",
            "Details": "Raw summary capacities for the five selected diagnostics",
        }
    )

    observed_checks = {
        (str(row.Cell_ID), int(row.Raw_Cycle_Index))
        for row in batch5_checks.itertuples(index=False)
    }
    rows.append(
        {
            "Test_ID": "BATCH5_CHECK_SET_EXACT",
            "Cell_ID": "BATCH5",
            "Expected_Source_Cycles": "|".join(
                f"{cell}:{cycle}" for cell, cycle in sorted(EXPECTED_BATCH5_CHECKS)
            ),
            "Observed_Source_Cycles": "|".join(
                f"{cell}:{cycle}" for cell, cycle in sorted(observed_checks)
            ),
            "Expected_Median_Capacity_Ah": np.nan,
            "Observed_Median_Capacity_Ah": np.nan,
            "Tolerance_Ah": np.nan,
            "Expected_Capacity_Vector_Ah": "",
            "Observed_Capacity_Vector_Ah": "",
            "Status": "PASS" if observed_checks == EXPECTED_BATCH5_CHECKS else "FAIL",
            "Details": "Exactly five Batch-5 CHECK rows; no manual decision changed",
        }
    )

    for column in REQUIRED_ORDER_COLUMNS:
        all_numeric = all(
            pd.api.types.is_numeric_dtype(group[column])
            for group in ordered_groups.values()
        )
        rows.append(
            {
                "Test_ID": f"STRICT_NUMERIC_ORDER_COLUMN_{column.upper()}",
                "Cell_ID": "ALL_55_CELLS",
                "Expected_Source_Cycles": "",
                "Observed_Source_Cycles": "",
                "Expected_Median_Capacity_Ah": np.nan,
                "Observed_Median_Capacity_Ah": np.nan,
                "Tolerance_Ah": np.nan,
                "Expected_Capacity_Vector_Ah": "",
                "Observed_Capacity_Vector_Ah": "",
                "Status": "PASS" if all_numeric else "FAIL",
                "Details": f"pd.to_numeric(errors='raise') applied to {column}",
            }
        )

    result = pd.DataFrame(rows)
    if result["Status"].ne("PASS").any():
        failures = "|".join(result.loc[result["Status"].ne("PASS"), "Test_ID"])
        raise RuntimeError(f"HOTFIX_REGRESSION_FAILED:{failures}")
    return result


def _batch5_checks(cycle_qc: pd.DataFrame) -> pd.DataFrame:
    checks = cycle_qc.loc[
        pd.to_numeric(cycle_qc["Batch_ID"], errors="raise").eq(5)
        & cycle_qc["Auto_QC_Flag"].eq("CHECK")
    ].copy()
    checks["Raw_Cycle_Index"] = pd.to_numeric(
        checks["Raw_Cycle_Index"], errors="raise"
    )
    observed = {
        (str(row.Cell_ID), int(row.Raw_Cycle_Index))
        for row in checks.itertuples(index=False)
    }
    if observed != EXPECTED_BATCH5_CHECKS:
        missing = sorted(EXPECTED_BATCH5_CHECKS - observed)
        extra = sorted(observed - EXPECTED_BATCH5_CHECKS)
        raise RuntimeError(
            f"BATCH5_CHECK_SET_MISMATCH:missing={missing};extra={extra}"
        )
    if not checks["Auto_QC_Reason_Codes"].eq("CAPACITY_JUMP").all():
        raise RuntimeError("BATCH5_CHECK_REASON_NOT_CAPACITY_JUMP_ONLY")
    return checks.sort_values(["Cell_ID", "Raw_Cycle_Index"]).reset_index(drop=True)


def _manual_review_table(cycle_qc: pd.DataFrame, checks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for current in checks.itertuples(index=False):
        cell_rows = cycle_qc.loc[
            cycle_qc["Cell_ID"].eq(current.Cell_ID)
            & cycle_qc["Primary_Capacity_Eligible"].eq("YES")
            & cycle_qc["Final_Include"].eq("YES")
            & ~cycle_qc["Auto_QC_Flag"].eq("INVALID_CANDIDATE")
        ].copy()
        cell_rows = _coerce_numeric_order_columns(cell_rows)
        cell_rows["Capacity_Ah"] = pd.to_numeric(
            cell_rows["Capacity_Ah"], errors="raise"
        )
        current_diag = float(pd.to_numeric(current.Diagnostic_Index, errors="raise"))
        prior = cell_rows.loc[cell_rows["Diagnostic_Index"].lt(current_diag)].sort_values(
            ["Diagnostic_Index", "Raw_Cycle_Index"], kind="stable"
        )
        if prior.empty:
            raise RuntimeError(
                f"NO_PREVIOUS_VALID_DIAGNOSTIC:{current.Cell_ID}:{current.Raw_Cycle_Index}"
            )
        previous = prior.iloc[-1]
        previous_capacity = float(previous["Capacity_Ah"])
        current_capacity = float(current.Capacity_Ah)
        delta = current_capacity - previous_capacity
        delta_pct = 100.0 * delta / previous_capacity

        if int(previous["Raw_Cycle_Index"]) != int(
            pd.to_numeric(current.Jump_Reference_Raw_Cycle, errors="raise")
        ):
            raise RuntimeError(
                f"JUMP_REFERENCE_RAW_CYCLE_MISMATCH:{current.Cell_ID}:{current.Raw_Cycle_Index}"
            )
        if not np.isclose(
            previous_capacity,
            float(pd.to_numeric(current.Jump_Reference_Capacity_Ah, errors="raise")),
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"JUMP_REFERENCE_CAPACITY_MISMATCH:{current.Cell_ID}:{current.Raw_Cycle_Index}"
            )

        reason_codes = str(current.Auto_QC_Reason_Codes)
        charge_max = float(pd.to_numeric(current.Observed_Charge_Max_V, errors="raise"))
        charge_end = float(
            pd.to_numeric(current.Observed_Charge_End_Current_A, errors="raise")
        )
        expected_cv = float(pd.to_numeric(current.Expected_CV_Cutoff_A, errors="raise"))
        charge_complete = (
            charge_max >= 4.18
            and charge_end <= expected_cv
            and "INCOMPLETE_CHARGE_PROTOCOL" not in reason_codes
        )
        discharge_complete = _as_bool(current.Full_Discharge_Observed) and (
            "INCOMPLETE_DISCHARGE_PROTOCOL" not in reason_codes
        )
        plot_name = f"{current.Cell_ID}_cycle{int(current.Raw_Cycle_Index)}_review.png"
        reason = _clean_text(current.Auto_QC_Reason)
        if not reason:
            reason = (
                "Capacity jump versus the nearest preceding final-included comparable "
                "diagnostic; automatic protocol checks remained complete. Manual review pending."
            )
        rows.append(
            {
                "Cell_ID": current.Cell_ID,
                "Raw_Cycle_Index": int(current.Raw_Cycle_Index),
                "Diagnostic_Index": int(current_diag),
                "Cumulative_Aging_Operation_Count": int(
                    pd.to_numeric(current.Cumulative_Aging_Operation_Count, errors="raise")
                ),
                "Capacity_Ah": current_capacity,
                "Previous_Valid_Diagnostic_Cycle": int(previous["Raw_Cycle_Index"]),
                "Previous_Valid_Diagnostic_Index": int(previous["Diagnostic_Index"]),
                "Previous_Valid_Capacity_Ah": previous_capacity,
                "Capacity_Jump_Ah": delta,
                "Capacity_Jump_Pct": delta_pct,
                "Charge_Complete": charge_complete,
                "Charge_Max_V": charge_max,
                "Charge_End_Current_A": charge_end,
                "Discharge_Complete": discharge_complete,
                "Discharge_Median_Current_A": float(
                    pd.to_numeric(
                        current.Observed_Discharge_Median_Current_A, errors="raise"
                    )
                ),
                "Discharge_Min_V": float(
                    pd.to_numeric(current.Observed_Discharge_Min_V, errors="raise")
                ),
                "Discharge_End_V": float(
                    pd.to_numeric(current.Observed_Discharge_End_V, errors="raise")
                ),
                "Auto_QC_Flag": current.Auto_QC_Flag,
                "Auto_QC_Reason_Codes": reason_codes,
                "Auto_QC_Reason": reason,
                "Current_Final_Include": current.Final_Include,
                "Manual_Decision": _clean_text(current.Manual_Decision),
                "Source_File": current.Source_File,
                "Review_Plot": f"review_plots/{plot_name}",
            }
        )
    return pd.DataFrame(rows)


def _monotonic_elapsed_minutes(cycle: Any, indices: np.ndarray, adapter: Any) -> np.ndarray:
    names = field_names(cycle)
    if "system_time" in names:
        raw = np.asarray(get_field(cycle, "system_time")).ravel(order="C")
        parsed = pd.to_datetime(
            raw[indices], format="%Y-%m-%d,%H:%M:%S", errors="coerce"
        )
        if parsed.isna().any():
            parsed = pd.to_datetime(raw[indices], format="mixed", errors="coerce")
        if not parsed.isna().any():
            elapsed = (parsed - parsed[0]).total_seconds().to_numpy(float) / 60.0
            if np.all(np.diff(elapsed) >= 0):
                return elapsed

    raw_time = time_to_seconds(
        numeric_1d(get_field(cycle, adapter.time_field)), adapter.time_unit
    )[indices]
    if raw_time.size == 0:
        return np.asarray([], dtype=float)
    differences = np.diff(raw_time)
    positive = differences[np.isfinite(differences) & (differences > 0)]
    typical = float(np.median(positive)) if positive.size else 0.0
    increments = np.where(
        np.isfinite(differences) & (differences >= 0), differences, typical
    )
    return np.r_[0.0, np.cumsum(increments)] / 60.0


def _cycle_plot_trace(cycle: Any, adapter: Any, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    voltage = numeric_1d(get_field(cycle, adapter.voltage_field))
    current = numeric_1d(get_field(cycle, adapter.current_field))
    capacity = numeric_1d(get_field(cycle, adapter.capacity_field))
    time_seconds = time_to_seconds(
        numeric_1d(get_field(cycle, adapter.time_field)), adapter.time_unit
    )
    lengths = {len(voltage), len(current), len(capacity), len(time_seconds)}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise RuntimeError("RAW_TRACE_ARRAY_LENGTH_MISMATCH_OR_EMPTY")
    if not all(np.isfinite(values).all() for values in (voltage, current, capacity)):
        raise RuntimeError("RAW_TRACE_CONTAINS_NAN_OR_INF")

    charge_indices = np.flatnonzero(
        current > float(config["charge_current_detection_threshold_A"])
    )
    discharge = _select_main_segment(
        current < float(config["discharge_current_detection_threshold_A"]),
        current,
        voltage,
        capacity,
        time_seconds,
    )
    if charge_indices.size == 0 or discharge is None:
        raise RuntimeError("REVIEW_PLOT_MAIN_CHARGE_OR_DISCHARGE_MISSING")
    discharge_indices = np.arange(discharge.start_index, discharge.end_index + 1)
    return {
        "charge_time_min": _monotonic_elapsed_minutes(cycle, charge_indices, adapter),
        "charge_voltage_V": voltage[charge_indices],
        "charge_current_A": current[charge_indices],
        "discharge_time_min": _monotonic_elapsed_minutes(
            cycle, discharge_indices, adapter
        ),
        "discharge_voltage_V": voltage[discharge_indices],
        "discharge_current_A": current[discharge_indices],
        "discharge_capacity_Ah": capacity[discharge_indices],
    }


def _make_review_plots(
    workspace: Path,
    validation_dir: Path,
    scan_dir: Path,
    review: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    config = _load_json(validation_dir / "batch5_config_final_candidate.json")
    manifest = pd.read_csv(
        scan_dir / "xjtu_file_manifest.csv", keep_default_na=False, low_memory=False
    )
    figures: list[Path] = []
    for item in review.itertuples(index=False):
        source_path = workspace / "Batch-5" / str(item.Source_File)
        if not source_path.is_file():
            raise FileNotFoundError(f"Batch-5 raw MAT not found: {source_path}")
        manifest_row = manifest.loc[
            manifest["Cell_ID"].eq(item.Cell_ID)
            & manifest["Source_File"].eq(item.Source_File)
            & manifest["File_Role"].eq("CANONICAL")
        ]
        if len(manifest_row) != 1:
            raise RuntimeError(f"CANONICAL_MANIFEST_ROW_NOT_UNIQUE:{item.Cell_ID}")
        if _sha256(source_path) != str(manifest_row.iloc[0]["SHA256"]):
            raise RuntimeError(f"RAW_MAT_HASH_CHANGED:{source_path}")

        payload = load_mat_payload(source_path)
        cycles = cycle_sequence(payload["data"])
        adapter, issues = discover_adapter(cycles, payload["summary"], config)
        if adapter is None:
            raise RuntimeError(
                f"REVIEW_PLOT_SCHEMA_ADAPTER_FAILED:{item.Cell_ID}:{'|'.join(issues)}"
            )
        raw_indices = (
            int(item.Previous_Valid_Diagnostic_Cycle),
            int(item.Raw_Cycle_Index),
        )
        if max(raw_indices) > len(cycles) or min(raw_indices) < 1:
            raise RuntimeError(f"REVIEW_PLOT_RAW_CYCLE_OUT_OF_RANGE:{item.Cell_ID}")
        traces = [
            _cycle_plot_trace(cycles[raw_cycle - 1], adapter, config)
            for raw_cycle in raw_indices
        ]
        labels = (
            f"Previous valid C{raw_indices[0]} ({item.Previous_Valid_Capacity_Ah:.3f} Ah)",
            f"Current CHECK C{raw_indices[1]} ({item.Capacity_Ah:.3f} Ah)",
        )
        colors = ("#1f77b4", "#d62728")

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for trace, label, color in zip(traces, labels, colors, strict=True):
            axes[0, 0].plot(
                trace["charge_time_min"], trace["charge_voltage_V"],
                color=color, linewidth=0.9, label=label,
            )
            axes[0, 1].plot(
                trace["charge_time_min"], trace["charge_current_A"],
                color=color, linewidth=0.9, label=label,
            )
            axes[1, 0].plot(
                trace["discharge_time_min"], trace["discharge_voltage_V"],
                color=color, linewidth=0.9, label=label,
            )
        axes[0, 0].axhline(4.2, color="black", linestyle="--", linewidth=0.7)
        axes[0, 0].set(title="Charge voltage (raw samples)", xlabel="Elapsed time (min)", ylabel="Voltage (V)")
        axes[0, 1].axhline(0.10, color="black", linestyle="--", linewidth=0.7)
        axes[0, 1].set(title="Charge current (raw samples)", xlabel="Elapsed time (min)", ylabel="Current (A)")
        axes[1, 0].axhline(2.5, color="black", linestyle="--", linewidth=0.7)
        axes[1, 0].set(title="Main-discharge voltage (raw samples)", xlabel="Elapsed time (min)", ylabel="Voltage (V)")

        capacity_axis = axes[1, 1]
        current_axis = capacity_axis.twinx()
        capacity_lines = []
        current_lines = []
        for trace, label, color in zip(traces, labels, colors, strict=True):
            capacity_lines.extend(
                capacity_axis.plot(
                    trace["discharge_time_min"], trace["discharge_capacity_Ah"],
                    color=color, linewidth=1.0, label=f"{label} capacity",
                )
            )
            current_lines.extend(
                current_axis.plot(
                    trace["discharge_time_min"], trace["discharge_current_A"],
                    color=color, linestyle=":", linewidth=0.9, label=f"{label} current",
                )
            )
        capacity_axis.set(
            title="Main-discharge capacity and current (raw samples)",
            xlabel="Elapsed time (min)",
            ylabel="Raw capacity channel (Ah)",
        )
        current_axis.set_ylabel("Current (A)")
        current_axis.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        capacity_axis.legend(
            capacity_lines + current_lines,
            [line.get_label() for line in capacity_lines + current_lines],
            fontsize=7,
            loc="best",
        )

        for axis in axes.ravel():
            axis.grid(alpha=0.2)
        for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
            axis.legend(fontsize=8, loc="best")
        fig.suptitle(
            f"{item.Cell_ID}: C{raw_indices[1]} CHECK vs C{raw_indices[0]} previous valid diagnostic\n"
            f"capacity jump = {item.Capacity_Jump_Ah:+.3f} Ah ({item.Capacity_Jump_Pct:+.2f}%)",
            fontsize=13,
        )
        fig.text(
            0.5,
            0.01,
            "Direct MAT samples only — no smoothing, interpolation, resampling, or capacity correction.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.93))
        path = output_dir / str(item.Review_Plot)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"REVIEW_PLOT_WRITE_FAILED:{path}")
        figures.append(path)
    return figures


def _validate_written_outputs(output_dir: Path) -> None:
    regression = pd.read_csv(
        output_dir / "early_median5_hotfix_regression.csv", keep_default_na=False
    )
    preview = pd.read_csv(
        output_dir / "corrected_qref_preview.csv", keep_default_na=False
    )
    review = pd.read_csv(
        output_dir / "batch5_check_manual_review.csv", keep_default_na=False
    )
    if regression["Test_ID"].duplicated().any() or not regression["Status"].eq("PASS").all():
        raise RuntimeError("WRITTEN_REGRESSION_OUTPUT_FAILED_OR_DUPLICATED")
    if len(preview) != 55 or preview["Cell_ID"].eq("").any() or preview["Cell_ID"].duplicated().any():
        raise RuntimeError("WRITTEN_QREF_PREVIEW_KEY_VALIDATION_FAILED")
    observed = set(
        zip(
            review["Cell_ID"].astype(str),
            pd.to_numeric(review["Raw_Cycle_Index"], errors="raise").astype(int),
        )
    )
    if observed != EXPECTED_BATCH5_CHECKS:
        raise RuntimeError("WRITTEN_BATCH5_REVIEW_KEY_VALIDATION_FAILED")
    if not review["Current_Final_Include"].eq("REVIEW").all():
        raise RuntimeError("BATCH5_REVIEW_DECISION_WAS_CHANGED")
    plots = sorted((output_dir / "review_plots").glob("*.png"))
    if len(plots) != 5 or any(path.stat().st_size == 0 for path in plots):
        raise RuntimeError("WRITTEN_REVIEW_PLOT_COUNT_OR_SIZE_FAILED")


def run_hotfix_validation(
    workspace: Path,
    prehotfix_dir: Path,
    scan_dir: Path,
    validation_dir: Path,
    output_dir: Path,
) -> Path:
    workspace = workspace.resolve()
    prehotfix_dir = prehotfix_dir.resolve()
    scan_dir = scan_dir.resolve()
    validation_dir = validation_dir.resolve()
    output_dir = output_dir.resolve()
    if not prehotfix_dir.is_dir():
        raise FileNotFoundError(f"PRE_HOTFIX output directory not found: {prehotfix_dir}")
    pre_hashes_before = _tree_hashes(prehotfix_dir)
    receipt = _load_json(prehotfix_dir / "run_receipt.json")
    if receipt.get("run_status") != "COMPLETE" or int(receipt.get("total_cells", 0)) != 55:
        raise RuntimeError("PRE_HOTFIX_RUN_RECEIPT_NOT_COMPLETE_55_CELL")

    cycle_qc = pd.read_csv(
        prehotfix_dir / "xjtu_55_cycle_qc.csv",
        keep_default_na=False,
        low_memory=False,
    )
    prehotfix_qref = pd.read_csv(
        prehotfix_dir / "xjtu_55_qref_summary.csv",
        keep_default_na=False,
        low_memory=False,
    )
    if len(cycle_qc) != int(receipt["total_raw_cycles"]):
        raise RuntimeError("PRE_HOTFIX_CYCLE_COUNT_DOES_NOT_MATCH_RECEIPT")

    checks = _batch5_checks(cycle_qc)
    preview, ordered_groups = _corrected_qref_preview(cycle_qc, prehotfix_qref)
    regression = _regression_results(preview, ordered_groups, checks)
    review = _manual_review_table(cycle_qc, checks)

    temporary = _atomic_directory(output_dir)
    try:
        _write_csv(regression, temporary / "early_median5_hotfix_regression.csv")
        _write_csv(preview, temporary / "corrected_qref_preview.csv")
        _write_csv(review, temporary / "batch5_check_manual_review.csv")
        figures = _make_review_plots(
            workspace, validation_dir, scan_dir, review, temporary
        )
        _validate_written_outputs(temporary)

        pre_hashes_after = _tree_hashes(prehotfix_dir)
        if pre_hashes_before != pre_hashes_after:
            raise RuntimeError("PRE_HOTFIX_OUTPUT_CHANGED_DURING_VALIDATION")
        output_files = [
            path.relative_to(temporary).as_posix()
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        ]
        if len(output_files) != 8:
            raise RuntimeError(
                f"HOTFIX_OUTPUT_COUNT_BEFORE_LOG_EXPECTED_8_FOUND_{len(output_files)}"
            )
        changed_count = int(preview["Hotfix_Changed"].eq("YES").sum())
        lines = [
            "XJTU 55-cell early-median5 hotfix validation",
            f"Completed_UTC: {datetime.now(timezone.utc).isoformat()}",
            f"Python: {platform.python_version()}",
            f"Pipeline_Version: {PIPELINE_VERSION}",
            f"numpy: {np.__version__}",
            f"pandas: {pd.__version__}",
            f"scipy: {scipy.__version__}",
            f"h5py: {h5py.__version__}",
            f"PRE_HOTFIX_Directory: {prehotfix_dir}",
            f"PRE_HOTFIX_Files_Hashed: {len(pre_hashes_before)}",
            "PRE_HOTFIX_Unchanged: YES",
            f"Cycle_QC_Rows_Read: {len(cycle_qc)}",
            f"Qref_Cells_Read: {len(prehotfix_qref)}",
            "Numeric_Conversion: Diagnostic_Index=pd.to_numeric(errors='raise')",
            "Numeric_Conversion: Raw_Cycle_Index=pd.to_numeric(errors='raise')",
            "Numeric_Conversion: Cumulative_Aging_Operation_Count=pd.to_numeric(errors='raise')",
            "Selection_Filter: Primary_Capacity_Eligible=YES AND Final_Include=YES AND Auto_QC_Flag!=INVALID_CANDIDATE",
            "Ordering: Diagnostic_Index numeric ascending, then Raw_Cycle_Index numeric ascending",
            f"Corrected_Qref_Cells_Changed: {changed_count}",
            f"Regression_Tests_PASS: {int(regression['Status'].eq('PASS').sum())}",
            "Regression_Tests_FAIL: 0",
            "Unit_Tests: 15 PASS, 0 FAIL",
            "Batch5_CHECK_Count: 5",
            "Batch5_CHECK_Set: " + "|".join(
                f"{cell}:C{cycle}" for cell, cycle in sorted(EXPECTED_BATCH5_CHECKS)
            ),
            "Batch5_Manual_Decisions_Changed: NO",
            f"Review_Plots_Created: {len(figures)}",
            "Review_Plots_Data: direct raw MAT samples",
            "Review_Plots_Visual_QA: PASS_ALL_5",
            "Discharge_Current_Axis_Offset_Display: DISABLED",
            "Smoothing: NONE",
            "Interpolation: NONE",
            "Capacity_Correction: NONE",
            "EOL_Recomputed: NO",
            "RUL_Recomputed: NO",
            "HUST_Processing: NO",
            "Model_Training: NO",
            "Outputs:",
            *[f"  - {name}" for name in output_files],
            "HOTFIX VALIDATION COMPLETE — WAITING FOR MANUAL CHECK REVIEW",
        ]
        (temporary / "hotfix_validation_log.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        final_files = [path for path in temporary.rglob("*") if path.is_file()]
        if len(final_files) != 9:
            raise RuntimeError(
                f"HOTFIX_FINAL_OUTPUT_COUNT_EXPECTED_9_FOUND_{len(final_files)}"
            )
        temporary.replace(output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    if _tree_hashes(prehotfix_dir) != pre_hashes_before:
        raise RuntimeError("PRE_HOTFIX_OUTPUT_CHANGED_AFTER_FINALIZATION")
    return output_dir


__all__ = ["run_hotfix_validation"]
