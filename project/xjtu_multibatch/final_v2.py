from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import scipy

from . import PIPELINE_VERSION
from .analysis import (
    QREF_DEFINITIONS,
    _atomic_directory,
    _coerce_numeric_order_columns,
    _finalize_qref_summary,
    _new_eol,
    _new_soh,
    _pairwise,
    _qref_row_from_observations,
    _rul,
    _run_regression_tests,
    _sha256,
    _write_csv,
)


MANUAL_REASON = (
    "Manual raw-curve review confirmed complete CC-CV charging, normal CV "
    "termination near 0.10 A, complete approximately 1C discharge to "
    "approximately 2.5 V, and no protocol truncation or abnormal voltage "
    "behavior. Capacity jump alone is not an exclusion criterion. Retain as "
    "a valid late-life capacity diagnostic."
)

EXPECTED_BATCH5_MANUAL_KEYS = {
    ("B5_RW_battery-4", 252),
    ("B5_RW_battery-5", 274),
    ("B5_RW_battery-6", 219),
    ("B5_RW_battery-7", 186),
    ("B5_RW_battery-8", 208),
}

EXPECTED_EOL_BOUNDS = {
    "B5_RW_battery-4": (220.0, 230.0, 252),
    "B5_RW_battery-5": (240.0, 250.0, 274),
    "B5_RW_battery-6": (190.0, 200.0, 219),
    "B5_RW_battery-7": (160.0, 170.0, 186),
    "B5_RW_battery-8": (170.0, 180.0, 197),
}

QREF_SOURCE_REGRESSIONS = {
    "B3_R2.5_battery-4": "3|8|13|18|23",
    "B4_R3_battery-4": "7|13|19|25|31",
    "B5_RW_battery-1": "21|32|43|54|65",
    "B6_Sim_satellite_battery-1": "7|13|19|24|30",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _tree_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _verify_prehotfix(prehotfix_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    receipt_path = prehotfix_dir / "run_receipt.json"
    receipt = _load_json(receipt_path)
    if receipt.get("run_status") != "COMPLETE":
        raise RuntimeError("PRE_HOTFIX_RUN_NOT_COMPLETE")
    if int(receipt.get("total_cells", 0)) != 55:
        raise RuntimeError("PRE_HOTFIX_CANONICAL_CELL_COUNT_NOT_55")
    for name, expected in receipt.get("output_hashes_excluding_receipt", {}).items():
        path = prehotfix_dir / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"PRE_HOTFIX_OUTPUT_HASH_MISMATCH:{name}")
    return receipt, _tree_hashes(prehotfix_dir)


def _verify_hotfix(hotfix_dir: Path) -> dict[str, str]:
    regressions = pd.read_csv(
        hotfix_dir / "early_median5_hotfix_regression.csv",
        keep_default_na=False,
        low_memory=False,
    )
    if regressions.empty or not regressions["Status"].eq("PASS").all():
        raise RuntimeError("HOTFIX_VALIDATION_REGRESSION_NOT_ALL_PASS")
    preview = pd.read_csv(
        hotfix_dir / "corrected_qref_preview.csv",
        keep_default_na=False,
        low_memory=False,
    )
    if len(preview) != 55 or preview["Cell_ID"].nunique() != 55:
        raise RuntimeError("HOTFIX_QREF_PREVIEW_NOT_55_UNIQUE_CELLS")
    review = pd.read_csv(
        hotfix_dir / "batch5_check_manual_review.csv",
        keep_default_na=False,
        low_memory=False,
    )
    observed = set(
        zip(
            review["Cell_ID"].astype(str),
            pd.to_numeric(review["Raw_Cycle_Index"], errors="raise").astype(int),
        )
    )
    if observed != EXPECTED_BATCH5_MANUAL_KEYS:
        raise RuntimeError("HOTFIX_BATCH5_REVIEW_KEY_SET_CHANGED")
    return _tree_hashes(hotfix_dir)


def _apply_batch5_manual_keep(
    cycle_qc: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, int], str]]:
    result = cycle_qc.copy()
    result["Raw_Cycle_Index"] = pd.to_numeric(
        result["Raw_Cycle_Index"], errors="raise"
    )
    auto_flags_before = result["Auto_QC_Flag"].copy()
    auto_reasons_before = result["Auto_QC_Reason_Codes"].copy()
    original_include: dict[tuple[str, int], str] = {}

    for cell, raw_cycle in sorted(EXPECTED_BATCH5_MANUAL_KEYS):
        mask = result["Cell_ID"].eq(cell) & result["Raw_Cycle_Index"].eq(raw_cycle)
        if int(mask.sum()) != 1:
            raise RuntimeError(f"MANUAL_REVIEW_KEY_NOT_UNIQUE:{cell}:{raw_cycle}")
        row = result.loc[mask].iloc[0]
        if row["Auto_QC_Flag"] != "CHECK":
            raise RuntimeError(f"MANUAL_REVIEW_AUTO_FLAG_NOT_CHECK:{cell}:{raw_cycle}")
        if row["Auto_QC_Reason_Codes"] != "CAPACITY_JUMP":
            raise RuntimeError(f"MANUAL_REVIEW_REASON_NOT_CAPACITY_JUMP:{cell}:{raw_cycle}")
        if row["Final_Include"] != "REVIEW":
            raise RuntimeError(f"MANUAL_REVIEW_ORIGINAL_INCLUDE_NOT_REVIEW:{cell}:{raw_cycle}")
        if row["Primary_Capacity_Eligible"] != "YES":
            raise RuntimeError(f"MANUAL_REVIEW_NOT_PRIMARY_ELIGIBLE:{cell}:{raw_cycle}")
        original_include[(cell, raw_cycle)] = str(row["Final_Include"])
        result.loc[mask, "Manual_Review_Status"] = "COMPLETED"
        result.loc[mask, "Manual_Decision"] = "KEEP"
        result.loc[mask, "Manual_Reason"] = MANUAL_REASON
        result.loc[mask, "Final_Include"] = "YES"
        result.loc[mask, "Final_Include_Basis"] = (
            "MANUAL_KEEP_PROTOCOL_COMPLETE_CAPACITY_JUMP_ONLY"
        )

    if not result["Auto_QC_Flag"].equals(auto_flags_before):
        raise RuntimeError("AUTO_QC_FLAG_CHANGED_DURING_MANUAL_REVIEW")
    if not result["Auto_QC_Reason_Codes"].equals(auto_reasons_before):
        raise RuntimeError("AUTO_QC_REASON_CHANGED_DURING_MANUAL_REVIEW")
    return result, original_include


def _align_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
    return result[columns]


def _new_batch_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[pd.to_numeric(frame["Batch_ID"], errors="raise").ne(1)].copy()


def _batch1_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[pd.to_numeric(frame["Batch_ID"], errors="raise").eq(1)].copy()


def _build_qref(
    cycle_qc: pd.DataFrame,
    prehotfix_qref: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    new_cycle_qc = _new_batch_rows(cycle_qc)
    new_rows = [
        _qref_row_from_observations(cell, int(rows["Batch_ID"].iloc[0]), rows)
        for cell, rows in new_cycle_qc.groupby("Cell_ID", sort=True)
    ]
    new_qref = _finalize_qref_summary(pd.DataFrame(new_rows))
    columns = list(prehotfix_qref.columns)
    qref = pd.concat(
        [
            _batch1_rows(prehotfix_qref)[columns],
            _align_columns(new_qref, columns),
        ],
        ignore_index=True,
        sort=False,
    )
    qref["Batch_ID"] = pd.to_numeric(qref["Batch_ID"], errors="raise")
    qref = qref.sort_values(["Batch_ID", "Cell_ID"], kind="stable").reset_index(
        drop=True
    )
    return qref, new_qref


def _build_derived_outputs(
    primary: pd.DataFrame,
    qref: pd.DataFrame,
    new_qref: pd.DataFrame,
    prehotfix_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    new_primary = _new_batch_rows(primary)

    pre_soh = pd.read_csv(
        prehotfix_dir / "xjtu_55_soh_labels_long.csv",
        keep_default_na=False,
        low_memory=False,
    )
    new_soh = _new_soh(new_primary, new_qref)
    soh = pd.concat(
        [
            _batch1_rows(pre_soh),
            _align_columns(new_soh, list(pre_soh.columns)),
        ],
        ignore_index=True,
        sort=False,
    )
    soh = soh.sort_values(
        ["Batch_ID", "Cell_ID", "Raw_Cycle_Index", "Qref_Definition"],
        kind="stable",
    ).reset_index(drop=True)

    pre_eol = pd.read_csv(
        prehotfix_dir / "xjtu_55_eol_sensitivity.csv",
        keep_default_na=False,
        low_memory=False,
    )
    new_eol, new_eol_lookup = _new_eol(new_primary, new_qref)
    eol = pd.concat(
        [
            _batch1_rows(pre_eol),
            _align_columns(new_eol, list(pre_eol.columns)),
        ],
        ignore_index=True,
        sort=False,
    )
    eol = eol.sort_values(
        ["Batch_ID", "Cell_ID", "Qref_Definition"], kind="stable"
    ).reset_index(drop=True)

    pre_pairwise = pd.read_csv(
        prehotfix_dir / "xjtu_55_eol_pairwise.csv",
        keep_default_na=False,
        low_memory=False,
    )
    recomputed_pairwise = _pairwise(eol, qref)
    pairwise = pd.concat(
        [
            _batch1_rows(pre_pairwise),
            _align_columns(
                _new_batch_rows(recomputed_pairwise), list(pre_pairwise.columns)
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    pairwise = pairwise.sort_values(
        ["Batch_ID", "Cell_ID", "Definition_A", "Definition_B"], kind="stable"
    ).reset_index(drop=True)

    pre_rul = pd.read_csv(
        prehotfix_dir / "xjtu_55_rul_labels.csv",
        keep_default_na=False,
        low_memory=False,
    )
    new_rul = _rul(new_primary, new_eol_lookup)
    rul = pd.concat(
        [
            _batch1_rows(pre_rul),
            _align_columns(new_rul, list(pre_rul.columns)),
        ],
        ignore_index=True,
        sort=False,
    )
    rul = rul.sort_values(
        ["Batch_ID", "Cell_ID", "Raw_Cycle_Index", "Qref_Definition"],
        kind="stable",
    ).reset_index(drop=True)
    return soh, eol, pairwise, rul


def _summaries(
    cycle_qc: pd.DataFrame,
    primary: pd.DataFrame,
    qref: pd.DataFrame,
    eol: pd.DataFrame,
    known_notes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    censoring = (
        eol.groupby(["Batch_ID", "Qref_Definition", "EOL_Status"])
        .size()
        .rename("Cell_Count")
        .reset_index()
    )

    batch_summary = (
        cycle_qc.groupby("Batch_ID")
        .agg(
            Canonical_Cell_Count=("Cell_ID", "nunique"),
            Total_Raw_Cycles=("Raw_Cycle_Index", "size"),
            PASS_Count=("Auto_QC_Flag", lambda values: int((values == "PASS").sum())),
            Invalid_Candidate_Count=(
                "Auto_QC_Flag",
                lambda values: int((values == "INVALID_CANDIDATE").sum()),
            ),
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
        .fillna({"Primary_Capacity_Observation_Count": 0, "Known_INFO_Count": 0})
    )
    for definition, column in (
        ("nominal", "Qref_nominal_Ah"),
        ("initial_test", "Qref_initial_test_Ah"),
        ("first_comparable", "Qref_first_comparable_Ah"),
        ("early_median5_comparable", "Qref_early_median5_comparable_Ah"),
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
            Invalid_Candidate_Count=(
                "Auto_QC_Flag",
                lambda values: int((values == "INVALID_CANDIDATE").sum()),
            ),
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
        .pivot(
            index=["Batch_ID", "Cell_ID"],
            columns="Qref_Definition",
            values="EOL_Status",
        )
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
    return censoring, batch_summary, cell_summary


def _selected_qref_source_keys(qref: pd.DataFrame) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for item in qref.itertuples(index=False):
        values = item._asdict()
        cell = str(values["Cell_ID"])
        for column in (
            "Qref_Initial_Test_Source_Cycle",
            "Qref_First_Comparable_Source_Cycle",
        ):
            value = pd.to_numeric(values.get(column), errors="coerce")
            if np.isfinite(value):
                keys.add((cell, int(value)))
        for value in str(
            values.get("Qref_Early_Median5_Comparable_Source_Cycles", "")
        ).split("|"):
            if value:
                keys.add((cell, int(value)))
    return keys


def _manual_audit(
    cycle_qc: pd.DataFrame,
    original_include: Mapping[tuple[str, int], str],
    primary: pd.DataFrame,
    qref: pd.DataFrame,
    soh: pd.DataFrame,
    eol: pd.DataFrame,
    rul: pd.DataFrame,
    frozen_dir: Path,
    hotfix_dir: Path,
    prehotfix_cycle_qc_sha256: str,
) -> pd.DataFrame:
    cycle_lookup = cycle_qc.set_index(["Cell_ID", "Raw_Cycle_Index"])
    primary_keys = {
        (str(row.Cell_ID), int(row.Raw_Cycle_Index))
        for row in primary.itertuples(index=False)
    }
    soh_keys = {
        (str(row.Cell_ID), int(row.Raw_Cycle_Index))
        for row in soh.itertuples(index=False)
    }
    rul_keys = {
        (str(row.Cell_ID), int(row.Raw_Cycle_Index))
        for row in rul.itertuples(index=False)
    }
    selected_qref_keys = _selected_qref_source_keys(qref)
    eol_cells = set(eol["Cell_ID"].astype(str))
    review = pd.read_csv(
        hotfix_dir / "batch5_check_manual_review.csv",
        keep_default_na=False,
        low_memory=False,
    ).set_index(["Cell_ID", "Raw_Cycle_Index"])
    frozen_manual = pd.read_csv(
        frozen_dir / "xjtu_manual_qc_review.csv",
        keep_default_na=False,
        low_memory=False,
    )

    rows: list[dict[str, Any]] = []
    for item in frozen_manual.itertuples(index=False):
        key = (str(item.Cell_ID), int(item.Raw_Cycle_Index))
        cycle = cycle_lookup.loc[key]
        rows.append(
            {
                "Batch_ID": 1,
                "Cell_ID": key[0],
                "Source_File": cycle["Source_File"],
                "Raw_Cycle_Index": key[1],
                "Aging_Cycle_Index": cycle.get("Aging_Cycle_Index", np.nan),
                "Aging_Operation_Index": cycle.get("Aging_Operation_Index", np.nan),
                "Diagnostic_Index": cycle.get("Diagnostic_Index", np.nan),
                "Cumulative_Aging_Operation_Count": cycle.get(
                    "Cumulative_Aging_Operation_Count", np.nan
                ),
                "Capacity_Ah": cycle["Capacity_Ah"],
                "Auto_QC_Flag": item.Auto_QC_Flag,
                "Auto_QC_Reason_Codes": item.Auto_QC_Reason_Codes,
                "Original_Final_Include": item.Final_Include,
                "Manual_Review_Status": item.Manual_Review_Status,
                "Manual_Decision": item.Manual_Decision,
                "Manual_Reason": item.Manual_Reason,
                "Final_Include": item.Final_Include,
                "Final_Include_Basis": item.Final_Include_Basis,
                "Primary_Capacity_Eligible": cycle["Primary_Capacity_Eligible"],
                "Charge_Complete": item.Charge_Max_V >= 4.18,
                "Charge_Max_V": item.Charge_Max_V,
                "Charge_End_Current_A": item.Charge_End_Current_A,
                "Discharge_Complete": item.Discharge_Complete,
                "Discharge_Median_Current_A": cycle.get(
                    "Discharge_Median_Current_A", np.nan
                ),
                "Discharge_Min_V": item.Discharge_Min_V,
                "Discharge_End_V": item.Discharge_End_V,
                "Previous_Valid_Diagnostic_Cycle": np.nan,
                "Previous_Valid_Capacity_Ah": np.nan,
                "Capacity_Jump_Ah": np.nan,
                "Capacity_Jump_Pct": np.nan,
                "Eligible_For_Qref_Source_Selection": "NO",
                "Selected_As_Qref_Source": "NO",
                "Included_In_SOH": "NO",
                "Included_In_EOL_Censoring": "NO",
                "Included_In_RUL_Bound_Analysis": "NO",
                "Review_Evidence": "FROZEN_BATCH1_MANUAL_REVIEW",
                "Source_Auto_QC_SHA256": item.Source_Auto_QC_SHA256,
            }
        )

    for key in sorted(EXPECTED_BATCH5_MANUAL_KEYS):
        cycle = cycle_lookup.loc[key]
        evidence = review.loc[key]
        rows.append(
            {
                "Batch_ID": 5,
                "Cell_ID": key[0],
                "Source_File": cycle["Source_File"],
                "Raw_Cycle_Index": key[1],
                "Aging_Cycle_Index": cycle.get("Aging_Cycle_Index", np.nan),
                "Aging_Operation_Index": cycle.get("Aging_Operation_Index", np.nan),
                "Diagnostic_Index": cycle["Diagnostic_Index"],
                "Cumulative_Aging_Operation_Count": cycle[
                    "Cumulative_Aging_Operation_Count"
                ],
                "Capacity_Ah": cycle["Capacity_Ah"],
                "Auto_QC_Flag": cycle["Auto_QC_Flag"],
                "Auto_QC_Reason_Codes": cycle["Auto_QC_Reason_Codes"],
                "Original_Final_Include": original_include[key],
                "Manual_Review_Status": cycle["Manual_Review_Status"],
                "Manual_Decision": cycle["Manual_Decision"],
                "Manual_Reason": cycle["Manual_Reason"],
                "Final_Include": cycle["Final_Include"],
                "Final_Include_Basis": cycle["Final_Include_Basis"],
                "Primary_Capacity_Eligible": cycle["Primary_Capacity_Eligible"],
                "Charge_Complete": evidence["Charge_Complete"],
                "Charge_Max_V": evidence["Charge_Max_V"],
                "Charge_End_Current_A": evidence["Charge_End_Current_A"],
                "Discharge_Complete": evidence["Discharge_Complete"],
                "Discharge_Median_Current_A": evidence[
                    "Discharge_Median_Current_A"
                ],
                "Discharge_Min_V": evidence["Discharge_Min_V"],
                "Discharge_End_V": evidence["Discharge_End_V"],
                "Previous_Valid_Diagnostic_Cycle": evidence[
                    "Previous_Valid_Diagnostic_Cycle"
                ],
                "Previous_Valid_Capacity_Ah": evidence[
                    "Previous_Valid_Capacity_Ah"
                ],
                "Capacity_Jump_Ah": evidence["Capacity_Jump_Ah"],
                "Capacity_Jump_Pct": evidence["Capacity_Jump_Pct"],
                "Eligible_For_Qref_Source_Selection": (
                    "YES" if key in primary_keys else "NO"
                ),
                "Selected_As_Qref_Source": (
                    "YES" if key in selected_qref_keys else "NO"
                ),
                "Included_In_SOH": "YES" if key in soh_keys else "NO",
                "Included_In_EOL_Censoring": (
                    "YES" if key[0] in eol_cells and key in primary_keys else "NO"
                ),
                "Included_In_RUL_Bound_Analysis": (
                    "YES" if key in rul_keys else "NO"
                ),
                "Review_Evidence": str(hotfix_dir / str(evidence["Review_Plot"])),
                "Source_Auto_QC_SHA256": prehotfix_cycle_qc_sha256,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Batch_ID", "Cell_ID", "Raw_Cycle_Index"], kind="stable"
    ).reset_index(drop=True)


def _frames_match(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> bool:
    try:
        left_sorted = left.sort_values(keys, kind="stable").reset_index(drop=True)
        right_sorted = right.sort_values(keys, kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(
            left_sorted[right_sorted.columns],
            right_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=0.0,
        )
        return True
    except (AssertionError, KeyError):
        return False


def _v2_regression_tests(
    base_tests: pd.DataFrame,
    cycle_qc: pd.DataFrame,
    primary: pd.DataFrame,
    qref: pd.DataFrame,
    soh: pd.DataFrame,
    eol: pd.DataFrame,
    pairwise: pd.DataFrame,
    rul: pd.DataFrame,
    audit: pd.DataFrame,
    prehotfix_dir: Path,
    prehotfix_hashes_unchanged: bool,
    hotfix_regression_pass: bool,
) -> pd.DataFrame:
    records = base_tests.to_dict("records")

    def add(
        test_id: str,
        scope: str,
        expected: Any,
        observed: Any,
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

    for cell, expected_sources in QREF_SOURCE_REGRESSIONS.items():
        rows = qref.loc[qref["Cell_ID"].eq(cell)]
        observed = (
            str(rows.iloc[0]["Qref_Early_Median5_Comparable_Source_Cycles"])
            if len(rows) == 1
            else f"rows={len(rows)}"
        )
        add(
            f"FINAL_V2_{cell.upper()}_EARLY_MEDIAN5_SOURCES",
            cell,
            expected_sources,
            observed,
            observed == expected_sources,
            "Numeric early-median5 source-cycle regression failed",
        )

    for cell, expected_median in (
        ("B5_RW_battery-1", 1.923),
        ("B6_Sim_satellite_battery-1", 1.942),
    ):
        value = pd.to_numeric(
            qref.loc[qref["Cell_ID"].eq(cell), "Qref_early_median5_comparable_Ah"],
            errors="coerce",
        )
        observed = float(value.iloc[0]) if len(value) == 1 else np.nan
        add(
            f"FINAL_V2_{cell.upper()}_EARLY_MEDIAN5_VALUE",
            cell,
            f"{expected_median:.3f}±0.002 Ah",
            observed,
            np.isfinite(observed)
            and bool(np.isclose(observed, expected_median, atol=0.002, rtol=0.0)),
            "Early-median5 capacity regression failed",
        )

    b6 = primary.loc[
        primary["Cell_ID"].eq("B6_Sim_satellite_battery-1")
    ].copy()
    b6 = _coerce_numeric_order_columns(b6).sort_values(
        ["Diagnostic_Index", "Raw_Cycle_Index"], kind="stable"
    ).head(5)
    observed_b6 = pd.to_numeric(b6["Capacity_Ah"], errors="raise").to_numpy(float)
    expected_b6 = np.asarray([1.922, 1.928, 1.942, 1.953, 1.962])
    add(
        "FINAL_V2_BATCH6_FIRST5_CAPACITY_VECTOR",
        "B6_Sim_satellite_battery-1",
        "1.922|1.928|1.942|1.953|1.962",
        "|".join(f"{value:.3f}" for value in observed_b6),
        observed_b6.shape == expected_b6.shape
        and bool(np.allclose(observed_b6, expected_b6, atol=0.002, rtol=0.0)),
        "Batch-6 first-five capacity vector changed",
    )

    reviewed = cycle_qc.loc[
        [
            (str(row.Cell_ID), int(row.Raw_Cycle_Index))
            in EXPECTED_BATCH5_MANUAL_KEYS
            for row in cycle_qc.itertuples(index=False)
        ]
    ]
    observed_keys = {
        (str(row.Cell_ID), int(row.Raw_Cycle_Index))
        for row in reviewed.itertuples(index=False)
    }
    add(
        "FINAL_V2_BATCH5_MANUAL_KEEP_KEY_SET",
        "Batch-5 manual review",
        5,
        len(observed_keys),
        observed_keys == EXPECTED_BATCH5_MANUAL_KEYS,
        "Manual KEEP key set changed",
    )
    manual_state_ok = (
        reviewed["Auto_QC_Flag"].eq("CHECK").all()
        and reviewed["Auto_QC_Reason_Codes"].eq("CAPACITY_JUMP").all()
        and reviewed["Manual_Decision"].eq("KEEP").all()
        and reviewed["Final_Include"].eq("YES").all()
        and reviewed["Primary_Capacity_Eligible"].eq("YES").all()
    )
    add(
        "FINAL_V2_BATCH5_MANUAL_KEEP_AUDIT_STATE",
        "Batch-5 manual review",
        "CHECK|CAPACITY_JUMP|KEEP|YES|eligible",
        "PASS" if manual_state_ok else "FAIL",
        manual_state_ok,
        "Automatic QC was changed or manual inclusion was not applied",
    )

    batch5_primary_count = int(
        primary.loc[pd.to_numeric(primary["Batch_ID"], errors="raise").eq(5)].shape[0]
    )
    add(
        "FINAL_V2_BATCH5_PRIMARY_COUNT_173",
        "Batch-5",
        173,
        batch5_primary_count,
        batch5_primary_count == 173,
        "The five KEEP decisions did not increase Batch-5 primary count from 168 to 173",
    )

    for cell, (lower, upper, first_below_raw) in EXPECTED_EOL_BOUNDS.items():
        for definition in ("nominal", "fixed_absolute_1p6Ah"):
            rows = eol.loc[
                eol["Cell_ID"].eq(cell)
                & eol["Qref_Definition"].eq(definition)
            ]
            if len(rows) == 1:
                row = rows.iloc[0]
                observed = (
                    f"{row['EOL_Status']};"
                    f"{row['EOL_Lower_Bound_Aging_Cycle']}<EOL<="
                    f"{row['EOL_Upper_Bound_Aging_Cycle']};"
                    f"first_below_raw={row['EOL_First_Below_Raw_Cycle']}"
                )
                passed = (
                    row["EOL_Status"] == "INTERVAL_CENSORED"
                    and pd.to_numeric(
                        row["EOL_Lower_Bound_Aging_Cycle"], errors="coerce"
                    )
                    == lower
                    and pd.to_numeric(
                        row["EOL_Upper_Bound_Aging_Cycle"], errors="coerce"
                    )
                    == upper
                    and pd.to_numeric(
                        row["EOL_First_Below_Raw_Cycle"], errors="coerce"
                    )
                    == first_below_raw
                    and not np.isfinite(
                        pd.to_numeric(row["EOL_Exact_Aging_Cycle"], errors="coerce")
                    )
                )
            else:
                observed = f"rows={len(rows)}"
                passed = False
            add(
                f"FINAL_V2_{cell.upper()}_{definition.upper()}_EOL",
                cell,
                f"INTERVAL_CENSORED;{lower:.0f}<EOL<={upper:.0f};first_below_raw={first_below_raw}",
                observed,
                passed,
                "Batch-5 nominal/fixed EOL regression failed",
            )

    manual_soh = soh.loc[
        [
            (str(row.Cell_ID), int(row.Raw_Cycle_Index))
            in EXPECTED_BATCH5_MANUAL_KEYS
            for row in soh.itertuples(index=False)
        ]
    ]
    manual_rul = rul.loc[
        [
            (str(row.Cell_ID), int(row.Raw_Cycle_Index))
            in EXPECTED_BATCH5_MANUAL_KEYS
            for row in rul.itertuples(index=False)
        ]
    ]
    add(
        "FINAL_V2_MANUAL_KEEP_ENTERED_SOH",
        "five Batch-5 KEEP cycles",
        "15 rows (3 available Qrefs per cycle)",
        len(manual_soh),
        len(manual_soh) == 15
        and manual_soh["Final_Include"].eq("YES").all(),
        "Manual KEEP cycles missing from SOH",
    )
    add(
        "FINAL_V2_MANUAL_KEEP_ENTERED_RUL_BOUNDS",
        "five Batch-5 KEEP cycles",
        "20 rows (4 Qref definitions per cycle)",
        len(manual_rul),
        len(manual_rul) == 20,
        "Manual KEEP cycles missing from RUL-bound analysis",
    )

    audit_b5 = audit.loc[pd.to_numeric(audit["Batch_ID"], errors="raise").eq(5)]
    audit_ok = (
        len(audit_b5) == 5
        and audit_b5["Auto_QC_Flag"].eq("CHECK").all()
        and audit_b5["Auto_QC_Reason_Codes"].eq("CAPACITY_JUMP").all()
        and audit_b5["Manual_Decision"].eq("KEEP").all()
        and audit_b5["Final_Include"].eq("YES").all()
        and audit_b5["Included_In_SOH"].eq("YES").all()
        and audit_b5["Included_In_EOL_Censoring"].eq("YES").all()
        and audit_b5["Included_In_RUL_Bound_Analysis"].eq("YES").all()
    )
    add(
        "FINAL_V2_MANUAL_AUDIT_COMPLETE",
        "xjtu_55_manual_qc_audit",
        "5 Batch-5 KEEP rows with original CHECK retained and all inclusion traces YES",
        "PASS" if audit_ok else "FAIL",
        audit_ok,
        "Manual audit is incomplete",
    )

    for name, keys in (
        ("QREF", ["Batch_ID", "Cell_ID"]),
        ("SOH", ["Batch_ID", "Cell_ID", "Raw_Cycle_Index", "Qref_Definition"]),
        ("EOL", ["Batch_ID", "Cell_ID", "Qref_Definition"]),
        ("PAIRWISE", ["Batch_ID", "Cell_ID", "Definition_A", "Definition_B"]),
        ("RUL", ["Batch_ID", "Cell_ID", "Raw_Cycle_Index", "Qref_Definition"]),
    ):
        current = {
            "QREF": qref,
            "SOH": soh,
            "EOL": eol,
            "PAIRWISE": pairwise,
            "RUL": rul,
        }[name]
        previous = pd.read_csv(
            prehotfix_dir
            / {
                "QREF": "xjtu_55_qref_summary.csv",
                "SOH": "xjtu_55_soh_labels_long.csv",
                "EOL": "xjtu_55_eol_sensitivity.csv",
                "PAIRWISE": "xjtu_55_eol_pairwise.csv",
                "RUL": "xjtu_55_rul_labels.csv",
            }[name],
            keep_default_na=False,
            low_memory=False,
        )
        matched = _frames_match(_batch1_rows(current), _batch1_rows(previous), keys)
        add(
            f"FINAL_V2_BATCH1_FROZEN_{name}_UNCHANGED",
            "Batch-1 frozen",
            "identical Batch-1 rows",
            "PASS" if matched else "FAIL",
            matched,
            f"Frozen Batch-1 {name} results changed",
        )

    add(
        "FINAL_V2_PRE_HOTFIX_HASHES_UNCHANGED",
        "PRE_HOTFIX outputs",
        "all hashes unchanged",
        "PASS" if prehotfix_hashes_unchanged else "FAIL",
        prehotfix_hashes_unchanged,
        "PRE_HOTFIX output changed",
    )
    add(
        "FINAL_V2_HOTFIX_REGRESSIONS_ALL_PASS",
        "hotfix validation",
        "all PASS",
        "PASS" if hotfix_regression_pass else "FAIL",
        hotfix_regression_pass,
        "Hotfix regression evidence is not all PASS",
    )
    return pd.DataFrame(records)


def _validate_written_outputs(output_dir: Path, prehotfix_dir: Path) -> None:
    qref = pd.read_csv(output_dir / "xjtu_55_qref_summary.csv", keep_default_na=False)
    soh = pd.read_csv(
        output_dir / "xjtu_55_soh_labels_long.csv",
        keep_default_na=False,
        low_memory=False,
    )
    eol = pd.read_csv(
        output_dir / "xjtu_55_eol_sensitivity.csv", keep_default_na=False
    )
    rul = pd.read_csv(
        output_dir / "xjtu_55_rul_labels.csv",
        keep_default_na=False,
        low_memory=False,
    )
    batch = pd.read_csv(
        output_dir / "xjtu_55_batch_summary.csv", keep_default_na=False
    )
    tests = pd.read_csv(
        output_dir / "xjtu_55_run_regression_tests.csv", keep_default_na=False
    )
    audit = pd.read_csv(
        output_dir / "xjtu_55_manual_qc_audit.csv", keep_default_na=False
    )
    if len(qref) != 55 or qref["Cell_ID"].nunique() != 55:
        raise RuntimeError("FINAL_V2_WRITTEN_QREF_NOT_55_UNIQUE_CELLS")
    if not tests["Status"].eq("PASS").all():
        raise RuntimeError("FINAL_V2_WRITTEN_REGRESSION_HAS_FAILURE")
    b5_count = int(
        batch.loc[
            pd.to_numeric(batch["Batch_ID"], errors="raise").eq(5),
            "Primary_Capacity_Observation_Count",
        ].iloc[0]
    )
    if b5_count != 173:
        raise RuntimeError("FINAL_V2_WRITTEN_BATCH5_PRIMARY_COUNT_NOT_173")
    censored_eol = eol["EOL_Status"].isin(
        ["INTERVAL_CENSORED", "RIGHT_CENSORED"]
    )
    point_columns = ["EOL_Raw_Cycle", "EOL_Aging_Cycle", "EOL_Exact_Aging_Cycle"]
    if (
        eol.loc[censored_eol, point_columns]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .any()
        .any()
    ):
        raise RuntimeError("FINAL_V2_WRITTEN_CENSORED_EOL_HAS_POINT")
    censored_rul = rul["EOL_Status"].isin(
        ["INTERVAL_CENSORED", "RIGHT_CENSORED"]
    )
    if pd.to_numeric(rul.loc[censored_rul, "RUL_Cycles"], errors="coerce").notna().any():
        raise RuntimeError("FINAL_V2_WRITTEN_CENSORED_RUL_HAS_POINT")
    if len(audit) != 7 or int(pd.to_numeric(audit["Batch_ID"]).eq(5).sum()) != 5:
        raise RuntimeError("FINAL_V2_WRITTEN_MANUAL_AUDIT_ROW_COUNT_FAILED")

    pre_qref = pd.read_csv(
        prehotfix_dir / "xjtu_55_qref_summary.csv",
        keep_default_na=False,
        low_memory=False,
    )
    if not _frames_match(
        _batch1_rows(qref), _batch1_rows(pre_qref), ["Batch_ID", "Cell_ID"]
    ):
        raise RuntimeError("FINAL_V2_WRITTEN_BATCH1_QREF_CHANGED")
    manual_keys = set(
        zip(
            audit.loc[pd.to_numeric(audit["Batch_ID"]).eq(5), "Cell_ID"].astype(str),
            pd.to_numeric(
                audit.loc[pd.to_numeric(audit["Batch_ID"]).eq(5), "Raw_Cycle_Index"],
                errors="raise",
            ).astype(int),
        )
    )
    if manual_keys != EXPECTED_BATCH5_MANUAL_KEYS:
        raise RuntimeError("FINAL_V2_WRITTEN_MANUAL_AUDIT_KEYS_CHANGED")


def run_final_v2(
    workspace: Path,
    prehotfix_dir: Path,
    hotfix_dir: Path,
    validation_dir: Path,
    output_dir: Path,
) -> Path:
    workspace = workspace.resolve()
    prehotfix_dir = prehotfix_dir.resolve()
    hotfix_dir = hotfix_dir.resolve()
    validation_dir = validation_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    receipt, pre_hashes_before = _verify_prehotfix(prehotfix_dir)
    hotfix_hashes_before = _verify_hotfix(hotfix_dir)
    hotfix_regressions = pd.read_csv(
        hotfix_dir / "early_median5_hotfix_regression.csv",
        keep_default_na=False,
    )

    cycle_qc = pd.read_csv(
        prehotfix_dir / "xjtu_55_cycle_qc.csv",
        keep_default_na=False,
        low_memory=False,
    )
    cycle_qc, original_include = _apply_batch5_manual_keep(cycle_qc)
    for column in (
        "Raw_Cycle_Index",
        "Diagnostic_Index",
        "Cumulative_Aging_Operation_Count",
    ):
        comparable = cycle_qc["Primary_Capacity_Eligible"].eq("YES")
        cycle_qc.loc[comparable, column] = pd.to_numeric(
            cycle_qc.loc[comparable, column], errors="raise"
        )

    primary = cycle_qc.loc[
        cycle_qc["Primary_Capacity_Eligible"].eq("YES")
        & cycle_qc["Final_Include"].eq("YES")
    ].copy()
    primary = _coerce_numeric_order_columns(primary)
    primary["Capacity_Ah"] = pd.to_numeric(primary["Capacity_Ah"], errors="raise")
    primary["Batch_ID"] = pd.to_numeric(primary["Batch_ID"], errors="raise")
    primary = primary.sort_values(
        [
            "Batch_ID",
            "Cell_ID",
            "Cumulative_Aging_Operation_Count",
            "Raw_Cycle_Index",
        ],
        kind="stable",
    ).reset_index(drop=True)

    prehotfix_qref = pd.read_csv(
        prehotfix_dir / "xjtu_55_qref_summary.csv",
        keep_default_na=False,
        low_memory=False,
    )
    qref, new_qref = _build_qref(cycle_qc, prehotfix_qref)
    soh, eol, pairwise, rul = _build_derived_outputs(
        primary, qref, new_qref, prehotfix_dir
    )
    known_notes = pd.read_csv(
        prehotfix_dir / "xjtu_55_known_data_quality_notes.csv",
        keep_default_na=False,
        low_memory=False,
    )
    censoring, batch_summary, cell_summary = _summaries(
        cycle_qc, primary, qref, eol, known_notes
    )

    validation_receipt = _load_json(validation_dir / "validation_receipt.json")
    frozen_dir = Path(str(validation_receipt["frozen_batch1_directory"]))
    manifest = pd.read_csv(
        prehotfix_dir / "xjtu_55_file_manifest.csv",
        keep_default_na=False,
        low_memory=False,
    )
    validated = pd.read_csv(
        validation_dir / "xjtu_protocol_inventory_validated.csv",
        keep_default_na=False,
        low_memory=False,
    )
    audit = _manual_audit(
        cycle_qc,
        original_include,
        primary,
        qref,
        soh,
        eol,
        rul,
        frozen_dir,
        hotfix_dir,
        receipt["output_hashes_excluding_receipt"]["xjtu_55_cycle_qc.csv"],
    )

    base_tests = _run_regression_tests(
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
    tests = _v2_regression_tests(
        base_tests,
        cycle_qc,
        primary,
        qref,
        soh,
        eol,
        pairwise,
        rul,
        audit,
        prehotfix_dir,
        _tree_hashes(prehotfix_dir) == pre_hashes_before,
        hotfix_regressions["Status"].eq("PASS").all(),
    )
    failures = tests.loc[tests["Status"].ne("PASS")]
    if not failures.empty:
        raise RuntimeError(
            "FINAL_V2_REGRESSION_FAILED:"
            + "|".join(failures["Test_ID"].astype(str))
        )

    temp_dir = _atomic_directory(output_dir)
    try:
        outputs = {
            "xjtu_55_qref_summary.csv": qref,
            "xjtu_55_soh_labels_long.csv": soh,
            "xjtu_55_eol_sensitivity.csv": eol,
            "xjtu_55_eol_pairwise.csv": pairwise,
            "xjtu_55_rul_labels.csv": rul,
            "xjtu_55_censoring_summary.csv": censoring,
            "xjtu_55_batch_summary.csv": batch_summary,
            "xjtu_55_cell_summary.csv": cell_summary,
            "xjtu_55_run_regression_tests.csv": tests,
            "xjtu_55_manual_qc_audit.csv": audit,
        }
        for name, frame in outputs.items():
            _write_csv(frame, temp_dir / name)
        _validate_written_outputs(temp_dir, prehotfix_dir)

        if _tree_hashes(prehotfix_dir) != pre_hashes_before:
            raise RuntimeError("PRE_HOTFIX_OUTPUT_CHANGED_DURING_FINAL_V2")
        if _tree_hashes(hotfix_dir) != hotfix_hashes_before:
            raise RuntimeError("HOTFIX_VALIDATION_OUTPUT_CHANGED_DURING_FINAL_V2")

        output_hashes = {
            path.name: _sha256(path)
            for path in sorted(temp_dir.glob("*.csv"))
        }
        eol_b5_lines: list[str] = []
        for cell in sorted(EXPECTED_EOL_BOUNDS):
            row = eol.loc[
                eol["Cell_ID"].eq(cell)
                & eol["Qref_Definition"].eq("nominal")
            ].iloc[0]
            eol_b5_lines.append(
                f"{cell}:{row['EOL_Status']}:"
                f"{row['EOL_Lower_Bound_Aging_Cycle']}<EOL<="
                f"{row['EOL_Upper_Bound_Aging_Cycle']}:"
                f"first_below_raw={row['EOL_First_Below_Raw_Cycle']}"
            )
        log_lines = [
            f"Pipeline_Version={PIPELINE_VERSION}",
            f"Python_Version={sys.version.replace(chr(10), ' ')}",
            f"Platform={platform.platform()}",
            f"NumPy_Version={np.__version__}",
            f"pandas_Version={pd.__version__}",
            f"SciPy_Version={scipy.__version__}",
            f"Run_Timestamp_UTC={datetime.now(timezone.utc).isoformat()}",
            f"PRE_HOTFIX_Directory={prehotfix_dir}",
            f"HOTFIX_Validation_Directory={hotfix_dir}",
            f"Validation_Directory={validation_dir}",
            f"Frozen_Batch1_Directory={frozen_dir}",
            "PRE_HOTFIX_Unchanged=YES",
            "HOTFIX_Validation_Outputs_Unchanged=YES",
            "Raw_MAT_Files_Modified=NO",
            "BATCH1_SOURCE=FROZEN_REFERENCE",
            "BATCH1_DERIVED_RESULTS_UNCHANGED=YES",
            "NO_INTERPOLATION=TRUE",
            "NO_SMOOTHING=TRUE",
            "MEASURED_CAPACITY_CORRECTION=FALSE",
            "RAW_CYCLE_RENUMBERING=FALSE",
            "Numeric_Sort_Diagnostic_Index=pd.to_numeric(errors='raise')",
            "Numeric_Sort_Raw_Cycle_Index=pd.to_numeric(errors='raise')",
            "Numeric_Sort_Cumulative_Aging_Operation_Count=pd.to_numeric(errors='raise')",
            "Early_Median5_Order=NUMERIC_DIAGNOSTIC_INDEX_ASCENDING",
            "Batch5_Manual_Review_Count=5",
            "Batch5_Manual_Decision=KEEP",
            "Batch5_Auto_QC_Preserved=CHECK|CAPACITY_JUMP",
            "Batch5_PRE_HOTFIX_Primary_Observation_Count=168",
            "Batch5_FINAL_V2_Primary_Observation_Count=173",
            f"Canonical_Cell_Count={qref['Cell_ID'].nunique()}",
            f"Total_Primary_Capacity_Observations={len(primary)}",
            "Primary_Observations_By_Batch="
            + "|".join(
                f"{int(key)}:{int(value)}"
                for key, value in primary.groupby("Batch_ID").size().items()
            ),
            "Batch5_Nominal_EOL=" + "|".join(eol_b5_lines),
            f"Regression_Tests_PASS={int(tests['Status'].eq('PASS').sum())}",
            "Regression_Tests_FAIL=0",
            "RIGHT_CENSORED_POINT_EOL_CREATED=NO",
            "INTERVAL_CENSORED_POINT_EOL_CREATED=NO",
            "CENSORED_POINT_RUL_CREATED=NO",
            "EOL_RUL_RECOMPUTED=YES",
            "HUST_Processing=NO",
            "Machine_Learning_Trained=NO",
            "Output_SHA256:",
            *[f"  {name}={value}" for name, value in sorted(output_hashes.items())],
            "XJTU 55-CELL FINAL V2 COMPLETE",
        ]
        (temp_dir / "processing_log.txt").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8"
        )
        expected_names = set(outputs) | {"processing_log.txt"}
        actual_names = {path.name for path in temp_dir.iterdir() if path.is_file()}
        if actual_names != expected_names:
            raise RuntimeError(
                f"FINAL_V2_OUTPUT_FILE_SET_MISMATCH:{sorted(actual_names)}"
            )
        temp_dir.rename(output_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    if _tree_hashes(prehotfix_dir) != pre_hashes_before:
        raise RuntimeError("PRE_HOTFIX_OUTPUT_CHANGED_AFTER_FINAL_V2")
    if _tree_hashes(hotfix_dir) != hotfix_hashes_before:
        raise RuntimeError("HOTFIX_VALIDATION_OUTPUT_CHANGED_AFTER_FINAL_V2")
    return output_dir


__all__ = [
    "EXPECTED_BATCH5_MANUAL_KEYS",
    "MANUAL_REASON",
    "_apply_batch5_manual_keep",
    "run_final_v2",
]
