"""Full raw extraction, QC, and distribution audit for frozen SOH features."""

from __future__ import annotations

import csv
import gc
import hashlib
import inspect
import json
import os
import platform
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hust_repro.safe_pickle import load_hust_pickle

from .audits import (
    complete_case_audit,
    distribution_by_xjtu_batch,
    distribution_summary,
    feature_values_sha256,
    missing_mask_sha256,
    outlier_candidates,
    protocol_shortcut_audit,
    target_association_audit,
)
from .config import (
    ALLOWED_EXTRACTION_STATUSES,
    FEATURE_NAMES,
    FEATURE_UNITS,
    HUST_EXPECTED_OBSERVATIONS,
    NEAR_TARGET_PROXY_ABS_SPEARMAN,
    OUTLIER_MAD_MULTIPLIER,
    PIPELINE_VERSION,
    RANDOM_SEED,
    SYSTEMATIC_MISSINGNESS_RATE_THRESHOLD,
    XJTU_EXPECTED_OBSERVATIONS,
)
from .extractors import (
    ExtractionResult,
    extract_hust_features,
    extract_xjtu_features,
    invalid_extraction_result,
    matrix_row,
    provenance_rows,
)


COMMON_QREFS = ("nominal", "first_comparable", "early_median5_comparable")

CSV_OUTPUTS = (
    "feature_matrix_xjtu_v1.csv",
    "feature_matrix_hust_v1.csv",
    "feature_extraction_provenance_xjtu.csv",
    "feature_extraction_provenance_hust.csv",
    "feature_missingness_summary.csv",
    "feature_distribution_summary.csv",
    "feature_distribution_by_xjtu_batch.csv",
    "feature_protocol_shortcut_audit.csv",
    "feature_target_association_audit.csv",
    "feature_complete_case_audit.csv",
    "feature_extraction_review_candidates.csv",
    "feature_target_join_audit.csv",
    "feature_extraction_regression_tests.csv",
)

PROVENANCE_COLUMNS = (
    "Dataset",
    "Batch_ID",
    "Cell_ID",
    "Raw_Cycle_Index",
    "Feature_Name",
    "Raw_Source_File",
    "Physical_Stage",
    "Stage_Start_Index",
    "Stage_End_Index",
    "Stage_Start_Time_s",
    "Stage_End_Time_s",
    "Raw_Sample_Count",
    "Valid_Sample_Count",
    "Feature_Value",
    "Feature_Unit",
    "Extraction_Status",
    "Extraction_Reason_Code",
    "Raw_Time_Unit",
    "Time_Conversion_Factor_To_Seconds",
    "Pre_Discharge_State_Type",
    "Extraction_Detail",
)


@dataclass(frozen=True)
class Paths:
    workspace: Path
    xjtu_final: Path
    hust_final: Path
    design: Path
    xjtu_manifest: Path
    hust_manifest: Path
    hust_fingerprint: Path

    @classmethod
    def from_workspace(cls, workspace: Path) -> "Paths":
        outputs = workspace / "outputs"
        return cls(
            workspace=workspace,
            xjtu_final=outputs / "xjtu_55_final_v2",
            hust_final=outputs / "hust_77_final_v1",
            design=outputs / "soh_ml_design_v1",
            xjtu_manifest=outputs / "xjtu_55_final_v1" / "xjtu_55_file_manifest.csv",
            hust_manifest=outputs / "hust_scan_v1" / "hust_file_manifest.csv",
            hust_fingerprint=outputs / "hust_scan_v1" / "hust_protocol_fingerprint.csv",
        )


class FeatureExtractionError(RuntimeError):
    pass


def _require(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FeatureExtractionError("MISSING_REQUIRED_INPUTS:" + "|".join(missing))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_snapshot(directory: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(directory).as_posix(): (path.stat().st_size, _sha256_file(path))
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
    }


def _snapshot_digest(snapshot: dict[str, tuple[int, str]]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _raw_stat_snapshot(paths: Iterable[Path]) -> dict[str, tuple[int, int]]:
    return {
        str(path.resolve()): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(set(paths))
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.columns.duplicated().any():
        raise FeatureExtractionError(f"DUPLICATE_OUTPUT_COLUMNS:{path.name}")
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


def _numeric_int(series: pd.Series, label: str) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors="raise")
    except Exception as exc:
        raise FeatureExtractionError(f"NONNUMERIC_{label}:{exc}") from exc
    if not np.all(np.isfinite(values.to_numpy(dtype=float))):
        raise FeatureExtractionError(f"NONFINITE_{label}")
    if not np.allclose(values, np.round(values), rtol=0, atol=0):
        raise FeatureExtractionError(f"NONINTEGER_{label}")
    return values.astype("int64")


def _load_design_and_keys(paths: Paths) -> dict[str, Any]:
    feature_plan_path = paths.design / "ml_harmonized_feature_plan.csv"
    xjtu_folds_path = paths.design / "ml_cell_fold_assignments_xjtu.csv"
    hust_folds_path = paths.design / "ml_cell_fold_assignments_hust.csv"
    xjtu_labels_path = paths.xjtu_final / "xjtu_55_soh_labels_long.csv"
    hust_labels_path = paths.hust_final / "hust_77_soh_labels_long.csv"
    _require(
        [
            feature_plan_path,
            xjtu_folds_path,
            hust_folds_path,
            xjtu_labels_path,
            hust_labels_path,
            paths.xjtu_manifest,
            paths.hust_manifest,
            paths.hust_fingerprint,
        ]
    )
    plan = pd.read_csv(feature_plan_path, low_memory=False)
    frozen_features = tuple(
        plan.loc[
            plan["Feature_Set_ID"].eq("FEATURE_SET_A_PHYSICS_ONLY")
        ]
        .sort_values("Feature_Order")["Feature_Name"]
        .astype(str)
    )
    if frozen_features != FEATURE_NAMES:
        raise FeatureExtractionError(
            f"FROZEN_FEATURE_SET_MISMATCH:design={frozen_features}:code={FEATURE_NAMES}"
        )
    xfold = pd.read_csv(xjtu_folds_path, low_memory=False)
    hfold = pd.read_csv(hust_folds_path, low_memory=False)
    for frame, name in ((xfold, "XJTU"), (hfold, "HUST")):
        frame["Fold_ID"] = _numeric_int(frame["Fold_ID"], f"{name}_Fold_ID")
        if frame["Cell_ID"].duplicated().any():
            raise FeatureExtractionError(f"DUPLICATE_{name}_FOLD_CELL")

    # Only physical keys and Qref definition are loaded before extraction.
    # Target capacity/SOH columns are deliberately omitted until all raw
    # feature values have been frozen.
    xlabels_keys = pd.read_csv(
        xjtu_labels_path,
        usecols=["Batch_ID", "Cell_ID", "Raw_Cycle_Index", "Qref_Definition"],
        low_memory=False,
    )
    hlabels_keys = pd.read_csv(
        hust_labels_path,
        usecols=["Cell_ID", "Raw_Cycle_Index", "Qref_Definition"],
        low_memory=False,
    )
    xlabels_keys["Raw_Cycle_Index"] = _numeric_int(
        xlabels_keys["Raw_Cycle_Index"], "XJTU_Raw_Cycle_Index"
    )
    hlabels_keys["Raw_Cycle_Index"] = _numeric_int(
        hlabels_keys["Raw_Cycle_Index"], "HUST_Raw_Cycle_Index"
    )
    xkeys = xlabels_keys.loc[
        xlabels_keys["Qref_Definition"].eq("nominal"),
        ["Batch_ID", "Cell_ID", "Raw_Cycle_Index"],
    ].reset_index(drop=True)
    hkeys = hlabels_keys.loc[
        hlabels_keys["Qref_Definition"].eq("nominal"),
        ["Cell_ID", "Raw_Cycle_Index"],
    ].reset_index(drop=True)
    if xkeys.duplicated(["Batch_ID", "Cell_ID", "Raw_Cycle_Index"]).any():
        raise FeatureExtractionError("DUPLICATE_XJTU_FROZEN_KEY")
    if hkeys.duplicated(["Cell_ID", "Raw_Cycle_Index"]).any():
        raise FeatureExtractionError("DUPLICATE_HUST_FROZEN_KEY")
    if len(xkeys) != XJTU_EXPECTED_OBSERVATIONS:
        raise FeatureExtractionError(f"XJTU_KEY_COUNT:{len(xkeys)}")
    if len(hkeys) != HUST_EXPECTED_OBSERVATIONS:
        raise FeatureExtractionError(f"HUST_KEY_COUNT:{len(hkeys)}")
    return {
        "plan": plan,
        "xfold": xfold,
        "hfold": hfold,
        "xlabels_keys": xlabels_keys,
        "hlabels_keys": hlabels_keys,
        "xkeys": xkeys,
        "hkeys": hkeys,
        "xlabels_path": xjtu_labels_path,
        "hlabels_path": hust_labels_path,
    }


def _raw_file_maps(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    xmanifest = pd.read_csv(paths.xjtu_manifest, low_memory=False)
    xcanonical = xmanifest.loc[
        xmanifest["Included_In_Analysis"].astype(str).str.upper().eq("YES")
    ].copy()
    if len(xcanonical) != 55 or xcanonical["Cell_ID"].nunique() != 55:
        raise FeatureExtractionError("XJTU_CANONICAL_FILE_COUNT_NOT_55")
    xcanonical["Raw_Path"] = xcanonical.apply(
        lambda row: str(
            (
                paths.workspace / "Batch-1" / str(row["Source_File"])
                if int(row["Batch_ID"]) == 1
                else paths.workspace
                / f"Batch-{int(row['Batch_ID'])}"
                / str(row["Source_File"])
            ).resolve()
        ),
        axis=1,
    )
    hmanifest = pd.read_csv(paths.hust_manifest, low_memory=False)
    hcanonical = hmanifest.loc[
        hmanifest["Included_In_Analysis"].astype(str).str.upper().eq("YES")
    ].copy()
    if len(hcanonical) != 77 or hcanonical["Cell_ID"].nunique() != 77:
        raise FeatureExtractionError("HUST_CANONICAL_FILE_COUNT_NOT_77")
    hcanonical["Raw_Path"] = hcanonical["File_Name"].map(
        lambda name: str((paths.workspace / "HUST 77" / str(name)).resolve())
    )
    for frame, dataset in ((xcanonical, "XJTU"), (hcanonical, "HUST")):
        for row in frame.itertuples(index=False):
            path = Path(row.Raw_Path)
            if not path.is_file():
                raise FeatureExtractionError(f"{dataset}_RAW_FILE_MISSING:{path}")
            if path.stat().st_size != int(row.File_Size_Bytes):
                raise FeatureExtractionError(f"{dataset}_RAW_SIZE_MISMATCH:{path.name}")
    return xcanonical, hcanonical


def _representative_plan(
    xkeys: pd.DataFrame, hkeys: pd.DataFrame, paths: Paths
) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], str]]:
    x_requested = {
        1: ("2C_battery-1", 2),
        2: ("B2_3C_battery-3", 2),
        3: ("B3_R2.5_battery-4", 3),
        4: ("B4_R3_battery-4", 7),
        5: ("B5_RW_battery-1", 21),
        6: ("B6_Sim_satellite_battery-1", 7),
    }
    xplan: dict[tuple[str, int], str] = {}
    for batch, (cell, cycle) in x_requested.items():
        exists = (
            xkeys["Batch_ID"].eq(batch)
            & xkeys["Cell_ID"].eq(cell)
            & xkeys["Raw_Cycle_Index"].eq(cycle)
        ).any()
        if not exists:
            raise FeatureExtractionError(f"XJTU_REPRESENTATIVE_KEY_MISSING:{batch}:{cell}:{cycle}")
        xplan[(cell, cycle)] = f"XJTU_BATCH_{batch}_VALIDATED_PROTOCOL_REPRESENTATIVE"

    requested_cells = pd.read_csv(
        paths.workspace / "outputs" / "hust_scan_v1" / "representative_validation_plan.csv"
    )["Cell_ID"].astype(str).tolist()
    hplan: dict[tuple[str, int], str] = {}
    for position, cell in enumerate(requested_cells):
        cycles = sorted(
            hkeys.loc[hkeys["Cell_ID"].eq(cell), "Raw_Cycle_Index"].astype(int).tolist()
        )
        if not cycles:
            raise FeatureExtractionError(f"HUST_REPRESENTATIVE_CELL_MISSING:{cell}")
        fractions = (0.10, 0.50, 0.90)
        fraction = fractions[position % len(fractions)]
        index = min(len(cycles) - 1, int(round((len(cycles) - 1) * fraction)))
        cycle = int(cycles[index])
        hplan[(cell, cycle)] = (
            f"HUST_PREVIOUSLY_VALIDATED_REPRESENTATIVE_CELL;LIFE_POSITION={fraction:.2f}"
        )
    return xplan, hplan


def _save_stage_plot(
    result: ExtractionResult,
    path: Path,
    title: str,
) -> None:
    arrays = result.arrays_for_plot
    x = arrays["plot_elapsed_seconds"]
    current = arrays["current_A"]
    voltage = arrays["voltage_V"]
    boundaries = result.boundaries
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)
    axes[0].plot(x, current, color="#1F2937", linewidth=0.8)
    axes[1].plot(x, voltage, color="#0F766E", linewidth=0.8)
    axes[0].set_ylabel("Current (A)")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].set_xlabel("Sequential elapsed time (s)")
    axes[0].axhline(0, color="#9CA3AF", linewidth=0.6)

    def span(start_name: str, end_name: str, color: str, label: str) -> None:
        start = boundaries.get(start_name)
        end = boundaries.get(end_name)
        if start is not None and end is not None:
            for axis in axes:
                axis.axvspan(x[start], x[end], color=color, alpha=0.14, label=label)

    span("cc_start_index", "cc_end_index", "#2563EB", "Final pre-CV CC")
    span("cv_start_index", "cv_end_index", "#F59E0B", "CV")
    markers = (
        ("charge_end_index", "#111827", ":", "Charge end"),
        ("pre_discharge_index", "#7C3AED", "--", "Pre-discharge point"),
        ("discharge_start_index", "#DC2626", "-.", "Discharge start"),
    )
    for key, color, style, label in markers:
        index = boundaries.get(key)
        if index is not None:
            for axis in axes:
                axis.axvline(x[index], color=color, linestyle=style, linewidth=1.0, label=label)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=5,
            fontsize=8,
        )
    fig.suptitle(title, fontsize=11, y=0.935)
    axes[0].grid(True, color="#E5E7EB", linewidth=0.5)
    axes[1].grid(True, color="#E5E7EB", linewidth=0.5)
    fig.text(
        0.01,
        0.01,
        "Raw samples only; no smoothing or interpolation.",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


MATRIX_COMMON_COLUMNS = (
    "Dataset",
    "Cell_ID",
    "Raw_Cycle_Index",
    "Fold_ID",
    "Physical_Observation_Key",
)

MATRIX_AUDIT_COLUMNS = (
    "Feature_Set_B_Raw_Cycle_Index",
    "Feature_Extraction_Flag",
    "Feature_Extraction_Reason_Codes",
    "Feature_Set_A_Valid_Count",
    "Feature_Set_A_Missing_Count",
    "Feature_Set_A_Complete",
    "Pre_Discharge_State_Type",
)

REVIEW_COLUMNS = (
    "Dataset",
    "Batch_ID",
    "Protocol_Fingerprint_ID",
    "Cell_ID",
    "Raw_Cycle_Index",
    "Feature_Name",
    "Feature_Value",
    "Extraction_Status",
    "Candidate_Type",
    "Reason_Code",
    "Review_Lower_Bound",
    "Review_Upper_Bound",
    "Automatic_Exclusion",
    "Raw_Source_File",
)


def _matrix_columns(dataset: str) -> list[str]:
    columns = list(MATRIX_COMMON_COLUMNS)
    if dataset == "XJTU":
        columns.insert(1, "Batch_ID")
    return columns + list(FEATURE_NAMES) + list(MATRIX_AUDIT_COLUMNS)


class MissingnessAccumulator:
    """Accumulate feature-status summaries without retaining long provenance."""

    def __init__(self) -> None:
        self.counts: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
        self.affected_cells: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
        self.group_cells: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    def observe(
        self,
        *,
        dataset: str,
        batch_id: int | str | None,
        protocol_fingerprint: str,
        cell_id: str,
        result: ExtractionResult,
    ) -> None:
        groups: list[tuple[str, str]] = [("DATASET", "ALL")]
        if dataset == "XJTU":
            groups.append(("BATCH_ID", str(int(batch_id))))
        else:
            groups.append(("PROTOCOL_FINGERPRINT", protocol_fingerprint))
        for level, group_value in groups:
            group_key = (dataset, level, group_value)
            self.group_cells[group_key].add(cell_id)
            for feature_name in FEATURE_NAMES:
                feature = result.features[feature_name]
                key = (dataset, level, group_value, feature_name)
                self.counts[key][feature.status] += 1
                if not np.isfinite(feature.value):
                    self.counts[key]["VALUE_MISSING"] += 1
                if feature.status != "PASS" or not np.isfinite(feature.value):
                    self.affected_cells[key].add(cell_id)

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for key in sorted(self.counts):
            dataset, level, group_value, feature_name = key
            counter = self.counts[key]
            total = sum(counter.get(status, 0) for status in ALLOWED_EXTRACTION_STATUSES)
            missing_count = counter.get("VALUE_MISSING", 0)
            rows.append(
                {
                    "Dataset": dataset,
                    "Audit_Level": level,
                    "Group_Value": group_value,
                    "Batch_ID": group_value if level == "BATCH_ID" else "",
                    "Protocol_Fingerprint_ID": (
                        group_value if level == "PROTOCOL_FINGERPRINT" else ""
                    ),
                    "Feature_Name": feature_name,
                    "Total_Expected_Observations": total,
                    "PASS_Count": counter.get("PASS", 0),
                    "CHECK_Count": counter.get("CHECK", 0),
                    "MISSING_Count": counter.get("MISSING", 0),
                    "INVALID_EXTRACTION_Count": counter.get("INVALID_EXTRACTION", 0),
                    "Value_Missing_Count": missing_count,
                    "Missing_Rate": missing_count / total if total else np.nan,
                    "Affected_Cell_Count": len(self.affected_cells[key]),
                    "Total_Cell_Count": len(self.group_cells[(dataset, level, group_value)]),
                    "Systematic_Missingness_Threshold": SYSTEMATIC_MISSINGNESS_RATE_THRESHOLD,
                    "Systematic_Missingness_Flag": (
                        "REVIEW"
                        if total and missing_count / total > SYSTEMATIC_MISSINGNESS_RATE_THRESHOLD
                        else "NO"
                    ),
                    "Automatic_Observation_Exclusion": "NO",
                }
            )
        return pd.DataFrame(rows)


@dataclass
class RuntimeAudit:
    missingness: MissingnessAccumulator
    feature_key_count: int = 0
    post_discharge_boundary_violations: int = 0
    time_unit_counts: Counter[tuple[str, str, float]] | None = None
    extraction_exception_count: int = 0
    nonpass_feature_count: int = 0

    def __post_init__(self) -> None:
        if self.time_unit_counts is None:
            self.time_unit_counts = Counter()

    def observe(
        self,
        *,
        dataset: str,
        batch_id: int | str | None,
        protocol_fingerprint: str,
        cell_id: str,
        result: ExtractionResult,
    ) -> None:
        self.missingness.observe(
            dataset=dataset,
            batch_id=batch_id,
            protocol_fingerprint=protocol_fingerprint,
            cell_id=cell_id,
            result=result,
        )
        self.feature_key_count += len(FEATURE_NAMES)
        self.nonpass_feature_count += sum(
            result.features[name].status != "PASS" for name in FEATURE_NAMES
        )
        assert self.time_unit_counts is not None
        self.time_unit_counts[
            (
                dataset,
                result.raw_time_unit,
                float(result.time_conversion_factor_to_seconds),
            )
        ] += 1
        discharge_start = result.boundaries.get("discharge_start_index")
        if discharge_start is not None:
            for feature in result.features.values():
                if (
                    feature.stage_end_index is not None
                    and feature.stage_end_index >= discharge_start
                ):
                    self.post_discharge_boundary_violations += 1


def _write_observation(
    *,
    matrix_writer: csv.DictWriter,
    provenance_writer: csv.DictWriter,
    review_writer: csv.DictWriter,
    audit: RuntimeAudit,
    result: ExtractionResult,
    dataset: str,
    batch_id: int | str | None,
    protocol_fingerprint: str,
    cell_id: str,
    raw_cycle_index: int,
    fold_id: int,
    raw_source_file: str,
) -> None:
    matrix_writer.writerow(
        matrix_row(
            result=result,
            dataset=dataset,
            batch_id=batch_id,
            cell_id=cell_id,
            raw_cycle_index=raw_cycle_index,
            fold_id=fold_id,
        )
    )
    long_rows = provenance_rows(
        result=result,
        dataset=dataset,
        batch_id=batch_id,
        cell_id=cell_id,
        raw_cycle_index=raw_cycle_index,
        raw_source_file=raw_source_file,
    )
    provenance_writer.writerows(long_rows)
    for row in long_rows:
        if row["Extraction_Status"] != "PASS":
            review_writer.writerow(
                {
                    "Dataset": dataset,
                    "Batch_ID": batch_id if batch_id is not None else "",
                    "Protocol_Fingerprint_ID": protocol_fingerprint,
                    "Cell_ID": cell_id,
                    "Raw_Cycle_Index": raw_cycle_index,
                    "Feature_Name": row["Feature_Name"],
                    "Feature_Value": row["Feature_Value"],
                    "Extraction_Status": row["Extraction_Status"],
                    "Candidate_Type": "FEATURE_EXTRACTION_STATUS_REVIEW",
                    "Reason_Code": row["Extraction_Reason_Code"],
                    "Review_Lower_Bound": "",
                    "Review_Upper_Bound": "",
                    "Automatic_Exclusion": "NO",
                    "Raw_Source_File": raw_source_file,
                }
            )
    audit.observe(
        dataset=dataset,
        batch_id=batch_id,
        protocol_fingerprint=protocol_fingerprint,
        cell_id=cell_id,
        result=result,
    )


def _representative_boundary_row(
    *,
    dataset: str,
    batch_id: int | str | None,
    cell_id: str,
    cycle: int,
    selection_reason: str,
    result: ExtractionResult,
    plot_file: str,
) -> dict[str, Any]:
    boundary = result.boundaries
    ordered_values = [
        boundary.get("charge_start_index"),
        boundary.get("cc_start_index"),
        boundary.get("cc_end_index"),
        boundary.get("cv_start_index"),
        boundary.get("cv_end_index"),
        boundary.get("charge_end_index"),
        boundary.get("pre_discharge_index"),
        boundary.get("discharge_start_index"),
    ]
    complete = all(value is not None for value in ordered_values)
    order_valid = bool(
        complete
        and ordered_values[0] <= ordered_values[1]
        and ordered_values[1] <= ordered_values[2]
        and ordered_values[2] < ordered_values[3]
        and ordered_values[3] <= ordered_values[4]
        and ordered_values[4] <= ordered_values[5]
        and ordered_values[5] <= ordered_values[6]
        and ordered_values[6] < ordered_values[7]
    )
    expected_time = ("MINUTES", 60.0) if dataset == "XJTU" else ("SECONDS", 1.0)
    time_valid = (
        result.raw_time_unit == expected_time[0]
        and result.time_conversion_factor_to_seconds == expected_time[1]
    )
    cv_feature_names = (
        "CV_Current_Taper_Slope_A_per_s",
        "CV_Duration_s",
        "CV_End_Current_A",
        "CV_Median_Current_A",
    )
    cv_values_valid = all(np.isfinite(result.features[name].value) for name in cv_feature_names)
    return {
        "Test_Category": "REPRESENTATIVE_RAW_BOUNDARY",
        "Dataset": dataset,
        "Batch_ID": batch_id if batch_id is not None else "",
        "Cell_ID": cell_id,
        "Raw_Cycle_Index": cycle,
        "Selection_Reason": selection_reason,
        "Raw_Time_Unit": result.raw_time_unit,
        "Time_Conversion_Factor_To_Seconds": result.time_conversion_factor_to_seconds,
        "Charge_Start_Index": boundary.get("charge_start_index"),
        "CC_Start_Index": boundary.get("cc_start_index"),
        "CC_End_Index": boundary.get("cc_end_index"),
        "CV_Start_Index": boundary.get("cv_start_index"),
        "CV_End_Index": boundary.get("cv_end_index"),
        "Charge_End_Index": boundary.get("charge_end_index"),
        "Pre_Discharge_Index": boundary.get("pre_discharge_index"),
        "Discharge_Start_Index": boundary.get("discharge_start_index"),
        "Boundary_Order_Valid": "YES" if order_valid else "NO",
        "CV_Features_Extracted": "YES" if cv_values_valid else "NO",
        "Time_Unit_Validated": "YES" if time_valid else "NO",
        "Plot_File": plot_file,
        "Status": "PASS" if order_valid and cv_values_valid and time_valid else "FAIL",
        "Details": result.overall_reason_codes,
    }


def _safe_plot_name(dataset: str, batch_id: Any, cell_id: str, cycle: int) -> str:
    batch = f"_batch{batch_id}" if batch_id not in (None, "") else ""
    safe_cell = "".join(character if character.isalnum() or character in "-_" else "_" for character in cell_id)
    return f"{dataset.lower()}{batch}_{safe_cell}_cycle_{cycle}.png"


def _process_xjtu(
    *,
    paths: Paths,
    staging: Path,
    xcanonical: pd.DataFrame,
    xkeys: pd.DataFrame,
    xfold: pd.DataFrame,
    representative_plan: dict[tuple[str, int], str],
    audit: RuntimeAudit,
    review_writer: csv.DictWriter,
) -> tuple[Path, list[dict[str, Any]]]:
    from scipy.io import loadmat

    matrix_path = staging / "_feature_matrix_xjtu_unsorted.csv"
    provenance_path = staging / "feature_extraction_provenance_xjtu.csv"
    matrix_columns = _matrix_columns("XJTU")
    fold_map = dict(zip(xfold["Cell_ID"].astype(str), xfold["Fold_ID"].astype(int)))
    source_map = xcanonical.set_index("Cell_ID").to_dict("index")
    representative_rows: list[dict[str, Any]] = []
    plots_dir = staging / "review_plots"

    with matrix_path.open("w", newline="", encoding="utf-8-sig") as matrix_handle, provenance_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as provenance_handle:
        matrix_writer = csv.DictWriter(matrix_handle, fieldnames=matrix_columns)
        provenance_writer = csv.DictWriter(provenance_handle, fieldnames=PROVENANCE_COLUMNS)
        matrix_writer.writeheader()
        provenance_writer.writeheader()
        grouped = xkeys.groupby(["Batch_ID", "Cell_ID"], sort=True)
        for file_number, ((batch_id, cell_id), key_group) in enumerate(grouped, start=1):
            batch_id = int(batch_id)
            cell_id = str(cell_id)
            source = source_map[cell_id]
            raw_path = Path(source["Raw_Path"])
            print(
                f"[XJTU {file_number:02d}/55] batch={batch_id} cell={cell_id} "
                f"observations={len(key_group)}",
                flush=True,
            )
            try:
                loaded = loadmat(
                    raw_path,
                    squeeze_me=True,
                    struct_as_record=False,
                    variable_names=["data"],
                )
                data = np.atleast_1d(loaded["data"]).ravel(order="C")
            except Exception as exc:
                loaded = None
                data = np.asarray([], dtype=object)
                load_reason = f"RAW_FILE_LOAD_ERROR:{type(exc).__name__}:{exc}"
            else:
                load_reason = ""
            for cycle in key_group["Raw_Cycle_Index"].astype(int):
                try:
                    if load_reason:
                        raise RuntimeError(load_reason)
                    if cycle < 1 or cycle > len(data):
                        raise IndexError(f"RAW_CYCLE_NOT_FOUND:{cycle}:data_length={len(data)}")
                    raw_cycle = data[cycle - 1]
                    result = extract_xjtu_features(
                        relative_time_min=getattr(raw_cycle, "relative_time_min"),
                        current_A=getattr(raw_cycle, "current_A"),
                        voltage_V=getattr(raw_cycle, "voltage_V"),
                    )
                except Exception as exc:
                    audit.extraction_exception_count += 1
                    reason = f"XJTU_EXTRACTION_ERROR:{type(exc).__name__}:{exc}"
                    result = invalid_extraction_result(
                        reason=reason,
                        raw_time_unit="MINUTES",
                        time_conversion_factor_to_seconds=60.0,
                    )
                _write_observation(
                    matrix_writer=matrix_writer,
                    provenance_writer=provenance_writer,
                    review_writer=review_writer,
                    audit=audit,
                    result=result,
                    dataset="XJTU",
                    batch_id=batch_id,
                    protocol_fingerprint="",
                    cell_id=cell_id,
                    raw_cycle_index=int(cycle),
                    fold_id=fold_map[cell_id],
                    raw_source_file=str(source["Source_File"]),
                )
                representative_key = (cell_id, int(cycle))
                if representative_key in representative_plan:
                    plot_name = _safe_plot_name("XJTU", batch_id, cell_id, int(cycle))
                    if result.arrays_for_plot["time_seconds"].size:
                        _save_stage_plot(
                            result,
                            plots_dir / plot_name,
                            f"XJTU Batch {batch_id} | {cell_id} | Raw Cycle {cycle}",
                        )
                    representative_rows.append(
                        _representative_boundary_row(
                            dataset="XJTU",
                            batch_id=batch_id,
                            cell_id=cell_id,
                            cycle=int(cycle),
                            selection_reason=representative_plan[representative_key],
                            result=result,
                            plot_file=f"review_plots/{plot_name}",
                        )
                    )
            del data
            if loaded is not None:
                del loaded
            gc.collect()
    return matrix_path, representative_rows


def _process_hust(
    *,
    paths: Paths,
    staging: Path,
    hcanonical: pd.DataFrame,
    hkeys: pd.DataFrame,
    hfold: pd.DataFrame,
    representative_plan: dict[tuple[str, int], str],
    audit: RuntimeAudit,
    review_writer: csv.DictWriter,
) -> tuple[Path, list[dict[str, Any]]]:
    matrix_path = staging / "_feature_matrix_hust_unsorted.csv"
    provenance_path = staging / "feature_extraction_provenance_hust.csv"
    matrix_columns = _matrix_columns("HUST")
    fold_map = dict(zip(hfold["Cell_ID"].astype(str), hfold["Fold_ID"].astype(int)))
    source_map = hcanonical.set_index("Cell_ID").to_dict("index")
    fingerprint_frame = pd.read_csv(paths.hust_fingerprint, low_memory=False)
    if fingerprint_frame["Cell_ID"].duplicated().any():
        raise FeatureExtractionError("DUPLICATE_HUST_PROTOCOL_FINGERPRINT_CELL")
    fingerprint_map = dict(
        zip(
            fingerprint_frame["Cell_ID"].astype(str),
            fingerprint_frame["Protocol_Fingerprint_ID"].astype(str),
        )
    )
    representative_rows: list[dict[str, Any]] = []
    plots_dir = staging / "review_plots"

    with matrix_path.open("w", newline="", encoding="utf-8-sig") as matrix_handle, provenance_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as provenance_handle:
        matrix_writer = csv.DictWriter(matrix_handle, fieldnames=matrix_columns)
        provenance_writer = csv.DictWriter(provenance_handle, fieldnames=PROVENANCE_COLUMNS)
        matrix_writer.writeheader()
        provenance_writer.writeheader()
        grouped = hkeys.groupby("Cell_ID", sort=True)
        for file_number, (cell_id, key_group) in enumerate(grouped, start=1):
            cell_id = str(cell_id)
            source = source_map[cell_id]
            raw_path = Path(source["Raw_Path"])
            fingerprint = fingerprint_map.get(cell_id, "UNAVAILABLE")
            print(
                f"[HUST {file_number:02d}/77] cell={cell_id} observations={len(key_group)}",
                flush=True,
            )
            try:
                loaded = load_hust_pickle(raw_path)
                native_id = str(source["Native_Cell_ID"])
                if native_id in loaded:
                    bundle = loaded[native_id]
                elif len(loaded) == 1:
                    bundle = next(iter(loaded.values()))
                else:
                    raise KeyError(f"NATIVE_CELL_ID_NOT_FOUND:{native_id}")
                # Access only the raw per-cycle data object.  The sibling dq and
                # rul mappings are intentionally never read by this stage.
                cycle_data = bundle["data"]
            except Exception as exc:
                loaded = None
                cycle_data = {}
                load_reason = f"RAW_FILE_LOAD_ERROR:{type(exc).__name__}:{exc}"
            else:
                load_reason = ""
            for cycle in key_group["Raw_Cycle_Index"].astype(int):
                try:
                    if load_reason:
                        raise RuntimeError(load_reason)
                    if int(cycle) not in cycle_data:
                        raise KeyError(f"RAW_CYCLE_NOT_FOUND:{cycle}")
                    raw_cycle = cycle_data[int(cycle)]
                    required = {"Status", "Current (mA)", "Voltage (V)", "Time (s)"}
                    missing_columns = sorted(required.difference(raw_cycle.columns))
                    if missing_columns:
                        raise KeyError(f"RAW_COLUMNS_MISSING:{missing_columns}")
                    result = extract_hust_features(
                        time_s=raw_cycle["Time (s)"].to_numpy(copy=False),
                        current_mA=raw_cycle["Current (mA)"].to_numpy(copy=False),
                        voltage_V=raw_cycle["Voltage (V)"].to_numpy(copy=False),
                        status=raw_cycle["Status"].to_numpy(copy=False),
                    )
                except Exception as exc:
                    audit.extraction_exception_count += 1
                    reason = f"HUST_EXTRACTION_ERROR:{type(exc).__name__}:{exc}"
                    result = invalid_extraction_result(
                        reason=reason,
                        raw_time_unit="SECONDS",
                        time_conversion_factor_to_seconds=1.0,
                    )
                _write_observation(
                    matrix_writer=matrix_writer,
                    provenance_writer=provenance_writer,
                    review_writer=review_writer,
                    audit=audit,
                    result=result,
                    dataset="HUST",
                    batch_id=None,
                    protocol_fingerprint=fingerprint,
                    cell_id=cell_id,
                    raw_cycle_index=int(cycle),
                    fold_id=fold_map[cell_id],
                    raw_source_file=str(source["File_Name"]),
                )
                representative_key = (cell_id, int(cycle))
                if representative_key in representative_plan:
                    plot_name = _safe_plot_name("HUST", None, cell_id, int(cycle))
                    if result.arrays_for_plot["time_seconds"].size:
                        _save_stage_plot(
                            result,
                            plots_dir / plot_name,
                            f"HUST | {cell_id} | Raw Cycle {cycle}",
                        )
                    representative_rows.append(
                        _representative_boundary_row(
                            dataset="HUST",
                            batch_id=None,
                            cell_id=cell_id,
                            cycle=int(cycle),
                            selection_reason=representative_plan[representative_key],
                            result=result,
                            plot_file=f"review_plots/{plot_name}",
                        )
                    )
            del cycle_data
            if loaded is not None:
                del loaded
            gc.collect()
    return matrix_path, representative_rows


def _order_and_validate_matrix(
    *,
    unsorted_path: Path,
    frozen_keys: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    matrix = pd.read_csv(unsorted_path, low_memory=False)
    matrix["Raw_Cycle_Index"] = _numeric_int(
        matrix["Raw_Cycle_Index"], f"{dataset}_MATRIX_RAW_CYCLE_INDEX"
    )
    key_columns = (
        ["Batch_ID", "Cell_ID", "Raw_Cycle_Index"]
        if dataset == "XJTU"
        else ["Cell_ID", "Raw_Cycle_Index"]
    )
    if dataset == "XJTU":
        matrix["Batch_ID"] = _numeric_int(matrix["Batch_ID"], "XJTU_MATRIX_BATCH_ID")
    if matrix.duplicated(key_columns).any():
        raise FeatureExtractionError(f"DUPLICATE_{dataset}_MATRIX_KEY")
    order = frozen_keys[key_columns].copy()
    order["_Frozen_Order"] = np.arange(len(order), dtype=np.int64)
    ordered = order.merge(matrix, on=key_columns, how="left", validate="one_to_one", sort=False)
    if ordered["Physical_Observation_Key"].isna().any():
        missing = ordered.loc[ordered["Physical_Observation_Key"].isna(), key_columns].head(5)
        raise FeatureExtractionError(
            f"{dataset}_MATRIX_MISSING_FROZEN_KEYS:{missing.to_dict('records')}"
        )
    if len(ordered) != len(frozen_keys) or len(matrix) != len(frozen_keys):
        raise FeatureExtractionError(
            f"{dataset}_MATRIX_ROW_COUNT:matrix={len(matrix)}:frozen={len(frozen_keys)}"
        )
    ordered = ordered.sort_values("_Frozen_Order", kind="stable").drop(columns="_Frozen_Order")
    ordered = ordered.loc[:, _matrix_columns(dataset)]
    for feature in FEATURE_NAMES:
        ordered[feature] = pd.to_numeric(ordered[feature], errors="coerce")
    return ordered.reset_index(drop=True)


def _load_targets_after_extraction(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load labels only after every physical feature has been extracted."""
    xlabels = pd.read_csv(
        paths.xjtu_final / "xjtu_55_soh_labels_long.csv",
        usecols=[
            "Batch_ID",
            "Cell_ID",
            "Raw_Cycle_Index",
            "Capacity_Ah",
            "Qref_Definition",
            "Qref_Ah",
            "SOH_Fraction",
        ],
        low_memory=False,
    )
    hlabels = pd.read_csv(
        paths.hust_final / "hust_77_soh_labels_long.csv",
        usecols=[
            "Cell_ID",
            "Raw_Cycle_Index",
            "Capacity_Ah",
            "Qref_Definition",
            "Qref_Ah",
            "SOH_Fraction",
        ],
        low_memory=False,
    )
    for labels, dataset in ((xlabels, "XJTU"), (hlabels, "HUST")):
        labels["Raw_Cycle_Index"] = _numeric_int(
            labels["Raw_Cycle_Index"], f"{dataset}_TARGET_RAW_CYCLE_INDEX"
        )
        keys = (
            ["Batch_ID", "Cell_ID", "Raw_Cycle_Index", "Qref_Definition"]
            if dataset == "XJTU"
            else ["Cell_ID", "Raw_Cycle_Index", "Qref_Definition"]
        )
        if labels.duplicated(keys).any():
            raise FeatureExtractionError(f"DUPLICATE_{dataset}_TARGET_KEY")
    xlabels["Batch_ID"] = _numeric_int(xlabels["Batch_ID"], "XJTU_TARGET_BATCH_ID")
    return xlabels, hlabels


def _sha256_text_values(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _target_join_audit(
    *,
    matrix: pd.DataFrame,
    labels: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    key_columns = (
        ["Batch_ID", "Cell_ID", "Raw_Cycle_Index"]
        if dataset == "XJTU"
        else ["Cell_ID", "Raw_Cycle_Index"]
    )
    order = matrix[key_columns].copy()
    order["_Matrix_Order"] = np.arange(len(order), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    nominal_hashes: dict[str, str] | None = None
    for qref, label_group in labels.groupby("Qref_Definition", sort=True):
        label_group = label_group.copy()
        joined = label_group.merge(order, on=key_columns, how="left", validate="one_to_one")
        missing_feature_rows = int(joined["_Matrix_Order"].isna().sum())
        joined = joined.dropna(subset=["_Matrix_Order"]).copy()
        joined["_Matrix_Order"] = joined["_Matrix_Order"].astype("int64")
        joined = joined.sort_values("_Matrix_Order", kind="stable")
        feature_subset = matrix.iloc[joined["_Matrix_Order"].to_numpy()].reset_index(drop=True)
        label_keys = joined[key_columns].reset_index(drop=True)
        feature_keys = feature_subset[key_columns].reset_index(drop=True)
        keys_match = label_keys.equals(feature_keys)
        key_hash = _sha256_text_values(
            feature_subset["Physical_Observation_Key"].astype(str).tolist()
        )
        fold_hash = _sha256_text_values(feature_subset["Fold_ID"].astype(int).tolist())
        value_hash = feature_values_sha256(feature_subset)
        mask_hash = missing_mask_sha256(feature_subset)
        current = {
            "key": key_hash,
            "fold": fold_hash,
            "value": value_hash,
            "mask": mask_hash,
        }
        if qref == "nominal":
            nominal_hashes = current
        common = qref in COMMON_QREFS
        rows.append(
            {
                "Dataset": dataset,
                "Qref_Definition": qref,
                "Qref_Independent_Feature_Extraction": "YES",
                "Frozen_Label_Row_Count": len(label_group),
                "Matched_Feature_Row_Count": len(feature_subset),
                "Missing_Feature_Row_Count": missing_feature_rows,
                "Physical_Keys_In_Label_Order_Match_Matrix": "YES" if keys_match else "NO",
                "Physical_Key_Sequence_SHA256": key_hash,
                "Fold_ID_Sequence_SHA256": fold_hash,
                "Feature_Values_SHA256": value_hash,
                "Missingness_Mask_SHA256": mask_hash,
                "Population_Role": (
                    "COMMON_PRIMARY_POPULATION"
                    if common
                    else "FROZEN_QREF_SPECIFIC_SECONDARY_POPULATION"
                ),
                "Expected_Common_Population_Row_Count": (
                    len(matrix) if common else np.nan
                ),
                "Common_Population_Row_Count_Match": (
                    "YES" if common and len(feature_subset) == len(matrix) else "NOT_APPLICABLE"
                ),
                "Same_Physical_Rows_As_Nominal": "PENDING_BASELINE",
                "Same_Row_Order_As_Nominal": "PENDING_BASELINE",
                "Same_Fold_ID_As_Nominal": "PENDING_BASELINE",
                "Same_Feature_Values_As_Nominal": "PENDING_BASELINE",
                "Same_Missingness_As_Nominal": "PENDING_BASELINE",
                "Only_Target_Column_Changes_Across_Common_Qrefs": "PENDING_BASELINE",
                "Qref_Specific_Row_Dropping": "NO",
                "Target_Column": "SOH_Fraction",
                "Target_Nonmissing_Count": int(joined["SOH_Fraction"].notna().sum()),
                "Join_Status": (
                    "PASS" if missing_feature_rows == 0 and keys_match else "FAIL"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if nominal_hashes is None:
        raise FeatureExtractionError(f"{dataset}_NOMINAL_TARGET_DEFINITION_MISSING")
    for index, row in frame.iterrows():
        common = row["Qref_Definition"] in COMMON_QREFS
        if common:
            same_key = row["Physical_Key_Sequence_SHA256"] == nominal_hashes["key"]
            same_fold = row["Fold_ID_Sequence_SHA256"] == nominal_hashes["fold"]
            same_value = row["Feature_Values_SHA256"] == nominal_hashes["value"]
            same_mask = row["Missingness_Mask_SHA256"] == nominal_hashes["mask"]
            frame.at[index, "Same_Physical_Rows_As_Nominal"] = "YES" if same_key else "NO"
            frame.at[index, "Same_Row_Order_As_Nominal"] = "YES" if same_key else "NO"
            frame.at[index, "Same_Fold_ID_As_Nominal"] = "YES" if same_fold else "NO"
            frame.at[index, "Same_Feature_Values_As_Nominal"] = "YES" if same_value else "NO"
            frame.at[index, "Same_Missingness_As_Nominal"] = "YES" if same_mask else "NO"
            frame.at[index, "Only_Target_Column_Changes_Across_Common_Qrefs"] = (
                "YES" if same_key and same_fold and same_value and same_mask else "NO"
            )
        else:
            frame.at[index, "Same_Physical_Rows_As_Nominal"] = "NOT_APPLICABLE_SECONDARY_SUBSET"
            frame.at[index, "Same_Row_Order_As_Nominal"] = "NOT_APPLICABLE_SECONDARY_SUBSET"
            frame.at[index, "Same_Fold_ID_As_Nominal"] = "NOT_APPLICABLE_SECONDARY_SUBSET"
            frame.at[index, "Same_Feature_Values_As_Nominal"] = (
                "MATCHES_SINGLE_FROZEN_FEATURE_MATRIX_SUBSET"
            )
            frame.at[index, "Same_Missingness_As_Nominal"] = (
                "MATCHES_SINGLE_FROZEN_FEATURE_MATRIX_SUBSET"
            )
            frame.at[index, "Only_Target_Column_Changes_Across_Common_Qrefs"] = "NOT_APPLICABLE"
    return frame


def _append_outlier_reviews(
    *,
    review_path: Path,
    xmatrix: pd.DataFrame,
    hmatrix: pd.DataFrame,
) -> int:
    candidates = pd.concat(
        [
            outlier_candidates(xmatrix, "XJTU"),
            outlier_candidates(hmatrix, "HUST"),
        ],
        ignore_index=True,
        sort=False,
    )
    if candidates.empty:
        return 0
    with review_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        for row in candidates.to_dict("records"):
            writer.writerow(
                {
                    "Dataset": row.get("Dataset", ""),
                    "Batch_ID": row.get("Batch_ID", ""),
                    "Protocol_Fingerprint_ID": "",
                    "Cell_ID": row.get("Cell_ID", ""),
                    "Raw_Cycle_Index": row.get("Raw_Cycle_Index", ""),
                    "Feature_Name": row.get("Feature_Name", ""),
                    "Feature_Value": row.get("Feature_Value", ""),
                    "Extraction_Status": row.get("Extraction_Status", "PASS"),
                    "Candidate_Type": row.get("Candidate_Type", "ROBUST_EXTREME_VALUE_REVIEW"),
                    "Reason_Code": row.get("Reason_Code", ""),
                    "Review_Lower_Bound": row.get("Review_Lower_Bound", ""),
                    "Review_Upper_Bound": row.get("Review_Upper_Bound", ""),
                    "Automatic_Exclusion": "NO",
                    "Raw_Source_File": "",
                }
            )
    return len(candidates)


def _test_row(
    test_id: str,
    passed: bool,
    expected: Any,
    observed: Any,
    details: str = "",
) -> dict[str, Any]:
    return {
        "Test_ID": test_id,
        "Test_Category": "HARD_REGRESSION",
        "Dataset": "ALL",
        "Batch_ID": "",
        "Cell_ID": "",
        "Raw_Cycle_Index": "",
        "Status": "PASS" if passed else "FAIL",
        "Expected": expected,
        "Observed": observed,
        "Details": details,
    }


def _fold_assignment_match(
    matrix: pd.DataFrame, folds: pd.DataFrame, dataset: str
) -> bool:
    expected = folds[["Cell_ID", "Fold_ID"]].copy()
    expected["Fold_ID"] = pd.to_numeric(expected["Fold_ID"], errors="raise").astype(int)
    actual = matrix[["Cell_ID", "Fold_ID"]].drop_duplicates()
    actual["Fold_ID"] = pd.to_numeric(actual["Fold_ID"], errors="raise").astype(int)
    joined = actual.merge(expected, on="Cell_ID", suffixes=("_Actual", "_Expected"), validate="one_to_one")
    return (
        len(joined) == matrix["Cell_ID"].nunique()
        and len(joined) == len(expected)
        and joined["Fold_ID_Actual"].equals(joined["Fold_ID_Expected"])
    )


def _physical_keys_match(
    matrix: pd.DataFrame, keys: pd.DataFrame, dataset: str
) -> bool:
    columns = (
        ["Batch_ID", "Cell_ID", "Raw_Cycle_Index"]
        if dataset == "XJTU"
        else ["Cell_ID", "Raw_Cycle_Index"]
    )
    left = matrix[columns].reset_index(drop=True).copy()
    right = keys[columns].reset_index(drop=True).copy()
    for frame in (left, right):
        frame["Raw_Cycle_Index"] = pd.to_numeric(
            frame["Raw_Cycle_Index"], errors="raise"
        ).astype(int)
        if dataset == "XJTU":
            frame["Batch_ID"] = pd.to_numeric(frame["Batch_ID"], errors="raise").astype(int)
    return left.equals(right)


def _build_regression_tests(
    *,
    xmatrix: pd.DataFrame,
    hmatrix: pd.DataFrame,
    xkeys: pd.DataFrame,
    hkeys: pd.DataFrame,
    xfold: pd.DataFrame,
    hfold: pd.DataFrame,
    target_join: pd.DataFrame,
    representative_rows: list[dict[str, Any]],
    audit: RuntimeAudit,
    raw_inputs_unchanged: bool,
    frozen_inputs_unchanged: bool,
    staging: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        _test_row(
            "XJTU_EXPECTED_OBSERVATION_KEYS",
            len(xmatrix) == XJTU_EXPECTED_OBSERVATIONS,
            XJTU_EXPECTED_OBSERVATIONS,
            len(xmatrix),
        )
    )
    rows.append(
        _test_row(
            "HUST_EXPECTED_OBSERVATION_KEYS",
            len(hmatrix) == HUST_EXPECTED_OBSERVATIONS,
            HUST_EXPECTED_OBSERVATIONS,
            len(hmatrix),
        )
    )
    duplicate_count = int(
        xmatrix.duplicated(["Batch_ID", "Cell_ID", "Raw_Cycle_Index"]).sum()
        + hmatrix.duplicated(["Cell_ID", "Raw_Cycle_Index"]).sum()
    )
    expected_feature_keys = (len(xmatrix) + len(hmatrix)) * len(FEATURE_NAMES)
    rows.append(
        _test_row(
            "NO_DUPLICATE_FEATURE_KEYS",
            duplicate_count == 0 and audit.feature_key_count == expected_feature_keys,
            f"0 duplicates and {expected_feature_keys} long keys",
            f"{duplicate_count} duplicates and {audit.feature_key_count} long keys",
            "Each unique physical observation is emitted exactly once for each frozen feature.",
        )
    )
    x_parameters = tuple(inspect.signature(extract_xjtu_features).parameters)
    h_parameters = tuple(inspect.signature(extract_hust_features).parameters)
    allowed_x = ("relative_time_min", "current_A", "voltage_V")
    allowed_h = ("time_s", "current_mA", "voltage_V", "status")
    rows.append(
        _test_row(
            "NO_TARGET_CAPACITY_USED_DURING_EXTRACTION",
            x_parameters == allowed_x and h_parameters == allowed_h,
            f"XJTU={allowed_x};HUST={allowed_h}",
            f"XJTU={x_parameters};HUST={h_parameters}",
            "Target capacity columns are loaded only after both raw feature matrices are frozen.",
        )
    )
    rows.append(
        _test_row(
            "NO_QREF_USED_DURING_EXTRACTION",
            x_parameters == allowed_x and h_parameters == allowed_h,
            "No Qref argument in either extractor",
            "No Qref argument in either extractor" if x_parameters == allowed_x and h_parameters == allowed_h else "Unexpected extractor arguments",
        )
    )
    rows.append(
        _test_row(
            "NO_POST_DISCHARGE_INFORMATION_USED",
            audit.post_discharge_boundary_violations == 0,
            0,
            audit.post_discharge_boundary_violations,
            "Every emitted feature stage ends strictly before the target discharge starts.",
        )
    )
    expected_time_counts = Counter(
        {
            ("XJTU", "MINUTES", 60.0): len(xmatrix),
            ("HUST", "SECONDS", 1.0): len(hmatrix),
        }
    )
    rows.append(
        _test_row(
            "RAW_TIME_UNITS_EXPLICITLY_VALIDATED",
            audit.time_unit_counts == expected_time_counts,
            dict(expected_time_counts),
            dict(audit.time_unit_counts or {}),
        )
    )
    rows.append(
        _test_row(
            "FEATURE_SET_A_EXACTLY_11_FEATURES",
            len(FEATURE_NAMES) == 11
            and all(feature in xmatrix.columns for feature in FEATURE_NAMES)
            and all(feature in hmatrix.columns for feature in FEATURE_NAMES),
            "Exactly the 11 frozen SAFE_CORE names",
            "|".join(FEATURE_NAMES),
        )
    )
    b_match = bool(
        np.array_equal(
            xmatrix["Feature_Set_B_Raw_Cycle_Index"].to_numpy(dtype=int),
            xmatrix["Raw_Cycle_Index"].to_numpy(dtype=int),
        )
        and np.array_equal(
            hmatrix["Feature_Set_B_Raw_Cycle_Index"].to_numpy(dtype=int),
            hmatrix["Raw_Cycle_Index"].to_numpy(dtype=int),
        )
    )
    rows.append(
        _test_row(
            "FEATURE_SET_B_EQUALS_A_PLUS_CYCLE_INDEX",
            b_match,
            "Feature Set A plus an unchanged Raw_Cycle_Index sensitivity column",
            "MATCH" if b_match else "MISMATCH",
        )
    )
    folds_match = _fold_assignment_match(xmatrix, xfold, "XJTU") and _fold_assignment_match(
        hmatrix, hfold, "HUST"
    )
    rows.append(
        _test_row(
            "FOLD_IDS_MATCH_FROZEN_DESIGN",
            folds_match,
            "Exact cell-level frozen Fold_ID assignments",
            "MATCH" if folds_match else "MISMATCH",
        )
    )
    keys_match = _physical_keys_match(xmatrix, xkeys, "XJTU") and _physical_keys_match(
        hmatrix, hkeys, "HUST"
    )
    rows.append(
        _test_row(
            "PHYSICAL_KEYS_MATCH_FROZEN_POPULATION",
            keys_match,
            "Exact frozen physical key sequence",
            "MATCH" if keys_match else "MISMATCH",
        )
    )
    common_join = target_join.loc[target_join["Qref_Definition"].isin(COMMON_QREFS)]
    same_values = bool(
        len(common_join) == 2 * len(COMMON_QREFS)
        and common_join["Same_Feature_Values_As_Nominal"].eq("YES").all()
    )
    same_missing = bool(
        len(common_join) == 2 * len(COMMON_QREFS)
        and common_join["Same_Missingness_As_Nominal"].eq("YES").all()
    )
    join_preserves = bool(
        target_join["Join_Status"].eq("PASS").all()
        and common_join["Common_Population_Row_Count_Match"].eq("YES").all()
        and common_join["Same_Row_Order_As_Nominal"].eq("YES").all()
        and common_join["Same_Fold_ID_As_Nominal"].eq("YES").all()
    )
    rows.append(
        _test_row(
            "NO_QREF_SPECIFIC_FEATURE_VALUES",
            same_values,
            "Identical hashes for all common-Qref feature values",
            "MATCH" if same_values else "MISMATCH",
        )
    )
    rows.append(
        _test_row(
            "NO_QREF_SPECIFIC_MISSINGNESS",
            same_missing,
            "Identical hashes for all common-Qref missingness masks",
            "MATCH" if same_missing else "MISMATCH",
        )
    )
    rows.append(
        _test_row(
            "TARGET_JOIN_PRESERVES_FEATURE_ROWS",
            join_preserves,
            "Same common physical rows, order, folds, values, and missingness",
            "PASS" if join_preserves else "FAIL",
        )
    )
    representative_pass = bool(
        len([row for row in representative_rows if row["Dataset"] == "XJTU"]) == 6
        and len([row for row in representative_rows if row["Dataset"] == "HUST"]) >= 8
        and all(row["Status"] == "PASS" for row in representative_rows)
    )
    rows.append(
        _test_row(
            "REPRESENTATIVE_RAW_BOUNDARIES_VALIDATED",
            representative_pass,
            "6 XJTU batch representatives and >=8 HUST representatives all PASS",
            (
                f"XJTU={sum(row['Dataset']=='XJTU' for row in representative_rows)};"
                f"HUST={sum(row['Dataset']=='HUST' for row in representative_rows)};"
                f"failures={sum(row['Status']!='PASS' for row in representative_rows)}"
            ),
        )
    )
    model_artifacts = [
        path.name
        for path in staging.rglob("*")
        if path.is_file() and path.suffix.lower() in {".joblib", ".onnx", ".pt", ".pth"}
    ]
    rows.append(
        _test_row(
            "NO_FINAL_MODELS_TRAINED",
            not model_artifacts,
            "No trained-model artifacts",
            "NONE" if not model_artifacts else "|".join(model_artifacts),
        )
    )
    rows.append(
        _test_row(
            "RAW_INPUT_FILES_UNCHANGED",
            raw_inputs_unchanged,
            "All raw file sizes and mtimes unchanged",
            "UNCHANGED" if raw_inputs_unchanged else "CHANGED",
        )
    )
    rows.append(
        _test_row(
            "FROZEN_INPUT_OUTPUTS_UNCHANGED",
            frozen_inputs_unchanged,
            "Frozen XJTU, HUST, and design trees byte-identical",
            "UNCHANGED" if frozen_inputs_unchanged else "CHANGED",
        )
    )
    for index, representative in enumerate(representative_rows, start=1):
        detail = (
            f"charge={representative['Charge_Start_Index']}-{representative['Charge_End_Index']};"
            f"cc={representative['CC_Start_Index']}-{representative['CC_End_Index']};"
            f"cv={representative['CV_Start_Index']}-{representative['CV_End_Index']};"
            f"pre_discharge={representative['Pre_Discharge_Index']};"
            f"discharge_start={representative['Discharge_Start_Index']};"
            f"time={representative['Raw_Time_Unit']}*{representative['Time_Conversion_Factor_To_Seconds']};"
            f"plot={representative['Plot_File']};reasons={representative['Details']}"
        )
        rows.append(
            {
                "Test_ID": f"REPRESENTATIVE_{index:02d}_{representative['Dataset']}_{representative['Cell_ID']}_C{representative['Raw_Cycle_Index']}",
                "Test_Category": representative["Test_Category"],
                "Dataset": representative["Dataset"],
                "Batch_ID": representative["Batch_ID"],
                "Cell_ID": representative["Cell_ID"],
                "Raw_Cycle_Index": representative["Raw_Cycle_Index"],
                "Status": representative["Status"],
                "Expected": "Ordered physical CC/CV/charge-end/pre-discharge boundaries and explicit time unit",
                "Observed": (
                    f"boundary_order={representative['Boundary_Order_Valid']};"
                    f"cv_features={representative['CV_Features_Extracted']};"
                    f"time_unit={representative['Time_Unit_Validated']}"
                ),
                "Details": detail,
            }
        )
    return pd.DataFrame(rows)


def _software_versions() -> dict[str, str]:
    import matplotlib as matplotlib_package
    import scipy

    return {
        "Python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib_package.__version__,
    }


def _config_values() -> dict[str, Any]:
    from . import config

    values: dict[str, Any] = {}
    for name in sorted(dir(config)):
        if not name.isupper():
            continue
        value = getattr(config, name)
        if isinstance(value, (str, int, float, bool, tuple, list, dict)):
            values[name] = value
    return values


def _write_processing_log(
    *,
    path: Path,
    started: datetime,
    finished: datetime,
    xcanonical: pd.DataFrame,
    hcanonical: pd.DataFrame,
    xmatrix: pd.DataFrame,
    hmatrix: pd.DataFrame,
    missingness: pd.DataFrame,
    distributions: pd.DataFrame,
    target_association: pd.DataFrame,
    target_join: pd.DataFrame,
    regression: pd.DataFrame,
    audit: RuntimeAudit,
    outlier_count: int,
    decision: str,
    decision_reasons: list[str],
    snapshots: dict[str, str],
) -> None:
    lines = [
        f"Pipeline_Version: {PIPELINE_VERSION}",
        f"Random_Seed: {RANDOM_SEED}",
        f"Started_Asia_Shanghai: {started.isoformat()}",
        f"Finished_Asia_Shanghai: {finished.isoformat()}",
        f"Decision: {decision}",
        f"Decision_Reasons: {'|'.join(decision_reasons) if decision_reasons else 'NONE'}",
        "Final_Models_Trained: NO",
        "Raw_Files_Modified: NO",
        "Capacity_or_Target_Used_During_Extraction: NO",
        "Qref_Used_During_Extraction: NO",
        "Post_Discharge_Information_Used: NO",
        "",
        "[SOFTWARE]",
    ]
    lines.extend(f"{name}: {version}" for name, version in _software_versions().items())
    lines.extend(["", "[INPUT_COUNTS]"])
    lines.extend(
        [
            f"XJTU_Canonical_Raw_Files: {len(xcanonical)}",
            f"HUST_Canonical_Raw_Files: {len(hcanonical)}",
            f"XJTU_Eligible_Physical_Observations: {len(xmatrix)}",
            f"HUST_Eligible_Physical_Observations: {len(hmatrix)}",
            f"Feature_Set_A_Count: {len(FEATURE_NAMES)}",
            f"Long_Form_Feature_Key_Count: {audit.feature_key_count}",
            f"NonPASS_Feature_Status_Count: {audit.nonpass_feature_count}",
            f"Extraction_Exception_Observation_Count: {audit.extraction_exception_count}",
            f"Post_Discharge_Boundary_Violations: {audit.post_discharge_boundary_violations}",
            f"Robust_Outlier_Review_Candidate_Count: {outlier_count}",
        ]
    )
    lines.extend(["", "[TIME_UNIT_AUDIT]"])
    for key, count in sorted((audit.time_unit_counts or {}).items()):
        lines.append(f"{key[0]}: raw_unit={key[1]};factor_to_seconds={key[2]};observations={count}")
    lines.extend(["", "[FROZEN_INPUT_SNAPSHOT_SHA256]"])
    lines.extend(f"{name}: {digest}" for name, digest in sorted(snapshots.items()))
    lines.extend(["", "[CONFIG]"])
    lines.append(json.dumps(_config_values(), ensure_ascii=False, sort_keys=True, indent=2))
    lines.extend(["", "[MISSINGNESS_DATASET_LEVEL]"])
    dataset_missing = missingness.loc[missingness["Audit_Level"].eq("DATASET")]
    for row in dataset_missing.itertuples(index=False):
        lines.append(
            f"{row.Dataset}|{row.Feature_Name}: PASS={row.PASS_Count};CHECK={row.CHECK_Count};"
            f"MISSING={row.MISSING_Count};INVALID={row.INVALID_EXTRACTION_Count};"
            f"value_missing_rate={row.Missing_Rate:.12g};affected_cells={row.Affected_Cell_Count}"
        )
    lines.extend(["", "[VARIATION_FLAGS]"])
    for row in distributions.itertuples(index=False):
        lines.append(f"{row.Dataset}|{row.Feature_Name}: {row.Variation_Flag}")
    lines.extend(["", "[TARGET_ASSOCIATION_FLAGS]"])
    for row in target_association.itertuples(index=False):
        lines.append(
            f"{row.Dataset}|{row.Feature_Name}: rho_capacity={row.Spearman_Rho_vs_Primary_Capacity_Ah};"
            f"rho_nominal_soh={row.Spearman_Rho_vs_Nominal_SOH};flag={row.Association_Flag}"
        )
    lines.extend(["", "[TARGET_JOIN]"])
    for row in target_join.itertuples(index=False):
        lines.append(
            f"{row.Dataset}|{row.Qref_Definition}: rows={row.Matched_Feature_Row_Count};"
            f"join={row.Join_Status};same_values={row.Same_Feature_Values_As_Nominal};"
            f"same_missing={row.Same_Missingness_As_Nominal}"
        )
    lines.extend(["", "[REGRESSION_TESTS]"])
    for row in regression.itertuples(index=False):
        lines.append(f"{row.Test_ID}: {row.Status} | observed={row.Observed}")
    lines.append("")
    lines.append(
        "ALL HARD REGRESSION TESTS PASSED"
        if regression["Status"].eq("PASS").all()
        else "PIPELINE FAILED HARD REGRESSION TESTS"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decision_reasons(
    *,
    missingness: pd.DataFrame,
    distributions: pd.DataFrame,
    target_association: pd.DataFrame,
) -> list[str]:
    reasons: list[str] = []
    systematic = missingness.loc[
        missingness["Systematic_Missingness_Flag"].eq("REVIEW")
    ]
    for row in systematic.itertuples(index=False):
        reasons.append(
            f"SYSTEMATIC_MISSINGNESS:{row.Dataset}:{row.Audit_Level}:{row.Group_Value}:"
            f"{row.Feature_Name}:rate={row.Missing_Rate:.12g}"
        )
    for row in target_association.loc[
        target_association["Association_Flag"].eq("POTENTIAL_NEAR_TARGET_PROXY_REVIEW")
    ].itertuples(index=False):
        reasons.append(
            f"POTENTIAL_NEAR_TARGET_PROXY_REVIEW:{row.Dataset}:{row.Feature_Name}:"
            f"rho_capacity={row.Spearman_Rho_vs_Primary_Capacity_Ah}:"
            f"rho_soh={row.Spearman_Rho_vs_Nominal_SOH}"
        )
    for row in distributions.loc[
        distributions["Variation_Flag"].eq("CONSTANT")
    ].itertuples(index=False):
        reasons.append(f"CONSTANT_FEATURE:{row.Dataset}:{row.Feature_Name}")
    return sorted(set(reasons))


def _validate_output_headers(staging: Path) -> None:
    for name in CSV_OUTPUTS:
        path = staging / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FeatureExtractionError(f"OUTPUT_MISSING_OR_EMPTY:{name}")
        header = pd.read_csv(path, nrows=2, low_memory=False)
        if header.columns.duplicated().any():
            raise FeatureExtractionError(f"OUTPUT_DUPLICATE_COLUMNS:{name}")


def run_feature_extraction(
    workspace: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    paths = Paths.from_workspace(workspace)
    target = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (workspace / "outputs" / "soh_feature_extraction_v1").resolve()
    )
    if target.exists():
        raise FeatureExtractionError(f"OUTPUT_DIRECTORY_ALREADY_EXISTS:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".soh_feature_extraction_v1_staging_",
            dir=target.parent,
        )
    )
    (staging / "review_plots").mkdir(parents=True, exist_ok=False)
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    print(f"[{PIPELINE_VERSION}] staging={staging}", flush=True)

    log_path = staging / "processing_log.txt"
    try:
        design = _load_design_and_keys(paths)
        xcanonical, hcanonical = _raw_file_maps(paths)
        raw_paths = [Path(value) for value in xcanonical["Raw_Path"]] + [
            Path(value) for value in hcanonical["Raw_Path"]
        ]
        raw_before = _raw_stat_snapshot(raw_paths)
        xjtu_frozen_before = _tree_snapshot(paths.xjtu_final)
        hust_frozen_before = _tree_snapshot(paths.hust_final)
        design_frozen_before = _tree_snapshot(paths.design)
        snapshot_digests = {
            "XJTU_Final_V2_Before": _snapshot_digest(xjtu_frozen_before),
            "HUST_Final_V1_Before": _snapshot_digest(hust_frozen_before),
            "SOH_ML_Design_V1_Before": _snapshot_digest(design_frozen_before),
        }
        xplan, hplan = _representative_plan(
            design["xkeys"], design["hkeys"], paths
        )
        audit = RuntimeAudit(missingness=MissingnessAccumulator())
        review_path = staging / "feature_extraction_review_candidates.csv"
        with review_path.open("w", newline="", encoding="utf-8-sig") as review_handle:
            review_writer = csv.DictWriter(review_handle, fieldnames=REVIEW_COLUMNS)
            review_writer.writeheader()
            x_unsorted, x_representatives = _process_xjtu(
                paths=paths,
                staging=staging,
                xcanonical=xcanonical,
                xkeys=design["xkeys"],
                xfold=design["xfold"],
                representative_plan=xplan,
                audit=audit,
                review_writer=review_writer,
            )
            h_unsorted, h_representatives = _process_hust(
                paths=paths,
                staging=staging,
                hcanonical=hcanonical,
                hkeys=design["hkeys"],
                hfold=design["hfold"],
                representative_plan=hplan,
                audit=audit,
                review_writer=review_writer,
            )

        print("[AUDIT] ordering matrices against frozen physical populations", flush=True)
        xmatrix = _order_and_validate_matrix(
            unsorted_path=x_unsorted,
            frozen_keys=design["xkeys"],
            dataset="XJTU",
        )
        hmatrix = _order_and_validate_matrix(
            unsorted_path=h_unsorted,
            frozen_keys=design["hkeys"],
            dataset="HUST",
        )
        _write_csv(xmatrix, staging / "feature_matrix_xjtu_v1.csv")
        _write_csv(hmatrix, staging / "feature_matrix_hust_v1.csv")
        x_unsorted.unlink()
        h_unsorted.unlink()

        # Target values become visible only here, after both physical feature
        # matrices have been extracted, ordered, and written.
        print("[AUDIT] raw extraction frozen; loading labels for post-extraction audits", flush=True)
        xlabels, hlabels = _load_targets_after_extraction(paths)
        xnominal = xlabels.loc[xlabels["Qref_Definition"].eq("nominal")].copy()
        hnominal = hlabels.loc[hlabels["Qref_Definition"].eq("nominal")].copy()

        missingness = audit.missingness.to_frame()
        distributions = pd.concat(
            [
                distribution_summary(xmatrix, "XJTU"),
                distribution_summary(hmatrix, "HUST"),
            ],
            ignore_index=True,
        )
        xjtu_batch_distribution = distribution_by_xjtu_batch(xmatrix)
        shortcut = protocol_shortcut_audit(xmatrix)
        target_association = pd.concat(
            [
                target_association_audit(
                    xmatrix,
                    xnominal,
                    "XJTU",
                    ["Batch_ID", "Cell_ID", "Raw_Cycle_Index"],
                ),
                target_association_audit(
                    hmatrix,
                    hnominal,
                    "HUST",
                    ["Cell_ID", "Raw_Cycle_Index"],
                ),
            ],
            ignore_index=True,
        )
        complete_case = pd.concat(
            [
                complete_case_audit(xmatrix, "XJTU"),
                complete_case_audit(hmatrix, "HUST"),
            ],
            ignore_index=True,
        )
        target_join = pd.concat(
            [
                _target_join_audit(matrix=xmatrix, labels=xlabels, dataset="XJTU"),
                _target_join_audit(matrix=hmatrix, labels=hlabels, dataset="HUST"),
            ],
            ignore_index=True,
        )

        _write_csv(missingness, staging / "feature_missingness_summary.csv")
        _write_csv(distributions, staging / "feature_distribution_summary.csv")
        _write_csv(
            xjtu_batch_distribution,
            staging / "feature_distribution_by_xjtu_batch.csv",
        )
        _write_csv(shortcut, staging / "feature_protocol_shortcut_audit.csv")
        _write_csv(
            target_association,
            staging / "feature_target_association_audit.csv",
        )
        _write_csv(complete_case, staging / "feature_complete_case_audit.csv")
        _write_csv(target_join, staging / "feature_target_join_audit.csv")
        outlier_count = _append_outlier_reviews(
            review_path=review_path,
            xmatrix=xmatrix,
            hmatrix=hmatrix,
        )

        raw_after = _raw_stat_snapshot(raw_paths)
        xjtu_frozen_after = _tree_snapshot(paths.xjtu_final)
        hust_frozen_after = _tree_snapshot(paths.hust_final)
        design_frozen_after = _tree_snapshot(paths.design)
        snapshot_digests.update(
            {
                "XJTU_Final_V2_After": _snapshot_digest(xjtu_frozen_after),
                "HUST_Final_V1_After": _snapshot_digest(hust_frozen_after),
                "SOH_ML_Design_V1_After": _snapshot_digest(design_frozen_after),
            }
        )
        raw_unchanged = raw_before == raw_after
        frozen_unchanged = (
            xjtu_frozen_before == xjtu_frozen_after
            and hust_frozen_before == hust_frozen_after
            and design_frozen_before == design_frozen_after
        )
        representatives = x_representatives + h_representatives
        regression = _build_regression_tests(
            xmatrix=xmatrix,
            hmatrix=hmatrix,
            xkeys=design["xkeys"],
            hkeys=design["hkeys"],
            xfold=design["xfold"],
            hfold=design["hfold"],
            target_join=target_join,
            representative_rows=representatives,
            audit=audit,
            raw_inputs_unchanged=raw_unchanged,
            frozen_inputs_unchanged=frozen_unchanged,
            staging=staging,
        )
        _write_csv(regression, staging / "feature_extraction_regression_tests.csv")

        hard_failures = regression.loc[regression["Status"].ne("PASS")]
        decision_reasons = _decision_reasons(
            missingness=missingness,
            distributions=distributions,
            target_association=target_association,
        )
        if hard_failures.empty:
            decision = (
                "COMPLETE_READY_FOR_HUMAN_REVIEW"
                if not decision_reasons
                else "NEEDS_REVIEW_DO_NOT_TRAIN_FINAL_MODELS"
            )
        else:
            decision = "PIPELINE_FAILED_HARD_REGRESSION_TESTS"
            decision_reasons.extend(
                f"HARD_REGRESSION_FAILURE:{test_id}"
                for test_id in hard_failures["Test_ID"].astype(str)
            )
        finished = datetime.now(ZoneInfo("Asia/Shanghai"))
        _write_processing_log(
            path=log_path,
            started=started,
            finished=finished,
            xcanonical=xcanonical,
            hcanonical=hcanonical,
            xmatrix=xmatrix,
            hmatrix=hmatrix,
            missingness=missingness,
            distributions=distributions,
            target_association=target_association,
            target_join=target_join,
            regression=regression,
            audit=audit,
            outlier_count=outlier_count,
            decision=decision,
            decision_reasons=decision_reasons,
            snapshots=snapshot_digests,
        )
        _validate_output_headers(staging)
        if not hard_failures.empty:
            raise FeatureExtractionError(
                "HARD_REGRESSION_TESTS_FAILED:"
                + "|".join(hard_failures["Test_ID"].astype(str))
                + f":staging={staging}"
            )
        os.replace(staging, target)
        if decision == "COMPLETE_READY_FOR_HUMAN_REVIEW":
            final_stdout = (
                "SOH FEATURE EXTRACTION V1 COMPLETE —\n"
                "READY FOR HUMAN REVIEW —\n"
                "NO FINAL MODELS TRAINED"
            )
        else:
            final_stdout = (
                "SOH FEATURE EXTRACTION V1 NEEDS REVIEW —\n"
                "DO NOT TRAIN FINAL MODELS"
            )
        return {
            "output_dir": str(target),
            "decision": decision,
            "decision_reasons": decision_reasons,
            "xjtu_observations": len(xmatrix),
            "hust_observations": len(hmatrix),
            "nonpass_feature_count": audit.nonpass_feature_count,
            "outlier_review_count": outlier_count,
            "final_stdout": final_stdout,
        }
    except Exception as exc:
        if not log_path.exists():
            failure_lines = [
                f"Pipeline_Version: {PIPELINE_VERSION}",
                f"Started_Asia_Shanghai: {started.isoformat()}",
                f"Failed_Asia_Shanghai: {datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}",
                f"PIPELINE FAILED: {type(exc).__name__}: {exc}",
                f"Staging_Directory_Preserved: {staging}",
                "Final_Models_Trained: NO",
            ]
            log_path.write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
        raise


__all__ = ["FeatureExtractionError", "run_feature_extraction"]
