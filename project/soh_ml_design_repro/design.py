"""Create an auditable SOH benchmark design without fitting any model.

The module consumes frozen XJTU Final V2 and HUST Final V1 label tables.  It
never writes into a frozen directory and it does not import a modelling
library.  Candidate raw-curve features are inventoried only; extraction and
model fitting are deliberately deferred until human approval.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


DESIGN_VERSION = "SOH_ML_DESIGN_V1"
RANDOM_SEED = 20260928
N_FOLDS = 5
COMMON_QREFS = ("nominal", "first_comparable", "early_median5_comparable")
XJTU_SECONDARY_QREF = "initial_test"
OUTPUT_NAMES = (
    "ml_feature_inventory.csv",
    "ml_feature_leakage_audit.csv",
    "ml_harmonized_feature_plan.csv",
    "ml_dataset_specific_feature_plan.csv",
    "ml_observation_intersection_audit.csv",
    "ml_cell_fold_assignments_xjtu.csv",
    "ml_cell_fold_assignments_hust.csv",
    "ml_xjtu_initial_test_47cell_fold_assignments.csv",
    "ml_label_definition_plan.csv",
    "ml_preprocessing_plan.csv",
    "ml_model_hyperparameter_plan.csv",
    "ml_metric_plan.csv",
    "ml_fairness_matrix.csv",
    "ml_design_regression_tests.csv",
)


@dataclass(frozen=True)
class InputPaths:
    workspace: Path
    xjtu_frozen: Path
    hust_frozen: Path
    cross_frozen: Path
    xjtu_availability: Path
    hust_availability: Path

    @classmethod
    def from_workspace(cls, workspace: Path) -> "InputPaths":
        outputs = workspace / "outputs"
        return cls(
            workspace=workspace,
            xjtu_frozen=outputs / "xjtu_55_final_v2",
            hust_frozen=outputs / "hust_77_final_v1",
            cross_frozen=outputs / "cross_dataset_summary_v1",
            xjtu_availability=outputs / "xjtu_55_final_v1" / "xjtu_55_cycle_qc.csv",
            hust_availability=outputs / "hust_scan_v1" / "hust_cycle_inventory.csv",
        )


@dataclass
class DesignContext:
    paths: InputPaths
    xjtu_labels: pd.DataFrame
    hust_labels: pd.DataFrame
    xjtu_qref: pd.DataFrame
    hust_qref: pd.DataFrame
    xjtu_availability: pd.DataFrame
    hust_availability: pd.DataFrame
    xjtu_common_keys: pd.DataFrame
    hust_common_keys: pd.DataFrame
    xjtu_initial_cells: list[str]
    availability_stats: dict[str, Any]


class DesignError(RuntimeError):
    """Raised when a hard-stop design condition is encountered."""


def _require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise DesignError("Missing required frozen inputs: " + " | ".join(missing))


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_snapshot(directory: Path) -> dict[str, tuple[int, str]]:
    """Return size and SHA-256 for every file in a frozen directory."""
    snapshot: dict[str, tuple[int, str]] = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        relative = path.relative_to(directory).as_posix()
        snapshot[relative] = (path.stat().st_size, _sha256_file(path))
    return snapshot


def _read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    except Exception as exc:  # pragma: no cover - message is for research audit
        raise DesignError(f"Failed to read {path}: {exc}") from exc


def _numeric(series: pd.Series, label: str) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="raise")
    except Exception as exc:
        raise DesignError(f"Non-numeric values in {label}: {exc}") from exc


def _key_set(frame: pd.DataFrame, columns: list[str]) -> set[tuple[Any, ...]]:
    return set(map(tuple, frame[columns].itertuples(index=False, name=None)))


def _load_context(paths: InputPaths) -> DesignContext:
    xjtu_soh = paths.xjtu_frozen / "xjtu_55_soh_labels_long.csv"
    hust_soh = paths.hust_frozen / "hust_77_soh_labels_long.csv"
    xjtu_qref_path = paths.xjtu_frozen / "xjtu_55_qref_summary.csv"
    hust_qref_path = paths.hust_frozen / "hust_77_qref_summary.csv"
    cross_cell = paths.cross_frozen / "cross_dataset_cell_level.csv"
    _require_paths(
        [
            xjtu_soh,
            hust_soh,
            xjtu_qref_path,
            hust_qref_path,
            cross_cell,
            paths.xjtu_availability,
            paths.hust_availability,
        ]
    )

    xjtu_labels = _read_csv(xjtu_soh)
    hust_labels = _read_csv(hust_soh)
    xjtu_qref = _read_csv(xjtu_qref_path)
    hust_qref = _read_csv(hust_qref_path)

    for label, frame, columns in (
        (
            "XJTU labels",
            xjtu_labels,
            [
                "Batch_ID",
                "Cell_ID",
                "Raw_Cycle_Index",
                "Capacity_Ah",
                "Qref_Definition",
                "Qref_Ah",
                "SOH_Fraction",
            ],
        ),
        (
            "HUST labels",
            hust_labels,
            [
                "Cell_ID",
                "Raw_Cycle_Index",
                "Capacity_Ah",
                "Qref_Definition",
                "Qref_Ah",
                "SOH_Fraction",
            ],
        ),
    ):
        absent = [column for column in columns if column not in frame.columns]
        if absent:
            raise DesignError(f"{label} missing columns: {absent}")
        frame["Raw_Cycle_Index"] = _numeric(
            frame["Raw_Cycle_Index"], f"{label}.Raw_Cycle_Index"
        ).astype("int64")
        if frame["Cell_ID"].isna().any():
            raise DesignError(f"{label} contains empty Cell_ID")

    xjtu_common_keys = (
        xjtu_labels.loc[
            xjtu_labels["Qref_Definition"].eq("nominal"),
            ["Batch_ID", "Cell_ID", "Raw_Cycle_Index"],
        ]
        .drop_duplicates()
        .sort_values(["Batch_ID", "Cell_ID", "Raw_Cycle_Index"])
        .reset_index(drop=True)
    )
    hust_common_keys = (
        hust_labels.loc[
            hust_labels["Qref_Definition"].eq("nominal"),
            ["Cell_ID", "Raw_Cycle_Index"],
        ]
        .drop_duplicates()
        .sort_values(["Cell_ID", "Raw_Cycle_Index"])
        .reset_index(drop=True)
    )
    xjtu_initial_cells = sorted(
        xjtu_labels.loc[
            xjtu_labels["Qref_Definition"].eq(XJTU_SECONDARY_QREF), "Cell_ID"
        ].unique()
    )

    xjtu_availability = _read_csv(paths.xjtu_availability)
    hust_availability = _read_csv(paths.hust_availability)
    availability_stats = _availability_audit(
        xjtu_common_keys,
        hust_common_keys,
        xjtu_availability,
        hust_availability,
    )
    return DesignContext(
        paths=paths,
        xjtu_labels=xjtu_labels,
        hust_labels=hust_labels,
        xjtu_qref=xjtu_qref,
        hust_qref=hust_qref,
        xjtu_availability=xjtu_availability,
        hust_availability=hust_availability,
        xjtu_common_keys=xjtu_common_keys,
        hust_common_keys=hust_common_keys,
        xjtu_initial_cells=xjtu_initial_cells,
        availability_stats=availability_stats,
    )


def _availability_audit(
    xjtu_keys: pd.DataFrame,
    hust_keys: pd.DataFrame,
    xjtu_source: pd.DataFrame,
    hust_source: pd.DataFrame,
) -> dict[str, Any]:
    """Validate that every frozen observation has raw-curve audit evidence."""
    x_cols = [
        "Batch_ID",
        "Cell_ID",
        "Raw_Cycle_Index",
        "Charge_Max_V",
        "Charge_End_Current_A",
        "Charge_Duration_s",
        "Observed_Charge_Current_A",
        "Observed_Charge_Max_V",
        "Observed_Charge_End_Current_A",
        "Observed_Charge_Duration_s",
        "Observed_Rest_Segment_Count",
        "Observed_Total_Rest_Duration_s",
    ]
    missing_x = [column for column in x_cols if column not in xjtu_source.columns]
    if missing_x:
        raise DesignError(f"XJTU availability table missing columns: {missing_x}")
    xjtu_source = xjtu_source[x_cols].copy()
    xjtu_source["Raw_Cycle_Index"] = _numeric(
        xjtu_source["Raw_Cycle_Index"], "XJTU availability Raw_Cycle_Index"
    ).astype("int64")
    xm = xjtu_keys.merge(
        xjtu_source,
        on=["Batch_ID", "Cell_ID", "Raw_Cycle_Index"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not xm["_merge"].eq("both").all():
        raise DesignError("XJTU frozen observations do not all join to availability evidence")
    x_charge_max = xm["Charge_Max_V"].combine_first(xm["Observed_Charge_Max_V"])
    x_charge_end_current = xm["Charge_End_Current_A"].combine_first(
        xm["Observed_Charge_End_Current_A"]
    )
    x_charge_duration = xm["Charge_Duration_s"].combine_first(
        xm["Observed_Charge_Duration_s"]
    )
    if not (
        x_charge_max.notna().all()
        and x_charge_end_current.notna().all()
        and x_charge_duration.notna().all()
    ):
        raise DesignError("XJTU charge availability evidence is incomplete")

    h_cols = [
        "Cell_ID",
        "Raw_Cycle_Index",
        "Charge_Stage1_Median_Current_A",
        "Charge_Stage2_Plateau_Median_Current_A",
        "Charge_Max_V",
        "CV_End_Current_A",
        "Charge_Protocol_Traceable",
        "Full_Discharge_Traceable",
        "Native_Discharge_Capacity_mAh",
        "Native_RUL_Cycles",
    ]
    missing_h = [column for column in h_cols if column not in hust_source.columns]
    if missing_h:
        raise DesignError(f"HUST availability table missing columns: {missing_h}")
    hust_source = hust_source[h_cols].copy()
    hust_source["Raw_Cycle_Index"] = _numeric(
        hust_source["Raw_Cycle_Index"], "HUST availability Raw_Cycle_Index"
    ).astype("int64")
    hm = hust_keys.merge(
        hust_source,
        on=["Cell_ID", "Raw_Cycle_Index"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not hm["_merge"].eq("both").all():
        raise DesignError("HUST frozen observations do not all join to availability evidence")
    for column in (
        "Charge_Stage1_Median_Current_A",
        "Charge_Stage2_Plateau_Median_Current_A",
        "Charge_Max_V",
        "CV_End_Current_A",
    ):
        if not hm[column].notna().all():
            raise DesignError(f"HUST availability evidence incomplete for {column}")

    x_rest = xm["Observed_Rest_Segment_Count"].notna()
    return {
        "XJTU": {
            "cells": int(xjtu_keys["Cell_ID"].nunique()),
            "observations": int(len(xjtu_keys)),
            "charge_cells": int(xjtu_keys["Cell_ID"].nunique()),
            "charge_observations": int(len(xjtu_keys)),
            "rest_cells_precomputed": int(xm.loc[x_rest, "Cell_ID"].nunique()),
            "rest_observations_precomputed": int(x_rest.sum()),
        },
        "HUST": {
            "cells": int(hust_keys["Cell_ID"].nunique()),
            "observations": int(len(hust_keys)),
            "charge_cells": int(hust_keys["Cell_ID"].nunique()),
            "charge_observations": int(len(hust_keys)),
            "rest_cells_precomputed": 0,
            "rest_observations_precomputed": 0,
        },
    }


def assign_xjtu_folds(cells: pd.DataFrame) -> pd.DataFrame:
    """Deterministically balance XJTU protocol batches across five folds."""
    required = {"Batch_ID", "Cell_ID"}
    if not required.issubset(cells.columns):
        raise DesignError("XJTU fold input must contain Batch_ID and Cell_ID")
    unique = cells[["Batch_ID", "Cell_ID"]].drop_duplicates()
    if unique["Cell_ID"].duplicated().any():
        raise DesignError("An XJTU Cell_ID is associated with multiple batches")
    fold_total = {fold: 0 for fold in range(1, N_FOLDS + 1)}
    batch_fold: dict[tuple[Any, int], int] = {}
    assignments: list[dict[str, Any]] = []
    for batch_id in sorted(unique["Batch_ID"].unique(), key=lambda value: str(value)):
        group = unique.loc[unique["Batch_ID"].eq(batch_id), "Cell_ID"].tolist()
        group.sort(key=lambda cell: _stable_hash(f"{RANDOM_SEED}|XJTU|{batch_id}|{cell}"))
        for cell in group:
            fold = min(
                range(1, N_FOLDS + 1),
                key=lambda candidate: (
                    batch_fold.get((batch_id, candidate), 0),
                    fold_total[candidate],
                    _stable_hash(f"{RANDOM_SEED}|{cell}|{candidate}"),
                ),
            )
            batch_fold[(batch_id, fold)] = batch_fold.get((batch_id, fold), 0) + 1
            fold_total[fold] += 1
            assignments.append(
                {"Batch_ID": batch_id, "Cell_ID": cell, "Fold_ID": fold}
            )
    result = pd.DataFrame(assignments)
    return result.sort_values(["Fold_ID", "Batch_ID", "Cell_ID"]).reset_index(drop=True)


def assign_hust_folds(cells: Iterable[str]) -> pd.DataFrame:
    """Deterministically assign HUST cells without using cycle rows."""
    unique = sorted(set(cells), key=lambda cell: _stable_hash(f"{RANDOM_SEED}|HUST|{cell}"))
    assignments = [
        {"Cell_ID": cell, "Fold_ID": (index % N_FOLDS) + 1}
        for index, cell in enumerate(unique)
    ]
    return pd.DataFrame(assignments).sort_values(["Fold_ID", "Cell_ID"]).reset_index(
        drop=True
    )


def _feature_specs() -> list[dict[str, Any]]:
    """Return the frozen candidate feature catalogue.

    Availability is an audit of channels/validated precursor fields, not a
    claim that a final ML matrix has already been extracted.
    """
    specs: list[dict[str, Any]] = []

    def add(
        name: str,
        family: str,
        units: str,
        meaning: str,
        source_x: str,
        source_h: str,
        availability: str,
        raw: str,
        uses_target: str,
        future: str,
        proxy: str,
        protocol: str,
        harmonizable: str,
        risk: str,
        decision: str,
        extraction: str,
        absolute: str,
        normalization: str,
        notes: str,
    ) -> None:
        specs.append(
            {
                "Feature_Name": name,
                "Feature_Family": family,
                "Units": units,
                "Physical_Meaning": meaning,
                "Source_XJTU": source_x,
                "Source_HUST": source_h,
                "Availability_Mode": availability,
                "Requires_Raw_Time_Series": raw,
                "Uses_Target_Capacity": uses_target,
                "Uses_Future_Information": future,
                "Potential_Capacity_Proxy": proxy,
                "Protocol_Dependent": protocol,
                "Cross_Dataset_Harmonizable": harmonizable,
                "Leakage_Risk": risk,
                "Decision": decision,
                "Extraction_Readiness": extraction,
                "Absolute_Value_Comparable": absolute,
                "Required_Normalization": normalization,
                "Notes": notes,
            }
        )

    # Harmonized, pre-discharge physics features.
    add(
        "Pre_CV_CC_Median_Current_A",
        "CHARGE",
        "A",
        "Median current in the final constant-current stage immediately before CV.",
        "current_A|voltage_V|relative_time_min; main CC before CV",
        "current|voltage|status|time; low-rate CC stage immediately before CV",
        "FULL",
        "YES",
        "NO",
        "NO",
        "NO",
        "YES",
        "YES_CONCEPT_ONLY",
        "SAFE",
        "SAFE_CORE",
        "RAW_EXTRACTION_REQUIRED",
        "NO_PROTOCOL_SETPOINT_DIFFERS",
        "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET",
        "XJTU and HUST are modelled separately; harmonization is by physical stage, not equal C-rate.",
    )
    add(
        "CV_Median_Current_A",
        "CHARGE",
        "A",
        "Median positive current during the high-voltage CV taper.",
        "current_A|voltage_V|relative_time_min",
        "current|voltage|status|time; CV status",
        "FULL",
        "YES",
        "NO",
        "NO",
        "NO",
        "YES",
        "YES_CONCEPT_ONLY",
        "SAFE",
        "SAFE_CORE",
        "RAW_EXTRACTION_REQUIRED",
        "NO_PROTOCOL_SETPOINT_DIFFERS",
        "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET",
        "Positive-current CV samples only; never integrate charge quantity.",
    )
    add(
        "CV_End_Current_A",
        "CHARGE",
        "A",
        "Terminal positive current of the CV stage before rest/discharge.",
        "Charge_End_Current_A or Observed_Charge_End_Current_A",
        "CV_End_Current_A",
        "FULL",
        "NO",
        "NO",
        "NO",
        "NO",
        "YES",
        "YES_CONCEPT_ONLY",
        "SAFE",
        "SAFE_CORE",
        "AVAILABLE_IN_AUDIT_TABLE",
        "NO_PROTOCOL_CUTOFF_DIFFERS",
        "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET",
        "Protocol-completion feature measured before target discharge.",
    )
    add(
        "Charge_Start_V",
        "CHARGE",
        "V",
        "Voltage at the first positive-current sample of the main charge.",
        "voltage_V|current_A",
        "voltage|current|status",
        "FULL",
        "YES",
        "NO",
        "NO",
        "NO",
        "YES",
        "YES",
        "SAFE",
        "SAFE_CORE",
        "RAW_EXTRACTION_REQUIRED",
        "CONDITIONALLY_WITHIN_DATASET",
        "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET",
        "May reflect preceding protocol state; retain protocol metadata for audit only.",
    )
    add(
        "Charge_End_V",
        "CHARGE",
        "V",
        "Voltage at the final positive-current sample of the main charge.",
        "voltage_V|current_A",
        "voltage|current|status",
        "FULL",
        "YES",
        "NO",
        "NO",
        "NO",
        "YES",
        "YES",
        "SAFE",
        "SAFE_CORE",
        "RAW_EXTRACTION_REQUIRED",
        "CONDITIONALLY_WITHIN_DATASET",
        "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET",
        "Pre-discharge terminal charge voltage.",
    )
    add(
        "Charge_Max_V",
        "CHARGE",
        "V",
        "Maximum voltage during the main charge.",
        "Charge_Max_V or Observed_Charge_Max_V",
        "Charge_Max_V",
        "FULL",
        "NO",
        "NO",
        "NO",
        "NO",
        "YES",
        "YES",
        "SAFE",
        "SAFE_CORE",
        "AVAILABLE_IN_AUDIT_TABLE",
        "CONDITIONALLY_WITHIN_DATASET",
        "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET",
        "Used as observed, without protocol-normalizing away dataset differences.",
    )
    add(
        "CV_Duration_s",
        "CHARGE",
        "s",
        "Elapsed time from CV onset to the terminal CV sample.",
        "relative_time_min|voltage_V|current_A",
        "time|status|voltage|current",
        "FULL",
        "YES",
        "NO",
        "NO",
        "WEAK_CHARGE_PROCESS_PROXY",
        "YES",
        "YES_CONCEPT_ONLY",
        "LOW",
        "SAFE_CORE",
        "RAW_EXTRACTION_REQUIRED",
        "NO_SAMPLING_AND_PROTOCOL_DIFFER",
        "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET",
        "Pre-discharge feature; audited separately from prohibited full-discharge duration.",
    )
    add(
        "Charge_Voltage_Rise_Slope_V_per_s",
        "CHARGE",
        "V/s",
        "Robust voltage slope over the central 20-80% time span of the final CC stage.",
        "relative_time_min|voltage_V|current_A",
        "time|voltage|current|status",
        "FULL",
        "YES",
        "NO",
        "NO",
        "NO",
        "YES",
        "YES_CONCEPT_ONLY",
        "LOW",
        "SAFE_CORE",
        "RAW_EXTRACTION_REQUIRED",
        "NO_PROTOCOL_RATE_DIFFERS",
        "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET",
        "Freeze a robust linear-slope extractor before model training.",
    )
    add(
        "CV_Current_Taper_Slope_A_per_s",
        "CHARGE",
        "A/s",
        "Robust current slope over the central 20-80% time span of CV.",
        "relative_time_min|voltage_V|current_A",
        "time|voltage|current|status",
        "FULL",
        "YES",
        "NO",
        "NO",
        "NO",
        "YES",
        "YES_CONCEPT_ONLY",
        "LOW",
        "SAFE_CORE",
        "RAW_EXTRACTION_REQUIRED",
        "NO_PROTOCOL_RATE_DIFFERS",
        "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET",
        "No charge integration is permitted.",
    )
    add(
        "Charge_Terminal_Voltage_Std_V",
        "CHARGE",
        "V",
        "Voltage standard deviation in the final 30 seconds of CV.",
        "relative_time_min|voltage_V|current_A",
        "time|voltage|current|status",
        "FULL",
        "YES",
        "NO",
        "NO",
        "NO",
        "YES",
        "YES",
        "SAFE",
        "SAFE_CORE",
        "RAW_EXTRACTION_REQUIRED",
        "YES_AFTER_SAME_TIME_WINDOW",
        "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET",
        "Use a fixed physical-time window, not a fixed record count.",
    )
    add(
        "Pre_Discharge_Voltage_V",
        "PRE_DISCHARGE_STATE",
        "V",
        "Last voltage observed before the main negative-current discharge begins.",
        "relative_time_min|voltage_V|current_A",
        "time|voltage|current|status",
        "FULL",
        "YES",
        "NO",
        "NO",
        "NO",
        "YES",
        "YES_CONCEPT_ONLY",
        "LOW",
        "SAFE_CORE",
        "RAW_EXTRACTION_REQUIRED",
        "NO_REST_PROTOCOL_DIFFERS",
        "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET",
        "Physical timing is pre-discharge; document presence/absence of intervening rest.",
    )

    # Charging features deliberately kept outside the primary safe core.
    for name, meaning, source_x, source_h in (
        (
            "CC_Charge_Duration_s",
            "Duration of the complete final CC charge stage.",
            "relative_time_min|current_A|voltage_V",
            "time|current|voltage|status",
        ),
        (
            "Total_Charge_Duration_s",
            "Duration of the complete charge process.",
            "Charge_Duration_s or Observed_Charge_Duration_s",
            "time|status",
        ),
        (
            "Time_To_Charge_4p10V_s",
            "Elapsed charge time to first reach 4.10 V.",
            "relative_time_min|voltage_V|current_A",
            "time|voltage|current|status",
        ),
    ):
        add(
            name,
            "CHARGE",
            "s",
            meaning,
            source_x,
            source_h,
            "FULL",
            "YES",
            "NO",
            "NO",
            "YES_CHARGE_THROUGHPUT_PROXY",
            "YES",
            "YES_CONCEPT_ONLY",
            "MEDIUM",
            "SECONDARY_ONLY",
            "RAW_EXTRACTION_REQUIRED",
            "NO",
            "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET",
            "Pre-discharge but strongly related to charged throughput; excluded from primary SAFE_CORE.",
        )

    # XJTU-specific rest family.  Existing audited derivatives cover batches 2-6;
    # Batch-1 would require a fresh read-only raw extraction after approval.
    for name, units, meaning in (
        ("Rest_Duration_s", "s", "Duration of the final rest immediately before discharge."),
        ("Pre_Discharge_Rest_End_V", "V", "Voltage at the end of the pre-discharge rest."),
        ("Rest_Relaxation_Delta_V", "V", "End-minus-start voltage across the pre-discharge rest."),
        (
            "Rest_Relaxation_Slope_V_per_s",
            "V/s",
            "Robust voltage slope during the pre-discharge rest.",
        ),
    ):
        add(
            name,
            "REST",
            units,
            meaning,
            "relative_time_min|voltage_V|current_A; validated rest segments",
            "No rest status in audited public raw protocol",
            "REST_PRECOMPUTED",
            "YES",
            "NO",
            "NO",
            "NO",
            "YES",
            "NO",
            "LOW",
            "SAFE_EXTENDED",
            "XJTU_PARTIAL_AUDIT_HUST_UNAVAILABLE",
            "NOT_CROSS_DATASET",
            "TRAIN_FOLD_STANDARDIZATION_WITHIN_XJTU",
            "XJTU secondary only; do not fabricate a HUST rest feature.",
        )

    # Early/partial discharge features are not full-capacity measurements, but
    # stay outside the primary core until their windows are validated.
    for name, units, meaning in (
        (
            "Early_Discharge_Voltage_30s",
            "V",
            "Voltage 30 seconds after main-discharge onset.",
        ),
        (
            "Early_Discharge_Voltage_60s",
            "V",
            "Voltage 60 seconds after main-discharge onset.",
        ),
        (
            "Early_Discharge_Voltage_Slope_0_60s",
            "V/s",
            "Robust voltage slope over the first 60 seconds of discharge.",
        ),
        (
            "Discharge_Onset_Voltage_Drop_10s",
            "V",
            "Voltage change over the first 10 seconds of discharge.",
        ),
        (
            "Early_Discharge_Median_Current_0_60s",
            "A",
            "Median current over the first 60 seconds of main discharge.",
        ),
    ):
        add(
            name,
            "PARTIAL_DISCHARGE",
            units,
            meaning,
            "relative_time_min|voltage_V|current_A",
            "time|voltage|current|status",
            "FULL",
            "YES",
            "NO",
            "NO",
            "LOW_FIXED_EARLY_WINDOW",
            "YES",
            "YES_CONCEPT_ONLY",
            "LOW",
            "SAFE_EXTENDED",
            "RAW_EXTRACTION_REQUIRED",
            "NO_PROTOCOL_RATE_DIFFERS",
            "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET",
            "Fixed clock-time window only; never define the window as a fraction of full discharge.",
        )

    # Metadata sensitivity features.
    add(
        "Raw_Cycle_Index",
        "METADATA",
        "cycle",
        "Recorded cycle/diagnostic index used as an explicit aging shortcut sensitivity.",
        "Raw_Cycle_Index",
        "Raw_Cycle_Index",
        "FULL",
        "NO",
        "NO",
        "NO",
        "AGING_SHORTCUT",
        "YES",
        "YES",
        "MEDIUM",
        "SECONDARY_ONLY",
        "AVAILABLE_IN_FROZEN_LABEL_TABLE",
        "YES_WITHIN_DATASET",
        "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET",
        "Included only in FEATURE_SET_B; excluded from primary FEATURE_SET_A.",
    )
    add(
        "Batch_ID",
        "METADATA",
        "category",
        "XJTU protocol/batch group identifier.",
        "Batch_ID",
        "Not applicable",
        "XJTU_FULL",
        "NO",
        "NO",
        "NO",
        "PROTOCOL_SHORTCUT",
        "YES",
        "NO",
        "MEDIUM",
        "SECONDARY_ONLY",
        "AVAILABLE_IN_FROZEN_LABEL_TABLE",
        "NOT_CROSS_DATASET",
        "TRAIN_FOLD_CATEGORY_MAP_ONLY",
        "Used for fold balance/audit; not part of primary feature sets.",
    )
    add(
        "Protocol_Fingerprint_ID",
        "METADATA",
        "category",
        "HUST raw protocol fingerprint identifier.",
        "Not applicable",
        "Raw_Status_Sequence and stage currents",
        "HUST_FULL",
        "NO",
        "NO",
        "NO",
        "PROTOCOL_SHORTCUT",
        "YES",
        "NO",
        "MEDIUM",
        "SECONDARY_ONLY",
        "DERIVABLE_FROM_AUDIT_TABLE",
        "NOT_CROSS_DATASET",
        "TRAIN_FOLD_CATEGORY_MAP_ONLY",
        "Record for audit; do not force exact fingerprint stratification.",
    )

    # Explicit leakage/proxy audit rows.
    prohibited = (
        (
            "Primary_Capacity_Ah",
            "TARGET",
            "Ah",
            "Discharge capacity that mathematically defines the SOH target.",
            "TARGET_FULL",
            "YES",
            "NO",
            "DIRECT_TARGET",
            "PROHIBITED",
        ),
        (
            "Native_Discharge_Capacity",
            "NATIVE_LABEL",
            "Ah or mAh",
            "Dataset-native discharge capacity/dq measurement.",
            "TARGET_FULL",
            "YES",
            "NO",
            "DIRECT_TARGET_OR_EQUIVALENT",
            "PROHIBITED",
        ),
        (
            "Charge_Capacity_Ah",
            "CAPACITY_PROXY",
            "Ah",
            "Integrated charge capacity when mathematically tied to the target quantity.",
            "NOT_EXTRACTED",
            "YES",
            "NO",
            "NEAR_DIRECT_CAPACITY_PROXY",
            "PROHIBITED",
        ),
        (
            "SOH_Fraction",
            "TARGET",
            "fraction",
            "Reference-normalized SOH target.",
            "SOH_FULL",
            "YES",
            "NO",
            "DIRECT_TARGET",
            "PROHIBITED",
        ),
        (
            "Native_SOH",
            "NATIVE_LABEL",
            "fraction or percent",
            "Any dataset-native SOH label.",
            "NOT_EXTRACTED",
            "YES",
            "NO",
            "DIRECT_OR_ALTERNATE_TARGET",
            "PROHIBITED",
        ),
        (
            "Qref_Ah",
            "LABEL_DEFINITION",
            "Ah",
            "Reference capacity used to define the target label.",
            "QREF_FULL",
            "NO",
            "NO",
            "LABEL_DEFINITION_INPUT",
            "PROHIBITED",
        ),
        (
            "Native_RUL_Cycles",
            "FUTURE_LABEL",
            "cycle",
            "Dataset-native remaining useful life.",
            "NATIVE_RUL",
            "NO",
            "YES",
            "FUTURE_LIFE_LABEL",
            "PROHIBITED",
        ),
        (
            "EOL_Cycle_or_Status",
            "FUTURE_LABEL",
            "cycle/status",
            "Qref-defined EOL cycle, interval, censoring, or status.",
            "EOL_FULL",
            "NO",
            "YES",
            "FUTURE_LIFE_LABEL",
            "PROHIBITED",
        ),
        (
            "Final_or_Last_Observed_Cycle",
            "FUTURE_LABEL",
            "cycle",
            "Final cycle index or last observed cycle for the cell.",
            "EOL_FULL",
            "NO",
            "YES",
            "FUTURE_LIFE_LABEL",
            "PROHIBITED",
        ),
        (
            "Capacity_Retention_or_Normalized_Capacity",
            "TARGET_TRANSFORM",
            "fraction",
            "Capacity divided by nominal, initial, first, or other reference.",
            "DERIVABLE_TARGET",
            "YES",
            "NO",
            "TRANSFORMED_TARGET",
            "PROHIBITED",
        ),
        (
            "Capacity_Fade_or_Difference",
            "TARGET_TRANSFORM",
            "Ah",
            "Difference between current capacity and a prior/reference capacity.",
            "DERIVABLE_TARGET",
            "YES",
            "NO",
            "TRANSFORMED_TARGET",
            "PROHIBITED",
        ),
        (
            "Rolling_or_Smoothed_Capacity",
            "TARGET_TRANSFORM",
            "Ah",
            "Rolling, smoothed, interpolated, or filtered target capacity.",
            "DERIVABLE_TARGET",
            "YES",
            "POSSIBLE",
            "TRANSFORMED_TARGET",
            "PROHIBITED",
        ),
        (
            "Capacity_Degradation_Slope",
            "TARGET_TRANSFORM",
            "Ah/cycle",
            "Slope or derivative computed from target capacity history/future.",
            "DERIVABLE_TARGET",
            "YES",
            "POSSIBLE",
            "TRANSFORMED_TARGET",
            "PROHIBITED",
        ),
        (
            "Full_Discharge_Duration_s",
            "FULL_DISCHARGE_PROXY",
            "s",
            "Duration of the complete near-constant-current target discharge.",
            "FULL",
            "NO",
            "NO",
            "HIGH_LEAKAGE_RISK_CAPACITY_PROXY",
            "HIGH",
        ),
        (
            "Integrated_Discharge_Current_Ah",
            "FULL_DISCHARGE_PROXY",
            "Ah",
            "Time integral of current across the target discharge.",
            "DERIVABLE_TARGET",
            "YES",
            "NO",
            "MATHEMATICALLY_EQUIVALENT_TO_CAPACITY",
            "PROHIBITED",
        ),
        (
            "Total_Discharge_Energy_Wh",
            "FULL_DISCHARGE_PROXY",
            "Wh",
            "Integrated energy across the complete target discharge.",
            "DERIVABLE_TARGET",
            "YES",
            "NO",
            "HIGH_LEAKAGE_RISK_CAPACITY_PROXY",
            "PROHIBITED",
        ),
        (
            "Full_Discharge_Stage_Duration_Sum_s",
            "FULL_DISCHARGE_PROXY",
            "s",
            "Sum of durations over all complete discharge stages.",
            "DERIVABLE_TARGET",
            "NO",
            "NO",
            "HIGH_LEAKAGE_RISK_CAPACITY_PROXY",
            "HIGH",
        ),
        (
            "Time_To_Terminal_Discharge_Voltage_s",
            "FULL_DISCHARGE_PROXY",
            "s",
            "Elapsed time from discharge onset to terminal voltage.",
            "DERIVABLE_TARGET",
            "NO",
            "NO",
            "HIGH_LEAKAGE_RISK_CAPACITY_PROXY",
            "HIGH",
        ),
        (
            "Voltage_at_Fraction_of_Full_Discharge",
            "FULL_DISCHARGE_PROXY",
            "V",
            "Voltage at a time defined using a fraction of eventual full-discharge duration.",
            "DERIVABLE_TARGET",
            "NO",
            "YES",
            "USES_TARGET_COMPLETION_TO_DEFINE_WINDOW",
            "PROHIBITED",
        ),
    )
    for name, family, units, meaning, availability, uses_target, future, proxy, risk in prohibited:
        add(
            name,
            family,
            units,
            meaning,
            "Frozen labels/raw curve/derived series as applicable",
            "Frozen labels/raw curve/derived series as applicable",
            availability,
            "VARIES",
            uses_target,
            future,
            proxy,
            "VARIES",
            "NO",
            risk,
            "PROHIBITED",
            "INTENTIONALLY_NOT_EXTRACTED_FOR_ML",
            "NOT_APPLICABLE",
            "NOT_PERMITTED",
            "Audit-only row; transformed variants are equally prohibited.",
        )
    return specs


def _availability_for(
    mode: str, dataset: str, stats: dict[str, Any]
) -> tuple[int, int, str]:
    ds = stats[dataset]
    full = (ds["cells"], ds["observations"], "FROZEN_KEYS_PLUS_VALIDATED_RAW_CHANNEL_AUDIT")
    if mode in {"FULL", "TARGET_FULL", "SOH_FULL", "QREF_FULL", "EOL_FULL", "DERIVABLE_TARGET"}:
        return full
    if mode == "REST_PRECOMPUTED":
        if dataset == "XJTU":
            return (
                ds["rest_cells_precomputed"],
                ds["rest_observations_precomputed"],
                "PRECOMPUTED_FOR_XJTU_BATCHES_2_TO_6; BATCH1_RAW_REEXTRACTION_DEFERRED",
            )
        return (0, 0, "NO_REST_STATUS_IN_AUDITED_HUST_PUBLIC_RAW_PROTOCOL")
    if mode == "XJTU_FULL":
        return full if dataset == "XJTU" else (0, 0, "NOT_APPLICABLE")
    if mode == "HUST_FULL":
        return full if dataset == "HUST" else (0, 0, "NOT_APPLICABLE")
    if mode == "NATIVE_RUL":
        return full if dataset == "HUST" else (0, 0, "NO_NATIVE_RUL_IN_XJTU_FROZEN_LABELS")
    if mode == "NOT_EXTRACTED":
        return (0, 0, "PROHIBITED_FEATURE_NOT_EXTRACTED")
    raise DesignError(f"Unknown availability mode: {mode}")


def build_feature_inventory(ctx: DesignContext) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in _feature_specs():
        for dataset in ("XJTU", "HUST"):
            cells, observations, evidence = _availability_for(
                spec["Availability_Mode"], dataset, ctx.availability_stats
            )
            total = ctx.availability_stats[dataset]["observations"]
            source = spec["Source_XJTU"] if dataset == "XJTU" else spec["Source_HUST"]
            row = {
                "Dataset": dataset,
                "Feature_Name": spec["Feature_Name"],
                "Feature_Family": spec["Feature_Family"],
                "Raw_Source_Channels": source,
                "Physical_Meaning": spec["Physical_Meaning"],
                "Units": spec["Units"],
                "Available_Cell_Count": cells,
                "Available_Observation_Count": observations,
                "Missing_Rate": 1.0 - (observations / total if total else 0.0),
                "Requires_Raw_Time_Series": spec["Requires_Raw_Time_Series"],
                "Uses_Target_Capacity": spec["Uses_Target_Capacity"],
                "Uses_Future_Information": spec["Uses_Future_Information"],
                "Potential_Capacity_Proxy": spec["Potential_Capacity_Proxy"],
                "Protocol_Dependent": spec["Protocol_Dependent"],
                "Cross_Dataset_Harmonizable": spec["Cross_Dataset_Harmonizable"],
                "Leakage_Risk": spec["Leakage_Risk"],
                "Decision": spec["Decision"],
                "Extraction_Readiness": spec["Extraction_Readiness"],
                "Availability_Evidence": evidence,
                "Absolute_Value_Comparable": spec["Absolute_Value_Comparable"],
                "Required_Normalization": spec["Required_Normalization"],
                "Notes": spec["Notes"],
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["Dataset", "Feature_Family", "Feature_Name"]
    ).reset_index(drop=True)


def build_leakage_audit(inventory: pd.DataFrame) -> pd.DataFrame:
    audit = inventory[
        [
            "Dataset",
            "Feature_Name",
            "Feature_Family",
            "Uses_Target_Capacity",
            "Uses_Future_Information",
            "Potential_Capacity_Proxy",
            "Leakage_Risk",
            "Decision",
        ]
    ].copy()

    def path(row: pd.Series) -> str:
        if row["Uses_Target_Capacity"] == "YES":
            return "DIRECT_OR_TRANSFORMED_TARGET_PATH"
        if row["Uses_Future_Information"] in {"YES", "POSSIBLE"}:
            return "FUTURE_INFORMATION_PATH"
        if row["Potential_Capacity_Proxy"] not in {"NO", "LOW_FIXED_EARLY_WINDOW"}:
            return "CAPACITY_PROXY_PATH"
        if row["Feature_Name"] == "Raw_Cycle_Index":
            return "AGING_SHORTCUT_PATH"
        return "NO_IDENTIFIED_LABEL_LEAKAGE_PATH"

    audit["Leakage_Path"] = audit.apply(path, axis=1)
    audit["Timing_Relative_To_Target_Full_Discharge"] = np.select(
        [
            audit["Feature_Family"].isin(
                ["CHARGE", "REST", "PRE_DISCHARGE_STATE", "METADATA"]
            ),
            audit["Feature_Family"].eq("PARTIAL_DISCHARGE"),
            audit["Feature_Family"].isin(
                ["TARGET", "TARGET_TRANSFORM", "FULL_DISCHARGE_PROXY", "FUTURE_LABEL"]
            ),
        ],
        ["BEFORE", "EARLY_DURING", "AT_OR_AFTER_TARGET_COMPLETION"],
        default="VARIES",
    )
    audit["Primary_Feature_Eligible"] = np.where(
        audit["Decision"].eq("SAFE_CORE"), "YES", "NO"
    )
    audit["Transformed_Variants_Prohibited"] = np.where(
        audit["Decision"].eq("PROHIBITED"), "YES", "NOT_APPLICABLE"
    )
    audit["Required_Control"] = np.select(
        [
            audit["Decision"].eq("PROHIBITED"),
            audit["Decision"].eq("SECONDARY_ONLY"),
            audit["Decision"].eq("SAFE_EXTENDED"),
        ],
        [
            "EXCLUDE_FROM_ALL_MODEL_INPUTS",
            "SEPARATE_PREDECLARED_SENSITIVITY_ONLY",
            "VALIDATE_FIXED_EARLY_OR_DATASET_SPECIFIC_EXTRACTION_BEFORE_USE",
        ],
        default="ALLOW_IN_FEATURE_SET_A_AND_B_AFTER_FROZEN_EXTRACTION",
    )
    audit["Audit_Result"] = np.where(
        (audit["Decision"].eq("SAFE_CORE"))
        & (
            audit["Uses_Target_Capacity"].eq("YES")
            | audit["Uses_Future_Information"].isin(["YES", "POSSIBLE"])
            | audit["Leakage_Risk"].isin(["HIGH", "PROHIBITED"])
        ),
        "FAIL",
        "PASS",
    )
    audit["Audit_Rationale"] = np.where(
        audit["Audit_Result"].eq("PASS"),
        "Decision is consistent with the identified leakage path and timing.",
        "Unsafe feature entered SAFE_CORE.",
    )
    columns = [
        "Dataset",
        "Feature_Name",
        "Feature_Family",
        "Timing_Relative_To_Target_Full_Discharge",
        "Uses_Target_Capacity",
        "Uses_Future_Information",
        "Potential_Capacity_Proxy",
        "Leakage_Path",
        "Leakage_Risk",
        "Decision",
        "Primary_Feature_Eligible",
        "Transformed_Variants_Prohibited",
        "Required_Control",
        "Audit_Result",
        "Audit_Rationale",
    ]
    return audit[columns].sort_values(["Dataset", "Feature_Name"]).reset_index(drop=True)


def build_harmonized_feature_plan(inventory: pd.DataFrame) -> pd.DataFrame:
    safe = inventory.loc[inventory["Decision"].eq("SAFE_CORE")]
    coverage = safe.groupby("Feature_Name").agg(
        Dataset_Count=("Dataset", "nunique"),
        Min_Cell_Count=("Available_Cell_Count", "min"),
        Min_Observation_Count=("Available_Observation_Count", "min"),
    )
    core_names = sorted(
        coverage.loc[
            coverage["Dataset_Count"].eq(2)
            & coverage["Min_Cell_Count"].gt(0)
            & coverage["Min_Observation_Count"].gt(0)
        ].index
    )
    if not core_names:
        raise DesignError("ML DESIGN NEEDS FEATURE REVIEW: no harmonized SAFE_CORE")

    exemplar = (
        inventory.loc[
            inventory["Feature_Name"].isin(core_names),
            [
                "Feature_Name",
                "Feature_Family",
                "Units",
                "Physical_Meaning",
                "Protocol_Dependent",
                "Required_Normalization",
                "Requires_Raw_Time_Series",
            ],
        ]
        .drop_duplicates("Feature_Name")
        .set_index("Feature_Name")
    )
    rows: list[dict[str, Any]] = []
    for feature_set, is_primary, names in (
        ("FEATURE_SET_A_PHYSICS_ONLY", "YES", core_names),
        ("FEATURE_SET_B_PHYSICS_PLUS_CYCLE_INDEX", "NO", core_names + ["Raw_Cycle_Index"]),
    ):
        for order, name in enumerate(names, start=1):
            if name == "Raw_Cycle_Index":
                family = "METADATA"
                units = "cycle"
                meaning = "Explicit aging-proxy sensitivity feature."
                protocol = "YES"
                normalization = "TRAIN_FOLD_STANDARDIZATION_WITHIN_DATASET"
                raw = "NO"
                sensitivity_role = "CYCLE_INDEX_SENSITIVITY_ONLY"
            else:
                item = exemplar.loc[name]
                family = item["Feature_Family"]
                units = item["Units"]
                meaning = item["Physical_Meaning"]
                protocol = item["Protocol_Dependent"]
                normalization = item["Required_Normalization"]
                raw = item["Requires_Raw_Time_Series"]
                sensitivity_role = "PRIMARY_PHYSICS_FEATURE"
            rows.append(
                {
                    "Feature_Set_ID": feature_set,
                    "Primary_Feature_Set": is_primary,
                    "Feature_Order": order,
                    "Feature_Name": name,
                    "Feature_Family": family,
                    "Units": units,
                    "Physical_Meaning": meaning,
                    "Available_In_XJTU": "YES",
                    "Available_In_HUST": "YES",
                    "Datasets_Trained_Separately": "YES",
                    "Requires_Raw_Time_Series": raw,
                    "Protocol_Dependent": protocol,
                    "Normalization_Policy": normalization,
                    "Qref_Independent_Input": "YES",
                    "Feature_Set_Role": sensitivity_role,
                    "Frozen_Status": "DESIGN_FROZEN_PENDING_HUMAN_APPROVAL",
                }
            )
    return pd.DataFrame(rows)


def build_dataset_specific_plan() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        dataset: str,
        feature: str,
        family: str,
        decision: str,
        availability: str,
        extraction: str,
        reason: str,
        population: str = "SAME_QREF_INDEPENDENT_PHYSICAL_OBSERVATIONS",
    ) -> None:
        rows.append(
            {
                "Dataset": dataset,
                "Plan_ID": "DATASET_SPECIFIC_SAFE_EXTENDED",
                "Feature_Name": feature,
                "Feature_Family": family,
                "Decision": decision,
                "Availability_Status": availability,
                "Extraction_Policy": extraction,
                "Observation_Population_Policy": population,
                "Preprocessing_Policy": "FIT_ON_TRAINING_CELLS_ONLY",
                "Qref_Independent_Input": "YES",
                "Scientific_Rationale": reason,
                "Primary_Result_Eligible": "NO",
                "Frozen_Status": "PLANNED_SECONDARY_ANALYSIS",
            }
        )

    for feature in (
        "Rest_Duration_s",
        "Pre_Discharge_Rest_End_V",
        "Rest_Relaxation_Delta_V",
        "Rest_Relaxation_Slope_V_per_s",
    ):
        add(
            "XJTU",
            feature,
            "REST",
            "SAFE_EXTENDED",
            "PRECOMPUTED_BATCHES_2_TO_6; BATCH1_RAW_EXTRACTION_REQUIRED",
            "DO_NOT_USE_UNTIL_FIXED_EXTRACTOR_AND_MISSINGNESS_AUDIT_ARE_APPROVED",
            "XJTU has a documented pre-discharge rest; HUST audited protocol does not.",
        )
    for dataset in ("XJTU", "HUST"):
        for feature in (
            "Early_Discharge_Voltage_30s",
            "Early_Discharge_Voltage_60s",
            "Early_Discharge_Voltage_Slope_0_60s",
            "Discharge_Onset_Voltage_Drop_10s",
            "Early_Discharge_Median_Current_0_60s",
        ):
            add(
                dataset,
                feature,
                "PARTIAL_DISCHARGE",
                "SAFE_EXTENDED",
                "RAW_CHANNELS_AVAILABLE_FOR_FROZEN_OBSERVATIONS",
                "FIXED_CLOCK_TIME_WINDOWS; NO_FULL_DISCHARGE_FRACTION_WINDOWS",
                "Early response is not the completed capacity, but remains secondary pending extraction validation.",
            )
    for feature in ("CC_Charge_Duration_s", "Total_Charge_Duration_s", "Time_To_Charge_4p10V_s"):
        for dataset in ("XJTU", "HUST"):
            add(
                dataset,
                feature,
                "CHARGE",
                "SECONDARY_ONLY",
                "RAW_CHANNELS_AVAILABLE_FOR_FROZEN_OBSERVATIONS",
                "AUDIT_CORRELATION_WITH_TARGET_CAPACITY_BEFORE_ANY_USE",
                "Pre-discharge timing can approximate charged throughput and is excluded from SAFE_CORE.",
            )
    add(
        "HUST",
        "Charge_Stage1_Median_Current_A",
        "CHARGE",
        "SAFE_EXTENDED",
        "AVAILABLE_IN_HUST_SCAN_AUDIT",
        "USE_ONLY_IN_HUST_SECONDARY_MODEL",
        "HUST-specific high-rate first charge stage has no XJTU analogue.",
    )
    add(
        "HUST",
        "Protocol_Fingerprint_ID",
        "METADATA",
        "SECONDARY_ONLY",
        "DERIVABLE_FROM_HUST_SCAN_AUDIT",
        "AUDIT_OR_EXPLICIT_PROTOCOL_SENSITIVITY_ONLY",
        "Potential protocol shortcut; record but do not include in primary models.",
    )
    add(
        "XJTU",
        "Batch_ID",
        "METADATA",
        "SECONDARY_ONLY",
        "AVAILABLE_IN_FROZEN_LABELS",
        "FOLD_BALANCING_AND_AUDIT_ONLY",
        "Potential protocol shortcut; not a primary predictor.",
    )
    return pd.DataFrame(rows).sort_values(["Dataset", "Feature_Family", "Feature_Name"])


def build_observation_intersection_audit(ctx: DesignContext) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_pair(
        dataset: str,
        frame: pd.DataFrame,
        keys: list[str],
        definition_a: str,
        definition_b: str,
        population: str,
        allowed_cells: set[str] | None = None,
    ) -> None:
        subset = frame
        if allowed_cells is not None:
            subset = subset.loc[subset["Cell_ID"].isin(allowed_cells)]
        a = subset.loc[subset["Qref_Definition"].eq(definition_a)]
        b = subset.loc[subset["Qref_Definition"].eq(definition_b)]
        a_set = _key_set(a, keys)
        b_set = _key_set(b, keys)
        intersection = a_set & b_set
        rows.append(
            {
                "Dataset": dataset,
                "Comparison_ID": f"{dataset}_{definition_a}_VS_{definition_b}",
                "Population_ID": population,
                "Definition_A": definition_a,
                "Definition_B": definition_b,
                "Key_Columns": "|".join(keys),
                "Cell_Count_A": int(a["Cell_ID"].nunique()),
                "Cell_Count_B": int(b["Cell_ID"].nunique()),
                "Observation_Count_A": len(a_set),
                "Observation_Count_B": len(b_set),
                "Intersection_Count": len(intersection),
                "Only_A_Count": len(a_set - b_set),
                "Only_B_Count": len(b_set - a_set),
                "Duplicate_Key_Count_A": int(a.duplicated(keys).sum()),
                "Duplicate_Key_Count_B": int(b.duplicated(keys).sum()),
                "Physical_Observation_Keys_Identical": "YES" if a_set == b_set else "NO",
                "Definition_Specific_EOL_Truncation": "NO",
                "Audit_Status": "PASS" if a_set == b_set and a_set else "FAIL",
                "Notes": "Only Qref-derived label values may differ.",
            }
        )

    for dataset, frame, keys in (
        ("XJTU", ctx.xjtu_labels, ["Batch_ID", "Cell_ID", "Raw_Cycle_Index"]),
        ("HUST", ctx.hust_labels, ["Cell_ID", "Raw_Cycle_Index"]),
    ):
        add_pair(dataset, frame, keys, COMMON_QREFS[0], COMMON_QREFS[1], f"{dataset}_COMMON_PRIMARY")
        add_pair(dataset, frame, keys, COMMON_QREFS[0], COMMON_QREFS[2], f"{dataset}_COMMON_PRIMARY")
        add_pair(dataset, frame, keys, COMMON_QREFS[1], COMMON_QREFS[2], f"{dataset}_COMMON_PRIMARY")
    add_pair(
        "XJTU",
        ctx.xjtu_labels,
        ["Batch_ID", "Cell_ID", "Raw_Cycle_Index"],
        "nominal",
        XJTU_SECONDARY_QREF,
        "XJTU_COMMON_47_INITIAL_TEST",
        set(ctx.xjtu_initial_cells),
    )
    rows.append(
        {
            "Dataset": "HUST",
            "Comparison_ID": "HUST_INITIAL_TEST_AVAILABILITY",
            "Population_ID": "NOT_APPLICABLE",
            "Definition_A": "initial_test",
            "Definition_B": "NOT_CREATED",
            "Key_Columns": "Cell_ID|Raw_Cycle_Index",
            "Cell_Count_A": 0,
            "Cell_Count_B": 0,
            "Observation_Count_A": 0,
            "Observation_Count_B": 0,
            "Intersection_Count": 0,
            "Only_A_Count": 0,
            "Only_B_Count": 0,
            "Duplicate_Key_Count_A": 0,
            "Duplicate_Key_Count_B": 0,
            "Physical_Observation_Keys_Identical": "NOT_APPLICABLE",
            "Definition_Specific_EOL_Truncation": "NO",
            "Audit_Status": "PASS",
            "Notes": "HUST initial_test is unavailable and was not invented.",
        }
    )
    return pd.DataFrame(rows).sort_values(["Dataset", "Comparison_ID"]).reset_index(drop=True)


def _format_xjtu_fold_assignments(ctx: DesignContext) -> pd.DataFrame:
    cells = ctx.xjtu_common_keys[["Batch_ID", "Cell_ID"]].drop_duplicates()
    folds = assign_xjtu_folds(cells)
    observation_counts = ctx.xjtu_common_keys.groupby("Cell_ID").size().rename(
        "Observation_Count_Common_Qrefs"
    )
    folds = folds.merge(observation_counts, on="Cell_ID", validate="one_to_one")
    folds.insert(0, "Dataset", "XJTU")
    folds["Held_Out_When_Fold"] = folds["Fold_ID"]
    folds["Training_Folds"] = folds["Fold_ID"].map(
        lambda held: "|".join(str(fold) for fold in range(1, N_FOLDS + 1) if fold != held)
    )
    folds["Fold_Cell_Count"] = folds.groupby("Fold_ID")["Cell_ID"].transform("size")
    folds["Split_Unit"] = "Cell_ID"
    folds["CV_Design"] = "DETERMINISTIC_5_FOLD_GROUPED_CV"
    folds["Assignment_Method"] = "GREEDY_BATCH_BALANCE_WITH_SHA256_TIE_BREAK"
    folds["Random_Seed"] = RANDOM_SEED
    folds["Applicable_Qref_Definitions"] = "|".join(COMMON_QREFS)
    folds["Observation_Population_ID"] = "XJTU_COMMON_PRIMARY_55_CELL"
    folds["Cycles_From_Cell_Can_Cross_Folds"] = "NO"
    folds["No_Cell_Leakage_Verified"] = "YES"
    return folds.sort_values(["Fold_ID", "Batch_ID", "Cell_ID"]).reset_index(drop=True)


def _format_hust_fold_assignments(ctx: DesignContext) -> pd.DataFrame:
    folds = assign_hust_folds(ctx.hust_common_keys["Cell_ID"].unique())
    observation_counts = ctx.hust_common_keys.groupby("Cell_ID").size().rename(
        "Observation_Count_Common_Qrefs"
    )
    folds = folds.merge(observation_counts, on="Cell_ID", validate="one_to_one")
    folds.insert(0, "Dataset", "HUST")
    folds["Held_Out_When_Fold"] = folds["Fold_ID"]
    folds["Training_Folds"] = folds["Fold_ID"].map(
        lambda held: "|".join(str(fold) for fold in range(1, N_FOLDS + 1) if fold != held)
    )
    folds["Fold_Cell_Count"] = folds.groupby("Fold_ID")["Cell_ID"].transform("size")
    folds["Split_Unit"] = "Cell_ID"
    folds["CV_Design"] = "DETERMINISTIC_5_FOLD_GROUPED_CV"
    folds["Assignment_Method"] = "SHA256_SEEDED_CELL_ORDER_ROUND_ROBIN"
    folds["Random_Seed"] = RANDOM_SEED
    folds["Applicable_Qref_Definitions"] = "|".join(COMMON_QREFS)
    folds["Observation_Population_ID"] = "HUST_COMMON_PRIMARY_77_CELL"
    folds["Protocol_Stratification"] = "RECORDED_NOT_FORCED"
    folds["Cycles_From_Cell_Can_Cross_Folds"] = "NO"
    folds["No_Cell_Leakage_Verified"] = "YES"
    return folds.sort_values(["Fold_ID", "Cell_ID"]).reset_index(drop=True)


def _format_xjtu_initial_folds(
    ctx: DesignContext, xjtu_folds: pd.DataFrame
) -> pd.DataFrame:
    cells = set(ctx.xjtu_initial_cells)
    result = xjtu_folds.loc[xjtu_folds["Cell_ID"].isin(cells)].copy()
    initial_counts = (
        ctx.xjtu_labels.loc[
            ctx.xjtu_labels["Qref_Definition"].eq(XJTU_SECONDARY_QREF)
        ]
        .groupby("Cell_ID")
        .size()
        .rename("Observation_Count_Common_47_Subset")
    )
    result = result.drop(columns=["Observation_Count_Common_Qrefs"]).merge(
        initial_counts, on="Cell_ID", validate="one_to_one"
    )
    result["Fold_Cell_Count"] = result.groupby("Fold_ID")["Cell_ID"].transform("size")
    result["Applicable_Qref_Definitions"] = "nominal|initial_test"
    result["Observation_Population_ID"] = "XJTU_COMMON_47_INITIAL_TEST"
    result["Assignment_Method"] = "INHERITED_FROM_XJTU_55_CELL_PRIMARY_ASSIGNMENT"
    result["Common_47_Cell_Subset_Verified"] = "YES"
    return result.sort_values(["Fold_ID", "Batch_ID", "Cell_ID"]).reset_index(drop=True)


def build_label_definition_plan(ctx: DesignContext) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        dataset: str,
        qref: str,
        tier: str,
        cell_count: int,
        observation_count: int,
        population: str,
        fold_file: str,
        qref_source: str,
    ) -> None:
        lookahead = qref == "early_median5_comparable"
        rows.append(
            {
                "Dataset": dataset,
                "Benchmark_Tier": tier,
                "Qref_Definition": qref,
                "Qref_Source_Rule": qref_source,
                "Included_Cell_Count": cell_count,
                "Physical_Observation_Count": observation_count,
                "Observation_Population_ID": population,
                "Capacity_Field": "Primary_Capacity_Ah (frozen Capacity_Ah column)",
                "SOH_Target_Formula": "SOH_Fraction=Primary_Capacity_Ah/Qref_Ah",
                "Model_Target": "SOH_Fraction",
                "Reporting_Label": "SOH_Percent=100*SOH_Fraction",
                "Clip_Above_One": "NO",
                "Round_Target": "NO",
                "Use_Dataset_Native_SOH": "NO",
                "Qref_As_Input_Feature": "NO",
                "Definition_Specific_EOL_Truncation": "NO",
                "Fold_Assignment_File": fold_file,
                "EARLY_MEDIAN5_BOL_LOOKAHEAD": "YES" if lookahead else "NO",
                "Planned_BOL_Lookahead_Sensitivity": (
                    "REPEAT_METRICS_EXCLUDING_RAW_CYCLE_INDEX_1_TO_5"
                    if lookahead
                    else "NOT_APPLICABLE"
                ),
                "Frozen_Status": "DESIGN_FROZEN_PENDING_HUMAN_APPROVAL",
            }
        )

    x_count = len(ctx.xjtu_common_keys)
    h_count = len(ctx.hust_common_keys)
    source_rules = {
        "nominal": "DATASET_NOMINAL_CAPACITY",
        "first_comparable": "FIRST_FINAL_INCLUDED_PROTOCOL_COMPARABLE_CAPACITY",
        "early_median5_comparable": "MEDIAN_OF_FIRST_FIVE_NUMERICALLY_ORDERED_FINAL_INCLUDED_COMPARABLE_CAPACITIES",
        "initial_test": "VALID_INITIAL_DIAGNOSTIC_CAPACITY; XJTU_ONLY",
    }
    for qref in COMMON_QREFS:
        add(
            "XJTU",
            qref,
            "PRIMARY_CROSS_DATASET",
            55,
            x_count,
            "XJTU_COMMON_PRIMARY_55_CELL",
            "ml_cell_fold_assignments_xjtu.csv",
            source_rules[qref],
        )
        add(
            "HUST",
            qref,
            "PRIMARY_CROSS_DATASET",
            77,
            h_count,
            "HUST_COMMON_PRIMARY_77_CELL",
            "ml_cell_fold_assignments_hust.csv",
            source_rules[qref],
        )
    initial_count = int(
        ctx.xjtu_labels["Qref_Definition"].eq(XJTU_SECONDARY_QREF).sum()
    )
    add(
        "XJTU",
        XJTU_SECONDARY_QREF,
        "SECONDARY_XJTU_ONLY_COMMON_47",
        len(ctx.xjtu_initial_cells),
        initial_count,
        "XJTU_COMMON_47_INITIAL_TEST",
        "ml_xjtu_initial_test_47cell_fold_assignments.csv",
        source_rules[XJTU_SECONDARY_QREF],
    )
    return pd.DataFrame(rows).sort_values(["Dataset", "Benchmark_Tier", "Qref_Definition"])


def build_preprocessing_plan() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        order: int,
        model: str,
        operation: str,
        applied_to: str,
        fit_scope: str,
        parameters: str,
        leakage_control: str,
    ) -> None:
        rows.append(
            {
                "Step_Order": order,
                "Model_Family": model,
                "Applicable_Feature_Sets": "FEATURE_SET_A_PHYSICS_ONLY|FEATURE_SET_B_PHYSICS_PLUS_CYCLE_INDEX",
                "Operation": operation,
                "Applied_To": applied_to,
                "Fit_Scope": fit_scope,
                "Held_Out_Cell_Data_Used_For_Fit": "NO",
                "Parameters": parameters,
                "Leakage_Control": leakage_control,
                "Implementation_State": "PLANNED_NOT_EXECUTED",
            }
        )

    for model in ("Ridge Regression", "Random Forest Regressor", "Gradient Boosting Regressor"):
        add(
            1,
            model,
            "VALIDATE_AND_SELECT_FROZEN_FEATURE_COLUMNS",
            "ALL_INPUT_ROWS",
            "NO_STATISTICAL_FIT; SCHEMA_ONLY",
            "Exact approved feature-name allowlist; Qref_Ah excluded",
            "Selection is identical across Qrefs and independent of labels.",
        )
        add(
            2,
            model,
            "MEDIAN_IMPUTATION_IF_REQUIRED",
            "NUMERIC_FEATURES",
            "CURRENT_OUTER_TRAINING_FOLDS_ONLY",
            "Per-feature median; optional missing indicator fixed from training schema",
            "Imputer statistics are never computed from held-out cells.",
        )
        if model == "Ridge Regression":
            add(
                3,
                model,
                "STANDARDIZE",
                "NUMERIC_FEATURES",
                "CURRENT_OUTER_TRAINING_FOLDS_ONLY",
                "Mean=0 and variance=1 using training-cell statistics",
                "Scaler is refit independently inside every outer fold.",
            )
        else:
            add(
                3,
                model,
                "NO_SCALING",
                "NUMERIC_FEATURES",
                "NOT_APPLICABLE",
                "Tree model receives imputed physical-unit features",
                "No global scaling statistics exist.",
            )
        add(
            4,
            model,
            "FIT_MODEL",
            "SOH_Fraction target",
            "CURRENT_OUTER_TRAINING_FOLDS_ONLY",
            "Fixed hyperparameters from ml_model_hyperparameter_plan.csv",
            "No test-fold tuning and no cross-Qref feature changes.",
        )
        add(
            5,
            model,
            "PREDICT_HELD_OUT_CELLS",
            "CURRENT_OUTER_TEST_FOLD",
            "TRANSFORM_ONLY_WITH_TRAINING_FIT_OBJECTS",
            "One prediction per frozen physical observation",
            "Every cell is predicted only while held out.",
        )
        add(
            6,
            model,
            "BACK_TRANSFORM_TO_CAPACITY_FOR_EVALUATION",
            "PREDICTIONS_ONLY",
            "NO_FIT",
            "Predicted_Capacity_Ah=Predicted_SOH_Fraction*Qref_Ah",
            "Qref is used after prediction for metrics, never as a predictor.",
        )
    return pd.DataFrame(rows).sort_values(["Model_Family", "Step_Order"])


def build_model_hyperparameter_plan() -> pd.DataFrame:
    rows = [
        {
            "Model_Order": 1,
            "Model_Family": "Ridge Regression",
            "Planned_Implementation": "sklearn.linear_model.Ridge",
            "Hyperparameter_Mode": "FIXED",
            "Hyperparameters_JSON": json.dumps(
                {"alpha": 1.0, "fit_intercept": True, "solver": "auto"},
                sort_keys=True,
            ),
            "Required_Preprocessing": "TRAIN_FOLD_MEDIAN_IMPUTER_AND_STANDARD_SCALER",
            "Random_Seed": "NOT_APPLICABLE_DETERMINISTIC_SOLVER",
            "Test_Fold_Tuning": "PROHIBITED",
            "Qref_Specific_Tuning": "NO",
            "Rationale": "Fixed regularized linear baseline isolates label-definition effects.",
            "Training_Status": "NOT_TRAINED_IN_DESIGN_RUN",
        },
        {
            "Model_Order": 2,
            "Model_Family": "Random Forest Regressor",
            "Planned_Implementation": "sklearn.ensemble.RandomForestRegressor",
            "Hyperparameter_Mode": "FIXED",
            "Hyperparameters_JSON": json.dumps(
                {
                    "bootstrap": True,
                    "max_depth": 12,
                    "max_features": "sqrt",
                    "min_samples_leaf": 3,
                    "n_estimators": 300,
                    "n_jobs": 1,
                    "random_state": RANDOM_SEED,
                },
                sort_keys=True,
            ),
            "Required_Preprocessing": "TRAIN_FOLD_MEDIAN_IMPUTER; NO_SCALER",
            "Random_Seed": RANDOM_SEED,
            "Test_Fold_Tuning": "PROHIBITED",
            "Qref_Specific_Tuning": "NO",
            "Rationale": "Conservative fixed ensemble; single-thread setting maximizes reproducibility.",
            "Training_Status": "NOT_TRAINED_IN_DESIGN_RUN",
        },
        {
            "Model_Order": 3,
            "Model_Family": "Gradient Boosting Regressor",
            "Planned_Implementation": "sklearn.ensemble.GradientBoostingRegressor",
            "Hyperparameter_Mode": "FIXED",
            "Hyperparameters_JSON": json.dumps(
                {
                    "learning_rate": 0.05,
                    "loss": "squared_error",
                    "max_depth": 3,
                    "min_samples_leaf": 3,
                    "n_estimators": 200,
                    "random_state": RANDOM_SEED,
                    "subsample": 1.0,
                },
                sort_keys=True,
            ),
            "Required_Preprocessing": "TRAIN_FOLD_MEDIAN_IMPUTER; NO_SCALER",
            "Random_Seed": RANDOM_SEED,
            "Test_Fold_Tuning": "PROHIBITED",
            "Qref_Specific_Tuning": "NO",
            "Rationale": "Small fixed boosting model avoids broad hyperparameter search.",
            "Training_Status": "NOT_TRAINED_IN_DESIGN_RUN",
        },
    ]
    return pd.DataFrame(rows)


def build_metric_plan() -> pd.DataFrame:
    rows = [
        (
            "MAE_SOH",
            "SOH_LABEL_SPACE",
            "mean(abs(y_soh-pred_soh)) within each held-out cell",
            "NO",
        ),
        (
            "RMSE_SOH",
            "SOH_LABEL_SPACE",
            "sqrt(mean((y_soh-pred_soh)^2)) within each held-out cell",
            "NO",
        ),
        (
            "R2_SOH",
            "SOH_LABEL_SPACE",
            "1-SSE/SST within each held-out cell; NA when fewer than 2 observations or zero variance",
            "NO",
        ),
        (
            "MAE_SOH_percentage_points",
            "SOH_LABEL_SPACE",
            "100*MAE_SOH within each held-out cell",
            "NO",
        ),
        (
            "MAE_Q_Ah",
            "PHYSICAL_CAPACITY_SPACE",
            "mean(abs(Capacity_Ah-Predicted_SOH_Fraction*Qref_Ah)) within each held-out cell",
            "YES",
        ),
        (
            "RMSE_Q_Ah",
            "PHYSICAL_CAPACITY_SPACE",
            "sqrt(mean((Capacity_Ah-Predicted_SOH_Fraction*Qref_Ah)^2)) within each held-out cell",
            "NO",
        ),
        (
            "R2_Q",
            "PHYSICAL_CAPACITY_SPACE",
            "1-SSE/SST in Ah within each held-out cell; NA when fewer than 2 observations or zero variance",
            "NO",
        ),
    ]
    records: list[dict[str, Any]] = []
    for order, (name, space, formula, primary) in enumerate(rows, start=1):
        records.append(
            {
                "Metric_Order": order,
                "Metric_Name": name,
                "Metric_Space": space,
                "Formula": formula,
                "Primary_Model_Ranking_Metric": primary,
                "First_Level_Aggregation_Unit": "HELD_OUT_Cell_ID",
                "Cross_Cell_Summary": "MEDIAN|IQR_Q1_Q3|MEAN|CELL_COUNT",
                "Cycle_Weighting_Across_Cells": "NONE; COMPUTE_EACH_CELL_FIRST",
                "EOL_Truncation": "NO",
                "Censored_EOL_Required": "NO; SOH benchmark retains all frozen observations",
                "Future_Confidence_Interval_Unit": "RESAMPLE_CELLS_NOT_CYCLES",
                "Ranking_Direction": "LOWER_IS_BETTER" if not name.startswith("R2") else "HIGHER_IS_BETTER",
                "Ranking_Change_Definition": (
                    "YES only when model ordering by median cell-level MAE_Q_Ah differs across Qrefs"
                    if name == "MAE_Q_Ah"
                    else "NOT_PRIMARY_RANKING_RULE"
                ),
                "Frozen_Status": "PLANNED_NOT_COMPUTED",
            }
        )
    return pd.DataFrame(records)


def build_fairness_matrix() -> pd.DataFrame:
    comparisons = [
        ("XJTU", "nominal", "first_comparable", "XJTU_COMMON_PRIMARY_55_CELL"),
        ("XJTU", "nominal", "early_median5_comparable", "XJTU_COMMON_PRIMARY_55_CELL"),
        ("XJTU", "first_comparable", "early_median5_comparable", "XJTU_COMMON_PRIMARY_55_CELL"),
        ("HUST", "nominal", "first_comparable", "HUST_COMMON_PRIMARY_77_CELL"),
        ("HUST", "nominal", "early_median5_comparable", "HUST_COMMON_PRIMARY_77_CELL"),
        ("HUST", "first_comparable", "early_median5_comparable", "HUST_COMMON_PRIMARY_77_CELL"),
        ("XJTU", "nominal", "initial_test", "XJTU_COMMON_47_INITIAL_TEST"),
    ]
    rows: list[dict[str, Any]] = []
    for dataset, a, b, population in comparisons:
        for feature_set in (
            "FEATURE_SET_A_PHYSICS_ONLY",
            "FEATURE_SET_B_PHYSICS_PLUS_CYCLE_INDEX",
        ):
            rows.append(
                {
                    "Dataset": dataset,
                    "Comparison_ID": f"{dataset}_{a}_VS_{b}_{feature_set}",
                    "Qref_Definition_A": a,
                    "Qref_Definition_B": b,
                    "Observation_Population_ID": population,
                    "Feature_Set_ID": feature_set,
                    "Same_Cell_ID_Set": "YES",
                    "Same_Physical_Observation_Keys": "YES",
                    "Same_Fold_Assignment": "YES",
                    "Same_Features": "YES",
                    "Same_Preprocessing_Rules": "YES",
                    "Same_Model_Parameters": "YES",
                    "Same_Metric_Definitions": "YES",
                    "Definition_Specific_EOL_Truncation": "NO",
                    "Only_Allowed_Difference": "Qref-derived SOH target label",
                    "Fair_Comparison": "YES",
                    "Notes": (
                        "XJTU initial_test comparison is restricted to the common 47-cell subset."
                        if b == "initial_test"
                        else "Primary common-Qref comparison."
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["Dataset", "Comparison_ID"])


def build_regression_tests(
    ctx: DesignContext,
    inventory: pd.DataFrame,
    leakage: pd.DataFrame,
    harmonized: pd.DataFrame,
    intersections: pd.DataFrame,
    xjtu_folds: pd.DataFrame,
    hust_folds: pd.DataFrame,
    initial_folds: pd.DataFrame,
    labels: pd.DataFrame,
    preprocessing: pd.DataFrame,
    models: pd.DataFrame,
    metrics: pd.DataFrame,
    fairness: pd.DataFrame,
    frozen_inputs_unchanged: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        test_id: str,
        category: str,
        passed: bool,
        expected: str,
        observed: str,
        evidence: str,
        hard_stop: str = "YES",
    ) -> None:
        rows.append(
            {
                "Test_ID": test_id,
                "Category": category,
                "Expected": expected,
                "Observed": observed,
                "Status": "PASS" if passed else "FAIL",
                "Hard_Stop_On_Failure": hard_stop,
                "Evidence": evidence,
            }
        )

    core = inventory.loc[inventory["Decision"].eq("SAFE_CORE")].copy()
    core_names = sorted(core["Feature_Name"].unique())
    core_upper = "|".join(core_names).upper()
    add(
        "NO_TARGET_CAPACITY_AS_FEATURE",
        "LEAKAGE",
        not core["Uses_Target_Capacity"].eq("YES").any()
        and "CAPACITY_AH" not in core_upper,
        "No target capacity or transformed capacity in SAFE_CORE",
        "SAFE_CORE=" + "|".join(core_names),
        "ml_feature_inventory.csv and ml_feature_leakage_audit.csv",
    )
    add(
        "NO_SOH_AS_FEATURE",
        "LEAKAGE",
        "SOH" not in core_upper,
        "No SOH label or SOH transform in SAFE_CORE",
        f"SOH name matches={sum('SOH' in name.upper() for name in core_names)}",
        "Harmonized feature allowlist",
    )
    add(
        "NO_RUL_OR_EOL_AS_FEATURE",
        "LEAKAGE",
        not any(token in core_upper for token in ("RUL", "EOL")),
        "No RUL/EOL/future-life feature in SAFE_CORE",
        f"RUL/EOL matches={sum(any(token in name.upper() for token in ('RUL','EOL')) for name in core_names)}",
        "Harmonized feature allowlist",
    )
    add(
        "NO_NATIVE_LABEL_AS_FEATURE",
        "LEAKAGE",
        "NATIVE" not in core_upper,
        "No dataset-native label in SAFE_CORE",
        f"Native matches={sum('NATIVE' in name.upper() for name in core_names)}",
        "Harmonized feature allowlist",
    )
    add(
        "NO_FULL_DISCHARGE_DURATION_IN_SAFE_CORE",
        "CAPACITY_PROXY",
        "Full_Discharge_Duration_s" not in core_names,
        "Full-discharge duration excluded from SAFE_CORE",
        "Excluded; classified HIGH_LEAKAGE_RISK_CAPACITY_PROXY",
        "ml_feature_leakage_audit.csv",
    )
    add(
        "NO_QREF_AS_PRIMARY_INPUT_FEATURE",
        "LEAKAGE",
        not any("QREF" in name.upper() for name in core_names)
        and harmonized["Qref_Independent_Input"].eq("YES").all(),
        "Qref determines labels only and is absent from predictor sets",
        f"Qref-named SAFE_CORE features={sum('QREF' in name.upper() for name in core_names)}",
        "ml_harmonized_feature_plan.csv",
    )

    fold_unique = (
        not xjtu_folds["Cell_ID"].duplicated().any()
        and not hust_folds["Cell_ID"].duplicated().any()
        and xjtu_folds["No_Cell_Leakage_Verified"].eq("YES").all()
        and hust_folds["No_Cell_Leakage_Verified"].eq("YES").all()
    )
    add(
        "NO_SAME_CELL_ACROSS_TRAIN_TEST",
        "SPLIT",
        fold_unique,
        "Each cell assigned to exactly one held-out fold",
        f"XJTU unique={xjtu_folds.Cell_ID.nunique()}/{len(xjtu_folds)}; HUST unique={hust_folds.Cell_ID.nunique()}/{len(hust_folds)}",
        "Cell-level assignment files",
    )
    primary_intersections = intersections.loc[
        intersections["Definition_A"].isin(COMMON_QREFS)
        & intersections["Definition_B"].isin(COMMON_QREFS)
    ]
    add(
        "PHYSICAL_OBSERVATION_KEYS_IDENTICAL_ACROSS_COMMON_QREFS",
        "OBSERVATION_POPULATION",
        len(primary_intersections) == 6
        and primary_intersections["Physical_Observation_Keys_Identical"].eq("YES").all(),
        "All three common Qrefs have identical physical keys within each dataset",
        f"Passing pairwise audits={int(primary_intersections.Audit_Status.eq('PASS').sum())}/6",
        "ml_observation_intersection_audit.csv",
    )
    expected_qrefs = "|".join(COMMON_QREFS)
    add(
        "FOLD_ASSIGNMENTS_IDENTICAL_ACROSS_QREFS",
        "SPLIT",
        xjtu_folds["Applicable_Qref_Definitions"].eq(expected_qrefs).all()
        and hust_folds["Applicable_Qref_Definitions"].eq(expected_qrefs).all(),
        "One fixed assignment per dataset applies to all common Qrefs",
        f"Applicable common Qrefs={expected_qrefs}",
        "XJTU and HUST fold assignment files",
    )
    add(
        "FEATURES_IDENTICAL_ACROSS_QREFS",
        "FAIRNESS",
        fairness["Same_Features"].eq("YES").all()
        and harmonized["Qref_Independent_Input"].eq("YES").all(),
        "Feature names and extraction definitions fixed across Qrefs",
        f"Fair comparisons with same features={int(fairness.Same_Features.eq('YES').sum())}/{len(fairness)}",
        "ml_fairness_matrix.csv and ml_harmonized_feature_plan.csv",
    )
    initial_audit = intersections.loc[
        intersections["Comparison_ID"].eq("XJTU_nominal_VS_initial_test")
    ]
    add(
        "XJTU_INITIAL_TEST_USES_COMMON_47_CELL_SUBSET",
        "SECONDARY_BENCHMARK",
        len(initial_folds) == 47
        and initial_folds["Cell_ID"].nunique() == 47
        and len(initial_audit) == 1
        and initial_audit.iloc[0]["Physical_Observation_Keys_Identical"] == "YES",
        "Nominal versus initial_test uses the same observations and inherited folds for 47 cells",
        f"Cells={initial_folds.Cell_ID.nunique()}; key audit={initial_audit.iloc[0]['Audit_Status'] if len(initial_audit) else 'MISSING'}",
        "Initial-test assignment and intersection audit",
    )
    statistical_ops = preprocessing.loc[
        preprocessing["Operation"].isin(["MEDIAN_IMPUTATION_IF_REQUIRED", "STANDARDIZE"])
    ]
    add(
        "NO_GLOBAL_PREPROCESSING_BEFORE_CV",
        "PREPROCESSING",
        statistical_ops["Fit_Scope"].eq("CURRENT_OUTER_TRAINING_FOLDS_ONLY").all()
        and preprocessing["Held_Out_Cell_Data_Used_For_Fit"].eq("NO").all(),
        "All statistical preprocessing fitted on current training cells only",
        f"Train-only statistical operations={int(statistical_ops.Fit_Scope.eq('CURRENT_OUTER_TRAINING_FOLDS_ONLY').sum())}/{len(statistical_ops)}",
        "ml_preprocessing_plan.csv",
    )
    primary_metric = metrics.loc[metrics["Primary_Model_Ranking_Metric"].eq("YES")]
    add(
        "PRIMARY_METRIC_INCLUDES_PHYSICAL_CAPACITY_SPACE",
        "METRICS",
        len(primary_metric) == 1
        and primary_metric.iloc[0]["Metric_Name"] == "MAE_Q_Ah"
        and primary_metric.iloc[0]["Metric_Space"] == "PHYSICAL_CAPACITY_SPACE",
        "Primary ranking metric is physical-capacity MAE",
        "|".join(primary_metric["Metric_Name"].astype(str)),
        "ml_metric_plan.csv",
    )
    add(
        "CELL_LEVEL_METRIC_AGGREGATION_REQUIRED",
        "METRICS",
        metrics["First_Level_Aggregation_Unit"].eq("HELD_OUT_Cell_ID").all()
        and metrics["Cross_Cell_Summary"].str.contains("MEDIAN").all(),
        "Metrics computed per held-out cell before cross-cell summaries",
        f"Cell-first metrics={int(metrics.First_Level_Aggregation_Unit.eq('HELD_OUT_Cell_ID').sum())}/{len(metrics)}",
        "ml_metric_plan.csv",
    )
    hust_qrefs = set(ctx.hust_labels["Qref_Definition"].unique())
    add(
        "HUST_INITIAL_TEST_NOT_INVENTED",
        "LABEL_DEFINITION",
        XJTU_SECONDARY_QREF not in hust_qrefs
        and not (
            (labels["Dataset"] == "HUST")
            & (labels["Qref_Definition"] == XJTU_SECONDARY_QREF)
        ).any(),
        "No HUST initial_test label or plan row",
        "HUST Qrefs=" + "|".join(sorted(hust_qrefs)),
        "Frozen HUST labels and ml_label_definition_plan.csv",
    )
    early_rows = labels.loc[labels["Qref_Definition"].eq("early_median5_comparable")]
    add(
        "EARLY_MEDIAN5_LOOKAHEAD_DOCUMENTED",
        "LABEL_DEFINITION",
        len(early_rows) == 2
        and early_rows["EARLY_MEDIAN5_BOL_LOOKAHEAD"].eq("YES").all()
        and early_rows["Planned_BOL_Lookahead_Sensitivity"].str.contains("1_TO_5").all(),
        "BOL look-ahead flagged with planned exclusion-of-cycles-1-to-5 sensitivity",
        f"Documented rows={int(early_rows.EARLY_MEDIAN5_BOL_LOOKAHEAD.eq('YES').sum())}/{len(early_rows)}",
        "ml_label_definition_plan.csv",
    )

    # Additional hard-stop and reproducibility assertions.
    both_core = set(
        inventory.loc[
            inventory["Decision"].eq("SAFE_CORE")
            & inventory["Available_Observation_Count"].gt(0),
            "Feature_Name",
        ].unique()
    )
    add(
        "HARMONIZED_SAFE_CORE_NONEMPTY",
        "FEATURE_AVAILABILITY",
        len(both_core) > 0 and len(harmonized) > 0,
        "At least one leakage-safe harmonized feature",
        f"Harmonized SAFE_CORE feature count={len(core_names)}",
        "ml_feature_inventory.csv",
    )
    add(
        "ALL_FEATURE_LEAKAGE_AUDITS_PASS",
        "LEAKAGE",
        leakage["Audit_Result"].eq("PASS").all(),
        "No unsafe feature classified as SAFE_CORE",
        f"Pass={int(leakage.Audit_Result.eq('PASS').sum())}/{len(leakage)}",
        "ml_feature_leakage_audit.csv",
    )
    add(
        "COMMON_QREF_DEFINITIONS_EXACT",
        "LABEL_DEFINITION",
        set(ctx.xjtu_labels["Qref_Definition"].unique())
        == set(COMMON_QREFS) | {XJTU_SECONDARY_QREF}
        and hust_qrefs == set(COMMON_QREFS),
        "Primary common Qrefs exactly nominal, first_comparable, early_median5_comparable",
        f"XJTU={sorted(ctx.xjtu_labels.Qref_Definition.unique())}; HUST={sorted(hust_qrefs)}",
        "Frozen SOH label tables",
    )

    formula_errors = []
    for frame in (ctx.xjtu_labels, ctx.hust_labels):
        expected_soh = frame["Capacity_Ah"] / frame["Qref_Ah"]
        formula_errors.append(float(np.nanmax(np.abs(frame["SOH_Fraction"] - expected_soh))))
    add(
        "SOH_FORMULA_MATCHES_FROZEN_LABELS",
        "LABEL_DEFINITION",
        max(formula_errors) <= 1e-9,
        "SOH_Fraction equals Capacity_Ah/Qref_Ah within 1e-9",
        f"Max absolute error={max(formula_errors):.12g}",
        "Frozen SOH label tables",
    )
    add(
        "NO_DEFINITION_SPECIFIC_EOL_TRUNCATION",
        "OBSERVATION_POPULATION",
        labels["Definition_Specific_EOL_Truncation"].eq("NO").all()
        and intersections["Definition_Specific_EOL_Truncation"].eq("NO").all(),
        "No Qref-specific EOL truncation",
        "All label-plan and intersection rows state NO",
        "Label and observation plans",
    )
    x_fold_counts = xjtu_folds.groupby("Fold_ID")["Cell_ID"].nunique()
    h_fold_counts = hust_folds.groupby("Fold_ID")["Cell_ID"].nunique()
    x_batch_balance = xjtu_folds.groupby(["Batch_ID", "Fold_ID"])["Cell_ID"].nunique().unstack(fill_value=0)
    add(
        "DETERMINISTIC_FIVE_FOLD_COUNTS_VALID",
        "SPLIT",
        set(x_fold_counts.index) == set(range(1, 6))
        and set(h_fold_counts.index) == set(range(1, 6))
        and x_fold_counts.max() - x_fold_counts.min() <= 1
        and h_fold_counts.max() - h_fold_counts.min() <= 1
        and (x_batch_balance.max(axis=1) - x_batch_balance.min(axis=1)).max() <= 1,
        "Five folds, total imbalance <=1, and XJTU within-batch imbalance <=1",
        f"XJTU={x_fold_counts.to_dict()}; HUST={h_fold_counts.to_dict()}",
        "Fold assignment files",
    )
    inherited = initial_folds[["Cell_ID", "Fold_ID"]].merge(
        xjtu_folds[["Cell_ID", "Fold_ID"]],
        on="Cell_ID",
        suffixes=("_initial", "_primary"),
        validate="one_to_one",
    )
    add(
        "XJTU_INITIAL_FOLDS_INHERIT_PRIMARY_ASSIGNMENTS",
        "SPLIT",
        inherited["Fold_ID_initial"].eq(inherited["Fold_ID_primary"]).all(),
        "The common 47 cells retain their primary XJTU Fold_ID",
        f"Matching assignments={int(inherited.Fold_ID_initial.eq(inherited.Fold_ID_primary).sum())}/{len(inherited)}",
        "XJTU primary and 47-cell assignment files",
    )
    add(
        "XJTU_AND_HUST_TRAINED_SEPARATELY",
        "PROTOCOL_CONFOUNDING",
        harmonized["Datasets_Trained_Separately"].eq("YES").all(),
        "No pooled raw-cycle training table",
        "All harmonized plan rows say YES",
        "ml_harmonized_feature_plan.csv",
    )
    add(
        "NO_FINAL_MODEL_TRAINING_EXECUTED",
        "RUN_SCOPE",
        models["Training_Status"].eq("NOT_TRAINED_IN_DESIGN_RUN").all(),
        "Design only; zero model fits",
        f"Planned model families={len(models)}; fitted models=0",
        "ml_model_hyperparameter_plan.csv and source-code scope",
    )
    add(
        "FROZEN_INPUTS_UNMODIFIED",
        "INPUT_INTEGRITY",
        frozen_inputs_unchanged,
        "SHA-256/size snapshots identical before and after design generation",
        "UNCHANGED" if frozen_inputs_unchanged else "CHANGED",
        "Pre/post recursive frozen-directory snapshot",
    )
    tests = pd.DataFrame(rows)
    return tests.sort_values(["Category", "Test_ID"]).reset_index(drop=True)


def _snapshot_digest(snapshot: dict[str, tuple[int, str]]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.columns.duplicated().any():
        raise DesignError(f"Duplicate columns in planned output {path.name}")
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def _build_processing_log(
    ctx: DesignContext,
    tables: dict[str, pd.DataFrame],
    snapshots_before: dict[str, dict[str, tuple[int, str]]],
    snapshots_after: dict[str, dict[str, tuple[int, str]]],
) -> str:
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    regression = tables["ml_design_regression_tests.csv"]
    xfold = tables["ml_cell_fold_assignments_xjtu.csv"]
    hfold = tables["ml_cell_fold_assignments_hust.csv"]
    inventory = tables["ml_feature_inventory.csv"]
    harmonized = tables["ml_harmonized_feature_plan.csv"]
    lines = [
        f"Design_Version: {DESIGN_VERSION}",
        f"Processing_Timestamp_Asia_Shanghai: {timestamp}",
        f"Python_Version: {platform.python_version()}",
        f"Python_Executable: {sys.executable}",
        f"Platform: {platform.platform()}",
        f"pandas_Version: {pd.__version__}",
        f"numpy_Version: {np.__version__}",
        f"Random_Seed: {RANDOM_SEED}",
        f"Grouped_CV_Folds: {N_FOLDS}",
        "Run_Scope: DESIGN_ONLY; NO FINAL MODELS TRAINED",
        "Raw_Data_Action: NOT_REPROCESSED; candidate availability inferred from frozen labels and validated audit tables",
        "Frozen_Inputs_Write_Action: NONE",
        "Datasets_Training_Policy: XJTU_AND_HUST_SEPARATE",
        "",
        "[FROZEN INPUT SNAPSHOTS]",
    ]
    for name in ("XJTU_FINAL_V2", "HUST_FINAL_V1", "CROSS_DATASET_SUMMARY_V1"):
        before = snapshots_before[name]
        after = snapshots_after[name]
        lines.extend(
            [
                f"{name}_File_Count: {len(before)}",
                f"{name}_Before_Digest_SHA256: {_snapshot_digest(before)}",
                f"{name}_After_Digest_SHA256: {_snapshot_digest(after)}",
                f"{name}_Unchanged: {'YES' if before == after else 'NO'}",
            ]
        )
    lines.extend(
        [
            "",
            "[FROZEN OBSERVATION POPULATIONS]",
            f"XJTU_Common_Qref_Cells: {ctx.xjtu_common_keys.Cell_ID.nunique()}",
            f"XJTU_Common_Qref_Observations: {len(ctx.xjtu_common_keys)}",
            f"XJTU_Initial_Test_Common_Cells: {len(ctx.xjtu_initial_cells)}",
            f"XJTU_Initial_Test_Observations: {int(ctx.xjtu_labels.Qref_Definition.eq(XJTU_SECONDARY_QREF).sum())}",
            f"HUST_Common_Qref_Cells: {ctx.hust_common_keys.Cell_ID.nunique()}",
            f"HUST_Common_Qref_Observations: {len(ctx.hust_common_keys)}",
            "Common_Primary_Qrefs: " + "|".join(COMMON_QREFS),
            "HUST_Initial_Test: NOT_AVAILABLE_NOT_INVENTED",
            "Definition_Specific_EOL_Truncation: NO",
            "",
            "[FEATURE DESIGN]",
            f"Feature_Inventory_Rows: {len(inventory)}",
            f"Harmonized_SAFE_CORE_Feature_Count: {harmonized.loc[harmonized.Feature_Set_ID.eq('FEATURE_SET_A_PHYSICS_ONLY'),'Feature_Name'].nunique()}",
            "Primary_Feature_Set: FEATURE_SET_A_PHYSICS_ONLY",
            "Sensitivity_Feature_Set: FEATURE_SET_B_PHYSICS_PLUS_CYCLE_INDEX",
            "Full_Discharge_Duration: HIGH_LEAKAGE_RISK_CAPACITY_PROXY; PROHIBITED_FROM_SAFE_CORE",
            "Qref_As_Input: NO",
            "",
            "[CELL-LEVEL FOLDS]",
            f"XJTU_Fold_Cell_Counts: {xfold.groupby('Fold_ID').Cell_ID.nunique().to_dict()}",
            f"HUST_Fold_Cell_Counts: {hfold.groupby('Fold_ID').Cell_ID.nunique().to_dict()}",
            "Split_Unit: Cell_ID",
            "Cycle_Level_Random_Split: PROHIBITED",
            "",
            "[OUTPUT TABLES]",
        ]
    )
    for name in OUTPUT_NAMES:
        lines.append(f"{name}: {len(tables[name])} rows")
    lines.extend(
        [
            "",
            "[REGRESSION TESTS]",
            f"PASS: {int(regression.Status.eq('PASS').sum())}",
            f"FAIL: {int(regression.Status.eq('FAIL').sum())}",
        ]
    )
    for row in regression.itertuples(index=False):
        lines.append(f"{row.Test_ID}: {row.Status} | {row.Observed}")
    lines.extend(
        [
            "",
            "ALL ML DESIGN REGRESSION TESTS PASSED"
            if regression["Status"].eq("PASS").all()
            else "ML DESIGN REGRESSION TESTS FAILED",
            "SOH ML DESIGN V1 COMPLETE —",
            "NO FINAL MODELS TRAINED —",
            "WAITING FOR HUMAN APPROVAL",
        ]
    )
    return "\n".join(lines) + "\n"


def run_design(workspace: Path, output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Generate all design artifacts atomically after hard-stop checks pass."""
    workspace = workspace.resolve()
    paths = InputPaths.from_workspace(workspace)
    for frozen_dir in (paths.xjtu_frozen, paths.hust_frozen, paths.cross_frozen):
        if not frozen_dir.is_dir():
            raise DesignError(f"Missing frozen directory: {frozen_dir}")
    final_dir = (
        output_dir.resolve()
        if output_dir is not None
        else (workspace / "outputs" / "soh_ml_design_v1").resolve()
    )
    if final_dir.exists():
        raise DesignError(f"Refusing to overwrite existing output directory: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)

    snapshots_before = {
        "XJTU_FINAL_V2": _directory_snapshot(paths.xjtu_frozen),
        "HUST_FINAL_V1": _directory_snapshot(paths.hust_frozen),
        "CROSS_DATASET_SUMMARY_V1": _directory_snapshot(paths.cross_frozen),
    }
    ctx = _load_context(paths)
    inventory = build_feature_inventory(ctx)
    leakage = build_leakage_audit(inventory)
    harmonized = build_harmonized_feature_plan(inventory)
    dataset_specific = build_dataset_specific_plan()
    intersections = build_observation_intersection_audit(ctx)
    xjtu_folds = _format_xjtu_fold_assignments(ctx)
    hust_folds = _format_hust_fold_assignments(ctx)
    initial_folds = _format_xjtu_initial_folds(ctx, xjtu_folds)
    labels = build_label_definition_plan(ctx)
    preprocessing = build_preprocessing_plan()
    models = build_model_hyperparameter_plan()
    metrics = build_metric_plan()
    fairness = build_fairness_matrix()

    tables: dict[str, pd.DataFrame] = {
        "ml_feature_inventory.csv": inventory,
        "ml_feature_leakage_audit.csv": leakage,
        "ml_harmonized_feature_plan.csv": harmonized,
        "ml_dataset_specific_feature_plan.csv": dataset_specific,
        "ml_observation_intersection_audit.csv": intersections,
        "ml_cell_fold_assignments_xjtu.csv": xjtu_folds,
        "ml_cell_fold_assignments_hust.csv": hust_folds,
        "ml_xjtu_initial_test_47cell_fold_assignments.csv": initial_folds,
        "ml_label_definition_plan.csv": labels,
        "ml_preprocessing_plan.csv": preprocessing,
        "ml_model_hyperparameter_plan.csv": models,
        "ml_metric_plan.csv": metrics,
        "ml_fairness_matrix.csv": fairness,
    }
    stage = Path(
        tempfile.mkdtemp(prefix=".soh_ml_design_v1_staging_", dir=final_dir.parent)
    ).resolve()
    try:
        for name, frame in tables.items():
            _write_csv(frame, stage / name)

        snapshots_after = {
            "XJTU_FINAL_V2": _directory_snapshot(paths.xjtu_frozen),
            "HUST_FINAL_V1": _directory_snapshot(paths.hust_frozen),
            "CROSS_DATASET_SUMMARY_V1": _directory_snapshot(paths.cross_frozen),
        }
        frozen_unchanged = snapshots_before == snapshots_after
        regression = build_regression_tests(
            ctx,
            inventory,
            leakage,
            harmonized,
            intersections,
            xjtu_folds,
            hust_folds,
            initial_folds,
            labels,
            preprocessing,
            models,
            metrics,
            fairness,
            frozen_unchanged,
        )
        tables["ml_design_regression_tests.csv"] = regression
        _write_csv(regression, stage / "ml_design_regression_tests.csv")
        if not regression["Status"].eq("PASS").all():
            failed = regression.loc[regression["Status"].eq("FAIL"), "Test_ID"].tolist()
            raise DesignError("ML DESIGN NEEDS FEATURE REVIEW: " + "|".join(failed))

        log_text = _build_processing_log(
            ctx, tables, snapshots_before, snapshots_after
        )
        (stage / "processing_log.txt").write_text(log_text, encoding="utf-8")
        produced = sorted(path.name for path in stage.iterdir() if path.is_file())
        expected = sorted([*OUTPUT_NAMES, "processing_log.txt"])
        if produced != expected:
            raise DesignError(f"Output inventory mismatch: produced={produced}; expected={expected}")
        os.replace(stage, final_dir)
    except Exception:
        if stage.exists() and stage.parent == final_dir.parent and stage.name.startswith(
            ".soh_ml_design_v1_staging_"
        ):
            shutil.rmtree(stage)
        raise
    return tables
