from __future__ import annotations

import gc
import hashlib
import json
import platform
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import scipy

from . import PIPELINE_VERSION
from .config_templates import EXPECTED_CANONICAL_COUNTS, all_batch_configs
from .inventory import InventoryResult, scan_inventory
from .matio import MatFileScan, scan_mat_schema
from .protocols import description_catalog, protocol_rows_frame, scan_protocol_cycles


SCAN_OUTPUT_FILES = (
    "xjtu_file_manifest.csv",
    "xjtu_schema_inventory.csv",
    "xjtu_description_catalog.csv",
    "xjtu_protocol_inventory.csv",
    "xjtu_primary_capacity_candidate_inventory.csv",
    "xjtu_protocol_anomalies.csv",
    "xjtu_duplicate_report.csv",
    "batch2_config_draft.json",
    "batch3_config_draft.json",
    "batch4_config_draft.json",
    "batch5_config_draft.json",
    "batch6_config_draft.json",
    "processing_log_scan.txt",
    "representative_validation_plan.csv",
)


SCHEMA_COLUMNS = [
    "Batch_ID",
    "Cell_ID",
    "Source_File",
    "File_Size_Bytes",
    "Top_Level_Variables",
    "Data_Cycle_Count",
    "Summary_Cycle_Life",
    "Data_Fields",
    "Summary_Fields",
    "Description_Field",
    "Voltage_Field",
    "Current_Field",
    "Capacity_Field",
    "Time_Field",
    "Time_Unit",
    "Summary_Discharge_Capacity_Field",
    "Schema_Variant_Signature",
    "Schema_Variant_ID",
    "Schema_Status",
    "Schema_Issues",
    "Processing_Status",
]


ANOMALY_COLUMNS = [
    "Anomaly_Scope",
    "Batch_ID",
    "Cell_ID",
    "Source_File",
    "Raw_Cycle_Index",
    "Schema_Variant_ID",
    "Protocol_Class",
    "Severity",
    "Anomaly_Codes",
    "Details",
    "Requires_Human_Review",
]


REPRESENTATIVE_COLUMNS = [
    "Batch_ID",
    "Cell_ID",
    "Source_File",
    "Representative_Selection_Reason",
    "Validation_Sample_Type",
    "Raw_Cycle_Index",
    "Protocol_Class",
    "Primary_Capacity_Eligible",
    "Expected_Protocol_Match",
    "Current_Issues",
    "Human_Review_Decision",
    "Human_Review_Notes",
]


@dataclass(frozen=True)
class ScanResult:
    output_dir: Path
    scan_status: str
    canonical_counts: dict[int, int]
    duplicate_count: int
    schema_variant_counts: dict[str, int]
    cycle_counts: dict[int, int]
    protocol_counts: dict[str, int]
    primary_candidate_counts: dict[int, int]
    unknown_protocol_count: int
    anomaly_count: int
    representatives: dict[int, str]
    review_reasons: tuple[str, ...]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def _write_json(data: dict[str, Any], path: Path) -> None:
    encoded = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")


def _known_schema_signature() -> str:
    return json.dumps(
        {
            "capacity": "capacity_Ah",
            "current": "current_A",
            "description": "description",
            "relative_time": "relative_time_min",
            "summary_cycle_life": "cycle_life",
            "summary_discharge_capacity": "discharge_capacity_Ah",
            "voltage": "voltage_V",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _schema_variant_map(schema_rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    signatures = sorted(
        {
            str(row["Schema_Variant_Signature"])
            for row in schema_rows
            if row.get("Schema_Variant_Signature")
        }
    )
    known = _known_schema_signature()
    result: dict[str, str] = {}
    next_number = 1
    if known in signatures:
        result[known] = "XJTU_SCHEMA_V1"
        next_number = 2
    for signature in signatures:
        if signature not in result:
            result[signature] = f"XJTU_SCHEMA_V{next_number}"
            next_number += 1
    return result


def _frozen_batch1_schema_rows(frozen_dir: Path) -> list[dict[str, Any]]:
    manifest_path = frozen_dir.parent / "xjtu_batch1_pilot_v1" / "xjtu_file_manifest.csv"
    if not manifest_path.is_file():
        return []
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for item in manifest.loc[manifest["File_Role"].eq("CANONICAL")].itertuples(index=False):
        rows.append(
            {
                "Batch_ID": 1,
                "Cell_ID": str(item.Cell_ID),
                "Source_File": str(item.Source_File),
                "File_Size_Bytes": int(item.File_Size_Bytes),
                "Top_Level_Variables": "data|summary",
                "Data_Cycle_Count": int(item.Data_Cycle_Count),
                "Summary_Cycle_Life": int(item.Summary_Cycle_Life),
                "Data_Fields": (
                    "capacity_Ah|current_A|description|power_Wh|relative_time_min|"
                    "system_time|temperature_C|voltage_V"
                ),
                "Summary_Fields": (
                    "charge_capacity_Ah|charge_mean_voltage|charge_median_voltage|"
                    "charge_power_Wh|cycle_life|description|discharge_capacity_Ah|"
                    "discharge_mean_voltage|discharge_median_voltage|discharge_power_Wh"
                ),
                "Description_Field": "description",
                "Voltage_Field": "voltage_V",
                "Current_Field": "current_A",
                "Capacity_Field": "capacity_Ah",
                "Time_Field": "relative_time_min",
                "Time_Unit": "MINUTES",
                "Summary_Discharge_Capacity_Field": "discharge_capacity_Ah",
                "Schema_Variant_Signature": _known_schema_signature(),
                "Schema_Variant_ID": "XJTU_SCHEMA_V1",
                "Schema_Status": "FROZEN_REFERENCE",
                "Schema_Issues": "",
                "Processing_Status": "NOT_REPROCESSED_FROZEN_REFERENCE",
            }
        )
    return rows


def _canonical_counts(inventory: InventoryResult) -> dict[int, int]:
    result: dict[int, int] = {}
    for batch_id in range(1, 7):
        rows = inventory.manifest.loc[
            (inventory.manifest["Batch_ID"].eq(batch_id))
            & (inventory.manifest["File_Role"].eq("CANONICAL"))
            & (inventory.manifest["Included_In_Analysis"].eq("YES"))
        ]
        result[batch_id] = int(len(rows))
    return result


def _inventory_anomalies(inventory: InventoryResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for issue in inventory.issues:
        batch_id: int | pd._libs.missing.NAType = pd.NA
        for candidate in range(1, 7):
            if f"Batch-{candidate}" in issue:
                batch_id = candidate
                break
        rows.append(
            {
                "Anomaly_Scope": "INVENTORY",
                "Batch_ID": batch_id,
                "Cell_ID": "",
                "Source_File": "",
                "Raw_Cycle_Index": pd.NA,
                "Schema_Variant_ID": "",
                "Protocol_Class": "",
                "Severity": "HARD_STOP" if "DUPLICATE_CONTENT_MISMATCH" in issue else "REVIEW",
                "Anomaly_Codes": issue.split(":", 1)[0],
                "Details": issue,
                "Requires_Human_Review": "YES",
            }
        )
    return rows


def _schema_anomalies(schema: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if schema.empty:
        return rows
    failed = schema.loc[schema["Schema_Status"].eq("ERROR")]
    for item in failed.itertuples(index=False):
        rows.append(
            {
                "Anomaly_Scope": "SCHEMA",
                "Batch_ID": item.Batch_ID,
                "Cell_ID": item.Cell_ID,
                "Source_File": item.Source_File,
                "Raw_Cycle_Index": pd.NA,
                "Schema_Variant_ID": item.Schema_Variant_ID,
                "Protocol_Class": "",
                "Severity": "HARD_STOP",
                "Anomaly_Codes": "SCHEMA_ERROR",
                "Details": item.Schema_Issues,
                "Requires_Human_Review": "YES",
            }
        )
    return rows


def _cycle_anomalies(protocol: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if protocol.empty:
        return rows
    flagged = protocol.loc[
        protocol["Protocol_Class"].eq("UNKNOWN_PROTOCOL")
        | protocol["Primary_Capacity_Eligible"].eq("REVIEW")
        | protocol["Expected_Protocol_Match"].ne("YES")
        | protocol["Issues"].fillna("").ne("")
    ]
    for item in flagged.itertuples(index=False):
        codes = str(item.Issues)
        if item.Protocol_Class == "UNKNOWN_PROTOCOL" and "UNKNOWN_PROTOCOL" not in codes:
            codes = "|".join(filter(None, [codes, "UNKNOWN_PROTOCOL"]))
        code_set = {code for code in codes.split("|") if code}
        known_info = bool(code_set) and code_set.issubset(
            {
                "DESCRIPTION_CHARGE_CURRENT_MISMATCH",
                "CC_PLATEAU_NOT_OBSERVED_HIGH_SOC_START",
                "GEO_AGING_REACHED_FULL_CUTOFF",
            }
        )
        if known_info:
            if "DESCRIPTION_CHARGE_CURRENT_MISMATCH" in code_set:
                known_info = bool(
                    int(item.Batch_ID) == 5
                    and item.Protocol_Class == "RANDOM_WALK_AGING"
                    and item.Primary_Capacity_Eligible == "NO"
                )
            if code_set.intersection(
                {
                    "CC_PLATEAU_NOT_OBSERVED_HIGH_SOC_START",
                    "GEO_AGING_REACHED_FULL_CUTOFF",
                }
            ):
                known_info = bool(
                    known_info
                    and int(item.Batch_ID) == 6
                    and item.Protocol_Class == "GEO_AGING_PARTIAL"
                    and item.Primary_Capacity_Eligible == "NO"
                )
        severity = "INFO" if known_info else "REVIEW"
        requires_review = "NO" if known_info else "YES"
        rows.append(
            {
                "Anomaly_Scope": "CYCLE_PROTOCOL",
                "Batch_ID": item.Batch_ID,
                "Cell_ID": item.Cell_ID,
                "Source_File": item.Source_File,
                "Raw_Cycle_Index": item.Raw_Cycle_Index,
                "Schema_Variant_ID": item.Schema_Variant_ID,
                "Protocol_Class": item.Protocol_Class,
                "Severity": severity,
                "Anomaly_Codes": codes or "PROTOCOL_REVIEW",
                "Details": (
                    f"confidence={item.Protocol_Confidence};"
                    f"expected_match={item.Expected_Protocol_Match};"
                    f"primary_eligibility={item.Primary_Capacity_Eligible}"
                ),
                "Requires_Human_Review": requires_review,
            }
        )
    return rows


def _primary_candidates(protocol: pd.DataFrame) -> pd.DataFrame:
    if protocol.empty:
        return protocol.copy()
    mask = (
        protocol["Primary_Capacity_Eligible"].isin(["YES", "REVIEW"])
        | protocol["Protocol_Class"].eq("INITIAL_CAPACITY_TEST")
    )
    candidates = protocol.loc[mask].copy()
    candidates.insert(
        candidates.columns.get_loc("Protocol_Class") + 1,
        "Candidate_Role",
        np.where(
            candidates["Protocol_Class"].eq("INITIAL_CAPACITY_TEST"),
            "QREF_INITIAL_TEST_CANDIDATE",
            np.where(
                candidates["Primary_Capacity_Eligible"].eq("YES"),
                "PRIMARY_RECURRING_CAPACITY_CANDIDATE",
                "HUMAN_REVIEW_CANDIDATE",
            ),
        ),
    )
    return candidates.reset_index(drop=True)


def _representative_plan(
    protocol: pd.DataFrame,
    schema: pd.DataFrame,
    inventory: InventoryResult,
) -> tuple[pd.DataFrame, dict[int, str]]:
    rows: list[dict[str, Any]] = []
    selected: dict[int, str] = {}
    for batch_id in range(2, 7):
        batch_schema = schema.loc[
            schema["Batch_ID"].eq(batch_id) & schema["Schema_Status"].eq("PASS")
        ].copy()
        batch_schema["Data_Cycle_Count"] = pd.to_numeric(
            batch_schema["Data_Cycle_Count"], errors="coerce"
        )
        eligible_cells: list[str] = []
        for cell_id in batch_schema["Cell_ID"].astype(str):
            manifest_row = inventory.manifest.loc[
                inventory.manifest["Cell_ID"].eq(cell_id)
                & inventory.manifest["File_Role"].eq("CANONICAL")
            ]
            if not manifest_row.empty and manifest_row.iloc[0]["Inventory_Issues"] == "":
                eligible_cells.append(cell_id)
        batch_schema = batch_schema.loc[batch_schema["Cell_ID"].isin(eligible_cells)]
        if batch_schema.empty:
            selected[batch_id] = ""
            rows.append(
                {
                    "Batch_ID": batch_id,
                    "Cell_ID": "",
                    "Source_File": "",
                    "Representative_Selection_Reason": "NO_SCHEMA_VALID_REPRESENTATIVE",
                    "Validation_Sample_Type": "UNAVAILABLE",
                    "Raw_Cycle_Index": pd.NA,
                    "Protocol_Class": "",
                    "Primary_Capacity_Eligible": "",
                    "Expected_Protocol_Match": "",
                    "Current_Issues": "",
                    "Human_Review_Decision": "",
                    "Human_Review_Notes": "",
                }
            )
            continue
        median_count = float(batch_schema["Data_Cycle_Count"].median())
        batch_schema["Distance_To_Batch_Median"] = (
            batch_schema["Data_Cycle_Count"] - median_count
        ).abs()
        representative = batch_schema.sort_values(
            ["Distance_To_Batch_Median", "Cell_ID"], kind="stable"
        ).iloc[0]
        cell_id = str(representative["Cell_ID"])
        selected[batch_id] = cell_id
        cell_rows = protocol.loc[protocol["Cell_ID"].eq(cell_id)].sort_values(
            "Raw_Cycle_Index"
        )
        reason = (
            "SCHEMA_PASS;NO_DUPLICATE_AMBIGUITY;"
            f"CYCLE_COUNT_CLOSEST_TO_BATCH_MEDIAN_{median_count:g}"
        )

        selections: list[tuple[str, pd.DataFrame]] = [
            (
                "INITIAL_CYCLE",
                cell_rows.loc[cell_rows["Protocol_Class"].eq("INITIAL_CAPACITY_TEST")].head(1),
            ),
            (
                "NORMAL_PRIMARY_DIAGNOSTIC",
                cell_rows.loc[
                    cell_rows["Primary_Capacity_Eligible"].eq("YES")
                    & cell_rows["Expected_Protocol_Match"].eq("YES")
                ].head(1),
            ),
            (
                "NONPRIMARY_AGING_CYCLE",
                cell_rows.loc[
                    cell_rows["Primary_Capacity_Eligible"].eq("NO")
                    & ~cell_rows["Protocol_Class"].eq("INITIAL_CAPACITY_TEST")
                ].head(1),
            ),
            (
                "LATE_LIFE_PRIMARY_DIAGNOSTIC",
                cell_rows.loc[
                    cell_rows["Primary_Capacity_Eligible"].eq("YES")
                    & cell_rows["Expected_Protocol_Match"].eq("YES")
                ].tail(1),
            ),
        ]
        anomaly_rows = cell_rows.loc[
            cell_rows["Protocol_Class"].eq("UNKNOWN_PROTOCOL")
            | cell_rows["Primary_Capacity_Eligible"].eq("REVIEW")
            | cell_rows["Expected_Protocol_Match"].ne("YES")
            | cell_rows["Issues"].fillna("").ne("")
        ]
        for item in anomaly_rows.itertuples(index=False):
            selections.append(
                ("ANOMALY_REQUIRES_REVIEW", cell_rows.loc[cell_rows.index == item.Index])
                if hasattr(item, "Index")
                else (
                    "ANOMALY_REQUIRES_REVIEW",
                    cell_rows.loc[cell_rows["Raw_Cycle_Index"].eq(item.Raw_Cycle_Index)],
                )
            )
        seen: set[tuple[str, int]] = set()
        for sample_type, selection in selections:
            if selection.empty:
                rows.append(
                    {
                        "Batch_ID": batch_id,
                        "Cell_ID": cell_id,
                        "Source_File": str(representative["Source_File"]),
                        "Representative_Selection_Reason": reason,
                        "Validation_Sample_Type": f"{sample_type}_UNAVAILABLE",
                        "Raw_Cycle_Index": pd.NA,
                        "Protocol_Class": "",
                        "Primary_Capacity_Eligible": "",
                        "Expected_Protocol_Match": "",
                        "Current_Issues": "",
                        "Human_Review_Decision": "",
                        "Human_Review_Notes": "",
                    }
                )
                continue
            item = selection.iloc[0]
            key = (sample_type, int(item["Raw_Cycle_Index"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "Batch_ID": batch_id,
                    "Cell_ID": cell_id,
                    "Source_File": item["Source_File"],
                    "Representative_Selection_Reason": reason,
                    "Validation_Sample_Type": sample_type,
                    "Raw_Cycle_Index": item["Raw_Cycle_Index"],
                    "Protocol_Class": item["Protocol_Class"],
                    "Primary_Capacity_Eligible": item["Primary_Capacity_Eligible"],
                    "Expected_Protocol_Match": item["Expected_Protocol_Match"],
                    "Current_Issues": item["Issues"],
                    "Human_Review_Decision": "",
                    "Human_Review_Notes": "",
                }
            )
    plan = pd.DataFrame(rows, columns=REPRESENTATIVE_COLUMNS)
    if not plan.empty:
        plan["Raw_Cycle_Index"] = pd.array(plan["Raw_Cycle_Index"], dtype="Int64")
    return plan, selected


def _review_reasons(
    inventory: InventoryResult,
    schema: pd.DataFrame,
    protocol: pd.DataFrame,
    canonical_counts: dict[int, int],
) -> list[str]:
    reasons = list(inventory.issues)
    for batch_id, expected in EXPECTED_CANONICAL_COUNTS.items():
        actual = canonical_counts.get(batch_id, 0)
        if actual != expected:
            reasons.append(
                f"CANONICAL_COUNT_MISMATCH:Batch-{batch_id}:found={actual};expected={expected}"
            )
    if not schema.empty:
        for item in schema.loc[schema["Schema_Status"].eq("ERROR")].itertuples(index=False):
            reasons.append(f"SCHEMA_UNREADABLE:{item.Cell_ID}:{item.Schema_Issues}")
    if protocol.empty:
        reasons.append("NO_PROTOCOL_ROWS_SCANNED")
        return list(dict.fromkeys(reasons))

    for batch_id in range(2, 7):
        group = protocol.loc[protocol["Batch_ID"].eq(batch_id)]
        if group.empty:
            reasons.append(f"NO_PROTOCOL_ROWS_SCANNED:Batch-{batch_id}")
            continue
        unknown_fraction = float(group["Protocol_Class"].eq("UNKNOWN_PROTOCOL").mean())
        review_fraction = float(group["Expected_Protocol_Match"].ne("YES").mean())
        if unknown_fraction > 0.01:
            reasons.append(
                f"SYSTEMIC_UNKNOWN_PROTOCOL_RATE:Batch-{batch_id}:{unknown_fraction:.3%}"
            )
        if review_fraction > 0.05:
            reasons.append(
                f"SYSTEMIC_PROTOCOL_REVIEW_RATE:Batch-{batch_id}:{review_fraction:.3%}"
            )

    expected_b3 = {
        "VARIABLE_FULL_DISCHARGE_0P5C",
        "VARIABLE_FULL_DISCHARGE_1C",
        "VARIABLE_FULL_DISCHARGE_2C",
        "VARIABLE_FULL_DISCHARGE_3C",
        "VARIABLE_FULL_DISCHARGE_5C",
    }
    observed_b3 = set(protocol.loc[protocol["Batch_ID"].eq(3), "Protocol_Class"])
    if not expected_b3.issubset(observed_b3):
        reasons.append(
            "BATCH3_DISCHARGE_RATE_CLASSES_MISSING:"
            + "|".join(sorted(expected_b3 - observed_b3))
        )
    for batch_id, diagnostic, aging, failure_code in (
        (4, "CAPACITY_DIAGNOSTIC_1C", "AGING_PARTIAL_DISCHARGE", "BATCH4_DIAGNOSTIC_PARTIAL_DISTINCTION_UNRESOLVED"),
        (5, "CAPACITY_DIAGNOSTIC_1C", "RANDOM_WALK_AGING", "BATCH5_RECURRING_DIAGNOSTIC_STRUCTURE_NOT_FOUND"),
        (6, "CAPACITY_DIAGNOSTIC_0P5C", "GEO_AGING_PARTIAL", "BATCH6_PERIODIC_CAPACITY_TESTS_NOT_FOUND"),
    ):
        classes = set(protocol.loc[protocol["Batch_ID"].eq(batch_id), "Protocol_Class"])
        if diagnostic not in classes or aging not in classes:
            reasons.append(failure_code)
        cell_groups = protocol.loc[protocol["Batch_ID"].eq(batch_id)].groupby("Cell_ID")
        missing_cells = [
            cell_id
            for cell_id, cell_rows in cell_groups
            if diagnostic not in set(cell_rows["Protocol_Class"])
        ]
        if missing_cells:
            reasons.append(f"{failure_code}:cells={'|'.join(sorted(missing_cells))}")
    return list(dict.fromkeys(reasons))


def _scan_log(
    *,
    started: datetime,
    finished: datetime,
    workspace: Path,
    output_dir: Path,
    inventory: InventoryResult,
    schema: pd.DataFrame,
    protocol: pd.DataFrame,
    primary: pd.DataFrame,
    anomalies: pd.DataFrame,
    configs: dict[int, dict[str, Any]],
    status: str,
    reasons: list[str],
    representatives: dict[int, str],
    per_file_timings: list[str],
) -> str:
    lines = [
        "XJTU 55-cell multi-batch protocol scan",
        f"Pipeline_Version={PIPELINE_VERSION}",
        f"Command=python xjtu_pipeline.py scan",
        f"Started_UTC={started.isoformat()}",
        f"Finished_UTC={finished.isoformat()}",
        f"Workspace={workspace}",
        f"Output_Directory={output_dir}",
        f"Python={sys.version.replace(chr(10), ' ')}",
        f"Platform={platform.platform()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"scipy={scipy.__version__}",
        f"h5py={h5py.__version__}",
        "",
        "RAW_INPUT_POLICY=READ_ONLY;NO_RENAME;NO_MOVE;NO_WRITE",
        "BATCH1_POLICY=FROZEN_REFERENCE;RAW_NOT_REPROCESSED;QC_NOT_RECALCULATED",
        f"SCAN_STATUS={status}",
        "",
        "CANONICAL_COUNTS",
    ]
    canonical = _canonical_counts(inventory)
    for batch_id in range(1, 7):
        lines.append(
            f"Batch-{batch_id}={canonical.get(batch_id, 0)};"
            f"expected={EXPECTED_CANONICAL_COUNTS[batch_id]}"
        )
    lines.extend(["", "INPUT_MANIFEST"])
    for item in inventory.manifest.itertuples(index=False):
        lines.append(
            f"Batch-{item.Batch_ID}|{item.Source_File}|role={item.File_Role}|"
            f"bytes={item.File_Size_Bytes}|sha256={item.SHA256}|included={item.Included_In_Analysis}"
        )
    lines.extend(["", "PER_FILE_SCAN"])
    lines.extend(per_file_timings or ["NONE"])
    lines.extend(["", "SCHEMA_SUMMARY"])
    if schema.empty:
        lines.append("NONE")
    else:
        for (variant, status_value), count in schema.groupby(
            ["Schema_Variant_ID", "Schema_Status"], dropna=False
        ).size().items():
            lines.append(f"{variant}|{status_value}|files={count}")
    lines.extend(["", "PROTOCOL_COUNTS"])
    if protocol.empty:
        lines.append("NONE")
    else:
        for (batch_id, protocol_class), count in protocol.groupby(
            ["Batch_ID", "Protocol_Class"]
        ).size().items():
            lines.append(f"Batch-{batch_id}|{protocol_class}|cycles={count}")
    lines.extend(
        [
            "",
            f"Primary_Candidate_Rows={len(primary)}",
            f"Protocol_Anomaly_Rows={len(anomalies)}",
            f"Unknown_Protocol_Rows={int(protocol['Protocol_Class'].eq('UNKNOWN_PROTOCOL').sum()) if not protocol.empty else 0}",
            "",
            "REPRESENTATIVE_BATTERIES",
        ]
    )
    for batch_id in range(2, 7):
        lines.append(f"Batch-{batch_id}={representatives.get(batch_id, '') or 'UNAVAILABLE'}")
    lines.extend(["", "CONFIG_CANONICAL_JSON_SHA256"])
    for batch_id, config in configs.items():
        encoded = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
        lines.append(f"batch{batch_id}_config_draft.json={_sha256_bytes(encoded)}")
    lines.extend(["", "REVIEW_REASONS"])
    lines.extend(reasons or ["NONE_BEYOND_REQUIRED_HUMAN_PROTOCOL_APPROVAL"])
    lines.extend(
        [
            "",
            "VALIDATE_EXECUTED=NO",
            "RUN_EXECUTED=NO",
            "SCAN COMPLETE — WAITING FOR HUMAN PROTOCOL REVIEW",
        ]
    )
    return "\n".join(lines) + "\n"


def run_scan(
    workspace: Path,
    output_dir: Path,
    frozen_batch1_dir: Path,
) -> ScanResult:
    workspace = workspace.resolve()
    output_dir = output_dir.resolve()
    frozen_batch1_dir = frozen_batch1_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex[:10]}"
    temp_dir.mkdir(parents=False, exist_ok=False)
    started = datetime.now(timezone.utc)
    inventory = scan_inventory(workspace, frozen_batch1_dir)
    configs = all_batch_configs()
    schema_rows = _frozen_batch1_schema_rows(frozen_batch1_dir)
    protocol_records: list[dict[str, Any]] = []
    per_file_timings: list[str] = []

    if not inventory.hard_stop:
        for batch_id in range(2, 7):
            config = configs[batch_id]
            for canonical in inventory.canonical_by_batch.get(batch_id, ()):  # read-only
                file_started = time.perf_counter()
                scan: MatFileScan = scan_mat_schema(
                    canonical.path, batch_id, canonical.cell_id, config
                )
                schema_rows.append(dict(scan.schema_row))
                cycle_rows: list[dict[str, Any]] = []
                if scan.schema_row["Schema_Status"] == "PASS":
                    cycle_rows = scan_protocol_cycles(scan, config)
                    protocol_records.extend(cycle_rows)
                per_file_timings.append(
                    f"Batch-{batch_id}|{canonical.path.name}|"
                    f"schema={scan.schema_row['Schema_Status']}|"
                    f"cycles={scan.schema_row['Data_Cycle_Count']}|"
                    f"protocol_rows={len(cycle_rows)}|elapsed_s={time.perf_counter() - file_started:.3f}"
                )
                del cycle_rows
                del scan
                gc.collect()

    variant_map = _schema_variant_map(schema_rows)
    for row in schema_rows:
        signature = str(row.get("Schema_Variant_Signature", ""))
        if signature:
            row["Schema_Variant_ID"] = variant_map[signature]
    for row in protocol_records:
        source_signature = next(
            (
                str(schema_row["Schema_Variant_Signature"])
                for schema_row in schema_rows
                if schema_row["Batch_ID"] == row["Batch_ID"]
                and schema_row["Cell_ID"] == row["Cell_ID"]
                and schema_row["Source_File"] == row["Source_File"]
            ),
            "",
        )
        row["Schema_Variant_ID"] = variant_map.get(source_signature, "UNRESOLVED")

    schema = pd.DataFrame(schema_rows, columns=SCHEMA_COLUMNS)
    if not schema.empty:
        schema["Batch_ID"] = pd.array(schema["Batch_ID"], dtype="Int64")
        schema["Data_Cycle_Count"] = pd.array(schema["Data_Cycle_Count"], dtype="Int64")
        schema["Summary_Cycle_Life"] = pd.array(schema["Summary_Cycle_Life"], dtype="Int64")
        schema = schema.sort_values(["Batch_ID", "Cell_ID"]).reset_index(drop=True)
    protocol = protocol_rows_frame(protocol_records)
    descriptions = description_catalog(protocol)
    primary = _primary_candidates(protocol)
    anomaly_rows = _inventory_anomalies(inventory)
    anomaly_rows.extend(_schema_anomalies(schema))
    anomaly_rows.extend(_cycle_anomalies(protocol))
    anomalies = pd.DataFrame(anomaly_rows, columns=ANOMALY_COLUMNS)
    if not anomalies.empty:
        anomalies["Batch_ID"] = pd.array(anomalies["Batch_ID"], dtype="Int64")
        anomalies["Raw_Cycle_Index"] = pd.array(
            anomalies["Raw_Cycle_Index"], dtype="Int64"
        )
        anomalies = anomalies.sort_values(
            ["Batch_ID", "Cell_ID", "Raw_Cycle_Index", "Anomaly_Scope"],
            na_position="first",
        ).reset_index(drop=True)
    representatives_plan, representatives = _representative_plan(
        protocol, schema, inventory
    )
    canonical_counts = _canonical_counts(inventory)
    reasons = _review_reasons(
        inventory, schema, protocol, canonical_counts
    )
    status = "NEEDS_REVIEW" if reasons else "AWAITING_HUMAN_PROTOCOL_REVIEW"

    protocol_counts_by_batch = (
        protocol.groupby(["Batch_ID", "Protocol_Class"]).size().to_dict()
        if not protocol.empty
        else {}
    )
    for batch_id, config in configs.items():
        config["scan_observation_summary"] = {
            "scan_status": status,
            "canonical_files_observed": canonical_counts.get(batch_id, 0),
            "cycles_scanned": int(protocol.loc[protocol["Batch_ID"].eq(batch_id)].shape[0])
            if not protocol.empty
            else 0,
            "protocol_class_counts": {
                protocol_class: int(count)
                for (observed_batch, protocol_class), count in protocol_counts_by_batch.items()
                if int(observed_batch) == batch_id
            },
            "unknown_protocol_count": int(
                protocol.loc[protocol["Batch_ID"].eq(batch_id), "Protocol_Class"]
                .eq("UNKNOWN_PROTOCOL")
                .sum()
            )
            if not protocol.empty
            else 0,
            "human_approval_required_before_validate_or_run": True,
        }

    finished = datetime.now(timezone.utc)
    log_text = _scan_log(
        started=started,
        finished=finished,
        workspace=workspace,
        output_dir=output_dir,
        inventory=inventory,
        schema=schema,
        protocol=protocol,
        primary=primary,
        anomalies=anomalies,
        configs=configs,
        status=status,
        reasons=reasons,
        representatives=representatives,
        per_file_timings=per_file_timings,
    )

    _write_csv(inventory.manifest, temp_dir / "xjtu_file_manifest.csv")
    _write_csv(schema, temp_dir / "xjtu_schema_inventory.csv")
    _write_csv(descriptions, temp_dir / "xjtu_description_catalog.csv")
    _write_csv(protocol, temp_dir / "xjtu_protocol_inventory.csv")
    _write_csv(primary, temp_dir / "xjtu_primary_capacity_candidate_inventory.csv")
    _write_csv(anomalies, temp_dir / "xjtu_protocol_anomalies.csv")
    _write_csv(inventory.duplicates, temp_dir / "xjtu_duplicate_report.csv")
    for batch_id, config in configs.items():
        _write_json(config, temp_dir / f"batch{batch_id}_config_draft.json")
    (temp_dir / "processing_log_scan.txt").write_text(log_text, encoding="utf-8")
    _write_csv(representatives_plan, temp_dir / "representative_validation_plan.csv")

    actual_files = {path.name for path in temp_dir.iterdir() if path.is_file()}
    expected_files = set(SCAN_OUTPUT_FILES)
    if actual_files != expected_files:
        raise RuntimeError(
            "SCAN_OUTPUT_SET_MISMATCH:"
            f"missing={sorted(expected_files - actual_files)};"
            f"extra={sorted(actual_files - expected_files)}"
        )
    for csv_path in temp_dir.glob("*.csv"):
        pd.read_csv(csv_path, keep_default_na=False, low_memory=False)
    if "SCAN COMPLETE — WAITING FOR HUMAN PROTOCOL REVIEW" not in log_text:
        raise RuntimeError("SCAN_COMPLETION_MARKER_MISSING")
    temp_dir.rename(output_dir)

    cycle_counts = {
        int(batch_id): int(count)
        for batch_id, count in protocol.groupby("Batch_ID").size().items()
    } if not protocol.empty else {}
    frozen_cycle_count = int(
        pd.to_numeric(
            schema.loc[schema["Batch_ID"].eq(1), "Data_Cycle_Count"], errors="coerce"
        ).sum()
    )
    cycle_counts[1] = frozen_cycle_count
    protocol_counts = {
        f"Batch-{int(batch_id)}:{protocol_class}": int(count)
        for (batch_id, protocol_class), count in protocol_counts_by_batch.items()
    }
    primary_counts = {
        batch_id: int(
            protocol.loc[
                protocol["Batch_ID"].eq(batch_id)
                & protocol["Primary_Capacity_Eligible"].eq("YES")
            ].shape[0]
        )
        for batch_id in range(2, 7)
    }
    schema_variant_counts = {
        str(variant): int(count)
        for variant, count in schema.groupby("Schema_Variant_ID").size().items()
    }
    return ScanResult(
        output_dir=output_dir,
        scan_status=status,
        canonical_counts=canonical_counts,
        duplicate_count=int(len(inventory.duplicates)),
        schema_variant_counts=schema_variant_counts,
        cycle_counts=cycle_counts,
        protocol_counts=protocol_counts,
        primary_candidate_counts=primary_counts,
        unknown_protocol_count=int(protocol["Protocol_Class"].eq("UNKNOWN_PROTOCOL").sum())
        if not protocol.empty
        else 0,
        anomaly_count=int(len(anomalies)),
        representatives=representatives,
        review_reasons=tuple(reasons),
    )


def output_hash_manifest(scan_dir: Path) -> dict[str, str]:
    """Hashes used by the explicit human protocol-approval gate."""
    required = [
        "xjtu_protocol_inventory.csv",
        *[f"batch{batch_id}_config_draft.json" for batch_id in range(2, 7)],
    ]
    return {name: _sha256_path(scan_dir / name) for name in required}
