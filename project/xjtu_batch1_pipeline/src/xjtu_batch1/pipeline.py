from __future__ import annotations

import json
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
import pandas as pd
import scipy

from . import PIPELINE_VERSION
from .config import canonical_config_text, load_config
from .mat_io import (
    InputInventory,
    InventoryRecord,
    SchemaError,
    SchemaResult,
    assert_input_hashes_unchanged,
    scan_inventory,
    validate_schema,
)
from .plots import generate_all_plots
from .qc import process_battery
from .regression import run_battery1_regressions
from .sensitivity import (
    QREF_DEFINITIONS,
    compute_eol_pairwise,
    compute_eol_sensitivity,
    compute_pilot_summary,
    compute_qref_summary,
    compute_rul_sensitivity,
    compute_soh_labels,
)


QC_COLUMNS = [
    "Cell_ID",
    "Source_File",
    "Raw_Cycle_Index",
    "Aging_Cycle_Index",
    "Protocol_Class",
    "Description",
    "Schema_Valid",
    "Summary_Discharge_Capacity_Ah",
    "Raw_Discharge_Capacity_Ah",
    "Capacity_Difference_Ah",
    "Capacity_Difference_Pct",
    "Charge_Found",
    "Charge_Complete",
    "Charge_Record_Count",
    "Charge_Max_V",
    "Charge_End_Current_A",
    "Charge_Duration_s",
    "Main_Discharge_Found",
    "Discharge_Complete",
    "Discharge_Record_Count",
    "Discharge_Median_Current_A",
    "Discharge_Mean_Current_A",
    "Discharge_Start_V",
    "Discharge_End_V",
    "Discharge_Min_V",
    "Discharge_Duration_s",
    "Voltage_Spike_During_Discharge",
    "Abnormal_Voltage_Jump",
    "Max_Abs_Voltage_Jump_V",
    "Capacity_Jump_Flag",
    "Jump_Reference_Raw_Cycle",
    "Jump_Reference_Capacity_Ah",
    "Jump_Delta_Ah",
    "Jump_Delta_Pct",
    "Jump_Reference_Status",
    "Auto_QC_Flag",
    "Auto_QC_Reason_Codes",
    "Auto_QC_Reason",
    "Primary_Include_Status",
    "Provisional_Include",
]


@dataclass(frozen=True)
class PipelineOutputs:
    output_dir: Path
    manifest: pd.DataFrame
    qc: pd.DataFrame
    flagged: pd.DataFrame
    qref: pd.DataFrame
    soh: pd.DataFrame
    eol: pd.DataFrame
    pairwise: pd.DataFrame
    rul: pd.DataFrame
    pilot: pd.DataFrame
    regression_results: dict[str, str]
    figure_paths: tuple[Path, ...]


class RunLog:
    def __init__(self, echo: bool = True) -> None:
        self.lines: list[str] = []
        self.echo = echo

    def write(self, message: str = "") -> None:
        self.lines.append(message)
        if self.echo:
            print(message, flush=True)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _build_manifest(
    inventory: InputInventory,
    schema_by_id: dict[int, SchemaResult],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in inventory.records:
        schema = schema_by_id[record.battery_id]
        is_canonical = record.file_role == "CANONICAL"
        rows.append(
            {
                "Cell_ID": record.cell_id,
                "Source_File": record.path.name,
                "File_Size_Bytes": record.path.stat().st_size,
                "File_Role": record.file_role,
                "Duplicate_Of": record.duplicate_of,
                "SHA256": record.sha256,
                "Duplicate_SHA256": record.duplicate_sha256,
                "Hash_Match": record.hash_match,
                "Included_In_Analysis": record.included_in_analysis,
                "Data_Cycle_Count": schema.data_cycle_count,
                "Summary_Cycle_Life": schema.summary_cycle_life,
                "Schema_Status": (
                    schema.schema_status
                    if is_canonical
                    else "HASH_EQUIVALENT_TO_CANONICAL_NOT_RELOADED"
                ),
                "Schema_Issues": (
                    "|".join(schema.schema_issues)
                    if is_canonical
                    else "DUPLICATE_CONFIRMED_BY_SHA256"
                ),
                "Processing_Status": (
                    "COMPLETED"
                    if is_canonical and schema.schema_status == "PASS"
                    else (
                        "STOPPED_SCHEMA_ERROR"
                        if is_canonical
                        else "EXCLUDED_DUPLICATE_CONFIRMED"
                    )
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["Data_Cycle_Count"] = pd.array(frame["Data_Cycle_Count"], dtype="Int64")
    frame["Summary_Cycle_Life"] = pd.array(frame["Summary_Cycle_Life"], dtype="Int64")
    return frame


def _log_environment(log: RunLog) -> None:
    log.write(f"Pipeline version: {PIPELINE_VERSION}")
    log.write(f"Processing timestamp UTC: {datetime.now(timezone.utc).isoformat()}")
    log.write(f"Python version: {platform.python_version()}")
    log.write(f"Python executable: {sys.executable}")
    log.write(f"numpy version: {np.__version__}")
    log.write(f"scipy version: {scipy.__version__}")
    log.write(f"pandas version: {pd.__version__}")
    log.write(f"matplotlib version: {matplotlib.__version__}")


def _write_csv(frame: pd.DataFrame, path: Path, config: dict[str, Any]) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        float_format=str(config["csv_float_format"]),
        na_rep="",
        lineterminator="\n",
    )


def _validate_in_memory(
    qc: pd.DataFrame,
    schema_by_id: dict[int, SchemaResult],
    qref: pd.DataFrame,
    eol: pd.DataFrame,
    rul: pd.DataFrame,
) -> list[str]:
    checks: list[str] = []
    if qc[["Cell_ID", "Raw_Cycle_Index"]].duplicated().any():
        raise AssertionError("Duplicate Cell_ID + Raw_Cycle_Index in QC table")
    if qc["Cell_ID"].isna().any() or (qc["Cell_ID"].astype(str).str.len() == 0).any():
        raise AssertionError("Unexpected empty Cell_ID in QC table")
    for battery_id, schema in schema_by_id.items():
        cell_id = f"2C_battery-{battery_id}"
        actual = qc.loc[qc["Cell_ID"] == cell_id, "Raw_Cycle_Index"].astype(int).tolist()
        expected = list(range(1, int(schema.data_cycle_count or 0) + 1))
        if actual != expected:
            raise AssertionError(f"Raw Cycle sequence changed for {cell_id}")
    checks.append("QC keys unique; Raw Cycle indices preserved without compression")

    invalid_keys = set(
        zip(
            qc.loc[qc["Auto_QC_Flag"] == "INVALID_CANDIDATE", "Cell_ID"],
            qc.loc[qc["Auto_QC_Flag"] == "INVALID_CANDIDATE", "Raw_Cycle_Index"].astype(int),
        )
    )
    observed = eol.loc[eol["EOL_Status"] == "OBSERVED"]
    for row in observed.itertuples(index=False):
        key = (row.Cell_ID, int(row.EOL_Raw_Cycle))
        if key in invalid_keys:
            raise AssertionError(f"INVALID cycle triggered EOL: {key}")
        qc_row = qc.loc[
            (qc["Cell_ID"] == row.Cell_ID)
            & (qc["Raw_Cycle_Index"] == int(row.EOL_Raw_Cycle))
        ].iloc[0]
        if qc_row["Primary_Include_Status"] != "YES":
            raise AssertionError(f"Non-primary cycle triggered EOL: {key}")
    checks.append("No INVALID or REVIEW cycle entered primary EOL detection")

    censored_eol = eol.loc[eol["EOL_Status"] == "RIGHT_CENSORED"]
    if censored_eol["EOL_Raw_Cycle"].notna().any() or censored_eol["EOL_Aging_Cycle"].notna().any():
        raise AssertionError("Right-censored EOL contains a fabricated cycle")
    censored_rul = rul.loc[rul["RUL_Status"] == "CENSORED_EOL"]
    if censored_rul["RUL_Cycles"].notna().any():
        raise AssertionError("Censored RUL contains fabricated numeric labels")
    checks.append("Right-censored EOL cycles and RUL labels remain NA")

    qref_lookup = qref.set_index("Cell_ID")
    for cell_id, row in qref_lookup.iterrows():
        cell = qc.loc[qc["Cell_ID"] == cell_id].set_index("Raw_Cycle_Index")
        if row["Qref_initial_test_Status"] == "AVAILABLE":
            source = int(row["Qref_Initial_Test_Source_Cycle"])
            if cell.loc[source, "Primary_Include_Status"] != "YES":
                raise AssertionError(f"Invalid initial-test Qref source for {cell_id}")
        if row["Qref_first_standard_Status"] == "AVAILABLE":
            source = int(row["Qref_First_Standard_Source_Cycle"])
            if cell.loc[source, "Primary_Include_Status"] != "YES":
                raise AssertionError(f"Invalid first-standard Qref source for {cell_id}")
        if row["Qref_early_median5_Status"] == "AVAILABLE":
            sources = [int(value) for value in str(row["Qref_Early_Median5_Source_Cycles"]).split("|")]
            if len(sources) != 5:
                raise AssertionError(f"Median-5 source count is not five for {cell_id}")
            for source in sources:
                if cell.loc[source, "Primary_Include_Status"] != "YES":
                    raise AssertionError(f"Invalid median-5 Qref source for {cell_id}: {source}")
    checks.append("All Qref source cycles are explicit and primary-included")
    return checks


def _validate_written_outputs(
    output_dir: Path,
    schema_by_id: dict[int, SchemaResult],
    log: RunLog,
) -> None:
    required = (
        "xjtu_file_manifest.csv",
        "xjtu_cycle_capacity_qc.csv",
        "xjtu_qc_flagged_cycles.csv",
        "xjtu_qref_summary.csv",
        "xjtu_soh_labels_long.csv",
        "xjtu_eol_sensitivity.csv",
        "xjtu_eol_pairwise_shift.csv",
        "xjtu_rul_sensitivity.csv",
        "xjtu_pilot_summary.csv",
        "xjtu_batch1_config.json",
    )
    for name in required:
        path = output_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty required output: {name}")

    manifest = pd.read_csv(output_dir / "xjtu_file_manifest.csv")
    qc = pd.read_csv(output_dir / "xjtu_cycle_capacity_qc.csv")
    flagged = pd.read_csv(output_dir / "xjtu_qc_flagged_cycles.csv")
    qref = pd.read_csv(output_dir / "xjtu_qref_summary.csv")
    soh = pd.read_csv(output_dir / "xjtu_soh_labels_long.csv")
    eol = pd.read_csv(output_dir / "xjtu_eol_sensitivity.csv")
    pairwise = pd.read_csv(output_dir / "xjtu_eol_pairwise_shift.csv")
    rul = pd.read_csv(output_dir / "xjtu_rul_sensitivity.csv")
    pilot = pd.read_csv(output_dir / "xjtu_pilot_summary.csv")

    if len(manifest.loc[manifest["File_Role"] == "CANONICAL"]) != 8:
        raise AssertionError("Manifest canonical count is not eight")
    duplicate_rows = manifest.loc[manifest["File_Role"] == "DUPLICATE_CONFIRMED"]
    if len(duplicate_rows) != 8:
        raise AssertionError("Manifest confirmed-duplicate count is not eight")
    if not (duplicate_rows["Included_In_Analysis"] == "NO").all():
        raise AssertionError("A duplicate file was included in analysis")
    if not duplicate_rows["Hash_Match"].astype(bool).all():
        raise AssertionError("A duplicate manifest row lacks Hash_Match=TRUE")
    if qc["Source_File"].astype(str).str.contains(r"\(1\)\.mat$", regex=True).any():
        raise AssertionError("Duplicate MAT file entered the cycle QC table")

    if qc[["Cell_ID", "Raw_Cycle_Index"]].duplicated().any():
        raise AssertionError("Duplicate QC key after CSV reload")
    if flagged[["Cell_ID", "Raw_Cycle_Index"]].duplicated().any():
        raise AssertionError("Duplicate flagged-cycle key after CSV reload")
    if (flagged["Auto_QC_Flag"] == "PASS").any():
        raise AssertionError("PASS row entered flagged-cycle output")
    expected_flagged = int((qc["Auto_QC_Flag"] != "PASS").sum())
    if len(flagged) != expected_flagged:
        raise AssertionError("Flagged-cycle output is incomplete")
    for battery_id, schema in schema_by_id.items():
        cell_id = f"2C_battery-{battery_id}"
        cycles = qc.loc[qc["Cell_ID"] == cell_id, "Raw_Cycle_Index"].astype(int).tolist()
        if cycles != list(range(1, int(schema.data_cycle_count or 0) + 1)):
            raise AssertionError(f"Raw Cycle sequence changed after CSV reload for {cell_id}")

    for frame, keys, name in (
        (soh, ["Cell_ID", "Raw_Cycle_Index", "Qref_Definition"], "SOH"),
        (eol, ["Cell_ID", "Qref_Definition"], "EOL"),
        (pairwise, ["Cell_ID", "Definition_A", "Definition_B"], "pairwise EOL"),
        (rul, ["Cell_ID", "Raw_Cycle_Index", "Qref_Definition"], "RUL"),
    ):
        if frame[keys].duplicated().any():
            raise AssertionError(f"Duplicate key in {name} output")
        if frame["Cell_ID"].isna().any():
            raise AssertionError(f"Empty Cell_ID in {name} output")
    if len(qref) != 8 or len(pilot) != 8:
        raise AssertionError("Qref or pilot summary does not contain eight batteries")

    censored_eol = eol.loc[eol["EOL_Status"] == "RIGHT_CENSORED"]
    if censored_eol["EOL_Raw_Cycle"].notna().any():
        raise AssertionError("Censored EOL was filled with a cycle after CSV reload")
    censored_rul = rul.loc[rul["RUL_Status"] == "CENSORED_EOL"]
    if censored_rul["RUL_Cycles"].notna().any():
        raise AssertionError("Censored RUL was filled after CSV reload")
    if len(list((output_dir / "figures").glob("*.png"))) != 10:
        raise AssertionError("Expected ten PNG figures")
    log.write("Post-write CSV validation: PASSED")
    log.write("No duplicate keys; Cell_ID complete; Raw Cycle indices preserved")
    log.write("INVALID cycles excluded from primary EOL; censored RUL values are NA")
    log.write("Qref source cycles remain traceable")


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    echo: bool = True,
) -> PipelineOutputs:
    output_dir = output_dir.resolve()
    input_dir = input_dir.resolve()
    config_path = config_path.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    config = load_config(config_path)
    log = RunLog(echo=echo)
    _log_environment(log)
    log.write(f"Input directory: {input_dir}")
    log.write(f"Requested output directory: {output_dir}")
    log.write("Config values:")
    for line in canonical_config_text(config_path).splitlines():
        log.write(f"  {line}")

    log.write("Phase 1: input inventory and independent schema validation")
    inventory = scan_inventory(input_dir, config)
    canonical_records = [record for record in inventory.records if record.file_role == "CANONICAL"]
    duplicate_records = [record for record in inventory.records if record.file_role == "DUPLICATE_CONFIRMED"]
    log.write(
        f"Input scan: {len(canonical_records)} canonical files and "
        f"{len(duplicate_records)} confirmed byte-identical duplicates"
    )
    for record in inventory.records:
        log.write(
            f"Input file: {record.path.name}; role={record.file_role}; "
            f"bytes={record.path.stat().st_size}; sha256={record.sha256}; "
            f"included={record.included_in_analysis}"
        )

    schema_by_id: dict[int, SchemaResult] = {}
    for battery_id, path in inventory.canonical_by_id.items():
        result = validate_schema(path, f"2C_battery-{battery_id}")
        schema_by_id[battery_id] = result
        log.write(
            f"Schema {path.name}: status={result.schema_status}; data_cycles="
            f"{result.data_cycle_count}; cycle_life={result.summary_cycle_life}; "
            f"issues={'|'.join(result.schema_issues) if result.schema_issues else 'NONE'}"
        )
    schema_failures = [result for result in schema_by_id.values() if result.schema_status != "PASS"]
    if schema_failures:
        names = "|".join(result.source_file for result in schema_failures)
        log.write(f"PIPELINE FAILED: schema errors in {names}")
        raise SchemaError(f"STOPPED_SCHEMA_ERROR:{names}")

    log.write("Phase 2: process canonical 2C_battery-1.mat only")
    battery1 = process_battery(inventory.canonical_by_id[1], 1, config)
    log.write(f"Battery 1 processed cycles: {len(battery1)}")

    log.write("Phase 3: battery-1 regression tests")
    battery1_qref = compute_qref_summary(battery1, config)
    battery1_eol = compute_eol_sensitivity(battery1, battery1_qref, config)
    regression_results = run_battery1_regressions(
        schema_by_id[1], battery1, battery1_qref, battery1_eol, config
    )
    for name, status in regression_results.items():
        log.write(f"Regression test - {name}: {status}")
    log.write("ALL BATTERY-1 REGRESSION TESTS PASSED")

    log.write("Phase 4: process canonical batteries 2-8")
    frames = [battery1]
    for battery_id in range(2, 9):
        frame = process_battery(inventory.canonical_by_id[battery_id], battery_id, config)
        frames.append(frame)
        log.write(f"Battery {battery_id} processed cycles: {len(frame)}")
    qc = pd.concat(frames, ignore_index=True)[QC_COLUMNS]
    qc = qc.sort_values(["Cell_ID", "Raw_Cycle_Index"]).reset_index(drop=True)

    log.write("Phase 5: finalize cycle-level QC tables")
    flagged = qc.loc[qc["Auto_QC_Flag"] != "PASS"].copy()
    flagged = flagged.sort_values(["Cell_ID", "Raw_Cycle_Index"]).reset_index(drop=True)
    for cell_id, cell in qc.groupby("Cell_ID", sort=True):
        counts = cell["Auto_QC_Flag"].value_counts()
        log.write(
            f"QC {cell_id}: cycles={len(cell)}; PASS={int(counts.get('PASS', 0))}; "
            f"CHECK={int(counts.get('CHECK', 0))}; "
            f"INVALID_CANDIDATE={int(counts.get('INVALID_CANDIDATE', 0))}"
        )

    log.write("Phase 6: calculate four Qref definitions")
    qref = compute_qref_summary(qc, config)
    log.write("Phase 7: calculate unclipped SOH labels for primary-included aging cycles")
    soh = compute_soh_labels(qc, qref)
    log.write("Phase 8: calculate EOL sensitivity and right censoring")
    eol = compute_eol_sensitivity(qc, qref, config)
    pairwise = compute_eol_pairwise(eol)
    log.write("Phase 9: calculate RUL labels without fabricating censored targets")
    rul = compute_rul_sensitivity(qc, eol)
    log.write("Phase 10: prepare pilot summary and figures")
    pilot = compute_pilot_summary(qc, qref, eol, pairwise)

    for check in _validate_in_memory(qc, schema_by_id, qref, eol, rul):
        log.write(f"In-memory validation: {check}: PASSED")
    manifest = _build_manifest(inventory, schema_by_id)
    assert_input_hashes_unchanged(inventory)
    log.write("Raw MAT SHA-256 recheck before output: PASSED")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}_tmp_", dir=str(output_dir.parent))
    )
    figure_paths: list[Path] = []
    try:
        _write_csv(manifest, temporary / "xjtu_file_manifest.csv", config)
        _write_csv(qc, temporary / "xjtu_cycle_capacity_qc.csv", config)
        _write_csv(flagged, temporary / "xjtu_qc_flagged_cycles.csv", config)
        _write_csv(qref, temporary / "xjtu_qref_summary.csv", config)
        _write_csv(soh, temporary / "xjtu_soh_labels_long.csv", config)
        _write_csv(eol, temporary / "xjtu_eol_sensitivity.csv", config)
        _write_csv(pairwise, temporary / "xjtu_eol_pairwise_shift.csv", config)
        _write_csv(rul, temporary / "xjtu_rul_sensitivity.csv", config)
        _write_csv(pilot, temporary / "xjtu_pilot_summary.csv", config)
        shutil.copy2(config_path, temporary / "xjtu_batch1_config.json")
        figure_paths = generate_all_plots(qc, qref, eol, temporary / "figures", config)

        log.write("Phase 11: re-read and validate all output CSV files")
        _validate_written_outputs(temporary, schema_by_id, log)
        assert_input_hashes_unchanged(inventory)
        log.write("Raw MAT SHA-256 final recheck: PASSED")
        log.write("ALL REGRESSION TESTS PASSED")
        log.write("PIPELINE COMPLETED SUCCESSFULLY")
        log.save(temporary / "processing_log.txt")
        temporary.rename(output_dir)
    except Exception as exc:
        log.write(f"PIPELINE FAILED: {type(exc).__name__}: {exc}")
        log.save(temporary / "processing_log.txt")
        raise

    final_figure_paths = tuple(output_dir / "figures" / path.name for path in figure_paths)
    return PipelineOutputs(
        output_dir=output_dir,
        manifest=manifest,
        qc=qc,
        flagged=flagged,
        qref=qref,
        soh=soh,
        eol=eol,
        pairwise=pairwise,
        rul=rul,
        pilot=pilot,
        regression_results=regression_results,
        figure_paths=final_figure_paths,
    )

