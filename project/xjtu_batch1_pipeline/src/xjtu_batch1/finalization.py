from __future__ import annotations

import hashlib
import platform
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import load_config
from .qc import STANDARD_PROTOCOL
from .sensitivity import (
    QREF_DEFINITIONS,
    compute_eol_pairwise,
    compute_eol_sensitivity,
    compute_pilot_summary,
    compute_qref_summary,
    compute_rul_sensitivity,
    compute_soh_labels,
)


FINALIZATION_VERSION = "1.0.0"

MANUAL_REVIEW_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "Cell_ID": "2C_battery-1",
        "Raw_Cycle_Index": 390,
        "Expected_Auto_QC_Flag": "INVALID_CANDIDATE",
        "Manual_Decision": "EXCLUDE",
        "Final_Include": "NO",
        "Manual_Reason": (
            "Main discharge did not reach the expected 2.5 V cutoff and showed "
            "abnormal voltage behavior during discharge. Keep the original "
            "INVALID_CANDIDATE status and exclude from primary analysis."
        ),
    },
    {
        "Cell_ID": "2C_battery-3",
        "Raw_Cycle_Index": 393,
        "Expected_Auto_QC_Flag": "INVALID_CANDIDATE",
        "Manual_Decision": "EXCLUDE",
        "Final_Include": "NO",
        "Manual_Reason": (
            "Charge reached approximately 4.2002 V but CV charging terminated "
            "at approximately 0.2018 A rather than the normal approximately "
            "0.10 A cutoff. The following discharge was complete, but the "
            "capacity is not protocol-comparable because the starting SOC was "
            "affected by incomplete charging. Do not impute or correct capacity."
        ),
    },
)

SOURCE_FILES = (
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
    "processing_log.txt",
)

OUTPUT_FILES = (
    "xjtu_manual_qc_review.csv",
    "xjtu_cycle_capacity_final.csv",
    "xjtu_qref_summary_final.csv",
    "xjtu_soh_labels_long_final.csv",
    "xjtu_eol_sensitivity_final.csv",
    "xjtu_eol_pairwise_shift_final.csv",
    "xjtu_rul_sensitivity_final.csv",
    "xjtu_pilot_summary_final.csv",
    "xjtu_pilot_final_report.txt",
    "processing_log_final.txt",
)


@dataclass(frozen=True)
class FinalizationOutputs:
    output_dir: Path
    manual_review: pd.DataFrame
    cycle_final: pd.DataFrame
    qref: pd.DataFrame
    soh: pd.DataFrame
    eol: pd.DataFrame
    pairwise: pd.DataFrame
    rul: pd.DataFrame
    pilot: pd.DataFrame
    regression_results: dict[str, str]


class FinalizationLog:
    def __init__(self, echo: bool = True) -> None:
        self.echo = echo
        self.lines: list[str] = []

    def write(self, message: str) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        self.lines.append(line)
        if self.echo:
            print(line, flush=True)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(pilot_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in SOURCE_FILES:
        path = pilot_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty pilot v1 source file: {path}")
        hashes[name] = sha256_file(path)
    return hashes


def _assert_hashes_unchanged(pilot_dir: Path, expected: Mapping[str, str]) -> None:
    for name, expected_hash in expected.items():
        actual_hash = sha256_file(pilot_dir / name)
        if actual_hash != expected_hash:
            raise AssertionError(f"Pilot v1 source changed during finalization: {name}")


def _verify_raw_mat_hashes(manifest: pd.DataFrame, raw_dir: Path) -> None:
    required = {"Source_File", "SHA256"}
    missing = required.difference(manifest.columns)
    if missing:
        raise AssertionError(f"Manifest lacks raw hash fields: {sorted(missing)}")
    for row in manifest.itertuples(index=False):
        path = raw_dir / str(row.Source_File)
        if not path.is_file():
            raise FileNotFoundError(f"Raw MAT file listed in manifest is missing: {path}")
        if sha256_file(path) != str(row.SHA256):
            raise AssertionError(f"Raw MAT SHA-256 mismatch: {path.name}")


def _read_sources(pilot_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "manifest": pd.read_csv(pilot_dir / "xjtu_file_manifest.csv"),
        "qc": pd.read_csv(pilot_dir / "xjtu_cycle_capacity_qc.csv"),
        "qc_text": pd.read_csv(
            pilot_dir / "xjtu_cycle_capacity_qc.csv",
            dtype=str,
            keep_default_na=False,
        ),
        "qref": pd.read_csv(pilot_dir / "xjtu_qref_summary.csv"),
        "soh": pd.read_csv(pilot_dir / "xjtu_soh_labels_long.csv"),
        "eol": pd.read_csv(pilot_dir / "xjtu_eol_sensitivity.csv"),
        "pairwise": pd.read_csv(pilot_dir / "xjtu_eol_pairwise_shift.csv"),
        "rul": pd.read_csv(pilot_dir / "xjtu_rul_sensitivity.csv"),
        "pilot": pd.read_csv(pilot_dir / "xjtu_pilot_summary.csv"),
    }


def apply_manual_review(
    qc: pd.DataFrame,
    qc_text: pd.DataFrame,
    manual_records: Sequence[Mapping[str, Any]] = MANUAL_REVIEW_RECORDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Append a manual-review layer without changing any automatic QC field."""

    required = {
        "Cell_ID",
        "Raw_Cycle_Index",
        "Aging_Cycle_Index",
        "Protocol_Class",
        "Summary_Discharge_Capacity_Ah",
        "Auto_QC_Flag",
        "Auto_QC_Reason_Codes",
    }
    missing = required.difference(qc.columns)
    if missing:
        raise AssertionError(f"Automatic QC source lacks required fields: {sorted(missing)}")
    if len(qc) != len(qc_text):
        raise AssertionError("Numeric and text QC reads have different row counts")
    if qc[["Cell_ID", "Raw_Cycle_Index"]].duplicated().any():
        raise AssertionError("Automatic QC source has duplicate cycle keys")

    numeric_keys = list(
        zip(qc["Cell_ID"].astype(str), qc["Raw_Cycle_Index"].astype(int))
    )
    text_keys = list(
        zip(qc_text["Cell_ID"].astype(str), qc_text["Raw_Cycle_Index"].astype(int))
    )
    if numeric_keys != text_keys:
        raise AssertionError("Numeric and text QC reads are not row-aligned")

    final = qc.copy(deep=True)
    pass_mask = final["Auto_QC_Flag"].eq("PASS")
    final["Manual_Review_Status"] = np.where(pass_mask, "NOT_REQUIRED", "PENDING")
    final["Manual_Decision"] = ""
    final["Manual_Reason"] = ""
    final["Final_Include"] = np.where(pass_mask, "YES", "")
    final["Final_Include_Basis"] = np.where(pass_mask, "AUTO_PASS", "")

    seen: set[tuple[str, int]] = set()
    for record in manual_records:
        cell_id = str(record["Cell_ID"])
        raw_cycle = int(record["Raw_Cycle_Index"])
        key = (cell_id, raw_cycle)
        if key in seen:
            raise AssertionError(f"Duplicate manual-review key: {key}")
        seen.add(key)
        mask = final["Cell_ID"].eq(cell_id) & final["Raw_Cycle_Index"].eq(raw_cycle)
        if int(mask.sum()) != 1:
            raise AssertionError(f"Manual-review key does not match one QC row: {key}")
        row = final.loc[mask].iloc[0]
        expected_flag = str(record["Expected_Auto_QC_Flag"])
        if str(row["Auto_QC_Flag"]) != expected_flag:
            raise AssertionError(
                f"Manual-review Auto_QC_Flag mismatch for {key}: "
                f"expected {expected_flag}, found {row['Auto_QC_Flag']}"
            )
        decision = str(record["Manual_Decision"])
        final_include = str(record["Final_Include"])
        if decision != "EXCLUDE" or final_include != "NO":
            raise AssertionError(f"Unsupported authorized manual decision for {key}")
        final.loc[mask, "Manual_Review_Status"] = "COMPLETED"
        final.loc[mask, "Manual_Decision"] = decision
        final.loc[mask, "Manual_Reason"] = str(record["Manual_Reason"])
        final.loc[mask, "Final_Include"] = final_include
        final.loc[mask, "Final_Include_Basis"] = "MANUAL_EXCLUDE"

    unresolved = final.loc[
        final["Auto_QC_Flag"].ne("PASS")
        & final["Manual_Review_Status"].ne("COMPLETED")
    ]
    if not unresolved.empty:
        keys = "|".join(
            f"{row.Cell_ID}:Raw{int(row.Raw_Cycle_Index)}"
            for row in unresolved.itertuples(index=False)
        )
        raise AssertionError(f"UNRESOLVED_NONPASS_MANUAL_REVIEW:{keys}")
    if not final["Final_Include"].isin(["YES", "NO"]).all():
        raise AssertionError("Final_Include contains an unresolved value")
    if not final.loc[pass_mask, "Final_Include"].eq("YES").all():
        raise AssertionError("A PASS cycle was not assigned Final_Include=YES")
    if final.loc[final["Auto_QC_Flag"].eq("INVALID_CANDIDATE"), "Final_Include"].eq("YES").any():
        raise AssertionError("An INVALID_CANDIDATE cycle entered final analysis")

    final_text = qc_text.copy(deep=True)
    appended = (
        "Manual_Review_Status",
        "Manual_Decision",
        "Manual_Reason",
        "Final_Include",
        "Final_Include_Basis",
    )
    for column in appended:
        final_text[column] = final[column].fillna("").astype(str).tolist()

    reviewed = final.loc[final["Manual_Review_Status"].eq("COMPLETED")].copy()
    manual_columns = [
        "Cell_ID",
        "Raw_Cycle_Index",
        "Aging_Cycle_Index",
        "Auto_QC_Flag",
        "Auto_QC_Reason_Codes",
        "Manual_Decision",
        "Final_Include",
        "Manual_Reason",
        "Summary_Discharge_Capacity_Ah",
        "Charge_Max_V",
        "Charge_End_Current_A",
        "Main_Discharge_Found",
        "Discharge_Complete",
        "Discharge_Min_V",
        "Discharge_End_V",
        "Voltage_Spike_During_Discharge",
        "Abnormal_Voltage_Jump",
        "Manual_Review_Status",
        "Final_Include_Basis",
    ]
    manual_review = reviewed[manual_columns].sort_values(
        ["Cell_ID", "Raw_Cycle_Index"]
    ).reset_index(drop=True)
    manual_review["Raw_Cycle_Index"] = pd.array(
        manual_review["Raw_Cycle_Index"], dtype="Int64"
    )
    manual_review["Aging_Cycle_Index"] = pd.array(
        manual_review["Aging_Cycle_Index"], dtype="Int64"
    )
    return final, final_text, manual_review


def compute_final_tables(
    cycle_final: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recalculate sensitivity outputs using Final_Include as the sole inclusion gate."""

    analysis_qc = cycle_final.copy(deep=True)
    analysis_qc["Primary_Include_Status"] = analysis_qc["Final_Include"]
    analysis_qc["Provisional_Include"] = analysis_qc["Final_Include"]

    qref = compute_qref_summary(analysis_qc, config)
    qref["Analysis_Inclusion_Field"] = "Final_Include"

    soh = compute_soh_labels(analysis_qc, qref)
    soh = soh.rename(columns={"Primary_Include_Status": "Final_Include"})

    eol = compute_eol_sensitivity(analysis_qc, qref, config)
    eol = eol.drop(
        columns=[
            "Provisional_EOL_Status",
            "Provisional_EOL_Raw_Cycle",
            "Provisional_EOL_Aging_Cycle",
        ]
    )
    eol["Analysis_Inclusion_Field"] = "Final_Include"

    pairwise = compute_eol_pairwise(eol)
    pairwise["Analysis_Inclusion_Field"] = "Final_Include"

    rul = compute_rul_sensitivity(analysis_qc, eol)
    rul.insert(3, "Final_Include", "YES")

    pilot = compute_pilot_summary(analysis_qc, qref, eol, pairwise)
    final_counts: list[dict[str, Any]] = []
    for cell_id, cell in cycle_final.groupby("Cell_ID", sort=True):
        final_counts.append(
            {
                "Cell_ID": cell_id,
                "Final_Include_YES_Count": int(cell["Final_Include"].eq("YES").sum()),
                "Final_Include_NO_Count": int(cell["Final_Include"].eq("NO").sum()),
                "Manual_Exclude_Count": int(cell["Manual_Decision"].eq("EXCLUDE").sum()),
                "Final_Included_Aging_Cycle_Count": int(
                    (
                        cell["Final_Include"].eq("YES")
                        & cell["Protocol_Class"].eq(STANDARD_PROTOCOL)
                    ).sum()
                ),
                "Unresolved_Manual_Review_Count": int(
                    cell["Manual_Review_Status"].eq("PENDING").sum()
                ),
            }
        )
    pilot = pilot.merge(pd.DataFrame(final_counts), on="Cell_ID", how="left", validate="one_to_one")
    return qref, soh, eol, pairwise, rul, pilot


def _assert_frame_matches(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    columns: Sequence[str],
    label: str,
    tolerance: float = 5e-9,
) -> None:
    actual_view = actual.loc[:, list(columns)].reset_index(drop=True)
    expected_view = expected.loc[:, list(columns)].reset_index(drop=True)
    for column in columns:
        if (
            pd.api.types.is_object_dtype(actual_view[column])
            or pd.api.types.is_string_dtype(actual_view[column])
            or pd.api.types.is_object_dtype(expected_view[column])
            or pd.api.types.is_string_dtype(expected_view[column])
        ):
            actual_view[column] = actual_view[column].astype("string").fillna("")
            expected_view[column] = expected_view[column].astype("string").fillna("")
        elif (
            pd.api.types.is_numeric_dtype(actual_view[column])
            and pd.api.types.is_numeric_dtype(expected_view[column])
        ):
            actual_view[column] = pd.to_numeric(actual_view[column]).astype(float)
            expected_view[column] = pd.to_numeric(expected_view[column]).astype(float)
    try:
        pd.testing.assert_frame_equal(
            actual_view,
            expected_view,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=tolerance,
        )
    except AssertionError as exc:
        raise AssertionError(f"Final recalculation differs from pilot v1 {label}: {exc}") from exc


def validate_final_tables(
    source: Mapping[str, pd.DataFrame],
    cycle_final: pd.DataFrame,
    cycle_final_text: pd.DataFrame,
    manual_review: pd.DataFrame,
    qref: pd.DataFrame,
    soh: pd.DataFrame,
    eol: pd.DataFrame,
    pairwise: pd.DataFrame,
    rul: pd.DataFrame,
    pilot: pd.DataFrame,
) -> dict[str, str]:
    """Run final manual-review regressions and cross-output integrity checks."""

    source_qc = source["qc"]
    source_qc_text = source["qc_text"]
    original_columns = source_qc_text.columns.tolist()
    if not cycle_final_text[original_columns].equals(source_qc_text[original_columns]):
        raise AssertionError("An original automatic QC value changed in the final cycle table")
    if not cycle_final["Auto_QC_Flag"].equals(source_qc["Auto_QC_Flag"]):
        raise AssertionError("Auto_QC_Flag changed during manual finalization")
    for column in ("Raw_Cycle_Index", "Aging_Cycle_Index"):
        left = cycle_final[column].reset_index(drop=True)
        right = source_qc[column].reset_index(drop=True)
        pd.testing.assert_series_equal(left, right, check_names=False, check_dtype=False)
    np.testing.assert_allclose(
        cycle_final["Summary_Discharge_Capacity_Ah"].to_numpy(float),
        source_qc["Summary_Discharge_Capacity_Ah"].to_numpy(float),
        rtol=0.0,
        atol=0.0,
    )

    if len(manual_review) != 2:
        raise AssertionError("Final manual review must contain exactly two rows")
    expected_manual = {
        ("2C_battery-1", 390),
        ("2C_battery-3", 393),
    }
    actual_manual = set(
        zip(manual_review["Cell_ID"], manual_review["Raw_Cycle_Index"].astype(int))
    )
    if actual_manual != expected_manual:
        raise AssertionError("Manual-review cycle keys differ from the authorized set")

    indexed = cycle_final.set_index(["Cell_ID", "Raw_Cycle_Index"])
    regression_results: dict[str, str] = {}
    for cell_id, raw_cycle, name in (
        ("2C_battery-1", 390, "battery1_cycle390_final_exclude"),
        ("2C_battery-3", 393, "battery3_cycle393_final_exclude"),
    ):
        row = indexed.loc[(cell_id, raw_cycle)]
        if row["Final_Include"] != "NO" or row["Manual_Decision"] != "EXCLUDE":
            raise AssertionError(f"{cell_id} Raw Cycle {raw_cycle} was not manually excluded")
        if row["Auto_QC_Flag"] != "INVALID_CANDIDATE":
            raise AssertionError(f"{cell_id} Raw Cycle {raw_cycle} Auto_QC_Flag changed")
        regression_results[name] = "PASSED"

    eol_indexed = eol.set_index(["Cell_ID", "Qref_Definition"])
    for cell_id, name in (
        ("2C_battery-1", "battery1_nominal_eol_right_censored"),
        ("2C_battery-3", "battery3_nominal_eol_right_censored"),
    ):
        row = eol_indexed.loc[(cell_id, "nominal")]
        if row["EOL_Status"] != "RIGHT_CENSORED" or pd.notna(row["EOL_Raw_Cycle"]):
            raise AssertionError(f"{cell_id} nominal EOL is not correctly right-censored")
        regression_results[name] = "PASSED"

    if cycle_final.loc[
        cycle_final["Auto_QC_Flag"].eq("INVALID_CANDIDATE"), "Final_Include"
    ].eq("YES").any():
        raise AssertionError("INVALID_CANDIDATE entered final analysis")
    if not soh["Final_Include"].eq("YES").all():
        raise AssertionError("SOH output contains a non-included cycle")
    valid_keys = set(
        zip(
            cycle_final.loc[cycle_final["Final_Include"].eq("YES"), "Cell_ID"],
            cycle_final.loc[cycle_final["Final_Include"].eq("YES"), "Raw_Cycle_Index"].astype(int),
        )
    )
    soh_keys = set(zip(soh["Cell_ID"], soh["Raw_Cycle_Index"].astype(int)))
    rul_keys = set(zip(rul["Cell_ID"], rul["Raw_Cycle_Index"].astype(int)))
    if not soh_keys.issubset(valid_keys) or not rul_keys.issubset(valid_keys):
        raise AssertionError("SOH or RUL contains a Final_Include=NO cycle")

    for row in qref.itertuples(index=False):
        cell = cycle_final.loc[cycle_final["Cell_ID"].eq(row.Cell_ID)].set_index(
            "Raw_Cycle_Index"
        )
        if row.Qref_initial_test_Status == "AVAILABLE":
            if cell.loc[int(row.Qref_Initial_Test_Source_Cycle), "Final_Include"] != "YES":
                raise AssertionError(f"Initial-test Qref source excluded for {row.Cell_ID}")
        if row.Qref_first_standard_Status == "AVAILABLE":
            if cell.loc[int(row.Qref_First_Standard_Source_Cycle), "Final_Include"] != "YES":
                raise AssertionError(f"First-standard Qref source excluded for {row.Cell_ID}")
        if row.Qref_early_median5_Status == "AVAILABLE":
            sources = [int(value) for value in str(row.Qref_Early_Median5_Source_Cycles).split("|")]
            if len(sources) != 5 or not cell.loc[sources, "Final_Include"].eq("YES").all():
                raise AssertionError(f"Median-5 Qref sources invalid for {row.Cell_ID}")

    observed = eol.loc[eol["EOL_Status"].eq("OBSERVED") & eol["EOL_Raw_Cycle"].notna()]
    for row in observed.itertuples(index=False):
        cycle = indexed.loc[(row.Cell_ID, int(row.EOL_Raw_Cycle))]
        if cycle["Final_Include"] != "YES" or cycle["Auto_QC_Flag"] == "INVALID_CANDIDATE":
            raise AssertionError("Observed EOL was triggered by an excluded cycle")
    censored_eol = eol.loc[eol["EOL_Status"].eq("RIGHT_CENSORED")]
    if censored_eol[["EOL_Raw_Cycle", "EOL_Aging_Cycle"]].notna().any().any():
        raise AssertionError("A right-censored EOL contains a fabricated cycle")
    censored_rul = rul.loc[rul["EOL_Status"].eq("RIGHT_CENSORED")]
    if censored_rul["RUL_Cycles"].notna().any():
        raise AssertionError("A censored RUL contains a fabricated numeric value")

    _assert_frame_matches(qref, source["qref"], source["qref"].columns, "Qref")
    soh_for_compare = soh.rename(columns={"Final_Include": "Primary_Include_Status"})
    _assert_frame_matches(soh_for_compare, source["soh"], source["soh"].columns, "SOH")
    eol_columns = [
        column
        for column in source["eol"].columns
        if not column.startswith("Provisional_EOL_")
    ]
    _assert_frame_matches(eol, source["eol"], eol_columns, "EOL")
    _assert_frame_matches(pairwise, source["pairwise"], source["pairwise"].columns, "pairwise EOL")
    _assert_frame_matches(rul, source["rul"], source["rul"].columns, "RUL")
    _assert_frame_matches(pilot, source["pilot"], source["pilot"].columns, "pilot summary")

    for frame, keys, label in (
        (cycle_final, ["Cell_ID", "Raw_Cycle_Index"], "final cycle"),
        (manual_review, ["Cell_ID", "Raw_Cycle_Index"], "manual review"),
        (soh, ["Cell_ID", "Raw_Cycle_Index", "Qref_Definition"], "SOH"),
        (eol, ["Cell_ID", "Qref_Definition"], "EOL"),
        (pairwise, ["Cell_ID", "Definition_A", "Definition_B"], "pairwise EOL"),
        (rul, ["Cell_ID", "Raw_Cycle_Index", "Qref_Definition"], "RUL"),
    ):
        if frame[keys].duplicated().any():
            raise AssertionError(f"Duplicate key in {label} output")
        if frame["Cell_ID"].isna().any():
            raise AssertionError(f"Empty Cell_ID in {label} output")
    if len(pilot) != 8 or len(qref) != 8:
        raise AssertionError("Final Qref or pilot summary does not contain eight batteries")

    regression_results["auto_qc_and_capacity_unchanged"] = "PASSED"
    regression_results["final_include_gate_only"] = "PASSED"
    regression_results["censored_eol_and_rul_preserved"] = "PASSED"
    regression_results["recalculation_matches_confirmed_v1_exclusions"] = "PASSED"
    regression_results["unique_keys_and_indices_preserved"] = "PASSED"
    return regression_results


def _write_csv(frame: pd.DataFrame, path: Path, config: Mapping[str, Any]) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format=str(config["csv_float_format"]),
        lineterminator="\n",
    )


def _descriptive_report(
    pilot_dir: Path,
    cycle_final: pd.DataFrame,
    manual_review: pd.DataFrame,
    qref: pd.DataFrame,
    eol: pd.DataFrame,
    pairwise: pd.DataFrame,
    regression_results: Mapping[str, str],
) -> str:
    lines = [
        "XJTU Batch-1 Pilot Final Manual-Review Report",
        "=" * 48,
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Source automatic output: {pilot_dir}",
        "Automatic QC flags were preserved without modification.",
        "No capacity smoothing, interpolation, imputation, correction, or cycle renumbering was performed.",
        "",
        "Manual review decisions",
        "-----------------------",
    ]
    for row in manual_review.itertuples(index=False):
        lines.extend(
            [
                f"{row.Cell_ID} Raw Cycle {int(row.Raw_Cycle_Index)}: "
                f"Manual_Decision={row.Manual_Decision}; Final_Include={row.Final_Include}",
                f"  Auto_QC_Flag={row.Auto_QC_Flag}",
                f"  Manual_Reason={row.Manual_Reason}",
            ]
        )

    lines.extend(["", "Cycle-level descriptive counts", "------------------------------"])
    for cell_id, cell in cycle_final.groupby("Cell_ID", sort=True):
        counts = cell["Auto_QC_Flag"].value_counts()
        lines.append(
            f"{cell_id}: total={len(cell)}; PASS={int(counts.get('PASS', 0))}; "
            f"CHECK={int(counts.get('CHECK', 0))}; "
            f"INVALID_CANDIDATE={int(counts.get('INVALID_CANDIDATE', 0))}; "
            f"Final_Include_YES={int(cell['Final_Include'].eq('YES').sum())}; "
            f"Final_Include_NO={int(cell['Final_Include'].eq('NO').sum())}"
        )

    lines.extend(["", "Qref ranges (Ah)", "----------------"])
    for label, column in (
        ("nominal", "Qref_nominal_Ah"),
        ("initial_test", "Qref_initial_test_Ah"),
        ("first_standard", "Qref_first_standard_Ah"),
        ("early_median5", "Qref_early_median5_Ah"),
    ):
        lines.append(f"{label}: {qref[column].min():.6f} to {qref[column].max():.6f}")

    lines.extend(["", "EOL status counts", "-----------------"])
    status_counts = (
        eol.groupby(["Qref_Definition", "EOL_Status"]).size().sort_index()
    )
    for (definition, status), count in status_counts.items():
        lines.append(f"{definition}: {status}={int(count)}")

    lines.extend(["", "Observed pairwise EOL shifts", "-----------------------------"])
    numeric = pairwise.loc[pairwise["Delta_EOL_Cycles"].notna()]
    if numeric.empty:
        lines.append("None")
    else:
        for row in numeric.itertuples(index=False):
            lines.append(
                f"{row.Cell_ID}: {row.Definition_A} -> {row.Definition_B}; "
                f"Delta_EOL_Cycles={float(row.Delta_EOL_Cycles):.0f}"
            )
    changes = pairwise.loc[
        pairwise["Censoring_Status_Change"].isin(
            ["OBSERVED_TO_CENSORED", "CENSORED_TO_OBSERVED"]
        ),
        "Cell_ID",
    ].drop_duplicates().tolist()
    lines.append(
        "Cells with observed/censored status change: "
        + ("|".join(changes) if changes else "None")
    )

    lines.extend(["", "Regression tests", "----------------"])
    for name, status in regression_results.items():
        lines.append(f"{name}: {status}")
    lines.extend(
        [
            "",
            "Final status",
            "------------",
            "ALL FINAL REGRESSION TESTS PASSED",
            "FINAL MANUAL-REVIEW DATASET COMPLETED",
        ]
    )
    return "\n".join(lines) + "\n"


def _validate_written_outputs(
    output_dir: Path,
    source_qc_text: pd.DataFrame,
) -> None:
    for name in OUTPUT_FILES:
        path = output_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty final output: {name}")

    cycle = pd.read_csv(
        output_dir / "xjtu_cycle_capacity_final.csv",
        dtype=str,
        keep_default_na=False,
    )
    original_columns = source_qc_text.columns.tolist()
    if not cycle[original_columns].equals(source_qc_text[original_columns]):
        raise AssertionError("Original automatic QC values changed after final CSV reload")
    manual = pd.read_csv(output_dir / "xjtu_manual_qc_review.csv")
    if len(manual) != 2 or not manual["Final_Include"].eq("NO").all():
        raise AssertionError("Written manual-review file is incomplete")
    if not cycle.loc[cycle["Auto_QC_Flag"].eq("PASS"), "Final_Include"].eq("YES").all():
        raise AssertionError("Written PASS cycle has Final_Include other than YES")
    if not cycle.loc[
        cycle["Auto_QC_Flag"].eq("INVALID_CANDIDATE"), "Final_Include"
    ].eq("NO").all():
        raise AssertionError("Written INVALID_CANDIDATE has Final_Include other than NO")

    eol = pd.read_csv(output_dir / "xjtu_eol_sensitivity_final.csv")
    censored_eol = eol.loc[eol["EOL_Status"].eq("RIGHT_CENSORED")]
    if censored_eol[["EOL_Raw_Cycle", "EOL_Aging_Cycle"]].notna().any().any():
        raise AssertionError("Written right-censored EOL contains a cycle")
    rul = pd.read_csv(output_dir / "xjtu_rul_sensitivity_final.csv")
    if rul.loc[rul["EOL_Status"].eq("RIGHT_CENSORED"), "RUL_Cycles"].notna().any():
        raise AssertionError("Written censored RUL contains a numeric value")

    for name, keys in (
        ("xjtu_cycle_capacity_final.csv", ["Cell_ID", "Raw_Cycle_Index"]),
        ("xjtu_manual_qc_review.csv", ["Cell_ID", "Raw_Cycle_Index"]),
        (
            "xjtu_soh_labels_long_final.csv",
            ["Cell_ID", "Raw_Cycle_Index", "Qref_Definition"],
        ),
        ("xjtu_eol_sensitivity_final.csv", ["Cell_ID", "Qref_Definition"]),
        (
            "xjtu_eol_pairwise_shift_final.csv",
            ["Cell_ID", "Definition_A", "Definition_B"],
        ),
        (
            "xjtu_rul_sensitivity_final.csv",
            ["Cell_ID", "Raw_Cycle_Index", "Qref_Definition"],
        ),
    ):
        frame = pd.read_csv(output_dir / name)
        if frame[keys].duplicated().any():
            raise AssertionError(f"Duplicate key after final CSV reload: {name}")
        if frame["Cell_ID"].isna().any():
            raise AssertionError(f"Empty Cell_ID after final CSV reload: {name}")


def run_finalization(
    pilot_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    config_path: Path,
    echo: bool = True,
) -> FinalizationOutputs:
    pilot_dir = pilot_dir.resolve()
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    config_path = config_path.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing final output directory: {output_dir}")

    log = FinalizationLog(echo=echo)
    log.write(f"Finalization version: {FINALIZATION_VERSION}")
    log.write(f"Python version: {platform.python_version()}")
    log.write(f"numpy version: {np.__version__}")
    log.write(f"pandas version: {pd.__version__}")
    log.write(f"Pilot v1 source directory: {pilot_dir}")
    log.write(f"Raw MAT directory: {raw_dir}")
    log.write(f"Requested final output directory: {output_dir}")

    config = load_config(config_path)
    source_hashes = _source_hashes(pilot_dir)
    for name, digest in source_hashes.items():
        log.write(f"Pilot v1 source SHA256: {name}={digest}")
    source = _read_sources(pilot_dir)
    _verify_raw_mat_hashes(source["manifest"], raw_dir)
    log.write("Raw MAT SHA-256 verification before finalization: PASSED")

    log.write("Phase F1: apply the two authorized manual-review decisions")
    cycle_final, cycle_final_text, manual_review = apply_manual_review(
        source["qc"], source["qc_text"]
    )
    log.write(
        f"Manual review rows={len(manual_review)}; EXCLUDE="
        f"{int(manual_review['Manual_Decision'].eq('EXCLUDE').sum())}"
    )

    log.write("Phase F2: recalculate Qref, SOH, EOL, pairwise EOL, RUL, and summary")
    qref, soh, eol, pairwise, rul, pilot = compute_final_tables(cycle_final, config)

    log.write("Phase F3: run final manual-review regression tests")
    regression_results = validate_final_tables(
        source,
        cycle_final,
        cycle_final_text,
        manual_review,
        qref,
        soh,
        eol,
        pairwise,
        rul,
        pilot,
    )
    for name, status in regression_results.items():
        log.write(f"Final regression test - {name}: {status}")
    log.write("ALL IN-MEMORY FINAL REGRESSION TESTS PASSED")

    _assert_hashes_unchanged(pilot_dir, source_hashes)
    _verify_raw_mat_hashes(source["manifest"], raw_dir)
    log.write("Pilot v1 and raw MAT hash recheck before output: PASSED")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}_tmp_", dir=str(output_dir.parent))
    )
    try:
        log.write("Phase F4: write the ten requested final outputs to a temporary directory")
        manual_review = manual_review.copy()
        manual_review["Source_Auto_QC_SHA256"] = source_hashes[
            "xjtu_cycle_capacity_qc.csv"
        ]
        _write_csv(manual_review, temporary / "xjtu_manual_qc_review.csv", config)
        cycle_final_text.to_csv(
            temporary / "xjtu_cycle_capacity_final.csv",
            index=False,
            encoding="utf-8-sig",
            lineterminator="\n",
        )
        _write_csv(qref, temporary / "xjtu_qref_summary_final.csv", config)
        _write_csv(soh, temporary / "xjtu_soh_labels_long_final.csv", config)
        _write_csv(eol, temporary / "xjtu_eol_sensitivity_final.csv", config)
        _write_csv(pairwise, temporary / "xjtu_eol_pairwise_shift_final.csv", config)
        _write_csv(rul, temporary / "xjtu_rul_sensitivity_final.csv", config)
        _write_csv(pilot, temporary / "xjtu_pilot_summary_final.csv", config)
        report = _descriptive_report(
            pilot_dir,
            cycle_final,
            manual_review,
            qref,
            eol,
            pairwise,
            regression_results,
        )
        (temporary / "xjtu_pilot_final_report.txt").write_text(
            report, encoding="utf-8"
        )
        log.save(temporary / "processing_log_final.txt")

        log.write("Phase F5: re-read and validate all final CSV outputs")
        _validate_written_outputs(temporary, source["qc_text"])
        log.write("Post-write final CSV validation: PASSED")
        _assert_hashes_unchanged(pilot_dir, source_hashes)
        _verify_raw_mat_hashes(source["manifest"], raw_dir)
        log.write("Pilot v1 automatic outputs unchanged: PASSED")
        log.write("Raw MAT SHA-256 final recheck: PASSED")
        log.write("ALL FINAL REGRESSION TESTS PASSED")
        log.write("FINALIZATION COMPLETED SUCCESSFULLY")
        log.save(temporary / "processing_log_final.txt")
        temporary.rename(output_dir)
    except Exception as exc:
        log.write(f"FINALIZATION FAILED: {type(exc).__name__}: {exc}")
        log.save(temporary / "processing_log_final.txt")
        raise

    return FinalizationOutputs(
        output_dir=output_dir,
        manual_review=manual_review,
        cycle_final=cycle_final,
        qref=qref,
        soh=soh,
        eol=eol,
        pairwise=pairwise,
        rul=rul,
        pilot=pilot,
        regression_results=regression_results,
    )
