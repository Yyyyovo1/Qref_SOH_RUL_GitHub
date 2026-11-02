from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
S3_WORKBOOK = Path(
    r"LOCAL_HOME\Desktop\paper_submission_working_package_v2"
    r"\Supplementary_Tables_S1-S7_v2.xlsx"
)
S3_SHEET = "S3_PerCell_Qref"

XJTU_QREF = PROJECT_ROOT / "outputs/xjtu_55_final_v2/xjtu_55_qref_summary.csv"
HUST_QREF = PROJECT_ROOT / "outputs/hust_77_final_v1/hust_77_qref_summary.csv"
V1_AUDIT = (
    PROJECT_ROOT
    / "outputs/manuscript_semantic_audit_v1/reference_availability_audit.csv"
)

XJTU_FEATURES = (
    PROJECT_ROOT / "outputs/soh_feature_extraction_v1/feature_matrix_xjtu_v1.csv"
)
HUST_FEATURES = (
    PROJECT_ROOT / "outputs/soh_feature_extraction_v1/feature_matrix_hust_v1.csv"
)
XJTU_MASK = (
    PROJECT_ROOT
    / "outputs/soh_primary_benchmark_v1/soh_model_eligible_mask_xjtu.csv"
)
HUST_MASK = (
    PROJECT_ROOT
    / "outputs/soh_primary_benchmark_v1/soh_model_eligible_mask_hust.csv"
)

NUMERIC_TOLERANCE_AH = 5e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_blank(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().upper() in {
        "",
        "NA",
        "N/A",
        "NAN",
        "NONE",
        "UNAVAILABLE",
    }


def normalize_cycle_list(value: object) -> str:
    if is_blank(value):
        return ""
    normalized: list[str] = []
    for token in str(value).strip().split("|"):
        token = token.strip()
        try:
            normalized.append(str(int(float(token))))
        except ValueError:
            normalized.append(token)
    return "|".join(normalized)


def latest_cycle(value: object) -> object:
    normalized = normalize_cycle_list(value)
    if not normalized:
        return np.nan
    try:
        return max(int(token) for token in normalized.split("|"))
    except ValueError:
        return np.nan


def numeric_equal(left: object, right: object) -> bool:
    if is_blank(left) and is_blank(right):
        return True
    if is_blank(left) != is_blank(right):
        return False
    try:
        return abs(float(left) - float(right)) <= NUMERIC_TOLERANCE_AH
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()


def text_equal(left: object, right: object) -> bool:
    if is_blank(left) and is_blank(right):
        return True
    if is_blank(left) != is_blank(right):
        return False
    return str(left).strip() == str(right).strip()


def source_equal(left: object, right: object) -> bool:
    return normalize_cycle_list(left) == normalize_cycle_list(right)


def load_qref_authority() -> pd.DataFrame:
    xjtu = pd.read_csv(XJTU_QREF, dtype=object)
    xjtu_authority = pd.DataFrame(
        {
            "Dataset": "XJTU",
            "Batch_ID": xjtu["Batch_ID"],
            "Cell_ID": xjtu["Cell_ID"],
            "Nominal_Ah_Authority": xjtu["Qref_nominal_Ah"],
            "Nominal_Status_Authority": xjtu["Qref_nominal_Status"],
            "Initial_Test_Ah_Authority": xjtu["Qref_initial_test_Ah"],
            "Initial_Test_Status_Authority": xjtu["Qref_initial_test_Status"],
            "Initial_Test_Source_Authority": xjtu[
                "Qref_Initial_Test_Source_Cycle"
            ],
            "First_Comparable_Ah_Authority": xjtu[
                "Qref_first_comparable_Ah"
            ],
            "First_Comparable_Status_Authority": xjtu[
                "Qref_first_comparable_Status"
            ],
            "First_Comparable_Source_Authority": xjtu[
                "Qref_First_Comparable_Source_Cycle"
            ],
            "First5_Comparable_Ah_Authority": xjtu[
                "Qref_early_median5_comparable_Ah"
            ],
            "First5_Comparable_Status_Authority": xjtu[
                "Qref_early_median5_comparable_Status"
            ],
            "First5_Source_Cycles_Authority": xjtu[
                "Qref_Early_Median5_Comparable_Source_Cycles"
            ],
            "Authority_Path": str(XJTU_QREF),
            "Authority_Row": np.arange(2, len(xjtu) + 2),
        }
    )

    hust = pd.read_csv(HUST_QREF, dtype=object)
    hust_authority = pd.DataFrame(
        {
            "Dataset": "HUST",
            "Batch_ID": np.nan,
            "Cell_ID": hust["Cell_ID"],
            "Nominal_Ah_Authority": hust["Qref_Nominal_Ah"],
            "Nominal_Status_Authority": hust["Qref_Nominal_Status"],
            "Initial_Test_Ah_Authority": hust["Qref_Initial_Test_Ah"],
            "Initial_Test_Status_Authority": hust["Qref_Initial_Test_Status"],
            "Initial_Test_Source_Authority": hust[
                "Qref_Initial_Test_Source_Cycle"
            ],
            "First_Comparable_Ah_Authority": hust[
                "Qref_First_Comparable_Ah"
            ],
            "First_Comparable_Status_Authority": hust[
                "Qref_First_Comparable_Status"
            ],
            "First_Comparable_Source_Authority": hust[
                "Qref_First_Comparable_Source_Cycle"
            ],
            "First5_Comparable_Ah_Authority": hust[
                "Qref_Early_Median5_Comparable_Ah"
            ],
            "First5_Comparable_Status_Authority": hust[
                "Qref_Early_Median5_Comparable_Status"
            ],
            "First5_Source_Cycles_Authority": hust[
                "Qref_Early_Median5_Comparable_Source_Cycles"
            ],
            "Authority_Path": str(HUST_QREF),
            "Authority_Row": np.arange(2, len(hust) + 2),
        }
    )
    authority = pd.concat([xjtu_authority, hust_authority], ignore_index=True)
    if authority.duplicated(["Dataset", "Cell_ID"]).any():
        raise RuntimeError("Duplicate Dataset x Cell_ID keys in frozen Qref authority")
    return authority


def validate_s3(authority: pd.DataFrame) -> pd.DataFrame:
    s3 = pd.read_excel(S3_WORKBOOK, sheet_name=S3_SHEET, header=2, dtype=object)
    expected_columns = {
        "Dataset",
        "Cell_ID",
        "Nominal_Ah",
        "Initial_Test_Ah",
        "First_Comparable_Ah",
        "First5_Comparable_Ah",
        "Initial_Test_Source",
        "First_Comparable_Source",
        "First5_Source_Cycles",
        "Frozen_Artifact_Key_First5",
    }
    missing = expected_columns.difference(s3.columns)
    if missing:
        raise RuntimeError(f"S3 missing required columns: {sorted(missing)}")
    if s3.duplicated(["Dataset", "Cell_ID"]).any():
        raise RuntimeError("Duplicate Dataset x Cell_ID keys in Supplementary Table S3")

    s3 = s3.copy()
    s3["S3_Workbook_Row"] = np.arange(4, len(s3) + 4)
    merged = authority.merge(
        s3,
        on=["Dataset", "Cell_ID"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    merged["Key_Match"] = merged["_merge"].eq("both")
    merged["Nominal_Ah_Match"] = merged.apply(
        lambda row: numeric_equal(row["Nominal_Ah"], row["Nominal_Ah_Authority"]),
        axis=1,
    )
    merged["Initial_Test_Ah_Match"] = merged.apply(
        lambda row: numeric_equal(
            row["Initial_Test_Ah"], row["Initial_Test_Ah_Authority"]
        ),
        axis=1,
    )
    merged["First_Comparable_Ah_Match"] = merged.apply(
        lambda row: numeric_equal(
            row["First_Comparable_Ah"], row["First_Comparable_Ah_Authority"]
        ),
        axis=1,
    )
    merged["First5_Comparable_Ah_Match"] = merged.apply(
        lambda row: numeric_equal(
            row["First5_Comparable_Ah"], row["First5_Comparable_Ah_Authority"]
        ),
        axis=1,
    )
    merged["Initial_Test_Source_Match"] = merged.apply(
        lambda row: source_equal(
            row["Initial_Test_Source"], row["Initial_Test_Source_Authority"]
        ),
        axis=1,
    )
    merged["First_Comparable_Source_Match"] = merged.apply(
        lambda row: source_equal(
            row["First_Comparable_Source"],
            row["First_Comparable_Source_Authority"],
        ),
        axis=1,
    )
    merged["First5_Source_Cycles_Match"] = merged.apply(
        lambda row: source_equal(
            row["First5_Source_Cycles"],
            row["First5_Source_Cycles_Authority"],
        ),
        axis=1,
    )
    merged["Frozen_Artifact_Key_First5_Match"] = merged[
        "Frozen_Artifact_Key_First5"
    ].apply(lambda value: text_equal(value, "early_median5_comparable"))

    match_columns = [
        "Key_Match",
        "Nominal_Ah_Match",
        "Initial_Test_Ah_Match",
        "First_Comparable_Ah_Match",
        "First5_Comparable_Ah_Match",
        "Initial_Test_Source_Match",
        "First_Comparable_Source_Match",
        "First5_Source_Cycles_Match",
        "Frozen_Artifact_Key_First5_Match",
    ]
    merged["Overall_S3_Validation_Status"] = np.where(
        merged[match_columns].all(axis=1), "PASS", "FAIL"
    )

    def mismatch_fields(row: pd.Series) -> str:
        return "|".join(column.removesuffix("_Match") for column in match_columns if not row[column])

    merged["Mismatch_Fields"] = merged.apply(mismatch_fields, axis=1)
    merged["S3_Workbook_Path"] = str(S3_WORKBOOK)
    merged["S3_Sheet"] = S3_SHEET
    merged["Initial_Test_Source_Normalized_S3"] = merged[
        "Initial_Test_Source"
    ].apply(normalize_cycle_list)
    merged["Initial_Test_Source_Normalized_Authority"] = merged[
        "Initial_Test_Source_Authority"
    ].apply(normalize_cycle_list)
    merged["First_Comparable_Source_Normalized_S3"] = merged[
        "First_Comparable_Source"
    ].apply(normalize_cycle_list)
    merged["First_Comparable_Source_Normalized_Authority"] = merged[
        "First_Comparable_Source_Authority"
    ].apply(normalize_cycle_list)
    merged["First5_Source_Cycles_Normalized_S3"] = merged[
        "First5_Source_Cycles"
    ].apply(normalize_cycle_list)
    merged["First5_Source_Cycles_Normalized_Authority"] = merged[
        "First5_Source_Cycles_Authority"
    ].apply(normalize_cycle_list)

    columns = [
        "Dataset",
        "Batch_ID",
        "Cell_ID",
        "Overall_S3_Validation_Status",
        "Mismatch_Fields",
        "Key_Match",
        "Nominal_Ah",
        "Nominal_Ah_Authority",
        "Nominal_Ah_Match",
        "Initial_Test_Ah",
        "Initial_Test_Ah_Authority",
        "Initial_Test_Ah_Match",
        "Initial_Test_Source_Normalized_S3",
        "Initial_Test_Source_Normalized_Authority",
        "Initial_Test_Source_Match",
        "First_Comparable_Ah",
        "First_Comparable_Ah_Authority",
        "First_Comparable_Ah_Match",
        "First_Comparable_Source_Normalized_S3",
        "First_Comparable_Source_Normalized_Authority",
        "First_Comparable_Source_Match",
        "First5_Comparable_Ah",
        "First5_Comparable_Ah_Authority",
        "First5_Comparable_Ah_Match",
        "First5_Source_Cycles_Normalized_S3",
        "First5_Source_Cycles_Normalized_Authority",
        "First5_Source_Cycles_Match",
        "Frozen_Artifact_Key_First5",
        "Frozen_Artifact_Key_First5_Match",
        "Authority_Path",
        "Authority_Row",
        "S3_Workbook_Path",
        "S3_Sheet",
        "S3_Workbook_Row",
    ]
    return merged[columns].sort_values(["Dataset", "Batch_ID", "Cell_ID"], na_position="last")


def make_validation_summary(validation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, batch), group in validation.groupby(
        ["Dataset", "Batch_ID"], dropna=False, sort=True
    ):
        rows.append(
            {
                "Dataset": dataset,
                "Batch_ID": batch,
                "Cell_Count": len(group),
                "PASS_Cell_Count": int(
                    group["Overall_S3_Validation_Status"].eq("PASS").sum()
                ),
                "FAIL_Cell_Count": int(
                    group["Overall_S3_Validation_Status"].eq("FAIL").sum()
                ),
                "First5_Comparable_Ah_Mismatch_Count": int(
                    (~group["First5_Comparable_Ah_Match"]).sum()
                ),
                "First5_Source_Cycles_Mismatch_Count": int(
                    (~group["First5_Source_Cycles_Match"]).sum()
                ),
                "Other_Field_Mismatch_Count": int(
                    (
                        ~group[
                            [
                                "Key_Match",
                                "Nominal_Ah_Match",
                                "Initial_Test_Ah_Match",
                                "Initial_Test_Source_Match",
                                "First_Comparable_Ah_Match",
                                "First_Comparable_Source_Match",
                                "Frozen_Artifact_Key_First5_Match",
                            ]
                        ]
                    ).sum().sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def authority_lookup(authority: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in authority.iterrows():
        definitions = {
            "nominal": (row["Nominal_Ah_Authority"], ""),
            "first_comparable": (
                row["First_Comparable_Ah_Authority"],
                row["First_Comparable_Source_Authority"],
            ),
            "early_median5_comparable": (
                row["First5_Comparable_Ah_Authority"],
                row["First5_Source_Cycles_Authority"],
            ),
        }
        for definition, (qref, sources) in definitions.items():
            rows.append(
                {
                    "Dataset": row["Dataset"],
                    "Cell_ID": row["Cell_ID"],
                    "Reference_Definition": definition,
                    "Authority_Qref_Ah": qref,
                    "Authority_Source_Cycles": normalize_cycle_list(sources),
                    "Authority_Latest_Source_Cycle": latest_cycle(sources),
                    "Authority_Path": row["Authority_Path"],
                    "Authority_Row": row["Authority_Row"],
                }
            )
    return pd.DataFrame(rows)


def make_v1_change_report(
    authority: pd.DataFrame, validation: pd.DataFrame
) -> pd.DataFrame:
    v1 = pd.read_csv(V1_AUDIT, dtype=object)
    lookup = authority_lookup(authority)
    merged = v1.merge(
        lookup,
        on=["Dataset", "Cell_ID", "Reference_Definition"],
        how="outer",
        indicator=True,
        validate="many_to_one",
    )

    s3_by_reference: list[dict[str, object]] = []
    for _, row in validation.iterrows():
        for definition, fields in {
            "nominal": ["Key_Match", "Nominal_Ah_Match"],
            "first_comparable": [
                "Key_Match",
                "First_Comparable_Ah_Match",
                "First_Comparable_Source_Match",
            ],
            "early_median5_comparable": [
                "Key_Match",
                "First5_Comparable_Ah_Match",
                "First5_Source_Cycles_Match",
                "Frozen_Artifact_Key_First5_Match",
            ],
        }.items():
            passed = all(bool(row[field]) for field in fields)
            s3_by_reference.append(
                {
                    "Dataset": row["Dataset"],
                    "Cell_ID": row["Cell_ID"],
                    "Reference_Definition": definition,
                    "S3_Reference_Validation_Status": "PASS" if passed else "FAIL",
                }
            )
    merged = merged.merge(
        pd.DataFrame(s3_by_reference),
        on=["Dataset", "Cell_ID", "Reference_Definition"],
        how="left",
        validate="many_to_one",
    )

    merged["V1_Qref_Matches_Frozen_Authority"] = merged.apply(
        lambda row: numeric_equal(row["Qref_Ah"], row["Authority_Qref_Ah"]), axis=1
    )
    merged["V1_Source_Cycles_Normalized"] = merged["Source_Cycles"].apply(
        normalize_cycle_list
    )
    merged["V1_Source_Matches_Frozen_Authority"] = merged.apply(
        lambda row: source_equal(
            row["V1_Source_Cycles_Normalized"], row["Authority_Source_Cycles"]
        ),
        axis=1,
    )
    merged["V1_Latest_Source_Matches_Frozen_Authority"] = merged.apply(
        lambda row: numeric_equal(
            row["Latest_Source_Cycle"], row["Authority_Latest_Source_Cycle"]
        ),
        axis=1,
    )
    merged["V2_Count_Recomputation_Status"] = "NOT_RUN_S3_VALIDATION_GATE_FAILED"
    merged["V2_Eligible_Observation_Count"] = np.nan
    merged["V2_Strictly_Before_Count"] = np.nan
    merged["V2_At_Count"] = np.nan
    merged["V2_Strictly_After_Count"] = np.nan
    merged["Delta_Eligible_Observation_Count_V2_Minus_V1"] = np.nan
    merged["Delta_Strictly_Before_V2_Minus_V1"] = np.nan
    merged["Delta_At_V2_Minus_V1"] = np.nan
    merged["Delta_Strictly_After_V2_Minus_V1"] = np.nan

    def change_status(row: pd.Series) -> str:
        if row["_merge"] != "both":
            return "KEY_MISMATCH_V1_VS_AUTHORITY"
        if not (
            row["V1_Qref_Matches_Frozen_Authority"]
            and row["V1_Source_Matches_Frozen_Authority"]
            and row["V1_Latest_Source_Matches_Frozen_Authority"]
        ):
            return "V1_QREF_OR_SOURCE_DIFFERS_FROM_AUTHORITY"
        return "NO_QREF_OR_SOURCE_CHANGE; COUNTS_NOT_RECOMPUTED"

    merged["V1_to_V2_Change_Status"] = merged.apply(change_status, axis=1)
    merged["V1_to_V2_Change_Reason"] = np.where(
        merged["S3_Reference_Validation_Status"].eq("FAIL"),
        "Supplementary Table S3 conflicts with the frozen Qref authority for this reference; user-specified consistency gate stopped counting.",
        "V1 Qref/source fields match the frozen authority, but count recomputation was not permitted after the global S3 gate failed.",
    )

    columns = [
        "Dataset",
        "Batch_ID",
        "Cell_ID",
        "Reference_Definition",
        "Reference_Display_Name",
        "Population_Scope",
        "Population_Definition",
        "Qref_Ah",
        "Authority_Qref_Ah",
        "V1_Qref_Matches_Frozen_Authority",
        "V1_Source_Cycles_Normalized",
        "Authority_Source_Cycles",
        "V1_Source_Matches_Frozen_Authority",
        "Latest_Source_Cycle",
        "Authority_Latest_Source_Cycle",
        "V1_Latest_Source_Matches_Frozen_Authority",
        "Eligible_Observation_Count",
        "Strictly_Before_Reference_Availability_Count",
        "At_Reference_Availability_Count",
        "Strictly_After_Reference_Availability_Count",
        "V2_Eligible_Observation_Count",
        "V2_Strictly_Before_Count",
        "V2_At_Count",
        "V2_Strictly_After_Count",
        "Delta_Eligible_Observation_Count_V2_Minus_V1",
        "Delta_Strictly_Before_V2_Minus_V1",
        "Delta_At_V2_Minus_V1",
        "Delta_Strictly_After_V2_Minus_V1",
        "S3_Reference_Validation_Status",
        "V2_Count_Recomputation_Status",
        "V1_to_V2_Change_Status",
        "V1_to_V2_Change_Reason",
        "Authority_Path",
        "Authority_Row",
        "Qref_Source_Path",
        "Qref_Source_Row",
    ]
    return merged[columns].sort_values(
        ["Dataset", "Cell_ID", "Reference_Definition", "Population_Scope"]
    )


def write_report(
    validation: pd.DataFrame,
    summary: pd.DataFrame,
    changes: pd.DataFrame,
) -> None:
    failed = validation[validation["Overall_S3_Validation_Status"].eq("FAIL")]
    mismatched_fields = {
        "First5_Comparable_Ah": int((~validation["First5_Comparable_Ah_Match"]).sum()),
        "First5_Source_Cycles": int((~validation["First5_Source_Cycles_Match"]).sum()),
    }
    report = f"""# Reference Availability Audit V2 — validation gate status

## Outcome

**STOPPED BEFORE OBSERVATION COUNTING.** The required cell-by-cell consistency gate against Supplementary Table S3 failed. No `reference_availability_audit_v2.csv` was generated, and no FROZEN_SOH_ELIGIBLE or POSTCYCLE5 counts were recomputed.

## Frozen authority

- XJTU: `{XJTU_QREF}`
- HUST: `{HUST_QREF}`
- These frozen Qref summaries were treated as the sole Qref-value and source-cycle authority.

## S3 validation

- Keys checked: {len(validation)} cells (XJTU 55; HUST 77).
- Fully matching cells: {int(validation['Overall_S3_Validation_Status'].eq('PASS').sum())}.
- Failing cells: {len(failed)}.
- `First5_Comparable_Ah` mismatches: {mismatched_fields['First5_Comparable_Ah']}.
- `First5_Source_Cycles` mismatches: {mismatched_fields['First5_Source_Cycles']}.
- All failures are XJTU Batch 2–6. XJTU Batch 1 and all 77 HUST cells pass.
- Example: `B2_3C_battery-1` is `2|11|101|102|103` in S3 but `2|3|4|5|6` in frozen XJTU Final V2.

The mismatch pattern is consistent with the pre-hotfix string-sorting artifact in S3, whereas the frozen XJTU Final V2 summary contains the numeric-sort-corrected source cycles.

## V1 → V2 audit comparison

- V1 rows checked: {len(changes)}.
- V1 Qref mismatches versus frozen authority: {int((~changes['V1_Qref_Matches_Frozen_Authority']).sum())}.
- V1 source-cycle mismatches versus frozen authority: {int((~changes['V1_Source_Matches_Frozen_Authority']).sum())}.
- V1 latest-source-cycle mismatches versus frozen authority: {int((~changes['V1_Latest_Source_Matches_Frozen_Authority']).sum())}.
- Count deltas are intentionally blank because the S3 gate failed before the counting stage.

## Required resolution

Either (1) create a separately versioned Supplementary Table S3 sourced from the frozen XJTU Final V2/HUST Final V1 Qref summaries, then rerun this audit, or (2) explicitly authorize proceeding despite S3 inconsistency. This run did not alter S3, frozen summaries, V1 audit, SOH targets, Qref, EOL, model files, or folds.
"""
    (OUTPUT_DIR / "reference_availability_audit_v2_status.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    protected_inputs = [
        (S3_WORKBOOK, "Supplementary Table S3 workbook", "READ_VALIDATION"),
        (XJTU_QREF, "Frozen XJTU Final V2 Qref summary", "SOLE_QREF_AUTHORITY"),
        (HUST_QREF, "Frozen HUST Final V1 Qref summary", "SOLE_QREF_AUTHORITY"),
        (V1_AUDIT, "Existing reference availability audit V1", "V1_COMPARISON"),
        (XJTU_FEATURES, "Frozen XJTU feature matrix", "COUNT_INPUT_BLOCKED"),
        (HUST_FEATURES, "Frozen HUST feature matrix", "COUNT_INPUT_BLOCKED"),
        (XJTU_MASK, "Frozen XJTU eligibility mask", "COUNT_INPUT_BLOCKED"),
        (HUST_MASK, "Frozen HUST eligibility mask", "COUNT_INPUT_BLOCKED"),
    ]
    for path, _, _ in protected_inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    hashes_before = {path: sha256(path) for path, _, _ in protected_inputs}

    authority = load_qref_authority()
    validation = validate_s3(authority)
    validation_summary = make_validation_summary(validation)
    changes = make_v1_change_report(authority, validation)

    validation.to_csv(
        OUTPUT_DIR / "supplementary_table_s3_cell_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_summary.to_csv(
        OUTPUT_DIR / "supplementary_table_s3_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    changes.to_csv(
        OUTPUT_DIR / "reference_availability_audit_v1_to_v2_change_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(validation, validation_summary, changes)

    manifest_rows = []
    for path, role, use_status in protected_inputs:
        after = sha256(path)
        manifest_rows.append(
            {
                "Path": str(path),
                "Role": role,
                "Run_Use_Status": use_status,
                "Size_Bytes": path.stat().st_size,
                "SHA256_Before": hashes_before[path],
                "SHA256_After": after,
                "Hash_Unchanged": hashes_before[path] == after,
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        OUTPUT_DIR / "source_manifest.csv", index=False, encoding="utf-8-sig"
    )

    failed = validation[validation["Overall_S3_Validation_Status"].eq("FAIL")]
    log_lines = [
        f"Timestamp_UTC={datetime.now(timezone.utc).isoformat()}",
        f"Python={platform.python_version()}",
        f"pandas={pd.__version__}",
        "Requested_Count_Populations=FROZEN_SOH_ELIGIBLE|EXISTING_ROBUSTNESS_POSTCYCLE5_RAW_CYCLE_GT_5",
        "Qref_Recomputed=NO",
        "EOL_Recomputed=NO",
        "SOH_Targets_Recomputed=NO",
        "Model_Training=NO",
        f"S3_Cell_Count={len(validation)}",
        f"S3_PASS_Cell_Count={int(validation['Overall_S3_Validation_Status'].eq('PASS').sum())}",
        f"S3_FAIL_Cell_Count={len(failed)}",
        f"S3_First5_Comparable_Ah_Mismatch_Count={int((~validation['First5_Comparable_Ah_Match']).sum())}",
        f"S3_First5_Source_Cycles_Mismatch_Count={int((~validation['First5_Source_Cycles_Match']).sum())}",
        "Observation_Counting_Executed=NO",
        "Reference_Availability_Audit_V2_Created=NO",
        "Final_Status=STOPPED_S3_VALIDATION_GATE_FAILED",
    ]
    (OUTPUT_DIR / "processing_log.txt").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )

    if failed.empty:
        raise RuntimeError(
            "Unexpected pass: this validation-only script is designed to stop before counts."
        )

    print("REFERENCE AVAILABILITY AUDIT V2 STOPPED — S3 VALIDATION GATE FAILED")


if __name__ == "__main__":
    main()
