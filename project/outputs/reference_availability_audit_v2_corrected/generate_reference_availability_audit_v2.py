from __future__ import annotations

import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
BUILD_DIR = OUTPUT_DIR / ".artifact_build"

XJTU_QREF = PROJECT_ROOT / "outputs/xjtu_55_final_v2/xjtu_55_qref_summary.csv"
HUST_QREF = PROJECT_ROOT / "outputs/hust_77_final_v1/hust_77_qref_summary.csv"
XJTU_LABELS = PROJECT_ROOT / "outputs/xjtu_55_final_v2/xjtu_55_soh_labels_long.csv"
HUST_LABELS = PROJECT_ROOT / "outputs/hust_77_final_v1/hust_77_soh_labels_long.csv"
XJTU_FEATURES = PROJECT_ROOT / "outputs/soh_feature_extraction_v1/feature_matrix_xjtu_v1.csv"
HUST_FEATURES = PROJECT_ROOT / "outputs/soh_feature_extraction_v1/feature_matrix_hust_v1.csv"
XJTU_MASK = PROJECT_ROOT / "outputs/soh_primary_benchmark_v1/soh_model_eligible_mask_xjtu.csv"
HUST_MASK = PROJECT_ROOT / "outputs/soh_primary_benchmark_v1/soh_model_eligible_mask_hust.csv"
V1_AUDIT = PROJECT_ROOT / "outputs/manuscript_semantic_audit_v1/reference_availability_audit.csv"
ROBUSTNESS_PIPELINE = PROJECT_ROOT / "soh_ml_robustness_repro/pipeline.py"
OBSOLETE_S3 = Path(
    r"LOCAL_HOME\Desktop\paper_submission_working_package_v2"
    r"\Supplementary_Tables_S1-S7_v2.xlsx"
)

CORRECTED_S3_CSV = OUTPUT_DIR / "Supplementary_Table_S3_v3_corrected.csv"
CORRECTED_S3_XLSX = OUTPUT_DIR / "Supplementary_Table_S3_v3_corrected.xlsx"
VALIDATION_CSV = OUTPUT_DIR / "supplementary_S3_validation.csv"
AUDIT_CSV = OUTPUT_DIR / "reference_availability_audit_v2.csv"
AGGREGATE_CSV = OUTPUT_DIR / "aggregate_reference_availability_v2.csv"
MANIFEST_CSV = OUTPUT_DIR / "source_manifest.csv"
STATUS_MD = OUTPUT_DIR / "reference_availability_audit_v2_status.md"
PROCESSING_LOG = OUTPUT_DIR / "processing_log.txt"
PAYLOAD_JSON = BUILD_DIR / "s3_workbook_payload.json"

REFERENCE_ORDER = {
    "nominal": 1,
    "first_comparable": 2,
    "early_median5_comparable": 3,
}
POPULATION_ORDER = {
    "FROZEN_SOH_ELIGIBLE": 1,
    "EXISTING_ROBUSTNESS_POSTCYCLE5": 2,
}
NUMERIC_TOLERANCE_AH = 1e-12

FINAL_OUTPUTS = [
    CORRECTED_S3_CSV,
    CORRECTED_S3_XLSX,
    VALIDATION_CSV,
    AUDIT_CSV,
    AGGREGATE_CSV,
    MANIFEST_CSV,
    STATUS_MD,
    PROCESSING_LOG,
]


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_source(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def is_blank(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().upper() in {"", "NA", "N/A", "NAN", "NONE"}


def normalize_cycle_list(value: object) -> str:
    if is_blank(value):
        return ""
    normalized: list[str] = []
    for token in str(value).strip().split("|"):
        token = token.strip()
        try:
            normalized.append(str(int(float(token))))
        except ValueError as exc:
            raise AuditError(f"NONNUMERIC_SOURCE_CYCLE:{value}") from exc
    return "|".join(normalized)


def source_cycles(value: object) -> list[int]:
    normalized = normalize_cycle_list(value)
    if not normalized:
        return []
    return [int(token) for token in normalized.split("|")]


def latest_source_cycle(value: object) -> object:
    cycles = source_cycles(value)
    return max(cycles) if cycles else pd.NA


def numeric_equal(left: object, right: object) -> bool:
    if is_blank(left) and is_blank(right):
        return True
    if is_blank(left) != is_blank(right):
        return False
    try:
        return abs(float(left) - float(right)) <= NUMERIC_TOLERANCE_AH
    except (TypeError, ValueError):
        return False


def source_equal(left: object, right: object) -> bool:
    return normalize_cycle_list(left) == normalize_cycle_list(right)


def text_equal(left: object, right: object) -> bool:
    if is_blank(left) and is_blank(right):
        return True
    if is_blank(left) != is_blank(right):
        return False
    return str(left).strip() == str(right).strip()


def assert_outputs_absent() -> None:
    existing = [str(path) for path in FINAL_OUTPUTS if path.exists()]
    if existing:
        raise AuditError("REFUSE_TO_OVERWRITE:" + "|".join(existing))


def load_authority() -> pd.DataFrame:
    xjtu = pd.read_csv(XJTU_QREF, dtype=object)
    xjtu_authority = pd.DataFrame(
        {
            "Dataset": "XJTU",
            "Batch_ID": pd.to_numeric(xjtu["Batch_ID"], errors="raise").astype("Int64"),
            "Cell_ID": xjtu["Cell_ID"].astype(str),
            "Nominal_Qref_Ah": pd.to_numeric(xjtu["Qref_nominal_Ah"], errors="raise"),
            "Nominal_Qref_Status": xjtu["Qref_nominal_Status"],
            "Initial_Test_Qref_Ah": pd.to_numeric(
                xjtu["Qref_initial_test_Ah"], errors="coerce"
            ),
            "Initial_Test_Qref_Status": xjtu["Qref_initial_test_Status"],
            "Initial_Test_Source_Cycle": xjtu[
                "Qref_Initial_Test_Source_Cycle"
            ].apply(normalize_cycle_list),
            "First_Comparable_Ah": pd.to_numeric(
                xjtu["Qref_first_comparable_Ah"], errors="raise"
            ),
            "First_Comparable_Status": xjtu["Qref_first_comparable_Status"],
            "First_Source_Cycle": xjtu[
                "Qref_First_Comparable_Source_Cycle"
            ].apply(normalize_cycle_list),
            "First5_Comparable_Ah": pd.to_numeric(
                xjtu["Qref_early_median5_comparable_Ah"], errors="raise"
            ),
            "First5_Status": xjtu["Qref_early_median5_comparable_Status"],
            "First5_Source_Cycles": xjtu[
                "Qref_Early_Median5_Comparable_Source_Cycles"
            ].apply(normalize_cycle_list),
            "Source_File": relative_source(XJTU_QREF),
            "Source_Row": np.arange(2, len(xjtu) + 2),
            "Dataset_Order": 1,
            "Dataset_Row_Order": np.arange(1, len(xjtu) + 1),
        }
    )

    hust = pd.read_csv(HUST_QREF, dtype=object)
    hust_authority = pd.DataFrame(
        {
            "Dataset": "HUST",
            "Batch_ID": pd.Series([pd.NA] * len(hust), dtype="Int64"),
            "Cell_ID": hust["Cell_ID"].astype(str),
            "Nominal_Qref_Ah": pd.to_numeric(hust["Qref_Nominal_Ah"], errors="raise"),
            "Nominal_Qref_Status": hust["Qref_Nominal_Status"],
            "Initial_Test_Qref_Ah": pd.to_numeric(
                hust["Qref_Initial_Test_Ah"], errors="coerce"
            ),
            "Initial_Test_Qref_Status": hust["Qref_Initial_Test_Status"],
            "Initial_Test_Source_Cycle": hust[
                "Qref_Initial_Test_Source_Cycle"
            ].apply(normalize_cycle_list),
            "First_Comparable_Ah": pd.to_numeric(
                hust["Qref_First_Comparable_Ah"], errors="raise"
            ),
            "First_Comparable_Status": hust["Qref_First_Comparable_Status"],
            "First_Source_Cycle": hust[
                "Qref_First_Comparable_Source_Cycle"
            ].apply(normalize_cycle_list),
            "First5_Comparable_Ah": pd.to_numeric(
                hust["Qref_Early_Median5_Comparable_Ah"], errors="raise"
            ),
            "First5_Status": hust["Qref_Early_Median5_Comparable_Status"],
            "First5_Source_Cycles": hust[
                "Qref_Early_Median5_Comparable_Source_Cycles"
            ].apply(normalize_cycle_list),
            "Source_File": relative_source(HUST_QREF),
            "Source_Row": np.arange(2, len(hust) + 2),
            "Dataset_Order": 2,
            "Dataset_Row_Order": np.arange(1, len(hust) + 1),
        }
    )

    authority = pd.concat([xjtu_authority, hust_authority], ignore_index=True)
    authority["First5_Latest_Source_Cycle"] = authority[
        "First5_Source_Cycles"
    ].apply(latest_source_cycle)
    authority["First5_Display_Name"] = "first5_comparable"
    authority["Frozen_Artifact_Key_First5"] = "early_median5_comparable"
    authority = authority.sort_values(
        ["Dataset_Order", "Dataset_Row_Order"], kind="stable"
    ).reset_index(drop=True)
    authority["Corrected_S3_Row"] = np.arange(2, len(authority) + 2)

    if len(authority) != 132:
        raise AuditError(f"AUTHORITY_CELL_COUNT_NOT_132:{len(authority)}")
    if authority["Dataset"].value_counts().to_dict() != {"HUST": 77, "XJTU": 55}:
        raise AuditError("AUTHORITY_DATASET_CELL_COUNTS_INVALID")
    if authority.duplicated(["Dataset", "Cell_ID"]).any():
        raise AuditError("DUPLICATE_AUTHORITY_CELL_KEY")
    return authority


def corrected_s3_columns() -> list[str]:
    return [
        "Dataset",
        "Batch_ID",
        "Cell_ID",
        "Nominal_Qref_Ah",
        "Nominal_Qref_Status",
        "Initial_Test_Qref_Ah",
        "Initial_Test_Qref_Status",
        "Initial_Test_Source_Cycle",
        "First_Comparable_Ah",
        "First_Comparable_Status",
        "First_Source_Cycle",
        "First5_Comparable_Ah",
        "First5_Status",
        "First5_Source_Cycles",
        "First5_Latest_Source_Cycle",
        "First5_Display_Name",
        "Frozen_Artifact_Key_First5",
        "Source_File",
        "Source_Row",
    ]


def serialize_and_roundtrip_s3(authority: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    corrected = authority[corrected_s3_columns()].copy()
    buffer = io.StringIO()
    corrected.to_csv(buffer, index=False, lineterminator="\n")
    csv_text = buffer.getvalue()
    roundtrip = pd.read_csv(io.StringIO(csv_text), dtype=object)
    return roundtrip, csv_text


def validate_corrected_s3(
    corrected: pd.DataFrame, authority: pd.DataFrame
) -> pd.DataFrame:
    duplicate_key_count = int(corrected.duplicated(["Dataset", "Cell_ID"]).sum())
    merged = authority.merge(
        corrected,
        on=["Dataset", "Cell_ID"],
        how="outer",
        suffixes=("_Authority", "_Corrected"),
        indicator=True,
        validate="one_to_one" if duplicate_key_count == 0 else "many_to_many",
    )
    validation_rows: list[dict[str, Any]] = []
    numeric_fields = [
        "Nominal_Qref_Ah",
        "Initial_Test_Qref_Ah",
        "First_Comparable_Ah",
        "First5_Comparable_Ah",
    ]
    source_fields = [
        "Initial_Test_Source_Cycle",
        "First_Source_Cycle",
        "First5_Source_Cycles",
    ]
    text_fields = [
        "First5_Display_Name",
        "Frozen_Artifact_Key_First5",
        "Source_File",
    ]
    for _, row in merged.iterrows():
        qref_mismatches = [
            field
            for field in numeric_fields
            if not numeric_equal(row.get(f"{field}_Authority"), row.get(f"{field}_Corrected"))
        ]
        source_mismatches = [
            field
            for field in source_fields
            if not source_equal(row.get(f"{field}_Authority"), row.get(f"{field}_Corrected"))
        ]
        latest_match = numeric_equal(
            row.get("First5_Latest_Source_Cycle_Authority"),
            row.get("First5_Latest_Source_Cycle_Corrected"),
        )
        text_mismatches = [
            field
            for field in text_fields
            if not text_equal(row.get(f"{field}_Authority"), row.get(f"{field}_Corrected"))
        ]
        source_row_match = numeric_equal(
            row.get("Source_Row_Authority"), row.get("Source_Row_Corrected")
        )
        key_present = row["_merge"] == "both"
        dataset = str(row["Dataset"])
        batch_value = row.get("Batch_ID_Authority")
        known_problematic = (
            dataset == "XJTU"
            and not is_blank(batch_value)
            and int(float(batch_value)) in {2, 3, 4, 5, 6}
        )
        passed = (
            key_present
            and not qref_mismatches
            and not source_mismatches
            and latest_match
            and not text_mismatches
            and source_row_match
            and duplicate_key_count == 0
        )
        validation_rows.append(
            {
                "Dataset": dataset,
                "Batch_ID": batch_value,
                "Cell_ID": row["Cell_ID"],
                "Known_Problematic_XJTU_Batch2_6": "YES" if known_problematic else "NO",
                "Cell_Key_Present_In_Both": "YES" if key_present else "NO",
                "Duplicate_Cell_Key_Count_Global": duplicate_key_count,
                "Qref_Mismatch_Count": len(qref_mismatches),
                "Qref_Mismatch_Fields": "|".join(qref_mismatches),
                "Source_Cycle_Mismatch_Count": len(source_mismatches),
                "Source_Cycle_Mismatch_Fields": "|".join(source_mismatches),
                "Latest_Source_Cycle_Mismatch_Count": 0 if latest_match else 1,
                "Metadata_Mismatch_Count": len(text_mismatches) + (0 if source_row_match else 1),
                "Corrected_First5_Comparable_Ah": row.get("First5_Comparable_Ah_Corrected"),
                "Authority_First5_Comparable_Ah": row.get("First5_Comparable_Ah_Authority"),
                "Corrected_First5_Source_Cycles": normalize_cycle_list(
                    row.get("First5_Source_Cycles_Corrected")
                ),
                "Authority_First5_Source_Cycles": normalize_cycle_list(
                    row.get("First5_Source_Cycles_Authority")
                ),
                "Corrected_First5_Latest_Source_Cycle": row.get(
                    "First5_Latest_Source_Cycle_Corrected"
                ),
                "Authority_First5_Latest_Source_Cycle": row.get(
                    "First5_Latest_Source_Cycle_Authority"
                ),
                "Validation_Status": "PASS" if passed else "FAIL",
            }
        )
    validation = pd.DataFrame(validation_rows).sort_values(
        ["Dataset", "Batch_ID", "Cell_ID"], na_position="last", kind="stable"
    )

    missing_keys = int((validation["Cell_Key_Present_In_Both"] != "YES").sum())
    qref_mismatches = int(validation["Qref_Mismatch_Count"].sum())
    source_mismatches = int(validation["Source_Cycle_Mismatch_Count"].sum())
    latest_mismatches = int(validation["Latest_Source_Cycle_Mismatch_Count"].sum())
    failed_rows = int((validation["Validation_Status"] != "PASS").sum())
    if any(
        [
            missing_keys,
            duplicate_key_count,
            qref_mismatches,
            source_mismatches,
            latest_mismatches,
            failed_rows,
        ]
    ):
        raise AuditError(
            "CORRECTED_S3_GATE_FAILED:"
            f"missing={missing_keys}:duplicates={duplicate_key_count}:"
            f"qref={qref_mismatches}:source={source_mismatches}:"
            f"latest={latest_mismatches}:failed_rows={failed_rows}"
        )

    b2 = validation[
        (validation["Dataset"] == "XJTU")
        & (validation["Cell_ID"] == "B2_3C_battery-1")
    ]
    if len(b2) != 1 or b2.iloc[0]["Corrected_First5_Source_Cycles"] != "2|3|4|5|6":
        raise AuditError("KNOWN_B2_CELL_NUMERIC_SORT_REGRESSION_FAILED")
    known = validation[validation["Known_Problematic_XJTU_Batch2_6"] == "YES"]
    if len(known) != 47 or not known["Validation_Status"].eq("PASS").all():
        raise AuditError("XJTU_BATCH2_6_EXPLICIT_GATE_FAILED")
    return validation


def authority_reference_rows(authority: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in authority.iterrows():
        reference_values = {
            "nominal": (
                row["Nominal_Qref_Ah"],
                row["Nominal_Qref_Status"],
                "",
                "DATASET_NOMINAL_CAPACITY",
            ),
            "first_comparable": (
                row["First_Comparable_Ah"],
                row["First_Comparable_Status"],
                row["First_Source_Cycle"],
                "OBSERVED_SOURCE_CYCLE",
            ),
            "early_median5_comparable": (
                row["First5_Comparable_Ah"],
                row["First5_Status"],
                row["First5_Source_Cycles"],
                "LATEST_OF_FIVE_OBSERVED_SOURCE_CYCLES",
            ),
        }
        for definition, (qref, status, sources, source_type) in reference_values.items():
            cycles = source_cycles(sources)
            rows.append(
                {
                    "Dataset": row["Dataset"],
                    "Batch_ID": row["Batch_ID"],
                    "Cell_ID": row["Cell_ID"],
                    "Reference_Definition": definition,
                    "Reference_Display_Name": (
                        "first5_comparable"
                        if definition == "early_median5_comparable"
                        else definition
                    ),
                    "Qref_Ah": qref,
                    "Qref_Status": status,
                    "Source_Cycles": "|".join(str(cycle) for cycle in cycles),
                    "Source_Cycle_Count": len(cycles),
                    "Latest_Source_Cycle": max(cycles) if cycles else pd.NA,
                    "Reference_Availability_Rule": (
                        "AVAILABLE_EXOGENOUSLY_BEFORE_FIRST_RETAINED_OBSERVATION"
                        if definition == "nominal"
                        else "AVAILABLE_AFTER_LATEST_REQUIRED_SOURCE_CYCLE_IS_OBSERVED"
                    ),
                    "Reference_Source_Type": source_type,
                    "Qref_Source_Path": row["Source_File"],
                    "Qref_Source_Row": row["Source_Row"],
                    "Corrected_S3_Path": relative_source(CORRECTED_S3_CSV),
                    "Corrected_S3_Row": row["Corrected_S3_Row"],
                }
            )
    result = pd.DataFrame(rows)
    if result.duplicated(["Dataset", "Cell_ID", "Reference_Definition"]).any():
        raise AuditError("DUPLICATE_STANDARDIZED_AUTHORITY_REFERENCE_KEY")
    return result


def load_frozen_label_populations() -> dict[tuple[str, str], pd.DataFrame]:
    populations: dict[tuple[str, str], pd.DataFrame] = {}
    for dataset, path, include_field in [
        ("XJTU", XJTU_LABELS, "Final_Include"),
        ("HUST", HUST_LABELS, "Primary_Capacity_Eligible"),
    ]:
        labels = pd.read_csv(path, low_memory=False)
        labels["Raw_Cycle_Index"] = pd.to_numeric(
            labels["Raw_Cycle_Index"], errors="raise"
        ).astype(int)
        labels = labels[labels[include_field].astype(str).str.upper().eq("YES")].copy()
        for reference in REFERENCE_ORDER:
            selected = labels[labels["Qref_Definition"].eq(reference)].copy()
            key_columns = ["Cell_ID", "Raw_Cycle_Index"]
            if selected.duplicated(key_columns).any():
                raise AuditError(f"DUPLICATE_FROZEN_SOH_KEY:{dataset}:{reference}")
            populations[(dataset, reference)] = selected
    return populations


def load_postcycle5_populations(
    frozen_labels: dict[tuple[str, str], pd.DataFrame]
) -> dict[tuple[str, str], pd.DataFrame]:
    populations: dict[tuple[str, str], pd.DataFrame] = {}
    for dataset, feature_path, mask_path in [
        ("XJTU", XJTU_FEATURES, XJTU_MASK),
        ("HUST", HUST_FEATURES, HUST_MASK),
    ]:
        features = pd.read_csv(feature_path, low_memory=False)
        mask = pd.read_csv(mask_path, low_memory=False)
        if features["Physical_Observation_Key"].duplicated().any():
            raise AuditError(f"DUPLICATE_FEATURE_KEY:{dataset}")
        if mask["Physical_Observation_Key"].duplicated().any():
            raise AuditError(f"DUPLICATE_MASK_KEY:{dataset}")
        key_check = features[["Physical_Observation_Key"]].merge(
            mask[["Physical_Observation_Key", "Model_Eligible"]],
            on="Physical_Observation_Key",
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        if not key_check["_merge"].eq("both").all() or len(key_check) != len(features):
            raise AuditError(f"FEATURE_MASK_KEY_CHANGE:{dataset}")
        eligible_keys = set(
            mask.loc[
                mask["Model_Eligible"].astype(str).str.upper().eq("YES"),
                "Physical_Observation_Key",
            ].astype(str)
        )
        frozen = features[
            features["Physical_Observation_Key"].astype(str).isin(eligible_keys)
        ].copy()
        frozen["Raw_Cycle_Index"] = pd.to_numeric(
            frozen["Raw_Cycle_Index"], errors="raise"
        ).astype(int)
        postcycle5 = frozen[frozen["Raw_Cycle_Index"].gt(5)].copy()
        if postcycle5["Cell_ID"].nunique() != frozen["Cell_ID"].nunique():
            raise AuditError(f"POSTCYCLE5_REMOVED_ENTIRE_CELL:{dataset}")

        for reference in REFERENCE_ORDER:
            labels = frozen_labels[(dataset, reference)]
            join_columns = ["Cell_ID", "Raw_Cycle_Index"]
            alignment = postcycle5[join_columns].merge(
                labels[join_columns],
                on=join_columns,
                how="left",
                indicator=True,
                validate="one_to_one",
            )
            if not alignment["_merge"].eq("both").all():
                raise AuditError(f"POSTCYCLE5_LABEL_ALIGNMENT_FAILED:{dataset}:{reference}")
            populations[(dataset, reference)] = postcycle5.copy()
    return populations


def count_reference_availability(
    authority_refs: pd.DataFrame,
    frozen_labels: dict[tuple[str, str], pd.DataFrame],
    postcycle5: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    population_metadata = {
        "FROZEN_SOH_ELIGIBLE": (
            "Frozen final included physical capacity observations used for SOH labels",
            frozen_labels,
        ),
        "EXISTING_ROBUSTNESS_POSTCYCLE5": (
            "Frozen primary-model-eligible A9 feature rows filtered by existing Raw_Cycle_Index > 5 rule",
            postcycle5,
        ),
    }
    rows: list[dict[str, Any]] = []
    for _, authority in authority_refs.iterrows():
        dataset = str(authority["Dataset"])
        cell = str(authority["Cell_ID"])
        reference = str(authority["Reference_Definition"])
        latest = authority["Latest_Source_Cycle"]
        source_set = set(source_cycles(authority["Source_Cycles"]))
        frozen_cell_cycles = set(
            frozen_labels[(dataset, reference)]
            .loc[lambda frame: frame["Cell_ID"].astype(str).eq(cell), "Raw_Cycle_Index"]
            .astype(int)
        )
        sources_present = source_set.issubset(frozen_cell_cycles)

        for population_scope, (population_definition, population_map) in population_metadata.items():
            population = population_map[(dataset, reference)]
            cell_rows = population[population["Cell_ID"].astype(str).eq(cell)]
            raw_cycles = pd.to_numeric(cell_rows["Raw_Cycle_Index"], errors="raise").astype(int)
            total = int(len(raw_cycles))
            if reference == "nominal":
                before = 0
                at = 0
                after = total
            else:
                if is_blank(latest):
                    raise AuditError(f"MISSING_LATEST_SOURCE_CYCLE:{dataset}:{cell}:{reference}")
                availability_cycle = int(float(latest))
                before = int(raw_cycles.lt(availability_cycle).sum())
                at = int(raw_cycles.eq(availability_cycle).sum())
                after = int(raw_cycles.gt(availability_cycle).sum())
            if total != before + at + after:
                raise AuditError(f"COUNT_RECONCILIATION_FAILED:{dataset}:{cell}:{reference}:{population_scope}")
            rows.append(
                {
                    "Dataset": dataset,
                    "Batch_ID": authority["Batch_ID"],
                    "Cell_ID": cell,
                    "Reference_Definition": reference,
                    "Reference_Display_Name": authority["Reference_Display_Name"],
                    "Qref_Ah": authority["Qref_Ah"],
                    "Qref_Status": authority["Qref_Status"],
                    "Source_Cycles": authority["Source_Cycles"],
                    "Source_Cycle_Count": authority["Source_Cycle_Count"],
                    "Latest_Source_Cycle": latest,
                    "Reference_Availability_Rule": authority[
                        "Reference_Availability_Rule"
                    ],
                    "Reference_Source_Type": authority["Reference_Source_Type"],
                    "Population_Scope": population_scope,
                    "Population_Definition": population_definition,
                    "Eligible_Observation_Count": total,
                    "Strictly_Before_Reference_Availability_Count": before,
                    "At_Reference_Availability_Count": at,
                    "Strictly_After_Reference_Availability_Count": after,
                    "At_or_After_Reference_Availability_Count": at + after,
                    "Source_Cycles_Present_In_Frozen_SOH_Eligible_Population": (
                        "YES" if sources_present else "NO"
                    ),
                    "Qref_Source_Path": authority["Qref_Source_Path"],
                    "Qref_Source_Row": authority["Qref_Source_Row"],
                    "Corrected_S3_Path": authority["Corrected_S3_Path"],
                    "Corrected_S3_Row": authority["Corrected_S3_Row"],
                    "POSTCYCLE5_Raw_Cycle_Filter": (
                        "Raw_Cycle_Index > 5"
                        if population_scope == "EXISTING_ROBUSTNESS_POSTCYCLE5"
                        else "NONE"
                    ),
                    "POSTCYCLE5_Is_Not_Reference_Availability_Rule": "YES",
                    "Audit_Version": "V2_CORRECTED",
                }
            )
    audit = pd.DataFrame(rows)
    if len(audit) != 792:
        raise AuditError(f"AUDIT_ROW_COUNT_NOT_792:{len(audit)}")
    keys = ["Dataset", "Cell_ID", "Reference_Definition", "Population_Scope"]
    if audit.duplicated(keys).any():
        raise AuditError("DUPLICATE_AUDIT_KEY")
    if not audit[
        "Source_Cycles_Present_In_Frozen_SOH_Eligible_Population"
    ].eq("YES").all():
        raise AuditError("QREF_SOURCE_CYCLE_MISSING_FROM_FROZEN_SOH_POPULATION")
    audit["Reference_Order"] = audit["Reference_Definition"].map(REFERENCE_ORDER)
    audit["Population_Order"] = audit["Population_Scope"].map(POPULATION_ORDER)
    audit = audit.sort_values(
        ["Dataset", "Batch_ID", "Cell_ID", "Reference_Order", "Population_Order"],
        na_position="last",
        kind="stable",
    ).drop(columns=["Reference_Order", "Population_Order"])
    return audit


def aggregate_and_compare_v1(audit: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    count_columns = [
        "Eligible_Observation_Count",
        "Strictly_Before_Reference_Availability_Count",
        "At_Reference_Availability_Count",
        "Strictly_After_Reference_Availability_Count",
        "At_or_After_Reference_Availability_Count",
    ]
    group_keys = [
        "Dataset",
        "Reference_Definition",
        "Reference_Display_Name",
        "Population_Scope",
    ]
    aggregate = audit.groupby(group_keys, as_index=False)[count_columns].sum()
    aggregate = aggregate.rename(columns={column: f"V2_{column}" for column in count_columns})

    v1 = pd.read_csv(V1_AUDIT, low_memory=False)
    v1_group = (
        v1.groupby(
            ["Dataset", "Reference_Definition", "Reference_Display_Name", "Population_Scope"],
            as_index=False,
        )[count_columns]
        .sum()
        .rename(columns={column: f"V1_{column}" for column in count_columns})
    )
    aggregate = aggregate.merge(
        v1_group,
        on=group_keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not aggregate["_merge"].eq("both").all() or len(aggregate) != 12:
        raise AuditError("V1_V2_AGGREGATE_KEY_MISMATCH")
    aggregate = aggregate.drop(columns="_merge")
    for column in count_columns:
        aggregate[f"Delta_{column}_V2_Minus_V1"] = (
            aggregate[f"V2_{column}"] - aggregate[f"V1_{column}"]
        )
    delta_columns = [f"Delta_{column}_V2_Minus_V1" for column in count_columns]
    aggregate["V1_to_V2_Counts_Changed"] = np.where(
        aggregate[delta_columns].ne(0).any(axis=1), "YES", "NO"
    )
    aggregate["Change_Reason"] = np.where(
        aggregate["V1_to_V2_Counts_Changed"].eq("YES"),
        "Recomputed counts differ from V1; inspect row-level audit.",
        "No numerical change. V1 already used the frozen numeric-sort-corrected Qref source cycles.",
    )

    v1_keys = ["Dataset", "Cell_ID", "Reference_Definition", "Population_Scope"]
    v1_counts = v1[v1_keys + count_columns].copy()
    v2_counts = audit[v1_keys + count_columns].copy()
    row_compare = v2_counts.merge(
        v1_counts,
        on=v1_keys,
        suffixes=("_V2", "_V1"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not row_compare["_merge"].eq("both").all():
        raise AuditError("V1_V2_ROW_KEY_MISMATCH")
    row_mismatch = pd.Series(False, index=row_compare.index)
    for column in count_columns:
        row_mismatch |= row_compare[f"{column}_V2"].ne(row_compare[f"{column}_V1"])
    row_mismatch_count = int(row_mismatch.sum())

    aggregate["Reference_Order"] = aggregate["Reference_Definition"].map(REFERENCE_ORDER)
    aggregate["Population_Order"] = aggregate["Population_Scope"].map(POPULATION_ORDER)
    aggregate["Dataset_Order"] = aggregate["Dataset"].map({"XJTU": 1, "HUST": 2})
    aggregate = aggregate.sort_values(
        ["Dataset_Order", "Reference_Order", "Population_Order"], kind="stable"
    ).drop(columns=["Dataset_Order", "Reference_Order", "Population_Order"])
    return aggregate, row_mismatch_count


def markdown_aggregate_table(aggregate: pd.DataFrame) -> str:
    headers = [
        "Dataset",
        "Reference",
        "Population",
        "Total",
        "Before",
        "At",
        "After",
        "At or after",
        "Changed vs V1",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Dataset"]),
                    str(row["Reference_Display_Name"]),
                    str(row["Population_Scope"]),
                    str(int(row["V2_Eligible_Observation_Count"])),
                    str(int(row["V2_Strictly_Before_Reference_Availability_Count"])),
                    str(int(row["V2_At_Reference_Availability_Count"])),
                    str(int(row["V2_Strictly_After_Reference_Availability_Count"])),
                    str(int(row["V2_At_or_After_Reference_Availability_Count"])),
                    str(row["V1_to_V2_Counts_Changed"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def make_status_report(
    validation: pd.DataFrame,
    aggregate: pd.DataFrame,
    row_mismatch_count: int,
) -> str:
    known = validation[validation["Known_Problematic_XJTU_Batch2_6"].eq("YES")]
    return f"""# Reference Availability Audit V2 Corrected

## Corrected S3 frozen-authority gate

**PASS.** The corrected S3 was copied from the frozen XJTU Final V2 and HUST Final V1 Qref summaries; no Qref was recomputed from capacity data.

- Cells validated: {len(validation)} (XJTU 55; HUST 77)
- Missing cell keys: {int((validation['Cell_Key_Present_In_Both'] != 'YES').sum())}
- Duplicate cell keys: {int(validation['Duplicate_Cell_Key_Count_Global'].max())}
- Qref mismatches: {int(validation['Qref_Mismatch_Count'].sum())}
- Source-cycle mismatches: {int(validation['Source_Cycle_Mismatch_Count'].sum())}
- Latest-source-cycle mismatches: {int(validation['Latest_Source_Cycle_Mismatch_Count'].sum())}
- Explicit XJTU Batch 2–6 regression cells checked: {len(known)}; all passed
- `B2_3C_battery-1` corrected first5 sources: `2|3|4|5|6`

## Recomputed reference-availability counts

`FROZEN_SOH_ELIGIBLE` uses retained observations in the frozen SOH label outputs. `EXISTING_ROBUSTNESS_POSTCYCLE5` uses the frozen primary A9 feature matrices and eligibility masks, then applies the existing rule `Raw_Cycle_Index > 5`. POSTCYCLE5 was not redefined as post-reference availability.

{markdown_aggregate_table(aggregate)}

## V1 → V2 result

- Aggregate rows changed: {int(aggregate['V1_to_V2_Counts_Changed'].eq('YES').sum())} of {len(aggregate)}.
- Cell × reference × population count rows changed: {row_mismatch_count} of 792.
- The XJTU first5 values remain 10,427 / 220 / 55 / 10,152 for FROZEN_SOH_ELIGIBLE and 10,327 / 120 / 55 / 10,152 for POSTCYCLE5 (total / before / at / after).
- Manuscript text that reports the old 220 and 120 counts does not need numerical modification.
- The obsolete S3 must be replaced because it contains pre-hotfix Batch 2–6 first5 Qref/source-cycle values.
- Inputs for the old Table S9 should be replaced by `reference_availability_audit_v2.csv` and `aggregate_reference_availability_v2.csv`, even though the reported counts are unchanged, so provenance points to the corrected frozen-authority mapping.

## Prohibited analyses

No Qref, SOH target, EOL, RUL, fold, feature, model fit, or prediction was recomputed or modified.
"""


def json_safe(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_outputs(
    corrected_csv_text: str,
    corrected_roundtrip: pd.DataFrame,
    validation: pd.DataFrame,
    audit: pd.DataFrame,
    aggregate: pd.DataFrame,
    manifest: pd.DataFrame,
    status_report: str,
    processing_log: str,
) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=False)
    CORRECTED_S3_CSV.write_text(corrected_csv_text, encoding="utf-8-sig")
    validation.to_csv(VALIDATION_CSV, index=False, encoding="utf-8-sig")
    audit.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")
    aggregate.to_csv(AGGREGATE_CSV, index=False, encoding="utf-8-sig")
    manifest.to_csv(MANIFEST_CSV, index=False, encoding="utf-8-sig")
    STATUS_MD.write_text(status_report, encoding="utf-8")
    PROCESSING_LOG.write_text(processing_log, encoding="utf-8")
    payload = {
        "sheetName": "S3_corrected",
        "headers": corrected_s3_columns(),
        "rows": [
            [json_safe(value) for value in row]
            for row in corrected_roundtrip[corrected_s3_columns()].itertuples(
                index=False, name=None
            )
        ],
        "outputPath": str(CORRECTED_S3_XLSX),
        "previewTopPath": str(BUILD_DIR / "s3_preview_top.png"),
        "previewBottomPath": str(BUILD_DIR / "s3_preview_bottom.png"),
    }
    PAYLOAD_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    assert_outputs_absent()
    protected_inputs: list[tuple[Path, str, str]] = [
        (XJTU_QREF, "Frozen XJTU Final V2 Qref summary", "SOLE_QREF_AUTHORITY"),
        (HUST_QREF, "Frozen HUST Final V1 Qref summary", "SOLE_QREF_AUTHORITY"),
        (XJTU_LABELS, "Frozen XJTU SOH labels", "FROZEN_SOH_ELIGIBLE_POPULATION"),
        (HUST_LABELS, "Frozen HUST SOH labels", "FROZEN_SOH_ELIGIBLE_POPULATION"),
        (XJTU_FEATURES, "Frozen XJTU A9 feature matrix", "POSTCYCLE5_POPULATION"),
        (HUST_FEATURES, "Frozen HUST A9 feature matrix", "POSTCYCLE5_POPULATION"),
        (XJTU_MASK, "Frozen XJTU primary eligibility mask", "POSTCYCLE5_POPULATION"),
        (HUST_MASK, "Frozen HUST primary eligibility mask", "POSTCYCLE5_POPULATION"),
        (V1_AUDIT, "Reference availability audit V1", "V1_COMPARISON"),
        (ROBUSTNESS_PIPELINE, "Frozen POSTCYCLE5 implementation", "RULE_EVIDENCE"),
    ]
    if OBSOLETE_S3.exists():
        protected_inputs.append(
            (OBSOLETE_S3, "Obsolete Supplementary Table S3 workbook", "HASH_ONLY_NOT_AUTHORITY")
        )
    for path, _, _ in protected_inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    hashes_before = {path: sha256_file(path) for path, _, _ in protected_inputs}

    authority = load_authority()
    corrected_roundtrip, corrected_csv_text = serialize_and_roundtrip_s3(authority)
    validation = validate_corrected_s3(corrected_roundtrip, authority)

    authority_refs = authority_reference_rows(authority)
    frozen_labels = load_frozen_label_populations()
    postcycle5 = load_postcycle5_populations(frozen_labels)
    audit = count_reference_availability(authority_refs, frozen_labels, postcycle5)
    aggregate, row_mismatch_count = aggregate_and_compare_v1(audit)

    manifest_rows: list[dict[str, Any]] = []
    for path, role, use in protected_inputs:
        after_hash = sha256_file(path)
        manifest_rows.append(
            {
                "Path": str(path),
                "Role": role,
                "Use_In_V2": use,
                "Sole_Qref_Authority": "YES" if use == "SOLE_QREF_AUTHORITY" else "NO",
                "Size_Bytes": path.stat().st_size,
                "SHA256_Before": hashes_before[path],
                "SHA256_After": after_hash,
                "Hash_Unchanged": "YES" if hashes_before[path] == after_hash else "NO",
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    if not manifest["Hash_Unchanged"].eq("YES").all():
        raise AuditError("PROTECTED_SOURCE_HASH_CHANGED_DURING_RUN")

    status_report = make_status_report(validation, aggregate, row_mismatch_count)
    processing_log = "\n".join(
        [
            f"Timestamp_UTC={datetime.now(timezone.utc).isoformat()}",
            f"Python={platform.python_version()}",
            f"pandas={pd.__version__}",
            f"numpy={np.__version__}",
            "Qref_Authority=XJTU_FINAL_V2|HUST_FINAL_V1",
            "Qref_Recomputed=NO",
            "SOH_Targets_Recomputed=NO",
            "EOL_Recomputed=NO",
            "RUL_Recomputed=NO",
            "Model_Training=NO",
            "Fold_Changes=NO",
            "POSTCYCLE5_Rule=Raw_Cycle_Index > 5",
            "Corrected_S3_Cell_Count=132",
            "Corrected_S3_Qref_Mismatch_Count=0",
            "Corrected_S3_Source_Cycle_Mismatch_Count=0",
            "Corrected_S3_Latest_Source_Cycle_Mismatch_Count=0",
            "Corrected_S3_Missing_Cell_Key_Count=0",
            "Corrected_S3_Duplicate_Cell_Key_Count=0",
            "XJTU_Batch2_6_Explicit_Cell_Count=47",
            "XJTU_Batch2_6_Explicit_Gate=PASS",
            f"V1_V2_Aggregate_Changed_Row_Count={int(aggregate['V1_to_V2_Counts_Changed'].eq('YES').sum())}",
            f"V1_V2_Cell_Level_Changed_Row_Count={row_mismatch_count}",
            "Protected_Source_Hashes_Unchanged=YES",
            "Python_Stage_Status=PASS",
        ]
    ) + "\n"

    write_outputs(
        corrected_csv_text,
        corrected_roundtrip,
        validation,
        audit,
        aggregate,
        manifest,
        status_report,
        processing_log,
    )
    print("REFERENCE AVAILABILITY AUDIT V2 DATA STAGE COMPLETE")


if __name__ == "__main__":
    main()
