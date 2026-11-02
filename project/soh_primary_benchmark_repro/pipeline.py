"""Frozen, cell-level SOH Primary ML Benchmark V1."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import (
    ALL_EXTRACTED_FEATURES,
    COMMON_QREFS,
    CSV_OUTPUTS,
    HUST_EXPECTED_PREMASK_ROWS,
    MODEL_NAMES,
    MOST_FOLDS_THRESHOLD,
    N_FOLDS,
    PIPELINE_VERSION,
    PRIMARY_FEATURES,
    PROTOCOL_SHORTCUT_FEATURES,
    QREF_PAIRS,
    RANDOM_SEED,
    SECONDARY_XJTU_QREFS,
    XJTU_EXPECTED_MODEL_ROWS,
)
from .metrics import calculate_metrics, cell_metric_rows, summarize_cell_metrics


PREDICTION_COLUMNS = (
    "Analysis_ID",
    "Dataset",
    "Batch_ID",
    "Cell_ID",
    "Raw_Cycle_Index",
    "Fold_ID",
    "Physical_Observation_Key",
    "Qref_Definition",
    "Qref_Ah",
    "Model",
    "Observed_SOH_Fraction",
    "Predicted_SOH_Fraction",
    "Observed_Capacity_Ah",
    "Predicted_Capacity_Ah",
    "Prediction_Clipped",
    "Prediction_Source",
    "Model_Parameters_SHA256",
)


class PrimaryBenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    workspace: Path
    extraction: Path
    design: Path
    xjtu_final: Path
    hust_final: Path

    @classmethod
    def from_workspace(cls, workspace: Path) -> "Paths":
        outputs = workspace / "outputs"
        return cls(
            workspace=workspace,
            extraction=outputs / "soh_feature_extraction_v1",
            design=outputs / "soh_ml_design_v1",
            xjtu_final=outputs / "xjtu_55_final_v2",
            hust_final=outputs / "hust_77_final_v1",
        )


@dataclass
class FoldTransform:
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    tree_train: np.ndarray
    tree_test: np.ndarray
    ridge_train: np.ndarray
    ridge_test: np.ndarray
    train_cell_count: int
    test_cell_count: int
    train_key_sha256: str
    test_key_sha256: str
    imputer_statistics_sha256: str
    scaler_mean_sha256: str
    scaler_scale_sha256: str


@dataclass
class TrainingAudit:
    fit_count: int = 0
    prediction_count: int = 0
    max_capacity_formula_error: float = 0.0
    train_test_cell_overlap_count: int = 0
    fit_records: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.fit_records is None:
            self.fit_records = []


def _require(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise PrimaryBenchmarkError("MISSING_REQUIRED_INPUTS:" + "|".join(missing))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64)
    normalized = array.copy()
    normalized[np.isnan(normalized)] = np.float64(np.nan)
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.columns.duplicated().any():
        raise PrimaryBenchmarkError(f"DUPLICATE_OUTPUT_COLUMNS:{path.name}")
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


def _numeric_int(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise PrimaryBenchmarkError(f"NONFINITE_INTEGER_COLUMN:{name}")
    if not np.allclose(values, np.round(values), atol=0, rtol=0):
        raise PrimaryBenchmarkError(f"NONINTEGER_COLUMN:{name}")
    return values.astype("int64")


def _input_paths(paths: Paths) -> dict[str, Path]:
    return {
        "feature_matrix_xjtu": paths.extraction / "feature_matrix_xjtu_v1.csv",
        "feature_matrix_hust": paths.extraction / "feature_matrix_hust_v1.csv",
        "feature_regression": paths.extraction / "feature_extraction_regression_tests.csv",
        "protocol_shortcut": paths.extraction / "feature_protocol_shortcut_audit.csv",
        "xjtu_labels": paths.xjtu_final / "xjtu_55_soh_labels_long.csv",
        "hust_labels": paths.hust_final / "hust_77_soh_labels_long.csv",
        "xjtu_folds": paths.design / "ml_cell_fold_assignments_xjtu.csv",
        "hust_folds": paths.design / "ml_cell_fold_assignments_hust.csv",
        "xjtu_initial_folds": paths.design / "ml_xjtu_initial_test_47cell_fold_assignments.csv",
        "model_plan": paths.design / "ml_model_hyperparameter_plan.csv",
        "preprocessing_plan": paths.design / "ml_preprocessing_plan.csv",
        "metric_plan": paths.design / "ml_metric_plan.csv",
    }


def _load_inputs(paths: Paths) -> dict[str, Any]:
    input_paths = _input_paths(paths)
    _require(input_paths.values())
    feature_regression = pd.read_csv(input_paths["feature_regression"], low_memory=False)
    if not feature_regression["Status"].eq("PASS").all():
        failures = feature_regression.loc[
            feature_regression["Status"].ne("PASS"), "Test_ID"
        ].astype(str)
        raise PrimaryBenchmarkError(
            "FEATURE_EXTRACTION_NOT_FROZEN_PASS:" + "|".join(failures)
        )
    xmatrix = pd.read_csv(input_paths["feature_matrix_xjtu"], low_memory=False)
    hmatrix = pd.read_csv(input_paths["feature_matrix_hust"], low_memory=False)
    for matrix, dataset in ((xmatrix, "XJTU"), (hmatrix, "HUST")):
        matrix["Raw_Cycle_Index"] = _numeric_int(
            matrix["Raw_Cycle_Index"], f"{dataset}_Raw_Cycle_Index"
        )
        matrix["Fold_ID"] = _numeric_int(matrix["Fold_ID"], f"{dataset}_Fold_ID")
        missing_features = sorted(set(ALL_EXTRACTED_FEATURES).difference(matrix.columns))
        if missing_features:
            raise PrimaryBenchmarkError(
                f"{dataset}_EXTRACTED_FEATURES_MISSING:{missing_features}"
            )
    xmatrix["Batch_ID"] = _numeric_int(xmatrix["Batch_ID"], "XJTU_Batch_ID")
    if len(xmatrix) != XJTU_EXPECTED_MODEL_ROWS:
        raise PrimaryBenchmarkError(f"XJTU_MATRIX_ROWS:{len(xmatrix)}")
    if len(hmatrix) != HUST_EXPECTED_PREMASK_ROWS:
        raise PrimaryBenchmarkError(f"HUST_MATRIX_ROWS:{len(hmatrix)}")
    if xmatrix.duplicated(["Batch_ID", "Cell_ID", "Raw_Cycle_Index"]).any():
        raise PrimaryBenchmarkError("DUPLICATE_XJTU_PHYSICAL_KEY")
    if hmatrix.duplicated(["Cell_ID", "Raw_Cycle_Index"]).any():
        raise PrimaryBenchmarkError("DUPLICATE_HUST_PHYSICAL_KEY")

    xlabels = pd.read_csv(input_paths["xjtu_labels"], low_memory=False)
    hlabels = pd.read_csv(input_paths["hust_labels"], low_memory=False)
    for labels, dataset in ((xlabels, "XJTU"), (hlabels, "HUST")):
        labels["Raw_Cycle_Index"] = _numeric_int(
            labels["Raw_Cycle_Index"], f"{dataset}_Label_Raw_Cycle_Index"
        )
        for column in ("Capacity_Ah", "Qref_Ah", "SOH_Fraction"):
            labels[column] = pd.to_numeric(labels[column], errors="raise")
    xlabels["Batch_ID"] = _numeric_int(xlabels["Batch_ID"], "XJTU_Label_Batch_ID")

    xfold = pd.read_csv(input_paths["xjtu_folds"], low_memory=False)
    hfold = pd.read_csv(input_paths["hust_folds"], low_memory=False)
    ifold = pd.read_csv(input_paths["xjtu_initial_folds"], low_memory=False)
    for frame, name in ((xfold, "XJTU"), (hfold, "HUST"), (ifold, "XJTU_INITIAL")):
        frame["Fold_ID"] = _numeric_int(frame["Fold_ID"], f"{name}_Fold_ID")
        if frame["Cell_ID"].duplicated().any():
            raise PrimaryBenchmarkError(f"DUPLICATE_{name}_FOLD_CELL")
        if set(frame["Fold_ID"]) != set(range(1, N_FOLDS + 1)):
            raise PrimaryBenchmarkError(f"{name}_FOLD_IDS_NOT_1_TO_5")
    model_plan = pd.read_csv(input_paths["model_plan"], low_memory=False)
    if tuple(model_plan.sort_values("Model_Order")["Model_Family"].astype(str)) != MODEL_NAMES:
        raise PrimaryBenchmarkError("MODEL_PLAN_FAMILIES_CHANGED")
    if not model_plan["Hyperparameter_Mode"].eq("FIXED").all():
        raise PrimaryBenchmarkError("MODEL_PLAN_NOT_FIXED")
    if not model_plan["Test_Fold_Tuning"].eq("PROHIBITED").all():
        raise PrimaryBenchmarkError("TEST_FOLD_TUNING_NOT_PROHIBITED")
    model_parameters = {
        row.Model_Family: json.loads(row.Hyperparameters_JSON)
        for row in model_plan.itertuples(index=False)
    }
    shortcut = pd.read_csv(input_paths["protocol_shortcut"], low_memory=False)
    preprocessing = pd.read_csv(input_paths["preprocessing_plan"], low_memory=False)
    metric_plan = pd.read_csv(input_paths["metric_plan"], low_memory=False)
    return {
        "paths": input_paths,
        "xmatrix": xmatrix,
        "hmatrix": hmatrix,
        "xlabels": xlabels,
        "hlabels": hlabels,
        "xfold": xfold,
        "hfold": hfold,
        "ifold": ifold,
        "model_plan": model_plan,
        "model_parameters": model_parameters,
        "shortcut": shortcut,
        "preprocessing": preprocessing,
        "metric_plan": metric_plan,
    }


def _feature_freeze(shortcut: pd.DataFrame) -> pd.DataFrame:
    ratios = dict(
        zip(
            shortcut["Feature_Name"].astype(str),
            pd.to_numeric(
                shortcut["Between_To_Overall_Variance_Ratio"], errors="coerce"
            ),
        )
    )
    rows: list[dict[str, Any]] = []
    for feature in ALL_EXTRACTED_FEATURES:
        if feature in PRIMARY_FEATURES:
            status = "INCLUDED_PRIMARY_FEATURE_SET_A9"
            reason = (
                "Retained after Feature Extraction V1 human review; available before "
                "completion of the target discharge and not removed by the primary protocol audit."
            )
        else:
            ratio = ratios.get(feature, np.nan)
            status = "EXCLUDED_PRIMARY_RETAINED_PROTOCOL_SHORTCUT_SENSITIVITY"
            reason = (
                "Removed from the primary A9 benchmark because XJTU between-batch variation "
                f"dominated overall variation (ratio={ratio:.15g}); retained unchanged in the "
                "frozen extracted data for a later sensitivity analysis."
            )
        rows.append(
            {
                "Feature_Name": feature,
                "Primary_Status": status,
                "Reason": reason,
                "Source_Extraction_Status": (
                    "FROZEN_SAFE_CORE_FEATURE_EXTRACTION_V1_HUMAN_APPROVED"
                ),
            }
        )
    return pd.DataFrame(rows)


def _eligible_mask(matrix: pd.DataFrame, dataset: str) -> pd.DataFrame:
    missing = matrix[list(PRIMARY_FEATURES)].isna()
    missing_count = missing.sum(axis=1).astype(int)
    missing_names = missing.apply(
        lambda row: "|".join(np.asarray(PRIMARY_FEATURES)[row.to_numpy()]), axis=1
    )
    columns = ["Dataset"]
    if dataset == "XJTU":
        columns.append("Batch_ID")
    columns.extend(
        [
            "Cell_ID",
            "Raw_Cycle_Index",
            "Fold_ID",
            "Physical_Observation_Key",
        ]
    )
    mask = matrix[columns].copy()
    mask["Primary_Feature_Set"] = "PRIMARY_FEATURE_SET_A9"
    mask["Primary_Feature_Missing_Count"] = missing_count
    mask["Missing_Primary_Features"] = missing_names
    mask["Model_Eligible"] = np.where(missing_count.eq(0), "YES", "NO")
    mask["Exclusion_Reason"] = np.where(
        missing_count.eq(0), "INCLUDED_COMPLETE_PRIMARY_A9", "INCOMPLETE_PRIMARY_A9_NO_IMPUTATION"
    )
    mask["Mask_Qref_Independent"] = "YES"
    mask["Automatic_Data_Repair"] = "NO"
    return mask


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_frozen_folds(
    matrix: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    dataset: str,
) -> None:
    key_columns = ["Cell_ID"]
    if dataset == "XJTU":
        key_columns.insert(0, "Batch_ID")
    expected = fold_assignments[key_columns + ["Fold_ID"]].copy()
    expected["Fold_ID"] = _numeric_int(
        expected["Fold_ID"], f"{dataset}_Expected_Fold_ID"
    )
    observed = matrix[key_columns + ["Fold_ID"]].drop_duplicates().copy()
    if observed.duplicated(key_columns).any():
        raise PrimaryBenchmarkError(f"{dataset}_CELL_HAS_MULTIPLE_MATRIX_FOLDS")
    joined = observed.merge(
        expected,
        on=key_columns,
        how="outer",
        suffixes=("_Matrix", "_Frozen"),
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise PrimaryBenchmarkError(f"{dataset}_FOLD_CELL_SET_CHANGED")
    if not joined["Fold_ID_Matrix"].eq(joined["Fold_ID_Frozen"]).all():
        raise PrimaryBenchmarkError(f"{dataset}_FOLD_ASSIGNMENT_CHANGED")


def _key_columns(dataset: str) -> list[str]:
    return (["Batch_ID"] if dataset == "XJTU" else []) + [
        "Cell_ID",
        "Raw_Cycle_Index",
    ]


def _align_targets(
    matrix: pd.DataFrame,
    labels: pd.DataFrame,
    dataset: str,
    qref_definitions: Iterable[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Align frozen labels to one fixed physical observation matrix in matrix order."""
    key_columns = _key_columns(dataset)
    matrix_columns = list(matrix.columns)
    base = matrix.reset_index(drop=True).copy()
    base["_Matrix_Row_Order"] = np.arange(len(base), dtype=np.int64)
    aligned: dict[str, pd.DataFrame] = {}
    key_hashes: dict[str, str] = {}
    for qref in qref_definitions:
        label_subset = labels.loc[
            labels["Qref_Definition"].astype(str).eq(qref),
            key_columns
            + ["Qref_Definition", "Qref_Ah", "SOH_Fraction", "Capacity_Ah"],
        ].copy()
        if label_subset.duplicated(key_columns).any():
            raise PrimaryBenchmarkError(
                f"{dataset}_{qref}_DUPLICATE_TARGET_PHYSICAL_KEY"
            )
        merged = base.merge(
            label_subset,
            on=key_columns,
            how="left",
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
        if len(merged) != len(base) or not merged["_merge"].eq("both").all():
            missing = merged.loc[
                merged["_merge"].ne("both"), "Physical_Observation_Key"
            ].astype(str)
            raise PrimaryBenchmarkError(
                f"{dataset}_{qref}_TARGET_ALIGNMENT_FAILED:"
                + "|".join(missing.head(10))
            )
        merged = merged.sort_values("_Matrix_Row_Order", kind="stable").reset_index(
            drop=True
        )
        for column in ("Qref_Ah", "SOH_Fraction", "Capacity_Ah"):
            merged[column] = pd.to_numeric(merged[column], errors="raise")
        numeric = merged[["Qref_Ah", "SOH_Fraction", "Capacity_Ah"]].to_numpy(
            dtype=float
        )
        if not np.isfinite(numeric).all() or (merged["Qref_Ah"] <= 0).any():
            raise PrimaryBenchmarkError(f"{dataset}_{qref}_NONFINITE_TARGET")
        recomputed = merged["Capacity_Ah"].to_numpy(dtype=float) / merged[
            "Qref_Ah"
        ].to_numpy(dtype=float)
        if not np.allclose(
            recomputed,
            merged["SOH_Fraction"].to_numpy(dtype=float),
            atol=1e-9,
            rtol=1e-9,
        ):
            raise PrimaryBenchmarkError(f"{dataset}_{qref}_SOH_FORMULA_MISMATCH")
        if not merged[matrix_columns].equals(base[matrix_columns]):
            raise PrimaryBenchmarkError(f"{dataset}_{qref}_FEATURE_ROWS_CHANGED")
        aligned[qref] = merged.drop(columns=["_Matrix_Row_Order", "_merge"])
        key_hashes[qref] = _sha256_text(merged["Physical_Observation_Key"])
    if len(set(key_hashes.values())) != 1:
        raise PrimaryBenchmarkError(f"{dataset}_QREF_SPECIFIC_ROW_DROP")
    return aligned, key_hashes


def _prepare_fold_transforms(matrix: pd.DataFrame) -> dict[int, FoldTransform]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment-specific guard
        raise PrimaryBenchmarkError("SCIKIT_LEARN_UNAVAILABLE") from exc

    feature_values = matrix[list(PRIMARY_FEATURES)].to_numpy(dtype=float)
    transforms: dict[int, FoldTransform] = {}
    for fold_id in range(1, N_FOLDS + 1):
        test_indices = np.flatnonzero(matrix["Fold_ID"].to_numpy() == fold_id)
        train_indices = np.flatnonzero(matrix["Fold_ID"].to_numpy() != fold_id)
        if not len(test_indices) or not len(train_indices):
            raise PrimaryBenchmarkError(f"EMPTY_TRAIN_OR_TEST_FOLD:{fold_id}")
        train_cells = set(matrix.iloc[train_indices]["Cell_ID"].astype(str))
        test_cells = set(matrix.iloc[test_indices]["Cell_ID"].astype(str))
        overlap = train_cells.intersection(test_cells)
        if overlap:
            raise PrimaryBenchmarkError(
                f"SAME_CELL_TRAIN_TEST_FOLD_{fold_id}:" + "|".join(sorted(overlap))
            )
        imputer = SimpleImputer(strategy="median", add_indicator=False)
        tree_train = imputer.fit_transform(feature_values[train_indices])
        tree_test = imputer.transform(feature_values[test_indices])
        if tree_train.shape[1] != len(PRIMARY_FEATURES):
            raise PrimaryBenchmarkError(f"IMPUTER_DROPPED_FEATURE_FOLD:{fold_id}")
        scaler = StandardScaler(with_mean=True, with_std=True)
        ridge_train = scaler.fit_transform(tree_train)
        ridge_test = scaler.transform(tree_test)
        if not (
            np.isfinite(tree_train).all()
            and np.isfinite(tree_test).all()
            and np.isfinite(ridge_train).all()
            and np.isfinite(ridge_test).all()
        ):
            raise PrimaryBenchmarkError(f"NONFINITE_PREPROCESSED_VALUES:{fold_id}")
        transforms[fold_id] = FoldTransform(
            fold_id=fold_id,
            train_indices=train_indices,
            test_indices=test_indices,
            tree_train=tree_train,
            tree_test=tree_test,
            ridge_train=ridge_train,
            ridge_test=ridge_test,
            train_cell_count=len(train_cells),
            test_cell_count=len(test_cells),
            train_key_sha256=_sha256_text(
                matrix.iloc[train_indices]["Physical_Observation_Key"]
            ),
            test_key_sha256=_sha256_text(
                matrix.iloc[test_indices]["Physical_Observation_Key"]
            ),
            imputer_statistics_sha256=_sha256_array(imputer.statistics_),
            scaler_mean_sha256=_sha256_array(scaler.mean_),
            scaler_scale_sha256=_sha256_array(scaler.scale_),
        )
    return transforms


def _build_model(model_name: str, parameters: dict[str, Any]) -> Any:
    try:
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.linear_model import Ridge
    except ImportError as exc:  # pragma: no cover - environment-specific guard
        raise PrimaryBenchmarkError("SCIKIT_LEARN_UNAVAILABLE") from exc
    if model_name == "Ridge Regression":
        return Ridge(**parameters)
    if model_name == "Random Forest Regressor":
        return RandomForestRegressor(**parameters)
    if model_name == "Gradient Boosting Regressor":
        return GradientBoostingRegressor(**parameters)
    raise PrimaryBenchmarkError(f"UNAPPROVED_MODEL:{model_name}")


def _prediction_frame(
    rows: pd.DataFrame,
    *,
    analysis_id: str,
    dataset: str,
    qref_definition: str,
    model_name: str,
    predicted_soh: np.ndarray,
    parameter_hash: str,
) -> pd.DataFrame:
    predicted_soh = np.asarray(predicted_soh, dtype=float)
    qref = rows["Qref_Ah"].to_numpy(dtype=float)
    predicted_capacity = predicted_soh * qref
    batch_values: Any
    if "Batch_ID" in rows:
        batch_values = rows["Batch_ID"].to_numpy()
    else:
        batch_values = np.repeat("", len(rows))
    return pd.DataFrame(
        {
            "Analysis_ID": analysis_id,
            "Dataset": dataset,
            "Batch_ID": batch_values,
            "Cell_ID": rows["Cell_ID"].astype(str).to_numpy(),
            "Raw_Cycle_Index": rows["Raw_Cycle_Index"].to_numpy(dtype=np.int64),
            "Fold_ID": rows["Fold_ID"].to_numpy(dtype=np.int64),
            "Physical_Observation_Key": rows["Physical_Observation_Key"].astype(
                str
            ).to_numpy(),
            "Qref_Definition": qref_definition,
            "Qref_Ah": qref,
            "Model": model_name,
            "Observed_SOH_Fraction": rows["SOH_Fraction"].to_numpy(dtype=float),
            "Predicted_SOH_Fraction": predicted_soh,
            "Observed_Capacity_Ah": rows["Capacity_Ah"].to_numpy(dtype=float),
            "Predicted_Capacity_Ah": predicted_capacity,
            "Prediction_Clipped": "NO",
            "Prediction_Source": "HELD_OUT_CELL_FOLD_ONLY",
            "Model_Parameters_SHA256": parameter_hash,
        },
        columns=PREDICTION_COLUMNS,
    )


def _run_population(
    *,
    analysis_id: str,
    dataset: str,
    matrix: pd.DataFrame,
    aligned_targets: dict[str, pd.DataFrame],
    qref_definitions: Iterable[str],
    model_parameters: dict[str, dict[str, Any]],
    audit: TrainingAudit,
    prediction_handle: Any | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, FoldTransform]]:
    transforms = _prepare_fold_transforms(matrix)
    cell_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for qref_definition in qref_definitions:
        target = aligned_targets[qref_definition]
        y_all = target["SOH_Fraction"].to_numpy(dtype=float)
        for model_name in MODEL_NAMES:
            parameters = model_parameters[model_name]
            parameter_json = _canonical_json(parameters)
            parameter_hash = hashlib.sha256(parameter_json.encode("utf-8")).hexdigest()
            for fold_id in range(1, N_FOLDS + 1):
                transform = transforms[fold_id]
                train_indices = transform.train_indices
                test_indices = transform.test_indices
                if model_name == "Ridge Regression":
                    train_features = transform.ridge_train
                    test_features = transform.ridge_test
                    scaling = "TRAIN_FOLD_STANDARD_SCALER"
                else:
                    train_features = transform.tree_train
                    test_features = transform.tree_test
                    scaling = "NO_SCALING"
                model = _build_model(model_name, parameters)
                start = time.perf_counter()
                model.fit(train_features, y_all[train_indices])
                predicted_soh = np.asarray(model.predict(test_features), dtype=float)
                fit_seconds = time.perf_counter() - start
                test_target = target.iloc[test_indices].reset_index(drop=True)
                predictions = _prediction_frame(
                    test_target,
                    analysis_id=analysis_id,
                    dataset=dataset,
                    qref_definition=qref_definition,
                    model_name=model_name,
                    predicted_soh=predicted_soh,
                    parameter_hash=parameter_hash,
                )
                if not np.isfinite(
                    predictions[
                        ["Predicted_SOH_Fraction", "Predicted_Capacity_Ah"]
                    ].to_numpy(dtype=float)
                ).all():
                    raise PrimaryBenchmarkError(
                        f"NONFINITE_PREDICTION:{analysis_id}:{qref_definition}:"
                        f"{model_name}:{fold_id}"
                    )
                formula_error = float(
                    np.max(
                        np.abs(
                            predictions["Predicted_Capacity_Ah"].to_numpy(dtype=float)
                            - predictions["Predicted_SOH_Fraction"].to_numpy(
                                dtype=float
                            )
                            * predictions["Qref_Ah"].to_numpy(dtype=float)
                        )
                    )
                )
                audit.max_capacity_formula_error = max(
                    audit.max_capacity_formula_error, formula_error
                )
                if prediction_handle is not None:
                    predictions.to_csv(
                        prediction_handle,
                        index=False,
                        header=False,
                        float_format="%.15g",
                    )
                audit.fit_count += 1
                audit.prediction_count += len(predictions)
                per_cell = cell_metric_rows(
                    predictions,
                    analysis_id=analysis_id,
                    dataset=dataset,
                    qref_definition=qref_definition,
                    model_name=model_name,
                )
                cell_rows.extend(per_cell)
                model_cell = pd.DataFrame(per_cell)
                pooled_model = calculate_metrics(
                    predictions["Observed_SOH_Fraction"].to_numpy(dtype=float),
                    predictions["Predicted_SOH_Fraction"].to_numpy(dtype=float),
                    predictions["Observed_Capacity_Ah"].to_numpy(dtype=float),
                    predictions["Predicted_Capacity_Ah"].to_numpy(dtype=float),
                )

                baseline_soh_value = float(np.median(y_all[train_indices]))
                baseline_soh = np.full(len(test_indices), baseline_soh_value)
                baseline_capacity = (
                    baseline_soh * test_target["Qref_Ah"].to_numpy(dtype=float)
                )
                baseline_predictions = predictions.copy()
                baseline_predictions["Predicted_SOH_Fraction"] = baseline_soh
                baseline_predictions["Predicted_Capacity_Ah"] = baseline_capacity
                baseline_cells = pd.DataFrame(
                    cell_metric_rows(
                        baseline_predictions,
                        analysis_id=analysis_id,
                        dataset=dataset,
                        qref_definition=qref_definition,
                        model_name="Training-Fold Median SOH Constant Baseline",
                    )
                )
                pooled_baseline = calculate_metrics(
                    baseline_predictions["Observed_SOH_Fraction"].to_numpy(
                        dtype=float
                    ),
                    baseline_soh,
                    baseline_predictions["Observed_Capacity_Ah"].to_numpy(
                        dtype=float
                    ),
                    baseline_capacity,
                )
                model_median_mae_q = float(model_cell["MAE_Q_Ah"].median())
                baseline_median_mae_q = float(baseline_cells["MAE_Q_Ah"].median())
                train_cells = set(target.iloc[train_indices]["Cell_ID"].astype(str))
                test_cells = set(target.iloc[test_indices]["Cell_ID"].astype(str))
                overlap_count = len(train_cells.intersection(test_cells))
                audit.train_test_cell_overlap_count += overlap_count
                record = {
                    "Analysis_ID": analysis_id,
                    "Dataset": dataset,
                    "Qref_Definition": qref_definition,
                    "Model": model_name,
                    "Fold_ID": fold_id,
                    "Train_Observation_Count": len(train_indices),
                    "Test_Observation_Count": len(test_indices),
                    "Train_Cell_Count": transform.train_cell_count,
                    "Test_Cell_Count": transform.test_cell_count,
                    "Train_Test_Cell_Overlap_Count": overlap_count,
                    "Train_Physical_Key_SHA256": transform.train_key_sha256,
                    "Test_Physical_Key_SHA256": transform.test_key_sha256,
                    "Feature_Set": "PRIMARY_FEATURE_SET_A9",
                    "Feature_Count": len(PRIMARY_FEATURES),
                    "Median_Imputer_Fit_Scope": "CURRENT_TRAINING_FOLD_ONLY",
                    "Median_Imputer_Statistics_SHA256": transform.imputer_statistics_sha256,
                    "Scaling": scaling,
                    "Scaler_Fit_Scope": (
                        "CURRENT_TRAINING_FOLD_ONLY"
                        if model_name == "Ridge Regression"
                        else "NOT_APPLICABLE"
                    ),
                    "Scaler_Mean_SHA256": (
                        transform.scaler_mean_sha256
                        if model_name == "Ridge Regression"
                        else "NOT_APPLICABLE"
                    ),
                    "Scaler_Scale_SHA256": (
                        transform.scaler_scale_sha256
                        if model_name == "Ridge Regression"
                        else "NOT_APPLICABLE"
                    ),
                    "Model_Parameters_JSON": parameter_json,
                    "Model_Parameters_SHA256": parameter_hash,
                    "Training_Seconds": fit_seconds,
                }
                audit.fit_records.append(record)
                fold_rows.append(
                    {
                        **record,
                        "Held_Out_Cell_First_Median_MAE_Q_Ah": model_median_mae_q,
                        "Held_Out_Cell_First_Median_RMSE_Q_Ah": float(
                            model_cell["RMSE_Q_Ah"].median()
                        ),
                        "Held_Out_Cell_First_Median_R2_Q": float(
                            model_cell["R2_Q"].median()
                        ),
                        "Held_Out_Cell_First_Median_MAE_SOH_fraction": float(
                            model_cell["MAE_SOH_fraction"].median()
                        ),
                        "Held_Out_Cell_First_Median_MAE_SOH_percentage_points": float(
                            model_cell["MAE_SOH_percentage_points"].median()
                        ),
                        "Pooled_Descriptive_MAE_Q_Ah": pooled_model["MAE_Q_Ah"],
                        "Pooled_Descriptive_RMSE_Q_Ah": pooled_model["RMSE_Q_Ah"],
                        "Pooled_Descriptive_R2_Q": pooled_model["R2_Q"],
                        "Pooled_Descriptive_MAE_SOH_fraction": pooled_model[
                            "MAE_SOH_fraction"
                        ],
                        "Constant_Baseline_Training_Median_SOH": baseline_soh_value,
                        "Constant_Baseline_Cell_First_Median_MAE_Q_Ah": baseline_median_mae_q,
                        "Constant_Baseline_Pooled_Descriptive_MAE_Q_Ah": pooled_baseline[
                            "MAE_Q_Ah"
                        ],
                        "Model_Worse_Than_Constant_Baseline": (
                            "YES"
                            if model_median_mae_q > baseline_median_mae_q
                            else "NO"
                        ),
                        "Primary_Ranking_Metric": "Median held-out-cell MAE_Q_Ah",
                        "Pooled_Metrics_Role": "DESCRIPTIVE_ONLY_NOT_FOR_RANKING",
                    }
                )
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"fit complete {analysis_id} | {qref_definition} | "
                    f"{model_name} | fold {fold_id}/{N_FOLDS} | "
                    f"{fit_seconds:.2f}s",
                    flush=True,
                )
    return pd.DataFrame(cell_rows), pd.DataFrame(fold_rows), transforms


def _attach_sanity_status(
    summary_metrics: pd.DataFrame, fold_metrics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_metrics = fold_metrics.copy()
    summary_metrics = summary_metrics.copy()
    status_rows: list[dict[str, Any]] = []
    for (analysis_id, dataset, qref), group in fold_metrics.groupby(
        ["Analysis_ID", "Dataset", "Qref_Definition"], sort=False
    ):
        per_fold = group.groupby("Fold_ID")[
            "Model_Worse_Than_Constant_Baseline"
        ].apply(lambda values: bool(values.eq("YES").all()))
        all_models_worse_count = int(per_fold.sum())
        status = (
            "MODEL_BENCHMARK_NEEDS_REVIEW"
            if all_models_worse_count >= MOST_FOLDS_THRESHOLD
            else "SANITY_BASELINE_NOT_DOMINANT_ACROSS_MOST_FOLDS"
        )
        status_rows.append(
            {
                "Analysis_ID": analysis_id,
                "Dataset": dataset,
                "Qref_Definition": qref,
                "Folds_All_Three_Models_Worse_Than_Constant": all_models_worse_count,
                "Most_Folds_Threshold": MOST_FOLDS_THRESHOLD,
                "Benchmark_Sanity_Status": status,
            }
        )
    status_frame = pd.DataFrame(status_rows)
    summary_metrics = summary_metrics.merge(
        status_frame,
        on=["Analysis_ID", "Dataset", "Qref_Definition"],
        how="left",
        validate="many_to_one",
    )
    fold_metrics = fold_metrics.merge(
        status_frame,
        on=["Analysis_ID", "Dataset", "Qref_Definition"],
        how="left",
        validate="many_to_one",
    )
    return summary_metrics, fold_metrics


def _model_rankings(summary_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_order = {name: index for index, name in enumerate(MODEL_NAMES, start=1)}
    for (analysis_id, dataset, qref), group in summary_metrics.groupby(
        ["Analysis_ID", "Dataset", "Qref_Definition"], sort=False
    ):
        ranked = group.assign(
            _Model_Frozen_Order=group["Model"].map(model_order)
        ).sort_values(
            ["Median_MAE_Q_Ah", "_Model_Frozen_Order"], kind="stable"
        )
        for rank_position, row in enumerate(ranked.itertuples(index=False), start=1):
            rows.append(
                {
                    "Analysis_ID": analysis_id,
                    "Dataset": dataset,
                    "Qref_Definition": qref,
                    "Rank_Position": rank_position,
                    "Model": row.Model,
                    "Median_Cell_MAE_Q_Ah": row.Median_MAE_Q_Ah,
                    "Median_Cell_MAE_SOH_fraction": row.Median_MAE_SOH_fraction,
                    "Median_Cell_MAE_SOH_percentage_points": row.Median_MAE_SOH_percentage_points,
                    "Ranking_Metric": "Median held-out-cell MAE_Q_Ah",
                    "Ranking_Direction": "ASCENDING",
                    "Interpretation_Terminology": "NEUTRAL_RANK_POSITION_UNDER_PREDEFINED_METRIC",
                }
            )
    return pd.DataFrame(rows)


def _ranking_stability(
    rankings: pd.DataFrame, summary_metrics: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in rankings["Dataset"].drop_duplicates():
        dataset_rankings = rankings.loc[rankings["Dataset"].eq(dataset)]
        dataset_summary = summary_metrics.loc[summary_metrics["Dataset"].eq(dataset)]
        for qref_a, qref_b in QREF_PAIRS:
            ranked_a = dataset_rankings.loc[
                dataset_rankings["Qref_Definition"].eq(qref_a)
            ].sort_values("Rank_Position")
            ranked_b = dataset_rankings.loc[
                dataset_rankings["Qref_Definition"].eq(qref_b)
            ].sort_values("Rank_Position")
            order_a = " > ".join(ranked_a["Model"].astype(str))
            order_b = " > ".join(ranked_b["Model"].astype(str))
            changed = "YES" if order_a != order_b else "NO"
            for model_name in MODEL_NAMES:
                a = dataset_summary.loc[
                    dataset_summary["Qref_Definition"].eq(qref_a)
                    & dataset_summary["Model"].eq(model_name)
                ].iloc[0]
                b = dataset_summary.loc[
                    dataset_summary["Qref_Definition"].eq(qref_b)
                    & dataset_summary["Model"].eq(model_name)
                ].iloc[0]
                rank_a = int(
                    ranked_a.loc[ranked_a["Model"].eq(model_name), "Rank_Position"].iloc[
                        0
                    ]
                )
                rank_b = int(
                    ranked_b.loc[ranked_b["Model"].eq(model_name), "Rank_Position"].iloc[
                        0
                    ]
                )
                rows.append(
                    {
                        "Dataset": dataset,
                        "Qref_Definition_A": qref_a,
                        "Qref_Definition_B": qref_b,
                        "Ranking_Changed": changed,
                        "Ranking_Order_A": order_a,
                        "Ranking_Order_B": order_b,
                        "Model": model_name,
                        "Rank_Position_A": rank_a,
                        "Rank_Position_B": rank_b,
                        "Delta_Rank_Position_B_minus_A": rank_b - rank_a,
                        "Median_MAE_Q_Ah_A": a["Median_MAE_Q_Ah"],
                        "Median_MAE_Q_Ah_B": b["Median_MAE_Q_Ah"],
                        "Delta_Median_MAE_Q_Ah_B_minus_A": b["Median_MAE_Q_Ah"]
                        - a["Median_MAE_Q_Ah"],
                        "Median_MAE_SOH_fraction_A": a[
                            "Median_MAE_SOH_fraction"
                        ],
                        "Median_MAE_SOH_fraction_B": b[
                            "Median_MAE_SOH_fraction"
                        ],
                        "Delta_Median_MAE_SOH_fraction_B_minus_A": b[
                            "Median_MAE_SOH_fraction"
                        ]
                        - a["Median_MAE_SOH_fraction"],
                        "Interpretation": "NUMERIC_CHANGES_ONLY_PENDING_HUMAN_REVIEW",
                    }
                )
    return pd.DataFrame(rows)


def _paired_qref_effects(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ["MAE_Q_Ah", "RMSE_Q_Ah", "MAE_SOH_fraction"]
    for dataset in cell_metrics["Dataset"].drop_duplicates():
        dataset_rows = cell_metrics.loc[cell_metrics["Dataset"].eq(dataset)]
        for model_name in MODEL_NAMES:
            model_rows = dataset_rows.loc[dataset_rows["Model"].eq(model_name)]
            for qref_a, qref_b in QREF_PAIRS:
                left = model_rows.loc[
                    model_rows["Qref_Definition"].eq(qref_a),
                    ["Cell_ID", *metrics],
                ]
                right = model_rows.loc[
                    model_rows["Qref_Definition"].eq(qref_b),
                    ["Cell_ID", *metrics],
                ]
                paired = left.merge(
                    right,
                    on="Cell_ID",
                    how="outer",
                    suffixes=("_A", "_B"),
                    indicator=True,
                    validate="one_to_one",
                )
                if not paired["_merge"].eq("both").all():
                    raise PrimaryBenchmarkError(
                        f"PAIRED_QREF_CELL_SET_CHANGED:{dataset}:{model_name}:"
                        f"{qref_a}:{qref_b}"
                    )
                paired["Delta_MAE_Q_Ah_per_cell"] = (
                    paired["MAE_Q_Ah_B"] - paired["MAE_Q_Ah_A"]
                )
                paired["Delta_RMSE_Q_Ah_per_cell"] = (
                    paired["RMSE_Q_Ah_B"] - paired["RMSE_Q_Ah_A"]
                )
                paired["Delta_MAE_SOH_per_cell"] = (
                    paired["MAE_SOH_fraction_B"]
                    - paired["MAE_SOH_fraction_A"]
                )
                for row in paired.itertuples(index=False):
                    rows.append(
                        {
                            "Row_Type": "PAIRED_CELL",
                            "Dataset": dataset,
                            "Model": model_name,
                            "Qref_Definition_A": qref_a,
                            "Qref_Definition_B": qref_b,
                            "Cell_ID": row.Cell_ID,
                            "Paired_Cell_Count": 1,
                            "Delta_MAE_Q_Ah_per_cell": row.Delta_MAE_Q_Ah_per_cell,
                            "Delta_RMSE_Q_Ah_per_cell": row.Delta_RMSE_Q_Ah_per_cell,
                            "Delta_MAE_SOH_per_cell": row.Delta_MAE_SOH_per_cell,
                            "Summary_Statistic": "",
                            "Summary_Delta_MAE_Q_Ah": np.nan,
                            "Summary_Delta_RMSE_Q_Ah": np.nan,
                            "Summary_Delta_MAE_SOH": np.nan,
                        }
                    )
                statistics = {
                    "MEDIAN": lambda series: float(series.median()),
                    "Q1": lambda series: float(series.quantile(0.25)),
                    "Q3": lambda series: float(series.quantile(0.75)),
                    "IQR": lambda series: float(
                        series.quantile(0.75) - series.quantile(0.25)
                    ),
                    "MIN": lambda series: float(series.min()),
                    "MAX": lambda series: float(series.max()),
                }
                for statistic_name, statistic in statistics.items():
                    rows.append(
                        {
                            "Row_Type": "PAIRED_SUMMARY",
                            "Dataset": dataset,
                            "Model": model_name,
                            "Qref_Definition_A": qref_a,
                            "Qref_Definition_B": qref_b,
                            "Cell_ID": "",
                            "Paired_Cell_Count": len(paired),
                            "Delta_MAE_Q_Ah_per_cell": np.nan,
                            "Delta_RMSE_Q_Ah_per_cell": np.nan,
                            "Delta_MAE_SOH_per_cell": np.nan,
                            "Summary_Statistic": statistic_name,
                            "Summary_Delta_MAE_Q_Ah": statistic(
                                paired["Delta_MAE_Q_Ah_per_cell"]
                            ),
                            "Summary_Delta_RMSE_Q_Ah": statistic(
                                paired["Delta_RMSE_Q_Ah_per_cell"]
                            ),
                            "Summary_Delta_MAE_SOH": statistic(
                                paired["Delta_MAE_SOH_per_cell"]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _xjtu_batch_metrics(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    xjtu = cell_metrics.loc[cell_metrics["Dataset"].eq("XJTU")].copy()
    rows: list[dict[str, Any]] = []
    for row in xjtu.itertuples(index=False):
        rows.append(
            {
                "Row_Type": "HELD_OUT_CELL",
                "Batch_ID": row.Batch_ID,
                "Cell_ID": row.Cell_ID,
                "Qref_Definition": row.Qref_Definition,
                "Model": row.Model,
                "Cell_Count": 1,
                "Observation_Count": row.Observation_Count,
                "MAE_Q_Ah": row.MAE_Q_Ah,
                "RMSE_Q_Ah": row.RMSE_Q_Ah,
                "R2_Q": row.R2_Q,
                "MAE_SOH_fraction": row.MAE_SOH_fraction,
                "Median_Cell_MAE_Q_Ah": np.nan,
                "Q1_Cell_MAE_Q_Ah": np.nan,
                "Q3_Cell_MAE_Q_Ah": np.nan,
                "IQR_Cell_MAE_Q_Ah": np.nan,
                "Mean_Cell_MAE_Q_Ah": np.nan,
            }
        )
    for (batch_id, qref, model), group in xjtu.groupby(
        ["Batch_ID", "Qref_Definition", "Model"], sort=True
    ):
        q1 = float(group["MAE_Q_Ah"].quantile(0.25))
        q3 = float(group["MAE_Q_Ah"].quantile(0.75))
        rows.append(
            {
                "Row_Type": "BATCH_CELL_FIRST_SUMMARY",
                "Batch_ID": batch_id,
                "Cell_ID": "",
                "Qref_Definition": qref,
                "Model": model,
                "Cell_Count": int(group["Cell_ID"].nunique()),
                "Observation_Count": int(group["Observation_Count"].sum()),
                "MAE_Q_Ah": np.nan,
                "RMSE_Q_Ah": np.nan,
                "R2_Q": np.nan,
                "MAE_SOH_fraction": np.nan,
                "Median_Cell_MAE_Q_Ah": float(group["MAE_Q_Ah"].median()),
                "Q1_Cell_MAE_Q_Ah": q1,
                "Q3_Cell_MAE_Q_Ah": q3,
                "IQR_Cell_MAE_Q_Ah": q3 - q1,
                "Mean_Cell_MAE_Q_Ah": float(group["MAE_Q_Ah"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _secondary_metrics_output(
    cell_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    summary = summarize_cell_metrics(cell_metrics)
    summary, fold_metrics = _attach_sanity_status(summary, fold_metrics)
    cell_part = cell_metrics.copy()
    cell_part.insert(0, "Metric_Level", "HELD_OUT_CELL")
    fold_part = fold_metrics.copy()
    fold_part.insert(0, "Metric_Level", "HELD_OUT_FOLD")
    summary_part = summary.copy()
    summary_part.insert(0, "Metric_Level", "CELL_FIRST_CROSS_CELL_SUMMARY")
    return pd.concat(
        [cell_part, fold_part, summary_part], ignore_index=True, sort=False
    )


def _test_row(
    test_id: str,
    passed: bool,
    observed: Any,
    expected: Any,
    failure_message: str,
    *,
    hard_failure: bool = True,
) -> dict[str, Any]:
    return {
        "Test_ID": test_id,
        "Status": "PASS" if passed else "FAIL",
        "Observed": str(observed),
        "Expected": str(expected),
        "Failure_Message": "" if passed else failure_message,
        "Hard_Failure": "YES" if hard_failure else "NO",
    }


def _regression_tests(
    *,
    feature_freeze: pd.DataFrame,
    xmatrix: pd.DataFrame,
    hmatrix: pd.DataFrame,
    xmask: pd.DataFrame,
    hmask: pd.DataFrame,
    primary_key_hashes: dict[str, dict[str, str]],
    secondary_key_hashes: dict[str, str],
    primary_audit: TrainingAudit,
    secondary_audit: TrainingAudit,
    primary_cell_metrics: pd.DataFrame,
    primary_fold_metrics: pd.DataFrame,
    primary_summary_metrics: pd.DataFrame,
    rankings: pd.DataFrame,
    secondary_matrix: pd.DataFrame,
    model_parameters: dict[str, dict[str, Any]],
    input_hashes_before: dict[str, str],
    input_hashes_after: dict[str, str],
) -> pd.DataFrame:
    tests: list[dict[str, Any]] = []
    included_features = feature_freeze.loc[
        feature_freeze["Primary_Status"].eq("INCLUDED_PRIMARY_FEATURE_SET_A9"),
        "Feature_Name",
    ].tolist()
    tests.append(
        _test_row(
            "PRIMARY_FEATURE_COUNT",
            len(included_features) == 9 and tuple(included_features) == PRIMARY_FEATURES,
            len(included_features),
            "9 exact frozen A9 features in the approved order",
            "Primary feature allowlist changed.",
        )
    )
    tests.append(
        _test_row(
            "PROTOCOL_SHORTCUT_FEATURES_ABSENT_FROM_PRIMARY",
            set(PROTOCOL_SHORTCUT_FEATURES).isdisjoint(PRIMARY_FEATURES),
            "|".join(sorted(set(PROTOCOL_SHORTCUT_FEATURES).intersection(PRIMARY_FEATURES)))
            or "NONE",
            "NONE",
            "A protocol-shortcut feature entered the primary benchmark.",
        )
    )
    xeligible = int(xmask["Model_Eligible"].eq("YES").sum())
    heligible = int(hmask["Model_Eligible"].eq("YES").sum())
    tests.append(
        _test_row(
            "XJTU_MODEL_ROWS",
            len(xmatrix) == XJTU_EXPECTED_MODEL_ROWS
            and xeligible == XJTU_EXPECTED_MODEL_ROWS,
            f"premask={len(xmatrix)};eligible={xeligible}",
            f"premask={XJTU_EXPECTED_MODEL_ROWS};eligible={XJTU_EXPECTED_MODEL_ROWS}",
            "XJTU primary model population is not the frozen 10,427 rows.",
        )
    )
    tests.append(
        _test_row(
            "HUST_MODEL_MASK_QREF_INDEPENDENT",
            hmask["Mask_Qref_Independent"].eq("YES").all()
            and heligible == len(hmatrix) - int(
                hmask["Primary_Feature_Missing_Count"].gt(0).sum()
            ),
            f"premask={len(hmatrix)};eligible={heligible};mask_values="
            + "|".join(sorted(hmask["Mask_Qref_Independent"].unique())),
            "one A9-only mask shared by every Qref and model",
            "HUST eligibility mask is not Qref independent.",
        )
    )
    common_hashes_ok = all(
        len(set(hashes.values())) == 1 for hashes in primary_key_hashes.values()
    )
    tests.append(
        _test_row(
            "NO_QREF_SPECIFIC_ROW_DROP",
            common_hashes_ok,
            _canonical_json(primary_key_hashes),
            "one physical-key hash per dataset across all common Qrefs",
            "Physical observation keys differ by Qref.",
        )
    )
    lower_features = [name.lower() for name in PRIMARY_FEATURES]
    tests.append(
        _test_row(
            "NO_CAPACITY_AS_FEATURE",
            not any("capacity" in name for name in lower_features),
            "|".join(PRIMARY_FEATURES),
            "no capacity predictor",
            "Capacity entered the predictor matrix.",
        )
    )
    tests.append(
        _test_row(
            "NO_QREF_AS_FEATURE",
            not any("qref" in name for name in lower_features),
            "|".join(PRIMARY_FEATURES),
            "no Qref predictor",
            "Qref entered the predictor matrix.",
        )
    )
    tests.append(
        _test_row(
            "NO_CYCLE_INDEX_IN_PRIMARY",
            not any("cycle" in name or "index" in name for name in lower_features),
            "|".join(PRIMARY_FEATURES),
            "no cycle/index predictor",
            "A cycle index entered the primary predictor matrix.",
        )
    )
    tests.append(
        _test_row(
            "NO_EOLO_RUL_NATIVE_LABEL_FEATURES",
            not any(
                token in name
                for name in lower_features
                for token in ("eol", "rul", "soh")
            ),
            "|".join(PRIMARY_FEATURES),
            "no native EOL/RUL/SOH label feature",
            "A native outcome label entered the primary predictor matrix.",
        )
    )
    overlap = (
        primary_audit.train_test_cell_overlap_count
        + secondary_audit.train_test_cell_overlap_count
    )
    tests.append(
        _test_row(
            "NO_SAME_CELL_TRAIN_TEST",
            overlap == 0,
            overlap,
            0,
            "At least one cell crossed a training/test boundary.",
        )
    )
    tests.append(
        _test_row(
            "FOLDS_MATCH_FROZEN_DESIGN",
            True,
            "validated against all three frozen fold-assignment files",
            "exact match",
            "Frozen folds changed.",
        )
    )
    tests.append(
        _test_row(
            "FEATURE_VALUES_IDENTICAL_ACROSS_QREFS",
            common_hashes_ok,
            "A9 matrix reused; " + _canonical_json(primary_key_hashes),
            "identical feature rows and values across Qrefs",
            "Qref-specific feature values were detected.",
        )
    )
    fit_records = pd.DataFrame(
        [*(primary_audit.fit_records or []), *(secondary_audit.fit_records or [])]
    )
    parameter_consistency = bool(
        fit_records.groupby(["Analysis_ID", "Model"])[
            "Model_Parameters_SHA256"
        ].nunique().eq(1).all()
    )
    planned_hashes = {
        model: hashlib.sha256(
            _canonical_json(parameters).encode("utf-8")
        ).hexdigest()
        for model, parameters in model_parameters.items()
    }
    fitted_hashes = set(fit_records["Model_Parameters_SHA256"].astype(str))
    parameter_consistency = parameter_consistency and fitted_hashes == set(
        planned_hashes.values()
    )
    tests.append(
        _test_row(
            "MODEL_PARAMETERS_IDENTICAL_ACROSS_QREFS",
            parameter_consistency,
            _canonical_json(planned_hashes),
            "one frozen parameter hash per model across Qrefs",
            "Model parameters changed by Qref or departed from the frozen plan.",
        )
    )
    preprocessing_ok = bool(
        fit_records["Median_Imputer_Fit_Scope"]
        .eq("CURRENT_TRAINING_FOLD_ONLY")
        .all()
        and fit_records.loc[
            fit_records["Model"].eq("Ridge Regression"), "Scaler_Fit_Scope"
        ]
        .eq("CURRENT_TRAINING_FOLD_ONLY")
        .all()
        and fit_records["Train_Test_Cell_Overlap_Count"].eq(0).all()
    )
    tests.append(
        _test_row(
            "PREPROCESSING_FIT_ON_TRAIN_ONLY",
            preprocessing_ok,
            f"fit_records={len(fit_records)};overlap={overlap}",
            "all imputer/scaler fits use current training-fold rows only",
            "Held-out information entered preprocessing.",
        )
    )
    tests.append(
        _test_row(
            "PREDICTED_CAPACITY_FORMULA_CORRECT",
            primary_audit.max_capacity_formula_error <= 1e-12,
            primary_audit.max_capacity_formula_error,
            "<=1e-12 Ah absolute arithmetic error",
            "Predicted capacity is not Predicted_SOH_Fraction * Qref_Ah.",
        )
    )
    expected_cell_metric_rows = (
        xmatrix["Cell_ID"].nunique() + hmatrix["Cell_ID"].nunique()
    ) * len(COMMON_QREFS) * len(MODEL_NAMES)
    cell_first_ok = (
        len(primary_cell_metrics) == expected_cell_metric_rows
        and primary_cell_metrics.groupby(
            ["Dataset", "Cell_ID", "Qref_Definition", "Model"]
        ).size().eq(1).all()
        and primary_summary_metrics["Aggregation_Order"]
        .eq("CELL_FIRST_THEN_CROSS_CELL_SUMMARY")
        .all()
    )
    tests.append(
        _test_row(
            "CELL_FIRST_METRIC_AGGREGATION",
            cell_first_ok,
            f"cell_metric_rows={len(primary_cell_metrics)}",
            expected_cell_metric_rows,
            "Metrics were not calculated once per held-out cell before summarization.",
        )
    )
    ranking_ok = (
        rankings["Ranking_Metric"]
        .eq("Median held-out-cell MAE_Q_Ah")
        .all()
        and rankings.groupby(["Dataset", "Qref_Definition"])["Rank_Position"]
        .apply(lambda values: set(values) == {1, 2, 3})
        .all()
    )
    tests.append(
        _test_row(
            "PRIMARY_RANKING_USES_MEDIAN_CELL_MAE_Q",
            bool(ranking_ok),
            "Median held-out-cell MAE_Q_Ah",
            "Median held-out-cell MAE_Q_Ah with ranks 1,2,3",
            "Primary ranking did not use the frozen cell-first physical metric.",
        )
    )
    secondary_cells = secondary_matrix["Cell_ID"].nunique()
    secondary_common = len(set(secondary_key_hashes.values())) == 1
    tests.append(
        _test_row(
            "XJTU_INITIAL_TEST_COMMON_47_CELL_ONLY",
            secondary_cells == 47 and secondary_common,
            f"cells={secondary_cells};rows={len(secondary_matrix)};hashes="
            + _canonical_json(secondary_key_hashes),
            "47 cells and identical nominal/initial_test physical keys",
            "Secondary nominal and initial_test populations are not the same frozen 47 cells.",
        )
    )
    tests.append(
        _test_row(
            "NO_FINAL_RUL_MODEL",
            not any("rul" in name.lower() for name in MODEL_NAMES),
            "|".join(MODEL_NAMES),
            "Ridge/RF/Gradient Boosting SOH benchmark only",
            "An unauthorized RUL model was included.",
        )
    )
    missing_hust = hmask.loc[hmask["Model_Eligible"].eq("NO")]
    tests.append(
        _test_row(
            "HUST_ISOLATED_INCOMPLETE_A9_ROW_EXCLUDED_ONCE",
            len(missing_hust) == 1
            and missing_hust["Physical_Observation_Key"].iloc[0]
            == "HUST|HUST_7-7|1"
            and heligible == len(hmatrix) - 1,
            "|".join(missing_hust["Physical_Observation_Key"].astype(str)),
            "HUST|HUST_7-7|1 only",
            "The frozen HUST complete-case exclusion changed.",
        )
    )
    expected_prediction_rows = (
        xeligible + heligible
    ) * len(COMMON_QREFS) * len(MODEL_NAMES)
    tests.append(
        _test_row(
            "PRIMARY_HELD_OUT_PREDICTION_ROW_COUNT",
            primary_audit.prediction_count == expected_prediction_rows,
            primary_audit.prediction_count,
            expected_prediction_rows,
            "Held-out predictions are missing or duplicated.",
        )
    )
    expected_primary_fits = 2 * len(COMMON_QREFS) * len(MODEL_NAMES) * N_FOLDS
    expected_secondary_fits = (
        len(SECONDARY_XJTU_QREFS) * len(MODEL_NAMES) * N_FOLDS
    )
    tests.append(
        _test_row(
            "FROZEN_MODEL_FIT_COUNTS",
            primary_audit.fit_count == expected_primary_fits
            and secondary_audit.fit_count == expected_secondary_fits,
            f"primary={primary_audit.fit_count};secondary={secondary_audit.fit_count}",
            f"primary={expected_primary_fits};secondary={expected_secondary_fits}",
            "The fixed model/fold/Qref training grid is incomplete or duplicated.",
        )
    )
    tests.append(
        _test_row(
            "FROZEN_INPUT_FILES_UNCHANGED",
            input_hashes_before == input_hashes_after,
            _canonical_json(input_hashes_after),
            _canonical_json(input_hashes_before),
            "A frozen input file changed during model training.",
        )
    )
    return pd.DataFrame(tests)


def _validate_written_outputs(
    staging: Path,
    primary_audit: TrainingAudit,
) -> dict[str, Any]:
    missing = [name for name in CSV_OUTPUTS if not (staging / name).is_file()]
    if missing:
        raise PrimaryBenchmarkError("MISSING_OUTPUT_FILES:" + "|".join(missing))
    details: dict[str, Any] = {}
    for name in CSV_OUTPUTS:
        path = staging / name
        header = pd.read_csv(path, nrows=0)
        if header.columns.duplicated().any() or len(header.columns) == 0:
            raise PrimaryBenchmarkError(f"INVALID_OUTPUT_HEADER:{name}")
        details[name] = {
            "bytes": path.stat().st_size,
            "columns": len(header.columns),
        }
    prediction_path = staging / "soh_primary_predictions.csv"
    row_count = 0
    blank_cells = 0
    formula_max = 0.0
    for chunk in pd.read_csv(prediction_path, chunksize=200_000, low_memory=False):
        row_count += len(chunk)
        blank_cells += int(chunk["Cell_ID"].isna().sum())
        formula_max = max(
            formula_max,
            float(
                np.max(
                    np.abs(
                        chunk["Predicted_Capacity_Ah"].to_numpy(dtype=float)
                        - chunk["Predicted_SOH_Fraction"].to_numpy(dtype=float)
                        * chunk["Qref_Ah"].to_numpy(dtype=float)
                    )
                )
            ),
        )
    if row_count != primary_audit.prediction_count:
        raise PrimaryBenchmarkError(
            f"WRITTEN_PREDICTION_COUNT_MISMATCH:{row_count}:"
            f"{primary_audit.prediction_count}"
        )
    if blank_cells:
        raise PrimaryBenchmarkError(f"BLANK_PREDICTION_CELL_ID:{blank_cells}")
    if formula_max > 1e-12:
        raise PrimaryBenchmarkError(
            f"WRITTEN_PREDICTED_CAPACITY_FORMULA_ERROR:{formula_max}"
        )
    details["prediction_validation"] = {
        "rows": row_count,
        "blank_cell_ids": blank_cells,
        "max_capacity_formula_error": formula_max,
    }
    return details


def _processing_log(
    *,
    started: datetime,
    finished: datetime,
    input_paths: dict[str, Path],
    input_hashes: dict[str, str],
    xmask: pd.DataFrame,
    hmask: pd.DataFrame,
    primary_audit: TrainingAudit,
    secondary_audit: TrainingAudit,
    primary_fold_metrics: pd.DataFrame,
    regression_tests: pd.DataFrame,
    validation_details: dict[str, Any] | None,
) -> str:
    try:
        import scipy
        import sklearn
    except ImportError as exc:  # pragma: no cover
        raise PrimaryBenchmarkError("DEPENDENCY_VERSION_QUERY_FAILED") from exc
    sanity = (
        primary_fold_metrics[
            ["Dataset", "Qref_Definition", "Benchmark_Sanity_Status"]
        ]
        .drop_duplicates()
        .sort_values(["Dataset", "Qref_Definition"])
    )
    lines = [
        f"Pipeline: {PIPELINE_VERSION}",
        f"Status: {'ALL HARD REGRESSION TESTS PASSED' if regression_tests['Status'].eq('PASS').all() else 'PIPELINE FAILED'}",
        f"Started: {started.isoformat()}",
        f"Finished: {finished.isoformat()}",
        f"Elapsed_seconds: {(finished - started).total_seconds():.3f}",
        f"Python: {platform.python_version()}",
        f"numpy: {np.__version__}",
        f"pandas: {pd.__version__}",
        f"scipy: {scipy.__version__}",
        f"scikit-learn: {sklearn.__version__}",
        f"Random_seed: {RANDOM_SEED}",
        "Primary_feature_set: PRIMARY_FEATURE_SET_A9",
        "Primary_features: " + "|".join(PRIMARY_FEATURES),
        "Excluded_primary_protocol_shortcuts_retained_in_frozen_extraction: "
        + "|".join(PROTOCOL_SHORTCUT_FEATURES),
        "Primary_Qrefs: " + "|".join(COMMON_QREFS),
        "Secondary_XJTU_Qrefs: " + "|".join(SECONDARY_XJTU_QREFS),
        "Models: " + "|".join(MODEL_NAMES),
        "Cross_validation: frozen deterministic five-fold cell-level assignments",
        "Imputation: training-fold median fit only; HUST incomplete A9 row excluded before training",
        "Ridge_scaling: training-fold StandardScaler fit only",
        "RF_GB_scaling: none",
        "Prediction_clipping: none",
        "EOL_truncation: none",
        "Hyperparameter_tuning: none",
        "Raw_dataset_pooling: prohibited; XJTU and HUST trained separately",
        f"XJTU_mask_rows: {len(xmask)}",
        f"XJTU_model_eligible_rows: {int(xmask['Model_Eligible'].eq('YES').sum())}",
        f"HUST_mask_rows: {len(hmask)}",
        f"HUST_model_eligible_rows: {int(hmask['Model_Eligible'].eq('YES').sum())}",
        "HUST_excluded_keys: "
        + "|".join(
            hmask.loc[hmask["Model_Eligible"].eq("NO"), "Physical_Observation_Key"].astype(
                str
            )
        ),
        f"Primary_model_fits: {primary_audit.fit_count}",
        f"Secondary_model_fits: {secondary_audit.fit_count}",
        f"Primary_prediction_rows: {primary_audit.prediction_count}",
        f"Max_capacity_backtransform_error_Ah: {primary_audit.max_capacity_formula_error:.17g}",
        "Input_files_and_SHA256:",
    ]
    for name, path in input_paths.items():
        lines.append(
            f"  {name}: {path} | bytes={path.stat().st_size} | sha256={input_hashes[name]}"
        )
    lines.append("Sanity_baseline_status:")
    for row in sanity.itertuples(index=False):
        lines.append(
            f"  {row.Dataset}|{row.Qref_Definition}: {row.Benchmark_Sanity_Status}"
        )
    lines.append("Regression_tests:")
    for row in regression_tests.itertuples(index=False):
        lines.append(
            f"  {row.Test_ID}: {row.Status} | observed={row.Observed} | expected={row.Expected}"
        )
    if validation_details is not None:
        lines.append("Written_output_validation:")
        lines.append(_canonical_json(validation_details))
    lines.append(
        "Final_status: "
        + (
            "SOH PRIMARY BENCHMARK V1 COMPLETE — WAITING FOR HUMAN REVIEW"
            if regression_tests["Status"].eq("PASS").all()
            else "SOH PRIMARY BENCHMARK V1 FAILED"
        )
    )
    return "\n".join(lines) + "\n"


def run_primary_benchmark(
    workspace: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    paths = Paths.from_workspace(workspace)
    target = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (workspace / "outputs" / "soh_primary_benchmark_v1")
    )
    if target.exists():
        raise PrimaryBenchmarkError(f"OUTPUT_ALREADY_EXISTS:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".soh_primary_benchmark_v1_staging_", dir=target.parent
        )
    )
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    print(f"Staging output: {staging}", flush=True)
    input_paths = _input_paths(paths)
    _require(input_paths.values())
    input_hashes_before = {
        name: _sha256_file(path) for name, path in input_paths.items()
    }
    loaded = _load_inputs(paths)
    xmatrix = loaded["xmatrix"]
    hmatrix = loaded["hmatrix"]
    _validate_frozen_folds(xmatrix, loaded["xfold"], "XJTU")
    _validate_frozen_folds(hmatrix, loaded["hfold"], "HUST")

    feature_freeze = _feature_freeze(loaded["shortcut"])
    xmask = _eligible_mask(xmatrix, "XJTU")
    hmask = _eligible_mask(hmatrix, "HUST")
    _write_csv(feature_freeze, staging / "soh_primary_feature_freeze_v1.csv")
    _write_csv(xmask, staging / "soh_model_eligible_mask_xjtu.csv")
    _write_csv(hmask, staging / "soh_model_eligible_mask_hust.csv")
    xeligible = xmatrix.loc[xmask["Model_Eligible"].eq("YES")].reset_index(drop=True)
    heligible = hmatrix.loc[hmask["Model_Eligible"].eq("YES")].reset_index(drop=True)

    x_targets, x_key_hashes = _align_targets(
        xeligible, loaded["xlabels"], "XJTU", COMMON_QREFS
    )
    h_targets, h_key_hashes = _align_targets(
        heligible, loaded["hlabels"], "HUST", COMMON_QREFS
    )
    primary_key_hashes = {"XJTU": x_key_hashes, "HUST": h_key_hashes}
    primary_audit = TrainingAudit()
    prediction_path = staging / "soh_primary_predictions.csv"
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
        pd.DataFrame(columns=PREDICTION_COLUMNS).to_csv(handle, index=False)
        x_cell, x_fold_metrics, _ = _run_population(
            analysis_id="XJTU_PRIMARY_COMMON_QREFS_55_CELL",
            dataset="XJTU",
            matrix=xeligible,
            aligned_targets=x_targets,
            qref_definitions=COMMON_QREFS,
            model_parameters=loaded["model_parameters"],
            audit=primary_audit,
            prediction_handle=handle,
        )
        h_cell, h_fold_metrics, _ = _run_population(
            analysis_id="HUST_PRIMARY_COMMON_QREFS_77_CELL",
            dataset="HUST",
            matrix=heligible,
            aligned_targets=h_targets,
            qref_definitions=COMMON_QREFS,
            model_parameters=loaded["model_parameters"],
            audit=primary_audit,
            prediction_handle=handle,
        )
    primary_cell_metrics = pd.concat([x_cell, h_cell], ignore_index=True)
    primary_fold_metrics = pd.concat(
        [x_fold_metrics, h_fold_metrics], ignore_index=True
    )
    primary_summary_metrics = summarize_cell_metrics(primary_cell_metrics)
    primary_summary_metrics, primary_fold_metrics = _attach_sanity_status(
        primary_summary_metrics, primary_fold_metrics
    )
    rankings = _model_rankings(primary_summary_metrics)
    ranking_stability = _ranking_stability(rankings, primary_summary_metrics)
    paired_effects = _paired_qref_effects(primary_cell_metrics)
    xjtu_batch_metrics = _xjtu_batch_metrics(primary_cell_metrics)

    initial_cells = loaded["ifold"][["Batch_ID", "Cell_ID", "Fold_ID"]].copy()
    secondary_matrix = xmatrix.merge(
        initial_cells,
        on=["Batch_ID", "Cell_ID"],
        how="inner",
        suffixes=("", "_Initial_Frozen"),
        validate="many_to_one",
    )
    if not secondary_matrix["Fold_ID"].eq(
        secondary_matrix["Fold_ID_Initial_Frozen"]
    ).all():
        raise PrimaryBenchmarkError("XJTU_INITIAL47_FOLD_ASSIGNMENT_CHANGED")
    secondary_matrix = secondary_matrix.drop(columns="Fold_ID_Initial_Frozen").reset_index(
        drop=True
    )
    _validate_frozen_folds(secondary_matrix, loaded["ifold"], "XJTU")
    secondary_targets, secondary_key_hashes = _align_targets(
        secondary_matrix,
        loaded["xlabels"],
        "XJTU",
        SECONDARY_XJTU_QREFS,
    )
    secondary_audit = TrainingAudit()
    secondary_cell, secondary_fold, _ = _run_population(
        analysis_id="XJTU_SECONDARY_INITIAL_TEST_COMMON_47_CELL",
        dataset="XJTU",
        matrix=secondary_matrix,
        aligned_targets=secondary_targets,
        qref_definitions=SECONDARY_XJTU_QREFS,
        model_parameters=loaded["model_parameters"],
        audit=secondary_audit,
        prediction_handle=None,
    )
    secondary_output = _secondary_metrics_output(secondary_cell, secondary_fold)

    _write_csv(primary_cell_metrics, staging / "soh_primary_cell_metrics.csv")
    _write_csv(primary_fold_metrics, staging / "soh_primary_fold_metrics.csv")
    _write_csv(primary_summary_metrics, staging / "soh_primary_summary_metrics.csv")
    _write_csv(rankings, staging / "soh_primary_model_rankings.csv")
    _write_csv(ranking_stability, staging / "soh_primary_ranking_stability.csv")
    _write_csv(paired_effects, staging / "soh_primary_qref_paired_effects.csv")
    _write_csv(xjtu_batch_metrics, staging / "soh_primary_xjtu_batch_metrics.csv")
    _write_csv(
        secondary_output, staging / "soh_secondary_xjtu_initial_test_metrics.csv"
    )

    input_hashes_after = {
        name: _sha256_file(path) for name, path in input_paths.items()
    }
    regression_tests = _regression_tests(
        feature_freeze=feature_freeze,
        xmatrix=xmatrix,
        hmatrix=hmatrix,
        xmask=xmask,
        hmask=hmask,
        primary_key_hashes=primary_key_hashes,
        secondary_key_hashes=secondary_key_hashes,
        primary_audit=primary_audit,
        secondary_audit=secondary_audit,
        primary_cell_metrics=primary_cell_metrics,
        primary_fold_metrics=primary_fold_metrics,
        primary_summary_metrics=primary_summary_metrics,
        rankings=rankings,
        secondary_matrix=secondary_matrix,
        model_parameters=loaded["model_parameters"],
        input_hashes_before=input_hashes_before,
        input_hashes_after=input_hashes_after,
    )
    _write_csv(regression_tests, staging / "soh_primary_regression_tests.csv")
    finished = datetime.now(ZoneInfo("Asia/Shanghai"))
    preliminary_log = _processing_log(
        started=started,
        finished=finished,
        input_paths=input_paths,
        input_hashes=input_hashes_after,
        xmask=xmask,
        hmask=hmask,
        primary_audit=primary_audit,
        secondary_audit=secondary_audit,
        primary_fold_metrics=primary_fold_metrics,
        regression_tests=regression_tests,
        validation_details=None,
    )
    (staging / "processing_log.txt").write_text(preliminary_log, encoding="utf-8")
    failures = regression_tests.loc[
        regression_tests["Status"].ne("PASS")
        & regression_tests["Hard_Failure"].eq("YES")
    ]
    if not failures.empty:
        raise PrimaryBenchmarkError(
            "HARD_REGRESSION_FAILURES:"
            + "|".join(failures["Test_ID"].astype(str))
            + f";staging={staging}"
        )
    validation_details = _validate_written_outputs(staging, primary_audit)
    finished = datetime.now(ZoneInfo("Asia/Shanghai"))
    final_log = _processing_log(
        started=started,
        finished=finished,
        input_paths=input_paths,
        input_hashes=input_hashes_after,
        xmask=xmask,
        hmask=hmask,
        primary_audit=primary_audit,
        secondary_audit=secondary_audit,
        primary_fold_metrics=primary_fold_metrics,
        regression_tests=regression_tests,
        validation_details=validation_details,
    )
    (staging / "processing_log.txt").write_text(final_log, encoding="utf-8")
    staging.replace(target)
    return {
        "output_dir": str(target),
        "regression_tests_passed": True,
        "primary_prediction_rows": primary_audit.prediction_count,
        "final_stdout": (
            "SOH PRIMARY BENCHMARK V1 COMPLETE —\nWAITING FOR HUMAN REVIEW"
        ),
    }


__all__ = [
    "PrimaryBenchmarkError",
    "run_primary_benchmark",
    "_eligible_mask",
    "_feature_freeze",
]
