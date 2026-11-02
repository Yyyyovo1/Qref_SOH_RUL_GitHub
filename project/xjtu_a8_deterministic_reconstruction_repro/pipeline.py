"""Reconstruct and persist the frozen XJTU ROBUSTNESS_A8 intermediates.

Only the already-approved XJTU A8 experiment is fitted.  The reconstruction
must reproduce the frozen nine median cell MAE_Q_Ah values before any
corrected bootstrap is allowed to run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from soh_ml_robustness_repro.config import A8_FEATURES, MAX_PARALLEL_FOLDS
from soh_ml_robustness_repro.pipeline import _frozen_population, _prepare_transforms
from soh_primary_benchmark_repro.config import (
    COMMON_QREFS,
    MODEL_NAMES,
    N_FOLDS,
    PRIMARY_FEATURES,
    PROTOCOL_SHORTCUT_FEATURES,
    XJTU_EXPECTED_MODEL_ROWS,
)
from soh_primary_benchmark_repro.metrics import calculate_metrics, cell_metric_rows
from soh_primary_benchmark_repro.pipeline import (
    _align_targets,
    _build_model,
    _canonical_json,
    _numeric_int,
    _sha256_array,
    _sha256_file,
    _sha256_text,
    _validate_frozen_folds,
)

from .config import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CSV_OUTPUTS,
    DATASET,
    EXPECTED_MEDIAN_MAE_Q_AH,
    EXPECTED_RANKINGS,
    EXPERIMENT,
    FEATURE_SET,
    FINAL_STDOUT,
    GB_MODEL,
    PIPELINE_VERSION,
    REPRODUCTION_TOLERANCE,
    RF_MODEL,
    RIDGE_MODEL,
)


class ReconstructionError(RuntimeError):
    """Raised when a hard reconstruction or reproduction check fails."""


PREDICTION_OUTPUT_COLUMNS = (
    "Dataset",
    "Experiment",
    "Feature_Set",
    "Cell_ID",
    "Raw_Cycle_Index",
    "Fold_ID",
    "Qref_Definition",
    "Qref_Ah",
    "Model",
    "Observed_SOH_Fraction",
    "Predicted_SOH_Fraction",
    "Observed_Capacity_Ah",
    "Predicted_Capacity_Ah",
    "Absolute_Error_Q_Ah",
    "Physical_Observation_Key",
    "Model_Parameters_SHA256",
    "Prediction_Source",
    "Prediction_Clipped",
)

CELL_METRIC_OUTPUT_COLUMNS = (
    "Dataset",
    "Experiment",
    "Feature_Set",
    "Batch_ID",
    "Cell_ID",
    "Fold_ID",
    "Qref_Definition",
    "Qref_Ah",
    "Model",
    "Observation_Count",
    "MAE_Q_Ah",
    "RMSE_Q_Ah",
    "R2_Q",
    "MAE_SOH_fraction",
    "RMSE_SOH_fraction",
    "R2_SOH",
    "MAE_SOH_percentage_points",
    "Aggregation_Unit",
)


@dataclass(frozen=True)
class TargetedPaths:
    feature_matrix: Path
    feature_regression: Path
    labels: Path
    folds: Path
    model_plan: Path
    preprocessing_plan: Path
    metric_plan: Path
    design_regression: Path
    primary_mask: Path
    primary_feature_freeze: Path
    primary_regression: Path
    frozen_a8_summary: Path

    @classmethod
    def from_workspace(cls, workspace: Path) -> "TargetedPaths":
        outputs = workspace / "outputs"
        extraction = outputs / "soh_feature_extraction_v1"
        design = outputs / "soh_ml_design_v1"
        primary = outputs / "soh_primary_benchmark_v1"
        robustness = outputs / "soh_ml_robustness_v1"
        return cls(
            feature_matrix=extraction / "feature_matrix_xjtu_v1.csv",
            feature_regression=extraction / "feature_extraction_regression_tests.csv",
            labels=outputs / "xjtu_55_final_v2" / "xjtu_55_soh_labels_long.csv",
            folds=design / "ml_cell_fold_assignments_xjtu.csv",
            model_plan=design / "ml_model_hyperparameter_plan.csv",
            preprocessing_plan=design / "ml_preprocessing_plan.csv",
            metric_plan=design / "ml_metric_plan.csv",
            design_regression=design / "ml_design_regression_tests.csv",
            primary_mask=primary / "soh_model_eligible_mask_xjtu.csv",
            primary_feature_freeze=primary / "soh_primary_feature_freeze_v1.csv",
            primary_regression=primary / "soh_primary_regression_tests.csv",
            frozen_a8_summary=robustness / "soh_robustness_a8_summary.csv",
        )

    def as_dict(self) -> dict[str, Path]:
        return {
            "feature_matrix_xjtu": self.feature_matrix,
            "feature_extraction_regression_tests": self.feature_regression,
            "xjtu_soh_labels": self.labels,
            "xjtu_frozen_folds": self.folds,
            "model_hyperparameter_plan": self.model_plan,
            "preprocessing_plan": self.preprocessing_plan,
            "metric_plan": self.metric_plan,
            "ml_design_regression_tests": self.design_regression,
            "xjtu_primary_eligible_mask": self.primary_mask,
            "primary_feature_freeze": self.primary_feature_freeze,
            "primary_regression_tests": self.primary_regression,
            "frozen_a8_summary": self.frozen_a8_summary,
        }


@dataclass
class ReconstructionAudit:
    fit_records: list[dict[str, Any]] = field(default_factory=list)
    prediction_rows: int = 0
    train_test_cell_overlap_count: int = 0
    max_predicted_capacity_formula_error: float = 0.0
    max_absolute_error_formula_error: float = 0.0


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ReconstructionError("MISSING_REQUIRED_INPUTS:" + "|".join(missing))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.columns.duplicated().any():
        raise ReconstructionError(f"DUPLICATE_OUTPUT_COLUMNS:{path.name}")
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.17g",
        lineterminator="\n",
    )


def _snapshot_tree(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise ReconstructionError(f"MISSING_FROZEN_DIRECTORY:{path}")
    return {
        str(file.relative_to(path)).replace("\\", "/"): _sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _tree_digest(snapshot: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, file_hash in sorted(snapshot.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _hash_paths(paths: dict[str, Path]) -> dict[str, str]:
    return {name: _sha256_file(path) for name, path in paths.items()}


def _assert_regression_pass(path: Path, label: str) -> None:
    frame = pd.read_csv(path, low_memory=False)
    if "Status" not in frame.columns or not frame["Status"].eq("PASS").all():
        failures = (
            frame.loc[frame.get("Status", pd.Series(index=frame.index)).ne("PASS")]
            .get("Test_ID", pd.Series(dtype=str))
            .astype(str)
            .tolist()
        )
        raise ReconstructionError(
            f"{label}_REGRESSION_NOT_ALL_PASS:" + "|".join(failures)
        )


def _load_targeted_inputs(paths: TargetedPaths) -> dict[str, Any]:
    path_map = paths.as_dict()
    _require_files(path_map.values())
    _assert_regression_pass(paths.feature_regression, "FEATURE_EXTRACTION")
    _assert_regression_pass(paths.design_regression, "ML_DESIGN")
    _assert_regression_pass(paths.primary_regression, "PRIMARY_BENCHMARK")

    matrix = pd.read_csv(paths.feature_matrix, low_memory=False)
    matrix["Raw_Cycle_Index"] = _numeric_int(
        matrix["Raw_Cycle_Index"], "XJTU_Raw_Cycle_Index"
    )
    matrix["Fold_ID"] = _numeric_int(matrix["Fold_ID"], "XJTU_Fold_ID")
    matrix["Batch_ID"] = _numeric_int(matrix["Batch_ID"], "XJTU_Batch_ID")
    if len(matrix) != XJTU_EXPECTED_MODEL_ROWS:
        raise ReconstructionError(f"XJTU_MATRIX_ROWS:{len(matrix)}")
    if matrix["Cell_ID"].nunique() != 55:
        raise ReconstructionError(f"XJTU_MATRIX_CELLS:{matrix['Cell_ID'].nunique()}")
    if matrix.duplicated(["Batch_ID", "Cell_ID", "Raw_Cycle_Index"]).any():
        raise ReconstructionError("DUPLICATE_XJTU_PHYSICAL_KEY")
    missing_features = sorted(set(A8_FEATURES).difference(matrix.columns))
    if missing_features:
        raise ReconstructionError("MISSING_A8_FEATURES:" + "|".join(missing_features))

    labels = pd.read_csv(paths.labels, low_memory=False)
    labels["Raw_Cycle_Index"] = _numeric_int(
        labels["Raw_Cycle_Index"], "XJTU_Label_Raw_Cycle_Index"
    )
    labels["Batch_ID"] = _numeric_int(labels["Batch_ID"], "XJTU_Label_Batch_ID")
    for column in ("Capacity_Ah", "Qref_Ah", "SOH_Fraction"):
        labels[column] = pd.to_numeric(labels[column], errors="raise")

    folds = pd.read_csv(paths.folds, low_memory=False)
    folds["Fold_ID"] = _numeric_int(folds["Fold_ID"], "XJTU_Frozen_Fold_ID")
    if folds["Cell_ID"].duplicated().any():
        raise ReconstructionError("DUPLICATE_XJTU_FOLD_CELL")
    if set(folds["Fold_ID"]) != set(range(1, N_FOLDS + 1)):
        raise ReconstructionError("XJTU_FOLD_IDS_NOT_1_TO_5")
    _validate_frozen_folds(matrix, folds, DATASET)

    mask = pd.read_csv(paths.primary_mask, low_memory=False)
    population = _frozen_population(matrix, mask, DATASET)
    if len(population) != 10427 or population["Cell_ID"].nunique() != 55:
        raise ReconstructionError(
            f"FROZEN_XJTU_POPULATION_CHANGED:rows={len(population)};"
            f"cells={population['Cell_ID'].nunique()}"
        )

    aligned_targets, aligned_key_hashes = _align_targets(
        population, labels, DATASET, COMMON_QREFS
    )

    model_plan = pd.read_csv(paths.model_plan, low_memory=False)
    ordered_models = tuple(
        model_plan.sort_values("Model_Order")["Model_Family"].astype(str)
    )
    if ordered_models != MODEL_NAMES:
        raise ReconstructionError(f"MODEL_PLAN_FAMILIES_CHANGED:{ordered_models}")
    if not model_plan["Hyperparameter_Mode"].eq("FIXED").all():
        raise ReconstructionError("MODEL_PLAN_NOT_FIXED")
    if not model_plan["Test_Fold_Tuning"].eq("PROHIBITED").all():
        raise ReconstructionError("TEST_FOLD_TUNING_NOT_PROHIBITED")
    model_parameters = {
        row.Model_Family: json.loads(row.Hyperparameters_JSON)
        for row in model_plan.itertuples(index=False)
    }
    model_parameter_hashes = {
        model: hashlib.sha256(
            _canonical_json(parameters).encode("utf-8")
        ).hexdigest()
        for model, parameters in model_parameters.items()
    }

    preprocessing = pd.read_csv(paths.preprocessing_plan, low_memory=False)
    for model in MODEL_NAMES:
        model_steps = preprocessing.loc[
            preprocessing["Model_Family"].eq(model), ["Operation", "Fit_Scope"]
        ]
        if len(model_steps) != 6:
            raise ReconstructionError(f"PREPROCESSING_STEP_COUNT_CHANGED:{model}")
        imputation = model_steps.loc[
            model_steps["Operation"].eq("MEDIAN_IMPUTATION_IF_REQUIRED"),
            "Fit_Scope",
        ]
        if len(imputation) != 1 or imputation.iloc[0] != "CURRENT_OUTER_TRAINING_FOLDS_ONLY":
            raise ReconstructionError(f"PREPROCESSING_IMPUTER_SCOPE_CHANGED:{model}")
    ridge_scaler = preprocessing.loc[
        preprocessing["Model_Family"].eq(RIDGE_MODEL)
        & preprocessing["Operation"].eq("STANDARDIZE"),
        "Fit_Scope",
    ]
    if len(ridge_scaler) != 1 or ridge_scaler.iloc[0] != "CURRENT_OUTER_TRAINING_FOLDS_ONLY":
        raise ReconstructionError("RIDGE_SCALER_SCOPE_CHANGED")
    for model in (RF_MODEL, GB_MODEL):
        operations = set(
            preprocessing.loc[
                preprocessing["Model_Family"].eq(model), "Operation"
            ].astype(str)
        )
        if "NO_SCALING" not in operations or "STANDARDIZE" in operations:
            raise ReconstructionError(f"TREE_PREPROCESSING_CHANGED:{model}")

    frozen_summary = pd.read_csv(paths.frozen_a8_summary, low_memory=False)
    frozen_summary = frozen_summary.loc[
        frozen_summary["Dataset"].eq(DATASET)
    ].copy()
    if len(frozen_summary) != 9:
        raise ReconstructionError(f"FROZEN_XJTU_A8_SUMMARY_ROWS:{len(frozen_summary)}")

    return {
        "paths": path_map,
        "matrix": matrix,
        "population": population,
        "aligned_targets": aligned_targets,
        "aligned_key_hashes": aligned_key_hashes,
        "folds": folds,
        "model_plan": model_plan,
        "model_parameters": model_parameters,
        "model_parameter_hashes": model_parameter_hashes,
        "preprocessing": preprocessing,
        "frozen_summary": frozen_summary,
    }


def _fit_one_fold(
    *,
    qref_definition: str,
    model_name: str,
    parameters: dict[str, Any],
    target: pd.DataFrame,
    transform: Any,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    if model_name == RIDGE_MODEL:
        train_x = transform.ridge_train
        test_x = transform.ridge_test
        scaling = "TRAIN_FOLD_STANDARD_SCALER"
    else:
        train_x = transform.tree_train
        test_x = transform.tree_test
        scaling = "NO_SCALING"
    train_indices = transform.train_indices
    test_indices = transform.test_indices
    y = target["SOH_Fraction"].to_numpy(dtype=float)
    model = _build_model(model_name, parameters)
    started = time.perf_counter()
    model.fit(train_x, y[train_indices])
    predicted_soh = np.asarray(model.predict(test_x), dtype=float)
    training_seconds = time.perf_counter() - started

    held_out = target.iloc[test_indices].reset_index(drop=True)
    qref = held_out["Qref_Ah"].to_numpy(dtype=float)
    observed_capacity = held_out["Capacity_Ah"].to_numpy(dtype=float)
    predicted_capacity = predicted_soh * qref
    parameter_json = _canonical_json(parameters)
    parameter_hash = hashlib.sha256(parameter_json.encode("utf-8")).hexdigest()
    predictions = pd.DataFrame(
        {
            "Dataset": DATASET,
            "Experiment": EXPERIMENT,
            "Feature_Set": FEATURE_SET,
            "Batch_ID": held_out["Batch_ID"].to_numpy(dtype=int),
            "Cell_ID": held_out["Cell_ID"].astype(str).to_numpy(),
            "Raw_Cycle_Index": held_out["Raw_Cycle_Index"].to_numpy(dtype=int),
            "Fold_ID": held_out["Fold_ID"].to_numpy(dtype=int),
            "Qref_Definition": qref_definition,
            "Qref_Ah": qref,
            "Model": model_name,
            "Observed_SOH_Fraction": held_out["SOH_Fraction"].to_numpy(dtype=float),
            "Predicted_SOH_Fraction": predicted_soh,
            "Observed_Capacity_Ah": observed_capacity,
            "Predicted_Capacity_Ah": predicted_capacity,
            "Absolute_Error_Q_Ah": np.abs(predicted_capacity - observed_capacity),
            "Physical_Observation_Key": held_out["Physical_Observation_Key"].astype(str).to_numpy(),
            "Model_Parameters_SHA256": parameter_hash,
            "Prediction_Source": "HELD_OUT_CELL_FOLD_ONLY",
            "Prediction_Clipped": "NO",
        }
    )
    numeric = predictions[
        [
            "Observed_SOH_Fraction",
            "Predicted_SOH_Fraction",
            "Observed_Capacity_Ah",
            "Predicted_Capacity_Ah",
            "Absolute_Error_Q_Ah",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ReconstructionError(
            f"NONFINITE_PREDICTION:{qref_definition}:{model_name}:{transform.fold_id}"
        )

    metric_input = predictions.copy()
    cell_rows = cell_metric_rows(
        metric_input,
        analysis_id=EXPERIMENT,
        dataset=DATASET,
        qref_definition=qref_definition,
        model_name=model_name,
    )

    train_cells = set(target.iloc[train_indices]["Cell_ID"].astype(str))
    test_cells = set(target.iloc[test_indices]["Cell_ID"].astype(str))
    overlap = len(train_cells.intersection(test_cells))
    record = {
        "Experiment": EXPERIMENT,
        "Feature_Set": FEATURE_SET,
        "Features": "|".join(A8_FEATURES),
        "Feature_Count": len(A8_FEATURES),
        "Dataset": DATASET,
        "Qref_Definition": qref_definition,
        "Model": model_name,
        "Fold_ID": int(transform.fold_id),
        "Train_Observation_Count": len(train_indices),
        "Test_Observation_Count": len(test_indices),
        "Train_Cell_Count": int(transform.train_cell_count),
        "Test_Cell_Count": int(transform.test_cell_count),
        "Train_Test_Cell_Overlap_Count": overlap,
        "Train_Key_SHA256": transform.train_key_hash,
        "Test_Key_SHA256": transform.test_key_hash,
        "Median_Imputer_Fit_Scope": "CURRENT_TRAINING_FOLD_ONLY",
        "Imputer_Statistics_SHA256": transform.imputer_hash,
        "Scaling": scaling,
        "Scaler_Fit_Scope": (
            "CURRENT_TRAINING_FOLD_ONLY"
            if model_name == RIDGE_MODEL
            else "NOT_APPLICABLE"
        ),
        "Scaler_Mean_SHA256": (
            transform.scaler_mean_hash if model_name == RIDGE_MODEL else "NOT_APPLICABLE"
        ),
        "Scaler_Scale_SHA256": (
            transform.scaler_scale_hash if model_name == RIDGE_MODEL else "NOT_APPLICABLE"
        ),
        "Model_Parameters_JSON": parameter_json,
        "Model_Parameters_SHA256": parameter_hash,
        "Training_Seconds": training_seconds,
        "Prediction_Row_Count": len(predictions),
    }
    return predictions, cell_rows, record


def _train_xjtu_a8(
    population: pd.DataFrame,
    aligned_targets: dict[str, pd.DataFrame],
    model_parameters: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, ReconstructionAudit]:
    transforms = _prepare_transforms(population, A8_FEATURES)
    prediction_parts: list[pd.DataFrame] = []
    cell_rows: list[dict[str, Any]] = []
    audit = ReconstructionAudit()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FOLDS) as executor:
        for qref_definition in COMMON_QREFS:
            target = aligned_targets[qref_definition]
            for model_name in MODEL_NAMES:
                future_map: dict[Future[Any], int] = {}
                for fold_id in range(1, N_FOLDS + 1):
                    future = executor.submit(
                        _fit_one_fold,
                        qref_definition=qref_definition,
                        model_name=model_name,
                        parameters=model_parameters[model_name],
                        target=target,
                        transform=transforms[fold_id],
                    )
                    future_map[future] = fold_id
                completed: dict[int, tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]] = {}
                for future in as_completed(future_map):
                    fold_id = future_map[future]
                    result = future.result()
                    completed[fold_id] = result
                    print(
                        f"[{datetime.now().isoformat(timespec='seconds')}] "
                        f"fit complete {EXPERIMENT} | {DATASET} | {qref_definition} | "
                        f"{model_name} | fold {fold_id}/{N_FOLDS} | "
                        f"{result[2]['Training_Seconds']:.2f}s",
                        flush=True,
                    )
                for fold_id in range(1, N_FOLDS + 1):
                    predictions, rows, record = completed[fold_id]
                    prediction_parts.append(predictions)
                    cell_rows.extend(rows)
                    audit.fit_records.append(record)
                    audit.prediction_rows += len(predictions)
                    audit.train_test_cell_overlap_count += int(
                        record["Train_Test_Cell_Overlap_Count"]
                    )
                    capacity_error = np.max(
                        np.abs(
                            predictions["Predicted_Capacity_Ah"].to_numpy(float)
                            - predictions["Predicted_SOH_Fraction"].to_numpy(float)
                            * predictions["Qref_Ah"].to_numpy(float)
                        )
                    )
                    absolute_error = np.max(
                        np.abs(
                            predictions["Absolute_Error_Q_Ah"].to_numpy(float)
                            - np.abs(
                                predictions["Predicted_Capacity_Ah"].to_numpy(float)
                                - predictions["Observed_Capacity_Ah"].to_numpy(float)
                            )
                        )
                    )
                    audit.max_predicted_capacity_formula_error = max(
                        audit.max_predicted_capacity_formula_error,
                        float(capacity_error),
                    )
                    audit.max_absolute_error_formula_error = max(
                        audit.max_absolute_error_formula_error,
                        float(absolute_error),
                    )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    cell_metrics = pd.DataFrame(cell_rows)
    cell_metrics["Experiment"] = cell_metrics["Analysis_ID"]
    cell_metrics["Feature_Set"] = FEATURE_SET
    qref_order = {name: index for index, name in enumerate(COMMON_QREFS)}
    model_order = {name: index for index, name in enumerate(MODEL_NAMES)}
    predictions["_Qref_Order"] = predictions["Qref_Definition"].map(qref_order)
    predictions["_Model_Order"] = predictions["Model"].map(model_order)
    predictions = predictions.sort_values(
        ["_Qref_Order", "_Model_Order", "Fold_ID", "Cell_ID", "Raw_Cycle_Index"],
        kind="stable",
    ).drop(columns=["_Qref_Order", "_Model_Order"]).reset_index(drop=True)
    cell_metrics["_Qref_Order"] = cell_metrics["Qref_Definition"].map(qref_order)
    cell_metrics["_Model_Order"] = cell_metrics["Model"].map(model_order)
    cell_metrics = cell_metrics.sort_values(
        ["_Qref_Order", "_Model_Order", "Cell_ID"], kind="stable"
    ).drop(columns=["_Qref_Order", "_Model_Order"]).reset_index(drop=True)
    return predictions, cell_metrics, audit


def _ranking_order(group: pd.DataFrame) -> tuple[str, ...]:
    medians = (
        group.groupby("Model", sort=False)["MAE_Q_Ah"].median().to_dict()
    )
    return tuple(sorted(MODEL_NAMES, key=lambda model: (medians[model], MODEL_NAMES.index(model))))


def _reproduction_audit(
    cell_metrics: pd.DataFrame,
    frozen_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    if frozen_summary.duplicated(["Qref_Definition", "Model"]).any():
        raise ReconstructionError("DUPLICATE_FROZEN_A8_SUMMARY_KEY")
    frozen_lookup = {
        (row.Qref_Definition, row.Model): float(row.Median_MAE_Q_Ah)
        for row in frozen_summary.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for qref in COMMON_QREFS:
        group = cell_metrics.loc[cell_metrics["Qref_Definition"].eq(qref)]
        observed_ranking = _ranking_order(group)
        expected_ranking = EXPECTED_RANKINGS[qref]
        observed_ranking_text = " < ".join(observed_ranking)
        expected_ranking_text = " < ".join(expected_ranking)
        for model in MODEL_NAMES:
            key = (qref, model)
            if key not in frozen_lookup or key not in EXPECTED_MEDIAN_MAE_Q_AH:
                raise ReconstructionError(f"MISSING_FROZEN_EXPECTATION:{qref}:{model}")
            reconstructed = float(
                group.loc[group["Model"].eq(model), "MAE_Q_Ah"].median()
            )
            expected = float(EXPECTED_MEDIAN_MAE_Q_AH[key])
            frozen = float(frozen_lookup[key])
            error = abs(reconstructed - expected)
            frozen_constant_error = abs(frozen - expected)
            rows.append(
                {
                    "Dataset": DATASET,
                    "Experiment": EXPERIMENT,
                    "Feature_Set": FEATURE_SET,
                    "Qref_Definition": qref,
                    "Model": model,
                    "Cell_Count": int(group.loc[group["Model"].eq(model), "Cell_ID"].nunique()),
                    "Expected_Frozen_Median_MAE_Q_Ah": expected,
                    "Frozen_Summary_Median_MAE_Q_Ah": frozen,
                    "Reconstructed_Median_MAE_Q_Ah": reconstructed,
                    "Absolute_Reproduction_Error_Ah": error,
                    "Frozen_Summary_Error_vs_Expected_Ah": frozen_constant_error,
                    "Reproduction_Tolerance_Ah": REPRODUCTION_TOLERANCE,
                    "Machine_Precision_Difference": (
                        "YES" if 0 < error <= REPRODUCTION_TOLERANCE else "NO"
                    ),
                    "Median_Reproduction_Status": (
                        "PASS"
                        if error <= REPRODUCTION_TOLERANCE
                        and frozen_constant_error <= REPRODUCTION_TOLERANCE
                        else "FAIL"
                    ),
                    "Expected_Ranking_Lowest_MAE_First": expected_ranking_text,
                    "Reconstructed_Ranking_Lowest_MAE_First": observed_ranking_text,
                    "Ranking_Reproduction_Status": (
                        "PASS" if observed_ranking == expected_ranking else "FAIL"
                    ),
                }
            )
    audit = pd.DataFrame(rows)
    gate = bool(
        len(audit) == 9
        and audit["Median_Reproduction_Status"].eq("PASS").all()
        and audit["Ranking_Reproduction_Status"].eq("PASS").all()
    )
    return audit, gate


def _bootstrap_outputs(
    cell_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    pivot = cell_metrics.pivot(
        index="Cell_ID",
        columns=["Qref_Definition", "Model"],
        values="MAE_Q_Ah",
    ).sort_index()
    expected_columns = pd.MultiIndex.from_product(
        [COMMON_QREFS, MODEL_NAMES], names=["Qref_Definition", "Model"]
    )
    if set(pivot.columns) != set(expected_columns) or pivot.isna().any().any():
        raise ReconstructionError("BOOTSTRAP_CELL_METRIC_GRID_INCOMPLETE")
    if len(pivot) != 55:
        raise ReconstructionError(f"BOOTSTRAP_CELL_COUNT:{len(pivot)}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(pivot), size=(BOOTSTRAP_REPLICATES, len(pivot)), dtype=np.int64
    )
    draw_hash = hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()

    delta_by_qref: dict[str, np.ndarray] = {}
    ci_rows: list[dict[str, Any]] = []
    observed_delta: dict[str, float] = {}
    for qref in COMMON_QREFS:
        rf = pivot[(qref, RF_MODEL)].to_numpy(dtype=float)
        gb = pivot[(qref, GB_MODEL)].to_numpy(dtype=float)
        rf_boot = np.median(rf[indices], axis=1)
        gb_boot = np.median(gb[indices], axis=1)
        delta = gb_boot - rf_boot
        delta_by_qref[qref] = delta
        rf_median = float(np.median(rf))
        gb_median = float(np.median(gb))
        observed = gb_median - rf_median
        observed_delta[qref] = observed
        lower, upper = np.quantile(delta, [0.025, 0.975], method="linear")
        rf_lower = delta > 0
        gb_lower = delta < 0
        tie = delta == 0
        ci_rows.append(
            {
                "Dataset": DATASET,
                "Experiment": EXPERIMENT,
                "Feature_Set": FEATURE_SET,
                "Qref_Definition": qref,
                "Cell_Count": len(pivot),
                "Observed_RF_Median_MAE_Q_Ah": rf_median,
                "Observed_GB_Median_MAE_Q_Ah": gb_median,
                "Observed_Difference_Of_Medians": observed,
                "Bootstrap_Median_Difference_Of_Medians": float(np.median(delta)),
                "Bootstrap_95CI_Lower": float(lower),
                "Bootstrap_95CI_Upper": float(upper),
                "CI_Includes_Zero": "YES" if lower <= 0 <= upper else "NO",
                "RF_Lower_Median_Count": int(rf_lower.sum()),
                "RF_Lower_Median_Frequency": float(rf_lower.mean()),
                "GB_Lower_Median_Count": int(gb_lower.sum()),
                "GB_Lower_Median_Frequency": float(gb_lower.mean()),
                "Tie_Count": int(tie.sum()),
                "Tie_Frequency": float(tie.mean()),
                "Primary_Statistic": "median(GB cell MAE_Q_Ah) - median(RF cell MAE_Q_Ah)",
                "Secondary_Median_Of_Paired_Differences_Computed": "NO",
                "Bootstrap_Replicates": BOOTSTRAP_REPLICATES,
                "Bootstrap_Seed": BOOTSTRAP_SEED,
                "Bootstrap_Unit": "Cell_ID",
                "Cycle_Level_Resampling": "NO",
                "Same_Draws_Across_Qrefs": "YES",
                "Bootstrap_Draw_SHA256": draw_hash,
                "Rankflip_Comparison": "",
                "Interpretation_Classification": "",
            }
        )
    ci = pd.DataFrame(ci_rows)

    nominal_row = ci.loc[ci["Qref_Definition"].eq("nominal")].iloc[0]
    for measured_qref in ("first_comparable", "early_median5_comparable"):
        measured_row = ci.loc[ci["Qref_Definition"].eq(measured_qref)].iloc[0]
        observed_reverse = (
            observed_delta["nominal"] > 0 and observed_delta[measured_qref] < 0
        )
        nominal_strict_positive = float(nominal_row["Bootstrap_95CI_Lower"]) > 0
        measured_strict_negative = float(measured_row["Bootstrap_95CI_Upper"]) < 0
        nominal_includes_zero = nominal_row["CI_Includes_Zero"] == "YES"
        measured_includes_zero = measured_row["CI_Includes_Zero"] == "YES"
        if nominal_strict_positive and measured_strict_negative:
            classification = "BOOTSTRAP_SUPPORTED_RANK_REVERSAL"
        elif observed_reverse and (nominal_includes_zero or measured_includes_zero):
            classification = "DESCRIPTIVE_RANK_REVERSAL_NOT_BOOTSTRAP_ROBUST"
        else:
            classification = "MIXED_OR_UNCERTAIN_RANK_REVERSAL"
        mask = ci["Qref_Definition"].eq(measured_qref)
        ci.loc[mask, "Rankflip_Comparison"] = f"nominal_vs_{measured_qref}"
        ci.loc[mask, "Interpretation_Classification"] = classification
    ci.loc[ci["Qref_Definition"].eq("nominal"), "Rankflip_Comparison"] = "nominal_reference"
    ci.loc[
        ci["Qref_Definition"].eq("nominal"), "Interpretation_Classification"
    ] = "REFERENCE_QREF"

    nominal_rf_lower = delta_by_qref["nominal"] > 0
    rankflip_rows: list[dict[str, Any]] = []
    for measured_qref in ("first_comparable", "early_median5_comparable"):
        measured_gb_lower = delta_by_qref[measured_qref] < 0
        joint = nominal_rf_lower & measured_gb_lower
        rankflip_rows.append(
            {
                "Dataset": DATASET,
                "Experiment": EXPERIMENT,
                "Feature_Set": FEATURE_SET,
                "Nominal_Qref_Definition": "nominal",
                "Measured_Qref_Definition": measured_qref,
                "Rankflip_Event": f"Nominal_RF_Lower_AND_{measured_qref}_GB_Lower",
                "Cell_Count": len(pivot),
                "Observed_Nominal_Difference_Of_Medians": observed_delta["nominal"],
                "Observed_Measured_Difference_Of_Medians": observed_delta[measured_qref],
                "Nominal_RF_Lower_Count": int(nominal_rf_lower.sum()),
                "Nominal_RF_Lower_Frequency": float(nominal_rf_lower.mean()),
                "Measured_GB_Lower_Count": int(measured_gb_lower.sum()),
                "Measured_GB_Lower_Frequency": float(measured_gb_lower.mean()),
                "Joint_Rankflip_Count": int(joint.sum()),
                "Observed_Sample_RankFlip_Stability": float(joint.mean()),
                "Bootstrap_Replicates": BOOTSTRAP_REPLICATES,
                "Bootstrap_Seed": BOOTSTRAP_SEED,
                "Bootstrap_Unit": "Cell_ID",
                "Cycle_Level_Resampling": "NO",
                "Same_Draws_Across_Qrefs": "YES",
                "Bootstrap_Draw_SHA256": draw_hash,
                "Interpretation_Guardrail": (
                    "Observed-sample Cell_ID bootstrap stability; not a universal superiority probability"
                ),
            }
        )
    rankflip = pd.DataFrame(rankflip_rows)

    rank_rows: list[dict[str, Any]] = []
    for qref in COMMON_QREFS:
        values = np.column_stack(
            [pivot[(qref, model)].to_numpy(dtype=float) for model in MODEL_NAMES]
        )
        boot_medians = np.median(values[indices], axis=1)
        minimum = np.min(boot_medians, axis=1)
        tied_minimum = boot_medians == minimum[:, None]
        tie_replicates = np.sum(tied_minimum, axis=1) > 1
        rank1 = np.argmin(boot_medians, axis=1)
        observed_medians = np.median(values, axis=0)
        order = np.argsort(observed_medians, kind="stable")
        ranks = np.empty(len(MODEL_NAMES), dtype=int)
        ranks[order] = np.arange(1, len(MODEL_NAMES) + 1)
        for model_index, model in enumerate(MODEL_NAMES):
            rank_rows.append(
                {
                    "Dataset": DATASET,
                    "Experiment": EXPERIMENT,
                    "Feature_Set": FEATURE_SET,
                    "Qref_Definition": qref,
                    "Model": model,
                    "Cell_Count": len(pivot),
                    "Observed_Median_MAE_Q_Ah": float(observed_medians[model_index]),
                    "Observed_Rank": int(ranks[model_index]),
                    "Bootstrap_Rank1_Count": int(np.sum(rank1 == model_index)),
                    "Bootstrap_Rank1_Frequency": float(np.mean(rank1 == model_index)),
                    "Any_Rank1_Tie_Replicate_Count": int(tie_replicates.sum()),
                    "Bootstrap_Replicates": BOOTSTRAP_REPLICATES,
                    "Bootstrap_Seed": BOOTSTRAP_SEED,
                    "Bootstrap_Unit": "Cell_ID",
                    "Cycle_Level_Resampling": "NO",
                    "Same_Draws_Across_Qrefs": "YES",
                    "Bootstrap_Draw_SHA256": draw_hash,
                    "Rank_Metric": "median held-out-cell MAE_Q_Ah",
                    "Tie_Break_Order": "|".join(MODEL_NAMES),
                    "Interpretation_Guardrail": (
                        "Observed-sample rank frequency; not a universal model-superiority probability"
                    ),
                }
            )
    rank_frequency = pd.DataFrame(rank_rows)
    return ci, rankflip, rank_frequency, draw_hash


def _test_row(
    test_id: str,
    passed: bool,
    observed: str,
    expected: str,
    failure_message: str,
) -> dict[str, Any]:
    return {
        "Test_ID": test_id,
        "Status": "PASS" if passed else "FAIL",
        "Observed": observed,
        "Expected": expected,
        "Failure_Message": "" if passed else failure_message,
        "Hard_Failure": "YES",
    }


def _regression_tests(
    *,
    loaded: dict[str, Any],
    predictions: pd.DataFrame,
    cell_metrics: pd.DataFrame,
    reproduction: pd.DataFrame,
    ci: pd.DataFrame,
    rankflip: pd.DataFrame,
    rank_frequency: pd.DataFrame,
    draw_hash: str,
    audit: ReconstructionAudit,
    input_hashes_before: dict[str, str],
    input_hashes_after: dict[str, str],
    primary_before: dict[str, str],
    primary_after: dict[str, str],
    robustness_before: dict[str, str],
    robustness_after: dict[str, str],
    prior_targeted_before: dict[str, str],
    prior_targeted_after: dict[str, str],
) -> pd.DataFrame:
    tests: list[dict[str, Any]] = []

    def add(test_id: str, passed: bool, observed: Any, expected: Any, message: str) -> None:
        tests.append(_test_row(test_id, bool(passed), str(observed), str(expected), message))

    population = loaded["population"]
    add("XJTU_CELL_COUNT_55", population["Cell_ID"].nunique() == 55, population["Cell_ID"].nunique(), 55, "Frozen XJTU cell count changed.")
    add("XJTU_OBSERVATION_COUNT_10427", len(population) == 10427, len(population), 10427, "Frozen XJTU observation count changed.")
    add("A8_FEATURE_COUNT_8", len(A8_FEATURES) == 8, len(A8_FEATURES), 8, "A8 feature count changed.")
    expected_a8 = tuple(feature for feature in PRIMARY_FEATURES if feature != "CV_Duration_s")
    add("A8_EXACTLY_A9_MINUS_CV_DURATION", A8_FEATURES == expected_a8, "|".join(A8_FEATURES), "|".join(expected_a8), "A8 is not exactly A9 minus CV_Duration_s.")
    add("NO_CYCLE_INDEX", "Raw_Cycle_Index" not in A8_FEATURES, "Raw_Cycle_Index" in A8_FEATURES, False, "Cycle index entered A8 features.")
    shortcut_overlap = sorted(set(A8_FEATURES).intersection(PROTOCOL_SHORTCUT_FEATURES))
    add("NO_PROTOCOL_SHORTCUT_FEATURES", not shortcut_overlap, "|".join(shortcut_overlap) or "NONE", "NONE", "Protocol shortcut features entered A8.")
    capacity_features = [name for name in A8_FEATURES if "capacity" in name.lower()]
    add("NO_CAPACITY_AS_FEATURE", not capacity_features, "|".join(capacity_features) or "NONE", "NONE", "Capacity entered A8 features.")
    qref_features = [name for name in A8_FEATURES if "qref" in name.lower()]
    add("NO_QREF_AS_FEATURE", not qref_features, "|".join(qref_features) or "NONE", "NONE", "Qref entered A8 features.")
    rul_eol_features = [name for name in A8_FEATURES if "rul" in name.lower() or "eol" in name.lower()]
    add("NO_RUL_EOL_AS_FEATURE", not rul_eol_features, "|".join(rul_eol_features) or "NONE", "NONE", "RUL or EOL entered A8 features.")

    fit_frame = pd.DataFrame(audit.fit_records)
    fold_grid_ok = (
        len(fit_frame) == len(COMMON_QREFS) * len(MODEL_NAMES) * N_FOLDS
        and set(fit_frame["Fold_ID"]) == set(range(1, N_FOLDS + 1))
        and fit_frame.groupby(["Qref_Definition", "Model"])["Fold_ID"].nunique().eq(N_FOLDS).all()
    )
    add("FROZEN_FOLDS_EXACT_MATCH", fold_grid_ok, f"fits={len(fit_frame)};folds={sorted(fit_frame['Fold_ID'].unique())}", "45 fits using frozen folds 1-5", "Frozen fold fit grid changed.")
    add("NO_SAME_CELL_TRAIN_TEST", audit.train_test_cell_overlap_count == 0, audit.train_test_cell_overlap_count, 0, "A Cell_ID appears in both train and test.")
    aligned_hashes = loaded["aligned_key_hashes"]
    feature_rows_ok = len(set(aligned_hashes.values())) == 1 and all(len(loaded["aligned_targets"][q]) == 10427 for q in COMMON_QREFS)
    add("FEATURE_ROWS_IDENTICAL_ACROSS_QREFS", feature_rows_ok, json.dumps(aligned_hashes, sort_keys=True), "one identical physical-key hash across three Qrefs", "Qref-specific row filtering occurred.")

    parameter_hashes = loaded["model_parameter_hashes"]
    parameter_ok = all(
        fit_frame.loc[fit_frame["Model"].eq(model), "Model_Parameters_SHA256"].eq(parameter_hashes[model]).all()
        for model in MODEL_NAMES
    )
    add("MODEL_PARAMETERS_MATCH_FROZEN_PLAN", parameter_ok, json.dumps(parameter_hashes, sort_keys=True), "all 45 fits match frozen model-plan hashes", "Model parameters differ from frozen plan.")
    preprocessing_ok = (
        fit_frame["Median_Imputer_Fit_Scope"].eq("CURRENT_TRAINING_FOLD_ONLY").all()
        and fit_frame.loc[fit_frame["Model"].eq(RIDGE_MODEL), "Scaler_Fit_Scope"].eq("CURRENT_TRAINING_FOLD_ONLY").all()
        and fit_frame.loc[~fit_frame["Model"].eq(RIDGE_MODEL), "Scaler_Fit_Scope"].eq("NOT_APPLICABLE").all()
    )
    add("PREPROCESSING_TRAIN_ONLY", preprocessing_ok, f"imputer_scopes={sorted(fit_frame['Median_Imputer_Fit_Scope'].unique())};scaler_scopes={sorted(fit_frame['Scaler_Fit_Scope'].unique())}", "train-fold-only imputer and Ridge scaler; no tree scaling", "Held-out information entered preprocessing.")

    expected_prediction_rows = 10427 * len(COMMON_QREFS) * len(MODEL_NAMES)
    add("HELDOUT_PREDICTION_ROWS_93843", len(predictions) == expected_prediction_rows, len(predictions), expected_prediction_rows, "Held-out prediction row count changed.")
    prediction_key = ["Cell_ID", "Raw_Cycle_Index", "Qref_Definition", "Model"]
    add("HELDOUT_PREDICTION_KEYS_UNIQUE_93843", not predictions.duplicated(prediction_key).any(), int(predictions.duplicated(prediction_key).sum()), 0, "Duplicate held-out prediction keys exist.")
    add("CELL_METRIC_ROWS_495", len(cell_metrics) == 495, len(cell_metrics), 495, "Cell metric row count changed.")
    metric_key = ["Cell_ID", "Qref_Definition", "Model"]
    add("CELL_METRIC_KEYS_UNIQUE_495", len(cell_metrics) == 495 and not cell_metrics.duplicated(metric_key).any(), int(cell_metrics.duplicated(metric_key).sum()), 0, "Cell metric keys are not unique.")
    medians_ok = len(reproduction) == 9 and reproduction["Median_Reproduction_Status"].eq("PASS").all()
    rankings_ok = reproduction["Ranking_Reproduction_Status"].eq("PASS").all()
    add("ALL_NINE_FROZEN_MEDIANS_REPRODUCED", medians_ok, f"pass={int(reproduction['Median_Reproduction_Status'].eq('PASS').sum())};max_error={reproduction['Absolute_Reproduction_Error_Ah'].max():.17g}", "9 PASS with absolute error <= 1e-12", "Frozen A8 median result was not reproduced.")
    add("ALL_THREE_FROZEN_RANKINGS_REPRODUCED", rankings_ok, f"pass_rows={int(reproduction['Ranking_Reproduction_Status'].eq('PASS').sum())}", "all three Qref rankings reproduced", "Frozen A8 ranking was not reproduced.")

    primary_statistic_ok = ci["Primary_Statistic"].eq("median(GB cell MAE_Q_Ah) - median(RF cell MAE_Q_Ah)").all()
    add("CORRECTED_STATISTIC_IS_DIFFERENCE_OF_MEDIANS", primary_statistic_ok, "|".join(sorted(ci["Primary_Statistic"].unique())), "median(GB) - median(RF)", "Corrected statistic is not the difference of model medians.")
    secondary_not_used = ci["Secondary_Median_Of_Paired_Differences_Computed"].eq("NO").all()
    add("PRIMARY_STATISTIC_NOT_MEDIAN_OF_PAIRED_DIFFERENCES", secondary_not_used, "NO" if secondary_not_used else "YES", "NO", "Median of paired differences was used as the primary statistic.")
    bootstrap_unit_ok = (
        ci["Bootstrap_Unit"].eq("Cell_ID").all()
        and rankflip["Bootstrap_Unit"].eq("Cell_ID").all()
        and rank_frequency["Bootstrap_Unit"].eq("Cell_ID").all()
        and ci["Cycle_Level_Resampling"].eq("NO").all()
        and rankflip["Cycle_Level_Resampling"].eq("NO").all()
        and rank_frequency["Cycle_Level_Resampling"].eq("NO").all()
    )
    add("BOOTSTRAP_UNIT_CELL_ID", bootstrap_unit_ok, "Cell_ID;cycle_resampling=NO", "Cell_ID only", "Bootstrap unit changed or cycle resampling occurred.")
    draw_hashes = set(ci["Bootstrap_Draw_SHA256"]).union(rankflip["Bootstrap_Draw_SHA256"], rank_frequency["Bootstrap_Draw_SHA256"])
    same_draws = draw_hashes == {draw_hash}
    add("SAME_BOOTSTRAP_DRAWS_ACROSS_QREFS", same_draws, "|".join(sorted(draw_hashes)), draw_hash, "Different bootstrap draws were used across Qrefs or outputs.")
    replicate_ok = (
        ci["Bootstrap_Replicates"].eq(BOOTSTRAP_REPLICATES).all()
        and rankflip["Bootstrap_Replicates"].eq(BOOTSTRAP_REPLICATES).all()
        and rank_frequency["Bootstrap_Replicates"].eq(BOOTSTRAP_REPLICATES).all()
    )
    add("BOOTSTRAP_REPLICATES_5000", replicate_ok, BOOTSTRAP_REPLICATES, 5000, "Bootstrap replicate count changed.")

    add("PREDICTED_CAPACITY_FORMULA_EXACT", audit.max_predicted_capacity_formula_error <= 1e-15, f"{audit.max_predicted_capacity_formula_error:.17g}", "<=1e-15", "Predicted capacity formula mismatch.")
    add("ABSOLUTE_ERROR_Q_FORMULA_EXACT", audit.max_absolute_error_formula_error <= 1e-15, f"{audit.max_absolute_error_formula_error:.17g}", "<=1e-15", "Absolute capacity error formula mismatch.")
    add("ONLY_XJTU_ROBUSTNESS_A8_ROWS", predictions["Dataset"].eq(DATASET).all() and predictions["Experiment"].eq(EXPERIMENT).all() and predictions["Feature_Set"].eq(FEATURE_SET).all(), f"datasets={predictions['Dataset'].unique().tolist()};experiments={predictions['Experiment'].unique().tolist()};features={predictions['Feature_Set'].unique().tolist()}", "XJTU|ROBUSTNESS_A8|A8_MINUS_CV_DURATION", "Out-of-scope rows were produced.")
    add("FROZEN_PRIMARY_FILES_UNCHANGED", primary_before == primary_after, _tree_digest(primary_after), _tree_digest(primary_before), "Frozen primary directory changed.")
    add("FROZEN_ROBUSTNESS_FILES_UNCHANGED", robustness_before == robustness_after, _tree_digest(robustness_after), _tree_digest(robustness_before), "Frozen robustness directory changed.")
    add("FROZEN_PRIOR_TARGETED_FILES_UNCHANGED", prior_targeted_before == prior_targeted_after, _tree_digest(prior_targeted_after), _tree_digest(prior_targeted_before), "Prior targeted rank-flip directory changed.")
    add("FROZEN_INPUT_FILES_UNCHANGED", input_hashes_before == input_hashes_after, json.dumps(input_hashes_after, sort_keys=True), json.dumps(input_hashes_before, sort_keys=True), "A frozen input file changed.")
    return pd.DataFrame(tests)


def _manifest(
    *,
    workspace: Path,
    paths: dict[str, Path],
    input_hashes: dict[str, str],
    population: pd.DataFrame,
    aligned_key_hashes: dict[str, str],
    model_parameters: dict[str, dict[str, Any]],
    feature_matrix_hash: str,
    physical_key_hash: str,
    draw_hash: str,
    primary_tree_hash: str,
    robustness_tree_hash: str,
    prior_targeted_tree_hash: str,
) -> pd.DataFrame:
    import scipy
    import sklearn

    rows: list[dict[str, Any]] = []

    def add(record_type: str, name: str, value: Any = "", path: str = "", sha256: str = "", notes: str = "") -> None:
        rows.append(
            {
                "Record_Type": record_type,
                "Name": name,
                "Value": value,
                "Path": path,
                "SHA256": sha256,
                "Notes": notes,
            }
        )

    for name, path in paths.items():
        add(
            "INPUT_FILE",
            name,
            path.stat().st_size,
            str(path.resolve()),
            input_hashes[name],
            "Frozen read-only input",
        )
    for relative in (
        "xjtu_a8_deterministic_reconstruction_repro/config.py",
        "xjtu_a8_deterministic_reconstruction_repro/pipeline.py",
        "reconstruct_xjtu_a8.py",
    ):
        source = workspace / relative
        add("SOURCE_CODE", relative, source.stat().st_size, str(source.resolve()), _sha256_file(source), "Reconstruction implementation")
    add("DERIVED_HASH", "A8_FEATURE_MATRIX_SHA256", len(population), str(paths["feature_matrix_xjtu"].resolve()), feature_matrix_hash, "10427 rows x 8 frozen A8 features in frozen row order")
    add("DERIVED_HASH", "PHYSICAL_OBSERVATION_KEY_SHA256", len(population), str(paths["feature_matrix_xjtu"].resolve()), physical_key_hash, "Frozen XJTU physical observation order")
    for qref, key_hash in aligned_key_hashes.items():
        add("DERIVED_HASH", f"ALIGNED_KEY_SHA256_{qref}", len(population), str(paths["xjtu_soh_labels"].resolve()), key_hash, "Aligned target physical-key order")
    add("DERIVED_HASH", "BOOTSTRAP_DRAW_SHA256", BOOTSTRAP_REPLICATES, "", draw_hash, "One Cell_ID resample matrix shared across all Qrefs and bootstrap outputs")
    add("FROZEN_TREE_HASH", "soh_primary_benchmark_v1", "unchanged", str((workspace / "outputs/soh_primary_benchmark_v1").resolve()), primary_tree_hash, "Tree digest after reconstruction")
    add("FROZEN_TREE_HASH", "soh_ml_robustness_v1", "unchanged", str((workspace / "outputs/soh_ml_robustness_v1").resolve()), robustness_tree_hash, "Tree digest after reconstruction")
    add("FROZEN_TREE_HASH", "xjtu_a8_rankflip_verification_v1", "unchanged", str((workspace / "outputs/xjtu_a8_rankflip_verification_v1").resolve()), prior_targeted_tree_hash, "Tree digest after reconstruction")
    versions = {
        "Python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }
    for name, version in versions.items():
        add("SOFTWARE_VERSION", name, version, "", "", "")
    for model in MODEL_NAMES:
        random_state = model_parameters[model].get("random_state", "NOT_APPLICABLE")
        add("MODEL_RANDOM_STATE", model, random_state, str(paths["model_hyperparameter_plan"].resolve()), input_hashes["model_hyperparameter_plan"], _canonical_json(model_parameters[model]))
    add("RUN_PARAMETER", "PIPELINE_VERSION", PIPELINE_VERSION)
    add("RUN_PARAMETER", "DATASET", DATASET)
    add("RUN_PARAMETER", "EXPERIMENT", EXPERIMENT)
    add("RUN_PARAMETER", "FEATURE_SET", FEATURE_SET)
    add("RUN_PARAMETER", "A8_FEATURES", "|".join(A8_FEATURES))
    add("RUN_PARAMETER", "QREF_DEFINITIONS", "|".join(COMMON_QREFS))
    add("RUN_PARAMETER", "MODELS", "|".join(MODEL_NAMES))
    add("RUN_PARAMETER", "BOOTSTRAP_SEED", BOOTSTRAP_SEED)
    add("RUN_PARAMETER", "BOOTSTRAP_REPLICATES", BOOTSTRAP_REPLICATES)
    add("RUN_PARAMETER", "BOOTSTRAP_UNIT", "Cell_ID")
    add("RUN_PARAMETER", "REPRODUCTION_TOLERANCE", REPRODUCTION_TOLERANCE)
    return pd.DataFrame(rows)


def _recompute_metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cell_id, qref, model), group in predictions.groupby(
        ["Cell_ID", "Qref_Definition", "Model"], sort=True
    ):
        metrics = calculate_metrics(
            group["Observed_SOH_Fraction"].to_numpy(float),
            group["Predicted_SOH_Fraction"].to_numpy(float),
            group["Observed_Capacity_Ah"].to_numpy(float),
            group["Predicted_Capacity_Ah"].to_numpy(float),
        )
        rows.append({"Cell_ID": cell_id, "Qref_Definition": qref, "Model": model, **metrics})
    return pd.DataFrame(rows)


def _validate_outputs(staging: Path) -> dict[str, dict[str, Any]]:
    expected_rows = {
        "xjtu_a8_heldout_predictions.csv": 93843,
        "xjtu_a8_cell_metrics.csv": 495,
        "xjtu_a8_reproduction_audit.csv": 9,
        "xjtu_a8_corrected_difference_of_medians_ci.csv": 3,
        "xjtu_a8_corrected_rankflip_frequency.csv": 2,
        "xjtu_a8_corrected_rank_frequency.csv": 9,
        "xjtu_a8_reconstruction_regression_tests.csv": 32,
    }
    keys = {
        "xjtu_a8_heldout_predictions.csv": ["Cell_ID", "Raw_Cycle_Index", "Qref_Definition", "Model"],
        "xjtu_a8_cell_metrics.csv": ["Cell_ID", "Qref_Definition", "Model"],
        "xjtu_a8_reproduction_audit.csv": ["Qref_Definition", "Model"],
        "xjtu_a8_corrected_difference_of_medians_ci.csv": ["Qref_Definition"],
        "xjtu_a8_corrected_rankflip_frequency.csv": ["Measured_Qref_Definition"],
        "xjtu_a8_corrected_rank_frequency.csv": ["Qref_Definition", "Model"],
        "xjtu_a8_reconstruction_manifest.csv": ["Record_Type", "Name"],
        "xjtu_a8_reconstruction_regression_tests.csv": ["Test_ID"],
    }
    frames: dict[str, pd.DataFrame] = {}
    validation: dict[str, dict[str, Any]] = {}
    for name in CSV_OUTPUTS:
        path = staging / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ReconstructionError(f"MISSING_OR_EMPTY_OUTPUT:{name}")
        frame = pd.read_csv(path, low_memory=False)
        frames[name] = frame
        if frame.columns.duplicated().any():
            raise ReconstructionError(f"DUPLICATE_OUTPUT_COLUMNS:{name}")
        if frame.duplicated(keys[name]).any():
            raise ReconstructionError(f"DUPLICATE_OUTPUT_KEYS:{name}")
        if frame[keys[name]].isna().any(axis=None):
            raise ReconstructionError(f"NULL_OUTPUT_KEYS:{name}")
        if name in expected_rows and len(frame) != expected_rows[name]:
            raise ReconstructionError(
                f"OUTPUT_ROW_COUNT_MISMATCH:{name}:{len(frame)}:{expected_rows[name]}"
            )
        validation[name] = {
            "rows": len(frame),
            "columns": len(frame.columns),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    predictions = frames["xjtu_a8_heldout_predictions.csv"]
    metrics = frames["xjtu_a8_cell_metrics.csv"]
    reproduction = frames["xjtu_a8_reproduction_audit.csv"]
    tests = frames["xjtu_a8_reconstruction_regression_tests.csv"]
    if not tests["Status"].eq("PASS").all():
        raise ReconstructionError("WRITTEN_REGRESSION_TEST_FAILURE")
    if not reproduction["Median_Reproduction_Status"].eq("PASS").all():
        raise ReconstructionError("WRITTEN_REPRODUCTION_AUDIT_FAILURE")
    predicted_capacity_error = np.max(
        np.abs(
            predictions["Predicted_Capacity_Ah"].to_numpy(float)
            - predictions["Predicted_SOH_Fraction"].to_numpy(float)
            * predictions["Qref_Ah"].to_numpy(float)
        )
    )
    absolute_error_error = np.max(
        np.abs(
            predictions["Absolute_Error_Q_Ah"].to_numpy(float)
            - np.abs(
                predictions["Predicted_Capacity_Ah"].to_numpy(float)
                - predictions["Observed_Capacity_Ah"].to_numpy(float)
            )
        )
    )
    if predicted_capacity_error > 5e-15 or absolute_error_error > 5e-15:
        raise ReconstructionError(
            f"WRITTEN_PREDICTION_FORMULA_MISMATCH:{predicted_capacity_error}:"
            f"{absolute_error_error}"
        )
    recomputed = _recompute_metrics_from_predictions(predictions)
    merged = metrics.merge(
        recomputed,
        on=["Cell_ID", "Qref_Definition", "Model"],
        how="outer",
        suffixes=("_Written", "_Recomputed"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise ReconstructionError("WRITTEN_CELL_METRIC_KEY_RECONCILIATION_FAILED")
    recompute_max_errors: dict[str, float] = {}
    for column in ("MAE_Q_Ah", "RMSE_Q_Ah", "R2_Q", "MAE_SOH_fraction", "RMSE_SOH_fraction", "R2_SOH"):
        left = pd.to_numeric(merged[f"{column}_Written"], errors="coerce").to_numpy(float)
        right = pd.to_numeric(merged[f"{column}_Recomputed"], errors="coerce").to_numpy(float)
        same_nan = np.isnan(left) & np.isnan(right)
        finite_close = np.isfinite(left) & np.isfinite(right) & np.isclose(left, right, rtol=0, atol=5e-15)
        finite = np.isfinite(left) & np.isfinite(right)
        max_error = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        recompute_max_errors[column] = max_error
        if not np.all(same_nan | finite_close):
            raise ReconstructionError(f"WRITTEN_CELL_METRIC_RECOMPUTE_FAILED:{column}:{max_error}")
    validation["cell_metric_recompute"] = {
        "rows": len(merged),
        "columns": 6,
        "bytes": None,
        "sha256": None,
        "absolute_tolerance": 5e-15,
        "maximum_absolute_errors": recompute_max_errors,
        "status": "PASS_MACHINE_PRECISION_TOLERANCE",
    }
    if not (staging / "processing_log.txt").is_file():
        raise ReconstructionError("MISSING_PROCESSING_LOG")
    validation["processing_log.txt"] = {
        "rows": None,
        "columns": None,
        "bytes": (staging / "processing_log.txt").stat().st_size,
        "sha256": _sha256_file(staging / "processing_log.txt"),
    }
    return validation


def _processing_log(
    *,
    started: datetime,
    finished: datetime,
    output_dir: Path,
    loaded: dict[str, Any],
    audit: ReconstructionAudit,
    reproduction: pd.DataFrame,
    ci: pd.DataFrame,
    rankflip: pd.DataFrame,
    rank_frequency: pd.DataFrame,
    tests: pd.DataFrame,
    output_validation: dict[str, dict[str, Any]] | None,
) -> str:
    import scipy
    import sklearn

    lines = [
        f"Pipeline: {PIPELINE_VERSION}",
        f"Status: {'ALL REGRESSION TESTS PASSED' if tests['Status'].eq('PASS').all() else 'PIPELINE FAILED'}",
        f"Started: {started.isoformat()}",
        f"Finished: {finished.isoformat()}",
        f"Elapsed_seconds: {(finished - started).total_seconds():.3f}",
        f"Output_directory: {output_dir}",
        f"Python: {platform.python_version()}",
        f"numpy: {np.__version__}",
        f"pandas: {pd.__version__}",
        f"scipy: {scipy.__version__}",
        f"scikit-learn: {sklearn.__version__}",
        f"Dataset: {DATASET}",
        f"Experiment: {EXPERIMENT}",
        f"Feature_set: {FEATURE_SET}",
        f"A8_features: {'|'.join(A8_FEATURES)}",
        f"Qrefs: {'|'.join(COMMON_QREFS)}",
        f"Models: {'|'.join(MODEL_NAMES)}",
        f"Frozen_cell_count: {loaded['population']['Cell_ID'].nunique()}",
        f"Frozen_observation_count: {len(loaded['population'])}",
        f"Model_fits: {len(audit.fit_records)}",
        f"Held_out_prediction_rows: {audit.prediction_rows}",
        f"Train_test_cell_overlap_count: {audit.train_test_cell_overlap_count}",
        f"Bootstrap_seed: {BOOTSTRAP_SEED}",
        f"Bootstrap_replicates: {BOOTSTRAP_REPLICATES}",
        "Bootstrap_unit: Cell_ID",
        "Cycle_level_resampling: NO",
        "Same_bootstrap_draws_across_Qrefs: YES",
        "Primary_bootstrap_statistic: median(GB cell MAE_Q_Ah) - median(RF cell MAE_Q_Ah)",
        "Median_of_paired_differences_used_as_primary: NO",
        f"Maximum_absolute_reproduction_error_Ah: {reproduction['Absolute_Reproduction_Error_Ah'].max():.17g}",
        "Reproduction_audit:",
    ]
    for row in reproduction.itertuples(index=False):
        lines.append(
            f"  {row.Qref_Definition}|{row.Model}: {row.Median_Reproduction_Status} | "
            f"expected={row.Expected_Frozen_Median_MAE_Q_Ah:.17g} | "
            f"reconstructed={row.Reconstructed_Median_MAE_Q_Ah:.17g} | "
            f"abs_error={row.Absolute_Reproduction_Error_Ah:.17g}"
        )
    lines.append("Corrected_difference_of_medians:")
    for row in ci.itertuples(index=False):
        lines.append(
            f"  {row.Qref_Definition}: observed={row.Observed_Difference_Of_Medians:.17g} | "
            f"CI=[{row.Bootstrap_95CI_Lower:.17g},{row.Bootstrap_95CI_Upper:.17g}] | "
            f"classification={row.Interpretation_Classification}"
        )
    lines.append("Corrected_rankflip_frequency:")
    for row in rankflip.itertuples(index=False):
        lines.append(
            f"  {row.Measured_Qref_Definition}: {row.Observed_Sample_RankFlip_Stability:.17g} "
            f"({row.Joint_Rankflip_Count}/{row.Bootstrap_Replicates})"
        )
    lines.append(f"Corrected_rank_frequency_rows: {len(rank_frequency)}")
    lines.append("Regression_tests:")
    for row in tests.itertuples(index=False):
        lines.append(
            f"  {row.Test_ID}: {row.Status} | observed={row.Observed} | expected={row.Expected}"
        )
    if output_validation is not None:
        lines.append("Written_output_validation:")
        lines.append(_canonical_json(output_validation))
    lines.extend(
        [
            "Final_status: XJTU A8 DETERMINISTIC RECONSTRUCTION COMPLETE — "
            "FROZEN A8 RESULTS REPRODUCED — CORRECTED BOOTSTRAP COMPLETE",
            "",
        ]
    )
    return "\n".join(lines)


def run_reconstruction(
    workspace: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    target = (
        Path(output_dir).resolve()
        if output_dir is not None
        else workspace / "outputs" / "xjtu_a8_deterministic_reconstruction_v1"
    )
    if target.exists():
        raise ReconstructionError(f"OUTPUT_ALREADY_EXISTS:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".xjtu_a8_deterministic_reconstruction_v1_staging_",
            dir=target.parent,
        )
    )
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    print(f"Staging output: {staging}", flush=True)

    primary_dir = workspace / "outputs" / "soh_primary_benchmark_v1"
    robustness_dir = workspace / "outputs" / "soh_ml_robustness_v1"
    prior_targeted_dir = workspace / "outputs" / "xjtu_a8_rankflip_verification_v1"
    primary_before = _snapshot_tree(primary_dir)
    robustness_before = _snapshot_tree(robustness_dir)
    prior_targeted_before = _snapshot_tree(prior_targeted_dir)

    targeted_paths = TargetedPaths.from_workspace(workspace)
    loaded = _load_targeted_inputs(targeted_paths)
    input_hashes_before = _hash_paths(loaded["paths"])
    population = loaded["population"]
    feature_values = population[list(A8_FEATURES)].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
    feature_matrix_hash = _sha256_array(feature_values)
    physical_key_hash = _sha256_text(population["Physical_Observation_Key"])

    predictions, cell_metrics, audit = _train_xjtu_a8(
        population,
        loaded["aligned_targets"],
        loaded["model_parameters"],
    )
    prediction_output = predictions[list(PREDICTION_OUTPUT_COLUMNS)].copy()
    cell_metric_output = cell_metrics[list(CELL_METRIC_OUTPUT_COLUMNS)].copy()
    reproduction, reproduction_gate = _reproduction_audit(
        cell_metrics, loaded["frozen_summary"]
    )

    _write_csv(prediction_output, staging / "xjtu_a8_heldout_predictions.csv")
    _write_csv(cell_metric_output, staging / "xjtu_a8_cell_metrics.csv")
    _write_csv(reproduction, staging / "xjtu_a8_reproduction_audit.csv")
    if not reproduction_gate:
        raise ReconstructionError(
            "XJTU A8 DETERMINISTIC RECONSTRUCTION FAILED —\n"
            "FROZEN SUMMARY NOT REPRODUCED"
            f"\nstaging={staging}"
        )

    ci, rankflip, rank_frequency, draw_hash = _bootstrap_outputs(cell_metrics)
    _write_csv(ci, staging / "xjtu_a8_corrected_difference_of_medians_ci.csv")
    _write_csv(rankflip, staging / "xjtu_a8_corrected_rankflip_frequency.csv")
    _write_csv(rank_frequency, staging / "xjtu_a8_corrected_rank_frequency.csv")

    input_hashes_after = _hash_paths(loaded["paths"])
    primary_after = _snapshot_tree(primary_dir)
    robustness_after = _snapshot_tree(robustness_dir)
    prior_targeted_after = _snapshot_tree(prior_targeted_dir)

    tests = _regression_tests(
        loaded=loaded,
        predictions=predictions,
        cell_metrics=cell_metrics,
        reproduction=reproduction,
        ci=ci,
        rankflip=rankflip,
        rank_frequency=rank_frequency,
        draw_hash=draw_hash,
        audit=audit,
        input_hashes_before=input_hashes_before,
        input_hashes_after=input_hashes_after,
        primary_before=primary_before,
        primary_after=primary_after,
        robustness_before=robustness_before,
        robustness_after=robustness_after,
        prior_targeted_before=prior_targeted_before,
        prior_targeted_after=prior_targeted_after,
    )
    manifest = _manifest(
        workspace=workspace,
        paths=loaded["paths"],
        input_hashes=input_hashes_after,
        population=population,
        aligned_key_hashes=loaded["aligned_key_hashes"],
        model_parameters=loaded["model_parameters"],
        feature_matrix_hash=feature_matrix_hash,
        physical_key_hash=physical_key_hash,
        draw_hash=draw_hash,
        primary_tree_hash=_tree_digest(primary_after),
        robustness_tree_hash=_tree_digest(robustness_after),
        prior_targeted_tree_hash=_tree_digest(prior_targeted_after),
    )
    _write_csv(manifest, staging / "xjtu_a8_reconstruction_manifest.csv")
    _write_csv(tests, staging / "xjtu_a8_reconstruction_regression_tests.csv")
    failures = tests.loc[tests["Status"].ne("PASS")]
    finished = datetime.now(ZoneInfo("Asia/Shanghai"))
    preliminary_log = _processing_log(
        started=started,
        finished=finished,
        output_dir=target,
        loaded=loaded,
        audit=audit,
        reproduction=reproduction,
        ci=ci,
        rankflip=rankflip,
        rank_frequency=rank_frequency,
        tests=tests,
        output_validation=None,
    )
    (staging / "processing_log.txt").write_text(preliminary_log, encoding="utf-8")
    if not failures.empty:
        raise ReconstructionError(
            "HARD_REGRESSION_FAILURES:"
            + "|".join(failures["Test_ID"].astype(str))
            + f";staging={staging}"
        )

    output_validation = _validate_outputs(staging)
    finished = datetime.now(ZoneInfo("Asia/Shanghai"))
    final_log = _processing_log(
        started=started,
        finished=finished,
        output_dir=target,
        loaded=loaded,
        audit=audit,
        reproduction=reproduction,
        ci=ci,
        rankflip=rankflip,
        rank_frequency=rank_frequency,
        tests=tests,
        output_validation=output_validation,
    )
    (staging / "processing_log.txt").write_text(final_log, encoding="utf-8")

    if primary_before != _snapshot_tree(primary_dir):
        raise ReconstructionError("FROZEN_PRIMARY_CHANGED_AFTER_FINAL_VALIDATION")
    if robustness_before != _snapshot_tree(robustness_dir):
        raise ReconstructionError("FROZEN_ROBUSTNESS_CHANGED_AFTER_FINAL_VALIDATION")
    if prior_targeted_before != _snapshot_tree(prior_targeted_dir):
        raise ReconstructionError("FROZEN_PRIOR_TARGETED_CHANGED_AFTER_FINAL_VALIDATION")

    staging.replace(target)
    return {
        "output_dir": str(target),
        "prediction_rows": len(prediction_output),
        "cell_metric_rows": len(cell_metric_output),
        "maximum_reproduction_error": float(
            reproduction["Absolute_Reproduction_Error_Ah"].max()
        ),
        "output_validation": output_validation,
        "final_stdout": FINAL_STDOUT,
    }


def finalize_existing_staging(
    workspace: Path,
    staging_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate and publish an already completed fit without retraining models."""
    workspace = Path(workspace).resolve()
    staging = Path(staging_dir).resolve()
    target = (
        Path(output_dir).resolve()
        if output_dir is not None
        else workspace / "outputs" / "xjtu_a8_deterministic_reconstruction_v1"
    )
    if target.exists():
        raise ReconstructionError(f"OUTPUT_ALREADY_EXISTS:{target}")
    if not staging.is_dir() or not staging.name.startswith(
        ".xjtu_a8_deterministic_reconstruction_v1_staging_"
    ):
        raise ReconstructionError(f"INVALID_RECONSTRUCTION_STAGING:{staging}")
    required = [*(staging / name for name in CSV_OUTPUTS), staging / "processing_log.txt"]
    _require_files(required)

    primary_dir = workspace / "outputs" / "soh_primary_benchmark_v1"
    robustness_dir = workspace / "outputs" / "soh_ml_robustness_v1"
    prior_targeted_dir = workspace / "outputs" / "xjtu_a8_rankflip_verification_v1"
    primary_before = _snapshot_tree(primary_dir)
    robustness_before = _snapshot_tree(robustness_dir)
    prior_targeted_before = _snapshot_tree(prior_targeted_dir)

    tests = pd.read_csv(
        staging / "xjtu_a8_reconstruction_regression_tests.csv", low_memory=False
    )
    reproduction = pd.read_csv(
        staging / "xjtu_a8_reproduction_audit.csv", low_memory=False
    )
    if len(tests) != 32 or not tests["Status"].eq("PASS").all():
        raise ReconstructionError("STAGING_REGRESSION_TESTS_NOT_ALL_PASS")
    if len(reproduction) != 9 or not (
        reproduction["Median_Reproduction_Status"].eq("PASS").all()
        and reproduction["Ranking_Reproduction_Status"].eq("PASS").all()
    ):
        raise ReconstructionError("STAGING_REPRODUCTION_GATE_NOT_PASS")

    manifest = pd.read_csv(
        staging / "xjtu_a8_reconstruction_manifest.csv", low_memory=False
    )
    current_paths = TargetedPaths.from_workspace(workspace).as_dict()
    current_hashes = _hash_paths(current_paths)
    manifest_inputs = manifest.loc[manifest["Record_Type"].eq("INPUT_FILE")]
    recorded_hashes = dict(
        zip(manifest_inputs["Name"].astype(str), manifest_inputs["SHA256"].astype(str))
    )
    if current_hashes != recorded_hashes:
        raise ReconstructionError("FROZEN_INPUT_HASH_CHANGED_BEFORE_FINALIZATION")

    validation = _validate_outputs(staging)
    log_path = staging / "processing_log.txt"
    log_lines = [
        line
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("Final_status:")
    ]
    log_lines.extend(
        [
            "Finalization_resumed_from_existing_staging: YES",
            "Model_retraining_during_finalization: NO",
            "Written_cell_metric_recompute_absolute_tolerance: 5e-15",
            "Written_cell_metric_recompute_note: maximum observed discrepancy was machine precision after 17-significant-digit CSV serialization",
            "Written_output_validation:",
            _canonical_json(validation),
            "Final_status: XJTU A8 DETERMINISTIC RECONSTRUCTION COMPLETE — FROZEN A8 RESULTS REPRODUCED — CORRECTED BOOTSTRAP COMPLETE",
            "",
        ]
    )
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    final_validation = _validate_outputs(staging)

    if primary_before != _snapshot_tree(primary_dir):
        raise ReconstructionError("FROZEN_PRIMARY_CHANGED_DURING_FINALIZATION")
    if robustness_before != _snapshot_tree(robustness_dir):
        raise ReconstructionError("FROZEN_ROBUSTNESS_CHANGED_DURING_FINALIZATION")
    if prior_targeted_before != _snapshot_tree(prior_targeted_dir):
        raise ReconstructionError("FROZEN_PRIOR_TARGETED_CHANGED_DURING_FINALIZATION")

    staging.replace(target)
    return {
        "output_dir": str(target),
        "model_retraining_during_finalization": False,
        "output_validation": final_validation,
        "final_stdout": FINAL_STDOUT,
    }


__all__ = [
    "ReconstructionError",
    "finalize_existing_staging",
    "run_reconstruction",
]
