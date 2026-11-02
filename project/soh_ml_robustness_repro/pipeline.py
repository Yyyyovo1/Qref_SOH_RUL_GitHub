"""Frozen SOH robustness experiments and cell-level statistical uncertainty."""

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

from soh_primary_benchmark_repro.config import (
    COMMON_QREFS,
    MODEL_NAMES,
    N_FOLDS,
    PRIMARY_FEATURES,
    PROTOCOL_SHORTCUT_FEATURES,
    QREF_PAIRS,
)
from soh_primary_benchmark_repro.metrics import (
    cell_metric_rows,
    summarize_cell_metrics,
)
from soh_primary_benchmark_repro.pipeline import (
    Paths as PrimaryPaths,
    _align_targets,
    _build_model,
    _canonical_json,
    _load_inputs,
    _sha256_array,
    _sha256_file,
    _sha256_text,
    _validate_frozen_folds,
    _write_csv,
)

from .config import (
    A8_FEATURES,
    B10_FEATURES,
    BOOTSTRAP_REPLICATES,
    CSV_OUTPUTS,
    EXPERIMENTS,
    MAX_PARALLEL_FOLDS,
    MODEL_PAIRS,
    PIPELINE_VERSION,
    POSTCYCLE5_FEATURES,
    RANDOM_SEED,
)


class RobustnessError(RuntimeError):
    pass


@dataclass(frozen=True)
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
    train_key_hash: str
    test_key_hash: str
    imputer_hash: str
    scaler_mean_hash: str
    scaler_scale_hash: str


@dataclass
class TrainingAudit:
    fit_records: list[dict[str, Any]] = field(default_factory=list)
    prediction_rows: int = 0
    train_test_cell_overlap: int = 0
    population_key_hashes: dict[str, dict[str, dict[str, str]]] = field(
        default_factory=dict
    )
    population_feature_hashes: dict[str, dict[str, str]] = field(
        default_factory=dict
    )


def _snapshot_tree(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise RobustnessError(f"MISSING_FROZEN_PRIMARY_DIRECTORY:{path}")
    return {
        str(file.relative_to(path)).replace("\\", "/"): _sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _required_primary_outputs(primary_dir: Path) -> dict[str, Path]:
    paths = {
        "feature_freeze": primary_dir / "soh_primary_feature_freeze_v1.csv",
        "mask_xjtu": primary_dir / "soh_model_eligible_mask_xjtu.csv",
        "mask_hust": primary_dir / "soh_model_eligible_mask_hust.csv",
        "cell_metrics": primary_dir / "soh_primary_cell_metrics.csv",
        "summary_metrics": primary_dir / "soh_primary_summary_metrics.csv",
        "rankings": primary_dir / "soh_primary_model_rankings.csv",
        "regression_tests": primary_dir / "soh_primary_regression_tests.csv",
        "processing_log": primary_dir / "processing_log.txt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RobustnessError("MISSING_PRIMARY_V1_FILES:" + "|".join(missing))
    return paths


def _load_primary_outputs(primary_dir: Path) -> dict[str, Any]:
    paths = _required_primary_outputs(primary_dir)
    regression = pd.read_csv(paths["regression_tests"], low_memory=False)
    if not regression["Status"].eq("PASS").all():
        failed = regression.loc[regression["Status"].ne("PASS"), "Test_ID"]
        raise RobustnessError(
            "PRIMARY_V1_REGRESSION_NOT_ALL_PASS:" + "|".join(failed.astype(str))
        )
    return {
        "paths": paths,
        "feature_freeze": pd.read_csv(paths["feature_freeze"], low_memory=False),
        "mask_xjtu": pd.read_csv(paths["mask_xjtu"], low_memory=False),
        "mask_hust": pd.read_csv(paths["mask_hust"], low_memory=False),
        "cell_metrics": pd.read_csv(paths["cell_metrics"], low_memory=False),
        "summary_metrics": pd.read_csv(paths["summary_metrics"], low_memory=False),
        "rankings": pd.read_csv(paths["rankings"], low_memory=False),
        "regression_tests": regression,
    }


def _frozen_population(
    matrix: pd.DataFrame,
    mask: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    if mask["Physical_Observation_Key"].duplicated().any():
        raise RobustnessError(f"{dataset}_DUPLICATE_PRIMARY_MASK_KEY")
    if matrix["Physical_Observation_Key"].duplicated().any():
        raise RobustnessError(f"{dataset}_DUPLICATE_MATRIX_KEY")
    check = matrix[["Physical_Observation_Key"]].merge(
        mask[["Physical_Observation_Key", "Model_Eligible"]],
        on="Physical_Observation_Key",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if len(check) != len(matrix) or not check["_merge"].eq("both").all():
        raise RobustnessError(f"{dataset}_PRIMARY_MASK_KEYS_CHANGED")
    eligible_keys = set(
        mask.loc[mask["Model_Eligible"].eq("YES"), "Physical_Observation_Key"].astype(
            str
        )
    )
    frozen = matrix.loc[
        matrix["Physical_Observation_Key"].astype(str).isin(eligible_keys)
    ].reset_index(drop=True)
    if len(frozen) != len(eligible_keys):
        raise RobustnessError(f"{dataset}_PRIMARY_MASK_APPLICATION_FAILED")
    return frozen


def _prepare_transforms(
    matrix: pd.DataFrame,
    features: tuple[str, ...],
) -> dict[int, FoldTransform]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RobustnessError("SCIKIT_LEARN_UNAVAILABLE") from exc
    missing = sorted(set(features).difference(matrix.columns))
    if missing:
        raise RobustnessError("MISSING_SENSITIVITY_FEATURES:" + "|".join(missing))
    values = matrix[list(features)].apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=float
    )
    transforms: dict[int, FoldTransform] = {}
    for fold_id in range(1, N_FOLDS + 1):
        test_indices = np.flatnonzero(matrix["Fold_ID"].to_numpy() == fold_id)
        train_indices = np.flatnonzero(matrix["Fold_ID"].to_numpy() != fold_id)
        train_cells = set(matrix.iloc[train_indices]["Cell_ID"].astype(str))
        test_cells = set(matrix.iloc[test_indices]["Cell_ID"].astype(str))
        overlap = train_cells.intersection(test_cells)
        if overlap:
            raise RobustnessError(
                f"CELL_LEAKAGE_FOLD_{fold_id}:" + "|".join(sorted(overlap))
            )
        imputer = SimpleImputer(strategy="median", add_indicator=False)
        tree_train = imputer.fit_transform(values[train_indices])
        tree_test = imputer.transform(values[test_indices])
        if tree_train.shape[1] != len(features):
            raise RobustnessError(f"IMPUTER_DROPPED_FEATURE_FOLD:{fold_id}")
        scaler = StandardScaler(with_mean=True, with_std=True)
        ridge_train = scaler.fit_transform(tree_train)
        ridge_test = scaler.transform(tree_test)
        if not all(
            np.isfinite(array).all()
            for array in (tree_train, tree_test, ridge_train, ridge_test)
        ):
            raise RobustnessError(f"NONFINITE_PREPROCESSED_FEATURES:{fold_id}")
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
            train_key_hash=_sha256_text(
                matrix.iloc[train_indices]["Physical_Observation_Key"]
            ),
            test_key_hash=_sha256_text(
                matrix.iloc[test_indices]["Physical_Observation_Key"]
            ),
            imputer_hash=_sha256_array(imputer.statistics_),
            scaler_mean_hash=_sha256_array(scaler.mean_),
            scaler_scale_hash=_sha256_array(scaler.scale_),
        )
    return transforms


def _fit_one_fold(
    *,
    analysis_id: str,
    feature_set: str,
    features: tuple[str, ...],
    dataset: str,
    qref_definition: str,
    model_name: str,
    parameters: dict[str, Any],
    target: pd.DataFrame,
    transform: FoldTransform,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    if model_name == "Ridge Regression":
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
    start = time.perf_counter()
    model.fit(train_x, y[train_indices])
    predicted_soh = np.asarray(model.predict(test_x), dtype=float)
    elapsed = time.perf_counter() - start
    held_out = target.iloc[test_indices].reset_index(drop=True)
    qref = held_out["Qref_Ah"].to_numpy(dtype=float)
    batch = (
        held_out["Batch_ID"].to_numpy()
        if "Batch_ID" in held_out
        else np.repeat("", len(held_out))
    )
    predictions = pd.DataFrame(
        {
            "Analysis_ID": analysis_id,
            "Dataset": dataset,
            "Batch_ID": batch,
            "Cell_ID": held_out["Cell_ID"].astype(str).to_numpy(),
            "Raw_Cycle_Index": held_out["Raw_Cycle_Index"].to_numpy(dtype=int),
            "Fold_ID": held_out["Fold_ID"].to_numpy(dtype=int),
            "Qref_Definition": qref_definition,
            "Qref_Ah": qref,
            "Observed_SOH_Fraction": held_out["SOH_Fraction"].to_numpy(dtype=float),
            "Predicted_SOH_Fraction": predicted_soh,
            "Observed_Capacity_Ah": held_out["Capacity_Ah"].to_numpy(dtype=float),
            "Predicted_Capacity_Ah": predicted_soh * qref,
        }
    )
    if not np.isfinite(
        predictions[
            ["Predicted_SOH_Fraction", "Predicted_Capacity_Ah"]
        ].to_numpy(dtype=float)
    ).all():
        raise RobustnessError(
            f"NONFINITE_PREDICTION:{analysis_id}:{dataset}:{qref_definition}:"
            f"{model_name}:{transform.fold_id}"
        )
    cell_rows = cell_metric_rows(
        predictions,
        analysis_id=analysis_id,
        dataset=dataset,
        qref_definition=qref_definition,
        model_name=model_name,
    )
    parameter_json = _canonical_json(parameters)
    parameter_hash = hashlib.sha256(parameter_json.encode("utf-8")).hexdigest()
    train_cells = set(target.iloc[train_indices]["Cell_ID"].astype(str))
    test_cells = set(target.iloc[test_indices]["Cell_ID"].astype(str))
    overlap = len(train_cells.intersection(test_cells))
    record = {
        "Analysis_ID": analysis_id,
        "Feature_Set": feature_set,
        "Features": "|".join(features),
        "Feature_Count": len(features),
        "Dataset": dataset,
        "Qref_Definition": qref_definition,
        "Model": model_name,
        "Fold_ID": transform.fold_id,
        "Train_Observation_Count": len(train_indices),
        "Test_Observation_Count": len(test_indices),
        "Train_Cell_Count": transform.train_cell_count,
        "Test_Cell_Count": transform.test_cell_count,
        "Train_Test_Cell_Overlap_Count": overlap,
        "Train_Key_SHA256": transform.train_key_hash,
        "Test_Key_SHA256": transform.test_key_hash,
        "Median_Imputer_Fit_Scope": "CURRENT_TRAINING_FOLD_ONLY",
        "Imputer_Statistics_SHA256": transform.imputer_hash,
        "Scaling": scaling,
        "Scaler_Fit_Scope": (
            "CURRENT_TRAINING_FOLD_ONLY"
            if model_name == "Ridge Regression"
            else "NOT_APPLICABLE"
        ),
        "Scaler_Mean_SHA256": (
            transform.scaler_mean_hash
            if model_name == "Ridge Regression"
            else "NOT_APPLICABLE"
        ),
        "Scaler_Scale_SHA256": (
            transform.scaler_scale_hash
            if model_name == "Ridge Regression"
            else "NOT_APPLICABLE"
        ),
        "Model_Parameters_JSON": parameter_json,
        "Model_Parameters_SHA256": parameter_hash,
        "Training_Seconds": elapsed,
    }
    return cell_rows, record, len(predictions)


def _train_population(
    *,
    analysis_id: str,
    feature_set: str,
    features: tuple[str, ...],
    dataset: str,
    matrix: pd.DataFrame,
    aligned_targets: dict[str, pd.DataFrame],
    model_parameters: dict[str, dict[str, Any]],
    audit: TrainingAudit,
) -> pd.DataFrame:
    transforms = _prepare_transforms(matrix, features)
    cell_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FOLDS) as executor:
        for qref_definition in COMMON_QREFS:
            target = aligned_targets[qref_definition]
            for model_name in MODEL_NAMES:
                future_map: dict[Future[Any], int] = {}
                for fold_id in range(1, N_FOLDS + 1):
                    future = executor.submit(
                        _fit_one_fold,
                        analysis_id=analysis_id,
                        feature_set=feature_set,
                        features=features,
                        dataset=dataset,
                        qref_definition=qref_definition,
                        model_name=model_name,
                        parameters=model_parameters[model_name],
                        target=target,
                        transform=transforms[fold_id],
                    )
                    future_map[future] = fold_id
                completed: dict[int, tuple[list[dict[str, Any]], dict[str, Any], int]] = {}
                for future in as_completed(future_map):
                    fold_id = future_map[future]
                    result = future.result()
                    completed[fold_id] = result
                    print(
                        f"[{datetime.now().isoformat(timespec='seconds')}] "
                        f"fit complete {analysis_id} | {dataset} | {qref_definition} | "
                        f"{model_name} | fold {fold_id}/{N_FOLDS} | "
                        f"{result[1]['Training_Seconds']:.2f}s",
                        flush=True,
                    )
                for fold_id in range(1, N_FOLDS + 1):
                    rows, record, prediction_count = completed[fold_id]
                    cell_rows.extend(rows)
                    audit.fit_records.append(record)
                    audit.prediction_rows += prediction_count
                    audit.train_test_cell_overlap += int(
                        record["Train_Test_Cell_Overlap_Count"]
                    )
    return pd.DataFrame(cell_rows)


def _relative_change(current: float, reference: float) -> float:
    if not np.isfinite(reference) or reference == 0:
        return np.nan
    return 100.0 * (current - reference) / abs(reference)


def _sensitivity_summary(
    cell_metrics: pd.DataFrame,
    *,
    feature_set: str,
    experiment: str,
    primary_summary: pd.DataFrame,
    population_note: str,
    compare_to_primary: bool,
) -> pd.DataFrame:
    summary = summarize_cell_metrics(cell_metrics).copy()
    summary.insert(0, "Experiment", experiment)
    summary.insert(1, "Feature_Set", feature_set)
    summary["Population_Note"] = population_note
    summary["Primary_Comparison_Population_Identical"] = (
        "YES" if compare_to_primary else "NO_POSTCYCLE5_FILTER_APPLIED"
    )
    primary_columns = [
        "Dataset",
        "Qref_Definition",
        "Model",
        "Median_MAE_Q_Ah",
        "Median_MAE_SOH_fraction",
        "Median_RMSE_Q_Ah",
        "Median_R2_Q",
    ]
    primary = primary_summary[primary_columns].rename(
        columns={
            "Median_MAE_Q_Ah": "Primary_A9_Median_MAE_Q_Ah",
            "Median_MAE_SOH_fraction": "Primary_A9_Median_MAE_SOH_fraction",
            "Median_RMSE_Q_Ah": "Primary_A9_Median_RMSE_Q_Ah",
            "Median_R2_Q": "Primary_A9_Median_R2_Q",
        }
    )
    summary = summary.merge(
        primary,
        on=["Dataset", "Qref_Definition", "Model"],
        how="left",
        validate="one_to_one",
    )
    metric_pairs = (
        ("Median_MAE_Q_Ah", "Primary_A9_Median_MAE_Q_Ah"),
        ("Median_MAE_SOH_fraction", "Primary_A9_Median_MAE_SOH_fraction"),
        ("Median_RMSE_Q_Ah", "Primary_A9_Median_RMSE_Q_Ah"),
        ("Median_R2_Q", "Primary_A9_Median_R2_Q"),
    )
    for current, reference in metric_pairs:
        label = current.replace("Median_", "")
        summary[f"Absolute_Change_{label}_vs_Primary_A9"] = (
            summary[current] - summary[reference]
        )
        summary[f"Relative_Change_{label}_vs_Primary_A9_Pct"] = [
            _relative_change(float(value), float(ref))
            for value, ref in zip(summary[current], summary[reference])
        ]
    change_columns = [
        column
        for column in summary.columns
        if column.startswith("Absolute_Change_")
        or column.startswith("Relative_Change_")
    ]
    if compare_to_primary:
        summary["Relative_Change_Definition"] = (
            "100*(sensitivity-primary)/abs(primary); NA when primary equals zero"
        )
    else:
        summary[change_columns] = np.nan
        summary["Relative_Change_Definition"] = (
            "NOT_CALCULATED_FOR_POSTCYCLE5_POPULATION_CHANGE"
        )
    return summary


def _ranking_order(summary: pd.DataFrame) -> str:
    frozen_order = {model: index for index, model in enumerate(MODEL_NAMES)}
    ranked = summary.assign(
        _Frozen_Model_Order=summary["Model"].map(frozen_order)
    ).sort_values(["Median_MAE_Q_Ah", "_Frozen_Model_Order"], kind="stable")
    return " > ".join(ranked["Model"].astype(str))


def _robustness_rankings(
    sensitivity_summaries: Iterable[pd.DataFrame],
    primary_rankings: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary_orders: dict[tuple[str, str], str] = {}
    for (dataset, qref), group in primary_rankings.groupby(
        ["Dataset", "Qref_Definition"], sort=False
    ):
        primary_orders[(dataset, qref)] = " > ".join(
            group.sort_values("Rank_Position")["Model"].astype(str)
        )
    for summary in sensitivity_summaries:
        feature_set = str(summary["Feature_Set"].iloc[0])
        experiment = str(summary["Experiment"].iloc[0])
        orders: dict[tuple[str, str], str] = {}
        for (dataset, qref), group in summary.groupby(
            ["Dataset", "Qref_Definition"], sort=False
        ):
            order = _ranking_order(group)
            orders[(dataset, qref)] = order
            primary_order = primary_orders[(dataset, qref)]
            rows.append(
                {
                    "Row_Type": "QREF_RANKING",
                    "Experiment": experiment,
                    "Feature_Set": feature_set,
                    "Dataset": dataset,
                    "Qref_Definition": qref,
                    "Ranking_Order": order,
                    "Primary_A9_Ranking_Order": primary_order,
                    "Ranking_Changed_vs_Primary": (
                        "YES" if order != primary_order else "NO"
                    ),
                    "Qref_Definition_A": "",
                    "Qref_Definition_B": "",
                    "Ranking_Order_A": "",
                    "Ranking_Order_B": "",
                    "Ranking_Changed_Within_Feature_Set": "",
                    "Ranking_Metric": "Median held-out-cell MAE_Q_Ah ascending",
                }
            )
        for dataset in summary["Dataset"].drop_duplicates():
            for qref_a, qref_b in QREF_PAIRS:
                order_a = orders[(dataset, qref_a)]
                order_b = orders[(dataset, qref_b)]
                rows.append(
                    {
                        "Row_Type": "WITHIN_FEATURE_QREF_PAIR",
                        "Experiment": experiment,
                        "Feature_Set": feature_set,
                        "Dataset": dataset,
                        "Qref_Definition": "",
                        "Ranking_Order": "",
                        "Primary_A9_Ranking_Order": "",
                        "Ranking_Changed_vs_Primary": "",
                        "Qref_Definition_A": qref_a,
                        "Qref_Definition_B": qref_b,
                        "Ranking_Order_A": order_a,
                        "Ranking_Order_B": order_b,
                        "Ranking_Changed_Within_Feature_Set": (
                            "YES" if order_a != order_b else "NO"
                        ),
                        "Ranking_Metric": "Median held-out-cell MAE_Q_Ah ascending",
                    }
                )
    return pd.DataFrame(rows)


def _seed_for(*parts: Any) -> int:
    text = "|".join([str(RANDOM_SEED), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _bootstrap_median_difference(
    differences: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float, float]:
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or not len(differences):
        raise RobustnessError("EMPTY_BOOTSTRAP_DIFFERENCE_VECTOR")
    if not np.isfinite(differences).all():
        raise RobustnessError("NONFINITE_BOOTSTRAP_DIFFERENCE_VECTOR")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(differences), size=(BOOTSTRAP_REPLICATES, len(differences))
    )
    replicates = np.median(differences[indices], axis=1)
    lower, upper = np.quantile(replicates, [0.025, 0.975], method="linear")
    return float(np.median(differences)), float(lower), float(upper)


def _qref_paired_bootstrap(primary_cell_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_columns = (
        ("MAE_Q_Ah", "PHYSICAL_CAPACITY_SPACE_AH"),
        ("MAE_SOH_fraction", "SOH_LABEL_SPACE_FRACTION"),
    )
    for dataset in primary_cell_metrics["Dataset"].drop_duplicates():
        dataset_rows = primary_cell_metrics.loc[
            primary_cell_metrics["Dataset"].eq(dataset)
        ]
        for model_name in MODEL_NAMES:
            model_rows = dataset_rows.loc[dataset_rows["Model"].eq(model_name)]
            for qref_a, qref_b in QREF_PAIRS:
                for metric, metric_space in metric_columns:
                    left = model_rows.loc[
                        model_rows["Qref_Definition"].eq(qref_a),
                        ["Cell_ID", metric],
                    ]
                    right = model_rows.loc[
                        model_rows["Qref_Definition"].eq(qref_b),
                        ["Cell_ID", metric],
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
                        raise RobustnessError(
                            f"QREF_BOOTSTRAP_PAIRING_FAILED:{dataset}:{model_name}:"
                            f"{qref_a}:{qref_b}:{metric}"
                        )
                    differences = (
                        paired[f"{metric}_B"].to_numpy(dtype=float)
                        - paired[f"{metric}_A"].to_numpy(dtype=float)
                    )
                    seed = _seed_for(
                        "QREF_PAIRED", dataset, model_name, qref_a, qref_b, metric
                    )
                    median, lower, upper = _bootstrap_median_difference(
                        differences, seed=seed
                    )
                    rows.append(
                        {
                            "Dataset": dataset,
                            "Model": model_name,
                            "Qref_Definition_A": qref_a,
                            "Qref_Definition_B": qref_b,
                            "Metric": metric,
                            "Metric_Space": metric_space,
                            "Difference_Definition": "Metric_B_minus_Metric_A_per_paired_cell",
                            "Cell_Count": len(paired),
                            "Median_Paired_Difference": median,
                            "Bootstrap_95CI_Lower": lower,
                            "Bootstrap_95CI_Upper": upper,
                            "Bootstrap_Replicates": BOOTSTRAP_REPLICATES,
                            "Bootstrap_Seed": seed,
                            "Bootstrap_Unit": "Cell_ID",
                            "Paired_By_Cell": "YES",
                            "Cycle_Level_Resampling": "NO",
                        }
                    )
    return pd.DataFrame(rows)


def _model_pair_bootstrap(primary_cell_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, qref), group in primary_cell_metrics.groupby(
        ["Dataset", "Qref_Definition"], sort=False
    ):
        for model_a, model_b in MODEL_PAIRS:
            left = group.loc[
                group["Model"].eq(model_a), ["Cell_ID", "MAE_Q_Ah"]
            ]
            right = group.loc[
                group["Model"].eq(model_b), ["Cell_ID", "MAE_Q_Ah"]
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
                raise RobustnessError(
                    f"MODEL_BOOTSTRAP_PAIRING_FAILED:{dataset}:{qref}:"
                    f"{model_a}:{model_b}"
                )
            differences = (
                paired["MAE_Q_Ah_B"].to_numpy(dtype=float)
                - paired["MAE_Q_Ah_A"].to_numpy(dtype=float)
            )
            seed = _seed_for("MODEL_PAIR", dataset, qref, model_a, model_b)
            median, lower, upper = _bootstrap_median_difference(
                differences, seed=seed
            )
            rows.append(
                {
                    "Dataset": dataset,
                    "Qref_Definition": qref,
                    "Model_A": model_a,
                    "Model_B": model_b,
                    "Difference_Definition": "Cell_MAE_Q_Ah_ModelB_minus_ModelA",
                    "Cell_Count": len(paired),
                    "Median_Difference_ModelB_minus_ModelA": median,
                    "Bootstrap_95CI_Lower": lower,
                    "Bootstrap_95CI_Upper": upper,
                    "Bootstrap_Replicates": BOOTSTRAP_REPLICATES,
                    "Bootstrap_Seed": seed,
                    "Bootstrap_Unit": "Cell_ID",
                    "Paired_By_Cell": "YES",
                    "Cycle_Level_Resampling": "NO",
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_rank_frequency(primary_cell_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, qref), group in primary_cell_metrics.groupby(
        ["Dataset", "Qref_Definition"], sort=False
    ):
        pivot = group.pivot(index="Cell_ID", columns="Model", values="MAE_Q_Ah")
        if set(pivot.columns) != set(MODEL_NAMES) or pivot.isna().any().any():
            raise RobustnessError(f"BOOTSTRAP_RANK_CELL_MATRIX_INCOMPLETE:{dataset}:{qref}")
        values = pivot[list(MODEL_NAMES)].to_numpy(dtype=float)
        seed = _seed_for("RANK_FREQUENCY", dataset, qref)
        rng = np.random.default_rng(seed)
        indices = rng.integers(
            0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
        )
        replicate_medians = np.median(values[indices], axis=1)
        rank1 = np.argmin(replicate_medians, axis=1)
        for model_index, model_name in enumerate(MODEL_NAMES):
            rows.append(
                {
                    "Dataset": dataset,
                    "Qref_Definition": qref,
                    "Model": model_name,
                    "Cell_Count": len(values),
                    "Bootstrap_Rank1_Count": int(np.sum(rank1 == model_index)),
                    "Bootstrap_Rank1_Frequency": float(
                        np.mean(rank1 == model_index)
                    ),
                    "Bootstrap_Replicates": BOOTSTRAP_REPLICATES,
                    "Bootstrap_Seed": seed,
                    "Bootstrap_Unit": "Cell_ID",
                    "Cycle_Level_Resampling": "NO",
                    "Rank_Metric": "Median cell MAE_Q_Ah",
                    "Tie_Break_Order": "|".join(MODEL_NAMES),
                    "Interpretation_Guardrail": (
                        "Observed-cell bootstrap ranking stability; not universal superiority"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _scale_effect_audit(
    primary_summary: pd.DataFrame,
    sensitivity_summaries: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    primary = primary_summary.copy()
    primary.insert(0, "Experiment", "FROZEN_PRIMARY_V1")
    primary.insert(1, "Feature_Set", "PRIMARY_FEATURE_SET_A9")
    combined = pd.concat([primary, *sensitivity_summaries], ignore_index=True, sort=False)
    rows: list[dict[str, Any]] = []
    for (experiment, feature_set, dataset, model), group in combined.groupby(
        ["Experiment", "Feature_Set", "Dataset", "Model"], sort=False
    ):
        nominal = group.loc[group["Qref_Definition"].eq("nominal")]
        if len(nominal) != 1:
            raise RobustnessError(
                f"SCALE_AUDIT_NOMINAL_ROW_COUNT:{experiment}:{dataset}:{model}"
            )
        nominal_row = nominal.iloc[0]
        for measured_qref in ("first_comparable", "early_median5_comparable"):
            measured = group.loc[group["Qref_Definition"].eq(measured_qref)]
            if len(measured) != 1:
                raise RobustnessError(
                    f"SCALE_AUDIT_MEASURED_ROW_COUNT:{experiment}:{dataset}:"
                    f"{model}:{measured_qref}"
                )
            measured_row = measured.iloc[0]
            soh_nominal = float(nominal_row["Median_MAE_SOH_fraction"])
            soh_measured = float(measured_row["Median_MAE_SOH_fraction"])
            q_nominal = float(nominal_row["Median_MAE_Q_Ah"])
            q_measured = float(measured_row["Median_MAE_Q_Ah"])
            soh_relative = _relative_change(soh_measured, soh_nominal)
            q_relative = _relative_change(q_measured, q_nominal)
            difference = abs(soh_relative) - abs(q_relative)
            if np.isclose(difference, 0.0, atol=1e-12, rtol=0):
                comparison = "EQUAL_WITHIN_1E-12_PERCENTAGE_POINTS"
            elif difference > 0:
                comparison = "YES"
            else:
                comparison = "NO"
            rows.append(
                {
                    "Experiment": experiment,
                    "Feature_Set": feature_set,
                    "Dataset": dataset,
                    "Model": model,
                    "Nominal_Qref_Definition": "nominal",
                    "Measured_Qref_Definition": measured_qref,
                    "Nominal_Median_MAE_SOH_fraction": soh_nominal,
                    "Measured_Median_MAE_SOH_fraction": soh_measured,
                    "Absolute_Change_MAE_SOH_fraction": soh_measured - soh_nominal,
                    "Relative_Change_MAE_SOH_Pct": soh_relative,
                    "Nominal_Median_MAE_Q_Ah": q_nominal,
                    "Measured_Median_MAE_Q_Ah": q_measured,
                    "Absolute_Change_MAE_Q_Ah": q_measured - q_nominal,
                    "Relative_Change_MAE_Q_Pct": q_relative,
                    "Absolute_Relative_Change_MAE_SOH_Pct": abs(soh_relative),
                    "Absolute_Relative_Change_MAE_Q_Pct": abs(q_relative),
                    "SOH_Relative_Change_Magnitude_Greater_Than_Physical": comparison,
                    "Relative_Change_Definition": (
                        "100*(measuredQref-nominal)/abs(nominal)"
                    ),
                    "Interpretation_Status": "NUMERIC_AUDIT_ONLY_PENDING_HUMAN_REVIEW",
                }
            )
    return pd.DataFrame(rows)


def _test_row(
    test_id: str,
    passed: bool,
    observed: Any,
    expected: Any,
    failure_message: str,
) -> dict[str, Any]:
    return {
        "Test_ID": test_id,
        "Status": "PASS" if passed else "FAIL",
        "Observed": str(observed),
        "Expected": str(expected),
        "Failure_Message": "" if passed else failure_message,
        "Hard_Failure": "YES",
    }


def _regression_tests(
    *,
    audit: TrainingAudit,
    primary_hashes_before: dict[str, str],
    primary_hashes_after: dict[str, str],
    frozen_input_hashes_before: dict[str, str],
    frozen_input_hashes_after: dict[str, str],
    primary_dir: Path,
    target: Path,
    qref_bootstrap: pd.DataFrame,
    model_bootstrap: pd.DataFrame,
    rank_frequency: pd.DataFrame,
    rankings: pd.DataFrame,
    postcycle_matrices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    tests: list[dict[str, Any]] = []
    primary_unchanged = primary_hashes_before == primary_hashes_after
    tests.append(
        _test_row(
            "PRIMARY_V1_FILES_UNCHANGED",
            primary_unchanged,
            f"before_files={len(primary_hashes_before)};after_files={len(primary_hashes_after)};equal={primary_unchanged}",
            "all frozen Primary V1 file hashes identical before and after",
            "At least one frozen Primary V1 file changed.",
        )
    )
    tests.append(
        _test_row(
            "A8_EXACTLY_PRIMARY_MINUS_CV_DURATION",
            A8_FEATURES
            == tuple(
                feature
                for feature in PRIMARY_FEATURES
                if feature != "CV_Duration_s"
            )
            and len(A8_FEATURES) == 8,
            "|".join(A8_FEATURES),
            "Primary A9 in frozen order with CV_Duration_s removed only",
            "A8 feature allowlist is incorrect.",
        )
    )
    tests.append(
        _test_row(
            "B10_EXACTLY_PRIMARY_PLUS_CYCLE_INDEX",
            B10_FEATURES == (*PRIMARY_FEATURES, "Raw_Cycle_Index")
            and len(B10_FEATURES) == 10,
            "|".join(B10_FEATURES),
            "Primary A9 plus Raw_Cycle_Index only",
            "B10 feature allowlist is incorrect.",
        )
    )
    all_sensitivity_features = set(A8_FEATURES) | set(B10_FEATURES) | set(
        POSTCYCLE5_FEATURES
    )
    shortcuts_present = sorted(
        all_sensitivity_features.intersection(PROTOCOL_SHORTCUT_FEATURES)
    )
    tests.append(
        _test_row(
            "PROTOCOL_SHORTCUT_FEATURES_STILL_ABSENT",
            not shortcuts_present,
            "|".join(shortcuts_present) or "NONE",
            "NONE",
            "A frozen protocol-shortcut feature was reintroduced.",
        )
    )
    fit_records = pd.DataFrame(audit.fit_records)
    same_fold_hashes = bool(
        fit_records.groupby(["Feature_Set", "Dataset", "Fold_ID"])[
            ["Train_Key_SHA256", "Test_Key_SHA256"]
        ]
        .nunique()
        .eq(1)
        .all()
        .all()
    )
    tests.append(
        _test_row(
            "SAME_FROZEN_CELL_FOLDS",
            same_fold_hashes
            and set(fit_records["Fold_ID"].astype(int)) == set(range(1, N_FOLDS + 1)),
            f"fit_records={len(fit_records)};folds={sorted(fit_records['Fold_ID'].unique())}",
            "frozen folds 1-5 with identical train/test key hashes across Qrefs and models",
            "Sensitivity runs did not preserve frozen cell folds.",
        )
    )
    tests.append(
        _test_row(
            "NO_CYCLE_LEVEL_RANDOM_SPLIT",
            audit.train_test_cell_overlap == 0
            and fit_records["Train_Test_Cell_Overlap_Count"].eq(0).all(),
            audit.train_test_cell_overlap,
            0,
            "At least one cell appeared in both train and test within a fold.",
        )
    )
    qref_hashes_identical = all(
        len(set(qref_hashes.values())) == 1
        for experiment in audit.population_key_hashes.values()
        for qref_hashes in experiment.values()
    )
    tests.append(
        _test_row(
            "NO_QREF_SPECIFIC_ROWS",
            qref_hashes_identical,
            _canonical_json(audit.population_key_hashes),
            "one physical-key hash per experiment and dataset across all Qrefs",
            "A sensitivity run used Qref-specific rows.",
        )
    )
    bootstrap_unit_ok = (
        qref_bootstrap["Bootstrap_Unit"].eq("Cell_ID").all()
        and model_bootstrap["Bootstrap_Unit"].eq("Cell_ID").all()
        and rank_frequency["Bootstrap_Unit"].eq("Cell_ID").all()
        and qref_bootstrap["Cycle_Level_Resampling"].eq("NO").all()
        and model_bootstrap["Cycle_Level_Resampling"].eq("NO").all()
        and rank_frequency["Cycle_Level_Resampling"].eq("NO").all()
    )
    tests.append(
        _test_row(
            "BOOTSTRAP_RESAMPLES_CELLS_NOT_CYCLES",
            bootstrap_unit_ok,
            "Cell_ID; cycle-level resampling=NO",
            "Cell_ID resampling only",
            "Bootstrap unit was not Cell_ID.",
        )
    )
    expected_counts = {"XJTU": 55, "HUST": 77}
    paired_ok = (
        qref_bootstrap["Paired_By_Cell"].eq("YES").all()
        and model_bootstrap["Paired_By_Cell"].eq("YES").all()
        and all(
            int(row.Cell_Count) == expected_counts[str(row.Dataset)]
            for row in qref_bootstrap.itertuples(index=False)
        )
        and all(
            int(row.Cell_Count) == expected_counts[str(row.Dataset)]
            for row in model_bootstrap.itertuples(index=False)
        )
    )
    tests.append(
        _test_row(
            "BOOTSTRAP_PAIRED_BY_CELL",
            paired_ok,
            f"qref_rows={len(qref_bootstrap)};model_pair_rows={len(model_bootstrap)}",
            "all comparisons paired one-to-one by Cell_ID with 55 XJTU or 77 HUST cells",
            "Bootstrap pairing by Cell_ID failed.",
        )
    )
    post_hashes_ok = all(
        len(
            set(
                audit.population_key_hashes["ROBUSTNESS_POSTCYCLE5"][dataset].values()
            )
        )
        == 1
        and matrix["Raw_Cycle_Index"].gt(5).all()
        for dataset, matrix in postcycle_matrices.items()
    )
    tests.append(
        _test_row(
            "POSTCYCLE5_POPULATION_IDENTICAL_ACROSS_QREFS",
            post_hashes_ok,
            _canonical_json(audit.population_key_hashes["ROBUSTNESS_POSTCYCLE5"]),
            "identical Raw_Cycle_Index > 5 physical keys across all Qrefs",
            "Post-cycle-5 populations differ by Qref or retain cycles 1-5.",
        )
    )
    lower_features = [feature.lower() for feature in all_sensitivity_features]
    tests.append(
        _test_row(
            "NO_TARGET_CAPACITY_FEATURE",
            not any("capacity" in feature for feature in lower_features),
            "|".join(sorted(all_sensitivity_features)),
            "no capacity target feature",
            "Capacity entered a sensitivity predictor set.",
        )
    )
    tests.append(
        _test_row(
            "NO_QREF_FEATURE",
            not any("qref" in feature for feature in lower_features),
            "|".join(sorted(all_sensitivity_features)),
            "no Qref feature",
            "Qref entered a sensitivity predictor set.",
        )
    )
    tests.append(
        _test_row(
            "NO_RUL_OR_EOL_FEATURE",
            not any(
                token in feature
                for feature in lower_features
                for token in ("rul", "eol")
            ),
            "|".join(sorted(all_sensitivity_features)),
            "no RUL/EOL feature",
            "RUL or EOL entered a sensitivity predictor set.",
        )
    )
    tests.append(
        _test_row(
            "PRIMARY_RESULTS_NOT_OVERWRITTEN",
            primary_unchanged and primary_dir.resolve() != target.resolve(),
            f"hashes_equal={primary_unchanged};separate_target={primary_dir.resolve() != target.resolve()}",
            "frozen primary hashes unchanged and separate output directory",
            "Primary results were overwritten or target directory was not separate.",
        )
    )
    tests.append(
        _test_row(
            "FROZEN_FEATURE_LABEL_FOLD_INPUTS_UNCHANGED",
            frozen_input_hashes_before == frozen_input_hashes_after,
            f"files={len(frozen_input_hashes_after)};equal={frozen_input_hashes_before == frozen_input_hashes_after}",
            "all frozen feature, label, fold, model, preprocessing and metric input hashes unchanged",
            "A frozen benchmark input file changed during robustness analysis.",
        )
    )
    expected_fits = len(EXPERIMENTS) * 2 * len(COMMON_QREFS) * len(MODEL_NAMES) * N_FOLDS
    tests.append(
        _test_row(
            "FROZEN_ROBUSTNESS_FIT_GRID_COMPLETE",
            len(fit_records) == expected_fits,
            len(fit_records),
            expected_fits,
            "Robustness model/fold/Qref grid is incomplete or duplicated.",
        )
    )
    parameter_ok = bool(
        fit_records.groupby("Model")["Model_Parameters_SHA256"].nunique().eq(1).all()
    )
    tests.append(
        _test_row(
            "MODEL_PARAMETERS_IDENTICAL_ACROSS_EXPERIMENTS_AND_QREFS",
            parameter_ok,
            fit_records.groupby("Model")["Model_Parameters_SHA256"].nunique().to_dict(),
            "one frozen parameter hash per model",
            "Model parameters changed by experiment or Qref.",
        )
    )
    rank_frequency_sums = rank_frequency.groupby(
        ["Dataset", "Qref_Definition"]
    )["Bootstrap_Rank1_Frequency"].sum()
    bootstrap_shape_ok = (
        len(qref_bootstrap) == 36
        and len(model_bootstrap) == 18
        and len(rank_frequency) == 18
        and np.allclose(rank_frequency_sums.to_numpy(dtype=float), 1.0)
        and qref_bootstrap["Bootstrap_Replicates"].eq(BOOTSTRAP_REPLICATES).all()
        and model_bootstrap["Bootstrap_Replicates"].eq(BOOTSTRAP_REPLICATES).all()
        and rank_frequency["Bootstrap_Replicates"].eq(BOOTSTRAP_REPLICATES).all()
    )
    tests.append(
        _test_row(
            "BOOTSTRAP_OUTPUT_GRID_COMPLETE",
            bootstrap_shape_ok,
            f"qref={len(qref_bootstrap)};model_pairs={len(model_bootstrap)};rank_frequency={len(rank_frequency)}",
            "36 Qref-metric rows, 18 model-pair rows, 18 rank-frequency rows, each using 5000 replicates",
            "Bootstrap output grid is incomplete or rank frequencies do not sum to one.",
        )
    )
    tests.append(
        _test_row(
            "RANKING_ROBUSTNESS_GRID_COMPLETE",
            len(rankings) == 36,
            len(rankings),
            36,
            "Ranking robustness rows are incomplete.",
        )
    )
    return pd.DataFrame(tests)


def _validate_outputs(staging: Path) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for name in CSV_OUTPUTS:
        path = staging / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RobustnessError(f"MISSING_OR_EMPTY_OUTPUT:{name}")
        frame = pd.read_csv(path, low_memory=False)
        if frame.columns.duplicated().any() or frame.empty:
            raise RobustnessError(f"INVALID_OUTPUT_TABLE:{name}")
        details[name] = {
            "rows": len(frame),
            "columns": len(frame.columns),
            "bytes": path.stat().st_size,
        }
    tests = pd.read_csv(staging / "soh_robustness_regression_tests.csv")
    if not tests["Status"].eq("PASS").all():
        raise RobustnessError("WRITTEN_REGRESSION_TEST_FAILURE")
    return details


def _processing_log(
    *,
    started: datetime,
    finished: datetime,
    primary_dir: Path,
    primary_hashes: dict[str, str],
    audit: TrainingAudit,
    summaries: dict[str, pd.DataFrame],
    rankings: pd.DataFrame,
    qref_bootstrap: pd.DataFrame,
    model_bootstrap: pd.DataFrame,
    rank_frequency: pd.DataFrame,
    tests: pd.DataFrame,
    output_validation: dict[str, Any] | None,
) -> str:
    try:
        import scipy
        import sklearn
    except ImportError as exc:  # pragma: no cover
        raise RobustnessError("DEPENDENCY_VERSION_QUERY_FAILED") from exc
    fit_records = pd.DataFrame(audit.fit_records)
    lines = [
        f"Pipeline: {PIPELINE_VERSION}",
        f"Status: {'ALL REGRESSION TESTS PASSED' if tests['Status'].eq('PASS').all() else 'PIPELINE FAILED'}",
        f"Started: {started.isoformat()}",
        f"Finished: {finished.isoformat()}",
        f"Elapsed_seconds: {(finished - started).total_seconds():.3f}",
        f"Python: {platform.python_version()}",
        f"numpy: {np.__version__}",
        f"pandas: {pd.__version__}",
        f"scipy: {scipy.__version__}",
        f"scikit-learn: {sklearn.__version__}",
        f"Random_seed: {RANDOM_SEED}",
        f"Bootstrap_replicates: {BOOTSTRAP_REPLICATES}",
        "Bootstrap_unit: Cell_ID",
        "Bootstrap_cycle_resampling: NO",
        "Bootstrap_pairing: paired by Cell_ID",
        f"Parallel_fold_workers: {MAX_PARALLEL_FOLDS}",
        "Model_internal_parallelism: frozen parameters unchanged (RF n_jobs=1)",
        f"Frozen_primary_directory: {primary_dir}",
        f"Frozen_primary_file_count: {len(primary_hashes)}",
        "Frozen_primary_files_unchanged: YES",
        "A8_features: " + "|".join(A8_FEATURES),
        "B10_features: " + "|".join(B10_FEATURES),
        "Postcycle5_features: " + "|".join(POSTCYCLE5_FEATURES),
        "Protocol_shortcut_features_reintroduced: NO",
        "Qrefs: " + "|".join(COMMON_QREFS),
        "Models: " + "|".join(MODEL_NAMES),
        f"Model_fits: {len(fit_records)}",
        f"Internal_held_out_prediction_rows: {audit.prediction_rows}",
        f"Train_test_cell_overlap_count: {audit.train_test_cell_overlap}",
        "Sensitivity_summary_rows:",
    ]
    for name, summary in summaries.items():
        lines.append(f"  {name}: {len(summary)}")
    lines.extend(
        [
            f"Ranking_rows: {len(rankings)}",
            f"Qref_paired_bootstrap_rows: {len(qref_bootstrap)}",
            f"Model_pair_bootstrap_rows: {len(model_bootstrap)}",
            f"Bootstrap_rank_frequency_rows: {len(rank_frequency)}",
            "Ranking_changes_vs_primary:",
        ]
    )
    qref_ranking_rows = rankings.loc[rankings["Row_Type"].eq("QREF_RANKING")]
    for row in qref_ranking_rows.sort_values(
        ["Experiment", "Dataset", "Qref_Definition"]
    ).itertuples(index=False):
        lines.append(
            f"  {row.Experiment}|{row.Dataset}|{row.Qref_Definition}: "
            f"{row.Ranking_Changed_vs_Primary} | {row.Ranking_Order}"
        )
    lines.append("Regression_tests:")
    for row in tests.itertuples(index=False):
        lines.append(
            f"  {row.Test_ID}: {row.Status} | observed={row.Observed} | expected={row.Expected}"
        )
    if output_validation is not None:
        lines.append("Written_output_validation:")
        lines.append(_canonical_json(output_validation))
    lines.append(
        "Final_status: "
        + (
            "SOH ML ROBUSTNESS V1 COMPLETE — WAITING FOR HUMAN REVIEW"
            if tests["Status"].eq("PASS").all()
            else "SOH ML ROBUSTNESS V1 FAILED"
        )
    )
    return "\n".join(lines) + "\n"


def run_robustness(
    workspace: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    target = (
        Path(output_dir).resolve()
        if output_dir is not None
        else workspace / "outputs" / "soh_ml_robustness_v1"
    )
    if target.exists():
        raise RobustnessError(f"OUTPUT_ALREADY_EXISTS:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".soh_ml_robustness_v1_staging_", dir=target.parent)
    )
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    print(f"Staging output: {staging}", flush=True)

    primary_paths = PrimaryPaths.from_workspace(workspace)
    primary_dir = workspace / "outputs" / "soh_primary_benchmark_v1"
    primary_hashes_before = _snapshot_tree(primary_dir)
    primary = _load_primary_outputs(primary_dir)
    loaded = _load_inputs(primary_paths)
    frozen_input_hashes_before = {
        name: _sha256_file(path) for name, path in loaded["paths"].items()
    }
    _validate_frozen_folds(loaded["xmatrix"], loaded["xfold"], "XJTU")
    _validate_frozen_folds(loaded["hmatrix"], loaded["hfold"], "HUST")
    x_population = _frozen_population(
        loaded["xmatrix"], primary["mask_xjtu"], "XJTU"
    )
    h_population = _frozen_population(
        loaded["hmatrix"], primary["mask_hust"], "HUST"
    )
    base_populations = {"XJTU": x_population, "HUST": h_population}
    labels = {"XJTU": loaded["xlabels"], "HUST": loaded["hlabels"]}

    audit = TrainingAudit()
    experiment_cell_metrics: dict[str, pd.DataFrame] = {}
    postcycle_matrices: dict[str, pd.DataFrame] = {}
    for analysis_id, feature_set, features, postcycle5 in EXPERIMENTS:
        dataset_cells: list[pd.DataFrame] = []
        audit.population_key_hashes[analysis_id] = {}
        audit.population_feature_hashes[analysis_id] = {}
        for dataset in ("XJTU", "HUST"):
            matrix = base_populations[dataset].copy()
            if postcycle5:
                matrix = matrix.loc[matrix["Raw_Cycle_Index"].gt(5)].reset_index(
                    drop=True
                )
                if matrix["Cell_ID"].nunique() != base_populations[dataset][
                    "Cell_ID"
                ].nunique():
                    raise RobustnessError(
                        f"POSTCYCLE5_REMOVED_ENTIRE_CELL:{dataset}"
                    )
                postcycle_matrices[dataset] = matrix.copy()
            aligned, key_hashes = _align_targets(
                matrix, labels[dataset], dataset, COMMON_QREFS
            )
            audit.population_key_hashes[analysis_id][dataset] = key_hashes
            feature_values = matrix[list(features)].apply(
                pd.to_numeric, errors="raise"
            ).to_numpy(dtype=float)
            audit.population_feature_hashes[analysis_id][dataset] = _sha256_array(
                feature_values
            )
            dataset_cells.append(
                _train_population(
                    analysis_id=analysis_id,
                    feature_set=feature_set,
                    features=features,
                    dataset=dataset,
                    matrix=matrix,
                    aligned_targets=aligned,
                    model_parameters=loaded["model_parameters"],
                    audit=audit,
                )
            )
        experiment_cell_metrics[analysis_id] = pd.concat(
            dataset_cells, ignore_index=True
        )

    primary_summary = primary["summary_metrics"].copy()
    a8_summary = _sensitivity_summary(
        experiment_cell_metrics["ROBUSTNESS_A8"],
        feature_set="A8_MINUS_CV_DURATION",
        experiment="ROBUSTNESS_A8",
        primary_summary=primary_summary,
        population_note="Exact frozen Primary V1 physical observations",
        compare_to_primary=True,
    )
    b10_summary = _sensitivity_summary(
        experiment_cell_metrics["ROBUSTNESS_B10"],
        feature_set="B10_PHYSICS_PLUS_CYCLE_INDEX",
        experiment="ROBUSTNESS_B10",
        primary_summary=primary_summary,
        population_note="Exact frozen Primary V1 physical observations",
        compare_to_primary=True,
    )
    postcycle5_summary = _sensitivity_summary(
        experiment_cell_metrics["ROBUSTNESS_POSTCYCLE5"],
        feature_set="A9_POSTCYCLE5",
        experiment="ROBUSTNESS_POSTCYCLE5",
        primary_summary=primary_summary,
        population_note=(
            "Frozen Primary V1 observations filtered identically for every Qref to Raw_Cycle_Index > 5"
        ),
        compare_to_primary=False,
    )
    summaries = {
        "ROBUSTNESS_A8": a8_summary,
        "ROBUSTNESS_B10": b10_summary,
        "ROBUSTNESS_POSTCYCLE5": postcycle5_summary,
    }
    rankings = _robustness_rankings(
        summaries.values(), primary["rankings"]
    )
    qref_bootstrap = _qref_paired_bootstrap(primary["cell_metrics"])
    model_bootstrap = _model_pair_bootstrap(primary["cell_metrics"])
    rank_frequency = _bootstrap_rank_frequency(primary["cell_metrics"])
    scale_audit = _scale_effect_audit(primary_summary, summaries.values())

    _write_csv(a8_summary, staging / "soh_robustness_a8_summary.csv")
    _write_csv(b10_summary, staging / "soh_robustness_b10_summary.csv")
    _write_csv(
        postcycle5_summary, staging / "soh_robustness_postcycle5_summary.csv"
    )
    _write_csv(rankings, staging / "soh_robustness_rankings.csv")
    _write_csv(qref_bootstrap, staging / "soh_qref_paired_bootstrap_ci.csv")
    _write_csv(model_bootstrap, staging / "soh_model_pair_bootstrap_ci.csv")
    _write_csv(rank_frequency, staging / "soh_bootstrap_rank_frequency.csv")
    _write_csv(scale_audit, staging / "soh_scale_effect_audit.csv")

    primary_hashes_after = _snapshot_tree(primary_dir)
    frozen_input_hashes_after = {
        name: _sha256_file(path) for name, path in loaded["paths"].items()
    }
    tests = _regression_tests(
        audit=audit,
        primary_hashes_before=primary_hashes_before,
        primary_hashes_after=primary_hashes_after,
        frozen_input_hashes_before=frozen_input_hashes_before,
        frozen_input_hashes_after=frozen_input_hashes_after,
        primary_dir=primary_dir,
        target=target,
        qref_bootstrap=qref_bootstrap,
        model_bootstrap=model_bootstrap,
        rank_frequency=rank_frequency,
        rankings=rankings,
        postcycle_matrices=postcycle_matrices,
    )
    _write_csv(tests, staging / "soh_robustness_regression_tests.csv")
    finished = datetime.now(ZoneInfo("Asia/Shanghai"))
    preliminary_log = _processing_log(
        started=started,
        finished=finished,
        primary_dir=primary_dir,
        primary_hashes=primary_hashes_after,
        audit=audit,
        summaries=summaries,
        rankings=rankings,
        qref_bootstrap=qref_bootstrap,
        model_bootstrap=model_bootstrap,
        rank_frequency=rank_frequency,
        tests=tests,
        output_validation=None,
    )
    (staging / "processing_log.txt").write_text(preliminary_log, encoding="utf-8")
    failures = tests.loc[tests["Status"].ne("PASS")]
    if not failures.empty:
        raise RobustnessError(
            "HARD_REGRESSION_FAILURES:"
            + "|".join(failures["Test_ID"].astype(str))
            + f";staging={staging}"
        )
    output_validation = _validate_outputs(staging)
    finished = datetime.now(ZoneInfo("Asia/Shanghai"))
    final_log = _processing_log(
        started=started,
        finished=finished,
        primary_dir=primary_dir,
        primary_hashes=primary_hashes_after,
        audit=audit,
        summaries=summaries,
        rankings=rankings,
        qref_bootstrap=qref_bootstrap,
        model_bootstrap=model_bootstrap,
        rank_frequency=rank_frequency,
        tests=tests,
        output_validation=output_validation,
    )
    (staging / "processing_log.txt").write_text(final_log, encoding="utf-8")
    staging.replace(target)
    return {
        "output_dir": str(target),
        "regression_tests_passed": True,
        "final_stdout": (
            "SOH ML ROBUSTNESS V1 COMPLETE —\nWAITING FOR HUMAN REVIEW"
        ),
    }


__all__ = [
    "RobustnessError",
    "run_robustness",
    "_bootstrap_median_difference",
    "_qref_paired_bootstrap",
    "_model_pair_bootstrap",
    "_bootstrap_rank_frequency",
]
