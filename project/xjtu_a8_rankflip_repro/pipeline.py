"""Targeted verification of the XJTU A8 RF-versus-GB rank flip.

The frozen robustness run did not publish its intermediate A8 cell-level
metrics.  This module therefore reconstructs only the required XJTU A8 fits,
checks them against the frozen A8 summary, then bootstraps cells.  It never
invokes the full robustness benchmark.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from soh_ml_robustness_repro.config import A8_FEATURES
from soh_ml_robustness_repro.pipeline import (
    TrainingAudit,
    _frozen_population,
    _load_primary_outputs,
    _seed_for,
    _snapshot_tree,
    _train_population,
)
from soh_primary_benchmark_repro.config import COMMON_QREFS, MODEL_NAMES, N_FOLDS
from soh_primary_benchmark_repro.metrics import summarize_cell_metrics
from soh_primary_benchmark_repro.pipeline import (
    Paths as PrimaryPaths,
    _align_targets,
    _load_inputs,
    _sha256_file,
    _validate_frozen_folds,
    _write_csv,
)

from .config import (
    ANALYSIS_ID,
    BOOTSTRAP_REPLICATES,
    CSV_OUTPUTS,
    DATASET,
    FEATURE_SET,
    GB_MODEL,
    PIPELINE_VERSION,
    RANDOM_SEED,
    RF_MODEL,
    RIDGE_MODEL,
)


class RankFlipVerificationError(RuntimeError):
    """Raised when a hard verification condition is not met."""


def _frame_sha256(frame: pd.DataFrame) -> str:
    sort_columns = [
        column
        for column in (
            "Dataset",
            "Qref_Definition",
            "Model",
            "Cell_ID",
            "Test_ID",
        )
        if column in frame.columns
    ]
    ordered = frame.sort_values(sort_columns).reset_index(drop=True) if sort_columns else frame
    payload = ordered.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="<NA>",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _existing_a8_cell_metrics(robustness_dir: Path) -> tuple[pd.DataFrame | None, str]:
    candidates = (
        robustness_dir / "soh_robustness_a8_cell_metrics.csv",
        robustness_dir / "soh_robustness_cell_metrics.csv",
        robustness_dir / "a8_cell_metrics.csv",
    )
    for path in candidates:
        if not path.is_file():
            continue
        frame = pd.read_csv(path, low_memory=False)
        required = {
            "Analysis_ID",
            "Dataset",
            "Cell_ID",
            "Qref_Definition",
            "Model",
            "MAE_Q_Ah",
        }
        if not required.issubset(frame.columns):
            raise RankFlipVerificationError(
                f"EXISTING_A8_CELL_METRICS_SCHEMA_MISMATCH:{path}"
            )
        selected = frame.loc[
            frame["Analysis_ID"].eq(ANALYSIS_ID)
            & frame["Dataset"].eq(DATASET)
        ].copy()
        if selected.empty:
            raise RankFlipVerificationError(
                f"EXISTING_A8_CELL_METRICS_NO_XJTU_A8_ROWS:{path}"
            )
        return selected.reset_index(drop=True), str(path.resolve())
    return None, "MISSING_FROM_FROZEN_ROBUSTNESS_OUTPUT"


def _reconstruct_xjtu_a8(
    workspace: Path,
    primary: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    TrainingAudit,
    dict[str, Path],
    dict[str, str],
]:
    primary_paths = PrimaryPaths.from_workspace(workspace)
    loaded = _load_inputs(primary_paths)
    frozen_hashes_before = {
        name: _sha256_file(path) for name, path in loaded["paths"].items()
    }
    _validate_frozen_folds(loaded["xmatrix"], loaded["xfold"], DATASET)
    population = _frozen_population(
        loaded["xmatrix"], primary["mask_xjtu"], DATASET
    )
    aligned, _ = _align_targets(
        population, loaded["xlabels"], DATASET, COMMON_QREFS
    )
    audit = TrainingAudit()
    cell_metrics = _train_population(
        analysis_id=ANALYSIS_ID,
        feature_set=FEATURE_SET,
        features=A8_FEATURES,
        dataset=DATASET,
        matrix=population,
        aligned_targets=aligned,
        model_parameters=loaded["model_parameters"],
        audit=audit,
    )
    return (
        cell_metrics.reset_index(drop=True),
        audit,
        loaded["paths"],
        frozen_hashes_before,
    )


def _validate_cell_metric_grid(cell_metrics: pd.DataFrame) -> None:
    required = {
        "Analysis_ID",
        "Dataset",
        "Cell_ID",
        "Fold_ID",
        "Qref_Definition",
        "Model",
        "MAE_Q_Ah",
    }
    missing = sorted(required.difference(cell_metrics.columns))
    if missing:
        raise RankFlipVerificationError(
            "A8_CELL_METRICS_MISSING_COLUMNS:" + "|".join(missing)
        )
    if not cell_metrics["Analysis_ID"].eq(ANALYSIS_ID).all():
        raise RankFlipVerificationError("NON_A8_CELL_METRIC_ROWS")
    if not cell_metrics["Dataset"].eq(DATASET).all():
        raise RankFlipVerificationError("NON_XJTU_CELL_METRIC_ROWS")
    if cell_metrics.duplicated(["Cell_ID", "Qref_Definition", "Model"]).any():
        raise RankFlipVerificationError("DUPLICATE_A8_CELL_METRIC_KEY")
    expected = pd.MultiIndex.from_product(
        [
            sorted(cell_metrics["Cell_ID"].astype(str).unique()),
            COMMON_QREFS,
            MODEL_NAMES,
        ],
        names=["Cell_ID", "Qref_Definition", "Model"],
    )
    observed = pd.MultiIndex.from_frame(
        cell_metrics[["Cell_ID", "Qref_Definition", "Model"]].astype(str)
    )
    if len(expected) != 55 * len(COMMON_QREFS) * len(MODEL_NAMES):
        raise RankFlipVerificationError("XJTU_CELL_COUNT_NOT_55")
    if set(expected) != set(observed):
        raise RankFlipVerificationError("INCOMPLETE_A8_CELL_METRIC_GRID")
    if not np.isfinite(
        pd.to_numeric(cell_metrics["MAE_Q_Ah"], errors="raise").to_numpy(float)
    ).all():
        raise RankFlipVerificationError("NONFINITE_A8_CELL_MAE_Q")


def _compare_with_frozen_summary(
    cell_metrics: pd.DataFrame,
    robustness_dir: Path,
) -> tuple[bool, float, int]:
    path = robustness_dir / "soh_robustness_a8_summary.csv"
    if not path.is_file():
        raise RankFlipVerificationError(f"MISSING_FROZEN_A8_SUMMARY:{path}")
    frozen = pd.read_csv(path, low_memory=False)
    frozen = frozen.loc[frozen["Dataset"].eq(DATASET)].copy()
    generated = summarize_cell_metrics(cell_metrics)
    key = ["Dataset", "Qref_Definition", "Model"]
    compare_columns = [
        "Cell_Count",
        "Observation_Count",
        "Median_MAE_Q_Ah",
        "Q1_MAE_Q_Ah",
        "Q3_MAE_Q_Ah",
        "IQR_MAE_Q_Ah",
        "Mean_MAE_Q_Ah",
        "Median_RMSE_Q_Ah",
        "Median_R2_Q",
        "Median_MAE_SOH_fraction",
        "Median_MAE_SOH_percentage_points",
        "Median_RMSE_SOH_fraction",
        "Median_R2_SOH",
    ]
    merged = generated[key + compare_columns].merge(
        frozen[key + compare_columns],
        on=key,
        how="outer",
        suffixes=("_Generated", "_Frozen"),
        indicator=True,
        validate="one_to_one",
    )
    if len(merged) != len(COMMON_QREFS) * len(MODEL_NAMES):
        return False, float("inf"), len(merged)
    if not merged["_merge"].eq("both").all():
        return False, float("inf"), len(merged)
    max_abs = 0.0
    all_close = True
    for column in compare_columns:
        left = pd.to_numeric(merged[f"{column}_Generated"], errors="raise").to_numpy(float)
        right = pd.to_numeric(merged[f"{column}_Frozen"], errors="raise").to_numpy(float)
        if column in {"Cell_Count", "Observation_Count"}:
            close = np.array_equal(left, right)
        else:
            finite = np.isfinite(left) & np.isfinite(right)
            same_nan = np.isnan(left) & np.isnan(right)
            close = bool(
                np.all(same_nan | (finite & np.isclose(left, right, rtol=1e-10, atol=1e-12)))
            )
            if finite.any():
                max_abs = max(max_abs, float(np.max(np.abs(left[finite] - right[finite]))))
        all_close = all_close and bool(close)
    return all_close, max_abs, len(merged)


def _paired_rf_gb_bootstrap(
    cell_metrics: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for qref in COMMON_QREFS:
        subset = cell_metrics.loc[
            cell_metrics["Qref_Definition"].eq(qref)
            & cell_metrics["Model"].isin([RF_MODEL, GB_MODEL]),
            ["Cell_ID", "Model", "MAE_Q_Ah"],
        ]
        pivot = subset.pivot(index="Cell_ID", columns="Model", values="MAE_Q_Ah")
        if set(pivot.columns) != {RF_MODEL, GB_MODEL} or pivot.isna().any().any():
            raise RankFlipVerificationError(f"RF_GB_PAIRING_FAILED:{qref}")
        pivot = pivot.sort_index()
        rf = pivot[RF_MODEL].to_numpy(float)
        gb = pivot[GB_MODEL].to_numpy(float)
        differences = gb - rf
        seed = _seed_for(PIPELINE_VERSION, "RF_GB_PAIRED", qref)
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(pivot), size=(replicates, len(pivot)))
        boot = np.median(differences[indices], axis=1)
        lower, upper = np.quantile(boot, [0.025, 0.975], method="linear")
        rf_median = float(np.median(rf))
        gb_median = float(np.median(gb))
        rows.append(
            {
                "Dataset": DATASET,
                "Analysis_ID": ANALYSIS_ID,
                "Feature_Set": FEATURE_SET,
                "Qref_Definition": qref,
                "Model_A": RF_MODEL,
                "Model_B": GB_MODEL,
                "Metric": "MAE_Q_Ah",
                "Difference_Definition": "GB_MAE_Q_Ah_minus_RF_MAE_Q_Ah_per_paired_cell",
                "Cell_Count": len(pivot),
                "Observed_RF_Median_MAE_Q_Ah": rf_median,
                "Observed_GB_Median_MAE_Q_Ah": gb_median,
                "Observed_Ranking_By_Median_Cell_MAE_Q_Ah": (
                    "RF_LT_GB" if rf_median < gb_median else "GB_LT_RF" if gb_median < rf_median else "TIE"
                ),
                "Observed_Median_Difference": float(np.median(differences)),
                "Bootstrap_95CI_Lower": float(lower),
                "Bootstrap_95CI_Upper": float(upper),
                "Bootstrap_95CI_Includes_Zero": "YES" if lower <= 0 <= upper else "NO",
                "Bootstrap_Replicates": replicates,
                "Bootstrap_Seed": seed,
                "Bootstrap_Unit": "Cell_ID",
                "Paired_By_Cell_ID": "YES",
                "Cycle_Level_Resampling": "NO",
            }
        )
    return pd.DataFrame(rows)


def _rank_frequency(
    cell_metrics: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for qref in COMMON_QREFS:
        subset = cell_metrics.loc[cell_metrics["Qref_Definition"].eq(qref)]
        pivot = subset.pivot(index="Cell_ID", columns="Model", values="MAE_Q_Ah")
        if set(pivot.columns) != set(MODEL_NAMES) or pivot.isna().any().any():
            raise RankFlipVerificationError(f"RANK_FREQUENCY_GRID_FAILED:{qref}")
        pivot = pivot.sort_index()
        values = pivot[list(MODEL_NAMES)].to_numpy(float)
        seed = _seed_for(PIPELINE_VERSION, "JOINT_RANK_FREQUENCY", qref)
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(pivot), size=(replicates, len(pivot)))
        boot_medians = np.median(values[indices], axis=1)
        rank1 = np.argmin(boot_medians, axis=1)
        observed_medians = np.median(values, axis=0)
        observed_order = np.argsort(observed_medians, kind="stable")
        observed_ranks = np.empty(len(MODEL_NAMES), dtype=int)
        observed_ranks[observed_order] = np.arange(1, len(MODEL_NAMES) + 1)
        for index, model in enumerate(MODEL_NAMES):
            rows.append(
                {
                    "Dataset": DATASET,
                    "Analysis_ID": ANALYSIS_ID,
                    "Feature_Set": FEATURE_SET,
                    "Qref_Definition": qref,
                    "Model": model,
                    "Cell_Count": len(pivot),
                    "Observed_Median_MAE_Q_Ah": float(observed_medians[index]),
                    "Observed_Rank": int(observed_ranks[index]),
                    "Bootstrap_Rank1_Count": int(np.sum(rank1 == index)),
                    "Bootstrap_Rank1_Frequency": float(np.mean(rank1 == index)),
                    "Bootstrap_Replicates": replicates,
                    "Bootstrap_Seed": seed,
                    "Bootstrap_Unit": "Cell_ID",
                    "Joint_Three_Model_Resampling": "YES",
                    "Cycle_Level_Resampling": "NO",
                    "Rank_Metric": "Median cell MAE_Q_Ah",
                    "Tie_Break_Order": "|".join(MODEL_NAMES),
                    "Interpretation_Guardrail": (
                        "Observed-sample cell-bootstrap rank stability; not universal superiority probability"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _qref_rankflip_bootstrap(
    cell_metrics: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> pd.DataFrame:
    subset = cell_metrics.loc[
        cell_metrics["Model"].isin([RF_MODEL, GB_MODEL]),
        ["Cell_ID", "Qref_Definition", "Model", "MAE_Q_Ah"],
    ]
    pivot = subset.pivot(
        index="Cell_ID", columns=["Qref_Definition", "Model"], values="MAE_Q_Ah"
    ).sort_index()
    expected_columns = pd.MultiIndex.from_product(
        [COMMON_QREFS, (RF_MODEL, GB_MODEL)],
        names=["Qref_Definition", "Model"],
    )
    if not set(expected_columns).issubset(set(pivot.columns)):
        raise RankFlipVerificationError("QREF_RANKFLIP_GRID_INCOMPLETE")
    values = np.stack(
        [
            pivot[(qref, model)].to_numpy(float)
            for qref in COMMON_QREFS
            for model in (RF_MODEL, GB_MODEL)
        ],
        axis=1,
    ).reshape(len(pivot), len(COMMON_QREFS), 2)
    if not np.isfinite(values).all():
        raise RankFlipVerificationError("NONFINITE_QREF_RANKFLIP_VALUES")
    seed = _seed_for(PIPELINE_VERSION, "DIRECT_QREF_RANKFLIP")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pivot), size=(replicates, len(pivot)))
    boot_medians = np.median(values[indices], axis=1)
    qref_index = {qref: index for index, qref in enumerate(COMMON_QREFS)}
    nominal_index = qref_index["nominal"]
    nominal_rf_lt_gb = (
        boot_medians[:, nominal_index, 0] < boot_medians[:, nominal_index, 1]
    )
    observed_medians = np.median(values, axis=0)
    observed_nominal = (
        "RF_LT_GB"
        if observed_medians[nominal_index, 0] < observed_medians[nominal_index, 1]
        else "GB_LT_RF"
        if observed_medians[nominal_index, 1] < observed_medians[nominal_index, 0]
        else "TIE"
    )
    rows: list[dict[str, Any]] = []
    for measured_qref in ("first_comparable", "early_median5_comparable"):
        measured_index = qref_index[measured_qref]
        measured_gb_lt_rf = (
            boot_medians[:, measured_index, 1]
            < boot_medians[:, measured_index, 0]
        )
        joint = nominal_rf_lt_gb & measured_gb_lt_rf
        observed_measured = (
            "GB_LT_RF"
            if observed_medians[measured_index, 1] < observed_medians[measured_index, 0]
            else "RF_LT_GB"
            if observed_medians[measured_index, 0] < observed_medians[measured_index, 1]
            else "TIE"
        )
        rows.append(
            {
                "Dataset": DATASET,
                "Analysis_ID": ANALYSIS_ID,
                "Feature_Set": FEATURE_SET,
                "Metric": "Median cell MAE_Q_Ah",
                "Nominal_Qref_Definition": "nominal",
                "Measured_Qref_Definition": measured_qref,
                "RankFlip_Event": f"nominal_RF_LT_GB_AND_{measured_qref}_GB_LT_RF",
                "Cell_Count": len(pivot),
                "Observed_Nominal_Order": observed_nominal,
                "Observed_Measured_Order": observed_measured,
                "Nominal_RF_LT_GB_Count": int(nominal_rf_lt_gb.sum()),
                "Nominal_RF_LT_GB_Frequency": float(nominal_rf_lt_gb.mean()),
                "Measured_GB_LT_RF_Count": int(measured_gb_lt_rf.sum()),
                "Measured_GB_LT_RF_Frequency": float(measured_gb_lt_rf.mean()),
                "RankFlip_Count": int(joint.sum()),
                "Observed_Sample_RankFlip_Stability": float(joint.mean()),
                "Bootstrap_Replicates": replicates,
                "Bootstrap_Seed": seed,
                "Bootstrap_Unit": "Cell_ID",
                "Paired_Across_Qrefs_By_Cell_ID": "YES",
                "Cycle_Level_Resampling": "NO",
                "Interpretation_Guardrail": (
                    "Observed-sample rank-flip stability; not a population-level probability"
                ),
            }
        )
    return pd.DataFrame(rows)


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
    cell_metrics: pd.DataFrame,
    cell_hash_before: str,
    cell_hash_after: str,
    frozen_summary_match: bool,
    frozen_summary_max_abs: float,
    frozen_summary_rows: int,
    audit: TrainingAudit | None,
    retraining_required: bool,
    metrics_source: str,
    ci: pd.DataFrame,
    rank_frequency: pd.DataFrame,
    rankflip: pd.DataFrame,
    primary_hashes_before: dict[str, str],
    primary_hashes_after: dict[str, str],
    robustness_hashes_before: dict[str, str],
    robustness_hashes_after: dict[str, str],
    frozen_input_hashes_before: dict[str, str],
    frozen_input_hashes_after: dict[str, str],
) -> pd.DataFrame:
    tests: list[dict[str, Any]] = []
    tests.append(
        _test_row(
            "A8_CELL_METRICS_UNCHANGED",
            cell_hash_before == cell_hash_after and frozen_summary_match,
            f"before={cell_hash_before};after={cell_hash_after};frozen_summary_match={frozen_summary_match}",
            "cell metric hash unchanged during bootstrap and reconstructed summary matches frozen A8",
            "A8 cell metrics changed or failed frozen-summary reconciliation.",
        )
    )
    tests.append(
        _test_row(
            "XJTU_CELL_COUNT_55",
            cell_metrics["Cell_ID"].nunique() == 55,
            str(cell_metrics["Cell_ID"].nunique()),
            "55",
            "XJTU A8 cell count is not 55.",
        )
    )
    paired_ok = (
        ci["Paired_By_Cell_ID"].eq("YES").all()
        and ci["Cell_Count"].eq(55).all()
        and rank_frequency["Cell_Count"].eq(55).all()
        and rankflip["Paired_Across_Qrefs_By_Cell_ID"].eq("YES").all()
        and rankflip["Cell_Count"].eq(55).all()
    )
    tests.append(
        _test_row(
            "PAIRED_BY_CELL_ID",
            paired_ok,
            f"ci_rows={len(ci)};rank_rows={len(rank_frequency)};rankflip_rows={len(rankflip)}",
            "all resampling paired by the same 55 Cell_ID values",
            "Bootstrap pairing is not complete at Cell_ID level.",
        )
    )
    no_cycle = (
        ci["Cycle_Level_Resampling"].eq("NO").all()
        and rank_frequency["Cycle_Level_Resampling"].eq("NO").all()
        and rankflip["Cycle_Level_Resampling"].eq("NO").all()
    )
    tests.append(
        _test_row(
            "NO_CYCLE_RESAMPLING",
            no_cycle,
            "NO" if no_cycle else "YES",
            "NO",
            "Cycle-level resampling was detected.",
        )
    )
    if audit is None:
        same_fits = not retraining_required
        fit_observed = "retraining=NO;existing frozen cell metrics used"
    else:
        fit_frame = pd.DataFrame(audit.fit_records)
        same_fits = (
            retraining_required
            and len(fit_frame) == len(COMMON_QREFS) * len(MODEL_NAMES) * N_FOLDS
            and set(fit_frame["Fold_ID"]) == set(range(1, N_FOLDS + 1))
            and fit_frame["Features"].eq("|".join(A8_FEATURES)).all()
            and fit_frame["Train_Test_Cell_Overlap_Count"].eq(0).all()
            and fit_frame.groupby("Model")["Model_Parameters_SHA256"].nunique().eq(1).all()
            and fit_frame.groupby(["Qref_Definition", "Model"])["Fold_ID"].nunique().eq(N_FOLDS).all()
            and audit.prediction_rows == 10427 * len(COMMON_QREFS) * len(MODEL_NAMES)
            and frozen_summary_match
        )
        fit_observed = (
            f"fits={len(fit_frame)};prediction_rows={audit.prediction_rows};"
            f"overlap={audit.train_test_cell_overlap};summary_max_abs={frozen_summary_max_abs:.3g}"
        )
    tests.append(
        _test_row(
            "SAME_FROZEN_FOLDS_MODEL_FITS",
            same_fits,
            fit_observed,
            "45 exact XJTU A8 fits on frozen folds when reconstruction is required, with frozen-summary reconciliation",
            "Frozen folds, features, model parameters, or fit grid changed.",
        )
    )
    tests.append(
        _test_row(
            "NO_RETRAINING_UNLESS_REQUIRED",
            (retraining_required and metrics_source == "MISSING_FROM_FROZEN_ROBUSTNESS_OUTPUT")
            or (not retraining_required and metrics_source != "MISSING_FROM_FROZEN_ROBUSTNESS_OUTPUT"),
            f"retraining={'YES' if retraining_required else 'NO'};source={metrics_source}",
            "retrain only when frozen XJTU A8 cell metrics are absent",
            "Retraining requirement was not respected.",
        )
    )
    replicate_ok = (
        ci["Bootstrap_Replicates"].eq(BOOTSTRAP_REPLICATES).all()
        and rank_frequency["Bootstrap_Replicates"].eq(BOOTSTRAP_REPLICATES).all()
        and rankflip["Bootstrap_Replicates"].eq(BOOTSTRAP_REPLICATES).all()
    )
    tests.append(
        _test_row(
            "BOOTSTRAP_REPLICATES_5000",
            replicate_ok,
            str(BOOTSTRAP_REPLICATES),
            "5000 for every bootstrap output row",
            "Bootstrap replicate count is not 5000.",
        )
    )
    frozen_dirs_ok = (
        primary_hashes_before == primary_hashes_after
        and robustness_hashes_before == robustness_hashes_after
        and frozen_input_hashes_before == frozen_input_hashes_after
    )
    tests.append(
        _test_row(
            "PRIMARY_AND_ROBUSTNESS_FILES_UNCHANGED",
            frozen_dirs_ok,
            (
                f"primary_equal={primary_hashes_before == primary_hashes_after};"
                f"robustness_equal={robustness_hashes_before == robustness_hashes_after};"
                f"frozen_inputs_equal={frozen_input_hashes_before == frozen_input_hashes_after}"
            ),
            "all frozen Primary, Robustness, feature, label, fold, and model-plan hashes unchanged",
            "A frozen input or output file changed.",
        )
    )
    tests.append(
        _test_row(
            "A8_RECONSTRUCTION_MATCHES_FROZEN_SUMMARY",
            frozen_summary_match and frozen_summary_rows == 9,
            f"rows={frozen_summary_rows};max_abs_difference={frozen_summary_max_abs:.15g}",
            "9 XJTU A8 summary rows match within rtol=1e-10 and atol=1e-12",
            "Reconstructed A8 metrics do not reproduce the frozen summary.",
        )
    )
    observed_orders = dict(zip(ci["Qref_Definition"], ci["Observed_Ranking_By_Median_Cell_MAE_Q_Ah"]))
    ordering_ok = (
        observed_orders.get("nominal") == "RF_LT_GB"
        and observed_orders.get("first_comparable") == "GB_LT_RF"
        and observed_orders.get("early_median5_comparable") == "GB_LT_RF"
    )
    tests.append(
        _test_row(
            "OBSERVED_A8_RF_GB_ORDERING_REPRODUCED",
            ordering_ok,
            str(observed_orders),
            "nominal RF<GB; first_comparable GB<RF; early_median5_comparable GB<RF",
            "Observed XJTU A8 RF/GB ordering was not reproduced.",
        )
    )
    rank_sums = rank_frequency.groupby("Qref_Definition")["Bootstrap_Rank1_Frequency"].sum()
    grid_ok = (
        len(ci) == 3
        and len(rank_frequency) == 9
        and len(rankflip) == 2
        and np.allclose(rank_sums.to_numpy(float), 1.0, rtol=0, atol=1e-12)
    )
    tests.append(
        _test_row(
            "OUTPUT_GRID_COMPLETE",
            grid_ok,
            f"ci={len(ci)};rank={len(rank_frequency)};rankflip={len(rankflip)};rank_sums={rank_sums.to_dict()}",
            "3 CI rows, 9 rank-frequency rows, 2 rank-flip rows, rank frequencies sum to 1",
            "Targeted output grid is incomplete.",
        )
    )
    return pd.DataFrame(tests)


def _validate_outputs(staging: Path) -> dict[str, dict[str, int]]:
    expected_keys = {
        "xjtu_a8_rf_gb_bootstrap_ci.csv": ["Qref_Definition"],
        "xjtu_a8_rank_frequency.csv": ["Qref_Definition", "Model"],
        "xjtu_a8_qref_rankflip_bootstrap.csv": ["Measured_Qref_Definition"],
        "xjtu_a8_rankflip_regression_tests.csv": ["Test_ID"],
    }
    result: dict[str, dict[str, int]] = {}
    for name in CSV_OUTPUTS:
        path = staging / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RankFlipVerificationError(f"MISSING_OR_EMPTY_OUTPUT:{name}")
        frame = pd.read_csv(path, low_memory=False)
        if frame.columns.duplicated().any():
            raise RankFlipVerificationError(f"DUPLICATE_OUTPUT_COLUMNS:{name}")
        if frame.duplicated(expected_keys[name]).any():
            raise RankFlipVerificationError(f"DUPLICATE_OUTPUT_KEYS:{name}")
        if frame[expected_keys[name]].isna().any(axis=None):
            raise RankFlipVerificationError(f"NULL_OUTPUT_KEYS:{name}")
        result[name] = {
            "rows": len(frame),
            "columns": len(frame.columns),
            "bytes": path.stat().st_size,
        }
    tests = pd.read_csv(staging / "xjtu_a8_rankflip_regression_tests.csv")
    if not tests["Status"].eq("PASS").all():
        raise RankFlipVerificationError("WRITTEN_REGRESSION_TEST_FAILURE")
    return result


def run_verification(
    workspace: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    target = (
        Path(output_dir).resolve()
        if output_dir is not None
        else workspace / "outputs" / "xjtu_a8_rankflip_verification_v1"
    )
    if target.exists():
        raise RankFlipVerificationError(f"OUTPUT_ALREADY_EXISTS:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".xjtu_a8_rankflip_verification_v1_staging_",
            dir=target.parent,
        )
    )

    primary_dir = workspace / "outputs" / "soh_primary_benchmark_v1"
    robustness_dir = workspace / "outputs" / "soh_ml_robustness_v1"
    primary_hashes_before = _snapshot_tree(primary_dir)
    robustness_hashes_before = _snapshot_tree(robustness_dir)
    primary = _load_primary_outputs(primary_dir)

    existing, metrics_source = _existing_a8_cell_metrics(robustness_dir)
    retraining_required = existing is None
    audit: TrainingAudit | None = None
    frozen_input_hashes_before: dict[str, str] = {}
    frozen_input_hashes_after: dict[str, str] = {}
    if existing is None:
        (
            cell_metrics,
            audit,
            frozen_paths,
            frozen_input_hashes_before,
        ) = _reconstruct_xjtu_a8(workspace, primary)
    else:
        cell_metrics = existing

    _validate_cell_metric_grid(cell_metrics)
    cell_hash_before = _frame_sha256(cell_metrics)
    summary_match, summary_max_abs, summary_rows = _compare_with_frozen_summary(
        cell_metrics, robustness_dir
    )
    if not summary_match:
        raise RankFlipVerificationError(
            f"A8_RECONSTRUCTION_DOES_NOT_MATCH_FROZEN_SUMMARY:max_abs={summary_max_abs}"
        )

    ci = _paired_rf_gb_bootstrap(cell_metrics)
    rank_frequency = _rank_frequency(cell_metrics)
    rankflip = _qref_rankflip_bootstrap(cell_metrics)
    cell_hash_after = _frame_sha256(cell_metrics)

    _write_csv(ci, staging / "xjtu_a8_rf_gb_bootstrap_ci.csv")
    _write_csv(rank_frequency, staging / "xjtu_a8_rank_frequency.csv")
    _write_csv(rankflip, staging / "xjtu_a8_qref_rankflip_bootstrap.csv")

    primary_hashes_after = _snapshot_tree(primary_dir)
    robustness_hashes_after = _snapshot_tree(robustness_dir)
    if retraining_required:
        frozen_input_hashes_after = {
            name: _sha256_file(path) for name, path in frozen_paths.items()
        }

    tests = _regression_tests(
        cell_metrics=cell_metrics,
        cell_hash_before=cell_hash_before,
        cell_hash_after=cell_hash_after,
        frozen_summary_match=summary_match,
        frozen_summary_max_abs=summary_max_abs,
        frozen_summary_rows=summary_rows,
        audit=audit,
        retraining_required=retraining_required,
        metrics_source=metrics_source,
        ci=ci,
        rank_frequency=rank_frequency,
        rankflip=rankflip,
        primary_hashes_before=primary_hashes_before,
        primary_hashes_after=primary_hashes_after,
        robustness_hashes_before=robustness_hashes_before,
        robustness_hashes_after=robustness_hashes_after,
        frozen_input_hashes_before=frozen_input_hashes_before,
        frozen_input_hashes_after=frozen_input_hashes_after,
    )
    _write_csv(tests, staging / "xjtu_a8_rankflip_regression_tests.csv")
    failures = tests.loc[tests["Status"].ne("PASS")]
    if not failures.empty:
        raise RankFlipVerificationError(
            "HARD_REGRESSION_FAILURES:"
            + "|".join(failures["Test_ID"].astype(str))
            + f";staging={staging}"
        )
    output_validation = _validate_outputs(staging)
    staging.replace(target)
    return {
        "output_dir": str(target),
        "retraining_required": retraining_required,
        "metrics_source": metrics_source,
        "output_validation": output_validation,
        "final_stdout": (
            "XJTU A8 RANK-FLIP VERIFICATION COMPLETE —\n"
            "WAITING FOR HUMAN REVIEW"
        ),
    }


__all__ = [
    "RankFlipVerificationError",
    "run_verification",
    "_paired_rf_gb_bootstrap",
    "_rank_frequency",
    "_qref_rankflip_bootstrap",
]
