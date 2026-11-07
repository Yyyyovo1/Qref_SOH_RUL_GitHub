"""RF, TRAJ, WAFT and BASE with frozen cell folds (PROTOCOL.md section 4)."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import EXACT, INTERVAL, MIN_CELLS, N_FOLDS, RF_PARAMS, TRAJ_HORIZON_FACTOR, WAFT_PENALIZER
from .labels import DatasetInputs, PilotError


def _sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype=np.float64)).tobytes()).hexdigest()


@dataclass
class SettingResult:
    predictions: pd.DataFrame
    estimable: dict[str, bool]
    reasons: dict[str, str]
    audit: list[dict[str, Any]] = field(default_factory=list)
    traj_fallbacks: int = 0
    traj_missing: int = 0


def _fold_preprocess(x_train: np.ndarray, x_test: np.ndarray, scale: bool) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Median imputation (and standardization) with statistics from the training cells only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(x_train, axis=0)
    median = np.where(np.isnan(median), 0.0, median)  # a feature missing in every training cell carries no information
    tr = np.where(np.isnan(x_train), median, x_train)
    te = np.where(np.isnan(x_test), median, x_test)
    stats = {"Imputer_Median_SHA256": _sha(median)}
    if scale:
        mean, std = tr.mean(axis=0), tr.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        tr, te = (tr - mean) / std, (te - mean) / std
        stats.update({"Scaler_Mean_SHA256": _sha(mean), "Scaler_Scale_SHA256": _sha(std)})
    return tr, te, stats


def traj_predict(
    capacity: pd.DataFrame, t0: int, threshold_Ah: float, n_t0: float, fallback: float
) -> tuple[float, bool]:
    """OLS line of capacity vs endpoint index on raw cycles in [t0/2, t0]; crossing minus n(t0)."""
    window = capacity[(capacity["Raw_Cycle_Index"] <= t0) & (capacity["Raw_Cycle_Index"] >= t0 / 2.0)]
    x = window["Endpoint_Index"].to_numpy(float)
    y = window["Capacity_Ah"].to_numpy(float)
    if len(x) < 2 or np.unique(x).size < 2:
        return fallback, True
    xc = x - x.mean()
    slope = float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))
    intercept = float(y.mean() - slope * x.mean())
    if slope >= 0:
        return fallback, True
    crossing = (threshold_Ah - intercept) / slope
    return crossing - n_t0, False


def run_setting(
    data: DatasetInputs,
    labels: pd.DataFrame,
    design: pd.DataFrame,
    cal: str,
    methods: tuple[str, ...],
    t0: int,
    threshold_Ah: dict[str, float],
) -> SettingResult:
    """Cross-validated predictions for one dataset x theta x Qref x calibration setting.

    `labels` holds this setting's rows (one per cell, lexicographic order); `design` the
    features for the calibration condition. Only eligible cells are trained on or predicted.
    """
    from lifelines import WeibullAFTFitter
    from lifelines.exceptions import ConvergenceError
    from sklearn.ensemble import RandomForestRegressor

    lab = labels.set_index("Cell_ID").loc[data.cells]
    eligible = lab["Eligible"].to_numpy(bool)
    fold = lab["Fold_ID"].to_numpy(int)
    status = lab["Status"].to_numpy(object)
    rul = lab["RUL"].to_numpy(float)
    x_all = design.loc[data.cells].to_numpy(float)
    preds = {m: np.full(len(data.cells), np.nan) for m in methods}
    estimable = {m: True for m in methods}
    reasons = {m: "" for m in methods}
    result = SettingResult(pd.DataFrame(), estimable, reasons)
    exact_end = (lab["EOL_Exact_Index"]).to_numpy(float)
    cap_by_cell = {str(c): g for c, g in data.capacity.groupby("Cell_ID")}

    for k in range(1, N_FOLDS + 1):
        train = eligible & (fold != k)
        test = eligible & (fold == k)
        if set(np.flatnonzero(train)) & set(np.flatnonzero(test)):
            raise PilotError("TRAIN_TEST_OVERLAP")
        if not test.any():
            continue
        train_exact = train & (status == EXACT)
        n_exact = int(train_exact.sum())
        n_events = int((train & np.isin(status, [EXACT, INTERVAL])).sum())

        for m in ("RF", "BASE"):
            if m in methods and n_exact < MIN_CELLS:
                estimable[m], reasons[m] = False, f"FOLD_{k}_TRAIN_EXACT_{n_exact}_LT_{MIN_CELLS}"

        if "BASE" in methods and estimable["BASE"]:
            preds["BASE"][test] = float(np.median(rul[train_exact]))

        if "RF" in methods and estimable["RF"]:
            tr, te, stats = _fold_preprocess(x_all[train_exact], x_all[test], scale=False)
            model = RandomForestRegressor(**RF_PARAMS).fit(tr, rul[train_exact])
            preds["RF"][test] = model.predict(te)
            result.audit.append({"Method": "RF", "Fold_ID": k, "Train_Cells": "|".join(np.array(data.cells)[train_exact]),
                                 "Test_Cells": "|".join(np.array(data.cells)[test]), **stats})

        if "WAFT" in methods and estimable["WAFT"]:
            if n_events < MIN_CELLS:
                estimable["WAFT"], reasons["WAFT"] = False, f"FOLD_{k}_TRAIN_EVENTS_{n_events}_LT_{MIN_CELLS}"
            else:
                tr, te, stats = _fold_preprocess(x_all[train], x_all[test], scale=True)
                # Columns constant across the training cells (e.g. Qref under the nominal reference)
                # carry no information and break lifelines' internal normalization; drop them.
                keep = np.ptp(tr, axis=0) > 0
                tr, te = tr[:, keep], te[:, keep]
                cols = [f"x{i}" for i in range(tr.shape[1])]
                frame = pd.DataFrame(tr, columns=cols)
                frame["L"] = lab["RUL_Lower"].to_numpy(float)[train]
                frame["U"] = lab["RUL_Upper"].to_numpy(float)[train]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fitter = WeibullAFTFitter(penalizer=WAFT_PENALIZER).fit_interval_censoring(frame, "L", "U")
                        fold_pred = fitter.predict_median(pd.DataFrame(te, columns=cols)).to_numpy(float).ravel()
                    if not np.all(np.isfinite(fold_pred)) or not np.all(np.isfinite(fitter.params_.to_numpy(float))):
                        estimable["WAFT"], reasons["WAFT"] = False, f"FOLD_{k}_NONFINITE_FIT"
                    else:
                        preds["WAFT"][test] = fold_pred
                except (ConvergenceError, ValueError, np.linalg.LinAlgError) as exc:
                    estimable["WAFT"], reasons["WAFT"] = False, f"FOLD_{k}_{type(exc).__name__}"
                result.audit.append({"Method": "WAFT", "Fold_ID": k, "Train_Cells": "|".join(np.array(data.cells)[train]),
                                     "Test_Cells": "|".join(np.array(data.cells)[test]),
                                     "Dropped_Constant_Columns": int((~keep).sum()), **stats})

        if "TRAJ" in methods:
            horizon = TRAJ_HORIZON_FACTOR * float(np.max(exact_end[train_exact])) if n_exact else np.inf
            fallback = float(np.max(rul[train_exact])) if n_exact else np.nan
            for i in np.flatnonzero(test):
                cell = data.cells[i]
                n_t0 = float(lab["n_t0_Index"].iat[i])
                pred, used_fallback = traj_predict(cap_by_cell[cell], t0, threshold_Ah[cell], n_t0, fallback)
                if not used_fallback and pred + n_t0 > horizon:
                    pred, used_fallback = fallback, True
                result.traj_fallbacks += int(used_fallback)
                result.traj_missing += int(np.isnan(pred))
                preds["TRAJ"][i] = pred

    for m in methods:
        if not estimable[m]:
            preds[m][:] = np.nan
    result.predictions = pd.DataFrame({"Cell_ID": data.cells, **{f"Pred_{m}": preds[m] for m in methods}})
    return result
