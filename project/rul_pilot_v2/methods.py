"""RF, TRAJ, WAFT and BASE trained on (cell, t) samples with frozen cell folds (PROTOCOL_v2 section 5.1)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from rul_ranking_pilot.methods import _fold_preprocess, traj_predict

from .config import (
    EXACT,
    INTERVAL,
    MIN_CELLS,
    N_FOLDS,
    RF_PARAMS,
    TRAJ_HORIZON_FACTOR,
    TRAJ_MAX_FALLBACK_FRACTION,
    WAFT_PENALIZER,
)


@dataclass
class SettingResult:
    predictions: pd.DataFrame
    estimable: dict[str, bool]
    reasons: dict[str, str]
    traj_fallback_fraction: float = np.nan
    traj_missing_fraction: float = np.nan
    audit: list[dict[str, Any]] = field(default_factory=list)


def run_setting(samples: pd.DataFrame, x: np.ndarray, capacity: dict[str, pd.DataFrame], threshold: dict[str, float],
                t_values: tuple[int, int, int], traj_only: bool = False) -> SettingResult:
    """`samples` are the eligible (cell, t) rows of one dataset x Qref x theta, aligned with design rows `x`."""
    from lifelines import WeibullAFTFitter
    from lifelines.exceptions import ConvergenceError
    from sklearn.ensemble import RandomForestRegressor

    methods = ("TRAJ",) if traj_only else ("RF", "TRAJ", "WAFT", "BASE")
    cell = samples.Cell_ID.to_numpy(object)
    fold = samples.Fold_ID.to_numpy(int)
    tk = samples.t_Index.to_numpy(int)
    status = samples.Status.to_numpy(object)
    rul = samples.RUL.to_numpy(float)
    eol_exact = samples.EOL_Exact_Index.to_numpy(float)
    preds = {m: np.full(len(samples), np.nan) for m in methods}
    fallback = np.zeros(len(samples), bool)
    estimable = {m: True for m in methods}
    reasons = {m: "" for m in methods}
    result = SettingResult(pd.DataFrame(), estimable, reasons)

    for k in range(1, N_FOLDS + 1):
        train, test = fold != k, fold == k
        if set(cell[train]) & set(cell[test]):
            raise RuntimeError("TRAIN_TEST_CELL_OVERLAP")
        if not test.any():
            continue
        exact_tr = train & (status == EXACT)
        n_exact_cells = len(set(cell[exact_tr]))
        n_event_cells = len(set(cell[train & np.isin(status, [EXACT, INTERVAL])]))
        # Per-t training summaries used by BASE and as the TRAJ fallback.
        med_t = {t: float(np.median(rul[exact_tr & (tk == t)])) if (exact_tr & (tk == t)).any() else np.nan for t in range(3)}
        max_t = {t: float(np.max(rul[exact_tr & (tk == t)])) if (exact_tr & (tk == t)).any() else np.nan for t in range(3)}
        horizon = TRAJ_HORIZON_FACTOR * float(np.max(eol_exact[exact_tr])) if exact_tr.any() else np.inf

        if not traj_only:
            for m in ("RF", "BASE"):
                if n_exact_cells < MIN_CELLS:
                    estimable[m], reasons[m] = False, f"FOLD_{k}_TRAIN_EXACT_CELLS_{n_exact_cells}_LT_{MIN_CELLS}"
            if estimable["BASE"]:
                preds["BASE"][test] = [med_t[t] for t in tk[test]]
            if estimable["RF"]:
                tr, te, stats = _fold_preprocess(x[exact_tr], x[test], scale=False)
                preds["RF"][test] = RandomForestRegressor(**RF_PARAMS).fit(tr, rul[exact_tr]).predict(te)
                result.audit.append({"Method": "RF", "Fold_ID": k, "Train_Rows": np.flatnonzero(exact_tr).tolist(), **stats})
            if estimable["WAFT"]:
                if n_event_cells < MIN_CELLS:
                    estimable["WAFT"], reasons["WAFT"] = False, f"FOLD_{k}_TRAIN_EVENT_CELLS_{n_event_cells}_LT_{MIN_CELLS}"
                else:
                    tr, te, stats = _fold_preprocess(x[train], x[test], scale=True)
                    keep = np.ptp(tr, axis=0) > 0
                    tr, te = tr[:, keep], te[:, keep]
                    cols = [f"x{i}" for i in range(tr.shape[1])]
                    frame = pd.DataFrame(tr, columns=cols)
                    frame["L"] = samples.RUL_Lower.to_numpy(float)[train]
                    frame["U"] = samples.RUL_Upper.to_numpy(float)[train]
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fit = WeibullAFTFitter(penalizer=WAFT_PENALIZER).fit_interval_censoring(frame, "L", "U")
                            p = fit.predict_median(pd.DataFrame(te, columns=cols)).to_numpy(float).ravel()
                        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(fit.params_.to_numpy(float)))):
                            estimable["WAFT"], reasons["WAFT"] = False, f"FOLD_{k}_NONFINITE_FIT"
                        else:
                            preds["WAFT"][test] = p
                    except (ConvergenceError, ValueError, np.linalg.LinAlgError) as exc:
                        estimable["WAFT"], reasons["WAFT"] = False, f"FOLD_{k}_{type(exc).__name__}"
                    result.audit.append({"Method": "WAFT", "Fold_ID": k, "Train_Rows": np.flatnonzero(train).tolist(),
                                         "Dropped_Constant_Columns": int((~keep).sum()), **stats})

        for i in np.flatnonzero(test):
            fb = max_t[tk[i]]
            pred, used = traj_predict(capacity[cell[i]], t_values[tk[i]], threshold[cell[i]], float(samples.n_t_Index.iat[i]), fb)
            if not used and pred + float(samples.n_t_Index.iat[i]) > horizon:
                pred, used = fb, True
            preds["TRAJ"][i], fallback[i] = pred, used

    n = len(samples)
    result.traj_fallback_fraction = float(fallback.mean()) if n else np.nan
    result.traj_missing_fraction = float(np.isnan(preds["TRAJ"]).mean()) if n else np.nan
    if n and result.traj_fallback_fraction > TRAJ_MAX_FALLBACK_FRACTION:
        estimable["TRAJ"], reasons["TRAJ"] = False, f"TRAJ_FALLBACK_FRACTION_{result.traj_fallback_fraction:.3f}_GT_0.5"
    for m in methods:
        if not estimable[m]:
            preds[m][:] = np.nan
    result.predictions = pd.DataFrame({"Cell_ID": cell, "t_Index": tk, **{f"Pred_{m}": preds[m] for m in methods},
                                       "TRAJ_Fallback": fallback})
    return result
