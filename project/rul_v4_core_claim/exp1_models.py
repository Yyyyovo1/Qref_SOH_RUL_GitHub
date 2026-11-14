"""Experiment 1 models (PROTOCOL_v4 section 2.2): variance (OLS on log variance), discharge and full (elastic net,
z-standardization on the outer training fold, inner 4-fold CV repeated 10 times), outer frozen 5 folds or the original
Severson split. Fitting only: nothing in this module computes an error against the labels.

In development runs the inner-CV MSE, the selected (alpha, lambda), coefficients and predictions stay in memory; only
finiteness flags, warning counts and leakage-audit hashes leave this module (AMENDMENT_01 section 11).
"""

from __future__ import annotations

import hashlib
import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from rul_ranking_pilot.methods import _sha

from .config import ENET_EPS, ENET_L1_RATIOS, ENET_N_ALPHAS, INNER_FOLDS, INNER_REPEATS, N_FOLDS, RANDOM_SEED


def model_features(prep) -> dict[str, list[str]]:
    m = prep.MODELS
    return {"variance": list(m["variance"]), "discharge": list(m["discharge_candidates"]), "full": list(m["full_candidates"])}


def inner_splits(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Repeats r = 0..9 of KFold(4, shuffle=True, random_state=20260928 + r) over the n outer-training cells."""
    from sklearn.model_selection import KFold

    out = []
    for r in range(INNER_REPEATS):
        out.extend(KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED + r).split(np.arange(n)))
    return out


def splits_sha(splits) -> str:
    h = hashlib.sha256()
    for tr, va in splits:
        h.update(np.asarray(tr, "<i8").tobytes())
        h.update(b"|")
        h.update(np.asarray(va, "<i8").tobytes())
        h.update(b"#")
    return h.hexdigest()


def standardize(x_train: np.ndarray, x_test: np.ndarray):
    """z-standardization with training-fold statistics (StandardScaler formula: ddof 0, zero scale -> 1)."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    scale = np.where(std > 0, std, 1.0)
    return (x_train - mean) / scale, (x_test - mean) / scale, mean, scale


@dataclass
class FitResult:
    predictions: np.ndarray          # cycles (10 ** yhat), aligned with the test rows
    finite: bool
    warnings: Counter = field(default_factory=Counter)
    hyper: dict = field(default_factory=dict)   # selected l1_ratio / alpha (formal outputs only)
    audit: dict = field(default_factory=dict)


def fit_predict(model: str, x_train: np.ndarray, y_train_cycles: np.ndarray, x_test: np.ndarray,
                all_rows: np.ndarray | None = None) -> FitResult:
    """Fit on the outer training rows, predict the test rows. `all_rows` (training + test design) is used only for the
    negative control of the leakage audit."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNetCV, LinearRegression

    y = np.log10(np.asarray(y_train_cycles, float))
    counts: Counter = Counter()
    audit: dict = {}
    hyper: dict = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if model == "variance":
            reg = LinearRegression().fit(x_train, y)
            yhat = reg.predict(x_test)
            coef = np.r_[reg.coef_, reg.intercept_]
        else:
            tr, te, mean, scale = standardize(x_train, x_test)
            splits = inner_splits(len(x_train))
            reg = ElasticNetCV(l1_ratio=list(ENET_L1_RATIOS), eps=ENET_EPS, alphas=ENET_N_ALPHAS, cv=splits, n_jobs=1,
                               max_iter=1000, tol=1e-4, selection="cyclic").fit(tr, y)
            yhat = reg.predict(te)
            coef = np.r_[reg.coef_, reg.intercept_]
            hyper = {"Selected_L1_Ratio": float(reg.l1_ratio_), "Selected_Lambda": float(reg.alpha_)}
            re_mean, re_scale = x_train.mean(axis=0), np.where(x_train.std(axis=0) > 0, x_train.std(axis=0), 1.0)
            audit = {"Scaler_Mean_SHA256": _sha(mean), "Scaler_Recomputed_Match": _sha(re_mean) == _sha(mean) and _sha(re_scale) == _sha(scale),
                     "Inner_Splits_SHA256": splits_sha(splits),
                     "Inner_Splits_Reproducible": splits_sha(inner_splits(len(x_train))) == splits_sha(splits),
                     "Inner_Indices_Within_Train": all(int(np.max(np.r_[a, b])) < len(x_train) and int(np.min(np.r_[a, b])) >= 0 for a, b in splits),
                     "Inner_Validation_Partitions_Train": all(
                         np.array_equal(np.sort(np.concatenate([splits[r * INNER_FOLDS + j][1] for j in range(INNER_FOLDS)])), np.arange(len(x_train)))
                         for r in range(INNER_REPEATS))}
            if all_rows is not None:
                a_mean = all_rows.mean(axis=0)
                a_std = all_rows.std(axis=0)
                mean_differs = _sha(a_mean) != _sha(mean)
                audit["NegControl_Mean_Differs"] = mean_differs
                audit["NegControl_Std_Differs"] = (_sha(np.where(a_std > 0, a_std, 1.0)) != _sha(scale)) if not mean_differs else np.nan
                audit["NegControl_Pass"] = bool(mean_differs or audit["NegControl_Std_Differs"] == True)  # noqa: E712
    for w in caught:
        counts[w.category.__name__] += 1
    pred = np.power(10.0, yhat)
    finite = bool(np.all(np.isfinite(coef)) and np.all(np.isfinite(pred)))
    return FitResult(pred, finite, counts, hyper, audit)


@dataclass
class CVResult:
    cells: list[str]
    predictions: dict[str, float]    # cell -> predicted cycles (outer 5-fold), all cells of the setting
    estimable: bool
    reason: str
    warnings: Counter
    hyper_rows: list[dict]
    audit_rows: list[dict]


def run_outer_cv(model: str, cells: list[str], eol: dict[str, float], design: dict[str, np.ndarray],
                 folds: dict[str, int]) -> CVResult:
    """Outer frozen 5-fold predictions for one definition and one cell set. `design[cell]` is the feature vector."""
    cells = sorted(cells)
    preds: dict[str, float] = {}
    counts: Counter = Counter()
    hyper_rows, audit_rows = [], []
    estimable, reason = True, ""
    x_all = np.vstack([design[c] for c in cells]) if cells else np.zeros((0, 0))
    for k in range(1, N_FOLDS + 1):
        train = [c for c in cells if folds[c] != k]
        test = [c for c in cells if folds[c] == k]
        if not test:
            continue
        disjoint = not (set(train) & set(test))
        if len(train) < INNER_FOLDS:  # KFold(4) needs at least 4 training cells
            estimable, reason = False, f"FOLD_{k}_TRAIN_CELLS_{len(train)}"
            break
        xtr = np.vstack([design[c] for c in train])
        xte = np.vstack([design[c] for c in test])
        res = fit_predict(model, xtr, np.array([eol[c] for c in train]), xte, all_rows=x_all)
        counts.update(res.warnings)
        audit_rows.append({"Model": model, "Fold_ID": k, "Train_Cells": len(train), "Test_Cells": len(test),
                           "Train_Test_Disjoint": disjoint, "Fit_Finite": res.finite, **res.audit})
        if res.hyper:
            hyper_rows.append({"Model": model, "Fold_ID": k, **res.hyper})
        if not res.finite:
            estimable, reason = False, f"FOLD_{k}_NONFINITE_FIT"
            break
        preds.update(dict(zip(test, res.predictions.tolist())))
    if not estimable:
        preds = {}
    return CVResult(cells, preds, estimable, reason, counts, hyper_rows, audit_rows)


def run_original_split(model: str, cells_by_split: dict[str, list[str]], eol: dict[str, float],
                       design: dict[str, np.ndarray]) -> tuple[dict[str, float], bool, Counter]:
    """C condition: fit on the labeled training cells of the Severson split, predict train / test1 / test2."""
    train = sorted(cells_by_split.get("train", []))
    others = [c for s in ("train", "test1", "test2") for c in sorted(cells_by_split.get(s, []))]
    if len(train) < INNER_FOLDS or not others:
        return {}, False, Counter()
    xtr = np.vstack([design[c] for c in train])
    xte = np.vstack([design[c] for c in others])
    res = fit_predict(model, xtr, np.array([eol[c] for c in train]), xte)
    return (dict(zip(others, res.predictions.tolist())) if res.finite else {}), res.finite, res.warnings
