"""Evaluation sets, paired cell bootstrap and the preregistered decision rule (sections 5-7)."""

from __future__ import annotations

import hashlib
import warnings
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CAL_CONDITIONS,
    EXACT,
    INTERVAL,
    MAX_BOOT_MISSING_FRACTION,
    METHODS,
    METHODS_BY_CAL,
    MIN_CELLS,
    N_BOOTSTRAP,
    QREF_PAIRS,
    QREFS,
    RANDOM_SEED,
    RIGHT,
    STRONG_MIN_THETA_POINTS,
    THETAS,
    XJTU_DRAW_SHA256,
)
from .labels import PilotError


def draws_for(n_cells: int) -> tuple[np.ndarray, str]:
    draws = np.random.default_rng(RANDOM_SEED).integers(0, n_cells, size=(N_BOOTSTRAP, n_cells), dtype=np.int64)
    return draws, hashlib.sha256(draws.astype("<i8", copy=False).tobytes()).hexdigest()


def censoring_aware_error(pred: np.ndarray, status: np.ndarray, rul: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """E3 error: |e| for exact, distance to [lower, upper] for interval, max(0, c - pred) for right."""
    err = np.full(len(pred), np.nan)
    ex, iv, rt = status == EXACT, status == INTERVAL, status == RIGHT
    err[ex] = np.abs(pred[ex] - rul[ex])
    err[iv] = np.where(pred[iv] < lower[iv], lower[iv] - pred[iv], np.where(pred[iv] > upper[iv], pred[iv] - upper[iv], 0.0))
    err[rt] = np.maximum(0.0, lower[rt] - pred[rt])
    return err


def membership(labels: pd.DataFrame, dataset: str, theta: float, qref: str, eval_set: str, cells: list[str]) -> np.ndarray:
    sub = labels[(labels.Dataset == dataset) & (labels.Theta == theta)]
    if eval_set == "E1":
        ok = sub[sub.Eligible & sub.Status.eq(EXACT)].groupby("Cell_ID").Qref_Definition.nunique()
        members = set(ok[ok == len(QREFS)].index)
    else:
        own = sub[(sub.Qref_Definition == qref) & sub.Eligible]
        members = set(own[own.Status.eq(EXACT)].Cell_ID) if eval_set == "E2" else set(own.Cell_ID)
    return np.array([c in members for c in cells])


def error_vectors(labels: pd.DataFrame, predictions: pd.DataFrame, cells: list[str]) -> dict[tuple, np.ndarray]:
    """Per-cell error for every dataset x theta x Qref x cal x eval set x method (NaN outside the set)."""
    out: dict[tuple, np.ndarray] = {}
    for (d, theta, q, cal), p in predictions.groupby(["Dataset", "Theta", "Qref_Definition", "Cal"], sort=False):
        lab = labels[(labels.Dataset == d) & (labels.Theta == theta) & (labels.Qref_Definition == q)].set_index("Cell_ID").loc[cells[d]]
        p = p.set_index("Cell_ID").loc[cells[d]]
        st = lab.Status.to_numpy(object)
        rul, lo, hi = (lab[c].to_numpy(float) for c in ("RUL", "RUL_Lower", "RUL_Upper"))
        for e in ("E1", "E2", "E3"):
            member = membership(labels, d, theta, q, e, cells[d])
            for m in METHODS_BY_CAL[cal]:
                pred = p[f"Pred_{m}"].to_numpy(float)
                err = censoring_aware_error(pred, st, rul, lo, hi) if e == "E3" else np.abs(pred - rul)
                out[(d, theta, q, cal, e, m)] = np.where(member, err, np.nan)
    return out


def _boot_medians(err: np.ndarray, draws: np.ndarray) -> np.ndarray:
    sample = err[draws]
    valid = (~np.isnan(sample)).sum(axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(sample, axis=1)
    med[valid < MIN_CELLS] = np.nan
    return med


def pair_intervals(errors: dict[tuple, np.ndarray], draws: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for key in sorted({k[:5] for k in errors}, key=str):
        d, theta, q, cal, e = key
        for a, b in combinations([m for m in METHODS if m in METHODS_BY_CAL[cal]], 2):
            ea, eb = errors[key + (a,)], errors[key + (b,)]
            joint = ~np.isnan(ea) & ~np.isnan(eb)
            row = {"Dataset": d, "Theta": theta, "Qref_Definition": q, "Cal": cal, "Eval_Set": e, "Pair": f"{a}-{b}",
                   "Cells": int(joint.sum())}
            if joint.sum() < MIN_CELLS:
                rows.append({**row, "Estimable": False, "Reason": "FEWER_THAN_5_CELLS"})
                continue
            ea, eb = np.where(joint, ea, np.nan), np.where(joint, eb, np.nan)
            diff = _boot_medians(ea, draws[d]) - _boot_medians(eb, draws[d])
            missing = float(np.mean(np.isnan(diff)))
            if missing > MAX_BOOT_MISSING_FRACTION:
                rows.append({**row, "Estimable": False, "Reason": f"BOOT_MISSING_{missing:.3f}"})
                continue
            lo, hi = np.quantile(diff[~np.isnan(diff)], [0.025, 0.975], method="linear")
            rows.append({**row, "Estimable": True, "Reason": "",
                         "Observed_Difference": float(np.nanmedian(ea) - np.nanmedian(eb)),
                         "Median_A": float(np.nanmedian(ea)), "Median_B": float(np.nanmedian(eb)),
                         "CI95_Lower": float(lo), "CI95_Upper": float(hi), "Boot_Missing_Fraction": missing,
                         "CI_Sign": int(np.sign(lo)) if lo > 0 or hi < 0 else 0})
    return pd.DataFrame(rows)


def find_reversals(ci: pd.DataFrame) -> pd.DataFrame:
    """Supported reversals: both settings' CIs exclude zero with opposite signs (section 7.2)."""
    est = ci[ci.Estimable].set_index(["Dataset", "Theta", "Qref_Definition", "Cal", "Eval_Set", "Pair"])["CI_Sign"]
    sign = est.to_dict()
    rows = []
    keys = {(d, t, e, p) for d, t, _, _, e, p in est.index}
    for d, theta, e, pair in sorted(keys, key=str):
        comparisons = []
        for cal in CAL_CONDITIONS:
            for q1, q2 in QREF_PAIRS:
                comparisons.append(("Q", (q1, cal), (q2, cal)))
        if "TRAJ" not in pair:
            for q in QREFS:
                comparisons.append(("C", (q, "CAL"), (q, "NOCAL")))
        for kind, s1, s2 in comparisons:
            a = sign.get((d, theta, s1[0], s1[1], e, pair))
            b = sign.get((d, theta, s2[0], s2[1], e, pair))
            if a is None or b is None:
                continue
            rows.append({"Dataset": d, "Theta": theta, "Eval_Set": e, "Pair": pair, "Kind": kind,
                         "Setting_1": f"{s1[0]}|{s1[1]}", "Setting_2": f"{s2[0]}|{s2[1]}",
                         "Sign_1": a, "Sign_2": b, "Supported_Reversal": a != 0 and b != 0 and a == -b})
    return pd.DataFrame(rows)


def strong_rule(reversals: pd.DataFrame) -> pd.DataFrame:
    """|S(d, p, k)|: theta points with a supported E1 reversal confirmed (same settings, direction) on E2 or E3."""
    rev = reversals[reversals.Supported_Reversal]
    key = ["Dataset", "Theta", "Pair", "Kind", "Setting_1", "Setting_2", "Sign_1"]
    e1 = rev[rev.Eval_Set == "E1"][key].drop_duplicates()
    other = rev[rev.Eval_Set.isin(["E2", "E3"])][key].drop_duplicates()
    confirmed = e1.merge(other, on=key)
    rows = []
    for d in sorted(reversals.Dataset.unique()):
        for pair in sorted(reversals.Pair.unique()):
            for kind in ("Q", "C"):
                pts = sorted(confirmed[(confirmed.Dataset == d) & (confirmed.Pair == pair) & (confirmed.Kind == kind)].Theta.unique())
                e1_pts = sorted(e1[(e1.Dataset == d) & (e1.Pair == pair) & (e1.Kind == kind)].Theta.unique())
                rows.append({"Dataset": d, "Pair": pair, "Kind": kind, "S_Size": len(pts),
                             "S_Theta_Points": "|".join(f"{t:.2f}" for t in pts),
                             "E1_Only_Reversal_Theta_Points": "|".join(f"{t:.2f}" for t in e1_pts),
                             "Meets_Strong_Threshold": len(pts) >= STRONG_MIN_THETA_POINTS})
    return pd.DataFrame(rows)


def baseline_check(ci: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check 5: in E1/E2 estimable settings, best non-baseline minus BASE has CI entirely below 0 in >= half."""
    rows = []
    est = ci[ci.Estimable & ci.Eval_Set.isin(["E1", "E2"]) & ci.Pair.str.endswith("-BASE")]
    for key, g in est.groupby(["Dataset", "Theta", "Qref_Definition", "Cal", "Eval_Set"]):
        best = g.loc[g.Median_A.idxmin()]
        rows.append({"Dataset": key[0], "Theta": key[1], "Qref_Definition": key[2], "Cal": key[3], "Eval_Set": key[4],
                     "Best_Method": best.Pair.split("-")[0], "Best_Median": best.Median_A, "BASE_Median": best.Median_B,
                     "CI95_Lower": best.CI95_Lower, "CI95_Upper": best.CI95_Upper, "Best_Clearly_Better": best.CI95_Upper < 0})
    detail = pd.DataFrame(rows)
    share = float(detail.Best_Clearly_Better.mean()) if len(detail) else 0.0
    return detail, {"Settings": len(detail), "Clearly_Better_Share": share, "Pass": share >= 0.5}


def decide(strong: pd.DataFrame, check5: dict[str, Any]) -> str:
    if not check5["Pass"]:
        return "UNINFORMATIVE"
    return "STRONG" if strong.Meets_Strong_Threshold.any() else "WEAK"


def theta_grid_check() -> None:
    if len(THETAS) != 21 or THETAS[0] != 0.75 or THETAS[-1] != 0.95:
        raise PilotError("THETA_GRID")


def check_draw_hash(name: str, digest: str) -> None:
    if name == "XJTU" and digest != XJTU_DRAW_SHA256:
        raise PilotError("XJTU_DRAWS_DIFFER_FROM_S6_S7")
