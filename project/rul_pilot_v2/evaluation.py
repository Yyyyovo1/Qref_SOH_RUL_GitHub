"""Evaluation sets, C index, shared-draw cell bootstrap and the decision rule (PROTOCOL_v2 sections 1e, 6, 7)."""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from rul_ranking_pilot.evaluation import censoring_aware_error, draws_for

from .config import (
    C_INDEX_MIN_PAIRS,
    EXACT,
    MAX_BOOT_MISSING_FRACTION,
    METHODS,
    MIN_CELLS,
    QREF_PAIRS,
    QREFS,
    STRONG_MIN_THETA_POINTS,
)

PAIRS = [f"{a}-{b}" for a, b in combinations(METHODS, 2)]


def counts_matrix(draws: np.ndarray, n: int) -> np.ndarray:
    """Multiplicity of each cell in each bootstrap resample (resamples x cells)."""
    w = np.zeros((draws.shape[0], n))
    np.add.at(w, (np.repeat(np.arange(draws.shape[0]), draws.shape[1]), draws.ravel()), 1.0)
    return w


def _nanmedian_rows(values: np.ndarray) -> np.ndarray:
    valid = (~np.isnan(values)).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(values, axis=1)
    med[valid < MIN_CELLS] = np.nan
    return med


def _interval(diff: np.ndarray) -> dict:
    missing = float(np.mean(np.isnan(diff)))
    if missing > MAX_BOOT_MISSING_FRACTION:
        return {"Estimable": False, "Reason": f"BOOT_MISSING_{missing:.3f}"}
    lo, hi = np.quantile(diff[~np.isnan(diff)], [0.025, 0.975], method="linear")
    return {"Estimable": True, "Reason": "", "CI95_Lower": float(lo), "CI95_Upper": float(hi), "Boot_Missing_Fraction": missing}


def _better(row: dict, a: str, b: str, higher_is_better: bool) -> str:
    if not row.get("Estimable"):
        return ""
    lo, hi = row["CI95_Lower"], row["CI95_Upper"]
    if lo > 0:
        return a if higher_is_better else b
    if hi < 0:
        return b if higher_is_better else a
    return ""


class Arrays:
    """Per-setting arrays over cells x 3 t0: predictions, bounds, eligibility (NaN-safe)."""

    def __init__(self, cells: list[str], samples: pd.DataFrame, preds: pd.DataFrame):
        idx = {c: i for i, c in enumerate(cells)}
        n = len(cells)
        self.elig = np.zeros((n, 3), bool)
        self.status = np.full((n, 3), "", object)
        self.rul, self.lo, self.hi = (np.full((n, 3), np.nan) for _ in range(3))
        for r in samples.itertuples(index=False):
            i, k = idx[r.Cell_ID], r.t_Index
            self.elig[i, k] = r.Eligible
            self.status[i, k] = r.Status
            self.rul[i, k], self.lo[i, k], self.hi[i, k] = r.RUL, r.RUL_Lower, r.RUL_Upper
        self.pred = {m: np.full((n, 3), np.nan) for m in METHODS}
        if len(preds):
            ii = preds.Cell_ID.map(idx).to_numpy(int)
            kk = preds.t_Index.to_numpy(int)
            for m in METHODS:
                self.pred[m][ii, kk] = preds[f"Pred_{m}"].to_numpy(float)


def per_cell_errors(arr: Arrays, member: np.ndarray, t_mask: np.ndarray, a: str, b: str, one_sided: bool = False):
    """Within-cell mean absolute error over t0 where both methods predict (cells outside `member` -> NaN)."""
    pa, pb = arr.pred[a], arr.pred[b]
    use = t_mask & arr.elig & ~np.isnan(pa) & ~np.isnan(pb)
    if one_sided:
        ea = np.vstack([censoring_aware_error(pa[:, k], arr.status[:, k], arr.rul[:, k], arr.lo[:, k], arr.hi[:, k]) for k in range(3)]).T
        eb = np.vstack([censoring_aware_error(pb[:, k], arr.status[:, k], arr.rul[:, k], arr.lo[:, k], arr.hi[:, k]) for k in range(3)]).T
    else:
        ea, eb = np.abs(pa - arr.rul), np.abs(pb - arr.rul)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ca = np.nanmean(np.where(use, ea, np.nan), axis=1)
        cb = np.nanmean(np.where(use, eb, np.nan), axis=1)
    keep = member & ~np.isnan(ca) & ~np.isnan(cb)
    return np.where(keep, ca, np.nan), np.where(keep, cb, np.nan)


def error_pair_ci(ea: np.ndarray, eb: np.ndarray, draws: np.ndarray) -> dict:
    n = int((~np.isnan(ea)).sum())
    out = {"Cells": n}
    if n < MIN_CELLS:
        return {**out, "Estimable": False, "Reason": "FEWER_THAN_5_CELLS"}
    diff = _nanmedian_rows(ea[draws]) - _nanmedian_rows(eb[draws])
    return {**out, "Observed_A": float(np.nanmedian(ea)), "Observed_B": float(np.nanmedian(eb)),
            "Observed_Difference": float(np.nanmedian(ea) - np.nanmedian(eb)), **_interval(diff)}


def _c_parts(arr: Arrays, k: int, a: str, b: str):
    pa, pb = arr.pred[a][:, k], arr.pred[b][:, k]
    m = arr.elig[:, k] & ~np.isnan(pa) & ~np.isnan(pb)
    lo, hi = arr.lo[:, k], arr.hi[:, k]
    with np.errstate(invalid="ignore"):
        comp = (hi[:, None] < lo[None, :]) & m[:, None] & m[None, :]  # i's EOL known to precede j's
    conc = lambda p: np.where(p[:, None] < p[None, :], 1.0, np.where(p[:, None] == p[None, :], 0.5, 0.0))
    k_ = comp.astype(float)
    return k_, k_ * conc(np.nan_to_num(pa)), k_ * conc(np.nan_to_num(pb))


def c_index_pair_ci(arr: Arrays, a: str, b: str, w: np.ndarray) -> dict:
    """E3: C(A) - C(B), averaged over t0 with >= 10 comparable pairs; bootstrap via count-weighted pair sums."""
    obs_a, obs_b, boot_a, boot_b, pairs = [], [], [], [], []
    for k in range(3):
        kk, na, nb = _c_parts(arr, k, a, b)
        total = kk.sum()
        if total < C_INDEX_MIN_PAIRS:
            continue
        pairs.append(int(total))
        obs_a.append(na.sum() / total)
        obs_b.append(nb.sum() / total)
        den = ((w @ kk) * w).sum(axis=1)
        num_a = ((w @ na) * w).sum(axis=1)
        num_b = ((w @ nb) * w).sum(axis=1)
        ok = den >= C_INDEX_MIN_PAIRS
        boot_a.append(np.where(ok, num_a / np.where(ok, den, 1.0), np.nan))
        boot_b.append(np.where(ok, num_b / np.where(ok, den, 1.0), np.nan))
    out = {"Cells": int(arr.elig.any(axis=1).sum()), "Comparable_Pairs_per_t": "|".join(map(str, pairs))}
    if not pairs:
        return {**out, "Estimable": False, "Reason": "NO_t0_WITH_10_COMPARABLE_PAIRS"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ca, cb = np.nanmean(np.vstack(boot_a), axis=0), np.nanmean(np.vstack(boot_b), axis=0)
    return {**out, "Observed_A": float(np.mean(obs_a)), "Observed_B": float(np.mean(obs_b)),
            "Observed_Difference": float(np.mean(obs_a) - np.mean(obs_b)), **_interval(ca - cb)}


def membership(samples_dt: pd.DataFrame, cells: list[str], q: str, kind: str, t_k: int | None = None) -> np.ndarray:
    """E1 / E1_t / E2 / E3 membership for one dataset x theta (samples_dt holds all Qrefs and t0)."""
    if kind in ("E1", "E1_t"):
        t_set = [2] if kind == "E1" else [t_k]
        s = samples_dt[samples_dt.t_Index.isin(t_set)]
        ok = s.assign(ok=s.Eligible & s.Status.eq(EXACT)).groupby("Cell_ID").ok.agg(["all", "size"])
        need = len(QREFS) * len(t_set)
        members = set(ok[(ok["all"]) & (ok["size"] == need)].index)
    else:
        s = samples_dt[samples_dt.Qref_Definition.eq(q)]
        s = s[s.Eligible & (s.Status.eq(EXACT) if kind == "E2" else True)]
        members = set(s.Cell_ID)
    return np.array([c in members for c in cells])


def find_reversals(ci: pd.DataFrame) -> pd.DataFrame:
    """Q-type comparisons; a supported reversal = both settings determine a better method and they differ."""
    better = ci.set_index(["Dataset", "Theta", "Qref_Definition", "Eval_Set", "Pair"])["Better"].to_dict()
    rows = []
    for (d, theta, e, pair) in sorted({(r.Dataset, r.Theta, r.Eval_Set, r.Pair) for r in ci.itertuples()}, key=str):
        for q1, q2 in QREF_PAIRS:
            b1, b2 = better.get((d, theta, q1, e, pair)), better.get((d, theta, q2, e, pair))
            if b1 is None or b2 is None:
                continue
            rows.append({"Dataset": d, "Theta": theta, "Eval_Set": e, "Pair": pair, "Setting_1": q1, "Setting_2": q2,
                         "Better_1": b1, "Better_2": b2, "Supported_Reversal": bool(b1 and b2 and b1 != b2)})
    return pd.DataFrame(rows)


def strong_rule(rev: pd.DataFrame) -> pd.DataFrame:
    sup = rev[rev.Supported_Reversal]
    key = ["Dataset", "Theta", "Pair", "Setting_1", "Setting_2", "Better_1", "Better_2"]
    e1 = sup[sup.Eval_Set.eq("E1")][key].drop_duplicates()
    other = sup[sup.Eval_Set.isin(["E2", "E3"])][key].drop_duplicates()
    confirmed = e1.merge(other, on=key)
    rows = []
    for d in sorted(rev.Dataset.unique()):
        for pair in PAIRS:
            pts = sorted(confirmed[(confirmed.Dataset == d) & (confirmed.Pair == pair)].Theta.unique())
            e1p = sorted(e1[(e1.Dataset == d) & (e1.Pair == pair)].Theta.unique())
            rows.append({"Dataset": d, "Pair": pair, "S_Size": len(pts), "S_Theta_Points": "|".join(f"{t:.2f}" for t in pts),
                         "E1_Reversal_Theta_Points": "|".join(f"{t:.2f}" for t in e1p),
                         "Meets_Strong_Threshold": len(pts) >= STRONG_MIN_THETA_POINTS})
    return pd.DataFrame(rows)


def baseline_check(ci: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    est = ci[ci.Estimable & ci.Eval_Set.isin(["E1", "E2"]) & ci.Pair.str.endswith("-BASE")]
    rows = []
    for key, g in est.groupby(["Dataset", "Theta", "Qref_Definition", "Eval_Set"]):
        best = g.loc[g.Observed_A.idxmin()]
        rows.append({"Dataset": key[0], "Theta": key[1], "Qref_Definition": key[2], "Eval_Set": key[3],
                     "Best_Method": best.Pair.split("-")[0], "Best_Median": best.Observed_A, "BASE_Median": best.Observed_B,
                     "CI95_Lower": best.CI95_Lower, "CI95_Upper": best.CI95_Upper, "Best_Clearly_Better": bool(best.CI95_Upper < 0)})
    detail = pd.DataFrame(rows)
    share = float(detail.Best_Clearly_Better.mean()) if len(detail) else 0.0
    return detail, {"Settings": len(detail), "Clearly_Better_Share": share, "Pass": share >= 0.5}


__all__ = ["Arrays", "PAIRS", "baseline_check", "c_index_pair_ci", "counts_matrix", "draws_for", "error_pair_ci",
           "find_reversals", "membership", "per_cell_errors", "strong_rule"]
