"""RUL v2 part: trap A (E1/E2 surface changes and noise) and trap C (E3 one-sided error vs C index)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rul_pilot_v2.config import QREF_PAIRS, QREFS
from rul_pilot_v2.evaluation import PAIRS, Arrays, c_index_pair_ci, counts_matrix, membership, per_cell_errors

from .common import _better, _interval, _nanmedian_rows, draws_for, noise_baselines, s7_frequencies, surface_change


def load_v2(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = root / "outputs/rul_ranking_pilot_v2"
    kw = dict(encoding="utf-8-sig", low_memory=False, float_precision="round_trip")
    samples = pd.read_csv(out / "samples.csv", **kw)
    preds = pd.read_csv(out / "predictions.csv", **kw)
    for f in (samples, preds):
        f["Theta"] = f.Theta.round(2)
    return samples, preds


def _error_diff(ea: np.ndarray, eb: np.ndarray, draws: np.ndarray) -> tuple[dict, np.ndarray | None]:
    """Same statistic and rules as rul_pilot_v2.evaluation.error_pair_ci, also returning the bootstrap differences."""
    n = int((~np.isnan(ea)).sum())
    if n < 5:
        return {"Cells": n, "Estimable": False, "Reason": "FEWER_THAN_5_CELLS"}, None
    diff = _nanmedian_rows(ea[draws]) - _nanmedian_rows(eb[draws])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        obs = float(np.nanmedian(ea) - np.nanmedian(eb))
    return {"Cells": n, "Observed_A": float(np.nanmedian(ea)), "Observed_B": float(np.nanmedian(eb)),
            "Observed_Difference": obs, **_interval(diff)}, diff


def _mean_prediction(arr: Arrays, a: str, b: str) -> tuple[float, float]:
    """Within-cell then across-cell mean prediction over eligible (cell, t0) samples where both methods predict."""
    pa, pb = arr.pred[a], arr.pred[b]
    use = arr.elig & ~np.isnan(pa) & ~np.isnan(pb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ca = np.nanmean(np.where(use, pa, np.nan), axis=1)
        cb = np.nanmean(np.where(use, pb, np.nan), axis=1)
        return float(np.nanmean(ca)), float(np.nanmean(cb))


def rul_settings(samples: pd.DataFrame, preds: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Recompute E1 / E2 / E3(C index) intervals, plus E3 one-sided error intervals and mean predictions."""
    rows, diffs = [], {}
    for d in ["XJTU", "HUST", "MIT"]:
        sd = samples[samples.Dataset == d]
        pdd = preds[preds.Dataset == d]
        cells = sorted(sd.Cell_ID.unique())
        draws, _ = draws_for(len(cells))
        w = counts_matrix(draws, len(cells))
        all_t = np.ones((len(cells), 3), bool)
        for theta in sorted(sd.Theta.unique()):
            s_dt = sd[sd.Theta == theta]
            m_e1 = membership(s_dt, cells, "", "E1")
            for q in QREFS:
                arr = Arrays(cells, s_dt[s_dt.Qref_Definition == q], pdd[(pdd.Theta == theta) & (pdd.Qref_Definition == q)])
                m_e2, m_e3 = membership(s_dt, cells, q, "E2"), membership(s_dt, cells, q, "E3")
                for pair in PAIRS:
                    a, b = pair.split("-")
                    base = {"Dataset": d, "Theta": theta, "Qref_Definition": q, "Pair": pair}
                    mean_a, mean_b = _mean_prediction(arr, a, b)
                    means = {"Mean_Pred_A": mean_a, "Mean_Pred_B": mean_b}
                    for e, member, one_sided in (("E1", m_e1, False), ("E2", m_e2, False), ("E3_ONE_SIDED", m_e3, True)):
                        r, diff = _error_diff(*per_cell_errors(arr, member, all_t, a, b, one_sided=one_sided), draws)
                        r["Better"] = _better(r, a, b, higher_is_better=False)
                        rows.append({**base, "Eval_Set": e, **r, **(means if one_sided else {})})
                        if diff is not None and r.get("Estimable"):
                            diffs[(d, theta, q, e, pair)] = (r["Observed_Difference"], diff)
                    r3 = c_index_pair_ci(arr, a, b, w)
                    r3["Better"] = _better(r3, a, b, higher_is_better=True)
                    rows.append({**base, "Eval_Set": "E3", **r3, **means})
    return pd.DataFrame(rows), diffs


def comparisons(settings: pd.DataFrame, eval_sets: tuple[str, ...], diffs: dict | None = None) -> pd.DataFrame:
    """Q-type comparisons (v2 rule): surface change from point estimates, supported reversal from v2 'better method'."""
    idx = settings.set_index(["Dataset", "Theta", "Qref_Definition", "Eval_Set", "Pair"])
    rows = []
    for (d, theta, e, pair) in sorted({(r.Dataset, r.Theta, r.Eval_Set, r.Pair) for r in settings.itertuples()
                                       if r.Eval_Set in eval_sets}, key=str):
        for q1, q2 in QREF_PAIRS:
            r1, r2 = idx.loc[(d, theta, q1, e, pair)], idx.loc[(d, theta, q2, e, pair)]
            if not (r1.Estimable and r2.Estimable):
                continue
            o1, o2 = float(r1.Observed_Difference), float(r2.Observed_Difference)
            b1, b2 = (r1.Better or ""), (r2.Better or "")
            row = {"Dataset": d, "Theta": theta, "Eval_Set": e, "Pair": pair, "Setting_1": q1, "Setting_2": q2,
                   "Observed_1": o1, "Observed_2": o2, "Surface_Change": surface_change(o1, o2),
                   "Better_1": b1, "Better_2": b2, "Supported_Reversal": bool(b1 and b2 and b1 != b2)}
            if diffs is not None:
                row.update(s7_frequencies(diffs[(d, theta, q1, e, pair)][1], diffs[(d, theta, q2, e, pair)][1], o1, o2))
            rows.append(row)
    return pd.DataFrame(rows)


def noise_table(diffs: dict) -> pd.DataFrame:
    rows = [{"Dataset": k[0], "Theta": k[1], "Qref_Definition": k[2], "Eval_Set": k[3], "Pair": k[4], **noise_baselines(v[1])}
            for k, v in diffs.items()]
    return pd.DataFrame(rows)


def bias_table(settings: pd.DataFrame) -> pd.DataFrame:
    """Trap C: share of determined settings in which the method with the larger mean prediction is judged better."""
    s = settings[settings.Eval_Set.isin(["E3", "E3_ONE_SIDED"]) & settings.Estimable.astype(bool)].copy()
    s = s[s.Better.fillna("").ne("") & s.Mean_Pred_A.notna() & s.Mean_Pred_B.notna() & (s.Mean_Pred_A != s.Mean_Pred_B)]
    s["Larger_Prediction_Method"] = np.where(s.Mean_Pred_A > s.Mean_Pred_B, s.Pair.str.split("-").str[0], s.Pair.str.split("-").str[1])
    s["Larger_Prediction_Judged_Better"] = s.Larger_Prediction_Method == s.Better
    return s
