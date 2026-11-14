"""Experiment 2 (PROTOCOL_v4 section 3; AMENDMENT_01 sections 4, 5, 9): supported agreement (descriptive), equivalence
test (for H2) and detection power (descriptive; feeds the H3 low-power note). Formal run only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .blinding import evaluation
from .config import DELTA_FRACTION, DETECTION_FRACTIONS, DETECTION_SE_MULTIPLES, DETECTION_Z, H1_PAIRS, H3_PAIR
from .exp1 import interval90


@evaluation
def agreement(comp: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Section 3.1 on comparisons with both settings estimable: share where both settings determine a better method and
    it is the same; and, among comparisons where both determine one, the share that agree."""
    c = comp[comp.Estimable_Comparison].copy()
    c["Both_Determined"] = (c.Better_1.fillna("") != "") & (c.Better_2.fillna("") != "")
    c["Same_Better"] = c.Both_Determined & (c.Better_1 == c.Better_2)
    rows = []
    for key, g in ([("all", c)] if not group_cols else c.groupby(group_cols)):
        n, nd, ns = int(len(g)), int(g.Both_Determined.sum()), int(g.Same_Better.sum())
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,))) if group_cols else {"Scope": "all"}
        rows.append({**rec, "Estimable_Comparisons": n, "Both_Determined": nd, "Supported_Agreement": ns,
                     "Supported_Agreement_Share": ns / n if n else np.nan,
                     "Agreement_Share_Among_Determined": ns / nd if nd else np.nan})
    return pd.DataFrame(rows)


@evaluation
def classify(lo: float, hi: float, delta: float) -> str:
    if lo > -delta and hi < delta:
        return "EQUIVALENT"
    if hi < -delta or lo > delta:
        return "NOT_EQUIVALENT"
    return "IMPRECISE"


@evaluation
def equivalence_row(v1: np.ndarray, v2: np.ndarray, est1: bool, est2: bool, baseline: float) -> dict:
    """Amendment 01 section 5 order: delta undefined -> not estimable -> equivalent / not equivalent / imprecise."""
    if not np.isfinite(baseline):
        return {"Category": "DELTA_UNDEFINED", "Delta": np.nan}
    delta = DELTA_FRACTION * baseline
    if not (est1 and est2):
        return {"Category": "NOT_ESTIMABLE", "Delta": delta, "Reason": "SETTING_INTERVAL_NOT_ESTIMABLE"}
    r = interval90(v2 - v1)
    if not r["Estimable"]:
        return {"Category": "NOT_ESTIMABLE", "Delta": delta, "Reason": r["Reason"]}
    return {"Category": classify(r["CI90_Lower"], r["CI90_Upper"], delta), "Delta": delta,
            "CI90_Lower": r["CI90_Lower"], "CI90_Upper": r["CI90_Upper"], "Boot_Missing_Fraction": r["Boot_Missing_Fraction"]}


@evaluation
def detection_row(v1: np.ndarray, v2: np.ndarray, est1: bool, est2: bool, baseline: float, metric_is_error: bool = True) -> dict:
    """Section 3.3: P = Phi(m/SE1 - 1.96) * Phi(m/SE2 - 1.96). SE levels are constants; fraction levels need SE and
    a defined baseline (not for the C index, amendment 01 section 9)."""
    out = {f"P_{k:g}SE": float(norm.cdf(k - DETECTION_Z) ** 2) for k in DETECTION_SE_MULTIPLES}
    if not metric_is_error:
        return {**out, "SE_1": np.nan, "SE_2": np.nan, "Fraction_Levels": "NOT_APPLICABLE_C_INDEX"}
    if not (est1 and est2):
        return {**out, "SE_1": np.nan, "SE_2": np.nan, "Fraction_Levels": "NOT_ESTIMABLE"}
    se1 = float(np.nanstd(v1, ddof=1))
    se2 = float(np.nanstd(v2, ddof=1))
    out.update({"SE_1": se1, "SE_2": se2})
    if not np.isfinite(baseline) or se1 <= 0 or se2 <= 0:
        return {**out, "Fraction_Levels": "BASELINE_UNDEFINED" if not np.isfinite(baseline) else "ZERO_SE"}
    for f in DETECTION_FRACTIONS:
        m = f * baseline
        out[f"P_{int(round(f * 100))}pct"] = float(norm.cdf(m / se1 - DETECTION_Z) * norm.cdf(m / se2 - DETECTION_Z))
    return {**out, "Fraction_Levels": "OK"}


@evaluation
def exp1_equivalence_and_power(rev: pd.DataFrame, ci: pd.DataFrame, vectors: dict, base: pd.DataFrame) -> pd.DataFrame:
    """Experiment 1, decisive pairs (variance-discharge, variance-full), comparisons within the common windows."""
    est = ci.set_index(["Rule", "Pair", "Theta", "Qref_Definition", "Method_Pair"]).Estimable.to_dict()
    sest = ci.set_index(["Rule", "Pair", "Theta", "Qref_Definition", "Method_Pair"]).Setting_Estimable.to_dict()
    b = base.set_index(["Rule", "Pair", "Theta"]).Baseline_Mean_APE.to_dict()
    rows = []
    for r in rev[rev.Method_Pair.isin(H1_PAIRS)].itertuples(index=False):
        k1 = (r.Rule, r.Pair, r.Theta, r.Setting_1, r.Method_Pair)
        k2 = (r.Rule, r.Pair, r.Theta, r.Setting_2, r.Method_Pair)
        e1, e2 = bool(est[k1] and sest[k1]), bool(est[k2] and sest[k2])
        baseline = b.get((r.Rule, r.Pair, r.Theta), np.nan)
        rows.append({"Experiment": "1", "Dataset": "MIT", "Rule": r.Rule, "Pair": r.Pair, "Theta": r.Theta,
                     "Method_Pair": r.Method_Pair, "Eval_Set": "P_common_cells", "Baseline": baseline,
                     **equivalence_row(vectors[k1], vectors[k2], e1, e2, baseline),
                     **{f"Power_{k}": v for k, v in detection_row(vectors[k1], vectors[k2], e1, e2, baseline).items()}})
    return pd.DataFrame(rows)


@evaluation
def exp3_equivalence_and_power(comp: pd.DataFrame, vectors: dict, base: pd.DataFrame) -> pd.DataFrame:
    """Experiment 3, RF-GB within the strict common windows. Equivalence on E1 only; detection on E1, E2 (fraction
    levels) and E3 (SE levels only)."""
    b = base.set_index(["Dataset", "Theta", "Qref_Definition", "Eval_Set"]).BASE_Median_Error.to_dict()
    rows = []
    c = comp[comp.Pair.eq(H3_PAIR) & comp.In_Common_Window]
    for r in c.itertuples(index=False):
        rec = {"Experiment": "3", "Dataset": r.Dataset, "Rule": "strict", "Pair": r.Qref_Pair, "Theta": r.Theta,
               "Method_Pair": r.Pair, "Eval_Set": r.Eval_Set}
        if r.Eval_Set == "E3":
            pw = detection_row(None, None, r.Estimable_1, r.Estimable_2, np.nan, metric_is_error=False)
            rows.append({**rec, "Category": "NOT_IN_EQUIVALENCE_SCOPE", **{f"Power_{k}": v for k, v in pw.items()}})
            continue
        v1 = vectors[(r.Theta, r.Setting_1, r.Eval_Set, r.Pair, r.Dataset)]
        v2 = vectors[(r.Theta, r.Setting_2, r.Eval_Set, r.Pair, r.Dataset)]
        baseline = b.get((r.Dataset, r.Theta, "nominal", r.Eval_Set), np.nan)
        eq = equivalence_row(v1, v2, r.Estimable_1, r.Estimable_2, baseline) if r.Eval_Set == "E1" else {"Category": "NOT_IN_EQUIVALENCE_SCOPE"}
        pw = detection_row(v1, v2, r.Estimable_1, r.Estimable_2, baseline)
        rows.append({**rec, "Baseline": baseline, **eq, **{f"Power_{k}": v for k, v in pw.items()}})
    return pd.DataFrame(rows)


@evaluation
def power_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = [c for c in table.columns if c.startswith("Power_P_")]
    for (exp, d, e), g in table.groupby(["Experiment", "Dataset", "Eval_Set"]):
        for c in cols:
            v = pd.to_numeric(g[c], errors="coerce").dropna()
            rows.append({"Experiment": exp, "Dataset": d, "Eval_Set": e, "Level": c.replace("Power_", ""), "N": int(len(v)),
                         "Median": float(v.median()) if len(v) else np.nan,
                         "Q1": float(v.quantile(0.25)) if len(v) else np.nan, "Q3": float(v.quantile(0.75)) if len(v) else np.nan})
    return pd.DataFrame(rows)
