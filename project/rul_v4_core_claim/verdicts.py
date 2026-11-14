"""H1, H2, H3 (PROTOCOL_v4 section 5; AMENDMENT_01 sections 3, 5, 8, 9, 10). Formal run only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .blinding import evaluation
from .config import (
    H1_PAIRS,
    H1_REFUTE_THETA_POINTS,
    H1_SUPPORT_MIN,
    H2_EQUIVALENT_MIN,
    H2_IMPRECISE_MAX,
    H2_NONEQUIVALENT_MAX,
    H3_LOW_POWER_MEDIAN,
    H3_MAX_REVERSALS,
    H3_PAIR,
)

HOLDS, REFUTED, UNDECIDED, UNTESTABLE = "HOLDS", "REFUTED", "UNDECIDED", "UNTESTABLE"


@evaluation
def h1(rev: pd.DataFrame, units: pd.DataFrame, exp1_ok: bool, reason_if_not: str = "") -> dict:
    if not exp1_ok:
        return {"Verdict": UNTESTABLE, "Reason": reason_if_not}
    d = rev[rev.Method_Pair.isin(H1_PAIRS) & rev.Estimable_Comparison]
    sup = d[d.Supported_Reversal]
    pts = sup.groupby(["Rule", "Pair", "Method_Pair"]).Theta.nunique()
    refuted = bool((pts >= H1_REFUTE_THETA_POINTS).any())
    est = units[units.Estimable]
    share = float(est.Claim_Supported.mean()) if len(est) else np.nan
    holds = len(sup) == 0 and len(est) > 0 and share >= H1_SUPPORT_MIN
    verdict = REFUTED if refuted else (HOLDS if holds else UNDECIDED)
    return {"Verdict": verdict, "Supported_Reversals": int(len(sup)), "Estimable_Comparisons": int(len(d)),
            "Reversal_Theta_Points_By_Rule_Pair_Method": {"|".join(k): int(v) for k, v in pts.items()},
            "Estimable_Units": int(len(est)), "Claim_Supported_Units": int(est.Claim_Supported.sum()),
            "Claim_Supported_Share": share,
            "By_Claim": {c: {"estimable": int(len(g)), "supported": int(g.Claim_Supported.sum())} for c, g in est.groupby("Claim")}}


@evaluation
def h2(eq_rows: pd.DataFrame, exp1_ok: bool) -> dict:
    """Comparisons in scope: experiment 1 decisive pairs (if experiment 1 takes part) and experiment 3 RF-GB on E1."""
    scope = eq_rows[(eq_rows.Experiment.eq("1") & exp1_ok) | (eq_rows.Experiment.eq("3") & eq_rows.Eval_Set.eq("E1"))]
    scope = scope[scope.Category.ne("NOT_IN_EQUIVALENCE_SCOPE")]
    counts = scope.Category.value_counts().to_dict()
    used = scope[scope.Category.isin(["EQUIVALENT", "NOT_EQUIVALENT", "IMPRECISE"])]
    n = int(len(used))
    share = {k: (int((used.Category == k).sum()) / n if n else np.nan) for k in ("EQUIVALENT", "NOT_EQUIVALENT", "IMPRECISE")}
    if n == 0:
        verdict = UNTESTABLE
    elif share["NOT_EQUIVALENT"] > H2_NONEQUIVALENT_MAX:
        verdict = REFUTED
    elif share["IMPRECISE"] > H2_IMPRECISE_MAX:
        verdict = UNTESTABLE
    elif share["EQUIVALENT"] >= H2_EQUIVALENT_MIN:
        verdict = HOLDS
    else:
        verdict = UNDECIDED
    return {"Verdict": verdict, "Experiment_1_Included": bool(exp1_ok), "Denominator": n, "Shares": share,
            "Counts": {k: int(v) for k, v in counts.items()},
            "By_Experiment": {e: g.Category.value_counts().to_dict() for e, g in scope.groupby("Experiment")}}


@evaluation
def h3(comp: pd.DataFrame, power: pd.DataFrame, datasets_used: list[str]) -> dict:
    if not datasets_used:
        return {"Verdict": UNTESTABLE, "Reason": "NO_DATASET_PASSED_CHECKS_0_TO_5"}
    c = comp[comp.Pair.eq(H3_PAIR) & comp.In_Common_Window & comp.Estimable_Comparison & comp.Dataset.isin(datasets_used)]
    n_rev = int(c.Supported_Reversal.sum())
    p = power[power.Experiment.eq("3") & power.Dataset.isin(datasets_used) & power.Eval_Set.isin(["E1", "E2"])]
    p20 = pd.to_numeric(p.get("Power_P_20pct", pd.Series(dtype=float)), errors="coerce").dropna()
    med = float(p20.median()) if len(p20) else np.nan
    return {"Verdict": HOLDS if n_rev <= H3_MAX_REVERSALS else REFUTED, "Datasets_Used": list(datasets_used),
            "Supported_Reversals": n_rev, "Estimable_Comparisons": int(len(c)),
            "By_Dataset": {d: {"estimable": int(len(g)), "reversals": int(g.Supported_Reversal.sum())} for d, g in c.groupby("Dataset")},
            "Power_20pct_Median_E1_E2": med, "Power_20pct_N": int(len(p20)),
            "Low_Power_Note": bool(np.isfinite(med) and med < H3_LOW_POWER_MEDIAN)}
