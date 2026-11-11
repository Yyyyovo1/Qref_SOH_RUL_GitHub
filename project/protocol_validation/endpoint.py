"""Trap D: strict endpoint rule (v2) versus the tolerance rule of Lecu et al. (2026), BatteryKG qc.py @ 63b974c."""

from __future__ import annotations

import numpy as np
import pandas as pd

SMOOTH_WINDOW = 5
EOL_TOL = 0.01


def tolerance_rule(index: np.ndarray, capacity: np.ndarray, threshold: float) -> tuple[str, float]:
    """Rolling-median smoothing (window 5, centred, min_periods 1), linear interpolation of the first crossing,
    first-point rule, and the 1% termination tolerance on the last measured capacity (qc.py lines 60-90, 121-128)."""
    smoothed = pd.Series(capacity).rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).median().to_numpy()
    below = smoothed <= threshold
    if below.any():
        j = int(np.argmax(below))
        if j == 0:
            return "FIRST_BELOW", float(index[0])
        c0, c1 = smoothed[j - 1], smoothed[j]
        n0, n1 = index[j - 1], index[j]
        if c0 == c1:
            return "INTERPOLATED", float(n1)
        return "INTERPOLATED", float(n0 + (c0 - threshold) / (c0 - c1) * (n1 - n0))
    if capacity[-1] <= threshold * (1 + EOL_TOL):
        return "TOLERANCE", float(index[-1])
    return "NO_LABEL", np.nan


def tolerance_labels(data: dict, strict: pd.DataFrame) -> pd.DataFrame:
    """Apply the tolerance rule to every dataset x Qref x theta x cell of the strict label table."""
    series = {}
    for name, d in data.items():
        cap = d.capacity.sort_values(["Cell_ID", "Endpoint_Index", "Raw_Cycle_Index"], kind="stable")
        for c, g in cap.groupby("Cell_ID"):
            series[(name, c)] = (g.Endpoint_Index.to_numpy(float), g.Capacity_Ah.to_numpy(float))
    rows = []
    for r in strict.itertuples(index=False):
        idx, q = series[(r.Dataset, r.Cell_ID)]
        kind, eol = tolerance_rule(idx, q, r.Threshold_Ah)
        rows.append({"Dataset": r.Dataset, "Qref_Definition": r.Qref_Definition, "Theta": r.Theta, "Cell_ID": r.Cell_ID,
                     "Threshold_Ah": r.Threshold_Ah, "Strict_Status": r.Status, "Strict_EOL_Index": r.EOL_Exact_Index,
                     "Tolerance_Kind": kind, "Tolerance_EOL_Index": eol})
    out = pd.DataFrame(rows)
    out["Strict_Labeled"] = out.Strict_Status.eq("EXACT")
    out["Tolerance_Labeled"] = out.Tolerance_Kind.ne("NO_LABEL")
    out["EOL_Difference_Tolerance_Minus_Strict"] = np.where(out.Strict_Labeled & out.Tolerance_Labeled,
                                                            out.Tolerance_EOL_Index - out.Strict_EOL_Index, np.nan)
    return out


def setting_summary(lab: pd.DataFrame) -> pd.DataFrame:
    g = lab.groupby(["Dataset", "Qref_Definition", "Theta"])
    return pd.DataFrame({
        "Cells": g.size(), "Strict_Labeled": g.Strict_Labeled.sum(), "Tolerance_Labeled": g.Tolerance_Labeled.sum(),
        "Tolerance_Only_From_Tolerance_Clause": g.Tolerance_Kind.apply(lambda s: int((s == "TOLERANCE").sum())),
        "Both_Labeled": g.apply(lambda f: int((f.Strict_Labeled & f.Tolerance_Labeled).sum()), include_groups=False),
    }).reset_index()


def eol_difference_summary(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in lab.groupby("Dataset"):
        x = g.EOL_Difference_Tolerance_Minus_Strict.dropna().to_numpy()
        if not len(x):
            continue
        q = np.quantile(x, [0, 0.25, 0.5, 0.75, 1.0])
        rows.append({"Dataset": d, "Both_Labeled_Settings": len(x), "Min": q[0], "Q1": q[1], "Median": q[2], "Q3": q[3],
                     "Max": q[4], "Share_Abs_Le_1": float(np.mean(np.abs(x) <= 1))})
    return pd.DataFrame(rows)
