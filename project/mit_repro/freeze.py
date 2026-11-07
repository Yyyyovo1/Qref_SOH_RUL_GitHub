"""Cycle QC, Qref, endpoint labels, check 1, rebound counts, folds and t0 for MIT/Severson."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from eol_threshold_robustness_repro.config import EXACT, INTERVAL, LEFT, RIGHT
from eol_threshold_robustness_repro.pipeline import detect_hust_eol

from .config import (
    EXPECTED_DISCHARGE_CURRENT_C,
    N_FOLDS,
    NOMINAL_CAPACITY_AH,
    QC,
    QREFS,
    RANDOM_SEED,
    REBOUND_THETAS,
    T0_FRACTIONS,
    THETAS,
)


def qc_cycles(cycles: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen QC rules; capacity jumps are flagged only (HUST rule)."""
    out = cycles.sort_values(["Cell_ID", "Cycle_Index"], kind="stable").reset_index(drop=True).copy()
    expected = EXPECTED_DISCHARGE_CURRENT_C
    tol = QC["protocol_discharge_current_relative_tolerance"] * abs(expected)
    reasons = [[] for _ in range(len(out))]
    q = out["QDischarge_Ah"].to_numpy(float)
    cur = out["Discharge_Median_Current_C"].to_numpy(float)
    for i in range(len(out)):
        if QC["exclude_cycle_index_zero"] and out.at[i, "Cycle_Index"] == 0:
            reasons[i].append("CYCLE_INDEX_ZERO")
        if not np.isfinite(q[i]) or q[i] <= 0:
            reasons[i].append("CAPACITY_NONFINITE_OR_NONPOSITIVE")
        elif q[i] > QC["max_capacity_factor_of_nominal"] * NOMINAL_CAPACITY_AH:
            reasons[i].append("CAPACITY_ABOVE_1P5_NOMINAL")
        if not np.isfinite(cur[i]):
            reasons[i].append("NO_DISCHARGE_CURRENT_DETECTED")
        elif abs(cur[i] - expected) > tol:
            reasons[i].append("DISCHARGE_NOT_4C_PROTOCOL_INCONSISTENT")
    if out.duplicated(["Cell_ID", "Cycle_Index"]).any():
        raise ValueError("DUPLICATE_CELL_CYCLE_INDEX")
    out["Exclusion_Reasons"] = ["|".join(r) for r in reasons]
    out["Primary_Capacity_Eligible"] = np.where(out["Exclusion_Reasons"].eq(""), "YES", "NO")

    out["Capacity_Jump_Flag"] = "NO"
    out["Jump_Reference_Cycle"] = np.nan
    out["Capacity_Jump_Ah"] = np.nan
    for _, g in out[out.Primary_Capacity_Eligible.eq("YES")].groupby("Cell_ID", sort=False):
        ref = None
        for i in g.index:
            if ref is not None:
                delta = q[i] - q[ref]
                out.at[i, "Jump_Reference_Cycle"] = out.at[ref, "Cycle_Index"]
                out.at[i, "Capacity_Jump_Ah"] = delta
                if abs(delta) > max(QC["capacity_jump_abs_threshold_Ah"], QC["capacity_jump_relative_threshold"] * abs(q[ref])):
                    out.at[i, "Capacity_Jump_Flag"] = "YES"
            ref = i
    return out


def qref_table(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell, g in primary.groupby("Cell_ID", sort=True):
        g = g.sort_values("Cycle_Index")
        first5 = g.head(5)
        rows.append({
            "Cell_ID": cell,
            "Qref_Nominal_Ah": NOMINAL_CAPACITY_AH,
            "Qref_First_Comparable_Ah": float(g["QDischarge_Ah"].iloc[0]),
            "Qref_First_Comparable_Source_Cycle": int(g["Cycle_Index"].iloc[0]),
            "Qref_Early_Median5_Comparable_Ah": float(np.median(first5["QDischarge_Ah"])),
            "Qref_Early_Median5_Comparable_Source_Cycles": "|".join(str(int(c)) for c in first5["Cycle_Index"]),
            "Retained_Cycles": len(g),
            "Last_Retained_Cycle": int(g["Cycle_Index"].iloc[-1]),
            "Last_Retained_Capacity_Ah": float(g["QDischarge_Ah"].iloc[-1]),
        })
    return pd.DataFrame(rows)


QREF_COLUMN = {"nominal": "Qref_Nominal_Ah", "first_comparable": "Qref_First_Comparable_Ah",
               "early_median5_comparable": "Qref_Early_Median5_Comparable_Ah"}
STATUS = {EXACT: "EXACT", INTERVAL: "INTERVAL", RIGHT: "RIGHT", LEFT: "BASELINE"}


def endpoint_labels(primary: pd.DataFrame, qref: pd.DataFrame) -> pd.DataFrame:
    """Frozen detector (HUST rules): first retained cycle with capacity <= theta*Qref; no smoothing."""
    obs = primary.rename(columns={"Cycle_Index": "Raw_Cycle_Index", "QDischarge_Ah": "Primary_Capacity_Ah"})
    groups = {c: g[["Raw_Cycle_Index", "Primary_Capacity_Ah"]] for c, g in obs.groupby("Cell_ID")}
    rows = []
    for r in qref.itertuples(index=False):
        for q in QREFS:
            value = float(getattr(r, QREF_COLUMN[q]))
            for theta in THETAS:
                threshold = theta * value
                d = detect_hust_eol(groups[r.Cell_ID], threshold)
                first_hit = d.exact_cycle if d.exact_cycle is not None else (d.upper_bound if d.status in (INTERVAL, LEFT) else None)
                rows.append({"Cell_ID": r.Cell_ID, "Qref_Definition": q, "Theta": theta, "Qref_Ah": value,
                             "Threshold_Ah": threshold, "Status": STATUS[d.status],
                             "First_Hit_Cycle": np.nan if first_hit is None else float(first_hit),
                             "EOL_Lower_Cycle": float(d.lower_bound) if d.status == INTERVAL else np.nan,
                             "Last_Cycle": np.nan if d.last_cycle is None else float(d.last_cycle),
                             "Last_Capacity_Ah": np.nan if d.last_capacity_Ah is None else float(d.last_capacity_Ah)})
    return pd.DataFrame(rows)


def rebound_counts(primary: pd.DataFrame, labels: pd.DataFrame, thetas=REBOUND_THETAS) -> pd.DataFrame:
    """Settings (cell x Qref x theta) whose capacity returns above the threshold after the first hit."""
    cap = {c: g.sort_values("Cycle_Index") for c, g in primary.groupby("Cell_ID")}
    rows = []
    for r in labels[labels.Theta.isin(thetas) & labels.Status.isin(["EXACT", "INTERVAL"])].itertuples(index=False):
        g = cap[r.Cell_ID]
        after = g[g.Cycle_Index > r.First_Hit_Cycle]
        rows.append({"Qref_Definition": r.Qref_Definition, "Theta": r.Theta, "Status": r.Status, "Cell_ID": r.Cell_ID,
                     "Rebound": bool((after.QDischarge_Ah > r.Threshold_Ah).any())})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    return detail.groupby(["Status"]).Rebound.agg(Rebound_Settings="sum", Settings="size").reset_index()


def check1_cycle_life(labels: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Check 1 (PROTOCOL_v2 section 8): nominal theta=0.80 EXACT first hit vs authors' cycle_life."""
    sub = labels[(labels.Qref_Definition == "nominal") & np.isclose(labels.Theta, 0.80)]
    m = sub.merge(cells[["Cell_ID", "Author_Cycle_Life"]], on="Cell_ID")
    m["Difference_Cycles"] = m["First_Hit_Cycle"] - m["Author_Cycle_Life"]
    m["Last_Capacity_Minus_0p88_Ah"] = m["Last_Capacity_Ah"] - 0.88
    m["Within_1_Cycle"] = np.where(m.Status.eq("EXACT"), m["Difference_Cycles"].abs() <= 1, np.nan)
    return m[["Cell_ID", "Status", "First_Hit_Cycle", "Author_Cycle_Life", "Difference_Cycles", "Within_1_Cycle",
              "Last_Cycle", "Last_Capacity_Ah", "Last_Capacity_Minus_0p88_Ah"]]


def assign_folds(cells: list[str]) -> pd.DataFrame:
    """Same algorithm as soh_ml_design_v1 (HUST): SHA256-seeded cell order, round robin."""
    order = sorted(set(cells), key=lambda c: hashlib.sha256(f"{RANDOM_SEED}|MIT|{c}".encode()).hexdigest())
    return pd.DataFrame([{"Cell_ID": c, "Fold_ID": i % N_FOLDS + 1} for i, c in enumerate(order)]).sort_values(
        ["Fold_ID", "Cell_ID"]).reset_index(drop=True)


def t0_values(name: str, t_avail: int, labels_first_hit: pd.Series, last_cycles: pd.Series) -> dict:
    """PROTOCOL_v2 section 4: t0_1 = max(T_avail, E_min - 1); t0_2/t0_3 from floor(0.35/0.50 * M)."""
    e_min = float(labels_first_hit.min())
    t1 = int(max(t_avail, e_min - 1))
    m = float(np.median(last_cycles))
    t2 = max(t1 + 1, math.floor(T0_FRACTIONS[0] * m))
    t3 = max(t2 + 1, math.floor(T0_FRACTIONS[1] * m))
    return {"Dataset": name, "T_avail": int(t_avail), "E_min": e_min, "M_Median_Last_Cycle": m,
            "t0_1": t1, "t0_2": int(t2), "t0_3": int(t3)}
