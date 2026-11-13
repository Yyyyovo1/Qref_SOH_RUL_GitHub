"""TJU NCA / NCM label structure (read-only; capacity data only; no models, no method comparison).

Inputs: frozen outputs of commit 22801aa (project/outputs/tju_final_v2). Writes only to this directory.
Grid theta = 0.60-0.95, step 0.01. Strict rule = frozen dense detector used for HUST / MIT (detect_hust_eol);
tolerance rule = Lecu et al. 2026 (protocol_validation.endpoint.tolerance_rule).
Estimability: v2 rules (PROTOCOL_v2 sections 4-6) evaluated from labels only; TRAJ fallback cannot be known without
predictions and is treated as estimable (upper bound); the bootstrap-missing rule is not evaluated.
"""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "project"
sys.path.insert(0, str(PROJECT))
from eol_threshold_robustness_repro.config import EXACT as F_EXACT, INTERVAL as F_INTERVAL, LEFT as F_LEFT, RIGHT as F_RIGHT  # noqa: E402
from eol_threshold_robustness_repro.pipeline import detect_hust_eol  # noqa: E402
from protocol_validation.endpoint import tolerance_rule  # noqa: E402

OUT = PROJECT / "outputs" / "tju_final_v2"
GRID = np.round(np.arange(0.60, 0.9501, 0.01), 2)
QREFS = {"nominal": "Qref_Nominal_Ah", "first": "Qref_First_Comparable_Ah", "first5": "Qref_Early_Median5_Comparable_Ah"}
QPAIRS = [("nominal", "first"), ("nominal", "first5"), ("first", "first5")]
STATUS = {F_EXACT: "EXACT", F_INTERVAL: "INTERVAL", F_RIGHT: "RIGHT", F_LEFT: "BASELINE"}
METHODS = ("RF", "TRAJ", "WAFT", "BASE")
PAIRS = [f"{a}-{b}" for a, b in combinations(METHODS, 2)]
MIN_CELLS, C_MIN_PAIRS, N_FOLDS = 5, 10, 5
T0_FRACTIONS = (0.35, 0.50)
COLORS = {"nominal": "#2a78d6", "first": "#eb6834", "first5": "#1baf7a"}
MARKERS = {"nominal": "o", "first": "s", "first5": "^"}
TEXT, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"


def _write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(HERE / name, index=False, encoding="utf-8-sig", float_format="%.6g")


def load():
    cap = pd.read_csv(OUT / "tju_primary_capacity.csv", encoding="utf-8-sig")
    qref = pd.read_csv(OUT / "tju_qref_summary.csv", encoding="utf-8-sig")
    folds = pd.read_csv(OUT / "tju_cell_fold_assignments.csv", encoding="utf-8-sig")
    return cap, qref, folds


def positions(cap: pd.DataFrame, qref: pd.DataFrame) -> pd.DataFrame:
    first = cap.sort_values("Cycle_Index").groupby("Cell_ID").Capacity_Ah.first()
    last = cap.sort_values("Cycle_Index").groupby("Cell_ID").Capacity_Ah.last()
    rows = []
    for r in qref.itertuples(index=False):
        for q, col in QREFS.items():
            v = float(getattr(r, col))
            rows.append({"Dataset": r.Dataset, "Cell_ID": r.Cell_ID, "Qref": q, "Qref_Ah": v,
                         "Stop_Position": float(last[r.Cell_ID]) / v, "Initial_Position": float(first[r.Cell_ID]) / v})
    return pd.DataFrame(rows)


def describe(pos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (d, q), g in pos.groupby(["Dataset", "Qref"], sort=False):
        for col in ("Stop_Position", "Initial_Position"):
            x = g[col].to_numpy()
            p = np.quantile(x, [0, .1, .25, .5, .75, .9, 1])
            rows.append({"Dataset": d, "Qref": q, "Quantity": col, "Cells": len(x), "Min": p[0], "P10": p[1], "Q1": p[2],
                         "Median": p[3], "Q3": p[4], "P90": p[5], "Max": p[6]})
    return pd.DataFrame(rows)


def labels(cap: pd.DataFrame, qref: pd.DataFrame) -> pd.DataFrame:
    obs = {c: g.rename(columns={"Cycle_Index": "Raw_Cycle_Index", "Capacity_Ah": "Primary_Capacity_Ah"})[["Raw_Cycle_Index", "Primary_Capacity_Ah"]]
           for c, g in cap.sort_values("Cycle_Index").groupby("Cell_ID")}
    rows = []
    for r in qref.itertuples(index=False):
        o = obs[r.Cell_ID]
        idx, q_ah = o.Raw_Cycle_Index.to_numpy(float), o.Primary_Capacity_Ah.to_numpy(float)
        for q, col in QREFS.items():
            ref = float(getattr(r, col))
            for th in GRID:
                thr = float(th) * ref
                d = detect_hust_eol(o, thr)
                kind, tol_eol = tolerance_rule(idx, q_ah, thr)
                first_hit = d.exact_cycle if d.exact_cycle is not None else (d.upper_bound if d.status in (F_INTERVAL, F_LEFT) else None)
                rows.append({"Dataset": r.Dataset, "Cell_ID": r.Cell_ID, "Qref": q, "Theta": float(th), "Threshold_Ah": thr,
                             "Status": STATUS[d.status],
                             "EOL_Exact": float(d.exact_cycle) if d.exact_cycle is not None else np.nan,
                             "EOL_Lower": float(d.lower_bound) if d.status == F_INTERVAL else np.nan,
                             "EOL_Upper": float(d.upper_bound) if d.status == F_INTERVAL else np.nan,
                             "First_Hit": np.nan if first_hit is None else float(first_hit),
                             "Last_Cycle": float(idx[-1]), "Tolerance_Kind": kind, "Tolerance_EOL": tol_eol})
    lab = pd.DataFrame(rows)
    lab["Strict_Labeled"] = lab.Status.eq("EXACT")
    lab["Tolerance_Labeled"] = lab.Tolerance_Kind.ne("NO_LABEL")
    return lab


def shares(lab: pd.DataFrame) -> pd.DataFrame:
    g = lab.groupby(["Dataset", "Qref", "Theta"])
    out = g.agg(Cells=("Cell_ID", "size"), Strict_Labeled=("Strict_Labeled", "sum"), Tolerance_Labeled=("Tolerance_Labeled", "sum"),
                EXACT=("Status", lambda s: int((s == "EXACT").sum())), INTERVAL=("Status", lambda s: int((s == "INTERVAL").sum())),
                RIGHT=("Status", lambda s: int((s == "RIGHT").sum())), BASELINE=("Status", lambda s: int((s == "BASELINE").sum())),
                FIRST_BELOW=("Tolerance_Kind", lambda s: int((s == "FIRST_BELOW").sum())),
                TOLERANCE=("Tolerance_Kind", lambda s: int((s == "TOLERANCE").sum()))).reset_index()
    out["Strict_Share"] = out.Strict_Labeled / out.Cells
    out["Tolerance_Share"] = out.Tolerance_Labeled / out.Cells
    return out


def zone(a: float, b: float) -> str:
    m = min(a, b)
    return "low" if m < 0.30 else ("partial" if m < 0.90 else "sufficient")


def zones(sh: pd.DataFrame) -> pd.DataFrame:
    s = sh.set_index(["Dataset", "Qref", "Theta"])
    rows = []
    for d in sh.Dataset.unique():
        for q1, q2 in QPAIRS:
            for th in GRID:
                th = float(th)
                a, b = s.loc[(d, q1, th)], s.loc[(d, q2, th)]
                rows.append({"Dataset": d, "Qref_Pair": f"{q1}-{q2}", "Theta": th,
                             "Strict_Share_1": a.Strict_Share, "Strict_Share_2": b.Strict_Share,
                             "Strict_Zone": zone(a.Strict_Share, b.Strict_Share),
                             "Tolerance_Share_1": a.Tolerance_Share, "Tolerance_Share_2": b.Tolerance_Share,
                             "Tolerance_Zone": zone(a.Tolerance_Share, b.Tolerance_Share)})
    return pd.DataFrame(rows)


def t0_values(lab: pd.DataFrame, qref: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in qref.groupby("Dataset"):
        t_avail = max(int(s.split("|")[-1]) for s in g.Qref_Early_Median5_Comparable_Source_Cycles)
        e_min = float(lab[(lab.Dataset == d) & lab.Status.ne("BASELINE")].First_Hit.min())
        t1 = int(max(t_avail, e_min - 1))
        m = float(np.median(g.Last_Retained_Cycle))
        t2 = max(t1 + 1, math.floor(T0_FRACTIONS[0] * m))
        t3 = max(t2 + 1, math.floor(T0_FRACTIONS[1] * m))
        rows.append({"Dataset": d, "T_avail": t_avail, "E_min": e_min, "M_Median_Last_Cycle": m, "t0_1": t1, "t0_2": t2, "t0_3": t3})
    return pd.DataFrame(rows)


def samples(lab: pd.DataFrame, cap: pd.DataFrame, t0: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """v2 build_samples, label part only: one row per dataset x Qref x theta x cell x t0."""
    t0m = t0.set_index("Dataset")
    fold = dict(zip(folds.Cell_ID, folds.Fold_ID))
    frames = []
    for k in range(3):
        s = lab.copy()
        s["t_Index"] = k
        s["t_Raw_Cycle"] = s.Dataset.map(t0m[f"t0_{k + 1}"])
        nt = {}
        for d in s.Dataset.unique():
            t = int(t0m.at[d, f"t0_{k + 1}"])
            last = cap[(cap.Dataset == d) & (cap.Cycle_Index <= t)].groupby("Cell_ID").Cycle_Index.max()
            nt.update(last.to_dict())
        s["n_t"] = s.Cell_ID.map(nt)
        s["EOL_Before_t"] = s.First_Hit.le(s.t_Raw_Cycle).fillna(False)
        s["Eligible"] = ~s.EOL_Before_t & s.Status.ne("BASELINE") & s.n_t.notna()
        n = s.n_t
        s["RUL"] = np.where(s.Status.eq("EXACT"), s.EOL_Exact - n, np.nan)
        s["RUL_Lower"] = np.select([s.Status.eq("EXACT"), s.Status.eq("INTERVAL"), s.Status.eq("RIGHT")],
                                   [s.RUL, s.EOL_Lower - n, s.Last_Cycle - n], np.nan)
        s["RUL_Upper"] = np.select([s.Status.eq("EXACT"), s.Status.eq("INTERVAL"), s.Status.eq("RIGHT")],
                                   [s.RUL, s.EOL_Upper - n, np.inf], np.nan)
        frames.append(s)
    out = pd.concat(frames, ignore_index=True)
    out["Fold_ID"] = out.Cell_ID.map(fold)
    return out


def estimability(smp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per dataset x Qref x theta: method estimability (fold rules) and E1/E2/E3 sizes; then Q-type comparisons."""
    el = smp[smp.Eligible]
    setting_rows = []
    e1_members = {}
    for (d, th), g in el.groupby(["Dataset", "Theta"]):
        t2 = g[g.t_Index.eq(2)]
        ok = t2.assign(ok=t2.Status.eq("EXACT")).groupby("Cell_ID").ok.agg(["all", "size"])
        e1_members[(d, th)] = set(ok[ok["all"] & (ok["size"] == len(QREFS))].index)
    all_cells = smp.groupby("Dataset").Cell_ID.unique().to_dict()
    for (d, q, th), g in smp.groupby(["Dataset", "Qref", "Theta"]):
        e = g[g.Eligible]
        est = {m: True for m in METHODS}
        reasons = {m: "" for m in METHODS}
        for k in range(1, N_FOLDS + 1):
            test, train = e[e.Fold_ID.eq(k)], e[e.Fold_ID.ne(k)]
            if test.empty:
                continue
            n_exact = train[train.Status.eq("EXACT")].Cell_ID.nunique()
            n_event = train[train.Status.isin(["EXACT", "INTERVAL"])].Cell_ID.nunique()
            for m in ("RF", "BASE"):
                if n_exact < MIN_CELLS and est[m]:
                    est[m], reasons[m] = False, f"FOLD_{k}_TRAIN_EXACT_CELLS_{n_exact}"
            if n_event < MIN_CELLS and est["WAFT"]:
                est["WAFT"], reasons["WAFT"] = False, f"FOLD_{k}_TRAIN_EVENT_CELLS_{n_event}"
        if e.empty:
            est = {m: False for m in METHODS}
            reasons = {m: "NO_ELIGIBLE_SAMPLES" for m in METHODS}
        e2 = e[e.Status.eq("EXACT")].Cell_ID.nunique()
        e1 = len(e1_members.get((d, th), set()))
        pairs_t = []
        for k in range(3):
            ek = e[e.t_Index.eq(k)]
            lo, hi = ek.RUL_Lower.to_numpy(float), ek.RUL_Upper.to_numpy(float)
            pairs_t.append(int((hi[:, None] < lo[None, :]).sum()) if len(ek) else 0)
        setting_rows.append({"Dataset": d, "Qref": q, "Theta": th, "Eligible_Cells": e.Cell_ID.nunique(),
                             "E1_Cells": e1, "E2_Cells": e2, "E3_Pairs_per_t": "|".join(map(str, pairs_t)),
                             "E1_OK": e1 >= MIN_CELLS, "E2_OK": e2 >= MIN_CELLS, "E3_OK": max(pairs_t) >= C_MIN_PAIRS,
                             **{f"{m}_Estimable": est[m] for m in METHODS}, **{f"{m}_Reason": reasons[m] for m in METHODS}})
    st = pd.DataFrame(setting_rows)
    s = st.set_index(["Dataset", "Qref", "Theta"])
    comp = []
    for d in st.Dataset.unique():
        for q1, q2 in QPAIRS:
            for th in GRID:
                th = float(th)
                a, b = s.loc[(d, q1, th)], s.loc[(d, q2, th)]
                for ev in ("E1", "E2", "E3"):
                    for pair in PAIRS:
                        m1, m2 = pair.split("-")
                        ok = all(bool(x[f"{m}_Estimable"]) for x in (a, b) for m in (m1, m2)) and bool(a[f"{ev}_OK"]) and bool(b[f"{ev}_OK"])
                        comp.append({"Dataset": d, "Qref_Pair": f"{q1}-{q2}", "Theta": th, "Eval_Set": ev, "Pair": pair,
                                     "Includes_TRAJ": "TRAJ" in pair, "Estimable_Comparison": ok})
    return st, pd.DataFrame(comp)


def figure(sh: pd.DataFrame, pos: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, ds in zip(axes, ["NCA", "NCM"]):
        c = sh[sh.Dataset == ds]
        n = int(c.Cells.iat[0])
        ax.axhspan(0.90, 1.03, color="#eef4fb", zorder=0)
        ax.axhspan(-0.03, 0.30, color="#f6f5f2", zorder=0)
        for y in (0.30, 0.90):
            ax.axhline(y, color=MUTED, lw=0.8, ls=":", zorder=1)
        for q in ("nominal", "first", "first5"):
            s = c[c.Qref == q]
            ax.plot(s.Theta, s.Strict_Share, color=COLORS[q], lw=2, marker=MARKERS[q], ms=4, label=f"strict · {q}")
            ax.plot(s.Theta, s.Tolerance_Share, color=COLORS[q], lw=2, ls=(0, (4, 2)), marker=MARKERS[q], ms=4, mfc="white",
                    label=f"tolerance · {q}")
        x = float(pos[(pos.Dataset == ds) & (pos.Qref == "nominal")].Stop_Position.median())
        ax.axvline(x, color=MUTED, lw=1.1, ls=(0, (6, 3)), zorder=1)
        ax.annotate(f"stop (nominal, median) {x:.3f}", xy=(x, 1.0), xycoords=("data", "axes fraction"), xytext=(-2, 3),
                    textcoords="offset points", ha="left", va="bottom", fontsize=7.5, color=MUTED)
        ax.text(0.602, 0.915, "≥ 90%", fontsize=7.5, color=MUTED)
        ax.text(0.602, 0.315, "30%", fontsize=7.5, color=MUTED)
        ax.set_title(f"TJU {ds}  (n = {n} cells)", loc="left", fontsize=10.5, color=TEXT, fontweight="bold", pad=16)
        ax.set_xlim(0.595, 0.955)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(np.round(np.arange(0.60, 0.951, 0.05), 2))
        ax.set_xlabel("θ (EOL threshold fraction)", fontsize=9, color=MUTED)
        ax.grid(color=GRIDC, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("share of cells with a point EOL label", fontsize=9, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    order = [0, 2, 4, 1, 3, 5]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=6, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, 0.005, "Descriptive only, capacity data only. Solid: strict rule (frozen dense detector); dashed, open markers: "
             "tolerance rule (Lecu et al. 2026). Shaded: ≥ 90% (blue) and < 30% (grey) labeled share.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"tju_labeled_share_by_theta.{ext}", dpi=300, facecolor="white",
                    metadata={"Software": None} if ext == "png" else {"CreationDate": None})
    plt.close(fig)


def main() -> None:
    cap, qref, folds = load()
    pos = positions(cap, qref)
    _write(pos, "1_positions_per_cell.csv")
    _write(describe(pos), "1_position_distribution.csv")
    lab = labels(cap, qref)
    _write(lab, "2_labels_per_cell_setting.csv")
    sh = shares(lab)
    _write(sh, "2_labeled_share_by_theta.csv")
    zn = zones(sh)
    _write(zn, "3_coverage_zones.csv")
    t0 = t0_values(lab, qref)
    _write(t0, "4_t0_v2_rule.csv")
    smp = samples(lab, cap, t0, folds)
    st, comp = estimability(smp)
    _write(st, "4_setting_estimability.csv")
    _write(comp, "4_comparison_estimability.csv")
    cz = comp.merge(zn[["Dataset", "Qref_Pair", "Theta", "Strict_Zone", "Tolerance_Zone"]], on=["Dataset", "Qref_Pair", "Theta"])
    summ = []
    for traj in ("all_pairs", "excluding_TRAJ"):
        x = cz if traj == "all_pairs" else cz[~cz.Includes_TRAJ]
        t = x.groupby(["Dataset", "Strict_Zone", "Eval_Set"]).agg(Comparisons_Total=("Estimable_Comparison", "size"),
                                                                    Estimable=("Estimable_Comparison", "sum")).reset_index()
        t["Pairs_Scope"] = traj
        summ.append(t)
    summ = pd.concat(summ)
    _write(summ, "5_estimable_comparisons_by_zone.csv")
    t = cz.groupby(["Dataset", "Qref_Pair", "Strict_Zone", "Eval_Set"]).agg(
        Theta_Points=("Theta", "nunique"), Comparisons=("Estimable_Comparison", "size"), Estimable=("Estimable_Comparison", "sum")).reset_index()
    t2 = cz[~cz.Includes_TRAJ].groupby(["Dataset", "Qref_Pair", "Strict_Zone", "Eval_Set"]).Estimable_Comparison.sum()         .rename("Estimable_excl_TRAJ").reset_index()
    _write(t.merge(t2), "5b_estimable_by_pair_zone.csv")
    zc = zn.groupby(["Dataset", "Qref_Pair", "Strict_Zone"]).Theta.agg(Theta_Points="size",
                                                                         Thetas=lambda s: "|".join(f"{t:.2f}" for t in s)).reset_index()
    _write(zc, "3_zone_theta_points.csv")
    figure(sh, pos)
    info = {"grid": [float(GRID[0]), float(GRID[-1]), len(GRID)], "t0": t0.to_dict("records"),
            "cells": lab.groupby("Dataset").Cell_ID.nunique().to_dict(),
            "notes": ["TRAJ fallback fraction not computable from labels: TRAJ treated as estimable",
                      "bootstrap missing-fraction rule not evaluated (needs resampling of errors)"]}
    (HERE / "run_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(json.dumps(info, indent=2))
    print(describe(pos).round(4).to_string(index=False))
    print(zc.to_string(index=False))
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
