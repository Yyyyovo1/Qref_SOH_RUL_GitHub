"""Read-only follow-up 2: observable threshold windows, equivalent-threshold prediction, v2 reversals vs window,
and the annotated label-count figure. Descriptive only; writes only to project/outputs/followup_analyses_v1/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs" / "followup_analyses_v1"
sys.path.insert(0, str(ROOT))
from rul_pilot_v2.data import load_all  # noqa: E402

QREFS = ["nominal", "first_comparable", "early_median5_comparable"]
QL = {"nominal": "nominal", "first_comparable": "first", "early_median5_comparable": "first5"}
NOMINAL = {"XJTU": 2.0, "HUST": 1.1, "MIT": 1.1}
GRID = np.round(np.arange(0.75, 0.9501, 0.01), 2)
SHARE = 0.90
COLORS = {"nominal": "#2a78d6", "first": "#eb6834", "first5": "#1baf7a"}
MARKERS = {"nominal": "o", "first": "s", "first5": "^"}
TEXT, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"


def _write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.6g")


def positions(data) -> pd.DataFrame:
    """Per cell and Qref: stop position = last retained capacity / Qref; initial = first retained capacity / Qref."""
    rows = []
    for name, d in data.items():
        cap = d.capacity.sort_values(["Cell_ID", "Endpoint_Index", "Raw_Cycle_Index"], kind="stable")
        first, last = cap.groupby("Cell_ID").Capacity_Ah.first(), cap.groupby("Cell_ID").Capacity_Ah.last()
        for c in d.cells:
            group = name if name != "MIT" else ("MIT batch 2" if c.startswith("b2") else "MIT batches 1+3")
            batch = name if name != "MIT" else f"MIT {c[:2]}"
            for q in QREFS:
                qref = float(d.qref.at[c, q])
                rows.append({"Dataset": name, "Group": group, "Batch": batch, "Cell_ID": c, "Qref": QL[q], "Qref_Ah": qref,
                             "Stop_Position": float(last[c]) / qref, "Initial_Position": float(first[c]) / qref})
    return pd.DataFrame(rows)


def describe(pos: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in pos.groupby(by, sort=False):
        for col in ("Stop_Position", "Initial_Position"):
            q = np.quantile(g[col], [0, 0.25, 0.5, 0.75, 1])
            rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "Quantity": col, "Cells": len(g),
                         "Min": q[0], "Q1": q[1], "Median": q[2], "Q3": q[3], "Max": q[4]})
    return pd.DataFrame(rows)


def windows(counts: pd.DataFrame) -> pd.DataFrame:
    """theta points with labeled share >= 90% on the 21-point grid, per dataset x rule x Qref."""
    rows = []
    for (d, q), g in counts.groupby(["Dataset", "Qref"], sort=False):
        g = g.set_index("Theta").reindex(GRID)
        for rule, col in (("strict", "Strict_Rule_Labeled"), ("tolerance", "Tolerance_Rule_Labeled")):
            share = g[col] / g.Cells
            pts = [float(t) for t in GRID if share[t] >= SHARE]
            contiguous = bool(pts) and len(pts) == int(round((pts[-1] - pts[0]) / 0.01)) + 1
            best = float(share.max())
            rows.append({"Dataset": d, "Qref": q, "Rule": rule, "Window_Exists": bool(pts),
                         "Window_Low": pts[0] if pts else np.nan, "Window_High": pts[-1] if pts else np.nan,
                         "Contiguous": contiguous if pts else np.nan, "Points": len(pts),
                         "Low_At_Grid_Limit": bool(pts) and pts[0] == 0.75, "High_At_Grid_Limit": bool(pts) and pts[-1] == 0.95,
                         "Max_Share": best, "Theta_At_Max_Share": "|".join(f"{t:.2f}" for t in GRID if share[t] == best)})
    return pd.DataFrame(rows)


def equivalent_prediction(win: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """Predict measured-Qref window from the nominal window via theta' = theta * Qref / Qnom, using the
    median cell ratio r = Qref / Qnom: predicted window = nominal window / r."""
    rows = []
    for (d, rule), g in win.groupby(["Dataset", "Rule"], sort=False):
        nom = g[g.Qref == "nominal"].iloc[0]
        for q in ("first", "first5"):
            act = g[g.Qref == q].iloc[0]
            ratios = pos[(pos.Dataset == d) & (pos.Qref == q)].Qref_Ah / NOMINAL[d]
            r = float(np.median(ratios))
            row = {"Dataset": d, "Rule": rule, "Qref": q, "Median_Qref_over_Nominal": r,
                   "Ratio_Min": float(ratios.min()), "Ratio_Q1": float(ratios.quantile(0.25)),
                   "Ratio_Q3": float(ratios.quantile(0.75)), "Ratio_Max": float(ratios.max()),
                   "Nominal_Window": f"[{nom.Window_Low:.2f}, {nom.Window_High:.2f}]" if nom.Window_Exists else "none",
                   "Actual_Window": f"[{act.Window_Low:.2f}, {act.Window_High:.2f}]" if act.Window_Exists else "none"}
            if nom.Window_Exists:
                lo, hi = nom.Window_Low / r, nom.Window_High / r
                row.update({"Predicted_Low": lo, "Predicted_High": hi,
                            "Predicted_Low_Within_Grid": 0.75 <= lo <= 0.95, "Predicted_High_Within_Grid": 0.75 <= hi <= 0.95})
                if act.Window_Exists:
                    row.update({
                        "Deviation_Low_Actual_minus_Predicted": act.Window_Low - lo if not (nom.Low_At_Grid_Limit or act.Low_At_Grid_Limit) else np.nan,
                        "Deviation_High_Actual_minus_Predicted": act.Window_High - hi if not (nom.High_At_Grid_Limit or act.High_At_Grid_Limit) else np.nan,
                        "Note": "edges at the grid limit (0.75 / 0.95) are censored by the grid and not compared"})
            rows.append(row)
    return pd.DataFrame(rows)


def lower_edge_check(win: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """Descriptive cross-check: the lower window edge vs. the 90th percentile of the per-cell stop position
    (strict: first grid point above it; tolerance: first grid point at or above it / 1.01)."""
    rows = []
    for r in win.itertuples(index=False):
        p90 = float(np.quantile(pos[(pos.Dataset == r.Dataset) & (pos.Qref == r.Qref)].Stop_Position, 0.9))
        bound = p90 if r.Rule == "strict" else p90 / 1.01
        implied = next((float(t) for t in GRID if (t > bound if r.Rule == "strict" else t >= bound)), np.nan)
        rows.append({"Dataset": r.Dataset, "Qref": r.Qref, "Rule": r.Rule, "Stop_Position_P90": p90, "Bound": bound,
                     "Implied_Low": implied, "Actual_Low": r.Window_Low, "Match": implied == r.Window_Low})
    return pd.DataFrame(rows)


def classify_reversals(win: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Each v2 supported reversal vs. the strict-rule window of both Qrefs involved (one grid step = edge)."""
    rev = pd.read_csv(ROOT / "outputs/rul_ranking_pilot_v2/reversals.csv", encoding="utf-8-sig")
    rev = rev[rev.Supported_Reversal].copy()
    rev["Theta"] = rev.Theta.round(2)
    w = win[win.Rule == "strict"].set_index(["Dataset", "Qref"])

    def status(d, q, t):
        r = w.loc[(d, QL[q])]
        if not r.Window_Exists:
            return "outside (no window)"
        lo, hi = r.Window_Low, r.Window_High
        real_lo, real_hi = not r.Low_At_Grid_Limit, not r.High_At_Grid_Limit
        if t < lo - 0.005 or t > hi + 0.005:
            return "edge" if (real_lo and abs(t - lo) <= 0.0105) or (real_hi and abs(t - hi) <= 0.0105) else "outside"
        if (real_lo and t - lo <= 0.0105) or (real_hi and hi - t <= 0.0105):
            return "edge"
        return "inside (grid limit)" if (t == 0.95 and not real_hi) or (t == 0.75 and not real_lo) else "inside"

    rank = {"inside": 0, "inside (grid limit)": 1, "edge": 2, "outside": 3, "outside (no window)": 3}
    s1 = [status(d, q, t) for d, q, t in zip(rev.Dataset, rev.Setting_1, rev.Theta)]
    s2 = [status(d, q, t) for d, q, t in zip(rev.Dataset, rev.Setting_2, rev.Theta)]
    rev["Window_Status_Setting_1"], rev["Window_Status_Setting_2"] = s1, s2
    rev["Window_Class"] = [a if rank[a] >= rank[b] else b for a, b in zip(s1, s2)]
    cs = counts.set_index(["Dataset", "Qref", "Theta"])
    for k in (1, 2):
        key = list(zip(rev.Dataset, rev[f"Setting_{k}"].map(QL), rev.Theta))
        rev[f"Strict_Share_Setting_{k}"] = [cs.loc[i].Strict_Rule_Labeled / cs.loc[i].Cells for i in key]
        rev[f"Tolerance_Share_Setting_{k}"] = [cs.loc[i].Tolerance_Rule_Labeled / cs.loc[i].Cells for i in key]
    return rev[["Dataset", "Theta", "Eval_Set", "Pair", "Setting_1", "Setting_2", "Better_1", "Better_2",
                "Window_Status_Setting_1", "Window_Status_Setting_2", "Window_Class",
                "Strict_Share_Setting_1", "Strict_Share_Setting_2", "Tolerance_Share_Setting_1", "Tolerance_Share_Setting_2"]]


def figure(counts: pd.DataFrame, pos: pd.DataFrame, baseline: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), sharex=True)
    for ax, ds in zip(axes, ["XJTU", "HUST", "MIT"]):
        c = counts[counts.Dataset == ds]
        n = int(c.Cells.iat[0])
        for q in ("nominal", "first", "first5"):
            s = c[c.Qref == q]
            ax.plot(s.Theta, s.Strict_Rule_Labeled, color=COLORS[q], lw=2, marker=MARKERS[q], ms=4.5, label=f"strict · {q}")
            ax.plot(s.Theta, s.Tolerance_Rule_Labeled, color=COLORS[q], lw=2, ls=(0, (4, 2)), marker=MARKERS[q], ms=4.5,
                    mfc="white", label=f"tolerance · {q}")
        nominal = pos[(pos.Dataset == ds) & (pos.Qref == "nominal")]
        groups = ["MIT batch 2", "MIT batches 1+3"] if ds == "MIT" else [ds]
        for k, gname in enumerate(groups):
            x = float(nominal[nominal.Group == gname].Stop_Position.median())
            ax.axvline(x, color=MUTED, lw=1.1, ls=(0, (6, 3)), zorder=0)
            label = "stop" if ds != "MIT" else ("b2 stop" if k == 0 else "b1+b3 stop")
            # Label sits just above the plot area, centred on its line, so it never covers a curve.
            ax.annotate(f"{label} {x:.3f}", xy=(x, 1.0), xycoords=("data", "axes fraction"), xytext=(-2, 3),
                        textcoords="offset points", ha="left", va="bottom", fontsize=7.5, color=MUTED)
        if ds == "XJTU":
            row = c[(c.Qref == "nominal") & (c.Theta == 0.94)].iloc[0]
            ax.annotate(f"nominal, θ ≥ 0.90: first observation\nalready below threshold\n"
                        f"({baseline[0.90]} cells at 0.90 → {baseline[0.95]} at 0.95)",
                        xy=(0.94, row.Strict_Rule_Labeled), xytext=(0.948, n * 0.64), ha="right", va="center",
                        fontsize=7.5, color=TEXT,
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9, shrinkB=4))
        ax.set_title(f"{ds}  (n = {n} cells)", loc="left", fontsize=10.5, color=TEXT, fontweight="bold", pad=16)
        ax.set_ylim(-n * 0.03, n * 1.06)
        ax.set_xticks(np.round(np.arange(0.75, 0.951, 0.05), 2))
        ax.set_xlabel("θ (EOL threshold fraction)", fontsize=9, color=MUTED)
        ax.grid(color=GRIDC, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("cells with a point EOL label", fontsize=9, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [0, 2, 4, 1, 3, 5]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=6, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, 0.005, "Descriptive only. Solid: strict rule; dashed, open markers: tolerance rule (Lecu et al. 2026, "
             "BatteryKG qc.py @ 63b974c). Grey dashed lines: stop position = median last retained capacity / nominal capacity (MIT: batch 2 and batches 1+3 separately).",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"3_labeled_cells_by_theta_annotated.{ext}", dpi=300, facecolor="white",
                    metadata={"Software": None} if ext == "png" else {"CreationDate": None})
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all(ROOT)
    pos = positions(data)
    counts = pd.read_csv(OUT / "2_labeled_cells_by_theta.csv", encoding="utf-8-sig")
    counts["Theta"] = counts.Theta.round(2)
    _write(describe(pos, ["Dataset", "Qref"]), "4_position_distribution_by_dataset.csv")
    _write(describe(pos[pos.Dataset == "MIT"], ["Batch", "Qref"]), "4_position_distribution_mit_by_batch.csv")
    win = windows(counts)
    _write(win, "5_observable_windows.csv")
    pred = equivalent_prediction(win, pos)
    _write(pred, "6_equivalent_threshold_prediction.csv")
    _write(lower_edge_check(win, pos), "6b_lower_edge_vs_stop_p90.csv")
    rev = classify_reversals(win, counts)
    _write(rev, "7_v2_reversals_vs_window.csv")
    lab = pd.read_csv(ROOT / "outputs/protocol_validation_v1/trapD_labels.csv", encoding="utf-8-sig", float_precision="round_trip")
    lab["Theta"] = lab.Theta.round(2)
    x = lab[(lab.Dataset == "XJTU") & (lab.Qref_Definition == "nominal")]
    baseline = x[x.Strict_Status.eq("BASELINE")].groupby("Theta").size().reindex(GRID, fill_value=0).astype(int).to_dict()
    figure(counts, pos, baseline)
    summary = {"xjtu_nominal_strict_baseline_by_theta": {f"{k:.2f}": v for k, v in baseline.items()},
               "reversal_window_class_counts": rev.Window_Class.value_counts().to_dict(),
               "reversal_window_class_by_dataset": rev.groupby(["Dataset", "Window_Class"]).size().rename("n").reset_index().to_dict("records")}
    (OUT / "followup2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(describe(pos, ["Dataset", "Qref"]).round(4).to_string(index=False))
    print(describe(pos[pos.Dataset == "MIT"], ["Batch", "Qref"]).round(4).to_string(index=False))
    print(win.round(3).to_string(index=False))
    print(pred.round(4).to_string(index=False))
    print(rev.to_string(index=False))


if __name__ == "__main__":
    main()
