"""Read-only follow-up checks on protocol_validation_v1 (trap D). Writes only to project/outputs/followup_analyses_v1/.

Inputs (read-only): project/outputs/protocol_validation_v1/trapD_labels.csv (formal run, commit 6886053),
project/outputs/hust_77_final_v1/hust_77_primary_capacity.csv, project/outputs/mit_124_final_v1/mit_primary_capacity.csv.
Descriptive only; no new decision rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs" / "followup_analyses_v1"
QREFS = ["nominal", "first_comparable", "early_median5_comparable"]
QL = {"nominal": "nominal", "first_comparable": "first", "early_median5_comparable": "first5"}
COLORS = {"nominal": "#2a78d6", "first_comparable": "#eb6834", "early_median5_comparable": "#1baf7a"}
MARKERS = {"nominal": "o", "first_comparable": "s", "early_median5_comparable": "^"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.10g")


def last_capacity(path: Path, cell_col: str, cycle_col: str, cap_col: str, eligible_col: str | None) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if eligible_col:
        df = df[df[eligible_col].eq("YES")]
    return df.sort_values([cell_col, cycle_col]).groupby(cell_col)[cap_col].last()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(ROOT / "outputs/protocol_validation_v1/trapD_labels.csv", encoding="utf-8-sig",
                      float_precision="round_trip")
    lab["Theta"] = lab.Theta.round(2)
    t80 = lab[lab.Theta.eq(0.80)]
    out = {}

    # 1a. MIT strict rule: are the three sets of 43 exact cells the same cells?
    mit = t80[t80.Dataset.eq("MIT")]
    exact = {q: set(mit[(mit.Qref_Definition == q) & mit.Strict_Status.eq("EXACT")].Cell_ID) for q in QREFS}
    cells = sorted(set().union(*exact.values()))
    _write(pd.DataFrame({"Cell_ID": cells, **{f"Strict_Exact_{QL[q]}": [c in exact[q] for c in cells] for q in QREFS}}),
           "1a_mit_strict_exact_cells_theta080.csv")
    out["1a_mit_strict_exact"] = {"sizes": {QL[q]: len(exact[q]) for q in QREFS},
                                  "identical_sets": exact[QREFS[0]] == exact[QREFS[1]] == exact[QREFS[2]],
                                  "only_in": {QL[q]: sorted(exact[q] - set.intersection(*exact.values())) for q in QREFS}}

    # 1b. MIT tolerance rule: composition of nominal (124) and first / first5 (46).
    wide = mit.pivot_table(index="Cell_ID", columns="Qref_Definition",
                           values=["Strict_Status", "Tolerance_Kind", "Threshold_Ah"], aggfunc="first")
    wide.columns = [f"{a}_{QL[b]}" for a, b in wide.columns]
    mcap = last_capacity(ROOT / "outputs/mit_124_final_v1/mit_primary_capacity.csv", "Cell_ID", "Cycle_Index", "QDischarge_Ah", None)
    wide["Last_Capacity_Ah"] = mcap.reindex(wide.index)
    for q in QREFS:
        wide[f"Last_Minus_Threshold_Ah_{QL[q]}"] = wide.Last_Capacity_Ah - wide[f"Threshold_Ah_{QL[q]}"]
    _write(wide.reset_index(), "1b_mit_tolerance_composition_theta080.csv")
    comp = {}
    for q in QREFS:
        k = mit[mit.Qref_Definition == q]
        labeled = k[k.Tolerance_Kind.ne("NO_LABEL")]
        comp[QL[q]] = {
            "tolerance_rule_labels": int(len(labeled)),
            "by_kind": labeled.Tolerance_Kind.value_counts().to_dict(),
            "of_which_strict_exact": int(labeled.Strict_Status.eq("EXACT").sum()),
            "of_which_strict_right_censored": int(labeled.Strict_Status.eq("RIGHT").sum()),
            "strict_exact_not_labeled_by_tolerance_rule": int((k.Strict_Status.eq("EXACT") & k.Tolerance_Kind.eq("NO_LABEL")).sum()),
            "kind_cross_strict": {f"{s}->{t}": int(n) for (s, t), n in labeled.groupby(["Strict_Status", "Tolerance_Kind"]).size().items()},
        }
    tol = {q: set(mit[(mit.Qref_Definition == q) & mit.Tolerance_Kind.ne("NO_LABEL")].Cell_ID) for q in QREFS}
    comp["first_and_first5_same_46_cells"] = tol["first_comparable"] == tol["early_median5_comparable"]
    comp["measured_46_subset_of_nominal_124"] = tol["first_comparable"] <= tol["nominal"] and tol["early_median5_comparable"] <= tol["nominal"]
    comp["measured_46_subset_of_strict43_plus_tolerance_clause"] = all(
        tol[q] <= exact[q] | set(mit[(mit.Qref_Definition == q) & mit.Tolerance_Kind.eq("TOLERANCE")].Cell_ID)
        for q in ("first_comparable", "early_median5_comparable"))
    out["1b_mit_tolerance"] = comp

    # 1c. HUST tolerance rule, nominal: last capacity minus threshold for all 77 labels.
    hust = t80[t80.Dataset.eq("HUST") & t80.Qref_Definition.eq("nominal")].copy()
    hcap = last_capacity(ROOT / "outputs/hust_77_final_v1/hust_77_primary_capacity.csv", "Cell_ID", "Raw_Cycle_Index",
                         "Primary_Capacity_Ah", "Primary_Capacity_Eligible")
    hust["Last_Capacity_Ah"] = hust.Cell_ID.map(hcap)
    hust["Last_Minus_Threshold_Ah"] = hust.Last_Capacity_Ah - hust.Threshold_Ah
    hust["Last_Minus_Threshold_Pct_of_Threshold"] = 100 * hust.Last_Minus_Threshold_Ah / hust.Threshold_Ah
    _write(hust[["Cell_ID", "Threshold_Ah", "Last_Capacity_Ah", "Last_Minus_Threshold_Ah", "Last_Minus_Threshold_Pct_of_Threshold",
                 "Strict_Status", "Tolerance_Kind", "Tolerance_EOL_Index"]].sort_values("Cell_ID"), "1c_hust_nominal_tolerance_theta080.csv")
    d = hust.Last_Minus_Threshold_Ah
    out["1c_hust_nominal"] = {"labels": int(hust.Tolerance_Kind.ne("NO_LABEL").sum()),
                              "all_from_tolerance_clause": bool(hust.Tolerance_Kind.eq("TOLERANCE").all()),
                              "strict_status": hust.Strict_Status.value_counts().to_dict(),
                              "last_minus_threshold_Ah": {"min": float(d.min()), "median": float(d.median()), "max": float(d.max())},
                              "all_above_threshold_and_within_1pct": bool(((d > 0) & (d <= 0.01 * hust.Threshold_Ah)).all())}

    # 2. Labeled cells by dataset x rule x Qref over all 21 theta (descriptive).
    counts = []
    for (ds, q, th), g in lab.groupby(["Dataset", "Qref_Definition", "Theta"]):
        counts.append({"Dataset": ds, "Qref": QL[q], "Theta": th, "Cells": len(g),
                       "Strict_Rule_Labeled": int(g.Strict_Status.eq("EXACT").sum()),
                       "Tolerance_Rule_Labeled": int(g.Tolerance_Kind.ne("NO_LABEL").sum()),
                       "Tolerance_Clause_Only": int(g.Tolerance_Kind.eq("TOLERANCE").sum())})
    counts = pd.DataFrame(counts).sort_values(["Dataset", "Qref", "Theta"])
    _write(counts, "2_labeled_cells_by_theta.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True)
    for ax, ds in zip(axes, ["XJTU", "HUST", "MIT"]):
        c = counts[counts.Dataset == ds]
        n = int(c.Cells.iat[0])
        for q in QREFS:
            s = c[c.Qref == QL[q]]
            ax.plot(s.Theta, s.Strict_Rule_Labeled, color=COLORS[q], lw=2, marker=MARKERS[q], ms=4.5,
                    label=f"strict · {QL[q]}")
            ax.plot(s.Theta, s.Tolerance_Rule_Labeled, color=COLORS[q], lw=2, ls=(0, (4, 2)), marker=MARKERS[q], ms=4.5,
                    mfc="white", label=f"tolerance · {QL[q]}")
        ax.axvline(0.80, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"{ds}  (n = {n} cells)", loc="left", fontsize=10.5, color=TEXT, fontweight="bold")
        ax.set_ylim(-n * 0.03, n * 1.06)
        ax.set_xticks(np.round(np.arange(0.75, 0.951, 0.05), 2))
        ax.set_xlabel("θ (EOL threshold fraction)", fontsize=9, color=MUTED)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("cells with a point EOL label", fontsize=9, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [0, 2, 4, 1, 3, 5]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=6, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, 0.005, "Descriptive only. Solid: strict rule (exact endpoints); dashed, open markers: tolerance rule "
             "(Lecu et al. 2026, BatteryKG qc.py @ 63b974c). Dotted line: θ = 0.80.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"2_labeled_cells_by_theta.{ext}", dpi=300, facecolor="white",
                    metadata={"Software": None} if ext == "png" else {"CreationDate": None})
    plt.close(fig)

    (OUT / "followup_checks.json").write_text(json.dumps(out, indent=2, default=list), encoding="utf-8")
    print(json.dumps(out, indent=2, default=list))


if __name__ == "__main__":
    main()
