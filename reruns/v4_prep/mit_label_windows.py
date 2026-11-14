"""v4 preparation (read-only, capacity labels only): MIT share of cells labeled with EOL after cycle 100, common windows,
and estimability under the frozen MIT 5-fold split, for 3 Qref x 2 endpoint rules x theta 0.75-0.95.

Inputs: project/outputs/protocol_validation_v1/trapD_labels.csv (strict and tolerance labels, commit 6886053) and
project/outputs/mit_124_final_v1/mit_cell_fold_assignments.csv. No model is fitted, no error is computed.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "project"
QL = {"nominal": "nominal", "first_comparable": "first", "early_median5_comparable": "first5"}
GRID = np.round(np.arange(0.75, 0.9501, 0.01), 2)
PREDICTION_CYCLE = 100
SHARE = 0.90
MIN_CELLS = 5


def main() -> None:
    lab = pd.read_csv(PROJECT / "outputs/protocol_validation_v1/trapD_labels.csv", encoding="utf-8-sig", float_precision="round_trip")
    lab = lab[lab.Dataset.eq("MIT")].copy()
    lab["Theta"] = lab.Theta.round(2)
    lab["Qref"] = lab.Qref_Definition.map(QL)
    lab["strict"] = lab.Strict_Status.eq("EXACT") & (lab.Strict_EOL_Index > PREDICTION_CYCLE)
    lab["tolerance"] = lab.Tolerance_Kind.ne("NO_LABEL") & (lab.Tolerance_EOL_Index > PREDICTION_CYCLE)
    folds = pd.read_csv(PROJECT / "outputs/mit_124_final_v1/mit_cell_fold_assignments.csv", encoding="utf-8-sig")
    fold = dict(zip(folds.Cell_ID, folds.Fold_ID))
    n_cells = lab.Cell_ID.nunique()

    share_rows = []
    for (rule), _ in (("strict", 0), ("tolerance", 0)):
        for (q, th), g in lab.groupby(["Qref", "Theta"]):
            share_rows.append({"Rule": rule, "Qref": q, "Theta": th, "Cells": n_cells, "Labeled_EOL_gt_100": int(g[rule].sum()),
                               "Share": float(g[rule].mean()),
                               "Labeled_Any_EOL": int((g.Strict_Status.eq("EXACT") if rule == "strict" else g.Tolerance_Kind.ne("NO_LABEL")).sum())})
    sh = pd.DataFrame(share_rows)
    sh.to_csv(HERE / "3_mit_labeled_share_eol_gt_100.csv", index=False, encoding="utf-8-sig", float_format="%.6g")

    s = sh.set_index(["Rule", "Qref", "Theta"]).Share
    labeled = {(r, q, th): set(g.Cell_ID[g[r]]) for r in ("strict", "tolerance") for (q, th), g in lab.groupby(["Qref", "Theta"])}
    win_rows, est_rows = [], []
    for rule in ("strict", "tolerance"):
        for q1, q2 in combinations(["nominal", "first", "first5"], 2):
            pts = [float(t) for t in GRID if s[(rule, q1, t)] >= SHARE and s[(rule, q2, t)] >= SHARE]
            win_rows.append({"Rule": rule, "Pair": f"{q1}-{q2}", "Common_Window_Points": len(pts),
                             "Common_Window": "|".join(f"{t:.2f}" for t in pts)})
            for t in GRID:
                t = float(t)
                both = labeled[(rule, q1, t)] & labeled[(rule, q2, t)]
                per_fold_train = [len([c for c in both if fold[c] != k]) for k in range(1, 6)]
                per_fold_test = [len([c for c in both if fold[c] == k]) for k in range(1, 6)]
                est_rows.append({"Rule": rule, "Pair": f"{q1}-{q2}", "Theta": t, "In_Common_Window": t in pts,
                                 "Share_1": s[(rule, q1, t)], "Share_2": s[(rule, q2, t)], "Cells_Labeled_Both": len(both),
                                 "Min_Train_Cells_Per_Fold": min(per_fold_train), "Min_Test_Cells_Per_Fold": min(per_fold_test),
                                 "Estimable": min(per_fold_train) >= MIN_CELLS and len(both) >= MIN_CELLS})
    win = pd.DataFrame(win_rows)
    est = pd.DataFrame(est_rows)
    win.to_csv(HERE / "3_mit_common_windows.csv", index=False, encoding="utf-8-sig")
    est.to_csv(HERE / "3_mit_estimability_by_pair_theta.csv", index=False, encoding="utf-8-sig", float_format="%.6g")

    # Common-practice condition (C): theta = 0.80, original Severson split (train / test1 / test2).
    feat = pd.read_csv(HERE / "1_severson_features_mit124.csv", encoding="utf-8-sig") if (HERE / "1_severson_features_mit124.csv").exists() else None
    c_rows = []
    if feat is not None:
        sp = dict(zip(feat.Cell_ID, feat.Severson_Split))
        sub = lab[lab.Theta.eq(0.80)]
        for (q), g in sub.groupby("Qref"):
            for rule in ("strict", "tolerance"):
                cells = set(g.Cell_ID[g[rule]])
                c_rows.append({"Qref": q, "Rule": rule, "Theta": 0.80, **{f"Labeled_{k}": sum(sp[c] == k for c in cells)
                                                                          for k in ("train", "test1", "test2")},
                               "Labeled_Total": len(cells)})
        pd.DataFrame(c_rows).to_csv(HERE / "3_mit_theta080_original_split_coverage.csv", index=False, encoding="utf-8-sig")
    print(win.to_string(index=False))
    print(pd.DataFrame(c_rows).to_string(index=False))
    summary = {"cells": int(n_cells), "estimable_settings": int(est.Estimable.sum()), "settings": int(len(est)),
               "estimable_in_common_window": int((est.Estimable & est.In_Common_Window).sum()),
               "settings_in_common_window": int(est.In_Common_Window.sum())}
    (HERE / "3_mit_label_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
