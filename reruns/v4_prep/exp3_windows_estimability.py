"""v4 preparation (read-only): experiment 3 (RF vs GB in the v2 pipeline) common windows and estimability preview.

Strict-rule common windows (both Qref settings >= 90% exact) for HUST / MIT (v2 grid 0.75-0.95) and TJU-NCA (v3 grid
0.60-0.95). Estimability of the RF setting (GB uses the same rule: >= 5 exact training cells per fold) and the E1/E2/E3 cell
counts are read from the existing v2 / v3 outputs (method_estimability.csv; the 'Cells' / pair-count columns of pair_ci.csv
for the RF-BASE pair). No comparison, interval or reversal value is read.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "project"
QL = {"nominal": "nominal", "first_comparable": "first", "early_median5_comparable": "first5"}


def shares_v2() -> pd.DataFrame:
    c = pd.read_csv(PROJECT / "outputs/followup_analyses_v1/2_labeled_cells_by_theta.csv", encoding="utf-8-sig")
    c["Theta"] = c.Theta.round(2)
    c["Share"] = c.Strict_Rule_Labeled / c.Cells
    return c[c.Dataset.isin(["HUST", "MIT"])][["Dataset", "Qref", "Theta", "Share"]]


def shares_tju() -> pd.DataFrame:
    c = pd.read_csv(HERE.parent / "tju_label_structure/2_labeled_share_by_theta.csv", encoding="utf-8-sig")
    c["Theta"] = c.Theta.round(2)
    c = c[c.Dataset.eq("NCA")].assign(Dataset="TJU_NCA")
    return c.rename(columns={"Strict_Share": "Share"})[["Dataset", "Qref", "Theta", "Share"]]


def estimability() -> pd.DataFrame:
    rows = []
    for d, path in (("HUST", "outputs/rul_ranking_pilot_v2"), ("MIT", "outputs/rul_ranking_pilot_v2"), ("TJU_NCA", "outputs/rul_ranking_v3_tju")):
        m = pd.read_csv(PROJECT / path / "method_estimability.csv", encoding="utf-8-sig")
        ci = pd.read_csv(PROJECT / path / "pair_ci.csv", encoding="utf-8-sig", usecols=["Dataset", "Theta", "Qref_Definition", "Pair", "Eval_Set", "Cells"])
        src = "NCA" if d == "TJU_NCA" else d
        m, ci = m[m.Dataset.eq(src)], ci[ci.Dataset.eq(src) & ci.Pair.eq("RF-BASE")]
        for f in (m, ci):
            f["Theta"] = f.Theta.round(2)
        rf = m[m.Method.eq("RF")].set_index(["Theta", "Qref_Definition"]).Estimable
        cells = ci.pivot_table(index=["Theta", "Qref_Definition"], columns="Eval_Set", values="Cells", aggfunc="first")
        for (th, q), ok in rf.items():
            rows.append({"Dataset": d, "Theta": th, "Qref": QL[q], "RF_GB_Estimable": bool(ok),
                         **{f"{e}_Cells": cells.at[(th, q), e] if (th, q) in cells.index else np.nan for e in ("E1", "E2", "E3")}})
    return pd.DataFrame(rows)


def main() -> None:
    sh = pd.concat([shares_v2(), shares_tju()]).set_index(["Dataset", "Qref", "Theta"]).Share
    est = estimability().set_index(["Dataset", "Theta", "Qref"])
    rows = []
    for d in ("HUST", "MIT", "TJU_NCA"):
        thetas = sorted({t for (dd, _, t) in sh.index if dd == d})
        for q1, q2 in combinations(["nominal", "first", "first5"], 2):
            for t in thetas:
                inw = sh[(d, q1, t)] >= 0.90 and sh[(d, q2, t)] >= 0.90
                a, b = est.loc[(d, t, q1)], est.loc[(d, t, q2)]
                rows.append({"Dataset": d, "Pair": f"{q1}-{q2}", "Theta": t, "In_Common_Window": inw,
                             "Both_Settings_Estimable": bool(a.RF_GB_Estimable and b.RF_GB_Estimable),
                             "E1_Cells": a.E1_Cells, "E2_Cells_min": min(a.E2_Cells, b.E2_Cells), "E3_Cells_min": min(a.E3_Cells, b.E3_Cells)})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "4_exp3_windows_estimability.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
    w = out[out.In_Common_Window]
    summ = w.groupby(["Dataset", "Pair"]).agg(Window_Points=("Theta", "size"),
                                             Window=("Theta", lambda s: f"{s.min():.2f}-{s.max():.2f}" if len(s) else ""),
                                             Estimable_Points=("Both_Settings_Estimable", "sum"),
                                             E1_Cells_min=("E1_Cells", "min"), E1_Cells_max=("E1_Cells", "max")).reset_index()
    summ.to_csv(HERE / "4_exp3_window_summary.csv", index=False, encoding="utf-8-sig")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
