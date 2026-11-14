"""v4 preparation (read-only): Severson et al. (2019) candidate features from the first 100 cycles of the frozen MIT cells,
and a cell-by-cell comparison with the only public per-cell reference (Attia et al. 2021, petermattia/revisit-severson-et-al).

Feature formulas follow Severson et al. Supplementary Note 1 / Supplementary Table 1. Cycle k means the k-th element of the
authors' MATLAB struct array (h5py row k-1), as in the authors' data files and in Attia et al.'s generate_voltage_arrays.m.
No model is fitted. Usage: python severson_features.py <raw MIT dir> <Attia data dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "project"
FILES = {1: "2017-05-12_batchdata_updated_struct_errorcorrect.mat", 2: "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
         3: "2018-04-12_batchdata_updated_struct_errorcorrect.mat"}
DQ_FEATURES = ["DeltaQ_Min", "DeltaQ_Mean", "DeltaQ_Variance", "DeltaQ_Skewness", "DeltaQ_Kurtosis", "DeltaQ_Value_2V"]
FADE_FEATURES = ["Slope_2_100", "Intercept_2_100", "Slope_91_100", "Intercept_91_100", "QD_Cycle2", "QD_Max_minus_Cycle2",
                 "QD_Cycle100"]
OTHER_FEATURES = ["ChargeTime_Avg_2_6", "Tmax_2_100", "Tmin_2_100", "T_Integral_2_100", "IR_Cycle2", "IR_Min_2_100",
                  "IR_100_minus_2"]
MODELS = {"variance": ["DeltaQ_Variance"],
          "discharge_candidates": DQ_FEATURES + FADE_FEATURES,
          "full_candidates": DQ_FEATURES + FADE_FEATURES + OTHER_FEATURES,
          "discharge_selected_in_paper": ["DeltaQ_Min", "DeltaQ_Variance", "DeltaQ_Skewness", "DeltaQ_Kurtosis", "QD_Cycle2",
                                          "QD_Max_minus_Cycle2"],
          "full_selected_in_paper": ["DeltaQ_Min", "DeltaQ_Variance", "Slope_2_100", "Intercept_2_100", "QD_Cycle2",
                                     "ChargeTime_Avg_2_6", "T_Integral_2_100", "IR_Min_2_100", "IR_100_minus_2"]}


def dq_features(dq: np.ndarray) -> dict:
    """SI Note 1: log10 of |statistic| of Delta Q_100-10(V); variance with 1/(p-1); skewness/kurtosis with 1/p moments."""
    p = dq.size
    d = dq - dq.mean()
    m2, m3, m4 = np.mean(d ** 2), np.mean(d ** 3), np.mean(d ** 4)
    return {"DeltaQ_Min": np.log10(abs(dq.min())), "DeltaQ_Mean": np.log10(abs(dq.mean())),
            "DeltaQ_Variance": np.log10(abs(np.sum(d ** 2) / (p - 1))),
            "DeltaQ_Skewness": np.log10(abs(m3 / m2 ** 1.5)), "DeltaQ_Kurtosis": np.log10(abs(m4 / m2 ** 2)),
            "DeltaQ_Value_2V": np.log10(abs(dq[-1]))}


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    a = np.column_stack([x, np.ones_like(x)])
    w = np.linalg.lstsq(a, y, rcond=None)[0]
    return float(w[0]), float(w[1])


def cell_features(f, b, idx: int) -> tuple[dict, np.ndarray]:
    summ = f[b["summary"][idx, 0]]
    s = {k: np.hstack(summ[k][0, :].tolist()).astype(float) for k in ("QDischarge", "chargetime", "IR", "Tmax", "Tmin")}
    cyc = f[b["cycles"][idx, 0]]
    get = lambda k, j: np.asarray(f[cyc[k][j, 0]][()], dtype=float).ravel()  # noqa: E731  (j = row = cycle - 1)
    q10, q100 = get("Qdlin", 9), get("Qdlin", 99)
    dq = q100 - q10
    n = np.arange(2, 101)
    qd = s["QDischarge"][1:100]
    sl, ic = fit(n.astype(float), qd)
    sl91, ic91 = fit(np.arange(91, 101).astype(float), s["QDischarge"][90:100])
    t_int = 0.0
    for j in range(1, 100):
        t, temp = get("t", j), get("T", j)
        t_int += float(np.trapezoid(temp, t))
    feats = {**dq_features(dq), "Slope_2_100": sl, "Intercept_2_100": ic, "Slope_91_100": sl91, "Intercept_91_100": ic91,
             "QD_Cycle2": float(s["QDischarge"][1]), "QD_Max_minus_Cycle2": float(qd.max() - s["QDischarge"][1]),
             "QD_Cycle100": float(s["QDischarge"][99]), "ChargeTime_Avg_2_6": float(np.mean(s["chargetime"][1:6])),
             "Tmax_2_100": float(np.max(s["Tmax"][1:100])), "Tmin_2_100": float(np.min(s["Tmin"][1:100])),
             "T_Integral_2_100": t_int, "IR_Cycle2": float(s["IR"][1]), "IR_Min_2_100": float(np.min(s["IR"][1:100])),
             "IR_100_minus_2": float(s["IR"][99] - s["IR"][1])}
    arr = np.column_stack([get("Qdlin", j) for j in range(1, 100)])  # cycles 2..100, as in Attia et al.'s arrays
    return feats, arr


def main(raw_dir: Path, attia_dir: Path) -> None:
    import h5py

    inv = pd.read_csv(PROJECT / "outputs/mit_124_final_v1/mit_cell_inventory.csv", encoding="utf-8-sig")
    inv = inv[inv.Status.eq("INCLUDED")].sort_values(["Batch", "Batch_Index"])
    assert len(inv) == 124 and list(inv.groupby("Batch").size()) == [41, 43, 40]
    # Severson / Attia split: concatenated order b1, b2, b3; test1 = odd positions of the first 84 plus 84.
    order = inv.Source_Key.tolist()
    pos = np.arange(1, 125)
    test1 = set(list(range(1, 85, 2)) + [84])
    split = ["test1" if p in test1 else ("train" if p <= 84 else "test2") for p in pos]
    rank = {}
    for name in ("train", "test1", "test2"):
        k = [c for c, s in zip(order, split) if s == name]
        rank.update({c: (name, i + 1) for i, c in enumerate(k)})
    rows, arrays = [], {}
    for batch, g in inv.groupby("Batch"):
        with h5py.File(raw_dir / FILES[batch], "r") as f:
            b = f["batch"]
            for r in g.itertuples(index=False):
                feats, arr = cell_features(f, b, int(r.Batch_Index))
                arrays[r.Source_Key] = arr
                rows.append({"Cell_ID": r.Source_Key, "Batch": batch, "Severson_Split": rank[r.Source_Key][0],
                             "Split_Position": rank[r.Source_Key][1], "Author_Cycle_Life": r.Author_Cycle_Life, **feats})
        print("batch", batch, "done", flush=True)
    feat = pd.DataFrame(rows)
    feat.to_csv(HERE / "1_severson_features_mit124.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

    # Comparison with Attia et al. (2021): cycle lives per split and Qdlin arrays (cycles 2..100) per cell.
    cmp_rows = []
    for r in feat.itertuples(index=False):
        lives = pd.read_csv(attia_dir / "cycle_lives" / f"{r.Severson_Split}_cycle_lives.csv", header=None)[0].to_numpy()
        ref = pd.read_csv(attia_dir / r.Severson_Split / f"cell{r.Split_Position}.csv", header=None).to_numpy(float)
        mine = arrays[r.Cell_ID]
        ref_dq = ref[:, 98] - ref[:, 8]
        f_ref = dq_features(ref_dq)
        cmp_rows.append({"Cell_ID": r.Cell_ID, "Severson_Split": r.Severson_Split, "Split_Position": r.Split_Position,
                         "Cycle_Life_Attia": int(lives[r.Split_Position - 1]), "Author_Cycle_Life_Frozen": r.Author_Cycle_Life,
                         "Cycle_Life_Equal": int(lives[r.Split_Position - 1]) == int(r.Author_Cycle_Life),
                         "Array_Shape_Equal": ref.shape == mine.shape,
                         "Array_Max_Abs_Diff": float(np.max(np.abs(ref - mine))) if ref.shape == mine.shape else np.nan,
                         **{f"{k}_Abs_Diff": abs(f_ref[k] - getattr(r, k)) for k in DQ_FEATURES}})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(HERE / "1_feature_comparison_vs_attia2021.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
    lc = np.log10(feat.Author_Cycle_Life.astype(float))
    excl = feat.Author_Cycle_Life.astype(float) > feat.Author_Cycle_Life.astype(float).min()
    summary = {
        "cells": int(len(feat)), "split_sizes": feat.Severson_Split.value_counts().to_dict(),
        "cycle_life_equal_all": bool(cmp.Cycle_Life_Equal.all()),
        "array_shapes_equal_all": bool(cmp.Array_Shape_Equal.all()),
        "array_max_abs_diff": float(cmp.Array_Max_Abs_Diff.max()),
        "deltaq_feature_max_abs_diff": {k: float(cmp[f"{k}_Abs_Diff"].max()) for k in DQ_FEATURES},
        "corr_log_var_vs_log_cycle_life_excluding_shortest": float(np.corrcoef(feat.DeltaQ_Variance[excl], lc[excl])[0, 1]),
        "paper_reports_corr": -0.93,
        "models": MODELS,
        "nonfinite_features": {c: int((~np.isfinite(feat[c])).sum()) for c in DQ_FEATURES + FADE_FEATURES + OTHER_FEATURES},
    }
    (HERE / "1_feature_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
