"""SOH part: trap A (SOH comparisons, main and supplementary scope) and trap B (information condition)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import _better, _interval, draws_for, noise_baselines, s7_frequencies, surface_change

NOMINAL = {"XJTU": 2.0, "HUST": 1.1}
QREFS = ["nominal", "first_comparable", "early_median5_comparable"]
QLABEL = {"nominal": "nominal", "first_comparable": "first", "early_median5_comparable": "first5"}
MODELS = {"Random Forest Regressor": "RF", "Gradient Boosting Regressor": "GB", "Ridge Regression": "Ridge"}
PAIRS = [("RF", "GB"), ("RF", "Ridge"), ("GB", "Ridge")]
QPAIRS = [("nominal", "first"), ("nominal", "first5"), ("first", "first5")]


def soh_cell_mae(root: Path) -> pd.DataFrame:
    """Per-cell MAE_Q (Ah) for the primary A9 benchmark under OWN / TRAIN / NOM conversions, plus XJTU A8 (OWN)."""
    p = pd.read_csv(root / "outputs/soh_primary_benchmark_v1/soh_primary_predictions.csv", encoding="utf-8-sig",
                    low_memory=False, usecols=["Analysis_ID", "Dataset", "Cell_ID", "Fold_ID", "Qref_Definition",
                                               "Qref_Ah", "Model", "Predicted_SOH_Fraction", "Observed_Capacity_Ah"])
    p = p[p.Analysis_ID.str.contains("PRIMARY_COMMON")].copy()
    cellq = p.drop_duplicates(["Dataset", "Qref_Definition", "Cell_ID"])
    train_med = {(d, q, k): float(g.loc[g.Fold_ID != k, "Qref_Ah"].median())
                 for (d, q), g in cellq.groupby(["Dataset", "Qref_Definition"]) for k in range(1, 6)}
    scale = {"OWN": p.Qref_Ah,
             "TRAIN": pd.Series([train_med[t] for t in zip(p.Dataset, p.Qref_Definition, p.Fold_ID)], index=p.index),
             "NOM": p.Dataset.map(NOMINAL)}
    frames = []
    for conv, s in scale.items():
        err = (p.Predicted_SOH_Fraction * s - p.Observed_Capacity_Ah).abs()
        c = err.groupby([p.Dataset, p.Qref_Definition, p.Model, p.Cell_ID]).mean().rename("MAE_Q_Ah").reset_index()
        frames.append(c.assign(Scope=f"A9_{conv}", Conversion=conv))
    a8 = pd.read_csv(root / "outputs/xjtu_a8_deterministic_reconstruction_v1/xjtu_a8_cell_metrics.csv", encoding="utf-8-sig")
    a8 = a8[a8.Qref_Definition.isin(QREFS)][["Dataset", "Qref_Definition", "Model", "Cell_ID", "MAE_Q_Ah"]]
    frames.append(a8.assign(Scope="A8_OWN", Conversion="OWN"))
    out = pd.concat(frames, ignore_index=True)
    out["Model"] = out.Model.map(MODELS)
    out["Qref"] = out.Qref_Definition.map(QLABEL)
    return out


def _boot(values: np.ndarray, draws: np.ndarray) -> np.ndarray:
    return np.median(values[draws], axis=1)


def soh_pair_diffs(cell: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Bootstrap differences median(A) - median(B) per (scope, dataset, Qref, pair); same draws as reproduce_s6_s7."""
    diffs, rows = {}, []
    for (scope, d), g in cell.groupby(["Scope", "Dataset"], sort=True):
        cells = sorted(g.Cell_ID.unique())
        draws, _ = draws_for(len(cells))
        pv = g.pivot_table(index="Cell_ID", columns=["Qref", "Model"], values="MAE_Q_Ah").reindex(cells)
        for q in ["nominal", "first", "first5"]:
            for a, b in PAIRS:
                va, vb = pv[(q, a)].to_numpy(float), pv[(q, b)].to_numpy(float)
                diff = _boot(va, draws) - _boot(vb, draws)
                obs = float(np.median(va) - np.median(vb))
                key = (scope, d, q, f"{a}-{b}")
                diffs[key] = (obs, diff)
                r = {"Scope": scope, "Dataset": d, "Qref": q, "Pair": f"{a}-{b}", "Cells": len(cells),
                     "Observed_Difference": obs, **_interval(diff), **noise_baselines(diff)}
                r["Better"] = _better(r, a, b, higher_is_better=False)
                rows.append(r)
    return diffs, pd.DataFrame(rows)


def soh_comparisons(diffs: dict, settings: pd.DataFrame) -> pd.DataFrame:
    better = settings.set_index(["Scope", "Dataset", "Qref", "Pair"]).Better.to_dict()
    rows = []
    for scope, d in sorted({(k[0], k[1]) for k in diffs}):
        for a, b in PAIRS:
            pair = f"{a}-{b}"
            for q1, q2 in QPAIRS:
                o1, d1 = diffs[(scope, d, q1, pair)]
                o2, d2 = diffs[(scope, d, q2, pair)]
                b1, b2 = better[(scope, d, q1, pair)], better[(scope, d, q2, pair)]
                rows.append({"Scope": scope, "Dataset": d, "Pair": pair, "Setting_1": q1, "Setting_2": q2,
                             "Observed_1": o1, "Observed_2": o2, "Surface_Change": surface_change(o1, o2),
                             "Better_1": b1, "Better_2": b2, "Supported_Reversal": bool(b1 and b2 and b1 != b2),
                             **s7_frequencies(d1, d2, o1, o2)})
    return pd.DataFrame(rows)


def trap_b(cell: pd.DataFrame) -> pd.DataFrame:
    """Measured minus nominal median MAE_Q per model, OWN vs TRAIN (scaling_bootstrap convention)."""
    rows = []
    for (scope, d), g in cell[cell.Scope.isin(["A9_OWN", "A9_TRAIN"])].groupby(["Scope", "Dataset"]):
        cells = sorted(g.Cell_ID.unique())
        draws, _ = draws_for(len(cells))
        pv = g.pivot_table(index="Cell_ID", columns=["Qref", "Model"], values="MAE_Q_Ah").reindex(cells)
        for m in ["RF", "GB", "Ridge"]:
            base = pv[("nominal", m)].to_numpy(float)
            for q in ["first", "first5"]:
                v = pv[(q, m)].to_numpy(float)
                diff = _boot(v, draws) - _boot(base, draws)
                iv = _interval(diff)
                lo, hi = iv["CI95_Lower"], iv["CI95_Upper"]
                rows.append({"Dataset": d, "Model": m, "Measured_Qref": q, "Conversion": scope.split("_")[1],
                             "Observed_Measured_Minus_Nominal_Ah": float(np.median(v) - np.median(base)),
                             "CI95_Lower_Ah": lo, "CI95_Upper_Ah": hi,
                             "Conclusion": "HIGHER" if lo > 0 else "LOWER" if hi < 0 else "UNDETERMINED"})
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=["Dataset", "Model", "Measured_Qref"], columns="Conversion",
                            values=["Conclusion", "Observed_Measured_Minus_Nominal_Ah"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["Conclusion_Depends_On_Information_Condition"] = wide.Conclusion_OWN != wide.Conclusion_TRAIN
    wide["Point_Direction_OWN"] = np.sign(wide.Observed_Measured_Minus_Nominal_Ah_OWN.astype(float))
    wide["Point_Direction_TRAIN"] = np.sign(wide.Observed_Measured_Minus_Nominal_Ah_TRAIN.astype(float))
    return long, wide
