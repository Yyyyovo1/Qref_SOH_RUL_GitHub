"""TJU inputs in the v1/v2 DatasetInputs layout, endpoint detection on the v3 grid, t0 check, and features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rul_ranking_pilot.features import _level, _ols_slope
from rul_ranking_pilot.labels import BASELINE, DatasetInputs, _detection_record, detect

from .config import CAPACITY_FEATURES, DATASETS, NOMINAL_CAPACITY_AH, QREFS, STATIC_FEATURES, THETAS, TJU_DIR

QREF_COLUMN = {"nominal": "Qref_Nominal_Ah", "first_comparable": "Qref_First_Comparable_Ah",
               "early_median5_comparable": "Qref_Early_Median5_Comparable_Ah"}


def load_tju(root: Path) -> dict[str, DatasetInputs]:
    out = root / TJU_DIR
    cap = pd.read_csv(out / "tju_primary_capacity.csv", encoding="utf-8-sig")
    qtab = pd.read_csv(out / "tju_qref_summary.csv", encoding="utf-8-sig")
    folds = pd.read_csv(out / "tju_cell_fold_assignments.csv", encoding="utf-8-sig")
    inv = pd.read_csv(out / "tju_cell_inventory.csv", encoding="utf-8-sig")
    data = {}
    for d in DATASETS:
        q = qtab[qtab.Dataset.eq(d)]
        cells = sorted(q.Cell_ID.astype(str))
        c = cap[cap.Dataset.eq(d)].sort_values(["Cell_ID", "Cycle_Index"], kind="stable")
        capacity = pd.DataFrame({"Cell_ID": c.Cell_ID.astype(str), "Raw_Cycle_Index": c.Cycle_Index.astype(float),
                                 "Endpoint_Index": c.Cycle_Index.astype(float), "Capacity_Ah": c.Capacity_Ah.astype(float)})
        obs = capacity.rename(columns={"Capacity_Ah": "Primary_Capacity_Ah"})
        qref = pd.DataFrame({k: q[v].to_numpy(float) for k, v in QREF_COLUMN.items()}, index=q.Cell_ID.astype(str)).loc[cells]
        sources = {"first_comparable": {str(r.Cell_ID): [int(r.Qref_First_Comparable_Source_Cycle)] for r in q.itertuples()},
                   "early_median5_comparable": {str(r.Cell_ID): [int(x) for x in str(r.Qref_Early_Median5_Comparable_Source_Cycles).split("|")]
                                                for r in q.itertuples()}}
        f = folds[folds.Dataset.eq(d)]
        meta = inv[inv.Dataset.eq(d)].set_index("Cell_ID")
        static = pd.DataFrame({"Cell_ID": cells, "Temperature_C": meta.loc[cells, "Temperature_C"].to_numpy(float),
                               "Charge_CRate": meta.loc[cells, "Charge_CRate_Name"].to_numpy(float)})
        data[d] = DatasetInputs(
            name=d, cells=cells,
            observations={str(k): g[["Raw_Cycle_Index", "Primary_Capacity_Ah"]].reset_index(drop=True) for k, g in obs.groupby("Cell_ID")},
            capacity=capacity.reset_index(drop=True), qref=qref, source_cycles=sources, batch={},
            folds=dict(zip(f.Cell_ID.astype(str), f.Fold_ID.astype(int))), features=static)
    return data


def detect_all(data: dict[str, DatasetInputs], thetas=THETAS) -> pd.DataFrame:
    """v1/v2 detect_all on the v3 grid (frozen dense detector for non-XJTU datasets)."""
    rows = []
    for name, d in data.items():
        for q in QREFS:
            for theta in thetas:
                for cell in d.cells:
                    qref = float(d.qref.at[cell, q])
                    threshold = float(theta) * qref
                    rows.append({"Dataset": name, "Qref_Definition": q, "Theta": theta, "Cell_ID": cell, "Qref_Ah": qref,
                                 "Threshold_Ah": threshold, **_detection_record(detect(d, cell, threshold))})
    return pd.DataFrame(rows)


def compute_t0_1(data: dict[str, DatasetInputs], det: pd.DataFrame) -> dict:
    """v2 section 4 rule for t0_1 and M (same code path as v1 compute_t0), used by check 0."""
    out = {}
    for name, d in data.items():
        t_avail = max(max(v) for q in d.source_cycles for v in d.source_cycles[q].values())
        hits = det[(det.Dataset == name) & det.Status.ne(BASELINE)].dropna(subset=["First_Hit_Raw_Cycle"])
        e_min = float(hits.First_Hit_Raw_Cycle.min())
        m = float(np.median(d.capacity.groupby("Cell_ID").Raw_Cycle_Index.max()))
        t1 = int(max(t_avail, e_min - 1))
        t2 = max(t1 + 1, int(np.floor(0.35 * m)))
        t3 = max(t2 + 1, int(np.floor(0.50 * m)))
        out[name] = {"T_avail": int(t_avail), "E_min": e_min, "M": m, "t0": (t1, t2, t3)}
    return out


def base_features(d: DatasetInputs, t0s: tuple[int, int, int]) -> tuple[pd.DataFrame, float]:
    """Capacity-history features as in v2 (rul_pilot_v2.data.base_features) plus the static TJU test-design features."""
    cap = d.capacity
    static = d.features.set_index("Cell_ID")
    rows, max_used = [], 0.0
    for k, t in enumerate(t0s):
        c_t = cap[cap.Raw_Cycle_Index <= t]
        max_used = max(max_used, float(c_t.Raw_Cycle_Index.max()))
        groups = dict(tuple(c_t.groupby("Cell_ID")))
        for cell in d.cells:
            c = groups[cell]
            raw_c, q = c.Raw_Cycle_Index.to_numpy(float), c.Capacity_Ah.to_numpy(float)
            w = raw_c >= t / 2.0
            rec = {"Cell_ID": cell, "t_Index": k, "Capacity_t_Ah": _level(q),
                   "Capacity_Slope_Ah_per_cycle": _ols_slope(raw_c[w], q[w]), "Prediction_Time_Raw_Cycle": float(t)}
            for name in STATIC_FEATURES:
                rec[name] = float(static.at[cell, name])
            rows.append(rec)
    out = pd.DataFrame(rows)
    return out[["Cell_ID", "t_Index", *CAPACITY_FEATURES, *STATIC_FEATURES]], max_used


__all__ = ["NOMINAL_CAPACITY_AH", "base_features", "compute_t0_1", "detect_all", "load_tju"]
