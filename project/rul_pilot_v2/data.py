"""Inputs for v2: XJTU/HUST via the v1 loader, MIT from its frozen outputs; samples and features over t0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rul_ranking_pilot.features import _level, _ols_slope
from rul_ranking_pilot.labels import DatasetInputs, PilotError, load_inputs

from .config import (
    BASELINE,
    CAPACITY_FEATURES,
    DATASET_FEATURES,
    EXACT,
    INTERVAL,
    MIT_DIR,
    QREFS,
    RIGHT,
    T0_FILE,
)


def _mit_inputs(root: Path) -> DatasetInputs:
    out = root / MIT_DIR
    cap = pd.read_csv(out / "mit_primary_capacity.csv", encoding="utf-8-sig")
    qtab = pd.read_csv(out / "mit_qref_summary.csv", encoding="utf-8-sig")
    folds = pd.read_csv(out / "mit_cell_fold_assignments.csv", encoding="utf-8-sig")
    cells = sorted(qtab.Cell_ID.astype(str))
    obs = cap.rename(columns={"Cycle_Index": "Raw_Cycle_Index", "QDischarge_Ah": "Primary_Capacity_Ah"})
    capacity = pd.DataFrame({"Cell_ID": cap.Cell_ID.astype(str), "Raw_Cycle_Index": cap.Cycle_Index.astype(float),
                             "Endpoint_Index": cap.Cycle_Index.astype(float), "Capacity_Ah": cap.QDischarge_Ah.astype(float)})
    qref = pd.DataFrame({"nominal": qtab.Qref_Nominal_Ah.to_numpy(float),
                         "first_comparable": qtab.Qref_First_Comparable_Ah.to_numpy(float),
                         "early_median5_comparable": qtab.Qref_Early_Median5_Comparable_Ah.to_numpy(float)},
                        index=qtab.Cell_ID.astype(str)).loc[cells]
    sources = {
        "first_comparable": {str(c): [int(v)] for c, v in zip(qtab.Cell_ID, qtab.Qref_First_Comparable_Source_Cycle)},
        "early_median5_comparable": {str(c): [int(x) for x in str(v).split("|")]
                                     for c, v in zip(qtab.Cell_ID, qtab.Qref_Early_Median5_Comparable_Source_Cycles)},
    }
    features = cap[["Cell_ID", "Cycle_Index", *DATASET_FEATURES["MIT"]]].rename(columns={"Cycle_Index": "Raw_Cycle_Index"})
    return DatasetInputs(
        name="MIT", cells=cells,
        observations={str(c): g[["Raw_Cycle_Index", "Primary_Capacity_Ah"]].copy() for c, g in obs.groupby("Cell_ID")},
        capacity=capacity.sort_values(["Cell_ID", "Raw_Cycle_Index"]).reset_index(drop=True),
        qref=qref, source_cycles=sources, batch={},
        folds=dict(zip(folds.Cell_ID.astype(str), folds.Fold_ID.astype(int))), features=features)


def load_all(root: Path) -> dict[str, DatasetInputs]:
    data = load_inputs(root)
    data["MIT"] = _mit_inputs(root)
    return data


def load_t0(root: Path) -> dict[str, tuple[int, int, int]]:
    t = pd.read_csv(root / T0_FILE, encoding="utf-8-sig")
    return {r.Dataset: (int(r.t0_1), int(r.t0_2), int(r.t0_3)) for r in t.itertuples(index=False)}


def build_samples(data: dict[str, DatasetInputs], detections: pd.DataFrame, t0: dict) -> pd.DataFrame:
    """One row per dataset x Qref x theta x cell x t0: eligibility and t-relative RUL bounds."""
    n_t = {}
    for name, d in data.items():
        for k, t in enumerate(t0[name]):
            last = d.capacity[d.capacity.Raw_Cycle_Index <= t].groupby("Cell_ID").Endpoint_Index.max()
            if set(last.index) != set(d.cells):
                raise PilotError(f"{name}_CELL_WITHOUT_OBSERVATION_BEFORE_t0_{k + 1}")
            n_t.update({(name, k, c): float(v) for c, v in last.items()})
    frames = []
    for k in range(3):
        s = detections.copy()
        s["t_Index"] = k
        s["t_Raw_Cycle"] = [t0[d][k] for d in s.Dataset]
        s["n_t_Index"] = [n_t[(d, k, c)] for d, c in zip(s.Dataset, s.Cell_ID)]
        s["EOL_Before_t"] = s.First_Hit_Raw_Cycle.le(s.t_Raw_Cycle).fillna(False)
        s["Eligible"] = ~s.EOL_Before_t & s.Status.ne(BASELINE)
        n = s.n_t_Index
        s["RUL"] = np.where(s.Status.eq(EXACT), s.EOL_Exact_Index - n, np.nan)
        s["RUL_Lower"] = np.select([s.Status.eq(EXACT), s.Status.eq(INTERVAL), s.Status.eq(RIGHT)],
                                   [s.RUL, s.EOL_Lower_Index - n, s.Last_Observed_Index - n], np.nan)
        s["RUL_Upper"] = np.select([s.Status.eq(EXACT), s.Status.eq(INTERVAL), s.Status.eq(RIGHT)],
                                   [s.RUL, s.EOL_Upper_Index - n, np.inf], np.nan)
        frames.append(s)
    samples = pd.concat(frames, ignore_index=True)
    if (samples.Eligible & (samples.RUL_Upper <= 0)).any():
        raise PilotError("ELIGIBLE_SAMPLE_WITH_NONPOSITIVE_RUL")
    samples["Fold_ID"] = [data[d].folds[c] for d, c in zip(samples.Dataset, samples.Cell_ID)]
    return samples


def base_features(d: DatasetInputs, t0s: tuple[int, int, int]) -> tuple[pd.DataFrame, float]:
    """Qref-independent features per (cell, t): capacity history, prediction time, dataset-specific level/slope."""
    feat = d.features.copy()
    feat["Cell_ID"] = feat.Cell_ID.astype(str)
    cap = d.capacity
    rows, max_used = [], 0.0
    for k, t in enumerate(t0s):
        f_t = feat[feat.Raw_Cycle_Index <= t].sort_values(["Cell_ID", "Raw_Cycle_Index"], kind="stable")
        c_t = cap[cap.Raw_Cycle_Index <= t]
        max_used = max(max_used, float(f_t.Raw_Cycle_Index.max()), float(c_t.Raw_Cycle_Index.max()))
        f_groups = dict(tuple(f_t.groupby("Cell_ID")))
        c_groups = dict(tuple(c_t.groupby("Cell_ID")))
        for cell in d.cells:
            g, c = f_groups[cell], c_groups[cell]
            raw_c, q = c.Raw_Cycle_Index.to_numpy(float), c.Capacity_Ah.to_numpy(float)
            w = raw_c >= t / 2.0
            rec = {"Cell_ID": cell, "t_Index": k, "Capacity_t_Ah": _level(q),
                   "Capacity_Slope_Ah_per_cycle": _ols_slope(raw_c[w], q[w]), "Prediction_Time_Raw_Cycle": float(t)}
            raw_f = g.Raw_Cycle_Index.to_numpy(float)
            wf = raw_f >= t / 2.0
            for name in DATASET_FEATURES[d.name]:
                v = pd.to_numeric(g[name], errors="coerce").to_numpy(float)
                rec[f"{name}__level"] = _level(v)
                rec[f"{name}__slope"] = _ols_slope(raw_f[wf], v[wf])
            rows.append(rec)
    out = pd.DataFrame(rows)
    names = [*CAPACITY_FEATURES, *[f"{n}__{s}" for n in DATASET_FEATURES[d.name] for s in ("level", "slope")]]
    return out[["Cell_ID", "t_Index", *names]], max_used
