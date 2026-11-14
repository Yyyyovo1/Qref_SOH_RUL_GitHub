"""Synthetic inputs for the v4 tests (no real data is read by these builders)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rul_v4_core_claim import exp1 as E1
from rul_v4_core_claim import exp1_labels as L1

QREFS = ("nominal", "first_comparable", "early_median5_comparable")


def exp1_inputs(n_cells: int = 30, thetas=(0.90,), rules=("strict", "tolerance"), seed: int = 7, dims=(1, 4, 6)) -> E1.Exp1Inputs:
    rng = np.random.default_rng(seed)
    cells = [f"c{i:02d}" for i in range(n_cells)]
    x = {c: rng.normal(size=max(dims)) for c in cells}
    life = {c: 10 ** (2.9 + 0.25 * x[c][0] + 0.05 * rng.normal()) for c in cells}
    rows = []
    for rule in rules:
        for k, q in enumerate(QREFS):
            for t in thetas:
                for i, c in enumerate(cells):
                    labeled = not (q != "nominal" and i % 11 == k)  # a few unlabeled cells -> different P cell sets
                    eol = life[c] * (1.0 - 0.5 * (t - 0.85)) * (1.0 - 0.02 * k) * (1.01 if rule == "tolerance" else 1.0)
                    rows.append({"Rule": rule, "Qref_Definition": q, "Theta": t, "Cell_ID": c, "Labeled": labeled,
                                 "EOL": eol if labeled else np.nan})
    labels = pd.DataFrame(rows)
    labels["Labeled_EOL_gt_100"] = labels.Labeled & (labels.EOL > 100)
    sh = L1.shares(labels, n_cells)
    # every definition >= 90% labeled here, so every pair is in its window
    win = L1.windows(sh, thetas)
    sets = L1.cell_sets(labels, win, cells)
    design = {"variance": {c: x[c][: dims[0]] for c in cells}, "discharge": {c: x[c][: dims[1]] for c in cells},
              "full": {c: x[c][: dims[2]] for c in cells}}
    split = {c: ("train" if i < n_cells // 3 else "test1" if i < 2 * n_cells // 3 else "test2") for i, c in enumerate(cells)}
    return E1.Exp1Inputs(cells=cells, design=design, eol=L1.eol_lookup(labels), sets=sets,
                         folds={c: i % 5 + 1 for i, c in enumerate(cells)}, split=split), labels, sh, win


def rul_samples(n_cells: int = 30, seed: int = 3, exact_every: int = 1):
    """One dataset x theta x Qref worth of eligible (cell, t) samples, a design matrix and capacity curves."""
    rng = np.random.default_rng(seed)
    cells = [f"s{i:02d}" for i in range(n_cells)]
    rows, cap = [], {}
    t0 = (50, 100, 150)
    for i, c in enumerate(cells):
        eol = 300 + 40 * i + rng.integers(0, 20)
        n = np.arange(1, eol + 30)
        q = 1.1 - (0.25 / eol) * n + rng.normal(0, 0.002, n.size)
        cap[c] = pd.DataFrame({"Raw_Cycle_Index": n.astype(float), "Endpoint_Index": n.astype(float), "Capacity_Ah": q})
        status = "EXACT" if i % exact_every == 0 else "RIGHT"
        for k, t in enumerate(t0):
            rul = float(eol - t)
            rows.append({"Cell_ID": c, "t_Index": k, "Fold_ID": i % 5 + 1, "Status": status,
                         "RUL": rul if status == "EXACT" else np.nan, "RUL_Lower": rul, "RUL_Upper": rul if status == "EXACT" else np.inf,
                         "EOL_Exact_Index": float(eol) if status == "EXACT" else np.nan, "n_t_Index": float(t)})
    s = pd.DataFrame(rows)
    x = np.column_stack([rng.normal(size=len(s)), s.t_Index.to_numpy(float), rng.normal(size=len(s))])
    thr = {c: 0.88 for c in cells}
    return s, x, cap, thr, t0
