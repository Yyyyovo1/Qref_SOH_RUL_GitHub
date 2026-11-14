"""Experiment 1 features (Severson et al. 2019) and checks S and 0 (PROTOCOL_v4 section 6, AMENDMENT_01 sections 1, 2, 7).

The feature formulas are not re-implemented: `cell_features` / `dq_features` / `MODELS` are imported by path from the
archived prep script `reruns/v4_prep/severson_features.py` (SHA-256 checked by prereg.check_archived before use).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ATTIA_DIR,
    CHECK_S_ABS_TOL,
    CHECK_S_REL_TOL,
    MIT_INVENTORY,
    MIT_RAW_DIR,
    PREDICTION_CYCLE,
    SEVERSON_FEATURES_REF,
    SEVERSON_SCRIPT,
)

ID_COLUMNS = ["Cell_ID", "Batch", "Severson_Split", "Split_Position", "Author_Cycle_Life"]


def load_prep_module(root: Path):
    path = (Path(root) / SEVERSON_SCRIPT).resolve()
    spec = importlib.util.spec_from_file_location("v4_prep_severson_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def feature_names(prep) -> list[str]:
    return list(prep.DQ_FEATURES + prep.FADE_FEATURES + prep.OTHER_FEATURES)


def severson_split(order: list[str]) -> dict[str, tuple[str, int]]:
    """Severson / Attia split, as in the prep script: concatenated order b1, b2, b3; test1 = odd positions of the first
    84 plus position 84; train = the other first 84; test2 = batch 3. Returns cell -> (split, position within split)."""
    test1 = set(list(range(1, 85, 2)) + [84])
    split = ["test1" if p in test1 else ("train" if p <= 84 else "test2") for p in range(1, len(order) + 1)]
    out = {}
    for name in ("train", "test1", "test2"):
        k = [c for c, s in zip(order, split) if s == name]
        out.update({c: (name, i + 1) for i, c in enumerate(k)})
    return out


# ---------------------------------------------------------------- truncated view (check 0 of experiment 1)
class _RecordingRefs:
    """Cycle-level reference array truncated to the first n cycles; records every row index accessed."""

    def __init__(self, refs: np.ndarray, n: int, log: list[int]):
        self._refs, self._log = refs[:n], log

    def __getitem__(self, key):
        row = key[0] if isinstance(key, tuple) else key
        self._log.append(int(row))
        return self._refs[key]


class _TruncatedGroup:
    def __init__(self, group, n: int, kind: str, log: list[int]):
        self._g, self._n, self._kind, self._log = group, n, kind, log

    def __getitem__(self, key):
        values = self._g[key][()]
        if self._kind == "summary":
            return values[:, : self._n]          # (1, cycles): keep cycles 1..n
        return _RecordingRefs(values, self._n, self._log)  # (cycles, 1) references: rows 0..n-1


class TruncatedFile:
    """h5py file view exposing only the first n cycles of each cell's summary and cycle-level data."""

    def __init__(self, f, n: int = PREDICTION_CYCLE):
        self._f, self._n = f, n
        self.rows_accessed: list[int] = []

    def __getitem__(self, key):
        import h5py

        obj = self._f[key]
        if isinstance(obj, h5py.Group):
            keys = set(obj.keys())
            if "QDischarge" in keys:
                return _TruncatedGroup(obj, self._n, "summary", self.rows_accessed)
            if "Qdlin" in keys:
                return _TruncatedGroup(obj, self._n, "cycles", self.rows_accessed)
        return obj


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


# ---------------------------------------------------------------- feature recomputation
def compute_features(root: Path, prep, inventory: pd.DataFrame | None = None, raw_dir: Path | None = None,
                     files: dict | None = None):
    """Recompute the 20 features for the 124 frozen MIT cells (prep `main`, same order and split rule).

    Returns (features frame, {cell: Qdlin array cycles 2..100}, check-0 table)."""
    import h5py

    root = Path(root)
    if inventory is None:
        inventory = pd.read_csv(root / MIT_INVENTORY, encoding="utf-8-sig")
    inv = inventory[inventory.Status.eq("INCLUDED")].sort_values(["Batch", "Batch_Index"])
    raw_dir = Path(raw_dir) if raw_dir is not None else root / MIT_RAW_DIR
    files = files or prep.FILES
    rank = severson_split(inv.Source_Key.tolist())
    rows, arrays, check0 = [], {}, []
    for batch, g in inv.groupby("Batch"):
        with h5py.File(raw_dir / files[batch], "r") as f:
            b = f["batch"]
            for r in g.itertuples(index=False):
                idx = int(r.Batch_Index)
                feats, arr = prep.cell_features(f, b, idx)
                view = TruncatedFile(f)
                try:
                    t_feats, t_arr = prep.cell_features(view, b, idx)
                    ok = all(_same(feats[k], t_feats[k]) for k in feats) and np.array_equal(arr, t_arr)
                    reason = "" if ok else "FEATURES_DIFFER_WHEN_TRUNCATED_TO_100_CYCLES"
                except IndexError:
                    ok, reason = False, "ACCESS_BEYOND_CYCLE_100"
                max_row = max(view.rows_accessed) if view.rows_accessed else -1
                check0.append({"Cell_ID": r.Source_Key, "Truncated_Features_Identical": bool(ok),
                               "Max_Cycle_Row_Accessed": int(max_row), "Reason": reason,
                               "Pass": bool(ok and max_row <= PREDICTION_CYCLE - 1)})
                arrays[r.Source_Key] = arr
                rows.append({"Cell_ID": r.Source_Key, "Batch": batch, "Severson_Split": rank[r.Source_Key][0],
                             "Split_Position": rank[r.Source_Key][1], "Author_Cycle_Life": r.Author_Cycle_Life, **feats})
    return pd.DataFrame(rows), arrays, pd.DataFrame(check0)


# ---------------------------------------------------------------- check S
def check_s_features(feat: pd.DataFrame, ref: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, bool]:
    """Item 1 (amendment 01 section 1): relative difference <= 1e-11; absolute <= 1e-12 where the reference is 0."""
    a = feat.set_index("Cell_ID")
    r = ref.set_index("Cell_ID")
    ids_ok = set(a.index) == set(r.index) and len(a) == len(r)
    rows = []
    for c in ID_COLUMNS[1:]:
        rows.append({"Item": c, "Kind": "identity", "Cells": len(r),
                     "Mismatches": int((a.loc[r.index, c].astype(str).to_numpy() != r[c].astype(str).to_numpy()).sum()) if ids_ok else len(r),
                     "Recomputed_Max_Rel_Diff": np.nan, "Max_Abs_Diff_Zero_Reference": np.nan})
    for c in names:
        x = a.loc[r.index, c].to_numpy(float) if ids_ok else np.full(len(r), np.nan)
        y = r[c].to_numpy(float)
        zero = y == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(zero, np.nan, np.abs(x - y) / np.abs(y))
        absz = np.where(zero, np.abs(x), np.nan)
        bad = np.where(zero, ~(absz <= CHECK_S_ABS_TOL), ~(rel <= CHECK_S_REL_TOL))
        rows.append({"Item": c, "Kind": "feature", "Cells": len(r), "Mismatches": int(bad.sum()),
                     "Recomputed_Max_Rel_Diff": float(np.nanmax(rel)) if (~zero).any() else np.nan,
                     "Max_Abs_Diff_Zero_Reference": float(np.nanmax(absz)) if zero.any() else np.nan})
    table = pd.DataFrame(rows)
    return table, bool(ids_ok and (table.Mismatches == 0).all())


def round5(values: np.ndarray) -> np.ndarray:
    """Amendment 01 section 2: float(format(x, '.5g')) element-wise."""
    flat = np.asarray(values, dtype=float).ravel().tolist()
    return np.array([float(format(v, ".5g")) for v in flat], dtype=float).reshape(np.shape(values))


def check_s_arrays(feat: pd.DataFrame, arrays: dict, attia_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, bool, bool]:
    """Items 2 (Qdlin arrays, 5 significant digits) and 3 (cycle lives, split sizes)."""
    rows, life_rows = [], []
    lives = {s: pd.read_csv(attia_dir / "cycle_lives" / f"{s}_cycle_lives.csv", header=None, float_precision="round_trip")[0].to_numpy()
             for s in ("train", "test1", "test2")}
    for r in feat.itertuples(index=False):
        ref = pd.read_csv(attia_dir / r.Severson_Split / f"cell{r.Split_Position}.csv", header=None,
                          float_precision="round_trip").to_numpy(float)
        mine = arrays[r.Cell_ID]
        same_shape = ref.shape == mine.shape
        n_diff = int((round5(mine) != ref).sum()) if same_shape else int(ref.size)
        rows.append({"Cell_ID": r.Cell_ID, "Severson_Split": r.Severson_Split, "Split_Position": r.Split_Position,
                     "Shape_Equal": same_shape, "Elements": int(ref.size), "Elements_Not_Equal_After_Round5": n_diff,
                     "Pass": bool(same_shape and n_diff == 0)})
        attia_life = int(lives[r.Severson_Split][r.Split_Position - 1])
        life_rows.append({"Cell_ID": r.Cell_ID, "Severson_Split": r.Severson_Split, "Split_Position": r.Split_Position,
                          "Cycle_Life_Attia": attia_life, "Author_Cycle_Life": int(r.Author_Cycle_Life),
                          "Pass": attia_life == int(r.Author_Cycle_Life)})
    arr_t, life_t = pd.DataFrame(rows), pd.DataFrame(life_rows)
    sizes = feat.Severson_Split.value_counts().to_dict()
    sizes_ok = sizes == {"train": 41, "test1": 43, "test2": 40} and all(len(lives[s]) == sizes.get(s) for s in lives)
    return arr_t, life_t, bool(arr_t.Pass.all()), bool(life_t.Pass.all() and sizes_ok)


def run_check_s(root: Path, prep, feat: pd.DataFrame, arrays: dict, attia_dir: Path | None = None) -> dict:
    root = Path(root)
    ref = pd.read_csv(root / SEVERSON_FEATURES_REF, encoding="utf-8-sig", float_precision="round_trip")
    t1, ok1 = check_s_features(feat, ref, feature_names(prep))
    t2, t3, ok2, ok3 = check_s_arrays(feat, arrays, Path(attia_dir) if attia_dir else root / ATTIA_DIR)
    return {"features": t1, "qdlin": t2, "cycle_life": t3, "pass_1": ok1, "pass_2": ok2, "pass_3": ok3,
            "Pass": bool(ok1 and ok2 and ok3)}
