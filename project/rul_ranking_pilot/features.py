"""Cell-level features at t0 (PROTOCOL.md section 4.2) and the check-4 feature audits."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import A9_FEATURES, CAL_EXTRA_FEATURES, FORBIDDEN_NOCAL_TOKENS, LEVEL_LAST_N
from .labels import DatasetInputs, PilotError


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    keep = ~np.isnan(y)
    x, y = x[keep], y[keep]
    if len(x) < 2 or np.unique(x).size < 2:
        return np.nan
    xc = x - x.mean()
    return float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))


def _level(values: np.ndarray) -> float:
    tail = values[-LEVEL_LAST_N:]
    tail = tail[~np.isnan(tail)]
    return float(np.median(tail)) if len(tail) else np.nan


def nocal_features(data: DatasetInputs, t0: int) -> tuple[pd.DataFrame, float]:
    """18 features per cell from A9 rows with raw cycle <= t0; returns (frame, max raw cycle used)."""
    frame = data.features.copy()
    frame["Cell_ID"] = frame["Cell_ID"].astype(str)
    frame = frame[frame["Raw_Cycle_Index"] <= t0].sort_values(["Cell_ID", "Raw_Cycle_Index"], kind="stable")
    max_raw = float(frame["Raw_Cycle_Index"].max())
    rows = {}
    for cell in data.cells:
        g = frame[frame["Cell_ID"] == cell]
        raw = g["Raw_Cycle_Index"].to_numpy(float)
        window = raw >= t0 / 2.0
        record = {}
        for f in A9_FEATURES:
            v = pd.to_numeric(g[f], errors="coerce").to_numpy(float)
            record[f"{f}__level"] = _level(v)
            record[f"{f}__slope"] = _ols_slope(raw[window], v[window])
        rows[cell] = record
    out = pd.DataFrame.from_dict(rows, orient="index").loc[data.cells]
    out.index.name = "Cell_ID"
    return out, max_raw


def capacity_summary(data: DatasetInputs, t0: int) -> tuple[pd.DataFrame, float]:
    """C(t0) and capacity slope per cell from capacity observations with raw cycle <= t0."""
    cap = data.capacity[data.capacity["Raw_Cycle_Index"] <= t0]
    max_raw = float(cap["Raw_Cycle_Index"].max())
    rows = {}
    for cell, g in cap.groupby("Cell_ID"):
        raw = g["Raw_Cycle_Index"].to_numpy(float)
        q = g["Capacity_Ah"].to_numpy(float)
        window = raw >= t0 / 2.0
        rows[str(cell)] = {"Capacity_t0_Ah": _level(q), "Capacity_Slope_Ah_per_cycle": _ols_slope(raw[window], q[window])}
    out = pd.DataFrame.from_dict(rows, orient="index").loc[data.cells]
    out.index.name = "Cell_ID"
    return out, max_raw


def cal_features(nocal: pd.DataFrame, capacity: pd.DataFrame, qref: pd.Series, theta: float) -> pd.DataFrame:
    """NOCAL features plus C(t0), capacity slope, the cell's own Qref and C(t0) - theta*Qref."""
    out = nocal.copy()
    out["Capacity_t0_Ah"] = capacity["Capacity_t0_Ah"]
    out["Capacity_Slope_Ah_per_cycle"] = capacity["Capacity_Slope_Ah_per_cycle"]
    out["Qref_Ah"] = qref.loc[out.index].to_numpy(float)
    out["Margin_To_Threshold_Ah"] = out["Capacity_t0_Ah"] - float(theta) * out["Qref_Ah"]
    if list(out.columns[-4:]) != list(CAL_EXTRA_FEATURES):
        raise PilotError("CAL_FEATURE_ORDER")
    return out


def frame_sha256(frame: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update("|".join(map(str, frame.columns)).encode())
    h.update("|".join(map(str, frame.index)).encode())
    h.update(np.ascontiguousarray(frame.to_numpy(float)).tobytes())
    return h.hexdigest()


def audit_nocal(nocal: pd.DataFrame) -> None:
    bad = [c for c in nocal.columns if any(t in c.lower() for t in FORBIDDEN_NOCAL_TOKENS)]
    if bad:
        raise PilotError("NOCAL_FEATURES_CONTAIN_CALIBRATION_COLUMNS:" + "|".join(bad))
