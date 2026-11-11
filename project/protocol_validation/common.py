"""Shared bootstrap helpers; every rule follows PROTOCOL_v2 (no new decision rules)."""

from __future__ import annotations

import warnings

import numpy as np

from rul_pilot_v2.evaluation import _better, _interval, _nanmedian_rows  # noqa: F401  (v2 rules)
from rul_ranking_pilot.evaluation import draws_for  # noqa: F401  (seed 20260928, 5000 draws)

HALF = 2500


def signs(diff: np.ndarray) -> np.ndarray:
    """Sign of each bootstrap difference; NaN stays NaN."""
    out = np.sign(diff).astype(float)
    out[np.isnan(diff)] = np.nan
    return out


def noise_baselines(diff: np.ndarray) -> dict:
    """Single-resample disagreement (main: draw b vs b+2500) and half-vs-half disagreement (auxiliary)."""
    a, b = diff[:HALF], diff[HALF:2 * HALF]
    ok = ~np.isnan(a) & ~np.isnan(b)
    single = float(np.mean(np.sign(a[ok]) != np.sign(b[ok]))) if ok.any() else np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ma, mb = np.nanmedian(a), np.nanmedian(b)
    half = float(np.sign(ma) != np.sign(mb)) if np.isfinite(ma) and np.isfinite(mb) else np.nan
    return {"Noise_Single_Resample_Disagreement": single, "Noise_Half_Disagreement": half}


def s7_frequencies(d1: np.ndarray, d2: np.ndarray, obs1: float, obs2: float) -> dict:
    """Descriptive (S7 convention): per-resample sign flip between the two Qrefs, and joint agreement with the observed signs."""
    ok = ~np.isnan(d1) & ~np.isnan(d2)
    if not ok.any():
        return {"S7_Sign_Differs_Frequency": np.nan, "S7_Observed_Pattern_Frequency": np.nan}
    s1, s2 = np.sign(d1[ok]), np.sign(d2[ok])
    return {"S7_Sign_Differs_Frequency": float(np.mean(s1 != s2)),
            "S7_Observed_Pattern_Frequency": float(np.mean((s1 == np.sign(obs1)) & (s2 == np.sign(obs2))))}


def surface_change(obs1: float, obs2: float) -> bool:
    """Common practice: point-estimate order differs between the two Qrefs (exact zero differences are not counted)."""
    return bool(np.isfinite(obs1) and np.isfinite(obs2) and obs1 != 0 and obs2 != 0 and np.sign(obs1) != np.sign(obs2))
