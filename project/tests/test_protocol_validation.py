from __future__ import annotations

import numpy as np

from protocol_validation.common import noise_baselines, surface_change
from protocol_validation.endpoint import tolerance_rule


def test_tolerance_rule_interpolates_on_smoothed_curve() -> None:
    idx = np.arange(1, 11, dtype=float)
    cap = np.array([1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91])
    kind, eol = tolerance_rule(idx, cap, 0.945)
    assert kind == "INTERPOLATED" and np.isclose(eol, 6.5)


def test_single_dip_is_smoothed_away_and_tolerance_clause_applies() -> None:
    idx = np.arange(1, 11, dtype=float)
    cap = np.array([1.0, 0.99, 0.98, 0.80, 0.97, 0.96, 0.95, 0.94, 0.93, 0.9005])
    kind, eol = tolerance_rule(idx, cap, 0.90)          # dip removed by the median; last point within 1%
    assert kind == "TOLERANCE" and eol == 10.0
    assert tolerance_rule(idx, cap, 0.85)[0] == "NO_LABEL"
    assert tolerance_rule(idx, cap, 1.00) == ("FIRST_BELOW", 1.0)


def test_surface_change_and_noise_baselines() -> None:
    assert surface_change(-0.1, 0.2) and not surface_change(-0.1, -0.2) and not surface_change(0.0, 0.2)
    diff = np.r_[np.full(2500, -1.0), np.full(2500, 1.0)]
    n = noise_baselines(diff)
    assert n["Noise_Single_Resample_Disagreement"] == 1.0 and n["Noise_Half_Disagreement"] == 1.0
    assert noise_baselines(np.full(5000, -1.0))["Noise_Single_Resample_Disagreement"] == 0.0
