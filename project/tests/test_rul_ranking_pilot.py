from __future__ import annotations

import numpy as np
import pandas as pd

from rul_ranking_pilot.config import EXACT, INTERVAL, RIGHT, THETAS
from rul_ranking_pilot.evaluation import censoring_aware_error, draws_for, find_reversals, strong_rule
from rul_ranking_pilot.methods import traj_predict


def test_theta_grid_is_preregistered() -> None:
    assert len(THETAS) == 21 and THETAS[0] == 0.75 and THETAS[-1] == 0.95


def test_censoring_aware_error() -> None:
    status = np.array([EXACT, INTERVAL, INTERVAL, INTERVAL, RIGHT, RIGHT], dtype=object)
    pred = np.array([10.0, 5.0, 15.0, 25.0, 30.0, 50.0])
    rul = np.array([12.0, np.nan, np.nan, np.nan, np.nan, np.nan])
    lower = np.array([12.0, 10.0, 10.0, 10.0, 40.0, 40.0])
    upper = np.array([12.0, 20.0, 20.0, 20.0, np.inf, np.inf])
    err = censoring_aware_error(pred, status, rul, lower, upper)
    assert np.allclose(err, [2.0, 5.0, 0.0, 5.0, 10.0, 0.0])


def test_traj_line_crossing_and_fallback() -> None:
    cap = pd.DataFrame({"Raw_Cycle_Index": np.arange(1, 101, dtype=float),
                        "Endpoint_Index": np.arange(1, 101, dtype=float)})
    cap["Capacity_Ah"] = 2.0 - 0.001 * cap["Endpoint_Index"]
    pred, fb = traj_predict(cap, 100, 1.8, 100.0, fallback=-1.0)
    assert not fb and np.isclose(pred, 100.0)
    cap["Capacity_Ah"] = 2.0
    pred, fb = traj_predict(cap, 100, 1.8, 100.0, fallback=-1.0)
    assert fb and pred == -1.0


def test_xjtu_draws_match_s6_s7() -> None:
    _, digest = draws_for(55)
    assert digest == "2e8a1afa9101de2c4af7927c31f1e112adde5be2922437d03422aa90207e1ded"


def _ci(rows):
    return pd.DataFrame([{"Dataset": "D", "Theta": t, "Qref_Definition": q, "Cal": c, "Eval_Set": e, "Pair": "RF-WAFT",
                          "Estimable": True, "CI_Sign": s} for t, q, c, e, s in rows])


def test_reversal_requires_both_intervals_to_exclude_zero_with_opposite_signs() -> None:
    ci = _ci([(0.8, "nominal", "CAL", "E1", -1), (0.8, "first_comparable", "CAL", "E1", 1),
              (0.8, "early_median5_comparable", "CAL", "E1", 0)])
    rev = find_reversals(ci)
    hit = rev[rev.Supported_Reversal]
    assert len(hit) == 1 and hit.iloc[0].Setting_2 == "first_comparable|CAL"


def test_strong_rule_needs_confirmation_outside_e1() -> None:
    rows = []
    for t in THETAS[:6]:
        rows += [(t, "nominal", "CAL", "E1", -1), (t, "first_comparable", "CAL", "E1", 1)]
    s = strong_rule(find_reversals(_ci(rows)))
    assert not s.Meets_Strong_Threshold.any()
    for t in THETAS[:6]:
        rows += [(t, "nominal", "CAL", "E2", -1), (t, "first_comparable", "CAL", "E2", 1)]
    s = strong_rule(find_reversals(_ci(rows)))
    assert s[(s.Kind == "Q")].S_Size.max() == 6 and s.Meets_Strong_Threshold.any()
