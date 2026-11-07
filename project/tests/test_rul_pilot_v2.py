from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from rul_pilot_v2.evaluation import _better, _c_parts, counts_matrix, find_reversals, strong_rule


class _Arr:
    def __init__(self, lo, hi, pa, pb):
        n = len(lo)
        self.lo, self.hi = np.c_[lo, lo, lo], np.c_[hi, hi, hi]
        self.elig = np.ones((n, 3), bool)
        self.pred = {"A": np.c_[pa, pa, pa], "B": np.c_[pb, pb, pb]}


def test_c_index_matches_brute_force_and_weights() -> None:
    lo = np.array([10.0, 20.0, 30.0, 25.0, 5.0])
    hi = np.array([10.0, 20.0, np.inf, 28.0, 5.0])  # cell 2 right-censored, cell 3 interval
    pa = np.array([12.0, 18.0, 40.0, 26.0, 7.0])
    arr = _Arr(lo, hi, pa, pa[::-1])
    k, na, _ = _c_parts(arr, 0, "A", "B")
    num = den = 0.0
    for i, j in permutations(range(5), 2):
        if hi[i] < lo[j]:
            den += 1
            num += 1.0 if pa[i] < pa[j] else 0.5 if pa[i] == pa[j] else 0.0
    assert k.sum() == den and np.isclose(na.sum() / k.sum(), num / den)
    w = counts_matrix(np.array([[0, 0, 1, 2, 3]]), 5)  # cell 0 twice: its copies form no pair with each other
    assert w[0].tolist() == [2, 1, 1, 1, 0]
    assert np.isclose(((w @ k) * w).sum(), sum(w[0, i] * w[0, j] * k[i, j] for i in range(5) for j in range(5)))


def test_better_method_respects_metric_direction() -> None:
    lower = {"Estimable": True, "CI95_Lower": -2.0, "CI95_Upper": -0.5}
    assert _better(lower, "A", "B", higher_is_better=False) == "A"  # error: A - B < 0 -> A better
    assert _better(lower, "A", "B", higher_is_better=True) == "B"   # C index: A - B < 0 -> B better
    assert _better({"Estimable": True, "CI95_Lower": -1.0, "CI95_Upper": 1.0}, "A", "B", False) == ""


def _ci(rows):
    return pd.DataFrame([{"Dataset": "D", "Theta": t, "Qref_Definition": q, "Eval_Set": e, "Pair": "RF-WAFT", "Better": b}
                         for t, q, e, b in rows])


def test_strong_rule_compares_which_method_is_better_not_signs() -> None:
    thetas = [round(0.80 + 0.01 * i, 2) for i in range(6)]
    rows = []
    for t in thetas:
        rows += [(t, "nominal", "E1", "RF"), (t, "first_comparable", "E1", "WAFT"),
                 (t, "nominal", "E3", "RF"), (t, "first_comparable", "E3", "WAFT")]
    s = strong_rule(find_reversals(_ci(rows)))
    assert s.S_Size.max() == 6 and s.Meets_Strong_Threshold.any()
    flipped = [(t, q, e, ("WAFT" if b == "RF" else "RF") if e == "E3" else b) for t, q, e, b in rows]
    s2 = strong_rule(find_reversals(_ci(flipped)))
    assert s2.S_Size.max() == 0  # opposite conclusion on E3 does not confirm the E1 reversal
