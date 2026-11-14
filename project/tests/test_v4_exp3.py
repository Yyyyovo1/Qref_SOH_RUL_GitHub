"""v4 experiment 3 on synthetic data: GB trained like RF, the rewritten v3 replacement negative control, the
vector-returning interval equals v2, evaluation / comparisons / equivalence end to end, and the check-5 reproduction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import _v4_synthetic as SYN
from rul_pilot_v2.evaluation import error_pair_ci, per_cell_errors
from rul_pilot_v2.methods import run_setting
from rul_ranking_pilot.evaluation import draws_for
from rul_ranking_pilot.methods import _fold_preprocess, _sha
from rul_v4_core_claim import blinding
from rul_v4_core_claim import exp2 as E2
from rul_v4_core_claim import exp3 as E3
from rul_v4_core_claim import verdicts as V
from rul_v4_core_claim.config import GB_PARAMS


def test_gb_uses_rf_training_rows_and_rule() -> None:
    s, x, cap, thr, t0 = SYN.rul_samples(exact_every=2)
    res = run_setting(s, x, cap, thr, t0)
    pred, ok, reason, audit = E3.gb_fold_predictions(s, x, GB_PARAMS)
    rf = [a for a in res.audit if a["Method"] == "RF"]
    assert ok == res.estimable["RF"] and len(audit) == len(rf) == 5
    for a, b in zip(audit, rf):
        assert a["Train_Rows"] == b["Train_Rows"] and a["Imputer_Median_SHA256"] == b["Imputer_Median_SHA256"]
    assert np.isfinite(pred).all()
    # same estimability rule as RF: fewer than 5 exact training cells in a fold -> not estimable
    s2 = s.copy()
    s2.loc[~s2.Cell_ID.isin(sorted(s2.Cell_ID.unique())[:4]), "Status"] = "RIGHT"
    p2, ok2, reason2, _ = E3.gb_fold_predictions(s2, x, GB_PARAMS)
    r2 = run_setting(s2, x, cap, thr, t0)
    assert not ok2 and not r2.estimable["RF"] and np.isnan(p2).all() and reason2.startswith("FOLD_")


def _v3_inline(x, rows, recorded):
    """Copy of the inline logic in rul_pilot_v3_tju.pipeline.run (amendment 01), used as the reference."""
    all_med = np.where(np.isnan(np.nanmedian(x, axis=0)), 0.0, np.nanmedian(x, axis=0))
    all_mean = np.where(np.isnan(np.nanmean(x, axis=0)), 0.0, np.nanmean(x, axis=0))
    med = np.nanmedian(x[rows], axis=0) if len(rows) else np.full(x.shape[1], np.nan)
    med = np.where(np.isnan(med), 0.0, med)
    mean = np.nanmean(x[rows], axis=0) if len(rows) else np.full(x.shape[1], np.nan)
    mean = np.where(np.isnan(mean), 0.0, mean)
    neg_match = _sha(all_med) == recorded
    mean_differs = _sha(mean) != _sha(all_mean)
    return {"Imputer_Recomputed_Match": _sha(med) == recorded, "NegControl_All_Rows_Match": neg_match,
            "Replacement_Mean_Control_Differs": mean_differs if neg_match else np.nan,
            "NegControl_Pass_Original": not neg_match, "NegControl_Pass_Amended": (not neg_match) or mean_differs}


@pytest.mark.parametrize("case", ["medians_differ", "medians_equal_means_differ", "medians_and_means_equal"])
def test_negative_control_matches_v3(case) -> None:
    if case == "medians_differ":
        x = np.arange(20, dtype=float).reshape(10, 2)
        rows = list(range(6))
    elif case == "medians_equal_means_differ":
        x = np.array([[0.0], [5.0], [5.0], [5.0], [100.0]])
        rows = [0, 1, 2, 3]                                   # median 5 both ways, means differ
    else:
        x = np.array([[5.0], [5.0], [5.0], [5.0]])
        rows = [0, 1, 2]
    _, _, stats = _fold_preprocess(x[rows], x[rows], scale=False)
    got = E3.negative_control(x, rows, stats["Imputer_Median_SHA256"])
    ref = _v3_inline(x, rows, stats["Imputer_Median_SHA256"])
    assert got.keys() == ref.keys()
    for k in got:
        assert (got[k] == ref[k]) or (pd.isna(got[k]) and pd.isna(ref[k]))
    assert got["NegControl_Pass_Amended"] == (case != "medians_and_means_equal")


def test_interval_with_vector_equals_v2() -> None:
    rng = np.random.default_rng(4)
    ea, eb = rng.uniform(0, 100, 40), rng.uniform(0, 100, 40)
    ea[[1, 5, 9]] = eb[[1, 5, 9]] = np.nan
    draws, _ = draws_for(40)
    ref = error_pair_ci(ea, eb, draws)
    with blinding.allow_evaluation():
        got, vec = E3.error_pair_ci_with_draws(ea, eb, draws)
    assert got == ref and vec.shape == (5000,)


def _synthetic_dataset(n_cells: int = 30, thetas=(0.85, 0.90)) -> E3.Exp3Dataset:
    """Samples / predictions for one dataset over 2 thetas x 3 Qrefs (predictions are noise around the truth)."""
    rng = np.random.default_rng(9)
    cells = [f"s{i:02d}" for i in range(n_cells)]
    srows, prows = [], []
    for theta in thetas:
        for q in SYN.QREFS:
            for i, c in enumerate(cells):
                status = "EXACT" if i % 6 else "RIGHT"
                for k in range(3):
                    rul = 400.0 + 20 * i - 100 * k - 300 * (theta - 0.85)
                    srows.append({"Dataset": "SYN", "Theta": theta, "Qref_Definition": q, "Cell_ID": c, "t_Index": k, "Eligible": True,
                                  "Status": status, "RUL": rul if status == "EXACT" else np.nan, "RUL_Lower": rul,
                                  "RUL_Upper": rul if status == "EXACT" else np.inf, "Fold_ID": i % 5 + 1})
                    prows.append({"Dataset": "SYN", "Theta": theta, "Qref_Definition": q, "Cell_ID": c, "t_Index": k,
                                  **{f"Pred_{m}": rul + rng.normal(0, s) for m, s in (("RF", 30), ("TRAJ", 200), ("WAFT", 80), ("GB", 35))},
                                  "Pred_BASE": 380.0 - 100 * k})
    samples = pd.DataFrame(srows)
    return E3.Exp3Dataset("SYN", cells, (50, 100, 150), None, samples, None, pd.DataFrame(prows))


def test_exp3_evaluation_end_to_end() -> None:
    ds = _synthetic_dataset()
    with blinding.allow_evaluation():
        ci, e1t, sec, vec, base, _ = E3.evaluate_dataset(ds, [0.85, 0.90], E3.PAIRS)
        ci = pd.DataFrame(ci)
        assert len(ci) == 2 * 3 * len(E3.PAIRS) * 3 and set(ci.Pair) == set(E3.PAIRS)
        # rows for v2 pairs equal what v2's error_pair_ci gives on the same arrays
        arr = E3._arrays(ds, 0.85, "nominal")
        from rul_pilot_v2.evaluation import membership

        m_e2 = membership(ds.samples[ds.samples.Theta == 0.85], ds.cells, "nominal", "E2")
        ref = error_pair_ci(*per_cell_errors(arr, m_e2, np.ones((len(ds.cells), 3), bool), "RF", "BASE"), draws_for(len(ds.cells))[0])
        row = ci[(ci.Theta == 0.85) & (ci.Qref_Definition == "nominal") & (ci.Eval_Set == "E2") & (ci.Pair == "RF-BASE")].iloc[0]
        assert row.CI95_Lower == ref["CI95_Lower"] and row.Observed_A == ref["Observed_A"]
        win = pd.DataFrame([{"Dataset": "SYN", "Pair": f"{a}-{b}", "Setting_1": qa, "Setting_2": qb, "Theta": t, "In_Common_Window": t == 0.90}
                            for (qa, qb), (a, b) in zip(E3.QREF_PAIRS, [("nominal", "first"), ("nominal", "first5"), ("first", "first5")])
                            for t in (0.85, 0.90)])
        comp = E3.comparison_table(ci, win)
        assert set(comp[comp.In_Common_Window].Theta) == {0.90} and comp.Decisive_H3.sum() == 2 * 3 * 3
        vec5 = {(k[0], k[1], k[2], k[3], "SYN"): v for k, v in vec.items()}
        eq = E2.exp3_equivalence_and_power(comp, vec5, pd.DataFrame(base))
        assert len(eq) == 3 * 3 and set(eq[eq.Eval_Set.eq("E1")].Category) <= {"EQUIVALENT", "NOT_EQUIVALENT", "IMPRECISE", "NOT_ESTIMABLE", "DELTA_UNDEFINED"}
        assert eq[eq.Eval_Set.eq("E3")].Power_Fraction_Levels.eq("NOT_APPLICABLE_C_INDEX").all()
        h3 = V.h3(comp, eq, ["SYN"])
        assert h3["Verdict"] in {V.HOLDS, V.REFUTED} and h3["Estimable_Comparisons"] <= 9
        cp = E3.common_practice(ci.assign(Theta=ci.Theta.replace({0.85: 0.80})))
        assert len(cp) and "Apparent_Ranking_Change" in cp


def test_check5_reproduction(tmp_path, monkeypatch) -> None:
    ds = _synthetic_dataset()
    with blinding.allow_evaluation():
        ci, *_ = E3.evaluate_dataset(ds, [0.85, 0.90], list(E3.CHECK5_PAIRS))
        ci = pd.DataFrame(ci)
        from rul_pilot_v2.evaluation import baseline_check

        detail, _ = baseline_check(ci)
        (tmp_path / "outputs" / "rul_ranking_pilot_v2").mkdir(parents=True)
        detail.to_csv(tmp_path / E3.V2_CHECK5, index=False, encoding="utf-8-sig")
        monkeypatch.setitem(E3.CHECK5_EXPECTED, "SYN", (int(detail.Best_Clearly_Better.sum()), len(detail)))
        _, rep = E3.check5("SYN", ci, tmp_path)
        assert rep["Reproduced"] and rep["rows_equal_frozen"] and rep["counts_equal"]
        # GB-BASE must not enter the choice of the best method
        _, rep_gb = E3.check5("SYN", pd.concat([ci, ci[ci.Pair.eq("RF-BASE")].assign(Pair="GB-BASE", Observed_A=-1.0)]), tmp_path)
        assert rep_gb["Reproduced"]
        bad = detail.copy()
        bad.loc[bad.index[0], "Best_Method"] = "WAFT" if bad.Best_Method.iloc[0] != "WAFT" else "RF"
        bad.to_csv(tmp_path / E3.V2_CHECK5, index=False, encoding="utf-8-sig")
        assert not E3.check5("SYN", ci, tmp_path)[1]["Reproduced"]
