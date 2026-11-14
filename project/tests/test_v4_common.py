"""v4: archive check, blinding guard, development-run writer, equivalence / detection / verdict rules (synthetic only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rul_v4_core_claim import blinding
from rul_v4_core_claim import exp2 as E2
from rul_v4_core_claim import verdicts as V
from rul_v4_core_claim.config import ARCHIVED_SHA256
from rul_v4_core_claim.pipeline import DEV_FILES, Writer
from rul_v4_core_claim.prereg import record_hashes
from rul_ranking_pilot.labels import PilotError

REPO = Path(__file__).resolve().parents[2]


def test_config_hashes_equal_the_archive_records() -> None:
    assert record_hashes(REPO) == ARCHIVED_SHA256
    assert len(ARCHIVED_SHA256) == 17


def test_evaluation_functions_raise_while_blinded() -> None:
    assert not blinding.evaluation_allowed()
    with pytest.raises(blinding.BlindingError):
        E2.classify(-1.0, 1.0, 2.0)
    with pytest.raises(blinding.BlindingError):
        V.h3(pd.DataFrame(), pd.DataFrame(), [])
    with blinding.allow_evaluation():
        assert E2.classify(-1.0, 1.0, 2.0) == "EQUIVALENT"
    assert not blinding.evaluation_allowed()


def test_column_screen() -> None:
    bad = blinding.forbidden_columns(["Pred_RF", "CI95_Lower", "Observed_Difference", "Better", "Selected_Lambda",
                                      "Selected_L1_Ratio", "Mean_APE", "RMSE_Cycles", "Supported_Reversal"])
    assert len(bad) == 9
    ok = ["Cell_ID", "Theta", "Max_Rel_Diff", "CV_Median_Current_A__level", "Prediction_Time_Raw_Cycle", "DeltaQ_Mean",
          "Imputer_Median_SHA256", "Replacement_Mean_Control_Differs", "Estimable", "Fit_Finite", "TRAJ_Fallback_Fraction",
          "Labeled_EOL_gt_100", "Inner_Splits_SHA256", "NegControl_Pass", "Recomputed_Max_Rel_Diff"]
    assert blinding.forbidden_columns(ok) == []


def test_dev_writer_whitelist(tmp_path) -> None:
    w = Writer(tmp_path, dev=True)
    w.csv(pd.DataFrame({"Cell_ID": ["a"], "Estimable": [True]}), "exp1_estimability.csv")
    with pytest.raises(PilotError):
        w.csv(pd.DataFrame({"Cell_ID": ["a"]}), "exp1_pair_ci.csv")
    with pytest.raises(blinding.BlindingError):
        w.csv(pd.DataFrame({"Pred_GB": [1.0]}), "exp3_samples.csv")
    with pytest.raises(PilotError):
        w.json({"H1": 1}, "decision.json")
    assert "decision.json" not in DEV_FILES and not any("pair_ci" in f or "predictions" in f for f in DEV_FILES)


def test_classify_boundaries() -> None:
    with blinding.allow_evaluation():
        assert E2.classify(-0.99, 0.99, 1.0) == "EQUIVALENT"
        assert E2.classify(-1.0, 0.5, 1.0) == "IMPRECISE"        # touching -delta is not inside (-delta, delta)
        assert E2.classify(1.0, 2.0, 1.0) == "IMPRECISE"         # touching [-delta, delta] intersects
        assert E2.classify(1.0001, 2.0, 1.0) == "NOT_EQUIVALENT"
        assert E2.classify(-3.0, -1.0001, 1.0) == "NOT_EQUIVALENT"
        assert E2.classify(-3.0, 3.0, 1.0) == "IMPRECISE"


def test_equivalence_category_order() -> None:
    v = np.linspace(-1, 1, 5000)
    with blinding.allow_evaluation():
        assert E2.equivalence_row(v, v, False, False, np.nan)["Category"] == "DELTA_UNDEFINED"
        assert E2.equivalence_row(v, v, False, True, 10.0)["Category"] == "NOT_ESTIMABLE"
        nanv = v.copy()
        nanv[:300] = np.nan                                       # 6% missing -> not estimable
        assert E2.equivalence_row(v, nanv, True, True, 10.0)["Category"] == "NOT_ESTIMABLE"
        r = E2.equivalence_row(np.zeros(5000), np.full(5000, 0.05), True, True, 10.0)
        assert r["Category"] == "EQUIVALENT" and np.isclose(r["Delta"], 1.0)


def test_detection_probability_constants() -> None:
    with blinding.allow_evaluation():
        r = E2.detection_row(np.r_[0.0, 1.0] * 1, np.r_[0.0, 1.0], True, True, 10.0)
    assert abs(r["P_1SE"] - 0.0284) < 5e-4 and abs(r["P_2SE"] - 0.266) < 1e-3
    se = float(np.std([0.0, 1.0], ddof=1))
    from scipy.stats import norm
    assert np.isclose(r["P_20pct"], norm.cdf(2.0 / se - 1.96) ** 2)
    with blinding.allow_evaluation():
        e3 = E2.detection_row(None, None, True, True, np.nan, metric_is_error=False)
    assert e3["Fraction_Levels"] == "NOT_APPLICABLE_C_INDEX" and "P_20pct" not in e3


def _rev(rows):
    return pd.DataFrame(rows, columns=["Rule", "Pair", "Theta", "Method_Pair", "Estimable_Comparison", "Supported_Reversal"])


def test_h1_rules() -> None:
    units = pd.DataFrame({"Estimable": [True] * 4, "Claim_Supported": [True, True, False, False], "Claim": ["C1", "C2", "C1", "C2"]})
    none = _rev([["strict", "nominal-first", 0.9, "variance-full", True, False]])
    one = _rev([["strict", "nominal-first", 0.9, "variance-full", True, True]])
    two_same = _rev([["strict", "nominal-first", 0.9, "variance-full", True, True], ["strict", "nominal-first", 0.91, "variance-full", True, True]])
    two_rules = _rev([["strict", "nominal-first", 0.9, "variance-full", True, True], ["tolerance", "nominal-first", 0.91, "variance-full", True, True]])
    df_only = _rev([["strict", "nominal-first", 0.9, "discharge-full", True, True], ["strict", "nominal-first", 0.91, "discharge-full", True, True]])
    with blinding.allow_evaluation():
        assert V.h1(none, units, True)["Verdict"] == V.HOLDS                 # 2/4 supported = one half
        assert V.h1(none, units.assign(Claim_Supported=[True, False, False, False]), True)["Verdict"] == V.UNDECIDED
        assert V.h1(one, units, True)["Verdict"] == V.UNDECIDED
        assert V.h1(two_same, units, True)["Verdict"] == V.REFUTED
        assert V.h1(two_rules, units, True)["Verdict"] == V.UNDECIDED        # amendment 01 section 3.2
        assert V.h1(df_only, units, True)["Verdict"] == V.HOLDS              # discharge-full is descriptive only
        assert V.h1(none, units, False, "CHECK_S_FAILED")["Verdict"] == V.UNTESTABLE


def _eq(cats, exp="3", e="E1"):
    return pd.DataFrame({"Experiment": exp, "Eval_Set": e, "Category": cats})


def test_h2_rules() -> None:
    with blinding.allow_evaluation():
        assert V.h2(_eq(["EQUIVALENT"] * 95 + ["NOT_EQUIVALENT"] * 5), True)["Verdict"] == V.HOLDS     # exactly 5%
        assert V.h2(_eq(["EQUIVALENT"] * 94 + ["NOT_EQUIVALENT"] * 6), True)["Verdict"] == V.REFUTED
        assert V.h2(_eq(["EQUIVALENT"] * 50 + ["IMPRECISE"] * 50), True)["Verdict"] == V.HOLDS          # exactly 50%
        assert V.h2(_eq(["EQUIVALENT"] * 49 + ["IMPRECISE"] * 51), True)["Verdict"] == V.UNTESTABLE
        mixed = _eq(["EQUIVALENT"] * 40 + ["IMPRECISE"] * 40 + ["NOT_EQUIVALENT"] * 4 + ["DELTA_UNDEFINED"] * 16)
        r = V.h2(mixed, True)
        assert r["Denominator"] == 84 and r["Verdict"] == V.UNDECIDED          # 40/84 < 50%
        assert V.h2(_eq(["EQUIVALENT"] * 3, exp="1", e="P_common_cells"), False)["Verdict"] == V.UNTESTABLE
        assert V.h2(_eq(["EQUIVALENT"] * 3, e="E2"), True)["Denominator"] == 0


def test_h3_rules() -> None:
    comp = pd.DataFrame({"Pair": "RF-GB", "In_Common_Window": True, "Estimable_Comparison": True, "Dataset": ["HUST", "MIT", "NCA"],
                         "Supported_Reversal": [True, False, True]})
    power = pd.DataFrame({"Experiment": "3", "Dataset": ["HUST", "MIT"], "Eval_Set": "E1", "Power_P_20pct": [0.2, 0.4]})
    with blinding.allow_evaluation():
        assert V.h3(comp, power, ["HUST", "MIT"])["Verdict"] == V.HOLDS
        r = V.h3(comp, power, ["HUST", "MIT", "NCA"])
        assert r["Verdict"] == V.REFUTED and r["Low_Power_Note"] is True
        assert V.h3(comp, power, [])["Verdict"] == V.UNTESTABLE


def test_agreement_shares() -> None:
    comp = pd.DataFrame({"Estimable_Comparison": [True, True, True, False], "Better_1": ["A", "A", "", "A"],
                         "Better_2": ["A", "B", "B", "A"], "G": ["x", "x", "x", "x"]})
    with blinding.allow_evaluation():
        r = E2.agreement(comp, ["G"]).iloc[0]
    assert r.Estimable_Comparisons == 3 and r.Both_Determined == 2 and r.Supported_Agreement == 1
    assert np.isclose(r.Supported_Agreement_Share, 1 / 3) and np.isclose(r.Agreement_Share_Among_Determined, 0.5)
