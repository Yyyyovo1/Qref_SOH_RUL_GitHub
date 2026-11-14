"""v4 pipeline on synthetic inputs (data loaders replaced): the development run writes only whitelisted files and never
reaches an evaluation function; two formal runs are byte-identical and produce H1-H3."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import _v4_synthetic as SYN
import test_v4_exp3 as T3
from rul_pilot_v2.evaluation import baseline_check
from rul_v4_core_claim import blinding
from rul_v4_core_claim import exp1 as E1
from rul_v4_core_claim import exp1_models as M
from rul_v4_core_claim import exp3 as E3
from rul_v4_core_claim import pipeline as P


@pytest.fixture
def synthetic_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "INNER_REPEATS", 2)
    monkeypatch.setattr(M, "ENET_N_ALPHAS", 10)
    monkeypatch.setattr(P, "INPUT_FILES", ())
    monkeypatch.setattr(P, "check_archived", lambda repo: pd.DataFrame({"Path": ["x"], "Match": [True]}))
    monkeypatch.setitem(E3.GRID, "SYN", (0.85, 0.90))
    monkeypatch.setitem(E3.EXP3_LABEL, "SYN", "SYN")

    def prep1(root, thetas, checks, log):
        inp, labels, sh, win = SYN.exp1_inputs()
        fits = E1.fit_p_settings(inp, log=log)
        E1.fit_delta_settings(inp, fits)
        for c in ("0_FEATURES_USE_CYCLES_1_TO_100", "1_LABELS", "2_PHYSICAL_EQUIVALENCE", "3_MONOTONICITY", "4_NO_LEAKAGE_EXP1",
                  "WINDOWS_EQUAL_PREP", "S_SEVERSON_FEATURES"):
            P._check(checks, "1", "MIT", c, True, {})
        return {"inp": inp, "fits": fits, "labels": labels, "shares": sh, "windows": win, "sets": inp.sets,
                "feat": pd.DataFrame({"Cell_ID": inp.cells}), "check0": pd.DataFrame({"Cell_ID": inp.cells, "Pass": True}),
                "check_s": {"features": pd.DataFrame({"Item": ["x"], "Mismatches": [0]}), "qdlin": pd.DataFrame({"Cell_ID": ["c"]}),
                            "cycle_life": pd.DataFrame({"Cell_ID": ["c"]})},
                "tol": pd.DataFrame({"Cell_ID": ["c"], "EOL_Difference_Tolerance_Minus_Strict": [0.0]}),
                "estimability": E1.estimability_table(fits), "audit": E1.audit_table(fits)}

    def prep3(root, datasets, thetas, checks, log):
        ds = T3._synthetic_dataset()
        ds.det = pd.DataFrame({"Cell_ID": ds.cells})
        ds.features = pd.DataFrame({"Cell_ID": ds.cells})
        ds.estimability = pd.DataFrame({"Method": ["RF"], "Estimable": [True]})
        ds.audit = pd.DataFrame({"Method": ["RF"], "Train_Folds_Clean": [True]})
        ds.traj_eq = pd.DataFrame({"Max_Rel_Diff": [0.0]})
        ds.checks = {"0": True, "1": True, "2": True, "3": True, "4_amended": True, "4_original": True, "detail": {}}
        win = pd.DataFrame([{"Dataset": "SYN", "Pair": p, "Setting_1": a, "Setting_2": b, "Theta": t, "In_Common_Window": t == 0.90}
                            for (a, b), p in zip(E3.QREF_PAIRS, ("nominal-first", "nominal-first5", "first-first5")) for t in (0.85, 0.90)])
        return {"datasets": {"SYN": ds}, "windows": win}

    monkeypatch.setattr(P, "prepare_exp1", prep1)
    monkeypatch.setattr(P, "prepare_exp3", prep3)
    # frozen check-5 table for the synthetic dataset (what "v2" would have reported)
    root = tmp_path / "project"
    (root / "outputs" / "rul_ranking_pilot_v2").mkdir(parents=True)
    with blinding.allow_evaluation():
        ci, *_ = E3.evaluate_dataset(T3._synthetic_dataset(), [0.85, 0.90], list(E3.CHECK5_PAIRS))
    detail, _ = baseline_check(pd.DataFrame(ci))
    detail.to_csv(root / E3.V2_CHECK5, index=False, encoding="utf-8-sig")
    monkeypatch.setitem(E3.CHECK5_EXPECTED, "SYN", (int(detail.Best_Clearly_Better.sum()), len(detail)))

    return root


def test_dev_run_is_blind(synthetic_pipeline, tmp_path) -> None:
    root = synthetic_pipeline
    out = P.run(root, tmp_path / "dev", dev=True)
    names = {p.name for p in out.iterdir()}
    assert {n for n in names - {"processing_log.txt", "run_info.json"} if not n.startswith("exp3_features_")} <= P.DEV_FILES
    assert "decision.json" not in names and not any("pair_ci" in n or "predictions" in n for n in names)
    for p in out.glob("*.csv"):
        assert blinding.forbidden_columns(pd.read_csv(p, encoding="utf-8-sig", nrows=0).columns) == [], p.name
    log = (out / "processing_log.txt").read_text(encoding="utf-8")
    assert "evaluated" not in log and "verdict" not in log
    assert not blinding.evaluation_allowed()


def test_formal_runs_are_deterministic(synthetic_pipeline, tmp_path) -> None:
    root = synthetic_pipeline
    a = P.run(root, tmp_path / "formal_1", dev=False)
    b = P.run(root, tmp_path / "formal_2", dev=False)
    cmp = P.compare_outputs(a, b)
    assert cmp.Identical.all(), cmp[~cmp.Identical]
    dec = json.loads((a / "decision.json").read_text(encoding="utf-8"))
    assert {"H1", "H2", "H3"} <= set(dec) and all("Verdict" in dec[k] for k in ("H1", "H2", "H3"))
    assert dec["check5"]["SYN"]["Reproduced"] is True
    for f in ("exp1_pair_ci.csv", "exp1_h1_units.csv", "exp3_comparisons.csv", "exp2_equivalence_and_power.csv",
              "exp2_supported_agreement.csv", "exp3_check5_baseline.csv"):
        assert (a / f).exists(), f


def test_formal_run_stops_when_check5_not_reproduced(synthetic_pipeline, tmp_path, monkeypatch) -> None:
    root = synthetic_pipeline
    monkeypatch.setitem(E3.CHECK5_EXPECTED, "SYN", (0, 1))
    out = P.run(root, tmp_path / "formal_bad", dev=False)
    dec = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    assert dec["stopped"] == "CHECK_5_NOT_REPRODUCED" and "H1" not in dec
    assert not (out / "exp3_comparisons.csv").exists() and not (out / "exp1_pair_ci.csv").exists()


def test_subsets_refused_in_formal_runs(tmp_path) -> None:
    from rul_ranking_pilot.labels import PilotError

    with pytest.raises(PilotError):
        P.run(tmp_path, tmp_path / "x", dev=False, experiments=("1",))
