"""v4 PROTOCOL section 7 step 6: v2v3_agreement (supported agreement on the v2 and v3 results; synthetic data only)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rul_ranking_pilot.labels import PilotError
from rul_ranking_pilot.pipeline import _write
from rul_v4_core_claim import v2v3_agreement as A

QREFS = ("nominal", "first_comparable", "early_median5_comparable")
QPAIRS = (("nominal", "first_comparable"), ("nominal", "early_median5_comparable"),
          ("first_comparable", "early_median5_comparable"))
SHORT = {"nominal": "nominal", "first_comparable": "first", "early_median5_comparable": "first5"}
VERDICTS = {"H1": {"Verdict": "UNDECIDED"}, "H2": {"Verdict": "REFUTED"}, "H3": {"Verdict": "HOLDS"}}


def _write_inputs(root: Path, v2_settings: pd.DataFrame, v3_comp: pd.DataFrame) -> pd.DataFrame:
    """v2: pair_ci.csv (one row per setting) and reversals.csv (one row per comparison, without estimability, as
    frozen in v2); v3: comparisons.csv (already carries Estimable_Comparison and Qref_Pair). Written like the frozen
    files (utf-8-sig, %.17g). Returns the v2 reversal table (the expected row order)."""
    (root / A.V2_DIR).mkdir(parents=True)
    (root / A.V3_DIR).mkdir(parents=True)
    ci = v2_settings.assign(Metric="ABS_ERROR", Cells=10, Reason="")[
        ["Dataset", "Theta", "Qref_Definition", "Pair", "Eval_Set", "Metric", "Cells", "Estimable", "Reason", "Better"]]
    _write(ci, root / A.V2_DIR / "pair_ci.csv")
    s = v2_settings.set_index(["Dataset", "Theta", "Qref_Definition", "Eval_Set", "Pair"])
    rows = []
    for (d, t, e, p), _ in v2_settings.groupby(["Dataset", "Theta", "Eval_Set", "Pair"], sort=True):
        for q1, q2 in QPAIRS:
            r1, r2 = s.loc[(d, t, q1, e, p)], s.loc[(d, t, q2, e, p)]
            b1, b2 = r1.Better, r2.Better
            rows.append({"Dataset": d, "Theta": t, "Eval_Set": e, "Pair": p, "Setting_1": q1, "Setting_2": q2,
                         "Better_1": b1, "Better_2": b2,
                         "Supported_Reversal": bool(r1.Estimable and r2.Estimable and b1 and b2 and b1 != b2)})
    rev = pd.DataFrame(rows)
    _write(rev, root / A.V2_DIR / "reversals.csv")
    _write(v3_comp, root / A.V3_DIR / "comparisons.csv")
    return rev


def _decision(root: Path, content: dict) -> Path:
    p = root / "decision.json"
    p.write_text(json.dumps(content), encoding="utf-8")
    return p


def _explicit(root: Path) -> None:
    """Hand-built case with known shares.
    v2 HUST 0.80 E1 RF-BASE: nominal (est, RF), first (est, RF), first5 (est, undetermined)
        -> nominal-first agree; nominal-first5 and first-first5 estimable but not both determined.
    v2 MIT 0.80 E1 RF-BASE: nominal (not est), first (est, BASE), first5 (est, RF)
        -> nominal-* not estimable; first-first5 both determined, different.
    v2 total: estimable 4, both determined 2, agree 1 -> 0.25 and 0.5.
    v3 NCA: three rows (est, RF, RF), (est, RF, ""), (not est, RF, RF) -> estimable 2, both 1, agree 1 -> 0.5 and 1.0."""
    v2 = pd.DataFrame([
        ("HUST", 0.8, "nominal", "RF-BASE", "E1", True, "RF"),
        ("HUST", 0.8, "first_comparable", "RF-BASE", "E1", True, "RF"),
        ("HUST", 0.8, "early_median5_comparable", "RF-BASE", "E1", True, ""),
        ("MIT", 0.8, "nominal", "RF-BASE", "E1", False, ""),
        ("MIT", 0.8, "first_comparable", "RF-BASE", "E1", True, "BASE"),
        ("MIT", 0.8, "early_median5_comparable", "RF-BASE", "E1", True, "RF"),
    ], columns=["Dataset", "Theta", "Qref_Definition", "Pair", "Eval_Set", "Estimable", "Better"])
    v3 = pd.DataFrame([
        ("NCA", 0.6, "E2", "RF-BASE", "nominal", "first_comparable", "RF", "RF", False, True, "nominal-first"),
        ("NCA", 0.6, "E2", "RF-BASE", "nominal", "early_median5_comparable", "RF", "", False, True, "nominal-first5"),
        ("NCA", 0.6, "E2", "RF-BASE", "first_comparable", "early_median5_comparable", "RF", "RF", False, False, "first-first5"),
    ], columns=["Dataset", "Theta", "Eval_Set", "Pair", "Setting_1", "Setting_2", "Better_1", "Better_2",
                "Supported_Reversal", "Estimable_Comparison", "Qref_Pair"])
    _write_inputs(root, v2, v3)


def _random_inputs(root: Path, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    thetas = [round(0.75 + 0.01 * i, 2) for i in range(6)]
    v2 = []
    for d, t, q, p, e in itertools.product(("HUST", "MIT", "XJTU"), thetas, QREFS, ("RF-BASE", "RF-TRAJ"), ("E1", "E2", "E3")):
        est = bool(rng.random() < 0.7)
        better = str(rng.choice(["", p.split("-")[0], p.split("-")[1]])) if est else ""
        v2.append((d, t, q, p, e, est, better))
    v2 = pd.DataFrame(v2, columns=["Dataset", "Theta", "Qref_Definition", "Pair", "Eval_Set", "Estimable", "Better"])
    v3 = []
    for d, t, (q1, q2), p, e in itertools.product(("NCA", "NCM"), [0.6, 0.61, 0.62], QPAIRS, ("RF-GB", "TRAJ-BASE"), ("E1", "E2", "E3")):
        est = bool(rng.random() < 0.6)
        m = p.split("-")
        b1, b2 = (str(rng.choice(["", *m])) for _ in range(2))
        v3.append((d, t, e, p, q1, q2, b1, b2, bool(est and b1 and b2 and b1 != b2), est, f"{SHORT[q1]}-{SHORT[q2]}"))
    v3 = pd.DataFrame(v3, columns=["Dataset", "Theta", "Eval_Set", "Pair", "Setting_1", "Setting_2", "Better_1", "Better_2",
                                   "Supported_Reversal", "Estimable_Comparison", "Qref_Pair"])
    rev = _write_inputs(root, v2, v3)
    return v2, v3, rev


def _expected_shares(est: list[bool], b1: list[str], b2: list[str]) -> tuple[int, int, int]:
    n = nd = ns = 0
    for e, x, y in zip(est, b1, b2):
        if not e:
            continue
        n += 1
        if x and y:
            nd += 1
            ns += int(x == y)
    return n, nd, ns


# ---------------------------------------------------------------- refusal


def test_refuses_without_a_decision_file(tmp_path) -> None:
    _explicit(tmp_path)
    with pytest.raises(FileNotFoundError):
        A.run(tmp_path, tmp_path / "missing" / "decision.json", tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("content", [
    {},                                                                    # empty
    {"stopped": "CHECK_5_NOT_REPRODUCED", "check5": {}},                   # formal run stopped at check 5
    {"stopped": "COMMON_WINDOWS_DIFFER_FROM_PREP", "experiments": ["exp1"]},
    {"H1": {"Verdict": "UNDECIDED"}, "H2": {"Verdict": "REFUTED"}},        # H3 missing
    {"H1": {"Verdict": "UNDECIDED"}, "H2": {"Counts": {}}, "H3": {"Verdict": "HOLDS"}},  # H2 without a verdict
])
def test_refuses_without_h1_to_h3(tmp_path, content) -> None:
    _explicit(tmp_path)
    with pytest.raises(PilotError, match="V4_VERDICTS_NOT_FOUND"):
        A.run(tmp_path, _decision(tmp_path, content), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_refuses_to_overwrite_an_existing_output_directory(tmp_path) -> None:
    _explicit(tmp_path)
    (tmp_path / "out").mkdir()
    with pytest.raises(PilotError, match="OUTPUT_DIRECTORY_EXISTS"):
        A.run(tmp_path, _decision(tmp_path, VERDICTS), tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


# ---------------------------------------------------------------- reading v2 / v3


def test_v2_rows_and_setting_correspondence(tmp_path) -> None:
    v2, _, rev = _random_inputs(tmp_path)
    got = A.v2_comparisons(tmp_path)
    assert len(got) == len(rev) == len(v2)                      # 3 settings -> 3 definition pairs per (d, theta, e, p)
    est = {(r.Dataset, round(r.Theta, 2), r.Qref_Definition, r.Eval_Set, r.Pair): r.Estimable for r in v2.itertuples()}
    for g, r in zip(got.itertuples(), rev.itertuples()):
        assert (g.Dataset, round(g.Theta, 2), g.Eval_Set, g.Pair, g.Setting_1, g.Setting_2) == \
               (r.Dataset, round(r.Theta, 2), r.Eval_Set, r.Pair, r.Setting_1, r.Setting_2)
        k1 = (r.Dataset, round(r.Theta, 2), r.Setting_1, r.Eval_Set, r.Pair)
        k2 = (r.Dataset, round(r.Theta, 2), r.Setting_2, r.Eval_Set, r.Pair)
        assert g.Estimable_Comparison == (est[k1] and est[k2])
        assert g.Qref_Pair == f"{SHORT[r.Setting_1]}-{SHORT[r.Setting_2]}"
        assert g.Source == "v2"
    assert set(got.Qref_Pair) == {"nominal-first", "nominal-first5", "first-first5"}
    # estimability differs between settings often enough for a mix-up of Setting_1 / Setting_2 to be detected
    flipped = [est[(r.Dataset, round(r.Theta, 2), r.Setting_1, r.Eval_Set, r.Pair)] !=
               est[(r.Dataset, round(r.Theta, 2), r.Setting_2, r.Eval_Set, r.Pair)] for r in rev.itertuples()]
    assert sum(flipped) > 50


def test_v3_rows_read_unchanged(tmp_path) -> None:
    _, v3, _ = _random_inputs(tmp_path)
    got = A.v3_comparisons(tmp_path)
    assert len(got) == len(v3)
    assert got.Source.eq("v3").all()
    assert got.Estimable_Comparison.tolist() == v3.Estimable_Comparison.tolist()
    assert got.Qref_Pair.tolist() == v3.Qref_Pair.tolist()
    assert got.Better_1.fillna("").tolist() == v3.Better_1.tolist()
    assert got.Better_2.fillna("").tolist() == v3.Better_2.tolist()


# ---------------------------------------------------------------- shares (PROTOCOL section 3.1)


def test_explicit_shares_both_denominators(tmp_path) -> None:
    _explicit(tmp_path)
    out = A.run(tmp_path, _decision(tmp_path, VERDICTS), tmp_path / "out")
    t = pd.read_csv(out / "v2v3_supported_agreement.csv", encoding="utf-8-sig", float_precision="round_trip")
    src = t[t.Grouping.eq("Source")].set_index("Source")
    assert src.loc["v2", ["Estimable_Comparisons", "Both_Determined", "Supported_Agreement"]].tolist() == [4, 2, 1]
    assert src.loc["v2", "Supported_Agreement_Share"] == pytest.approx(0.25, abs=0)
    assert src.loc["v2", "Agreement_Share_Among_Determined"] == pytest.approx(0.5, abs=0)
    assert src.loc["v3", ["Estimable_Comparisons", "Both_Determined", "Supported_Agreement"]].tolist() == [2, 1, 1]
    assert src.loc["v3", "Supported_Agreement_Share"] == pytest.approx(0.5, abs=0)
    assert src.loc["v3", "Agreement_Share_Among_Determined"] == pytest.approx(1.0, abs=0)


def test_random_shares_match_a_direct_count(tmp_path) -> None:
    v2, v3, rev = _random_inputs(tmp_path)
    out = A.run(tmp_path, _decision(tmp_path, VERDICTS), tmp_path / "out")
    t = pd.read_csv(out / "v2v3_supported_agreement.csv", encoding="utf-8-sig", float_precision="round_trip")
    est = {(r.Dataset, round(r.Theta, 2), r.Qref_Definition, r.Eval_Set, r.Pair): r.Estimable for r in v2.itertuples()}
    v2_est = [est[(r.Dataset, round(r.Theta, 2), r.Setting_1, r.Eval_Set, r.Pair)] and
              est[(r.Dataset, round(r.Theta, 2), r.Setting_2, r.Eval_Set, r.Pair)] for r in rev.itertuples()]
    expected = {"v2": _expected_shares(v2_est, rev.Better_1.tolist(), rev.Better_2.tolist()),
                "v3": _expected_shares(v3.Estimable_Comparison.tolist(), v3.Better_1.tolist(), v3.Better_2.tolist())}
    src = t[t.Grouping.eq("Source")].set_index("Source")
    for s, (n, nd, ns) in expected.items():
        assert n > 0 and nd > ns > 0                              # both denominators and some disagreement exercised
        assert src.loc[s, ["Estimable_Comparisons", "Both_Determined", "Supported_Agreement"]].tolist() == [n, nd, ns]
        assert src.loc[s, "Supported_Agreement_Share"] == ns / n
        assert src.loc[s, "Agreement_Share_Among_Determined"] == ns / nd


def test_group_counts_add_up_to_the_totals(tmp_path) -> None:
    _random_inputs(tmp_path)
    out = A.run(tmp_path, _decision(tmp_path, VERDICTS), tmp_path / "out")
    t = pd.read_csv(out / "v2v3_supported_agreement.csv", encoding="utf-8-sig", float_precision="round_trip")
    assert set(t.Grouping) == {"+".join(g) for g in A.GROUPS}
    tot = t[t.Grouping.eq("Source")].set_index("Source")
    cols = ["Estimable_Comparisons", "Both_Determined", "Supported_Agreement"]
    for g in A.GROUPS:
        part = t[t.Grouping.eq("+".join(g))]
        assert part[g].notna().all().all()
        assert not part.duplicated(subset=g).any()
        sums = part.groupby("Source")[cols].sum()
        assert sums.loc[["v2", "v3"]].values.tolist() == tot.loc[["v2", "v3"], cols].values.tolist(), g
