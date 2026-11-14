"""v4 experiment 1 on synthetic data: inner CV, leakage audit, outer CV, bootstrap of the mean difference, H1 units,
the end-to-end evaluation, determinism, the truncated 100-cycle view, check S, and the dev-run blinding search."""

from __future__ import annotations

import io
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import _v4_synthetic as SYN
from rul_pilot_v2.evaluation import _interval
from rul_ranking_pilot.evaluation import draws_for
from rul_v4_core_claim import blinding
from rul_v4_core_claim import exp1 as E1
from rul_v4_core_claim import exp1_models as M
from rul_v4_core_claim import exp2 as E2
from rul_v4_core_claim import severson as S
from rul_v4_core_claim import verdicts as V
from rul_v4_core_claim.pipeline import Writer, write_exp1_dev_safe

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO / "project"


@pytest.fixture
def fast_enet(monkeypatch):
    """Fewer inner repeats / lambdas so the end-to-end tests stay quick (the selection test uses the real values)."""
    monkeypatch.setattr(M, "INNER_REPEATS", 2)
    monkeypatch.setattr(M, "ENET_N_ALPHAS", 10)


def test_inner_splits_are_repeated_seeded_partitions() -> None:
    s = M.inner_splits(37)
    assert len(s) == 40
    for r in range(10):
        val = np.sort(np.concatenate([s[r * 4 + j][1] for j in range(4)]))
        assert np.array_equal(val, np.arange(37))
        for j in range(4):
            tr, va = s[r * 4 + j]
            assert not set(tr) & set(va) and len(tr) + len(va) == 37
    assert M.splits_sha(s) == M.splits_sha(M.inner_splits(37))
    assert not np.array_equal(s[0][1], s[4][1])                      # different repeats use different seeds


def test_elastic_net_selection_equals_brute_force() -> None:
    """ElasticNetCV over the 40 inner splits picks the (l1_ratio, lambda) with the smallest MSE averaged over splits."""
    from sklearn.linear_model import ElasticNet, ElasticNetCV

    rng = np.random.default_rng(11)
    x = rng.normal(size=(48, 5))
    y_log = 2.8 + 0.3 * x[:, 0] - 0.1 * x[:, 1] + 0.05 * rng.normal(size=48)
    res = M.fit_predict("discharge", x, 10 ** y_log, x[:3])
    tr, _, _, _ = M.standardize(x, x[:3])
    splits = M.inner_splits(48)
    ref = ElasticNetCV(l1_ratio=list(M.ENET_L1_RATIOS), eps=1e-3, alphas=100, cv=splits, n_jobs=1).fit(tr, y_log)
    mse = ref.mse_path_.mean(axis=2)                                   # (l1_ratio, lambda) averaged over the 40 splits
    i, j = np.unravel_index(np.argmin(mse), mse.shape)
    assert res.hyper["Selected_L1_Ratio"] == M.ENET_L1_RATIOS[i] and res.hyper["Selected_Lambda"] == ref.alphas_[i][j]
    # independent recomputation of the chosen point's CV error and of a neighbour with plain ElasticNet fits
    def cv_mse(l1, a):
        e = []
        for tri, vai in splits:
            m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=10000, tol=1e-10).fit(tr[tri], y_log[tri])
            e.append(np.mean((m.predict(tr[vai]) - y_log[vai]) ** 2))
        return float(np.mean(e))
    chosen = cv_mse(M.ENET_L1_RATIOS[i], ref.alphas_[i][j])
    for jj in {max(j - 5, 0), min(j + 5, ref.alphas_.shape[1] - 1)} - {j}:
        assert chosen <= cv_mse(M.ENET_L1_RATIOS[i], ref.alphas_[i][jj]) + 1e-9
    assert np.isclose(chosen, mse[i, j], rtol=1e-3)


def test_standardize_matches_standardscaler() -> None:
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(2)
    x, z = rng.normal(size=(20, 4)), rng.normal(size=(5, 4))
    x[:, 3] = 1.0
    tr, te, mean, scale = M.standardize(x, z)
    sc = StandardScaler().fit(x)
    assert np.allclose(tr, sc.transform(x)) and np.allclose(te, sc.transform(z)) and np.allclose(scale, sc.scale_)


def test_leakage_audit_and_negative_control() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 4))
    y = 10 ** (2.9 + 0.2 * x[:, 0])
    ok = M.fit_predict("full", x[:24], y[:24], x[24:], all_rows=x)
    assert ok.audit["Scaler_Recomputed_Match"] and ok.audit["Inner_Indices_Within_Train"]
    assert ok.audit["Inner_Validation_Partitions_Train"] and ok.audit["Inner_Splits_Reproducible"]
    assert ok.audit["NegControl_Pass"]
    leaky = M.fit_predict("full", x[:24], y[:24], x[24:], all_rows=x[:24])   # "all rows" = training rows: control must fail
    assert not leaky.audit["NegControl_Pass"]


def test_outer_cv_uses_frozen_folds(fast_enet) -> None:
    inp, *_ = SYN.exp1_inputs()
    cells = inp.cells
    res = M.run_outer_cv("full", cells, inp.eol[("strict", "nominal", 0.90)], inp.design["full"], inp.folds)
    assert res.estimable and set(res.predictions) == set(cells)
    assert all(a["Train_Test_Disjoint"] for a in res.audit_rows) and len(res.audit_rows) == 5


def test_mean_pair_boot() -> None:
    rng = np.random.default_rng(1)
    ea, eb = rng.uniform(5, 20, 30), rng.uniform(5, 20, 30)
    ea[[3, 7]] = eb[[3, 7]] = np.nan
    draws, _ = draws_for(30)
    with blinding.allow_evaluation():
        r, vec = E1.mean_pair_boot(ea, eb, draws)
        assert r["Cells"] == 28 and np.isclose(r["Observed_Difference"], np.nanmean(ea) - np.nanmean(eb))
        manual = np.array([np.nanmean(ea[d]) - np.nanmean(eb[d]) for d in draws])
        assert np.allclose(vec, manual, equal_nan=True)
        assert r["CI95_Lower"] == _interval(vec)["CI95_Lower"]
        few = np.full(30, np.nan)
        few[:4] = 1.0
        assert not E1.mean_pair_boot(few, few, draws)[0]["Estimable"]
        r90 = E1.interval90(vec)
        lo, hi = np.quantile(vec[~np.isnan(vec)], [0.05, 0.95])
        assert r90["CI90_Lower"] == lo and r90["CI90_Upper"] == hi


def _formal_exp1(inp):
    fits = E1.fit_p_settings(inp)
    E1.fit_delta_settings(inp, fits)
    draws, _ = draws_for(len(inp.cells))
    with blinding.allow_evaluation():
        ci, vec, apes = E1.evaluate_p(inp, fits, draws)
        rev = E1.reversals(ci)
        units = E1.h1_units(ci)
        base = E1.delta_baselines(inp, fits, apes)
        eq = E2.exp1_equivalence_and_power(rev, ci, vec, base)
        h1 = V.h1(rev, units, True)
        h2 = V.h2(eq, True)
        cp = E1.common_practice(inp, 0.90, ("strict",), ("nominal",))
    return fits, ci, rev, units, base, eq, h1, h2, cp


def test_exp1_end_to_end_synthetic(fast_enet) -> None:
    inp, *_ = SYN.exp1_inputs()
    fits, ci, rev, units, base, eq, h1, h2, cp = _formal_exp1(inp)
    assert len(fits.settings) == 6                                    # 2 rules x 3 definition pairs x 1 theta
    assert len(ci) == 6 * 2 * 3 and len(rev) == 6 * 3
    assert len(units) == 6 * 2 * 2                                    # (rule, pair, Qref, theta) x {C1, C2}
    assert set(base.Baseline_Source) == {"pair_common_cells", "nominal_own_labeled_cells"}
    assert (eq.Method_Pair.isin(["variance-discharge", "variance-full"])).all() and len(eq) == 6 * 2
    assert h1["Verdict"] in {V.HOLDS, V.REFUTED, V.UNDECIDED} and h2["Verdict"] in {V.HOLDS, V.REFUTED, V.UNDECIDED, V.UNTESTABLE}
    ok = ci[ci.Estimable.astype(bool)]
    assert np.allclose(ok.Observed_Difference, ok.Observed_A - ok.Observed_B)
    assert set(cp.Subset) == {"train", "test1", "test2"}
    # determinism of the whole chain
    again = _formal_exp1(inp)
    pd.testing.assert_frame_equal(ci, again[1])
    pd.testing.assert_frame_equal(eq, again[5])


def test_dev_outputs_contain_no_model_quantities(fast_enet, tmp_path) -> None:
    """Amendment 01 section 11: in a development run the inner-CV MSE, selected (alpha, lambda), coefficients and
    predictions appear in no output file, not on stdout / stderr, not in the log."""
    inp, labels, sh, win = SYN.exp1_inputs()
    log_lines: list[str] = []
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        fits = E1.fit_p_settings(inp, log=lambda t: (log_lines.append(t), print(t)))
        E1.fit_delta_settings(inp, fits)
        x = {"feat": pd.DataFrame({"Cell_ID": inp.cells, "DeltaQ_Variance": [inp.design["variance"][c][0] for c in inp.cells]}),
             "check0": pd.DataFrame({"Cell_ID": inp.cells, "Pass": True}),
             "check_s": {"features": pd.DataFrame({"Item": ["x"], "Mismatches": [0]}), "qdlin": pd.DataFrame({"Cell_ID": ["c00"]}),
                         "cycle_life": pd.DataFrame({"Cell_ID": ["c00"]})},
             "labels": labels, "tol": pd.DataFrame({"Cell_ID": ["c00"], "EOL_Difference_Tolerance_Minus_Strict": [0.0]}),
             "shares": sh, "windows": win, "sets": inp.sets, "estimability": E1.estimability_table(fits),
             "audit": E1.audit_table(fits), "fits": fits}
        write_exp1_dev_safe(Writer(tmp_path, dev=True), x)
    text = "\n".join(p.read_text(encoding="utf-8-sig") for p in tmp_path.iterdir()) + out.getvalue() + err.getvalue() + "\n".join(log_lines)
    assert err.getvalue() == ""
    # Recompute, outside the dev path, the quantities that must not leak, and search for them.
    secrets = []
    from sklearn.linear_model import ElasticNetCV

    for (rule, q, theta, model, cells), res in fits.results.items():
        secrets += [v for v in res.predictions.values()]
        for h in res.hyper_rows:
            secrets.append(h["Selected_Lambda"])            # l1_ratio is a grid value (0.5, 0.9 ...) and collides with theta
        if model != "variance" and res.estimable:
            train = [c for c in cells if inp.folds[c] != 1]
            xtr = np.vstack([inp.design[model][c] for c in train])
            tr, _, _, _ = M.standardize(xtr, xtr[:1])
            y = np.log10([inp.eol[(rule, q, theta)][c] for c in train])
            ref = ElasticNetCV(l1_ratio=list(M.ENET_L1_RATIOS), eps=1e-3, alphas=M.ENET_N_ALPHAS, cv=M.inner_splits(len(train)), n_jobs=1).fit(tr, y)
            secrets += [float(ref.mse_path_.mean(axis=2).min()), *ref.coef_.tolist(), float(ref.intercept_)]
    assert len(secrets) > 100
    hits = []
    for v in secrets:
        if not np.isfinite(v) or v == 0:
            continue
        for s in {repr(float(v)), f"{v:.17g}", f"{v:.12g}", f"{v:.8g}"}:
            if len(s.lstrip("-").replace(".", "").lstrip("0")) >= 6 and s in text:
                hits.append(s)
    assert hits == []


# ---------------------------------------------------------------- truncated 100-cycle view and check S
def _write_matlab_like(path: Path, n_cycles: int = 110, n_cells: int = 2, seed: int = 0) -> None:
    """Tiny HDF5 file with the MATLAB v7.3 struct-of-references layout used by the MIT batch files."""
    import h5py

    rng = np.random.default_rng(seed)
    ref_dt = h5py.special_dtype(ref=h5py.Reference)
    with h5py.File(path, "w") as f:
        b = f.create_group("batch")
        summ_refs = np.empty((n_cells, 1), dtype=object)
        cyc_refs = np.empty((n_cells, 1), dtype=object)
        for i in range(n_cells):
            g = f.create_group(f"summary_{i}")
            q = 1.07 - 1e-4 * np.arange(n_cycles) + rng.normal(0, 1e-4, n_cycles)
            for k, v in {"QDischarge": q, "chargetime": 13 + rng.normal(0, 0.1, n_cycles), "IR": 0.016 + rng.normal(0, 1e-4, n_cycles),
                         "Tmax": 35 + rng.normal(0, 0.5, n_cycles), "Tmin": 29 + rng.normal(0, 0.5, n_cycles)}.items():
                g.create_dataset(k, data=v.reshape(1, -1))
            summ_refs[i, 0] = g.ref
            cg = f.create_group(f"cycles_{i}")
            for key, maker in {"Qdlin": lambda j: np.linspace(1.05, 0, 1000) * (1 - 2e-4 * j) + rng.normal(0, 1e-5, 1000),
                               "t": lambda j: np.linspace(0, 40, 50), "T": lambda j: 30 + rng.normal(0, 0.3, 50)}.items():
                refs = np.empty((n_cycles, 1), dtype=object)
                for j in range(n_cycles):
                    refs[j, 0] = f.create_dataset(f"{key}_{i}_{j}", data=maker(j).reshape(-1, 1)).ref
                cg.create_dataset(key, data=refs, dtype=ref_dt)
            cyc_refs[i, 0] = cg.ref
        b.create_dataset("summary", data=summ_refs, dtype=ref_dt)
        b.create_dataset("cycles", data=cyc_refs, dtype=ref_dt)


def test_truncated_view_and_feature_recompute(tmp_path) -> None:
    import h5py

    prep = S.load_prep_module(PROJECT)
    _write_matlab_like(tmp_path / "syn.mat")
    inv = pd.DataFrame({"Source_Key": ["b1c0", "b1c1"], "Batch": [1, 1], "Batch_Index": [0, 1], "Author_Cycle_Life": [900, 950],
                        "Status": "INCLUDED"})
    feat, arrays, check0 = S.compute_features(PROJECT, prep, inventory=inv, raw_dir=tmp_path, files={1: "syn.mat"})
    assert check0.Pass.all() and (check0.Max_Cycle_Row_Accessed == 99).all()
    assert len(feat) == 2 and arrays["b1c0"].shape == (1000, 99) and np.isfinite(feat[S.feature_names(prep)].to_numpy()).all()
    with h5py.File(tmp_path / "syn.mat", "r") as f:
        view = S.TruncatedFile(f)
        cyc = view[f["batch"]["cycles"][0, 0]]
        with pytest.raises(IndexError):
            cyc["Qdlin"][100, 0]                                           # cycle 101 is not visible
        assert view.rows_accessed == [100]
        summ = view[f["batch"]["summary"][0, 0]]
        assert summ["QDischarge"].shape == (1, 100)


def test_check_s_feature_tolerances() -> None:
    ref = pd.DataFrame({"Cell_ID": ["a", "b"], "Batch": [1, 1], "Severson_Split": ["train", "test1"], "Split_Position": [1, 1],
                        "Author_Cycle_Life": [500, 600], "F": [1.0, 0.0]})
    good = ref.assign(F=[1.0 + 5e-12, 1e-13])
    _, ok = S.check_s_features(good, ref, ["F"])
    assert ok
    for bad in (ref.assign(F=[1.0 + 2e-11, 0.0]), ref.assign(F=[1.0, 2e-12]), ref.assign(Author_Cycle_Life=[500, 601])):
        assert not S.check_s_features(bad, ref, ["F"])[1]


def test_round5_and_split_rule() -> None:
    assert S.round5(np.array([1.0706892, -0.000460034, 123456.7]).reshape(1, 3)).tolist() == [[1.0707, -0.00046003, 123460.0]]
    order = [f"b1c{i}" for i in range(41)] + [f"b2c{i}" for i in range(43)] + [f"b3c{i}" for i in range(40)]
    sp = S.severson_split(order)
    sizes = pd.Series([v[0] for v in sp.values()]).value_counts().to_dict()
    assert sizes == {"test1": 43, "train": 41, "test2": 40} and sp["b1c0"] == ("test1", 1) and sp["b2c42"][0] == "test1"
