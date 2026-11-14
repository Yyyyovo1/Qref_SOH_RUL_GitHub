"""Experiment 3 (PROTOCOL_v4 section 4; AMENDMENT_01 sections 6, 9): the v2 pipeline (HUST, MIT) and the v3 TJU data
adapter (NCA), all v2 methods unchanged (v2 `run_setting`) plus GB, checks 0-4 per dataset, check 5 reproduction of
v2/v3, and the evaluation (shared-draw cell bootstrap, E1/E2/E3, reversals within the strict common windows).

Fitting and checks 0-4 are allowed in development runs; evaluation functions are decorated with `@evaluation`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from rul_pilot_v2 import data as v2data
from rul_pilot_v2.config import MARGIN_FEATURE
from rul_pilot_v2.config import NOMINAL_CAPACITY_AH as V2_NOMINAL
from rul_pilot_v2.evaluation import (
    Arrays,
    _better,
    _interval,
    _nanmedian_rows,
    baseline_check,
    c_index_pair_ci,
    counts_matrix,
    draws_for,
    find_reversals,
    membership,
    per_cell_errors,
)
from rul_pilot_v2.methods import run_setting
from rul_pilot_v2.pipeline import _check_mit_labels, _physical_equivalence as v2_physical
from rul_pilot_v3_tju import config as v3config
from rul_pilot_v3_tju import data as v3data
from rul_pilot_v3_tju.pipeline import _check_labels as v3_check_labels
from rul_pilot_v3_tju.pipeline import _physical_equivalence as v3_physical
from rul_ranking_pilot.config import THETAS as V2_THETAS
from rul_ranking_pilot.labels import PilotError, check_frozen_labels, check_monotonicity, compute_t0
from rul_ranking_pilot.labels import detect_all as v1_detect_all
from rul_ranking_pilot.methods import _fold_preprocess, _sha

from .blinding import evaluation
from .config import (
    CHECK5_EXPECTED,
    CHECK5_PAIRS,
    EXACT,
    EXP3_C_THETA,
    EXP3_LABEL,
    EXP3_METHODS,
    EXP3_WINDOWS_REF,
    GB_PARAMS,
    GB_PLAN,
    MIN_CELLS,
    N_FOLDS,
    QREF_PAIRS,
    QREF_SHORT,
    QREFS,
    V2_CHECK5,
    V2_METHODS,
    V3_CHECK5,
    WINDOW_SHARE,
)

PAIRS = [f"{a}-{b}" for a, b in combinations(EXP3_METHODS, 2)]
NOMINAL = {"HUST": V2_NOMINAL["HUST"], "MIT": V2_NOMINAL["MIT"], "NCA": v3config.NOMINAL_CAPACITY_AH["NCA"]}
GRID = {"HUST": tuple(V2_THETAS), "MIT": tuple(V2_THETAS), "NCA": tuple(v3config.THETAS)}


# ================================================================ inputs, labels, samples, features
def load_inputs(root: Path, datasets) -> dict:
    root = Path(root)
    out = {}
    if {"HUST", "MIT"} & set(datasets):
        allv2 = v2data.load_all(root)
        out.update({d: allv2[d] for d in ("HUST", "MIT") if d in datasets})
    if "NCA" in datasets:
        out["NCA"] = v3data.load_tju(root)["NCA"]
    return out


def detections(data: dict, name: str) -> pd.DataFrame:
    if name == "NCA":
        det = v3data.detect_all({name: data[name]}, v3config.THETAS)
    else:
        det = v1_detect_all({name: data[name]})
    det["Theta"] = det.Theta.round(2)
    return det


def frozen_t0(root: Path, name: str) -> tuple[int, int, int]:
    return tuple(v3config.T0["NCA"]) if name == "NCA" else tuple(v2data.load_t0(Path(root))[name])


def base_features(data: dict, name: str, t0s) -> tuple[pd.DataFrame, float]:
    return (v3data if name == "NCA" else v2data).base_features(data[name], t0s)


def gb_params_from_plan(root: Path) -> dict:
    plan = pd.read_csv(Path(root) / GB_PLAN, encoding="utf-8-sig")
    row = plan[plan.apply(lambda r: "GradientBoostingRegressor" in " ".join(map(str, r.values)), axis=1)]
    if len(row) != 1:
        raise PilotError("GB_PLAN_ROW_NOT_FOUND")
    for v in row.iloc[0].values:
        s = str(v)
        if s.startswith("{") and "learning_rate" in s:
            return json.loads(s)
    raise PilotError("GB_PLAN_PARAMS_NOT_FOUND")


# ================================================================ checks 0-3 (per dataset, v2 / v3 definitions)
def checks_0_to_3(root: Path, data: dict, name: str, det: pd.DataFrame, t0) -> dict:
    root = Path(root)
    d = {name: data[name]}
    if name == "NCA":
        re = v3data.compute_t0_1(d, det)
        c0 = tuple(re[name]["t0"]) == tuple(t0)
        c0_detail = {"recomputed": list(re[name]["t0"]), "frozen": list(t0)}
        c1d = v3_check_labels(root, d, det)
        c1 = bool(c1d["per_dataset_pass"].get(name, False))
        phys = v3_physical(d, det)
    else:
        re = compute_t0(d, det).set_index("Dataset").t0_raw_cycle.to_dict()
        c0 = int(re[name]) == int(t0[0])
        c0_detail = {"recomputed_t0_1": int(re[name]), "frozen": list(t0)}
        if name == "HUST":
            fr = check_frozen_labels(root, det)
            fr = fr[fr.Dataset.eq("HUST")]
            c1, c1d = bool(len(fr) and fr.Pass.all()), {"rows": int(len(fr)), "pass_rows": int(fr.Pass.sum())}
        else:
            c1d = _check_mit_labels(root, det)
            c1 = bool(c1d["Pass"])
        phys = v2_physical(d, det)
    mono = check_monotonicity(det)
    return {"0": bool(c0), "1": bool(c1), "2_labels": bool(phys["Pass"]), "3": bool(mono.Pass.iat[0]),
            "detail": {"0": c0_detail, "1": c1d, "2": phys, "3": int(mono.Violations.iat[0])}}


# ================================================================ methods: v2 run_setting + GB
def gb_fold_predictions(samples: pd.DataFrame, x: np.ndarray, params: dict):
    """GB trained exactly like RF in v2 run_setting: exact-cell samples of the training folds, median imputation
    (no scaling), estimable only if every fold has >= 5 exact training cells."""
    from sklearn.ensemble import GradientBoostingRegressor

    cell = samples.Cell_ID.to_numpy(object)
    fold = samples.Fold_ID.to_numpy(int)
    status = samples.Status.to_numpy(object)
    rul = samples.RUL.to_numpy(float)
    pred = np.full(len(samples), np.nan)
    estimable, reason, audit = True, "", []
    for k in range(1, N_FOLDS + 1):
        train, test = fold != k, fold == k
        if set(cell[train]) & set(cell[test]):
            raise RuntimeError("TRAIN_TEST_CELL_OVERLAP")
        if not test.any():
            continue
        exact_tr = train & (status == EXACT)
        n_exact_cells = len(set(cell[exact_tr]))
        if n_exact_cells < MIN_CELLS:
            estimable, reason = False, f"FOLD_{k}_TRAIN_EXACT_CELLS_{n_exact_cells}_LT_{MIN_CELLS}"
        if estimable:
            tr, te, stats = _fold_preprocess(x[exact_tr], x[test], scale=False)
            pred[test] = GradientBoostingRegressor(**params).fit(tr, rul[exact_tr]).predict(te)
            audit.append({"Method": "GB", "Fold_ID": k, "Train_Rows": np.flatnonzero(exact_tr).tolist(), **stats})
    if not estimable:
        pred[:] = np.nan
    return pred, estimable, reason, audit


def negative_control(x: np.ndarray, rows: list[int], recorded_median_sha: str) -> dict:
    """v3 amendment 01 (rewritten as a function): original control on medians; replacement control on means when the
    training-row and all-row medians coincide."""
    all_med = np.where(np.isnan(np.nanmedian(x, axis=0)), 0.0, np.nanmedian(x, axis=0))
    all_mean = np.where(np.isnan(np.nanmean(x, axis=0)), 0.0, np.nanmean(x, axis=0))
    med = np.nanmedian(x[rows], axis=0) if len(rows) else np.full(x.shape[1], np.nan)
    med = np.where(np.isnan(med), 0.0, med)
    mean = np.nanmean(x[rows], axis=0) if len(rows) else np.full(x.shape[1], np.nan)
    mean = np.where(np.isnan(mean), 0.0, mean)
    neg_match = _sha(all_med) == recorded_median_sha
    mean_differs = _sha(mean) != _sha(all_mean)
    return {"Imputer_Recomputed_Match": _sha(med) == recorded_median_sha, "NegControl_All_Rows_Match": neg_match,
            "Replacement_Mean_Control_Differs": mean_differs if neg_match else np.nan,
            "NegControl_Pass_Original": not neg_match, "NegControl_Pass_Amended": (not neg_match) or mean_differs}


@dataclass
class Exp3Dataset:
    name: str
    cells: list[str]
    t0: tuple
    det: pd.DataFrame
    samples: pd.DataFrame
    features: pd.DataFrame
    predictions: pd.DataFrame = None
    estimability: pd.DataFrame = None
    audit: pd.DataFrame = None
    traj_eq: pd.DataFrame = None
    checks: dict = field(default_factory=dict)
    feature_sha: str = ""


def fit_dataset(root: Path, data: dict, name: str, gb_params: dict, thetas=None, log=None) -> Exp3Dataset:
    """Labels, samples, features and all fits for one dataset (v2 / v3 per-setting loop, unchanged, plus GB)."""
    t0 = frozen_t0(root, name)
    det_full = detections(data, name)
    c03 = checks_0_to_3(root, data, name, det_full, t0)
    use = GRID[name] if thetas is None else tuple(t for t in GRID[name] if round(t, 2) in {round(x, 2) for x in thetas})
    det = det_full[det_full.Theta.isin([round(t, 2) for t in use])].reset_index(drop=True)
    d = data[name]
    samples = v2data.build_samples({name: d}, det, {name: t0})
    feats, max_used = base_features(data, name, t0)
    design_cols = [c for c in feats.columns if c not in ("Cell_ID", "t_Index")] + [MARGIN_FEATURE]
    feat_sha = hashlib.sha256(feats.to_csv(index=False).encode()).hexdigest()
    cap_by_cell = {c: g for c, g in d.capacity.groupby("Cell_ID")}
    preds, est, audit, traj_eq, design_ok = [], [], [], [], []
    for theta in use:
        for q in QREFS:
            s = samples[(samples.Theta == theta) & (samples.Qref_Definition == q) & samples.Eligible].reset_index(drop=True)
            qref = d.qref[q]
            xs = s[["Cell_ID", "t_Index"]].merge(feats, on=["Cell_ID", "t_Index"], how="left")
            xs[MARGIN_FEATURE] = xs.Capacity_t_Ah - theta * xs.Cell_ID.map(qref).astype(float)
            x = xs[design_cols].to_numpy(float)
            design_ok.append(bool(xs.drop(columns=[MARGIN_FEATURE]).reset_index(drop=True).equals(
                s[["Cell_ID", "t_Index"]].merge(feats, on=["Cell_ID", "t_Index"], how="left").reset_index(drop=True))))
            thr = {c: theta * float(qref[c]) for c in d.cells}
            res = run_setting(s, x, cap_by_cell, thr, t0)
            gb_pred, gb_ok, gb_reason, gb_audit = gb_fold_predictions(s, x, gb_params)
            p = pd.concat([s[["Dataset", "Theta", "Qref_Definition", "Cell_ID", "t_Index", "Fold_ID", "Status", "RUL",
                              "RUL_Lower", "RUL_Upper"]], res.predictions.drop(columns=["Cell_ID", "t_Index"])], axis=1)
            p.insert(p.columns.get_loc("Pred_BASE"), "Pred_GB", gb_pred)
            preds.append(p)
            ests = {**res.estimable, "GB": gb_ok}
            reasons = {**res.reasons, "GB": gb_reason}
            for m in EXP3_METHODS:
                est.append({"Dataset": name, "Theta": theta, "Qref_Definition": q, "Method": m, "Estimable": ests[m],
                            "Reason": reasons[m], "Samples": len(s),
                            "TRAJ_Fallback_Fraction": res.traj_fallback_fraction if m == "TRAJ" else np.nan,
                            "TRAJ_Missing_Fraction": res.traj_missing_fraction if m == "TRAJ" else np.nan})
            for a in [*res.audit, *gb_audit]:
                rows = a["Train_Rows"]
                audit.append({"Dataset": name, "Theta": theta, "Qref_Definition": q, "Method": a["Method"], "Fold_ID": a["Fold_ID"],
                              "Train_Folds_Clean": bool((s.Fold_ID.to_numpy()[rows] != a["Fold_ID"]).all()),
                              **negative_control(x, rows, a["Imputer_Median_SHA256"]),
                              "Dropped_Constant_Columns": a.get("Dropped_Constant_Columns", np.nan)})
            if q != "nominal":
                nominal = NOMINAL[name]
                alt = {c: (theta * float(qref[c]) / nominal) * nominal for c in d.cells}
                r2 = run_setting(s, x, cap_by_cell, alt, t0, traj_only=True)
                a1, a2 = res.predictions.Pred_TRAJ.to_numpy(float), r2.predictions.Pred_TRAJ.to_numpy(float)
                both = ~np.isnan(a1) & ~np.isnan(a2)
                traj_eq.append({"Dataset": name, "Theta": theta, "Qref_Definition": q,
                                "Estimable_Equal": res.estimable["TRAJ"] == r2.estimable["TRAJ"],
                                "Max_Rel_Diff": float(np.max(np.abs(a1[both] - a2[both]) / np.maximum(1.0, np.abs(a1[both])))) if both.any() else 0.0})
        if log:
            log(f"exp3 {name} theta={theta:.2f} trained")
    out = Exp3Dataset(name, list(d.cells), t0, det, samples, feats, pd.concat(preds, ignore_index=True), pd.DataFrame(est),
                      pd.DataFrame(audit), pd.DataFrame(traj_eq), feature_sha=feat_sha)
    ad, td = out.audit, out.traj_eq
    common = {"fold_cells_disjoint": bool(ad.Train_Folds_Clean.all()), "max_raw_cycle_used_le_t0_3": bool(max_used <= t0[2]),
              "design_has_no_qref_column": not [c for c in design_cols if "qref" in c.lower()],
              "non_margin_design_identical_across_qref": bool(all(design_ok)),
              "fold_statistics_from_training_rows_only": bool(ad.Imputer_Recomputed_Match.all())}
    traj_ok = bool(td.Estimable_Equal.all() and (td.Max_Rel_Diff <= 1e-9).all()) if len(td) else True
    out.checks = {"0": c03["0"], "1": c03["1"], "2": bool(c03["2_labels"] and traj_ok), "3": c03["3"],
                  "4_amended": bool(all(common.values()) and ad.NegControl_Pass_Amended.all()),
                  "4_original": bool(all(common.values()) and ad.NegControl_Pass_Original.all()),
                  "detail": {**c03["detail"], "4": {**common, "negative_control_original_failures": int((~ad.NegControl_Pass_Original).sum()),
                                                    "negative_control_amended_failures": int((~ad.NegControl_Pass_Amended).sum()),
                                                    "folds_using_replacement_mean_control": int(ad.NegControl_All_Rows_Match.sum()),
                                                    "shared_feature_sha256": feat_sha},
                             "2_traj_max_rel_diff_le_1e-9": traj_ok}}
    return out


# ================================================================ strict common windows (protocol section 4)
def strict_shares(det: pd.DataFrame) -> pd.Series:
    return det.assign(ex=det.Status.eq(EXACT)).groupby(["Qref_Definition", "Theta"]).ex.mean()


def windows(name: str, det: pd.DataFrame) -> pd.DataFrame:
    sh = strict_shares(det)
    rows = []
    for q1, q2 in QREF_PAIRS:
        for t in sorted(det.Theta.unique()):
            rows.append({"Dataset": name, "Pair": f"{QREF_SHORT[q1]}-{QREF_SHORT[q2]}", "Setting_1": q1, "Setting_2": q2,
                         "Theta": float(t), "Strict_Share_1": float(sh[(q1, t)]), "Strict_Share_2": float(sh[(q2, t)]),
                         "In_Common_Window": bool(sh[(q1, t)] >= WINDOW_SHARE and sh[(q2, t)] >= WINDOW_SHARE)})
    return pd.DataFrame(rows)


def compare_windows_with_prep(root: Path, win: pd.DataFrame) -> dict:
    ref = pd.read_csv(Path(root) / EXP3_WINDOWS_REF, encoding="utf-8-sig")
    ref["Theta"] = ref.Theta.round(2)
    w = win.assign(Dataset=win.Dataset.map(EXP3_LABEL), Theta=win.Theta.round(2))
    ref = ref[ref.Dataset.isin(w.Dataset.unique())]
    m = w.merge(ref[["Dataset", "Pair", "Theta", "In_Common_Window"]], on=["Dataset", "Pair", "Theta"], suffixes=("", "_prep"),
                how="outer", indicator=True)
    same = bool((m._merge == "both").all() and (m.In_Common_Window.astype(bool) == m.In_Common_Window_prep.astype(bool)).all())
    return {"rows": int(len(m)), "mismatches": int((~((m._merge == "both") & (m.In_Common_Window == m.In_Common_Window_prep))).sum()),
            "Pass": same}


def window_thetas(win: pd.DataFrame) -> set[float]:
    return {round(float(t), 2) for t in win[win.In_Common_Window].Theta}


# ================================================================ evaluation (formal run only)
def _arrays(ds: Exp3Dataset, theta: float, q: str) -> Arrays:
    s_q = ds.samples[(ds.samples.Theta == theta) & (ds.samples.Qref_Definition == q)]
    p_q = ds.predictions[(ds.predictions.Theta == theta) & (ds.predictions.Qref_Definition == q)]
    arr = Arrays(ds.cells, s_q, p_q)
    arr.pred["GB"] = np.full((len(ds.cells), 3), np.nan)
    if len(p_q):
        idx = {c: i for i, c in enumerate(ds.cells)}
        arr.pred["GB"][p_q.Cell_ID.map(idx).to_numpy(int), p_q.t_Index.to_numpy(int)] = p_q.Pred_GB.to_numpy(float)
    return arr


@evaluation
def error_pair_ci_with_draws(ea: np.ndarray, eb: np.ndarray, draws: np.ndarray) -> tuple[dict, np.ndarray]:
    """v2 `error_pair_ci`, line for line, also returning the bootstrap vector of the median difference."""
    n = int((~np.isnan(ea)).sum())
    out = {"Cells": n}
    if n < MIN_CELLS:
        return {**out, "Estimable": False, "Reason": "FEWER_THAN_5_CELLS"}, np.full(draws.shape[0], np.nan)
    diff = _nanmedian_rows(ea[draws]) - _nanmedian_rows(eb[draws])
    return {**out, "Observed_A": float(np.nanmedian(ea)), "Observed_B": float(np.nanmedian(eb)),
            "Observed_Difference": float(np.nanmedian(ea) - np.nanmedian(eb)), **_interval(diff)}, diff


@evaluation
def single_method_median(arr: Arrays, member: np.ndarray, method: str) -> tuple[float, int]:
    """Median of per-cell errors of one method on an evaluation set (BASE baseline for delta and detection)."""
    e, _ = per_cell_errors(arr, member, np.ones((len(member), 3), bool), method, method)
    n = int((~np.isnan(e)).sum())
    return (float(np.nanmedian(e)) if n >= MIN_CELLS else np.nan), n


@evaluation
def evaluate_dataset(ds: Exp3Dataset, thetas, pairs, keep_vectors_for=("RF-GB",), log=None):
    """v2 evaluation loop for the given thetas and method pairs. Returns (ci rows, E1_t rows, one-sided rows,
    vectors {(theta, q, eval set, pair): vector}, BASE baselines rows)."""
    cells = ds.cells
    draws, draw_sha = draws_for(len(cells))
    w = counts_matrix(draws, len(cells))
    ci_rows, e1t_rows, sec_rows, vectors, base_rows = [], [], [], {}, []
    for theta in thetas:
        s_dt = ds.samples[ds.samples.Theta == theta]
        m_e1 = membership(s_dt, cells, "", "E1")
        m_e1t = [membership(s_dt, cells, "", "E1_t", k) for k in range(3)]
        for q in QREFS:
            arr = _arrays(ds, theta, q)
            m_e2 = membership(s_dt, cells, q, "E2")
            m_e3 = membership(s_dt, cells, q, "E3")
            all_t = np.ones((len(cells), 3), bool)
            for e, mem in (("E1", m_e1), ("E2", m_e2)):
                med, n = single_method_median(arr, mem, "BASE")
                base_rows.append({"Dataset": ds.name, "Theta": theta, "Qref_Definition": q, "Eval_Set": e,
                                  "BASE_Cells": n, "BASE_Median_Error": med})
            for pair in pairs:
                a, b = pair.split("-")
                base = {"Dataset": ds.name, "Theta": theta, "Qref_Definition": q, "Pair": pair}
                r1, v1 = error_pair_ci_with_draws(*per_cell_errors(arr, m_e1, all_t, a, b), draws)
                r2, v2 = error_pair_ci_with_draws(*per_cell_errors(arr, m_e2, all_t, a, b), draws)
                r3 = c_index_pair_ci(arr, a, b, w)
                if pair in keep_vectors_for:
                    vectors[(theta, q, "E1", pair)] = v1
                    vectors[(theta, q, "E2", pair)] = v2
                for e, r, hib in (("E1", r1, False), ("E2", r2, False), ("E3", r3, True)):
                    ci_rows.append({**base, "Eval_Set": e, "Metric": "C_INDEX" if e == "E3" else "ABS_ERROR", **r,
                                    "Better": _better(r, a, b, hib)})
                for k in range(3):
                    mask = np.zeros((len(cells), 3), bool)
                    mask[:, k] = True
                    rk, _ = error_pair_ci_with_draws(*per_cell_errors(arr, m_e1t[k], mask, a, b), draws)
                    e1t_rows.append({**base, "t_Index": k, **rk, "Better": _better(rk, a, b, False)})
                ea, eb = per_cell_errors(arr, m_e3, all_t, a, b, one_sided=True)
                if (~np.isnan(ea)).sum() >= 5:
                    sec_rows.append({**base, "Cells": int((~np.isnan(ea)).sum()), "Median_A": float(np.nanmedian(ea)),
                                     "Median_B": float(np.nanmedian(eb))})
        if log:
            log(f"exp3 {ds.name} theta={theta:.2f} evaluated ({len(pairs)} pairs)")
    return ci_rows, e1t_rows, sec_rows, vectors, base_rows, draw_sha


@evaluation
def check5(ds_name: str, ci: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict]:
    """Check 5 over the whole grid with the best method among RF, TRAJ, WAFT (amendment 01 section 6), and its
    reproduction of the frozen v2 / v3 check5_baseline.csv."""
    sub = ci[ci.Dataset.eq(ds_name) & ci.Pair.isin(CHECK5_PAIRS)]
    detail, c5 = baseline_check(sub)
    frozen = pd.read_csv(Path(root) / (V3_CHECK5 if ds_name == "NCA" else V2_CHECK5), encoding="utf-8-sig")
    frozen = frozen[frozen.Dataset.eq(ds_name)].copy()
    key = ["Theta", "Qref_Definition", "Eval_Set"]
    cols = ["Best_Method", "Best_Clearly_Better"]
    a = detail.assign(Theta=detail.Theta.round(2))[key + cols] if len(detail) else pd.DataFrame(columns=key + cols)
    f = frozen.assign(Theta=frozen.Theta.round(2))[key + cols]
    m = a.merge(f, on=key, suffixes=("", "_frozen"), how="outer", indicator=True)
    rows_same = bool((m._merge == "both").all() and (m.Best_Method == m.Best_Method_frozen).all()
                     and (m.Best_Clearly_Better.astype(bool) == m.Best_Clearly_Better_frozen.astype(bool)).all())
    n_pass, n_all = int(detail.Best_Clearly_Better.sum()) if len(detail) else 0, int(len(detail))
    expected = CHECK5_EXPECTED[ds_name]
    rep = {"settings": n_all, "passed": n_pass, "expected": list(expected), "counts_equal": (n_pass, n_all) == tuple(expected),
           "rows_equal_frozen": rows_same, "row_mismatches": int((~((m._merge == "both") & (m.Best_Method == m.Best_Method_frozen))).sum())}
    rep["Reproduced"] = bool(rep["counts_equal"] and rows_same)
    rep["Pass"] = bool(c5["Pass"])
    rep["Clearly_Better_Share"] = c5["Clearly_Better_Share"]
    return detail, rep


@evaluation
def comparison_table(ci: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    """v2 find_reversals plus estimability of both settings and the pair-specific strict common window."""
    est = ci.set_index(["Dataset", "Theta", "Qref_Definition", "Eval_Set", "Pair"]).Estimable.to_dict()
    rev = find_reversals(ci)
    if not len(rev):
        return rev
    rev["Estimable_1"] = [bool(est[(r.Dataset, r.Theta, r.Setting_1, r.Eval_Set, r.Pair)]) for r in rev.itertuples()]
    rev["Estimable_2"] = [bool(est[(r.Dataset, r.Theta, r.Setting_2, r.Eval_Set, r.Pair)]) for r in rev.itertuples()]
    rev["Estimable_Comparison"] = rev.Estimable_1 & rev.Estimable_2
    rev["Qref_Pair"] = rev.Setting_1.map(QREF_SHORT) + "-" + rev.Setting_2.map(QREF_SHORT)
    w = win.assign(Theta=win.Theta.round(2)).set_index(["Dataset", "Setting_1", "Setting_2", "Theta"]).In_Common_Window.to_dict()
    rev["In_Common_Window"] = [bool(w.get((r.Dataset, r.Setting_1, r.Setting_2, round(r.Theta, 2)), False)) for r in rev.itertuples()]
    rev["Decisive_H3"] = rev.Pair.eq("RF-GB")
    return rev


@evaluation
def common_practice(ci: pd.DataFrame) -> pd.DataFrame:
    """Section 4 C condition: theta 0.80, frozen folds, point estimates and apparent ranking changes only."""
    c = ci[np.isclose(ci.Theta, EXP3_C_THETA)]
    rows = []
    for (d, e, pair), g in c.groupby(["Dataset", "Eval_Set", "Pair"]):
        obs = g.set_index("Qref_Definition")
        for q1, q2 in QREF_PAIRS:
            if q1 not in obs.index or q2 not in obs.index:
                continue
            d1, d2 = obs.at[q1, "Observed_Difference"] if "Observed_Difference" in obs else np.nan, \
                obs.at[q2, "Observed_Difference"] if "Observed_Difference" in obs else np.nan
            rows.append({"Dataset": d, "Eval_Set": e, "Pair": pair, "Setting_1": q1, "Setting_2": q2,
                         "Observed_A_1": obs.at[q1, "Observed_A"] if "Observed_A" in obs else np.nan,
                         "Observed_B_1": obs.at[q1, "Observed_B"] if "Observed_B" in obs else np.nan,
                         "Observed_A_2": obs.at[q2, "Observed_A"] if "Observed_A" in obs else np.nan,
                         "Observed_B_2": obs.at[q2, "Observed_B"] if "Observed_B" in obs else np.nan,
                         "Observed_Difference_1": d1, "Observed_Difference_2": d2,
                         "Apparent_Ranking_Change": bool(np.isfinite(d1) and np.isfinite(d2) and d1 * d2 < 0)})
    return pd.DataFrame(rows)
