"""End-to-end run of the RUL ranking-sensitivity pilot v2 (PROTOCOL_v2.md)."""

from __future__ import annotations

import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rul_ranking_pilot.labels import (
    PilotError,
    _detection_record,
    check_frozen_labels,
    check_monotonicity,
    compute_t0,
    detect,
    detect_all,
)
from rul_ranking_pilot.methods import _sha
from rul_ranking_pilot.pipeline import INPUT_FILES as V1_INPUT_FILES, _file_sha, _log, _write

from .config import (
    DATASETS,
    MARGIN_FEATURE,
    METHODS,
    MIT_DIR,
    MIT_FREEZE_COMMIT,
    NOMINAL_CAPACITY_AH,
    OUTPUT_DIR_NAME,
    PIPELINE_VERSION,
    PROTOCOL_COMMIT,
    QREFS,
    T0_FILE,
    THETAS,
    XJTU_DRAW_SHA256,
)
from .data import base_features, build_samples, load_all, load_t0
from .evaluation import (
    PAIRS,
    Arrays,
    _better,
    baseline_check,
    c_index_pair_ci,
    counts_matrix,
    draws_for,
    error_pair_ci,
    find_reversals,
    membership,
    per_cell_errors,
    strong_rule,
)
from .methods import run_setting

INPUT_FILES = (*V1_INPUT_FILES, f"{MIT_DIR}/mit_primary_capacity.csv", f"{MIT_DIR}/mit_qref_summary.csv",
               f"{MIT_DIR}/mit_cell_fold_assignments.csv", f"{MIT_DIR}/mit_endpoint_labels.csv",
               f"{MIT_DIR}/mit_check1_cycle_life.csv", T0_FILE)


def _check_mit_labels(root: Path, det: pd.DataFrame) -> dict:
    frozen = pd.read_csv(root / MIT_DIR / "mit_endpoint_labels.csv", encoding="utf-8-sig")
    frozen["Theta"] = frozen.Theta.round(2)
    cur = det[det.Dataset.eq("MIT")].assign(Theta=lambda f: f.Theta.round(2))
    m = cur.merge(frozen, on=["Cell_ID", "Qref_Definition", "Theta"], suffixes=("", "_frozen"))
    same = m.Status.eq(m.Status_frozen) & ((m.First_Hit_Raw_Cycle == m.First_Hit_Cycle) | (m.First_Hit_Raw_Cycle.isna() & m.First_Hit_Cycle.isna()))
    c1 = pd.read_csv(root / MIT_DIR / "mit_check1_cycle_life.csv", encoding="utf-8-sig")
    ex = c1[c1.Status.eq("EXACT")]
    return {"rows_compared": int(len(m)), "rows_expected": int(len(cur)), "mismatches": int((~same).sum()),
            "check1_exact": int(len(ex)), "check1_exact_within_1": int((ex.Difference_Cycles.abs() <= 1).sum()),
            "Pass": bool(len(m) == len(cur) and same.all() and len(ex) and (ex.Difference_Cycles.abs() <= 1).all())}


def _physical_equivalence(data, det: pd.DataFrame) -> dict:
    compare = ["Status", "First_Hit_Raw_Cycle", "EOL_Exact_Index", "EOL_Lower_Index", "EOL_Upper_Index"]
    bad, n, max_diff = 0, 0, 0.0
    for r in det[det.Qref_Definition.ne("nominal")].itertuples(index=False):
        nominal = NOMINAL_CAPACITY_AH[r.Dataset]
        thr = (r.Theta * r.Qref_Ah / nominal) * nominal
        rec = _detection_record(detect(data[r.Dataset], r.Cell_ID, thr))
        n += 1
        max_diff = max(max_diff, abs(thr - r.Threshold_Ah))
        if not all(rec[k] == getattr(r, k) or (pd.isna(rec[k]) and pd.isna(getattr(r, k))) for k in compare):
            bad += 1
    return {"comparisons": n, "mismatches": bad, "max_threshold_diff_Ah": max_diff, "Pass": bad == 0}


def _mit_rebound(data, det: pd.DataFrame) -> pd.DataFrame:
    cap = {c: g.sort_values("Raw_Cycle_Index") for c, g in data["MIT"].capacity.groupby("Cell_ID")}
    rows = []
    for r in det[det.Dataset.eq("MIT") & det.Status.isin(["EXACT", "INTERVAL"])].itertuples(index=False):
        g = cap[r.Cell_ID]
        rows.append({"Theta": round(r.Theta, 2), "Qref_Definition": r.Qref_Definition, "Status": r.Status,
                     "Rebound": bool((g[g.Raw_Cycle_Index > r.First_Hit_Raw_Cycle].Capacity_Ah > r.Threshold_Ah).any())})
    d = pd.DataFrame(rows)
    return d.groupby(["Theta", "Status"]).Rebound.agg(Rebound_Settings="sum", Settings="size").reset_index().assign(
        Rebound_Fraction=lambda f: f.Rebound_Settings / f.Settings)


def run_pilot(root: Path, output_dir: Path | None = None, thetas: tuple[float, ...] = THETAS,
              datasets: tuple[str, ...] = DATASETS) -> Path:
    root = Path(root).resolve()
    target = Path(output_dir).resolve() if output_dir else root / "outputs" / OUTPUT_DIR_NAME
    if target.exists():
        raise PilotError(f"OUTPUT_DIRECTORY_EXISTS_REFUSING_TO_OVERWRITE:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}_staging_", dir=target.parent))
    log: list[str] = []
    before = {f: _file_sha(root / f) for f in INPUT_FILES}
    _log(log, f"{PIPELINE_VERSION}; protocol {PROTOCOL_COMMIT}; MIT freeze {MIT_FREEZE_COMMIT}; thetas={len(thetas)}; datasets={datasets}")

    data = load_all(root)
    t0 = load_t0(root)
    det = detect_all(data)
    det["Theta"] = det.Theta.round(2)
    recomputed = compute_t0(data, det).set_index("Dataset").t0_raw_cycle.to_dict()
    t0_ok = all(int(recomputed[d]) == t0[d][0] for d in DATASETS)
    samples = build_samples(data, det, t0)
    _write(det, staging / "labels.csv")
    _write(samples, staging / "samples.csv")

    check1 = check_frozen_labels(root, det)
    _write(check1, staging / "check1_frozen_labels_xjtu_hust.csv")
    mit1 = _check_mit_labels(root, det)
    phys = _physical_equivalence(data, det)
    mono = check_monotonicity(det)
    checks = [
        {"Check": "0_T0_MATCHES_FROZEN", "Detail": json.dumps({"recomputed_t0_1": recomputed, "frozen": t0}), "Pass": t0_ok},
        {"Check": "1_LABELS", "Detail": f"XJTU/HUST {int(check1.Pass.sum())}/{len(check1)} rows; MIT {json.dumps(mit1)}",
         "Pass": bool(check1.Pass.all()) and mit1["Pass"]},
        {"Check": "2_PHYSICAL_EQUIVALENCE", "Detail": json.dumps(phys), "Pass": phys["Pass"]},
        {"Check": "3_MONOTONICITY", "Detail": f"{int(mono.Violations.iat[0])} violations", "Pass": bool(mono.Pass.iat[0])},
    ]
    if not all(c["Pass"] for c in checks):
        _write(pd.DataFrame(checks), staging / "sanity_checks.csv")
        raise PilotError(f"LABEL_CHECKS_FAILED; partial outputs in {staging}")
    _log(log, "checks 0-3 passed")

    feats, max_used, design_cols = {}, {}, {}
    for d in datasets:
        feats[d], max_used[d] = base_features(data[d], t0[d])
        design_cols[d] = [c for c in feats[d].columns if c not in ("Cell_ID", "t_Index")] + [MARGIN_FEATURE]
        _write(feats[d], staging / f"features_{d.lower()}.csv")
    forbidden = [c for d in datasets for c in design_cols[d] if "qref" in c.lower()]

    preds, est, audit, traj_eq = [], [], [], []
    for d in datasets:
        cap_by_cell = {c: g for c, g in data[d].capacity.groupby("Cell_ID")}
        for theta in thetas:
            for q in QREFS:
                s = samples[(samples.Dataset == d) & (samples.Theta == theta) & (samples.Qref_Definition == q) & samples.Eligible]
                s = s.reset_index(drop=True)
                qref = data[d].qref[q]
                xs = s[["Cell_ID", "t_Index"]].merge(feats[d], on=["Cell_ID", "t_Index"], how="left")
                xs[MARGIN_FEATURE] = xs.Capacity_t_Ah - theta * xs.Cell_ID.map(qref).astype(float)
                x = xs[design_cols[d]].to_numpy(float)
                thr = {c: theta * float(qref[c]) for c in data[d].cells}
                res = run_setting(s, x, cap_by_cell, thr, t0[d])
                p = pd.concat([s[["Dataset", "Theta", "Qref_Definition", "Cell_ID", "t_Index", "Fold_ID", "Status", "RUL",
                                  "RUL_Lower", "RUL_Upper"]], res.predictions.drop(columns=["Cell_ID", "t_Index"])], axis=1)
                preds.append(p)
                for m in METHODS:
                    est.append({"Dataset": d, "Theta": theta, "Qref_Definition": q, "Method": m, "Estimable": res.estimable[m],
                                "Reason": res.reasons[m], "Samples": len(s),
                                "TRAJ_Fallback_Fraction": res.traj_fallback_fraction if m == "TRAJ" else np.nan,
                                "TRAJ_Missing_Fraction": res.traj_missing_fraction if m == "TRAJ" else np.nan})
                for a in res.audit:
                    rows = a["Train_Rows"]
                    med = np.nanmedian(x[rows], axis=0) if len(rows) else np.full(x.shape[1], np.nan)
                    med = np.where(np.isnan(med), 0.0, med)
                    all_med = np.where(np.isnan(np.nanmedian(x, axis=0)), 0.0, np.nanmedian(x, axis=0))
                    audit.append({"Dataset": d, "Theta": theta, "Qref_Definition": q, "Method": a["Method"], "Fold_ID": a["Fold_ID"],
                                  "Train_Folds_Clean": bool((s.Fold_ID.to_numpy()[rows] != a["Fold_ID"]).all()),
                                  "Imputer_Recomputed_Match": _sha(med) == a["Imputer_Median_SHA256"],
                                  "NegControl_All_Rows_Match": _sha(all_med) == a["Imputer_Median_SHA256"],
                                  "Dropped_Constant_Columns": a.get("Dropped_Constant_Columns", np.nan)})
                if q != "nominal":
                    nominal = NOMINAL_CAPACITY_AH[d]
                    alt = {c: (theta * float(qref[c]) / nominal) * nominal for c in data[d].cells}
                    r2 = run_setting(s, x, cap_by_cell, alt, t0[d], traj_only=True)
                    a1, a2 = res.predictions.Pred_TRAJ.to_numpy(float), r2.predictions.Pred_TRAJ.to_numpy(float)
                    both = ~np.isnan(a1) & ~np.isnan(a2)
                    traj_eq.append({"Dataset": d, "Theta": theta, "Qref_Definition": q,
                                    "Estimable_Equal": res.estimable["TRAJ"] == r2.estimable["TRAJ"],
                                    "Max_Rel_Diff": float(np.max(np.abs(a1[both] - a2[both]) / np.maximum(1.0, np.abs(a1[both])))) if both.any() else 0.0})
            _log(log, f"{d} theta={theta:.2f} trained")

    predictions = pd.concat(preds, ignore_index=True)
    est = pd.DataFrame(est)
    audit = pd.DataFrame(audit)
    traj_eq = pd.DataFrame(traj_eq)
    _write(predictions, staging / "predictions.csv")
    _write(est, staging / "method_estimability.csv")
    _write(audit, staging / "preprocessing_audit.csv")
    _write(traj_eq, staging / "check2_traj_equivalence.csv")
    check4 = {"fold_cells_disjoint": bool(audit.Train_Folds_Clean.all()),
              "max_raw_cycle_used_le_t0_3": all(max_used[d] <= t0[d][2] for d in datasets),
              "design_has_no_qref_column": not forbidden,
              "fold_statistics_from_training_rows_only": bool(audit.Imputer_Recomputed_Match.all()),
              "negative_control_all_rows_differs": bool((~audit.NegControl_All_Rows_Match).all())}
    traj_ok = bool(traj_eq.Estimable_Equal.all() and (traj_eq.Max_Rel_Diff <= 1e-9).all())
    checks[2]["Detail"] += f"; TRAJ max rel diff {traj_eq.Max_Rel_Diff.max():.3g}"
    checks[2]["Pass"] = checks[2]["Pass"] and traj_ok
    checks.append({"Check": "4_NO_LEAKAGE", "Detail": json.dumps(check4), "Pass": all(check4.values())})
    if not (checks[2]["Pass"] and checks[-1]["Pass"]):
        _write(pd.DataFrame(checks), staging / "sanity_checks.csv")
        raise PilotError(f"CHECK_2_OR_4_FAILED; partial outputs in {staging}")
    _log(log, "checks 2 (TRAJ) and 4 passed")

    ci_rows, e1t_rows, sec_rows, draw_hash = [], [], [], {}
    for d in datasets:
        cells = data[d].cells
        draws, draw_hash[d] = draws_for(len(cells))
        if d == "XJTU" and draw_hash[d] != XJTU_DRAW_SHA256:
            raise PilotError("XJTU_DRAWS_DIFFER_FROM_S6_S7")
        w = counts_matrix(draws, len(cells))
        for theta in thetas:
            s_dt = samples[(samples.Dataset == d) & (samples.Theta == theta)]
            m_e1 = membership(s_dt, cells, "", "E1")
            m_e1t = [membership(s_dt, cells, "", "E1_t", k) for k in range(3)]
            for q in QREFS:
                arr = Arrays(cells, s_dt[s_dt.Qref_Definition == q],
                             predictions[(predictions.Dataset == d) & (predictions.Theta == theta) & (predictions.Qref_Definition == q)])
                m_e2 = membership(s_dt, cells, q, "E2")
                m_e3 = membership(s_dt, cells, q, "E3")
                all_t = np.ones((len(cells), 3), bool)
                for pair in PAIRS:
                    a, b = pair.split("-")
                    base = {"Dataset": d, "Theta": theta, "Qref_Definition": q, "Pair": pair}
                    r1 = error_pair_ci(*per_cell_errors(arr, m_e1, all_t, a, b), draws)
                    r2 = error_pair_ci(*per_cell_errors(arr, m_e2, all_t, a, b), draws)
                    r3 = c_index_pair_ci(arr, a, b, w)
                    for e, r, hib in (("E1", r1, False), ("E2", r2, False), ("E3", r3, True)):
                        ci_rows.append({**base, "Eval_Set": e, "Metric": "C_INDEX" if e == "E3" else "ABS_ERROR", **r,
                                        "Better": _better(r, a, b, hib)})
                    for k in range(3):
                        mask = np.zeros((len(cells), 3), bool)
                        mask[:, k] = True
                        rk = error_pair_ci(*per_cell_errors(arr, m_e1t[k], mask, a, b), draws)
                        e1t_rows.append({**base, "t_Index": k, **rk, "Better": _better(rk, a, b, False)})
                    ea, eb = per_cell_errors(arr, m_e3, all_t, a, b, one_sided=True)
                    if (~np.isnan(ea)).sum() >= 5:
                        sec_rows.append({**base, "Cells": int((~np.isnan(ea)).sum()), "Median_A": float(np.nanmedian(ea)),
                                         "Median_B": float(np.nanmedian(eb))})
        _log(log, f"{d} bootstrap done")

    ci = pd.DataFrame(ci_rows)
    rev = find_reversals(ci)
    strong = strong_rule(rev)
    base_detail, check5 = baseline_check(ci)
    checks.append({"Check": "5_BASELINE_WORSE", "Detail": f"{check5['Clearly_Better_Share']:.3f} of {check5['Settings']} E1/E2 settings",
                   "Pass": check5["Pass"]})
    checks.append({"Check": "6_DETERMINISM", "Detail": "compare two independent output directories", "Pass": None})
    decision = "UNINFORMATIVE" if not check5["Pass"] else ("STRONG" if strong.Meets_Strong_Threshold.any() else "WEAK")

    sup = rev[rev.Supported_Reversal]
    by_theta = sup.groupby(["Dataset", "Eval_Set", "Theta"]).size().rename("Supported_Reversals").reset_index()
    _write(ci, staging / "pair_ci.csv")
    _write(rev, staging / "reversals.csv")
    _write(strong, staging / "strong_rule.csv")
    _write(base_detail, staging / "check5_baseline.csv")
    _write(pd.DataFrame(e1t_rows), staging / "descriptive_e1_by_t0.csv")
    _write(pd.DataFrame(sec_rows), staging / "descriptive_e3_one_sided_error.csv")
    _write(by_theta, staging / "descriptive_reversals_by_theta.csv")
    if "MIT" in datasets:
        _write(_mit_rebound(data, det), staging / "descriptive_mit_rebound_by_theta.csv")
    _write(pd.DataFrame(checks), staging / "sanity_checks.csv")
    result = {"decision": decision, "t0": t0, "comparisons": int(len(rev)), "supported_reversals": int(len(sup)),
              "supported_reversals_by_eval_set": {k: int(v) for k, v in sup.Eval_Set.value_counts().sort_index().items()},
              "max_S_size": int(strong.S_Size.max()) if len(strong) else 0,
              "estimable_intervals": int(ci.Estimable.sum()), "not_estimable_intervals": int((~ci.Estimable).sum()),
              "not_estimable_method_settings": int((~est.Estimable).sum()), "check5": check5}
    (staging / "decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if {f: _file_sha(root / f) for f in INPUT_FILES} != before:
        raise PilotError("INPUT_FILES_CHANGED_DURING_RUN")
    import lifelines
    import scipy
    import sklearn

    info = {"pipeline": PIPELINE_VERSION, "protocol_commit": PROTOCOL_COMMIT, "mit_freeze_commit": MIT_FREEZE_COMMIT,
            "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "lifelines": lifelines.__version__,
            "thetas": list(thetas), "datasets": list(datasets), "bootstrap_draw_sha256": draw_hash, "input_sha256": before}
    (staging / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    _log(log, f"decision: {decision}")
    (staging / "processing_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    staging.replace(target)
    return target
