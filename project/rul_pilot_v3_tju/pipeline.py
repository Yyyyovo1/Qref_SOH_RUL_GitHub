"""End-to-end run of the RUL ranking-sensitivity v3 on TJU (PROTOCOL_v3_TJU.md, commit 332dbf4).

Labels, samples, methods, evaluation and bootstrap reuse the v1/v2 code unchanged; only the data adapter, the
theta grid, the static TJU features and the P0 / P1 decision are new. `dev=True` stops after checks 0-4 and method
estimability (no bootstrap, no comparison, no reversal is computed).

Amendment 01 (commit 840bd85): replacement negative control in check 4 (fold means when fold medians coincide),
per-dataset exclusion instead of a global stop, and a main (amended check 4) and a secondary (original check 4) verdict.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from protocol_validation.endpoint import tolerance_rule
from rul_pilot_v2.data import build_samples
from rul_pilot_v2.evaluation import (
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
)
from rul_pilot_v2.methods import run_setting
from rul_ranking_pilot.labels import PilotError, _detection_record, check_monotonicity, detect
from rul_ranking_pilot.methods import _sha
from rul_ranking_pilot.pipeline import _file_sha, _log, _write

from .config import (
    DATASETS,
    LABEL_REFERENCE,
    MARGIN_FEATURE,
    METHODS,
    MIN_ESTIMABLE_COMPARISONS,
    NOMINAL_CAPACITY_AH,
    OUTPUT_DIR_NAME,
    P0_MAX_REVERSALS,
    P1_MIN_RATE_A,
    P1_MIN_RATIO,
    P1_SEGMENTS,
    PIPELINE_VERSION,
    PROTOCOL_COMMIT,
    QREF_SHORT,
    QREFS,
    T0,
    THETAS,
    TJU_DIR,
    TJU_FREEZE_COMMIT,
)
from .data import base_features, compute_t0_1, detect_all, load_tju

INPUT_FILES = (f"{TJU_DIR}/tju_primary_capacity.csv", f"{TJU_DIR}/tju_qref_summary.csv",
               f"{TJU_DIR}/tju_cell_fold_assignments.csv", f"{TJU_DIR}/tju_cell_inventory.csv", LABEL_REFERENCE)


# ---------------------------------------------------------------- checks 0-3
def _check_labels(root: Path, data, det: pd.DataFrame) -> dict:
    """Check 1: strict status, first hit and tolerance kind equal the label-structure reference (committed with the protocol)."""
    ref = pd.read_csv(root / LABEL_REFERENCE, encoding="utf-8-sig")
    ref["Theta"] = ref.Theta.round(2)
    ref["Qref_Definition"] = ref.Qref.map({v: k for k, v in QREF_SHORT.items()})
    cur = det.copy()
    cap = {(n, c): g for n, d in data.items() for c, g in d.capacity.groupby("Cell_ID")}
    cur["Tolerance_Kind"] = [tolerance_rule(cap[(n, c)].Endpoint_Index.to_numpy(float), cap[(n, c)].Capacity_Ah.to_numpy(float), t)[0]
                             for n, c, t in zip(cur.Dataset, cur.Cell_ID, cur.Threshold_Ah)]
    m = cur.merge(ref, on=["Dataset", "Cell_ID", "Qref_Definition", "Theta"], suffixes=("", "_ref"))
    same_hit = (m.First_Hit_Raw_Cycle == m.First_Hit) | (m.First_Hit_Raw_Cycle.isna() & m.First_Hit.isna())
    same = m.Status.eq(m.Status_ref) & same_hit & m.Tolerance_Kind.eq(m.Tolerance_Kind_ref)
    mism = m[~same].groupby("Dataset").size().to_dict()
    return {"rows_compared": int(len(m)), "rows_expected": int(len(cur)), "mismatches": int((~same).sum()),
            "mismatches_by_dataset": {k: int(v) for k, v in mism.items()}, "per_dataset_pass": {
                d: bool(len(m[m.Dataset == d]) == len(cur[cur.Dataset == d]) and d not in mism) for d in cur.Dataset.unique()},
            "Pass": bool(len(m) == len(cur) and same.all())}


def _physical_equivalence(data, det: pd.DataFrame) -> dict:
    compare = ["Status", "First_Hit_Raw_Cycle", "EOL_Exact_Index", "EOL_Lower_Index", "EOL_Upper_Index"]
    bad, n, max_diff, bad_by = 0, 0, 0.0, {}
    for r in det[det.Qref_Definition.ne("nominal")].itertuples(index=False):
        nominal = NOMINAL_CAPACITY_AH[r.Dataset]
        thr = (r.Theta * r.Qref_Ah / nominal) * nominal
        rec = _detection_record(detect(data[r.Dataset], r.Cell_ID, thr))
        n += 1
        max_diff = max(max_diff, abs(thr - r.Threshold_Ah))
        if not all(rec[k] == getattr(r, k) or (pd.isna(rec[k]) and pd.isna(getattr(r, k))) for k in compare):
            bad += 1
            bad_by[r.Dataset] = bad_by.get(r.Dataset, 0) + 1
    return {"comparisons": n, "mismatches": bad, "mismatches_by_dataset": bad_by, "max_threshold_diff_Ah": max_diff,
            "Pass": bad == 0}


# ---------------------------------------------------------------- P0 / P1
def strict_shares(det: pd.DataFrame) -> pd.Series:
    return det.assign(ex=det.Status.eq("EXACT")).groupby(["Dataset", "Qref_Definition", "Theta"]).ex.mean()


def tolerance_shares(root: Path) -> pd.Series:
    ref = pd.read_csv(root / LABEL_REFERENCE, encoding="utf-8-sig")
    ref["Theta"] = ref.Theta.round(2)
    ref["Qref_Definition"] = ref.Qref.map({v: k for k, v in QREF_SHORT.items()})
    return ref.assign(lab=ref.Tolerance_Kind.ne("NO_LABEL")).groupby(["Dataset", "Qref_Definition", "Theta"]).lab.mean()


def comparison_table(ci: pd.DataFrame, rev: pd.DataFrame, det: pd.DataFrame, tol: pd.Series) -> pd.DataFrame:
    est = ci.set_index(["Dataset", "Theta", "Qref_Definition", "Eval_Set", "Pair"]).Estimable.to_dict()
    sh = strict_shares(det)
    out = rev.copy()
    out["Estimable_1"] = [bool(est[(r.Dataset, r.Theta, r.Setting_1, r.Eval_Set, r.Pair)]) for r in out.itertuples()]
    out["Estimable_2"] = [bool(est[(r.Dataset, r.Theta, r.Setting_2, r.Eval_Set, r.Pair)]) for r in out.itertuples()]
    out["Estimable_Comparison"] = out.Estimable_1 & out.Estimable_2
    out["Qref_Pair"] = out.Setting_1.map(QREF_SHORT) + "-" + out.Setting_2.map(QREF_SHORT)
    out["Strict_Share_1"] = [float(sh[(r.Dataset, r.Setting_1, r.Theta)]) for r in out.itertuples()]
    out["Strict_Share_2"] = [float(sh[(r.Dataset, r.Setting_2, r.Theta)]) for r in out.itertuples()]
    out["Lower_Strict_Share"] = out[["Strict_Share_1", "Strict_Share_2"]].min(axis=1)
    out["Lower_Side"] = np.where(out.Strict_Share_1 < out.Strict_Share_2, "setting_1",
                                 np.where(out.Strict_Share_2 < out.Strict_Share_1, "setting_2", "tie"))
    out["Tolerance_Lower_Share"] = [min(float(tol[(r.Dataset, r.Setting_1, r.Theta)]), float(tol[(r.Dataset, r.Setting_2, r.Theta)]))
                                    for r in out.itertuples()]
    out["Segment"] = _segment(out.Lower_Strict_Share)
    out["Tolerance_Segment"] = _segment(out.Tolerance_Lower_Share)
    out["Includes_TRAJ"] = out.Pair.str.contains("TRAJ")
    return out


def _segment(r: pd.Series) -> np.ndarray:
    a_lo, a_hi = P1_SEGMENTS["A"]
    b_lo, b_hi = P1_SEGMENTS["B"]
    return np.select([r < a_lo, (r >= a_lo) & (r < a_hi), (r >= b_lo) & (r < b_hi)], ["below_30", "A_30_60", "B_60_90"], "at_least_90")


def _rate(g: pd.DataFrame) -> dict:
    e = g[g.Estimable_Comparison]
    n, k = int(len(e)), int(e.Supported_Reversal.sum())
    return {"Estimable_Comparisons": n, "Supported_Reversals": k, "Reversal_Rate": k / n if n else None}


def decide(comp: pd.DataFrame, check5: dict[str, bool], ok_datasets: list[str]) -> dict:
    """P0 over datasets passing checks 0-4; P1 over those that also pass check 5 (section 7, amendment 01)."""
    comp = comp[comp.Dataset.isin(ok_datasets)]
    check5 = {d: v for d, v in check5.items() if d in ok_datasets}
    ff = comp[comp.Qref_Pair.eq("first-first5")]
    p0 = _rate(ff)
    p0["Datasets_Used"] = list(ok_datasets)
    p0["Verdict"] = ("UNTESTABLE" if not ok_datasets or p0["Estimable_Comparisons"] < MIN_ESTIMABLE_COMPARISONS
                     else ("HOLDS" if p0["Supported_Reversals"] <= P0_MAX_REVERSALS else "DOES_NOT_HOLD"))
    nm = comp[comp.Setting_1.eq("nominal")]
    used = [d for d, ok in check5.items() if ok]
    seg = {s: _rate(nm[nm.Dataset.isin(used) & nm.Segment.eq(s)]) for s in ("A_30_60", "B_60_90")}
    a, b = seg["A_30_60"], seg["B_60_90"]
    if not used:
        verdict, reason = "UNTESTABLE", "NO_DATASET_PASSED_CHECK_5"
    elif a["Estimable_Comparisons"] < MIN_ESTIMABLE_COMPARISONS or b["Estimable_Comparisons"] < MIN_ESTIMABLE_COMPARISONS:
        verdict, reason = "UNTESTABLE", "A_OR_B_FEWER_THAN_100_ESTIMABLE_COMPARISONS"
    else:
        ra, rb = a["Reversal_Rate"], b["Reversal_Rate"]
        ok = ra >= P1_MIN_RATIO * rb and ra >= P1_MIN_RATE_A
        verdict, reason = ("HOLDS" if ok else "DOES_NOT_HOLD"), f"rate_A={ra:.4f}, rate_B={rb:.4f}"
    return {"P0": p0, "P1": {"Datasets_Used": used, "Check5_Pass": check5, "Segments": seg, "Verdict": verdict, "Detail": reason}}


def descriptives(comp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    nm = comp[comp.Setting_1.eq("nominal")]
    rows = []
    for scope, g0 in (("P0_first_first5", comp[comp.Qref_Pair.eq("first-first5")]), ("P1_nominal_measured", nm)):
        for by in (None, "Dataset", "Eval_Set", "Includes_TRAJ"):
            for seg, g1 in g0.groupby("Segment"):
                parts = [("all", g1)] if by is None else list(g1.groupby(by))
                for lvl, g in parts:
                    rows.append({"Scope": scope, "Segment": seg, "By": by or "all", "Level": str(lvl), **_rate(g)})
    by_seg = pd.DataFrame(rows)
    # Mechanism split of the nominal-vs-measured comparisons (descriptive).
    mech = nm.assign(Lower=np.where(nm.Lower_Side.eq("setting_1"), "nominal_lower",
                                    np.where(nm.Lower_Side.eq("setting_2"), "measured_lower", "tie")))
    mech = pd.DataFrame([{"Segment": s, "Lower": lo, **_rate(g)} for (s, lo), g in mech.groupby(["Segment", "Lower"])])
    # De-duplicated events: (dataset, theta, pair, Qref pair), any evaluation set.
    ev = comp.groupby(["Dataset", "Theta", "Pair", "Qref_Pair", "Segment"]).agg(
        Any_Estimable=("Estimable_Comparison", "any"),
        Any_Reversal=("Supported_Reversal", lambda s: bool((s & comp.loc[s.index, "Estimable_Comparison"]).any()))).reset_index()
    ev["Scope"] = np.where(ev.Qref_Pair.eq("first-first5"), "P0_first_first5", "P1_nominal_measured")
    events = ev[ev.Any_Estimable].groupby(["Scope", "Segment"]).agg(Estimable_Events=("Any_Estimable", "size"),
                                                                    Reversal_Events=("Any_Reversal", "sum")).reset_index()
    tol = pd.DataFrame([{"Scope": sc, "Tolerance_Segment": s, **_rate(g)}
                        for sc, g0 in (("P0_first_first5", comp[comp.Qref_Pair.eq("first-first5")]), ("P1_nominal_measured", nm))
                        for s, g in g0.groupby("Tolerance_Segment")])
    sup = comp[comp.Supported_Reversal & comp.Estimable_Comparison][
        ["Dataset", "Theta", "Eval_Set", "Pair", "Qref_Pair", "Better_1", "Better_2", "Strict_Share_1", "Strict_Share_2",
         "Lower_Strict_Share", "Segment", "Tolerance_Lower_Share"]]
    return {"descriptive_by_segment": by_seg, "descriptive_mechanism": mech, "descriptive_dedup_events": events,
            "descriptive_tolerance_segments": tol, "supported_reversals_list": sup}


# ---------------------------------------------------------------- run
def run(root: Path, output_dir: Path | None = None, thetas=THETAS, datasets=DATASETS, dev: bool = False) -> Path:
    root = Path(root).resolve()
    target = Path(output_dir).resolve() if output_dir else root / "outputs" / OUTPUT_DIR_NAME
    if target.exists():
        raise PilotError(f"OUTPUT_DIRECTORY_EXISTS_REFUSING_TO_OVERWRITE:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}_staging_", dir=target.parent))
    log: list[str] = []
    before = {f: _file_sha(root / f) for f in INPUT_FILES}
    _log(log, f"{PIPELINE_VERSION}; protocol {PROTOCOL_COMMIT}; TJU freeze {TJU_FREEZE_COMMIT}; thetas={len(thetas)}; "
              f"datasets={datasets}; dev={dev}")

    data = {d: v for d, v in load_tju(root).items() if d in datasets}
    det_full = detect_all(data, THETAS)
    det_full["Theta"] = det_full.Theta.round(2)
    t0_re = compute_t0_1(data, det_full)
    t0 = {d: T0[d] for d in datasets}
    t0_ok = all(tuple(t0_re[d]["t0"]) == T0[d] for d in datasets)
    det = det_full[det_full.Theta.isin([round(t, 2) for t in thetas])].reset_index(drop=True)
    samples = build_samples(data, det, t0)
    _write(det, staging / "labels.csv")
    _write(samples, staging / "samples.csv")

    c1 = _check_labels(root, data, det)
    phys = _physical_equivalence(data, det)
    mono = check_monotonicity(det)
    checks = [
        {"Check": "0_T0_MATCHES_FROZEN", "Detail": json.dumps({"recomputed": t0_re, "frozen": t0}, default=list), "Pass": t0_ok},
        {"Check": "1_LABELS", "Detail": json.dumps(c1), "Pass": c1["Pass"]},
        {"Check": "2_PHYSICAL_EQUIVALENCE", "Detail": json.dumps(phys), "Pass": phys["Pass"]},
        {"Check": "3_MONOTONICITY", "Detail": f"{int(mono.Violations.iat[0])} violations", "Pass": bool(mono.Pass.iat[0])},
    ]
    per_ds = {d: {"0": tuple(t0_re[d]["t0"]) == T0[d], "1": bool(c1["per_dataset_pass"].get(d, False)),
                  "2": d not in phys["mismatches_by_dataset"],
                  "3": bool(check_monotonicity(det[det.Dataset.eq(d)]).Pass.iat[0])} for d in datasets}
    _log(log, "checks 0-3: " + json.dumps(per_ds))

    feats, max_used, design_cols, feat_sha = {}, {}, {}, {}
    for d in datasets:
        feats[d], max_used[d] = base_features(data[d], t0[d])
        design_cols[d] = [c for c in feats[d].columns if c not in ("Cell_ID", "t_Index")] + [MARGIN_FEATURE]
        feat_sha[d] = hashlib.sha256(feats[d].to_csv(index=False).encode()).hexdigest()
        _write(feats[d], staging / f"features_{d.lower()}.csv")
    forbidden = [c for d in datasets for c in design_cols[d] if "qref" in c.lower()]

    preds, est, audit, traj_eq, design_sha = [], [], [], [], []
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
                design_sha.append({"Dataset": d, "Theta": theta, "Qref_Definition": q,
                                   "Non_Margin_Rows_Match_Shared_Features": bool(
                                       xs.drop(columns=[MARGIN_FEATURE]).reset_index(drop=True).equals(
                                           s[["Cell_ID", "t_Index"]].merge(feats[d], on=["Cell_ID", "t_Index"], how="left").reset_index(drop=True)))})
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
                all_med = np.where(np.isnan(np.nanmedian(x, axis=0)), 0.0, np.nanmedian(x, axis=0))
                all_mean = np.where(np.isnan(np.nanmean(x, axis=0)), 0.0, np.nanmean(x, axis=0))
                for a in res.audit:
                    rows = a["Train_Rows"]
                    med = np.nanmedian(x[rows], axis=0) if len(rows) else np.full(x.shape[1], np.nan)
                    med = np.where(np.isnan(med), 0.0, med)
                    mean = np.nanmean(x[rows], axis=0) if len(rows) else np.full(x.shape[1], np.nan)
                    mean = np.where(np.isnan(mean), 0.0, mean)
                    neg_match = _sha(all_med) == a["Imputer_Median_SHA256"]
                    mean_differs = _sha(mean) != _sha(all_mean)
                    audit.append({"Dataset": d, "Theta": theta, "Qref_Definition": q, "Method": a["Method"], "Fold_ID": a["Fold_ID"],
                                  "Train_Folds_Clean": bool((s.Fold_ID.to_numpy()[rows] != a["Fold_ID"]).all()),
                                  "Imputer_Recomputed_Match": _sha(med) == a["Imputer_Median_SHA256"],
                                  "NegControl_All_Rows_Match": neg_match,
                                  "Replacement_Mean_Control_Differs": mean_differs if neg_match else np.nan,
                                  "NegControl_Pass_Original": not neg_match,
                                  "NegControl_Pass_Amended": (not neg_match) or mean_differs,
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
    est, audit, traj_eq = pd.DataFrame(est), pd.DataFrame(audit), pd.DataFrame(traj_eq)
    _write(est, staging / "method_estimability.csv")
    _write(audit, staging / "preprocessing_audit.csv")
    _write(traj_eq, staging / "check2_traj_equivalence.csv")
    dsha = pd.DataFrame(design_sha)
    check4 = {}
    for d in datasets:
        ad, td = audit[audit.Dataset.eq(d)], traj_eq[traj_eq.Dataset.eq(d)]
        per_ds[d]["2"] = per_ds[d]["2"] and bool(td.Estimable_Equal.all() and (td.Max_Rel_Diff <= 1e-9).all())
        common = {"fold_cells_disjoint": bool(ad.Train_Folds_Clean.all()), "max_raw_cycle_used_le_t0_3": bool(max_used[d] <= t0[d][2]),
                  "design_has_no_qref_column": not [c for c in design_cols[d] if "qref" in c.lower()],
                  "non_margin_design_identical_across_qref": bool(dsha[dsha.Dataset.eq(d)].Non_Margin_Rows_Match_Shared_Features.all()),
                  "fold_statistics_from_training_rows_only": bool(ad.Imputer_Recomputed_Match.all())}
        check4[d] = {**common, "negative_control_original_failures": int((~ad.NegControl_Pass_Original).sum()),
                     "negative_control_amended_failures": int((~ad.NegControl_Pass_Amended).sum()),
                     "folds_using_replacement_mean_control": int(ad.NegControl_All_Rows_Match.sum())}
        per_ds[d]["4_original"] = all(common.values()) and bool(ad.NegControl_Pass_Original.all())
        per_ds[d]["4_amended"] = all(common.values()) and bool(ad.NegControl_Pass_Amended.all())
    checks[2]["Detail"] += f"; TRAJ max rel diff {traj_eq.Max_Rel_Diff.max():.3g}"
    checks[2]["Pass"] = all(per_ds[d]["2"] for d in datasets)
    checks.append({"Check": "4_NO_LEAKAGE_AMENDED", "Detail": json.dumps({"by_dataset": check4, "shared_feature_sha256": feat_sha}),
                   "Pass": all(per_ds[d]["4_amended"] for d in datasets)})
    checks.append({"Check": "4_NO_LEAKAGE_ORIGINAL", "Detail": "negative control on medians only (original section 7)",
                   "Pass": all(per_ds[d]["4_original"] for d in datasets)})
    for c in checks[:4]:
        c["Pass"] = all(per_ds[d][c["Check"][0]] for d in datasets)
    ok_main = [d for d in datasets if all(per_ds[d][k] for k in ("0", "1", "2", "3", "4_amended"))]
    ok_secondary = [d for d in datasets if all(per_ds[d][k] for k in ("0", "1", "2", "3", "4_original"))]
    _write(pd.DataFrame([{"Dataset": d, **{f"Check_{k}": v for k, v in per_ds[d].items()}, "In_Main_Verdict": d in ok_main,
                          "In_Secondary_Verdict": d in ok_secondary} for d in datasets]), staging / "dataset_checks.csv")
    _log(log, "checks per dataset evaluated; main datasets [" + ",".join(ok_main) + "]; secondary datasets ["
         + ",".join(ok_secondary) + "]")

    if dev:  # Development run: stop before any bootstrap / comparison.
        _write(pd.DataFrame(checks), staging / "sanity_checks.csv")
        summary = predictions.groupby(["Dataset"]).agg(Rows=("Cell_ID", "size")).reset_index()
        _write(summary, staging / "dev_prediction_rows.csv")
        (staging / "processing_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
        staging.replace(target)
        return target
    _write(predictions, staging / "predictions.csv")

    ci_rows, e1t_rows, sec_rows, draw_hash = [], [], [], {}
    for d in datasets:
        cells = data[d].cells
        draws, draw_hash[d] = draws_for(len(cells))
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
    check5, base_parts = {}, []
    for d in datasets:
        detail, c5 = baseline_check(ci[ci.Dataset.eq(d)])
        base_parts.append(detail)
        check5[d] = bool(c5["Pass"])
        checks.append({"Check": f"5_BASELINE_WORSE_{d}", "Detail": f"{c5['Clearly_Better_Share']:.3f} of {c5['Settings']} E1/E2 settings",
                       "Pass": c5["Pass"]})
    checks.append({"Check": "6_DETERMINISM", "Detail": "compare two independent output directories", "Pass": None})
    comp = comparison_table(ci, rev, det, tolerance_shares(root))
    result = {"main": decide(comp, check5, ok_main), "secondary": decide(comp, check5, ok_secondary),
              "dataset_checks": per_ds, "main_datasets": ok_main, "secondary_datasets": ok_secondary}
    desc = {f"main_{k}": v for k, v in descriptives(comp[comp.Dataset.isin(ok_main)]).items()}
    desc.update({f"secondary_{k}": v for k, v in descriptives(comp[comp.Dataset.isin(ok_secondary)]).items()})
    desc.update({f"all_datasets_{k}": v for k, v in descriptives(comp).items()})

    _write(ci, staging / "pair_ci.csv")
    _write(comp, staging / "comparisons.csv")
    _write(pd.concat(base_parts, ignore_index=True), staging / "check5_baseline.csv")
    _write(pd.DataFrame(e1t_rows), staging / "descriptive_e1_by_t0.csv")
    _write(pd.DataFrame(sec_rows), staging / "descriptive_e3_one_sided_error.csv")
    for name, frame in desc.items():
        _write(frame, staging / f"{name}.csv")
    _write(pd.DataFrame(checks), staging / "sanity_checks.csv")
    result.update({"t0": t0, "comparison_rows": int(len(comp)), "estimable_comparisons": int(comp.Estimable_Comparison.sum()),
                   "supported_reversals": int((comp.Supported_Reversal & comp.Estimable_Comparison).sum()),
                   "estimable_intervals": int(ci.Estimable.sum()), "not_estimable_intervals": int((~ci.Estimable).sum()),
                   "not_estimable_method_settings": int((~est.Estimable).sum())})
    (staging / "decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if {f: _file_sha(root / f) for f in INPUT_FILES} != before:
        raise PilotError("INPUT_FILES_CHANGED_DURING_RUN")
    import lifelines
    import scipy
    import sklearn

    info = {"pipeline": PIPELINE_VERSION, "protocol_commit": PROTOCOL_COMMIT, "tju_freeze_commit": TJU_FREEZE_COMMIT,
            "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "lifelines": lifelines.__version__,
            "thetas": list(thetas), "datasets": list(datasets), "bootstrap_draw_sha256": draw_hash, "input_sha256": before}
    (staging / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    _log(log, "formal run complete")
    (staging / "processing_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    staging.replace(target)
    return target
