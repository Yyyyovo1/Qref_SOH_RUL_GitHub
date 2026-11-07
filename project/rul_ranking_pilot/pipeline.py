"""End-to-end run of the preregistered RUL ranking-sensitivity pilot."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CAL_CONDITIONS,
    DATASETS,
    METHODS_BY_CAL,
    NOMINAL_CAPACITY_AH,
    OUTPUT_DIR_NAME,
    PIPELINE_VERSION,
    PROTOCOL_COMMIT,
    PROTOCOL_T0,
    QREFS,
    THETAS,
)
from .evaluation import (
    baseline_check,
    check_draw_hash,
    decide,
    draws_for,
    error_vectors,
    find_reversals,
    pair_intervals,
    strong_rule,
    theta_grid_check,
)
from .features import audit_nocal, cal_features, capacity_summary, frame_sha256, nocal_features
from .labels import (
    PilotError,
    build_labels,
    check_frozen_labels,
    check_monotonicity,
    check_physical_equivalence,
    compute_t0,
    detect_all,
    load_inputs,
)
from .methods import _sha, run_setting

INPUT_FILES = (
    "outputs/xjtu_55_final_v2/xjtu_55_soh_labels_long.csv",
    "outputs/xjtu_55_final_v2/xjtu_55_qref_summary.csv",
    "outputs/hust_77_final_v1/hust_77_primary_capacity.csv",
    "outputs/hust_77_final_v1/hust_77_qref_summary.csv",
    "outputs/soh_feature_extraction_v1/feature_matrix_xjtu_v1.csv",
    "outputs/soh_feature_extraction_v1/feature_matrix_hust_v1.csv",
    "outputs/soh_ml_design_v1/ml_cell_fold_assignments_xjtu.csv",
    "outputs/soh_ml_design_v1/ml_cell_fold_assignments_hust.csv",
    "outputs/eol_threshold_robustness_v1/eol_threshold_status_summary.csv",
)


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _log(lines: list[str], text: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    lines.append(f"[{stamp}] {text}")
    print(lines[-1], flush=True)


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def run_pilot(root: Path, output_dir: Path | None = None, thetas: tuple[float, ...] = THETAS) -> Path:
    root = Path(root).resolve()
    target = Path(output_dir).resolve() if output_dir else root / "outputs" / OUTPUT_DIR_NAME
    if target.exists():
        raise PilotError(f"OUTPUT_DIRECTORY_EXISTS_REFUSING_TO_OVERWRITE:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}_staging_", dir=target.parent))
    log: list[str] = []
    theta_grid_check()
    inputs_before = {f: _file_sha(root / f) for f in INPUT_FILES}
    _log(log, f"{PIPELINE_VERSION}; protocol commit {PROTOCOL_COMMIT}; thetas={len(thetas)}; staging={staging}")

    data = load_inputs(root)
    detections = detect_all(data)
    t0_table = compute_t0(data, detections)
    t0_table["Protocol_Drafting_Value"] = t0_table.Dataset.map(PROTOCOL_T0)
    t0_table["Matches_Protocol"] = t0_table.t0_raw_cycle.eq(t0_table.Protocol_Drafting_Value)
    t0 = dict(zip(t0_table.Dataset, t0_table.t0_raw_cycle.astype(int)))
    _write(t0_table, staging / "t0.csv")
    _log(log, "t0: " + json.dumps({d: int(v) for d, v in t0.items()}))
    labels = build_labels(data, detections, t0)
    _write(labels, staging / "labels.csv")

    check1 = check_frozen_labels(root, labels)
    check2 = check_physical_equivalence(data, labels)
    check3 = check_monotonicity(labels)
    _write(check1, staging / "check1_frozen_labels.csv")
    summary = [
        {"Check": "1_FROZEN_LABELS", "Detail": f"{int(check1.Pass.sum())}/{len(check1)} rows match", "Pass": bool(check1.Pass.all())},
        {"Check": "2_PHYSICAL_EQUIVALENCE_LABELS",
         "Detail": f"{int(check2.Mismatches.iat[0])} mismatches in {int(check2.Comparisons.iat[0])}; "
                   f"max threshold diff {check2.Max_Abs_Threshold_Difference_Ah.iat[0]:.3g} Ah",
         "Pass": bool(check2.Pass.iat[0])},
        {"Check": "3_MONOTONICITY", "Detail": f"{int(check3.Violations.iat[0])} violations in {int(check3.Cell_Qref_Series.iat[0])} series",
         "Pass": bool(check3.Pass.iat[0])},
    ]
    if not all(s["Pass"] for s in summary):
        _write(pd.DataFrame(summary), staging / "sanity_checks.csv")
        raise PilotError(f"LABEL_CHECKS_FAILED; partial outputs in {staging}")
    _log(log, "checks 1-3 passed")

    # Features (section 4.2) and check-4 feature audits.
    nocal, capsum, max_raw = {}, {}, {}
    for d in DATASETS:
        nocal[d], raw_a = nocal_features(data[d], t0[d])
        capsum[d], raw_b = capacity_summary(data[d], t0[d])
        audit_nocal(nocal[d])
        max_raw[d] = max(raw_a, raw_b)
        _write(nocal[d].reset_index(), staging / f"features_nocal_{d.lower()}.csv")
        _write(capsum[d].reset_index(), staging / f"capacity_t0_{d.lower()}.csv")

    pred_frames, estimability, audit_rows, traj_equiv, nocal_hashes = [], [], [], [], {d: set() for d in DATASETS}
    for d in DATASETS:
        for theta in thetas:
            for q in QREFS:
                lab = labels[(labels.Dataset == d) & (labels.Theta == theta) & (labels.Qref_Definition == q)]
                qref = data[d].qref[q]
                thresholds = {c: float(theta) * float(qref[c]) for c in data[d].cells}
                for cal in CAL_CONDITIONS:
                    design = nocal[d] if cal == "NOCAL" else cal_features(nocal[d], capsum[d], qref, theta)
                    if cal == "NOCAL":
                        nocal_hashes[d].add(frame_sha256(design))
                    res = run_setting(data[d], lab, design, cal, METHODS_BY_CAL[cal], t0[d], thresholds)
                    p = res.predictions.merge(lab[["Cell_ID", "Status", "Eligible", "Fold_ID", "RUL", "RUL_Lower", "RUL_Upper"]], on="Cell_ID")
                    p.insert(0, "Cal", cal); p.insert(0, "Qref_Definition", q); p.insert(0, "Theta", theta); p.insert(0, "Dataset", d)
                    pred_frames.append(p)
                    for m in METHODS_BY_CAL[cal]:
                        estimability.append({"Dataset": d, "Theta": theta, "Qref_Definition": q, "Cal": cal, "Method": m,
                                             "Estimable": res.estimable[m], "Reason": res.reasons[m],
                                             "TRAJ_Fallbacks": res.traj_fallbacks if m == "TRAJ" else np.nan,
                                             "TRAJ_Missing": res.traj_missing if m == "TRAJ" else np.nan})
                    # Check 4: recompute fold statistics from the training cells only; all-cell stats must differ.
                    x = design.to_numpy(float)
                    index = {c: i for i, c in enumerate(design.index)}
                    for a in res.audit:
                        rows = [index[c] for c in a["Train_Cells"].split("|")]
                        med = np.nanmedian(x[rows], axis=0)
                        med = np.where(np.isnan(med), 0.0, med)
                        all_med = np.nanmedian(x, axis=0)
                        all_med = np.where(np.isnan(all_med), 0.0, all_med)
                        overlap = set(a["Train_Cells"].split("|")) & set(a["Test_Cells"].split("|"))
                        audit_rows.append({"Dataset": d, "Theta": theta, "Qref_Definition": q, "Cal": cal, **{k: v for k, v in a.items() if k not in ("Train_Cells", "Test_Cells")},
                                           "Train_Cell_Count": len(rows), "Train_Test_Overlap": len(overlap),
                                           "Imputer_Recomputed_Match": _sha(med) == a["Imputer_Median_SHA256"],
                                           "NegControl_AllCells_Match": _sha(all_med) == a["Imputer_Median_SHA256"]})
                    # Check 2 (TRAJ): the same Ah threshold written as theta' x nominal must give the same prediction.
                    if cal == "CAL" and q != "nominal":
                        nominal = NOMINAL_CAPACITY_AH[d]
                        alt = {c: (float(theta) * float(qref[c]) / nominal) * nominal for c in data[d].cells}
                        res2 = run_setting(data[d], lab, design, cal, ("TRAJ",), t0[d], alt)
                        a1, a2 = res.predictions["Pred_TRAJ"].to_numpy(float), res2.predictions["Pred_TRAJ"].to_numpy(float)
                        both = ~np.isnan(a1) & ~np.isnan(a2)
                        traj_equiv.append({"Dataset": d, "Theta": theta, "Qref_Definition": q,
                                           "NaN_Pattern_Equal": bool((np.isnan(a1) == np.isnan(a2)).all()),
                                           "Bitwise_Equal_Count": int((a1[both] == a2[both]).sum()), "Compared": int(both.sum()),
                                           "Max_Abs_Diff": float(np.max(np.abs(a1[both] - a2[both]))) if both.any() else 0.0,
                                           "Max_Rel_Diff": float(np.max(np.abs(a1[both] - a2[both]) / np.maximum(1.0, np.abs(a1[both])))) if both.any() else 0.0})
            _log(log, f"{d} theta={theta:.2f} trained")

    predictions = pd.concat(pred_frames, ignore_index=True)
    _write(predictions, staging / "predictions.csv")
    estimability = pd.DataFrame(estimability)
    _write(estimability, staging / "method_estimability.csv")
    audit = pd.DataFrame(audit_rows)
    _write(audit, staging / "preprocessing_audit.csv")
    traj_equiv = pd.DataFrame(traj_equiv)
    _write(traj_equiv, staging / "check2_traj_equivalence.csv")

    folds_ok = all(not (set(c for c, f in data[d].folds.items() if f == k) & set(c for c, f in data[d].folds.items() if f != k))
                   for d in DATASETS for k in range(1, 6))
    check4 = {
        "fold_cells_disjoint": folds_ok and int(audit.Train_Test_Overlap.sum()) == 0,
        "max_raw_cycle_used_le_t0": all(max_raw[d] <= t0[d] for d in DATASETS),
        "nocal_matrix_identical_across_qref_theta": all(len(nocal_hashes[d]) == 1 for d in DATASETS),
        "fold_statistics_from_training_cells_only": bool(audit.Imputer_Recomputed_Match.all()),
        "negative_control_all_cells_differs": bool((~audit.NegControl_AllCells_Match).all()),
    }
    traj_ok = bool(traj_equiv.NaN_Pattern_Equal.all() and (traj_equiv.Max_Rel_Diff <= 1e-9).all())
    summary[1]["Detail"] += (f"; TRAJ {int(traj_equiv.Bitwise_Equal_Count.sum())}/{int(traj_equiv.Compared.sum())} bitwise equal, "
                             f"max rel diff {traj_equiv.Max_Rel_Diff.max():.3g}")
    summary[1]["Pass"] = summary[1]["Pass"] and traj_ok
    summary.append({"Check": "4_NO_LEAKAGE", "Detail": json.dumps(check4), "Pass": all(check4.values())})
    if not (summary[1]["Pass"] and summary[3]["Pass"]):
        _write(pd.DataFrame(summary), staging / "sanity_checks.csv")
        raise PilotError(f"CHECK_2_OR_4_FAILED; partial outputs in {staging}")
    _log(log, "checks 2 (TRAJ) and 4 passed")

    # Evaluation, bootstrap and decision (sections 5-7).
    cells = {d: data[d].cells for d in DATASETS}
    draws, draw_hash = {}, {}
    for d in DATASETS:
        draws[d], draw_hash[d] = draws_for(len(cells[d]))
        check_draw_hash(d, draw_hash[d])
    errors = error_vectors(labels, predictions, cells)
    ci = pair_intervals(errors, draws)
    _write(ci, staging / "pair_ci.csv")
    reversals = find_reversals(ci)
    _write(reversals, staging / "reversals.csv")
    strong = strong_rule(reversals)
    _write(strong, staging / "strong_rule.csv")
    base_detail, check5 = baseline_check(ci)
    _write(base_detail, staging / "check5_baseline.csv")
    summary.append({"Check": "5_BASELINE_WORSE", "Detail": f"{check5['Clearly_Better_Share']:.3f} of {check5['Settings']} E1/E2 settings",
                    "Pass": check5["Pass"]})
    summary.append({"Check": "6_DETERMINISM", "Detail": "run compare_runs on two independent output directories", "Pass": None})
    _write(pd.DataFrame(summary), staging / "sanity_checks.csv")
    decision = decide(strong, check5)

    supported = reversals[reversals.Supported_Reversal]
    result = {
        "decision": decision,
        "t0": {d: int(v) for d, v in t0.items()},
        "comparisons": int(len(reversals)),
        "supported_reversals": int(len(supported)),
        "supported_reversals_by_eval_set": {k: int(v) for k, v in supported.Eval_Set.value_counts().sort_index().items()},
        "max_S_size": int(strong.S_Size.max()) if len(strong) else 0,
        "not_estimable_pair_intervals": int((~ci.Estimable).sum()),
        "estimable_pair_intervals": int(ci.Estimable.sum()),
        "not_estimable_method_settings": int((~estimability.Estimable).sum()),
        "check5": check5,
    }
    (staging / "decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    inputs_after = {f: _file_sha(root / f) for f in INPUT_FILES}
    if inputs_after != inputs_before:
        raise PilotError("INPUT_FILES_CHANGED_DURING_RUN")
    import lifelines
    import scipy
    import sklearn

    info ={"pipeline": PIPELINE_VERSION, "protocol_commit": PROTOCOL_COMMIT, "python": sys.version.split()[0],
            "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "lifelines": lifelines.__version__, "thetas": list(thetas),
            "bootstrap_draw_sha256": draw_hash, "input_sha256": inputs_before}
    (staging / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    _log(log, f"decision: {decision}")
    (staging / "processing_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    staging.replace(target)
    return target


def compare_runs(a: Path, b: Path) -> pd.DataFrame:
    """Check 6: every output except the timing-bearing log and run info must be byte-identical."""
    skip = {"processing_log.txt", "run_info.json"}
    names = sorted({p.name for p in Path(a).iterdir()} | {p.name for p in Path(b).iterdir()})
    rows = []
    for n in names:
        if n in skip:
            continue
        pa, pb = Path(a) / n, Path(b) / n
        same = pa.exists() and pb.exists() and _file_sha(pa) == _file_sha(pb)
        rows.append({"File": n, "Identical": same})
    return pd.DataFrame(rows)
