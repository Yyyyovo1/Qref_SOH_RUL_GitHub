"""End-to-end v4 run (PROTOCOL_v4.md + AMENDMENT_01.md).

`dev=True` (AMENDMENT_01 section 11): archived-file check, checks 0-4, check S, the experiment 1 leakage check and
estimability; no evaluation function can be called (blinding guard) and every written column is screened.
Formal run: the same, then (inside `allow_evaluation`) check 5 and its reproduction of v2/v3, all comparisons,
experiment 2, H1-H3. Check 6 compares two formal output directories (`compare_runs`).
"""

from __future__ import annotations

import json
import platform
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from rul_pilot_v2.pipeline import INPUT_FILES as V2_INPUT_FILES
from rul_pilot_v3_tju.pipeline import INPUT_FILES as V3_INPUT_FILES
from rul_ranking_pilot.labels import PilotError
from rul_ranking_pilot.pipeline import _file_sha, _log, _write

from . import exp1 as E1
from . import exp1_labels as L1
from . import exp2 as E2
from . import exp3 as E3
from . import severson as S
from . import verdicts as V
from .blinding import allow_evaluation, check_columns
from .config import (
    AMENDMENT_01_COMMIT,
    CHECK5_PAIRS,
    DESCRIPTIVE_EXCLUDED_CELL,
    EXP1_C_THETA,
    EXP1_THETAS,
    EXP3_C_THETA,
    EXP3_DATASETS,
    EXP3_LABEL,
    GB_PARAMS,
    GB_PLAN,
    MIT_FOLDS,
    OUTPUT_ROOT,
    PIPELINE_VERSION,
    PROTOCOL_COMMIT,
    QREFS,
    RULES,
    SEVERSON_FEATURES_REF,
    TRAPD_LABELS,
    V2_CHECK5,
    V3_CHECK5,
)
from .prereg import check_archived

INPUT_FILES = tuple(dict.fromkeys((*V2_INPUT_FILES, *V3_INPUT_FILES, TRAPD_LABELS, GB_PLAN, V2_CHECK5, V3_CHECK5, MIT_FOLDS,
                                   SEVERSON_FEATURES_REF, "outputs/mit_124_final_v1/mit_cell_inventory.csv")))

# Files a development run may write (anything else raises).
DEV_FILES = frozenset({
    "prereg_hash_check.csv", "sanity_checks.csv", "dataset_checks.csv", "dev_prediction_rows.csv",
    "exp1_features.csv", "exp1_check0_cycles.csv", "exp1_checkS_features.csv", "exp1_checkS_qdlin.csv",
    "exp1_checkS_cycle_life.csv", "exp1_labels.csv", "exp1_labels_detail.csv", "exp1_label_shares.csv",
    "exp1_common_windows.csv", "exp1_cell_sets.csv", "exp1_estimability.csv", "exp1_leakage_audit.csv",
    "exp1_fit_warnings.csv",
    "exp3_labels.csv", "exp3_samples.csv", "exp3_features_hust.csv", "exp3_features_mit.csv", "exp3_features_nca.csv",
    "exp3_windows.csv", "exp3_method_estimability.csv", "exp3_preprocessing_audit.csv", "exp3_check2_traj_equivalence.csv",
})


class Writer:
    def __init__(self, folder: Path, dev: bool):
        self.folder, self.dev = folder, dev

    def csv(self, frame: pd.DataFrame, name: str) -> None:
        if self.dev:
            if name not in DEV_FILES and not re.fullmatch(r"exp3_features_[a-z0-9]+\.csv", name):
                raise PilotError(f"DEV_RUN_FILE_NOT_ALLOWED:{name}")
            check_columns(frame.columns, name)
        _write(frame, self.folder / name)

    def json(self, obj, name: str) -> None:
        if self.dev and name not in {"run_info.json"}:
            raise PilotError(f"DEV_RUN_FILE_NOT_ALLOWED:{name}")
        (self.folder / name).write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def _check(rows: list, experiment: str, dataset: str, check: str, ok, detail) -> None:
    rows.append({"Experiment": experiment, "Dataset": dataset, "Check": check, "Pass": ok,
                 "Detail": json.dumps(detail, default=_json_default) if not isinstance(detail, str) else detail})


# ================================================================ experiment 1 inputs (no evaluation)
def prepare_exp1(root: Path, thetas, checks: list, log) -> dict:
    mit = L1.load_mit(root)
    det, tol, labels = L1.build_labels(mit, thetas)
    c1, c2, c3 = L1.check1(root, det, tol), L1.check2(mit, det, tol), L1.check3(det, tol)
    _check(checks, "1", "MIT", "1_LABELS", c1["Pass"], c1)
    _check(checks, "1", "MIT", "2_PHYSICAL_EQUIVALENCE", c2["Pass"], c2)
    _check(checks, "1", "MIT", "3_MONOTONICITY", c3["Pass"], c3)
    log("exp1 labels and checks 1-3 done")
    prep = S.load_prep_module(root)
    feat, arrays, check0 = S.compute_features(root, prep)
    cs = S.run_check_s(root, prep, feat, arrays)
    _check(checks, "1", "MIT", "0_FEATURES_USE_CYCLES_1_TO_100", bool(check0.Pass.all()),
           {"cells": int(len(check0)), "failures": int((~check0.Pass).sum()), "max_row_accessed": int(check0.Max_Cycle_Row_Accessed.max())})
    _check(checks, "1", "MIT", "S_SEVERSON_FEATURES", cs["Pass"],
           {"item_1_features": cs["pass_1"], "item_2_qdlin_round5": cs["pass_2"], "item_3_cycle_life": cs["pass_3"]})
    log("exp1 features, check 0 and check S done")
    cells = sorted(mit.cells)
    sh = L1.shares(labels, len(cells))
    win = L1.windows(sh, thetas)
    full_grid = tuple(thetas) == tuple(EXP1_THETAS)
    wcmp = L1.compare_windows_with_prep(root, win, sh) if full_grid else {"Pass": None, "skipped": "theta subset"}
    _check(checks, "1", "MIT", "WINDOWS_EQUAL_PREP", wcmp["Pass"], wcmp)
    sets = L1.cell_sets(labels, win, cells)
    models = E1.EXP1_MODELS
    from .exp1_models import model_features

    mf = model_features(prep)
    fidx = feat.set_index("Cell_ID")
    design = {m: {c: fidx.loc[c, mf[m]].to_numpy(float) for c in cells} for m in models}
    folds = pd.read_csv(root / MIT_FOLDS, encoding="utf-8-sig")
    inp = E1.Exp1Inputs(cells=cells, design=design, eol=L1.eol_lookup(labels), sets=sets,
                        folds=dict(zip(folds.Cell_ID.astype(str), folds.Fold_ID.astype(int))),
                        split=dict(zip(feat.Cell_ID, feat.Severson_Split)))
    fits = E1.fit_p_settings(inp, log=log)
    E1.fit_delta_settings(inp, fits)
    audit = E1.audit_table(fits)
    enet = audit[audit.Model.ne("variance")]
    leak = {"train_test_disjoint": bool(audit.Train_Test_Disjoint.all()),
            "scaler_from_training_cells_only": bool(enet.Scaler_Recomputed_Match.all()),
            "inner_indices_within_training_fold": bool(enet.Inner_Indices_Within_Train.all()),
            "inner_validation_partitions_training_fold": bool(enet.Inner_Validation_Partitions_Train.all()),
            "inner_splits_reproducible": bool(enet.Inner_Splits_Reproducible.all()),
            "negative_control_all_cells_differs": bool(enet.NegControl_Pass.all()),
            "design_has_no_label_or_qref_column": not [c for m in models for c in mf[m] if any(t in c.lower() for t in ("qref", "eol", "life"))],
            "design_identical_across_definitions": True,  # one feature table per cell, independent of (rule, Qref, theta)
            "folds_cover_all_cells": set(inp.folds) >= set(cells)}
    _check(checks, "1", "MIT", "4_NO_LEAKAGE_EXP1", all(leak.values()), leak)
    est = E1.estimability_table(fits)
    log("exp1 fits done")
    return {"mit": mit, "det": det, "tol": tol, "labels": labels, "feat": feat, "check0": check0, "check_s": cs,
            "shares": sh, "windows": win, "sets": sets, "inp": inp, "fits": fits, "audit": audit, "estimability": est,
            "prep": prep}


def write_exp1_dev_safe(w: Writer, x: dict) -> None:
    w.csv(x["feat"], "exp1_features.csv")
    w.csv(x["check0"], "exp1_check0_cycles.csv")
    w.csv(x["check_s"]["features"], "exp1_checkS_features.csv")
    w.csv(x["check_s"]["qdlin"], "exp1_checkS_qdlin.csv")
    w.csv(x["check_s"]["cycle_life"], "exp1_checkS_cycle_life.csv")
    w.csv(x["labels"], "exp1_labels.csv")
    w.csv(x["tol"].drop(columns=["EOL_Difference_Tolerance_Minus_Strict"]), "exp1_labels_detail.csv")
    w.csv(x["shares"], "exp1_label_shares.csv")
    w.csv(x["windows"], "exp1_common_windows.csv")
    w.csv(x["sets"], "exp1_cell_sets.csv")
    w.csv(x["estimability"], "exp1_estimability.csv")
    w.csv(x["audit"], "exp1_leakage_audit.csv")
    w.csv(E1.warning_table(x["fits"]), "exp1_fit_warnings.csv")


# ================================================================ experiment 3 inputs (no evaluation)
def prepare_exp3(root: Path, datasets, thetas, checks: list, log) -> dict:
    gb_plan = E3.gb_params_from_plan(root)
    _check(checks, "3", "all", "GB_PARAMS_EQUAL_SOH_DESIGN_PLAN", gb_plan == GB_PARAMS, {"plan": gb_plan, "config": GB_PARAMS})
    if gb_plan != GB_PARAMS:
        raise PilotError("GB_PARAMS_DIFFER_FROM_SOH_DESIGN_PLAN")
    data = E3.load_inputs(root, datasets)
    out, wins = {}, []
    for name in datasets:
        ds = E3.fit_dataset(root, data, name, GB_PARAMS, thetas=thetas, log=log)
        for k in ("0", "1", "2", "3", "4_amended", "4_original"):
            _check(checks, "3", name, f"{k}", ds.checks[k], ds.checks["detail"].get(k[0], {}) if k[0] != "4" else ds.checks["detail"]["4"])
        win = E3.windows(name, ds.det)
        wins.append(win)
        out[name] = ds
    win = pd.concat(wins, ignore_index=True)
    full = thetas is None
    wcmp = E3.compare_windows_with_prep(root, win) if full else {"Pass": None, "skipped": "theta subset"}
    _check(checks, "3", "all", "WINDOWS_EQUAL_PREP", wcmp["Pass"], wcmp)
    return {"datasets": out, "windows": win}


def write_exp3_dev_safe(w: Writer, x: dict) -> None:
    dss = x["datasets"].values()
    w.csv(pd.concat([d.det for d in dss], ignore_index=True), "exp3_labels.csv")
    w.csv(pd.concat([d.samples for d in dss], ignore_index=True), "exp3_samples.csv")
    for d in dss:
        w.csv(d.features, f"exp3_features_{d.name.lower()}.csv")
    w.csv(x["windows"], "exp3_windows.csv")
    w.csv(pd.concat([d.estimability for d in dss], ignore_index=True), "exp3_method_estimability.csv")
    w.csv(pd.concat([d.audit for d in dss], ignore_index=True), "exp3_preprocessing_audit.csv")
    w.csv(pd.concat([d.traj_eq for d in dss], ignore_index=True), "exp3_check2_traj_equivalence.csv")


# ================================================================ run
def run(root: Path, output_dir: Path | None = None, dev: bool = False, experiments=("1", "3"), datasets=EXP3_DATASETS,
        exp1_thetas=EXP1_THETAS, exp3_thetas=None) -> Path:
    root = Path(root).resolve()
    repo = root.parent
    if not dev and (tuple(experiments) != ("1", "3") or tuple(datasets) != EXP3_DATASETS or tuple(exp1_thetas) != EXP1_THETAS
                    or exp3_thetas is not None):
        raise PilotError("SUBSETS_ARE_ALLOWED_ONLY_IN_DEVELOPMENT_RUNS")
    target = Path(output_dir).resolve() if output_dir else (root / OUTPUT_ROOT / ("dev_run" if dev else "formal_run")).resolve()
    if target.exists():
        raise PilotError(f"OUTPUT_DIRECTORY_EXISTS_REFUSING_TO_OVERWRITE:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}_staging_", dir=target.parent))
    w = Writer(staging, dev)
    lines: list[str] = []
    log = lambda text: _log(lines, text)  # noqa: E731
    log(f"{PIPELINE_VERSION}; protocol {PROTOCOL_COMMIT}; amendment 01 {AMENDMENT_01_COMMIT}; dev={dev}; experiments={list(experiments)}")

    w.csv(check_archived(repo), "prereg_hash_check.csv")
    log("archived files verified")
    before = {f: _file_sha(root / f) for f in INPUT_FILES}
    checks: list[dict] = []
    x1 = prepare_exp1(root, exp1_thetas, checks, log) if "1" in experiments else None
    x3 = prepare_exp3(root, datasets, exp3_thetas, checks, log) if "3" in experiments else None

    exp1_ok, exp1_reason = False, "EXPERIMENT_1_NOT_RUN"
    if x1 is not None:
        c = {r["Check"]: r["Pass"] for r in checks if r["Experiment"] == "1"}
        failed = [k for k in ("0_FEATURES_USE_CYCLES_1_TO_100", "1_LABELS", "2_PHYSICAL_EQUIVALENCE", "3_MONOTONICITY",
                              "4_NO_LEAKAGE_EXP1", "WINDOWS_EQUAL_PREP") if c.get(k) is False]
        exp1_ok = c.get("S_SEVERSON_FEATURES") is True and not failed
        exp1_reason = "" if exp1_ok else ("CHECK_S_FAILED" if c.get("S_SEVERSON_FEATURES") is not True else "CHECKS_FAILED:" + ",".join(failed))
    ok04 = []
    if x3 is not None:
        for name, ds in x3["datasets"].items():
            if all(ds.checks[k] for k in ("0", "1", "2", "3", "4_amended")):
                ok04.append(name)
    w.csv(pd.DataFrame([{"Experiment": "1", "Dataset": "MIT", "Passes_Checks": exp1_ok, "Reason": exp1_reason}]
                       + [{"Experiment": "3", "Dataset": n, "Passes_Checks": n in ok04,
                           **{f"Check_{k}": v for k, v in ds.checks.items() if k != "detail"}}
                          for n, ds in (x3["datasets"].items() if x3 else [])]), "dataset_checks.csv")
    if x1 is not None:
        write_exp1_dev_safe(w, x1)
    if x3 is not None:
        write_exp3_dev_safe(w, x3)

    if dev:
        rows = [{"Experiment": "3", "Dataset": n, "Fitted_Rows": int(len(ds.predictions))} for n, ds in (x3["datasets"].items() if x3 else [])]
        rows += [{"Experiment": "1", "Dataset": "MIT", "Fitted_Rows": int(sum(len(r.predictions) for r in x1["fits"].results.values()))}] if x1 else []
        w.csv(pd.DataFrame(rows, columns=["Experiment", "Dataset", "Fitted_Rows"]), "dev_prediction_rows.csv")
        w.csv(pd.DataFrame(checks), "sanity_checks.csv")
        _finish(w, staging, target, root, before, lines, log, dev, experiments, datasets, {})
        return target

    window_mismatch = [f"exp{r['Experiment']}" for r in checks if r["Check"] == "WINDOWS_EQUAL_PREP" and r["Pass"] is False]
    if window_mismatch:  # implementation notes 2.5 / 3.6: the recomputed windows must equal the archived prep windows
        log("common windows differ from the archived prep windows: stopping before any evaluation")
        decision = {"stopped": "COMMON_WINDOWS_DIFFER_FROM_PREP", "experiments": window_mismatch}
    else:
        with allow_evaluation():
            decision = _formal(w, root, x1, x3, exp1_ok, exp1_reason, ok04, checks, log)
    w.csv(pd.DataFrame(checks), "sanity_checks.csv")
    w.json(decision, "decision.json")
    _finish(w, staging, target, root, before, lines, log, dev, experiments, datasets, decision)
    return target


def _finish(w, staging, target, root, before, lines, log, dev, experiments, datasets, decision) -> None:
    if {f: _file_sha(root / f) for f in INPUT_FILES} != before:
        raise PilotError("INPUT_FILES_CHANGED_DURING_RUN")
    import h5py
    import lifelines
    import scipy
    import sklearn

    info = {"pipeline": PIPELINE_VERSION, "protocol_commit": PROTOCOL_COMMIT, "amendment_01_commit": AMENDMENT_01_COMMIT,
            "dev": dev, "experiments": list(experiments), "datasets": list(datasets),
            "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "lifelines": lifelines.__version__,
            "h5py": h5py.__version__, "input_sha256": before}
    w.json(info, "run_info.json")
    log("run complete")
    (staging / "processing_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    staging.replace(target)


def _formal(w: Writer, root: Path, x1, x3, exp1_ok: bool, exp1_reason: str, ok04: list, checks: list, log) -> dict:
    from rul_pilot_v2.evaluation import draws_for

    decision: dict = {"experiment_1_participates": exp1_ok, "experiment_1_reason": exp1_reason}
    # ---------------- experiment 3, step 1: check 5 and its reproduction (amendment 01 section 6)
    ci3_parts, e1t_parts, sec_parts, base_parts, vec3, draw_sha = [], [], [], [], {}, {}
    c5 = {}
    if x3 is not None:
        for name, ds in x3["datasets"].items():
            ci, e1t, sec, _, _, draw_sha[name] = E3.evaluate_dataset(ds, E3.GRID[name], list(CHECK5_PAIRS), keep_vectors_for=(), log=log)
            ci3_parts += ci
            e1t_parts += e1t
            sec_parts += sec
            detail, rep = E3.check5(name, pd.DataFrame(ci), root)
            c5[name] = (detail, rep)
            _check(checks, "3", name, "5_BASELINE_WORSE", rep["Pass"], rep)
            _check(checks, "3", name, "5_REPRODUCES_V2_V3", rep["Reproduced"], rep)
        w.csv(pd.concat([d for d, _ in c5.values()], ignore_index=True), "exp3_check5_baseline.csv")
        if not all(rep["Reproduced"] for _, rep in c5.values()):
            log("check 5 does not reproduce v2/v3: stopping before any further comparison")
            decision["stopped"] = "CHECK_5_NOT_REPRODUCED"
            decision["check5"] = {n: rep for n, (_, rep) in c5.items()}
            return decision
    # ---------------- experiment 1
    rev1 = ci1 = None
    eq_rows = []
    if x1 is not None:
        inp, fits = x1["inp"], x1["fits"]
        draws, sha1 = draws_for(len(inp.cells))
        ci1, vec1, apes = E1.evaluate_p(inp, fits, draws)
        rev1 = E1.reversals(ci1)
        units = E1.h1_units(ci1)
        base1 = E1.delta_baselines(inp, fits, apes)
        eq1 = E2.exp1_equivalence_and_power(rev1, ci1, vec1, base1)
        eq_rows.append(eq1)
        w.csv(E1.predictions_table(inp, fits), "exp1_predictions.csv")
        w.csv(E1.hyper_table(fits), "exp1_selected_hyperparameters.csv")
        w.csv(ci1, "exp1_pair_ci.csv")
        w.csv(rev1, "exp1_reversals.csv")
        w.csv(units, "exp1_h1_units.csv")
        w.csv(base1, "exp1_delta_baselines.csv")
        w.csv(E1.rmse_table(inp, fits), "exp1_descriptive_rmse.csv")
        w.csv(E1.common_practice(inp, EXP1_C_THETA, RULES, QREFS), "exp1_common_practice_theta080.csv")
        ex = frozenset({DESCRIPTIVE_EXCLUDED_CELL})
        fits_x = E1.fit_p_settings(inp, exclude=ex)
        ci_x, _, _ = E1.evaluate_p(inp, fits_x, draws, exclude=ex)
        w.csv(ci_x, "exp1_descriptive_excl_b1c18_pair_ci.csv")
        w.csv(E1.reversals(ci_x), "exp1_descriptive_excl_b1c18_reversals.csv")
        w.csv(E1.common_practice(inp, EXP1_C_THETA, RULES, QREFS, exclude=ex), "exp1_descriptive_excl_b1c18_common_practice.csv")
        decision["H1"] = V.h1(rev1, units, exp1_ok, exp1_reason)
        decision["exp1_bootstrap_draw_sha256"] = sha1
        log("exp1 evaluated")
    else:
        decision["H1"] = {"Verdict": V.UNTESTABLE, "Reason": exp1_reason}
    # ---------------- experiment 3, step 2: remaining pairs within windows and at theta 0.80
    comp3 = None
    if x3 is not None:
        win = x3["windows"]
        rest = [p for p in E3.PAIRS if p not in CHECK5_PAIRS]
        for name, ds in x3["datasets"].items():
            th = sorted(E3.window_thetas(win[win.Dataset.eq(name)]) | {EXP3_C_THETA})
            th = [t for t in E3.GRID[name] if round(t, 2) in th]
            ci, e1t, sec, vec, base, _ = E3.evaluate_dataset(ds, th, rest, keep_vectors_for=("RF-GB",), log=log)
            ci3_parts += ci
            e1t_parts += e1t
            sec_parts += sec
            base_parts += base
            vec3.update({(k[0], k[1], k[2], k[3], name): v for k, v in vec.items()})
        ci3 = pd.DataFrame(ci3_parts)
        comp3 = E3.comparison_table(ci3, win)
        base3 = pd.DataFrame(base_parts)
        eq3 = E2.exp3_equivalence_and_power(comp3, vec3, base3)
        eq_rows.append(eq3)
        w.csv(x3_predictions(x3), "exp3_predictions.csv")
        w.csv(ci3, "exp3_pair_ci.csv")
        w.csv(comp3, "exp3_comparisons.csv")
        w.csv(base3, "exp3_base_baselines.csv")
        w.csv(pd.DataFrame(e1t_parts), "exp3_descriptive_e1_by_t0.csv")
        w.csv(pd.DataFrame(sec_parts), "exp3_descriptive_e3_one_sided_error.csv")
        w.csv(E3.common_practice(ci3), "exp3_common_practice_theta080.csv")
        decision["exp3_bootstrap_draw_sha256"] = draw_sha
        log("exp3 evaluated")
    # ---------------- experiment 2 and verdicts
    eq = pd.concat(eq_rows, ignore_index=True) if eq_rows else pd.DataFrame()
    power = E2.power_summary(eq) if len(eq) else pd.DataFrame()
    w.csv(eq, "exp2_equivalence_and_power.csv")
    w.csv(power, "exp2_detection_summary.csv")
    agr = []
    if rev1 is not None:
        a = E2.agreement(rev1.assign(Experiment="1", Dataset="MIT", Eval_Set="P_common_cells", Pair_Methods=rev1.Method_Pair),
                         ["Experiment", "Dataset", "Pair_Methods", "Eval_Set"])
        agr.append(a)
        agr.append(E2.agreement(rev1.assign(Experiment="1"), ["Experiment"]))
    if comp3 is not None:
        cw = comp3[comp3.In_Common_Window].assign(Experiment="3", Pair_Methods=comp3.Pair)
        agr.append(E2.agreement(cw, ["Experiment", "Dataset", "Pair_Methods", "Eval_Set"]))
        agr.append(E2.agreement(cw, ["Experiment"]))
    w.csv(pd.concat(agr, ignore_index=True) if agr else pd.DataFrame(), "exp2_supported_agreement.csv")
    eq_for_h2 = eq if len(eq) else pd.DataFrame(columns=["Experiment", "Eval_Set", "Category"])
    decision["H2"] = V.h2(eq_for_h2, exp1_ok)
    if x3 is not None:
        used = [n for n in ok04 if c5[n][1]["Pass"]]
        decision["H3"] = V.h3(comp3, eq if len(eq) else pd.DataFrame(columns=["Experiment", "Dataset", "Eval_Set"]), used)
        decision["check5"] = {EXP3_LABEL[n]: rep for n, (_, rep) in c5.items()}
        decision["exp3_datasets_passing_checks_0_4"] = ok04
    else:
        decision["H3"] = {"Verdict": V.UNTESTABLE, "Reason": "EXPERIMENT_3_NOT_RUN"}
    log("verdicts written")
    return decision


def x3_predictions(x3: dict) -> pd.DataFrame:
    return pd.concat([ds.predictions for ds in x3["datasets"].values()], ignore_index=True)


def compare_outputs(a: Path, b: Path) -> pd.DataFrame:
    from rul_ranking_pilot.pipeline import compare_runs

    return compare_runs(a, b)
