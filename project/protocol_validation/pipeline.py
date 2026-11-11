"""Run the retrospective protocol validation (VALIDATION_PLAN.md); writes only to a new output directory."""

from __future__ import annotations

import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rul_pilot_v2.data import load_all
from rul_ranking_pilot.labels import detect_all
from rul_ranking_pilot.pipeline import _file_sha

from . import OUTPUT_DIR_NAME, PIPELINE_VERSION, PLAN_COMMIT
from .endpoint import eol_difference_summary, setting_summary, tolerance_labels
from .rul import bias_table, comparisons, load_v2, noise_table, rul_settings
from .soh import soh_cell_mae, soh_comparisons, soh_pair_diffs, trap_b

INPUTS = ("outputs/soh_primary_benchmark_v1/soh_primary_predictions.csv",
          "outputs/xjtu_a8_deterministic_reconstruction_v1/xjtu_a8_cell_metrics.csv",
          "outputs/rul_ranking_pilot_v2/samples.csv", "outputs/rul_ranking_pilot_v2/predictions.csv",
          "outputs/rul_ranking_pilot_v2/pair_ci.csv", "outputs/rul_ranking_pilot_v2/labels.csv")
SCOPE_TO_CONVERSION = {"A9_OWN": "OWN", "A9_TRAIN": "TRAIN", "A9_NOM": "NOM"}


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def _close(a, b, tol: float) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tol)))


def check_v2_intervals(root: Path, rul_set: pd.DataFrame) -> dict:
    """Check 1: recomputed E1 / E2 / E3 (C index) intervals equal the committed v2 pair_ci.csv (tolerance 1e-12)."""
    v2 = pd.read_csv(root / "outputs/rul_ranking_pilot_v2/pair_ci.csv", encoding="utf-8-sig", float_precision="round_trip")
    v2["Theta"] = v2.Theta.round(2)
    key = ["Dataset", "Theta", "Qref_Definition", "Eval_Set", "Pair"]
    m = rul_set[rul_set.Eval_Set.isin(["E1", "E2", "E3"])].merge(v2, on=key, suffixes=("", "_v2"))
    est = m.Estimable.astype(bool)
    same_est = bool((est == m.Estimable_v2.astype(bool)).all())
    e = m[est]
    values = all(_close(e[c], e[f"{c}_v2"], 1e-12) for c in ("Observed_Difference", "CI95_Lower", "CI95_Upper"))
    better = bool((m.Better.fillna("") == m.Better_v2.fillna("")).all())
    return {"rows": int(len(m)), "expected": int(len(v2)), "estimable": int(est.sum()), "estimability_equal": same_est,
            "values_within_1e-12": values, "better_method_equal": better,
            "Pass": bool(len(m) == len(v2) and same_est and values and better)}


def check_soh(root: Path, soh_set: pd.DataFrame, b_long: pd.DataFrame) -> dict:
    """Check 2: SOH intervals equal reruns/scaling_bootstrap (1e-9); A8 RF-GB equals the negated S6 statistic."""
    sb = root.parent / "reruns/scaling_bootstrap"
    ref = pd.read_csv(sb / "model_pair_ci.csv", encoding="utf-8-sig")
    mine = soh_set[soh_set.Scope.isin(SCOPE_TO_CONVERSION)].assign(Method=lambda f: f.Scope.map(SCOPE_TO_CONVERSION))
    m = mine.merge(ref, on=["Dataset", "Qref", "Method", "Pair"])
    pair_ok = len(m) == len(ref) and all(_close(m[a], m[b], 1e-9) for a, b in (
        ("Observed_Difference", "Observed_Difference_Of_Medians_Ah"), ("CI95_Lower", "CI95_Lower_Ah"), ("CI95_Upper", "CI95_Upper_Ah")))
    ref_b = pd.read_csv(sb / "measured_minus_nominal_ci.csv", encoding="utf-8-sig")
    mb = b_long.rename(columns={"Conversion": "Method"}).merge(ref_b, on=["Dataset", "Method", "Model", "Measured_Qref"], suffixes=("", "_ref"))
    b_ok = len(mb) == 12 * 2 and all(_close(mb[c], mb[f"{c}_ref"], 1e-9) for c in (
        "Observed_Measured_Minus_Nominal_Ah", "CI95_Lower_Ah", "CI95_Upper_Ah"))
    s6 = pd.read_csv(root.parent / "supplementary/csv/S6_A8_Corrected_CI.csv", encoding="utf-8-sig")
    s6["Qref"] = s6.Qref_Definition.map({"nominal": "nominal", "first_comparable": "first", "early_median5_comparable": "first5"})
    a8 = soh_set[(soh_set.Scope == "A8_OWN") & (soh_set.Pair == "RF-GB")].merge(s6, on="Qref")
    s6_ok = len(a8) == 3 and _close(a8.Observed_Difference, -a8.Observed_Difference_Of_Medians, 1e-12) and \
        _close(a8.CI95_Lower, -a8.Bootstrap_95CI_Upper, 1e-12) and _close(a8.CI95_Upper, -a8.Bootstrap_95CI_Lower, 1e-12)
    return {"pair_rows": int(len(m)), "pair_expected": int(len(ref)), "pairs_within_1e-9": bool(pair_ok),
            "measured_minus_nominal_rows": int(len(mb)), "measured_minus_nominal_within_1e-9": bool(b_ok),
            "a8_rf_gb_matches_negated_s6": bool(s6_ok), "Pass": bool(pair_ok and b_ok and s6_ok)}


def check_strict_labels(root: Path, strict: pd.DataFrame) -> dict:
    """Check 3: recomputed strict labels identical to the committed v2 labels.csv."""
    v2 = pd.read_csv(root / "outputs/rul_ranking_pilot_v2/labels.csv", encoding="utf-8-sig", float_precision="round_trip")
    v2["Theta"] = v2.Theta.round(2)
    key = ["Dataset", "Qref_Definition", "Theta", "Cell_ID"]
    cols = ["Status", "First_Hit_Raw_Cycle", "EOL_Exact_Index", "EOL_Lower_Index", "EOL_Upper_Index", "Threshold_Ah"]
    m = strict.merge(v2, on=key, suffixes=("", "_v2"))
    ok = len(m) == len(v2) == len(strict) and bool((m.Status == m.Status_v2).all()) and \
        all(_close(m[c], m[f"{c}_v2"], 0.0) for c in cols[1:])
    return {"rows": int(len(m)), "expected": int(len(v2)), "Pass": bool(ok)}


def check_mit_tolerance(tol: pd.DataFrame) -> dict:
    """Check 4: MIT nominal theta=0.80 should give 124 tolerance-rule labels, 81 from the tolerance clause."""
    s = tol[(tol.Dataset == "MIT") & (tol.Qref_Definition == "nominal") & np.isclose(tol.Theta, 0.80)]
    kinds = s.Tolerance_Kind.value_counts().to_dict()
    labeled, clause = int(s.Tolerance_Labeled.sum()), int(kinds.get("TOLERANCE", 0))
    detail = {"cells": int(len(s)), "tolerance_rule_labels": labeled, "from_tolerance_clause": clause,
              "kinds": {k: int(v) for k, v in kinds.items()}, "expected": {"labels": 124, "from_tolerance_clause": 81}}
    if clause != 81:
        sw = s[(s.Strict_Status == "EXACT") & s.Tolerance_Kind.eq("TOLERANCE")]
        detail["strict_exact_cells_that_need_the_clause_after_smoothing"] = int(len(sw))
    detail["Pass"] = labeled == 124 and clause == 81
    return detail


def _summary(soh_cmp, rul_a, b_wide, rul_c, tol) -> pd.DataFrame:
    rows = []
    def add(trap, group, common, still, note=""):
        rows.append({"Trap": trap, "Group": group, "Common_Practice_Results": int(common),
                     "Still_Holding_Under_Protocol": int(still), "Note": note})
    for scope, label in (("A9_OWN", "SOH main (OWN)"),):
        for d in ("XJTU", "HUST"):
            c = soh_cmp[(soh_cmp.Scope == scope) & (soh_cmp.Dataset == d)]
            add("A", f"{label} {d}", c.Surface_Change.sum(), (c.Surface_Change & c.Supported_Reversal).sum())
    for d in ("XJTU", "HUST", "MIT"):
        c = rul_a[rul_a.Dataset == d]
        add("A", f"RUL v2 E1+E2 {d}", c.Surface_Change.sum(), (c.Surface_Change & c.Supported_Reversal).sum())
    for scope, label in (("A9_TRAIN", "SOH supplementary (TRAIN)"), ("A9_NOM", "SOH supplementary (NOM)"), ("A8_OWN", "SOH supplementary (A8, OWN)")):
        c = soh_cmp[soh_cmp.Scope == scope]
        add("A", label, c.Surface_Change.sum(), (c.Surface_Change & c.Supported_Reversal).sum(), "supplementary scope; not pooled")
    k = int(b_wide.Conclusion_Depends_On_Information_Condition.sum())
    add("B", "SOH 12 settings", len(b_wide), len(b_wide) - k, f"{k} of {len(b_wide)} settings: conclusion depends on the information condition")
    one = rul_c[(rul_c.Eval_Set == "E3_ONE_SIDED") & rul_c.Supported_Reversal]
    cidx = rul_c[(rul_c.Eval_Set == "E3") & rul_c.Supported_Reversal]
    key = ["Dataset", "Theta", "Pair", "Setting_1", "Setting_2", "Better_1", "Better_2"]
    kept = one[key].merge(cidx[key], on=key)
    for d in ("XJTU", "HUST", "MIT"):
        add("C", f"RUL v2 E3 {d}", (one.Dataset == d).sum(), (kept.Dataset == d).sum())
    for d in ("XJTU", "HUST", "MIT"):
        t = tol[tol.Dataset == d]
        add("D", f"Endpoint labels {d}", t.Tolerance_Labeled.sum(), (t.Tolerance_Labeled & t.Strict_Labeled).sum())
    return pd.DataFrame(rows)


def _figure(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"A": "A  Noise read as ranking change", "B": "B  Undeclared information condition",
              "C": "C  Biased censored metric", "D": "D  Endpoint rule"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, trap in zip(axes.ravel(), "ABCD"):
        s = summary[summary.Trap == trap]
        x = np.arange(len(s))
        ax.bar(x - 0.2, s.Common_Practice_Results, 0.4, label="common practice", color="#9aa5b1")
        ax.bar(x + 0.2, s.Still_Holding_Under_Protocol, 0.4, label="still holds under protocol", color="#2f6f9f")
        ax.set_xticks(x, s.Group, rotation=30, ha="right", fontsize=8)
        ax.set_title(titles[trap], fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def run(root: Path, output_dir: Path | None = None) -> Path:
    root = Path(root).resolve()
    target = Path(output_dir).resolve() if output_dir else root / "outputs" / OUTPUT_DIR_NAME
    if target.exists():
        raise FileExistsError(f"Output directory exists; refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}_staging_", dir=target.parent))
    before = {f: _file_sha(root / f) for f in INPUTS}
    log = [f"{PIPELINE_VERSION}; plan commit {PLAN_COMMIT}"]

    cell = soh_cell_mae(root)
    diffs, soh_set = soh_pair_diffs(cell)
    soh_cmp = soh_comparisons(diffs, soh_set)
    b_long, b_wide = trap_b(cell)
    samples, preds = load_v2(root)
    rul_set, rul_diffs = rul_settings(samples, preds)
    rul_a = comparisons(rul_set, ("E1", "E2"), rul_diffs)
    rul_c = comparisons(rul_set, ("E3", "E3_ONE_SIDED"))
    noise = noise_table(rul_diffs)
    bias = bias_table(rul_set)
    data = load_all(root)
    strict = detect_all(data)
    strict["Theta"] = strict.Theta.round(2)
    tol = tolerance_labels(data, strict)
    log.append(time.strftime("%Y-%m-%dT%H:%M:%S") + " computations done")

    checks = {"1_v2_intervals": check_v2_intervals(root, rul_set), "2_soh_intervals": check_soh(root, soh_set, b_long),
              "3_strict_labels": check_strict_labels(root, strict), "4_mit_tolerance": check_mit_tolerance(tol),
              "5_determinism": {"Pass": None, "detail": "compare two independent output directories"}}
    (staging / "consistency_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not all(checks[k]["Pass"] for k in ("1_v2_intervals", "2_soh_intervals", "3_strict_labels")):
        raise RuntimeError(f"CONSISTENCY_CHECKS_1_3_FAILED; see {staging}")

    _write(soh_set, staging / "trapA_soh_settings.csv")
    _write(soh_cmp, staging / "trapA_soh_comparisons.csv")
    _write(rul_a, staging / "trapA_rul_comparisons.csv")
    _write(noise[noise.Eval_Set.isin(["E1", "E2"])], staging / "trapA_rul_noise_by_setting.csv")
    noise_sum = pd.concat([
        soh_set.groupby(["Scope", "Dataset"])[["Noise_Single_Resample_Disagreement", "Noise_Half_Disagreement"]].mean().reset_index().assign(Part="SOH"),
        noise[noise.Eval_Set.isin(["E1", "E2"])].groupby(["Dataset"])[["Noise_Single_Resample_Disagreement", "Noise_Half_Disagreement"]].mean().reset_index().assign(Part="RUL_E1_E2", Scope="RUL_v2")])
    _write(noise_sum, staging / "trapA_noise_summary.csv")
    _write(b_long, staging / "trapB_intervals.csv")
    _write(b_wide, staging / "trapB_settings.csv")
    _write(rul_set[rul_set.Eval_Set.isin(["E3", "E3_ONE_SIDED"])], staging / "trapC_e3_intervals.csv")
    _write(rul_c, staging / "trapC_comparisons.csv")
    _write(bias, staging / "trapC_prediction_bias_settings.csv")
    _write(bias.groupby(["Eval_Set", "Dataset"]).Larger_Prediction_Judged_Better.agg(["mean", "size"]).reset_index()
           .rename(columns={"mean": "Share_Larger_Prediction_Judged_Better", "size": "Determined_Settings"}), staging / "trapC_prediction_bias_summary.csv")
    _write(tol, staging / "trapD_labels.csv")
    _write(setting_summary(tol), staging / "trapD_setting_summary.csv")
    _write(pd.crosstab([tol.Dataset, tol.Strict_Status], tol.Tolerance_Kind).reset_index(), staging / "trapD_crosstab.csv")
    _write(eol_difference_summary(tol), staging / "trapD_eol_difference_summary.csv")
    summary = _summary(soh_cmp, rul_a, b_wide, rul_c, tol)
    _write(summary, staging / "summary_table.csv")
    _figure(summary, staging / "summary_figure.png")

    if {f: _file_sha(root / f) for f in INPUTS} != before:
        raise RuntimeError("INPUT_FILES_CHANGED_DURING_RUN")
    info = {"pipeline": PIPELINE_VERSION, "plan_commit": PLAN_COMMIT, "python": sys.version.split()[0],
            "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "input_sha256": before}
    (staging / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.append(time.strftime("%Y-%m-%dT%H:%M:%S") + " complete")
    (staging / "processing_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    staging.replace(target)
    return target
