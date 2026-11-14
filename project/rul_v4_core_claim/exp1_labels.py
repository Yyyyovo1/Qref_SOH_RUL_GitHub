"""Experiment 1 labels on MIT: strict (frozen dense detector) and tolerance (Lecu et al. 2026) endpoints, EOL > 100,
labeled shares, common windows, P cell sets, and checks 1-3 for experiment 1 (AMENDMENT_01 section 7)."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from protocol_validation.endpoint import tolerance_labels, tolerance_rule
from rul_pilot_v2.data import _mit_inputs
from rul_pilot_v2.pipeline import _check_mit_labels, _physical_equivalence
from rul_ranking_pilot.labels import DatasetInputs, _detection_record, check_monotonicity, detect

from .config import (
    EXACT,
    EXP1_THETAS,
    MIT_SHARES_REF,
    MIT_WINDOWS_REF,
    PREDICTION_CYCLE,
    QREF_SHORT,
    QREFS,
    RULES,
    TRAPD_LABELS,
    WINDOW_SHARE,
)

NOMINAL_AH = 1.1
SHORT_TO_QREF = {v: k for k, v in QREF_SHORT.items()}


def load_mit(root: Path) -> DatasetInputs:
    return _mit_inputs(Path(root))


def strict_detections(d: DatasetInputs, thetas=EXP1_THETAS) -> pd.DataFrame:
    rows = []
    for q in QREFS:
        for theta in thetas:
            for cell in d.cells:
                qref = float(d.qref.at[cell, q])
                thr = float(theta) * qref
                rows.append({"Dataset": d.name, "Qref_Definition": q, "Theta": theta, "Cell_ID": cell, "Qref_Ah": qref,
                             "Threshold_Ah": thr, **_detection_record(detect(d, cell, thr))})
    det = pd.DataFrame(rows)
    det["Theta"] = det.Theta.round(2)
    return det


def build_labels(d: DatasetInputs, thetas=EXP1_THETAS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (strict detections, tolerance table in the trapD layout, long label table)."""
    det = strict_detections(d, thetas)
    tol = tolerance_labels({d.name: d}, det)
    tol["Theta"] = tol.Theta.round(2)
    long = []
    for rule in RULES:
        lab = tol.copy()
        if rule == "strict":
            lab["Labeled"] = lab.Strict_Status.eq(EXACT)
            lab["EOL"] = np.where(lab.Labeled, lab.Strict_EOL_Index, np.nan)
        else:
            lab["Labeled"] = lab.Tolerance_Kind.ne("NO_LABEL")
            lab["EOL"] = np.where(lab.Labeled, lab.Tolerance_EOL_Index, np.nan)
        lab["Rule"] = rule
        lab["Labeled_EOL_gt_100"] = lab.Labeled & (lab.EOL > PREDICTION_CYCLE)
        long.append(lab[["Rule", "Qref_Definition", "Theta", "Cell_ID", "Labeled", "EOL", "Labeled_EOL_gt_100"]])
    return det, tol, pd.concat(long, ignore_index=True)


def shares(labels: pd.DataFrame, n_cells: int) -> pd.DataFrame:
    g = labels.groupby(["Rule", "Qref_Definition", "Theta"]).Labeled_EOL_gt_100
    out = g.sum().rename("Labeled_EOL_gt_100").reset_index()
    out["Cells"] = n_cells
    out["Share"] = out.Labeled_EOL_gt_100 / n_cells
    return out


def windows(sh: pd.DataFrame, thetas=EXP1_THETAS) -> pd.DataFrame:
    s = sh.set_index(["Rule", "Qref_Definition", "Theta"]).Share
    rows = []
    for rule in RULES:
        for q1, q2 in combinations(QREFS, 2):
            pts = [t for t in thetas if s[(rule, q1, t)] >= WINDOW_SHARE and s[(rule, q2, t)] >= WINDOW_SHARE]
            rows.append({"Rule": rule, "Pair": f"{QREF_SHORT[q1]}-{QREF_SHORT[q2]}", "Qref_1": q1, "Qref_2": q2,
                         "Common_Window_Points": len(pts), "Common_Window": "|".join(f"{t:.2f}" for t in pts)})
    return pd.DataFrame(rows)


def window_points(win: pd.DataFrame) -> dict[tuple[str, str, str], list[float]]:
    return {(r.Rule, r.Qref_1, r.Qref_2): [float(x) for x in r.Common_Window.split("|")] if r.Common_Window else []
            for r in win.itertuples(index=False)}


def cell_sets(labels: pd.DataFrame, win: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    """P cell set per (rule, definition pair, theta in window): labeled with EOL > 100 under both definitions."""
    ok = labels[labels.Labeled_EOL_gt_100].groupby(["Rule", "Qref_Definition", "Theta"]).Cell_ID.apply(frozenset).to_dict()
    rows = []
    for (rule, q1, q2), pts in window_points(win).items():
        for t in pts:
            both = sorted(ok.get((rule, q1, t), frozenset()) & ok.get((rule, q2, t), frozenset()))
            rows.append({"Rule": rule, "Qref_1": q1, "Qref_2": q2, "Pair": f"{QREF_SHORT[q1]}-{QREF_SHORT[q2]}", "Theta": t,
                         "Cells": len(both), "Excluded_Cells": len(cells) - len(both), "Cell_IDs": "|".join(both)})
    return pd.DataFrame(rows)


def eol_lookup(labels: pd.DataFrame) -> dict[tuple[str, str, float], dict[str, float]]:
    lab = labels[labels.Labeled_EOL_gt_100]
    return {k: dict(zip(g.Cell_ID, g.EOL.astype(float))) for k, g in lab.groupby(["Rule", "Qref_Definition", "Theta"])}


# ---------------------------------------------------------------- reference comparisons
def compare_windows_with_prep(root: Path, win: pd.DataFrame, sh: pd.DataFrame) -> dict:
    ref = pd.read_csv(Path(root) / MIT_WINDOWS_REF, encoding="utf-8-sig", dtype={"Common_Window": str}).fillna({"Common_Window": ""})
    m = win.merge(ref, on=["Rule", "Pair"], suffixes=("", "_prep"), how="outer", indicator=True)
    same_w = bool((m._merge == "both").all() and (m.Common_Window == m.Common_Window_prep).all()
                  and (m.Common_Window_Points == m.Common_Window_Points_prep).all())
    rs = pd.read_csv(Path(root) / MIT_SHARES_REF, encoding="utf-8-sig")
    rs["Theta"] = rs.Theta.round(2)
    rs["Qref_Definition"] = rs.Qref.map(SHORT_TO_QREF)
    ms = sh.merge(rs, on=["Rule", "Qref_Definition", "Theta"], suffixes=("", "_prep"))
    same_c = bool(len(ms) == len(sh) == len(rs) and (ms.Labeled_EOL_gt_100 == ms.Labeled_EOL_gt_100_prep).all())
    return {"windows_equal_prep": same_w, "labeled_counts_equal_prep": same_c, "Pass": same_w and same_c}


def check1(root: Path, det: pd.DataFrame, tol: pd.DataFrame) -> dict:
    """Strict: v2 MIT check 1 (frozen endpoint labels, cycle_life within +-1). Tolerance: trapD labels row by row."""
    strict = _check_mit_labels(Path(root), det)
    ref = pd.read_csv(Path(root) / TRAPD_LABELS, encoding="utf-8-sig", float_precision="round_trip")
    ref = ref[ref.Dataset.eq("MIT")].copy()
    ref["Theta"] = ref.Theta.round(2)
    key = ["Qref_Definition", "Theta", "Cell_ID"]
    cols = ["Strict_Status", "Strict_EOL_Index", "Tolerance_Kind", "Tolerance_EOL_Index"]
    m = tol.merge(ref[key + cols], on=key, suffixes=("", "_ref"))

    def eq(a, b):
        return (a == b) | (a.isna() & b.isna())

    same = np.logical_and.reduce([eq(m[c], m[f"{c}_ref"]) for c in cols])
    tol_res = {"rows_compared": int(len(m)), "rows_expected": int(len(tol)), "rows_reference": int(len(ref)),
               "mismatches": int((~same).sum())}
    tol_res["Pass"] = bool(len(m) == len(tol) == len(ref) and same.all())
    return {"strict": strict, "tolerance": tol_res, "Pass": bool(strict["Pass"] and tol_res["Pass"])}


def check2(d: DatasetInputs, det: pd.DataFrame, tol: pd.DataFrame) -> dict:
    """Physical equivalence: theta*Qref written as theta'*nominal gives identical labels (strict and tolerance)."""
    strict = _physical_equivalence({d.name: d}, det)
    cap = d.capacity.sort_values(["Cell_ID", "Endpoint_Index", "Raw_Cycle_Index"], kind="stable")
    series = {c: (g.Endpoint_Index.to_numpy(float), g.Capacity_Ah.to_numpy(float)) for c, g in cap.groupby("Cell_ID")}
    bad = n = 0
    qref = d.qref
    for r in tol[tol.Qref_Definition.ne("nominal")].itertuples(index=False):
        thr = (r.Theta * float(qref.at[r.Cell_ID, r.Qref_Definition]) / NOMINAL_AH) * NOMINAL_AH
        kind, eol = tolerance_rule(*series[r.Cell_ID], thr)
        n += 1
        if not (kind == r.Tolerance_Kind and (eol == r.Tolerance_EOL_Index or (np.isnan(eol) and np.isnan(r.Tolerance_EOL_Index)))):
            bad += 1
    tol_res = {"comparisons": n, "mismatches": bad, "Pass": bad == 0}
    return {"strict": strict, "tolerance": tol_res, "Pass": bool(strict["Pass"] and tol_res["Pass"])}


def check3(det: pd.DataFrame, tol: pd.DataFrame) -> dict:
    s = check_monotonicity(det)
    t = check_monotonicity(tol.assign(Dataset="MIT", First_Hit_Index=tol.Tolerance_EOL_Index.where(tol.Tolerance_Kind.ne("NO_LABEL"))))
    return {"strict_violations": int(s.Violations.iat[0]), "tolerance_violations": int(t.Violations.iat[0]),
            "Pass": bool(s.Pass.iat[0] and t.Pass.iat[0])}
