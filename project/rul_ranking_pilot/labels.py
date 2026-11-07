"""Inputs, t0 rule, endpoint labels over the theta grid, and sanity checks 1-3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eol_threshold_robustness_repro.config import (
    EXACT as FROZEN_EXACT,
    HUST_QREF_COLUMNS,
    INTERVAL as FROZEN_INTERVAL,
    LEFT as FROZEN_LEFT,
    RIGHT as FROZEN_RIGHT,
    XJTU_QREF_COLUMNS,
    PipelinePaths,
)
from eol_threshold_robustness_repro.pipeline import (
    EOLDetection,
    detect_hust_eol,
    detect_xjtu_eol,
    load_frozen_inputs,
)

from .config import (
    BASELINE,
    DATASETS,
    EXACT,
    INTERVAL,
    NOMINAL_CAPACITY_AH,
    QREFS,
    RIGHT,
    THETAS,
    XJTU_DENSE_BATCHES,
)

STATUS_MAP = {FROZEN_EXACT: EXACT, FROZEN_INTERVAL: INTERVAL, FROZEN_RIGHT: RIGHT, FROZEN_LEFT: BASELINE}
SOURCE_COLUMNS = {
    "XJTU": {
        "first_comparable": "Qref_First_Comparable_Source_Cycle",
        "early_median5_comparable": "Qref_Early_Median5_Comparable_Source_Cycles",
    },
    "HUST": {
        "first_comparable": "Qref_First_Comparable_Source_Cycle",
        "early_median5_comparable": "Qref_Early_Median5_Comparable_Source_Cycles",
    },
}


class PilotError(RuntimeError):
    """Raised when a preregistered check fails or an input is inconsistent."""


@dataclass
class DatasetInputs:
    name: str
    cells: list[str]                      # lexicographic Cell_ID order
    observations: dict[str, pd.DataFrame]  # per cell, in the frozen detector's column layout
    capacity: pd.DataFrame                # Cell_ID, Raw_Cycle_Index, Endpoint_Index, Capacity_Ah
    qref: pd.DataFrame                    # index Cell_ID, one column per Qref definition (Ah)
    source_cycles: dict[str, dict[str, list[int]]]
    batch: dict[str, int]
    folds: dict[str, int]
    features: pd.DataFrame                # Cell_ID, Raw_Cycle_Index, A9 features


def _source_list(value: Any) -> list[int]:
    return [int(float(v)) for v in str(value).split("|") if str(v).strip() not in ("", "nan")]


def load_inputs(root: Path) -> dict[str, DatasetInputs]:
    frozen = load_frozen_inputs(PipelinePaths(root))
    outputs = root / "outputs"
    result: dict[str, DatasetInputs] = {}
    for name in DATASETS:
        if name == "XJTU":
            obs = frozen.xjtu_observations.copy()
            qtab, columns = frozen.xjtu_qref.copy(), XJTU_QREF_COLUMNS
            cap = obs.rename(columns={"Cumulative_Aging_Operation_Count": "Endpoint_Index"})
            batch = {str(r.Cell_ID): int(float(r.Batch_ID)) for r in qtab.itertuples(index=False)}
        else:
            obs = frozen.hust_observations.copy()
            qtab, columns = frozen.hust_qref.copy(), HUST_QREF_COLUMNS
            cap = obs.rename(columns={"Primary_Capacity_Ah": "Capacity_Ah"})
            cap["Endpoint_Index"] = cap["Raw_Cycle_Index"]
            batch = {}
        cap = cap[["Cell_ID", "Raw_Cycle_Index", "Endpoint_Index", "Capacity_Ah"]].copy()
        for column in ["Raw_Cycle_Index", "Endpoint_Index", "Capacity_Ah"]:
            cap[column] = pd.to_numeric(cap[column], errors="raise").astype(float)
        cap = cap.sort_values(["Cell_ID", "Raw_Cycle_Index"], kind="stable").reset_index(drop=True)
        qtab["Cell_ID"] = qtab["Cell_ID"].astype(str)
        qref = pd.DataFrame(
            {q: pd.to_numeric(qtab[columns[q][0]], errors="raise").to_numpy(float) for q in QREFS},
            index=qtab["Cell_ID"],
        )
        if qref.isna().any().any():
            raise PilotError(f"{name}_COMMON_QREF_MISSING")
        sources = {
            q: {str(c): _source_list(v) for c, v in zip(qtab["Cell_ID"], qtab[col])}
            for q, col in SOURCE_COLUMNS[name].items()
        }
        folds_frame = pd.read_csv(
            outputs / "soh_ml_design_v1" / f"ml_cell_fold_assignments_{name.lower()}.csv", encoding="utf-8-sig"
        )
        folds = dict(zip(folds_frame["Cell_ID"].astype(str), folds_frame["Fold_ID"].astype(int)))
        features = pd.read_csv(
            outputs / "soh_feature_extraction_v1" / f"feature_matrix_{name.lower()}_v1.csv", low_memory=False
        )
        cells = sorted(qref.index.astype(str))
        if set(cells) != set(folds) or set(cells) != set(cap["Cell_ID"].astype(str)):
            raise PilotError(f"{name}_CELL_SETS_DO_NOT_MATCH")
        result[name] = DatasetInputs(
            name=name,
            cells=cells,
            observations={str(c): g.copy() for c, g in obs.groupby("Cell_ID", sort=False)},
            capacity=cap,
            qref=qref.loc[cells],
            source_cycles=sources,
            batch=batch,
            folds=folds,
            features=features,
        )
    return result


def detect(data: DatasetInputs, cell: str, threshold_Ah: float) -> EOLDetection:
    observations = data.observations[cell]
    if data.name == "XJTU":
        return detect_xjtu_eol(observations, threshold_Ah, dense_measurement=data.batch[cell] in XJTU_DENSE_BATCHES)
    return detect_hust_eol(observations, threshold_Ah)


def _first_hit_index(d: EOLDetection) -> float:
    if d.exact_cycle is not None:
        return float(d.exact_cycle)
    if d.status in (FROZEN_INTERVAL, FROZEN_LEFT) and d.upper_bound is not None:
        return float(d.upper_bound)
    return np.nan


def _detection_record(d: EOLDetection) -> dict[str, Any]:
    status = STATUS_MAP[d.status]
    return {
        "Status": status,
        "First_Hit_Raw_Cycle": np.nan if d.first_below_raw_cycle is None else float(d.first_below_raw_cycle),
        "First_Hit_Index": _first_hit_index(d),
        "EOL_Exact_Index": np.nan if d.exact_cycle is None else float(d.exact_cycle),
        "EOL_Lower_Index": float(d.lower_bound) if status == INTERVAL else np.nan,
        "EOL_Upper_Index": float(d.upper_bound) if status == INTERVAL else np.nan,
        "Last_Observed_Index": np.nan if d.last_cycle is None else float(d.last_cycle),
    }


def detect_all(data: dict[str, DatasetInputs]) -> pd.DataFrame:
    """Endpoint detection for every dataset x Qref x theta x cell (independent of t0)."""
    rows = []
    for name, d in data.items():
        for q in QREFS:
            for theta in THETAS:
                for cell in d.cells:
                    qref = float(d.qref.at[cell, q])
                    threshold = float(theta) * qref
                    rows.append({"Dataset": name, "Qref_Definition": q, "Theta": theta, "Cell_ID": cell,
                                 "Qref_Ah": qref, "Threshold_Ah": threshold,
                                 **_detection_record(detect(d, cell, threshold))})
    return pd.DataFrame(rows)


def compute_t0(data: dict[str, DatasetInputs], detections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, d in data.items():
        t_avail = max(max(v) for q in d.source_cycles for v in d.source_cycles[q].values())
        sub = detections[(detections.Dataset == name) & detections.Status.ne(BASELINE)]
        hits = sub.dropna(subset=["First_Hit_Raw_Cycle"])
        position = hits["First_Hit_Raw_Cycle"].idxmin()
        e_min = float(hits.at[position, "First_Hit_Raw_Cycle"])
        t0 = int(max(t_avail, e_min - 1))
        rows.append({"Dataset": name, "T_avail": int(t_avail), "E_min": e_min,
                     "E_min_Qref": hits.at[position, "Qref_Definition"], "E_min_Theta": hits.at[position, "Theta"],
                     "E_min_Cell": hits.at[position, "Cell_ID"], "t0_raw_cycle": t0})
    return pd.DataFrame(rows)


def build_labels(data: dict[str, DatasetInputs], detections: pd.DataFrame, t0: dict[str, int]) -> pd.DataFrame:
    """Attach t0-relative RUL targets; cells hitting EOL at raw cycle <= t0 are EOL_BEFORE_T0."""
    n_t0 = {}
    for name, d in data.items():
        cap = d.capacity[d.capacity["Raw_Cycle_Index"] <= t0[name]]
        last = cap.groupby("Cell_ID")["Endpoint_Index"].max()
        if set(last.index) != set(d.cells):
            raise PilotError(f"{name}_CELL_WITHOUT_OBSERVATION_BEFORE_T0")
        n_t0.update({(name, c): float(v) for c, v in last.items()})
    labels = detections.copy()
    labels["t0_raw_cycle"] = labels["Dataset"].map(t0).astype(int)
    labels["n_t0_Index"] = [n_t0[(a, c)] for a, c in zip(labels.Dataset, labels.Cell_ID)]
    labels["EOL_Before_t0"] = labels["First_Hit_Raw_Cycle"].le(labels["t0_raw_cycle"]).fillna(False)
    labels["Eligible"] = ~labels["EOL_Before_t0"] & labels["Status"].ne(BASELINE)
    n = labels["n_t0_Index"]
    labels["RUL"] = np.where(labels.Status.eq(EXACT), labels["EOL_Exact_Index"] - n, np.nan)
    labels["RUL_Lower"] = np.select(
        [labels.Status.eq(EXACT), labels.Status.eq(INTERVAL), labels.Status.eq(RIGHT)],
        [labels["RUL"], labels["EOL_Lower_Index"] - n, labels["Last_Observed_Index"] - n], np.nan)
    labels["RUL_Upper"] = np.select(
        [labels.Status.eq(EXACT), labels.Status.eq(INTERVAL), labels.Status.eq(RIGHT)],
        [labels["RUL"], labels["EOL_Upper_Index"] - n, np.inf], np.nan)
    bad = labels["Eligible"] & (labels["RUL_Upper"] <= 0)
    if bad.any():
        raise PilotError("ELIGIBLE_CELL_WITH_NONPOSITIVE_RUL")
    labels["Fold_ID"] = [data[a].folds[c] for a, c in zip(labels.Dataset, labels.Cell_ID)]
    return labels


# ---------------------------------------------------------------- checks 1-3

def check_frozen_labels(root: Path, labels: pd.DataFrame) -> pd.DataFrame:
    """Check 1: endpoint classes at 0.75/0.80/0.85 reproduce the frozen status summary."""
    frozen = pd.read_csv(root / "outputs/eol_threshold_robustness_v1/eol_threshold_status_summary.csv", encoding="utf-8-sig")
    rows = []
    for r in frozen[frozen.Qref_Definition.isin(QREFS)].itertuples(index=False):
        sub = labels[(labels.Dataset == r.Dataset) & (labels.Qref_Definition == r.Qref_Definition)
                     & np.isclose(labels.Theta, r.Threshold_Fraction)]
        counts = sub.Status.value_counts()
        current = (int(counts.get(EXACT, 0)), int(counts.get(INTERVAL, 0)), int(counts.get(RIGHT, 0)), int(counts.get(BASELINE, 0)))
        expected = (int(r.EXACT_OBSERVED_Count), int(r.INTERVAL_CENSORED_Count), int(r.RIGHT_CENSORED_Count),
                    int(r.LEFT_OR_BASELINE_UNRESOLVED_Count))
        rows.append({"Check": "1_FROZEN_LABELS", "Dataset": r.Dataset, "Qref_Definition": r.Qref_Definition,
                     "Theta": r.Threshold_Fraction, "Current_E_I_R_B": "/".join(map(str, current)),
                     "Frozen_E_I_R_B": "/".join(map(str, expected)),
                     "Point_RUL_Current": current[0], "Point_RUL_Frozen": int(r.Point_RUL_Available_Count),
                     "Pass": current == expected and current[0] == int(r.Point_RUL_Available_Count)})
    return pd.DataFrame(rows)


def check_physical_equivalence(data: dict[str, DatasetInputs], labels: pd.DataFrame) -> pd.DataFrame:
    """Check 2 (labels): (nominal, theta' = theta*Qref/Qnom) must give identical labels to (Qref, theta)."""
    compare = ["Status", "First_Hit_Raw_Cycle", "EOL_Exact_Index", "EOL_Lower_Index", "EOL_Upper_Index"]
    rows = []
    for r in labels[labels.Qref_Definition.ne("nominal")].itertuples(index=False):
        nominal = NOMINAL_CAPACITY_AH[r.Dataset]
        theta_prime = r.Theta * r.Qref_Ah / nominal
        threshold = theta_prime * nominal
        record = _detection_record(detect(data[r.Dataset], r.Cell_ID, threshold))
        same = all((record[k] == getattr(r, k)) or (pd.isna(record[k]) and pd.isna(getattr(r, k))) for k in compare)
        rows.append({"Dataset": r.Dataset, "Qref_Definition": r.Qref_Definition, "Theta": r.Theta, "Cell_ID": r.Cell_ID,
                     "Abs_Threshold_Difference_Ah": abs(threshold - r.Threshold_Ah), "Labels_Identical": same})
    detail = pd.DataFrame(rows)
    return pd.DataFrame([{
        "Check": "2_PHYSICAL_EQUIVALENCE_LABELS", "Comparisons": len(detail),
        "Mismatches": int((~detail.Labels_Identical).sum()),
        "Max_Abs_Threshold_Difference_Ah": float(detail.Abs_Threshold_Difference_Ah.max()),
        "Pass": bool(detail.Labels_Identical.all()),
    }])


def check_monotonicity(labels: pd.DataFrame) -> pd.DataFrame:
    """Check 3: first-hit index is non-increasing in theta; a hit at theta implies a hit at every larger theta."""
    violations = 0
    groups = 0
    for _, g in labels.sort_values("Theta").groupby(["Dataset", "Qref_Definition", "Cell_ID"]):
        groups += 1
        hit = g["First_Hit_Index"].to_numpy(float)
        has = ~np.isnan(hit)
        if has.any() and not has[np.argmax(has):].all():
            violations += 1
            continue
        h = hit[has]
        if len(h) > 1 and np.any(np.diff(h) > 0):
            violations += 1
    return pd.DataFrame([{"Check": "3_MONOTONICITY", "Cell_Qref_Series": groups, "Violations": violations,
                          "Pass": violations == 0}])
