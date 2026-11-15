"""Experiment 1 (PROTOCOL_v4 section 2; AMENDMENT_01 sections 3, 4, 9): fitting over the P settings, and the
evaluation (APE, paired cell bootstrap of the mean difference, better method, reversals, H1 units, C condition).

Fitting (`fit_p_settings`, `fit_delta_settings`) is allowed in development runs; everything that compares predictions
with labels is decorated with `@evaluation` and raises unless the formal run enabled it.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rul_pilot_v2.evaluation import _better, _interval

from .blinding import evaluation
from .config import EXP1_MODELS, EXP1_PAIRS, H1_PAIRS, MAX_BOOT_MISSING_FRACTION, MIN_CELLS, QREF_SHORT
from .exp1_models import CVResult, run_original_split, run_outer_cv


def cell_set_id(cells) -> str:
    """Short identifier of a cell set (several settings share a (rule, Qref, theta) with different cell sets)."""
    import hashlib

    return hashlib.sha256("|".join(cells).encode()).hexdigest()[:12]


@dataclass
class Exp1Inputs:
    cells: list[str]                                   # all 124 MIT cells, lexicographic (bootstrap order)
    design: dict[str, dict[str, np.ndarray]]           # model -> cell -> feature vector
    eol: dict[tuple[str, str, float], dict[str, float]]  # (rule, Qref, theta) -> cell -> EOL (labeled and EOL > 100)
    sets: pd.DataFrame                                 # exp1_labels.cell_sets
    folds: dict[str, int]
    split: dict[str, str]                              # cell -> train / test1 / test2


@dataclass
class Exp1Fits:
    results: dict = field(default_factory=dict)        # (rule, Qref, theta, model, cells) -> CVResult
    settings: list = field(default_factory=list)       # (rule, q1, q2, theta, cells) P settings
    warnings: Counter = field(default_factory=Counter)

    def get(self, rule, q, theta, model, cells) -> CVResult:
        return self.results[(rule, q, theta, model, tuple(cells))]


def fit_task(model: str, cells: tuple, eol: dict, design: dict, folds: dict, inner_repeats: int, n_alphas: int) -> CVResult:
    """One outer-CV fit, in this process or in a worker. BLAS is limited to 1 thread whatever the number of workers
    (implementation notes 6.11); the inner-CV settings travel with the task so that a worker uses the caller's."""
    from threadpoolctl import threadpool_limits

    from . import exp1_models as M

    M.INNER_REPEATS, M.ENET_N_ALPHAS = inner_repeats, n_alphas
    with threadpool_limits(limits=1):
        return run_outer_cv(model, list(cells), eol, design, folds)


def _fit_keys(fits: Exp1Fits, inp: Exp1Inputs, keys: list, workers: int, log=None) -> None:
    from . import exp1_models as M
    from .parallel import run_jobs

    todo = [k for k in dict.fromkeys(keys) if k not in fits.results]
    args = [(model, cells, inp.eol.get((rule, q, theta), {}), inp.design[model], inp.folds, M.INNER_REPEATS, M.ENET_N_ALPHAS)
            for rule, q, theta, model, cells in todo]
    prog = (lambda i, n: log(f"exp1 fitted {i}/{n}")) if log else None
    for key, res in zip(todo, run_jobs(fit_task, args, workers, progress=prog)):
        fits.results[key] = res
        fits.warnings.update(res.warnings)


def fit_p_settings(inp: Exp1Inputs, exclude: frozenset = frozenset(), log=None, workers: int = 1) -> Exp1Fits:
    fits = Exp1Fits()
    keys = []
    for r in inp.sets.itertuples(index=False):
        cells = tuple(c for c in r.Cell_IDs.split("|") if c and c not in exclude)
        fits.settings.append((r.Rule, r.Qref_1, r.Qref_2, float(r.Theta), cells))
        keys += [(r.Rule, q, float(r.Theta), m, cells) for q in (r.Qref_1, r.Qref_2) for m in EXP1_MODELS]
    _fit_keys(fits, inp, keys, workers, log)
    return fits


def nominal_own_cells(inp: Exp1Inputs, rule: str, theta: float, exclude: frozenset = frozenset()) -> list[str]:
    return sorted(c for c in inp.eol.get((rule, "nominal", theta), {}) if c not in exclude)


def fit_delta_settings(inp: Exp1Inputs, fits: Exp1Fits, exclude: frozenset = frozenset(), workers: int = 1) -> None:
    """Amendment 01 section 4: for first-first5, nominal variance model on nominal's own labeled EOL > 100 cells."""
    keys = [(rule, "nominal", theta, "variance", tuple(nominal_own_cells(inp, rule, theta, exclude)))
            for rule, q1, q2, theta, _ in fits.settings if q1 != "nominal"]
    _fit_keys(fits, inp, keys, workers)


def hash_table(fits: Exp1Fits) -> pd.DataFrame:
    """Per fit: SHA-256 of the prediction vector (sorted cells) and of the selected hyperparameters. Hashes only, no
    values (implementation notes 6.11; allowed in development outputs, amendment 01 section 11 is not affected)."""
    from rul_ranking_pilot.methods import _sha

    rows = []
    for (rule, q, theta, model, cells), res in fits.results.items():
        pv = np.array([res.predictions[c] for c in sorted(res.predictions)], float)
        hv = np.array([[h["Fold_ID"], h["Selected_L1_Ratio"], h["Selected_Lambda"]] for h in res.hyper_rows], float).ravel()
        rows.append({"Rule": rule, "Qref_Definition": q, "Theta": theta, "Cell_Set_ID": cell_set_id(cells), "Model": model,
                     "Estimable": res.estimable, "Predictions_SHA256": _sha(pv), "Hyperparameters_SHA256": _sha(hv)})
    return pd.DataFrame(rows).sort_values(["Rule", "Qref_Definition", "Theta", "Cell_Set_ID", "Model"], kind="stable").reset_index(drop=True)


def estimability_table(fits: Exp1Fits) -> pd.DataFrame:
    rows = []
    for rule, q1, q2, theta, cells in fits.settings:
        for q in (q1, q2):
            for m in EXP1_MODELS:
                res = fits.get(rule, q, theta, m, cells)
                rows.append({"Rule": rule, "Pair": f"{QREF_SHORT[q1]}-{QREF_SHORT[q2]}", "Theta": theta, "Qref_Definition": q,
                             "Model": m, "Cells": len(cells), "Model_Estimable": res.estimable, "Reason": res.reason,
                             "Setting_Estimable": all(fits.get(rule, q, theta, mm, cells).estimable for mm in EXP1_MODELS)
                             and len(cells) >= MIN_CELLS})
    return pd.DataFrame(rows)


def audit_table(fits: Exp1Fits) -> pd.DataFrame:
    rows = []
    for (rule, q, theta, model, cells), res in sorted(fits.results.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3], cell_set_id(kv[0][4]))):
        for a in res.audit_rows:
            rows.append({"Rule": rule, "Qref_Definition": q, "Theta": theta, "Setting_Cells": len(cells), "Cell_Set_ID": cell_set_id(cells), **a})
    return pd.DataFrame(rows)


def warning_table(fits: Exp1Fits) -> pd.DataFrame:
    """Warning counts by kind (implementation notes 6.10): inner-CV and final-refit convergence warnings separately,
    plus the number of fold fits whose final refit did not converge."""
    rows = [{"Warning_Category": k, "Count": int(v)} for k, v in sorted(fits.warnings.items())]
    folds = [a for res in fits.results.values() for a in res.audit_rows if "Final_Refit_Converged" in a]
    rows.append({"Warning_Category": "Fold_Fits_Elastic_Net", "Count": len(folds)})
    rows.append({"Warning_Category": "Fold_Fits_Final_Refit_Not_Converged", "Count": int(sum(not a["Final_Refit_Converged"] for a in folds))})
    rows.append({"Warning_Category": "Fold_Fits_Final_Refit_Repeat_Not_Identical", "Count": int(sum(not a["Final_Refit_Repeat_Identical"] for a in folds))})
    return pd.DataFrame(rows, columns=["Warning_Category", "Count"])


# ================================================================ evaluation (formal run only)
def _nanmean_rows(values: np.ndarray) -> np.ndarray:
    valid = (~np.isnan(values)).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(values, axis=1)
    m[valid < MIN_CELLS] = np.nan
    return m


@evaluation
def ape_vector(all_cells: list[str], res: CVResult, eol: dict[str, float]) -> np.ndarray:
    """Absolute percentage error per cell (NaN outside the setting), in bootstrap cell order."""
    out = np.full(len(all_cells), np.nan)
    if not res.estimable:
        return out
    idx = {c: i for i, c in enumerate(all_cells)}
    for c in res.cells:
        y = eol[c]
        out[idx[c]] = abs(res.predictions[c] - y) / y * 100.0
    return out


@evaluation
def mean_pair_boot(ea: np.ndarray, eb: np.ndarray, draws: np.ndarray) -> tuple[dict, np.ndarray]:
    """v2 error_pair_ci with the mean instead of the median (implementation notes 2.2). Returns the row and the
    bootstrap vector of D = mean(A) - mean(B)."""
    n = int((~np.isnan(ea) & ~np.isnan(eb)).sum())
    out = {"Cells": n}
    if n < MIN_CELLS:
        return {**out, "Estimable": False, "Reason": "FEWER_THAN_5_CELLS"}, np.full(draws.shape[0], np.nan)
    diff = _nanmean_rows(ea[draws]) - _nanmean_rows(eb[draws])
    return {**out, "Observed_A": float(np.nanmean(ea)), "Observed_B": float(np.nanmean(eb)),
            "Observed_Difference": float(np.nanmean(ea) - np.nanmean(eb)), **_interval(diff)}, diff


@evaluation
def interval90(diff: np.ndarray) -> dict:
    missing = float(np.mean(np.isnan(diff)))
    if missing > MAX_BOOT_MISSING_FRACTION:
        return {"Estimable": False, "Reason": f"BOOT_MISSING_{missing:.3f}", "Boot_Missing_Fraction": missing}
    lo, hi = np.quantile(diff[~np.isnan(diff)], [0.05, 0.95], method="linear")
    return {"Estimable": True, "Reason": "", "CI90_Lower": float(lo), "CI90_Upper": float(hi), "Boot_Missing_Fraction": missing}


@evaluation
def evaluate_p(inp: Exp1Inputs, fits: Exp1Fits, draws: np.ndarray, exclude: frozenset = frozenset()):
    """Pair intervals for every (rule, definition pair, theta, Qref, method pair); returns (ci table, vectors, ape)."""
    rows, vectors, apes = [], {}, {}
    for rule, q1, q2, theta, cells in fits.settings:
        pair = f"{QREF_SHORT[q1]}-{QREF_SHORT[q2]}"
        for q in (q1, q2):
            eol = inp.eol.get((rule, q, theta), {})
            res = {m: fits.get(rule, q, theta, m, cells) for m in EXP1_MODELS}
            setting_ok = all(r.estimable for r in res.values()) and len(cells) >= MIN_CELLS
            ape = {m: ape_vector(inp.cells, res[m], eol) for m in EXP1_MODELS}
            apes[(rule, pair, theta, q)] = ape
            for a, b in EXP1_PAIRS:
                mp = f"{a}-{b}"
                base = {"Rule": rule, "Pair": pair, "Theta": theta, "Qref_Definition": q, "Method_Pair": mp,
                        "Decisive_H1": mp in H1_PAIRS, "Setting_Estimable": setting_ok}
                if setting_ok:
                    r, vec = mean_pair_boot(ape[a], ape[b], draws)
                else:
                    r, vec = {"Cells": len(cells), "Estimable": False, "Reason": "SETTING_NOT_ESTIMABLE"}, np.full(draws.shape[0], np.nan)
                vectors[(rule, pair, theta, q, mp)] = vec
                rows.append({**base, **r, "Better": _better(r, a, b, False)})
    return pd.DataFrame(rows), vectors, apes


@evaluation
def reversals(ci: pd.DataFrame) -> pd.DataFrame:
    rows = []
    idx = ci.set_index(["Rule", "Pair", "Theta", "Qref_Definition", "Method_Pair"])
    for (rule, pair, theta, mp), g in ci.groupby(["Rule", "Pair", "Theta", "Method_Pair"], sort=True):
        q1, q2 = g.Qref_Definition.iloc[0], g.Qref_Definition.iloc[-1]
        r1, r2 = idx.loc[(rule, pair, theta, q1, mp)], idx.loc[(rule, pair, theta, q2, mp)]
        est = bool(r1.Estimable) and bool(r2.Estimable)
        rows.append({"Rule": rule, "Pair": pair, "Theta": theta, "Method_Pair": mp, "Decisive_H1": mp in H1_PAIRS,
                     "Setting_1": q1, "Setting_2": q2, "Better_1": r1.Better, "Better_2": r2.Better,
                     "Estimable_Comparison": est,
                     "Supported_Reversal": bool(est and r1.Better and r2.Better and r1.Better != r2.Better)})
    return pd.DataFrame(rows)


@evaluation
def h1_units(ci: pd.DataFrame) -> pd.DataFrame:
    """Amendment 01 section 3: unit (rule, definition pair, Qref, theta); C1 on variance-full, C2 on variance-discharge."""
    rows = []
    for claim, mp, winner in (("C1", "variance-full", "full"), ("C2", "variance-discharge", "discharge")):
        for r in ci[ci.Method_Pair.eq(mp)].itertuples(index=False):
            est = bool(r.Setting_Estimable and r.Estimable)
            rows.append({"Rule": r.Rule, "Pair": r.Pair, "Qref_Definition": r.Qref_Definition, "Theta": r.Theta, "Claim": claim,
                         "Estimable": est, "Claim_Supported": bool(est and r.Better == winner)})
    return pd.DataFrame(rows)


@evaluation
def delta_baselines(inp: Exp1Inputs, fits: Exp1Fits, apes: dict, exclude: frozenset = frozenset()) -> pd.DataFrame:
    """Baseline B = nominal variance mean APE (amendment 01 section 4); delta = 0.10 * B."""
    rows = []
    for rule, q1, q2, theta, cells in fits.settings:
        pair = f"{QREF_SHORT[q1]}-{QREF_SHORT[q2]}"
        if q1 == "nominal":
            v = apes[(rule, pair, theta, "nominal")]["variance"]
            source = "pair_common_cells"
            n = int((~np.isnan(v)).sum())
        else:
            own = nominal_own_cells(inp, rule, theta, exclude)
            res = fits.get(rule, "nominal", theta, "variance", own)
            v = ape_vector(inp.cells, res, inp.eol.get((rule, "nominal", theta), {}))
            source = "nominal_own_labeled_cells"
            n = int((~np.isnan(v)).sum())
        ok = n >= MIN_CELLS
        rows.append({"Rule": rule, "Pair": pair, "Theta": theta, "Baseline_Source": source, "Baseline_Cells": n,
                     "Baseline_Defined": ok, "Baseline_Mean_APE": float(np.nanmean(v)) if ok else np.nan})
    return pd.DataFrame(rows)


@evaluation
def rmse_table(inp: Exp1Inputs, fits: Exp1Fits) -> pd.DataFrame:
    rows = []
    for rule, q1, q2, theta, cells in fits.settings:
        for q in (q1, q2):
            eol = inp.eol.get((rule, q, theta), {})
            for m in EXP1_MODELS:
                res = fits.get(rule, q, theta, m, cells)
                if res.estimable and res.cells:
                    e = np.array([res.predictions[c] - eol[c] for c in res.cells])
                    rmse = float(np.sqrt(np.mean(e ** 2)))
                else:
                    rmse = np.nan
                rows.append({"Rule": rule, "Pair": f"{QREF_SHORT[q1]}-{QREF_SHORT[q2]}", "Theta": theta, "Qref_Definition": q,
                             "Model": m, "Cells": len(cells), "RMSE_Cycles": rmse})
    return pd.DataFrame(rows)


@evaluation
def common_practice(inp: Exp1Inputs, theta: float, rules, qrefs, exclude: frozenset = frozenset()) -> pd.DataFrame:
    """Section 2.4: original split, theta 0.80, 6 definitions x 3 models; point estimates only."""
    rows = []
    for rule in rules:
        for q in qrefs:
            eol = {c: v for c, v in inp.eol.get((rule, q, theta), {}).items() if c not in exclude}
            by_split = {s: sorted(c for c in eol if inp.split[c] == s) for s in ("train", "test1", "test2")}
            for m in EXP1_MODELS:
                from threadpoolctl import threadpool_limits

                with threadpool_limits(limits=1):
                    preds, ok, _ = run_original_split(m, by_split, eol, inp.design[m])
                for s in ("train", "test1", "test2"):
                    cs = by_split[s]
                    rec = {"Rule": rule, "Qref_Definition": q, "Theta": theta, "Model": m, "Subset": s, "Labeled_Cells": len(cs)}
                    if not cs:
                        rows.append({**rec, "Status": "NO_LABEL", "Mean_APE": np.nan, "RMSE_Cycles": np.nan})
                    elif not ok:
                        rows.append({**rec, "Status": "NOT_FITTED", "Mean_APE": np.nan, "RMSE_Cycles": np.nan})
                    else:
                        e = np.array([preds[c] - eol[c] for c in cs])
                        ape = np.abs(e) / np.array([eol[c] for c in cs]) * 100.0
                        rows.append({**rec, "Status": "OK", "Mean_APE": float(ape.mean()), "RMSE_Cycles": float(np.sqrt(np.mean(e ** 2)))})
    return pd.DataFrame(rows)


@evaluation
def hyper_table(fits: Exp1Fits) -> pd.DataFrame:
    rows = []
    for (rule, q, theta, model, cells), res in fits.results.items():
        for h in res.hyper_rows:
            rows.append({"Rule": rule, "Qref_Definition": q, "Theta": theta, "Setting_Cells": len(cells), "Cell_Set_ID": cell_set_id(cells), **h})
    return pd.DataFrame(rows).sort_values(["Rule", "Qref_Definition", "Theta", "Cell_Set_ID", "Model", "Fold_ID"],
                                          kind="stable").reset_index(drop=True) if rows else pd.DataFrame()


@evaluation
def predictions_table(inp: Exp1Inputs, fits: Exp1Fits) -> pd.DataFrame:
    rows = []
    for (rule, q, theta, model, cells), res in fits.results.items():
        eol = inp.eol.get((rule, q, theta), {})
        for c in res.cells:
            rows.append({"Rule": rule, "Qref_Definition": q, "Theta": theta, "Setting_Cells": len(cells), "Cell_Set_ID": cell_set_id(cells), "Model": model,
                         "Cell_ID": c, "Fold_ID": inp.folds[c], "EOL": eol[c], "Predicted_EOL": res.predictions.get(c, np.nan)})
    return pd.DataFrame(rows).sort_values(["Rule", "Qref_Definition", "Theta", "Cell_Set_ID", "Model", "Cell_ID"],
                                          kind="stable").reset_index(drop=True)
