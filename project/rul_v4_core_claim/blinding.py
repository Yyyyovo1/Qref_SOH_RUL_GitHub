"""Blinding guard (AMENDMENT_01 section 11).

Every function that computes a model error, a method difference, an interval, a reversal, an equivalence class, a
detection probability or a verdict is decorated with `@evaluation`. It raises unless the caller is inside
`allow_evaluation()`, which only the formal run (and tests) enter. Development-run outputs are written through
`check_columns`, which rejects column names that could carry such quantities.
"""

from __future__ import annotations

import contextlib
import functools
import re

_STATE = {"allowed": False}

FORBIDDEN_PARTS = frozenset({
    "Pred", "Prediction", "Predictions", "Error", "Errors", "APE", "RMSE", "MSE", "CI", "CI95", "CI90", "Diff",
    "Difference", "Better", "Reversal", "Reversals", "Verdict", "Alpha", "L1", "Ratio", "Coef", "Coefficient",
    "Lambda", "Observed", "Median", "Mean", "SE", "Equivalence", "Power", "Supported", "Delta",
})
# Columns that are allowed in development outputs although a part matches (AMENDMENT_01 section 11 and v3 practice).
ALLOWED_COLUMNS = frozenset({
    "Max_Rel_Diff",                 # check 2 TRAJ equivalence: same method under two threshold spellings
    "Prediction_Time_Raw_Cycle",    # an input feature (the prediction time t), not a model output
    "Imputer_Median_SHA256",
    "NegControl_All_Rows_Match", "Replacement_Mean_Control_Differs",
    "Scaler_Mean_SHA256", "NegControl_Mean_Differs", "NegControl_Std_Differs",
    "Recomputed_Max_Rel_Diff", "Max_Abs_Diff_Zero_Reference",  # check S: features vs archived reference
    "DeltaQ_Mean",                  # Severson input feature (mean of Delta Q), not a model output
})


class BlindingError(RuntimeError):
    pass


@contextlib.contextmanager
def allow_evaluation():
    prev = _STATE["allowed"]
    _STATE["allowed"] = True
    try:
        yield
    finally:
        _STATE["allowed"] = prev


def evaluation_allowed() -> bool:
    return _STATE["allowed"]


def evaluation(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _STATE["allowed"]:
            raise BlindingError(f"EVALUATION_CALLED_WHILE_BLINDED:{func.__module__}.{func.__name__}")
        return func(*args, **kwargs)
    return wrapper


def _parts(name: str) -> set[str]:
    return set(re.split(r"[_\W]+", str(name))) - {""}


def _is_input_feature(name: str) -> bool:
    """v2/v3 design features are named <raw feature>__level / __slope (e.g. CV_Median_Current_A__level)."""
    return bool(re.fullmatch(r".+__(level|slope)", str(name)))


def forbidden_columns(columns) -> list[str]:
    return [c for c in columns if c not in ALLOWED_COLUMNS and not _is_input_feature(c) and _parts(c) & FORBIDDEN_PARTS]


def check_columns(columns, where: str) -> None:
    bad = forbidden_columns(columns)
    if bad:
        raise BlindingError(f"FORBIDDEN_COLUMNS_IN_DEV_OUTPUT:{where}:{','.join(map(str, bad))}")
