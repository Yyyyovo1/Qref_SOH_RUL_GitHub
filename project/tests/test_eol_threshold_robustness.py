from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eol_threshold_robustness_repro.config import (
    EXACT,
    INTERVAL,
    LEFT,
    RIGHT,
    PipelinePaths,
)
from eol_threshold_robustness_repro.pipeline import (
    build_cell_level,
    compare_frozen_80,
    compute_effect_bounds,
    detect_hust_eol,
    detect_xjtu_eol,
    load_frozen_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


def _xjtu_observations(cycles: list[int], capacities: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Raw_Cycle_Index": cycles,
            "Diagnostic_Index": range(1, len(cycles) + 1),
            "Cumulative_Aging_Operation_Count": cycles,
            "Capacity_Ah": capacities,
        }
    )


def test_xjtu_exact_interval_right_and_left_rules() -> None:
    dense = _xjtu_observations([1, 2, 3], [2.0, 1.7, 1.5])
    exact = detect_xjtu_eol(dense, 1.6, dense_measurement=True)
    assert exact.status == EXACT
    assert exact.exact_cycle == 3

    sparse = _xjtu_observations([1, 6, 11], [2.0, 1.7, 1.5])
    interval = detect_xjtu_eol(sparse, 1.6, dense_measurement=False)
    assert interval.status == INTERVAL
    assert interval.lower_bound == 6
    assert interval.upper_bound == 11
    assert interval.exact_cycle is None

    right = detect_xjtu_eol(dense, 1.4, dense_measurement=True)
    assert right.status == RIGHT
    assert right.exact_cycle is None
    assert right.lower_bound == 3

    left = detect_xjtu_eol(dense, 2.1, dense_measurement=True)
    assert left.status == LEFT
    assert left.exact_cycle is None
    assert left.upper_bound == 1


def test_hust_dense_and_genuine_gap_rules() -> None:
    dense = pd.DataFrame(
        {"Raw_Cycle_Index": [1, 2, 3], "Primary_Capacity_Ah": [1.1, 1.0, 0.8]}
    )
    exact = detect_hust_eol(dense, 0.9)
    assert exact.status == EXACT
    assert exact.exact_cycle == 3

    gap = pd.DataFrame(
        {"Raw_Cycle_Index": [1, 2, 4], "Primary_Capacity_Ah": [1.1, 1.0, 0.8]}
    )
    interval = detect_hust_eol(gap, 0.9)
    assert interval.status == INTERVAL
    assert interval.lower_bound == 2
    assert interval.upper_bound == 4
    assert interval.exact_cycle is None


def test_effect_delta_exists_only_for_two_exact_events() -> None:
    nominal = {
        "EOL_Status": EXACT,
        "EOL_Exact_Cycle": 100,
        "EOL_Lower_Bound": np.nan,
        "EOL_Upper_Bound": np.nan,
        "Last_Observed_Cycle": 120,
    }
    measured_exact = {
        "EOL_Status": EXACT,
        "EOL_Exact_Cycle": 90,
        "EOL_Lower_Bound": np.nan,
        "EOL_Upper_Bound": np.nan,
        "Last_Observed_Cycle": 120,
    }
    exact = compute_effect_bounds(nominal, measured_exact)
    assert exact["Delta_EOL_Cycles"] == -10

    measured_interval = {
        "EOL_Status": INTERVAL,
        "EOL_Exact_Cycle": np.nan,
        "EOL_Lower_Bound": 85,
        "EOL_Upper_Bound": 95,
        "Last_Observed_Cycle": 120,
    }
    censored = compute_effect_bounds(nominal, measured_interval)
    assert pd.isna(censored["Delta_EOL_Cycles"])


def test_frozen_80_percent_results_are_reproduced() -> None:
    inputs = load_frozen_inputs(PipelinePaths(ROOT))
    cell = build_cell_level(inputs)
    passed, summary, detail = compare_frozen_80(cell, inputs.cross_cell_80)
    assert passed, (summary, detail.head(20).to_dict("records"))
    assert summary["row_count_current"] == 451
    assert summary["xjtu_mismatch_count"] == 0
    assert summary["hust_mismatch_count"] == 0
