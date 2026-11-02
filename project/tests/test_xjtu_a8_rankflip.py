from __future__ import annotations

import numpy as np
import pandas as pd

from soh_primary_benchmark_repro.config import COMMON_QREFS, MODEL_NAMES
from xjtu_a8_rankflip_repro.config import (
    GB_MODEL,
    RF_MODEL,
    RIDGE_MODEL,
)
from xjtu_a8_rankflip_repro.pipeline import (
    _paired_rf_gb_bootstrap,
    _qref_rankflip_bootstrap,
    _rank_frequency,
)


def _synthetic_cell_metrics() -> pd.DataFrame:
    rows = []
    for cell_index in range(7):
        cell_id = f"cell_{cell_index + 1}"
        base = 0.01 + 0.001 * cell_index
        for qref in COMMON_QREFS:
            if qref == "nominal":
                rf, gb = base, base + 0.004
            else:
                rf, gb = base + 0.004, base
            for model, value in (
                (RF_MODEL, rf),
                (GB_MODEL, gb),
                (RIDGE_MODEL, base + 0.02),
            ):
                rows.append(
                    {
                        "Analysis_ID": "ROBUSTNESS_A8",
                        "Dataset": "XJTU",
                        "Cell_ID": cell_id,
                        "Fold_ID": (cell_index % 5) + 1,
                        "Qref_Definition": qref,
                        "Model": model,
                        "MAE_Q_Ah": value,
                    }
                )
    return pd.DataFrame(rows)


def test_rf_gb_bootstrap_is_paired_and_deterministic() -> None:
    metrics = _synthetic_cell_metrics()
    first = _paired_rf_gb_bootstrap(metrics, replicates=200)
    second = _paired_rf_gb_bootstrap(metrics, replicates=200)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 3
    assert first["Paired_By_Cell_ID"].eq("YES").all()
    assert first["Cycle_Level_Resampling"].eq("NO").all()
    assert first["Cell_Count"].eq(7).all()
    nominal = first.loc[first["Qref_Definition"].eq("nominal")].iloc[0]
    assert np.isclose(nominal["Observed_Median_Difference"], 0.004)


def test_joint_rank_frequency_has_complete_grid_and_sums_to_one() -> None:
    result = _rank_frequency(_synthetic_cell_metrics(), replicates=200)
    assert len(result) == len(COMMON_QREFS) * len(MODEL_NAMES)
    sums = result.groupby("Qref_Definition")["Bootstrap_Rank1_Frequency"].sum()
    assert np.allclose(sums.to_numpy(float), 1.0)
    assert result["Joint_Three_Model_Resampling"].eq("YES").all()
    assert result["Cycle_Level_Resampling"].eq("NO").all()


def test_direct_rankflip_uses_the_same_cells_across_qrefs() -> None:
    result = _qref_rankflip_bootstrap(
        _synthetic_cell_metrics(), replicates=200
    )
    assert len(result) == 2
    assert result["Paired_Across_Qrefs_By_Cell_ID"].eq("YES").all()
    assert result["Observed_Nominal_Order"].eq("RF_LT_GB").all()
    assert result["Observed_Measured_Order"].eq("GB_LT_RF").all()
    assert result["Observed_Sample_RankFlip_Stability"].eq(1.0).all()
