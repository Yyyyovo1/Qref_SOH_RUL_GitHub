"""Constants fixed by PROTOCOL_v2.md (commit 4cdf0a2) and the MIT freeze (commit c4b2787)."""

from __future__ import annotations

from rul_ranking_pilot.config import (  # unchanged from v1 (PROTOCOL_v2 sections 1e, 5.1, 7)
    A9_FEATURES,
    BASELINE,
    EXACT,
    INTERVAL,
    LEVEL_LAST_N,
    MAX_BOOT_MISSING_FRACTION,
    METHODS,
    MIN_CELLS,
    N_BOOTSTRAP,
    N_FOLDS,
    QREF_PAIRS,
    QREF_SHORT,
    QREFS,
    RANDOM_SEED,
    RF_PARAMS,
    RIGHT,
    STRONG_MIN_THETA_POINTS,
    THETAS,
    TRAJ_HORIZON_FACTOR,
    WAFT_PENALIZER,
    XJTU_DRAW_SHA256,
)

PIPELINE_VERSION = "RUL_RANKING_PILOT_V2"
PROTOCOL_COMMIT = "4cdf0a2"
MIT_FREEZE_COMMIT = "c4b2787"
OUTPUT_DIR_NAME = "rul_ranking_pilot_v2"
MIT_DIR = "outputs/mit_124_final_v1"
T0_FILE = "outputs/mit_124_final_v1/rul_pilot_v2_t0_all_datasets.csv"

DATASETS = ("XJTU", "HUST", "MIT")
NOMINAL_CAPACITY_AH = {"XJTU": 2.0, "HUST": 1.1, "MIT": 1.1}
DATASET_FEATURES = {"XJTU": A9_FEATURES, "HUST": A9_FEATURES, "MIT": ("IR_Ohm", "Tavg_C", "Chargetime_min")}
CAPACITY_FEATURES = ("Capacity_t_Ah", "Capacity_Slope_Ah_per_cycle", "Prediction_Time_Raw_Cycle")
MARGIN_FEATURE = "Margin_To_Threshold_Ah"
EVAL_SETS = ("E1", "E2", "E3")
TRAJ_MAX_FALLBACK_FRACTION = 0.50
C_INDEX_MIN_PAIRS = 10
REBOUND_DESCRIPTIVE_THETAS = THETAS
