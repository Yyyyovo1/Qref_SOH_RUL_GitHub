"""Constants fixed by PROTOCOL.md (preregistered at commit 3a4ae5c)."""

from __future__ import annotations

from soh_primary_benchmark_repro.config import PRIMARY_FEATURES

PIPELINE_VERSION = "RUL_RANKING_PILOT_V1"
PROTOCOL_COMMIT = "3a4ae5c"
OUTPUT_DIR_NAME = "rul_ranking_pilot_v1"

DATASETS = ("XJTU", "HUST")
QREFS = ("nominal", "first_comparable", "early_median5_comparable")
QREF_SHORT = {"nominal": "nominal", "first_comparable": "first", "early_median5_comparable": "first5"}
QREF_PAIRS = (
    ("nominal", "first_comparable"),
    ("nominal", "early_median5_comparable"),
    ("first_comparable", "early_median5_comparable"),
)
THETAS = tuple(round(0.75 + 0.01 * i, 2) for i in range(21))
CAL_CONDITIONS = ("CAL", "NOCAL")
METHODS = ("RF", "TRAJ", "WAFT", "BASE")
METHODS_BY_CAL = {"CAL": METHODS, "NOCAL": ("RF", "WAFT", "BASE")}
EVAL_SETS = ("E1", "E2", "E3")

NOMINAL_CAPACITY_AH = {"XJTU": 2.0, "HUST": 1.1}
XJTU_DENSE_BATCHES = frozenset({1, 2})
# Values found by the read-only drafting check (PROTOCOL.md section 3); the run recomputes them.
PROTOCOL_T0 = {"XJTU": 107, "HUST": 294}

# Endpoint classes used in this pilot.
EXACT = "EXACT"
INTERVAL = "INTERVAL"
RIGHT = "RIGHT"
BASELINE = "BASELINE"

A9_FEATURES = tuple(PRIMARY_FEATURES)
LEVEL_LAST_N = 3
CAL_EXTRA_FEATURES = ("Capacity_t0_Ah", "Capacity_Slope_Ah_per_cycle", "Qref_Ah", "Margin_To_Threshold_Ah")
FORBIDDEN_NOCAL_TOKENS = ("capacity", "qref", "margin")

RANDOM_SEED = 20260928
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
}
WAFT_PENALIZER = 0.1
TRAJ_HORIZON_FACTOR = 5.0

MIN_CELLS = 5
N_BOOTSTRAP = 5000
MAX_BOOT_MISSING_FRACTION = 0.05
STRONG_MIN_THETA_POINTS = 6  # ceil(21 / 4)
XJTU_DRAW_SHA256 = "2e8a1afa9101de2c4af7927c31f1e112adde5be2922437d03422aa90207e1ded"
N_FOLDS = 5
