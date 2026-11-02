"""Frozen constants for SOH ML Robustness + Statistical Uncertainty V1."""

from __future__ import annotations

from soh_primary_benchmark_repro.config import PRIMARY_FEATURES


PIPELINE_VERSION = "SOH_ML_ROBUSTNESS_V1"
RANDOM_SEED = 20260928
BOOTSTRAP_REPLICATES = 5000
MAX_PARALLEL_FOLDS = 3

A8_FEATURES = tuple(
    feature for feature in PRIMARY_FEATURES if feature != "CV_Duration_s"
)
B10_FEATURES = (*PRIMARY_FEATURES, "Raw_Cycle_Index")
POSTCYCLE5_FEATURES = PRIMARY_FEATURES

EXPERIMENTS = (
    ("ROBUSTNESS_A8", "A8_MINUS_CV_DURATION", A8_FEATURES, False),
    ("ROBUSTNESS_B10", "B10_PHYSICS_PLUS_CYCLE_INDEX", B10_FEATURES, False),
    ("ROBUSTNESS_POSTCYCLE5", "A9_POSTCYCLE5", POSTCYCLE5_FEATURES, True),
)

MODEL_PAIRS = (
    ("Random Forest Regressor", "Gradient Boosting Regressor"),
    ("Random Forest Regressor", "Ridge Regression"),
    ("Gradient Boosting Regressor", "Ridge Regression"),
)

CSV_OUTPUTS = (
    "soh_robustness_a8_summary.csv",
    "soh_robustness_b10_summary.csv",
    "soh_robustness_postcycle5_summary.csv",
    "soh_robustness_rankings.csv",
    "soh_qref_paired_bootstrap_ci.csv",
    "soh_model_pair_bootstrap_ci.csv",
    "soh_bootstrap_rank_frequency.csv",
    "soh_scale_effect_audit.csv",
    "soh_robustness_regression_tests.csv",
)
