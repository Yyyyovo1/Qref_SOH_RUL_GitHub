"""Frozen constants for the targeted XJTU A8 rank-flip verification."""

from __future__ import annotations


PIPELINE_VERSION = "XJTU_A8_RANKFLIP_VERIFICATION_V1"
DATASET = "XJTU"
ANALYSIS_ID = "ROBUSTNESS_A8"
FEATURE_SET = "A8_MINUS_CV_DURATION"
RANDOM_SEED = 20260928
BOOTSTRAP_REPLICATES = 5000

RF_MODEL = "Random Forest Regressor"
GB_MODEL = "Gradient Boosting Regressor"
RIDGE_MODEL = "Ridge Regression"

CSV_OUTPUTS = (
    "xjtu_a8_rf_gb_bootstrap_ci.csv",
    "xjtu_a8_rank_frequency.csv",
    "xjtu_a8_qref_rankflip_bootstrap.csv",
    "xjtu_a8_rankflip_regression_tests.csv",
)
