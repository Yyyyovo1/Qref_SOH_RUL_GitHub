"""Frozen constants for XJTU A8 deterministic artifact reconstruction V1."""

from __future__ import annotations


PIPELINE_VERSION = "XJTU_A8_DETERMINISTIC_ARTIFACT_RECONSTRUCTION_V1"
DATASET = "XJTU"
EXPERIMENT = "ROBUSTNESS_A8"
FEATURE_SET = "A8_MINUS_CV_DURATION"
BOOTSTRAP_SEED = 20260928
BOOTSTRAP_REPLICATES = 5000
REPRODUCTION_TOLERANCE = 1e-12

RIDGE_MODEL = "Ridge Regression"
RF_MODEL = "Random Forest Regressor"
GB_MODEL = "Gradient Boosting Regressor"

EXPECTED_MEDIAN_MAE_Q_AH = {
    ("nominal", RIDGE_MODEL): 0.0384226186278847,
    ("nominal", RF_MODEL): 0.029654953676739,
    ("nominal", GB_MODEL): 0.0303989740702258,
    ("first_comparable", RIDGE_MODEL): 0.0398772340115457,
    ("first_comparable", RF_MODEL): 0.0330133475977013,
    ("first_comparable", GB_MODEL): 0.0304567669789295,
    ("early_median5_comparable", RIDGE_MODEL): 0.0372485544148703,
    ("early_median5_comparable", RF_MODEL): 0.0299277220226493,
    ("early_median5_comparable", GB_MODEL): 0.0278657242046231,
}

EXPECTED_RANKINGS = {
    "nominal": (RF_MODEL, GB_MODEL, RIDGE_MODEL),
    "first_comparable": (GB_MODEL, RF_MODEL, RIDGE_MODEL),
    "early_median5_comparable": (GB_MODEL, RF_MODEL, RIDGE_MODEL),
}

CSV_OUTPUTS = (
    "xjtu_a8_heldout_predictions.csv",
    "xjtu_a8_cell_metrics.csv",
    "xjtu_a8_reproduction_audit.csv",
    "xjtu_a8_corrected_difference_of_medians_ci.csv",
    "xjtu_a8_corrected_rankflip_frequency.csv",
    "xjtu_a8_corrected_rank_frequency.csv",
    "xjtu_a8_reconstruction_manifest.csv",
    "xjtu_a8_reconstruction_regression_tests.csv",
)

FINAL_STDOUT = (
    "XJTU A8 DETERMINISTIC RECONSTRUCTION COMPLETE —\n"
    "FROZEN A8 RESULTS REPRODUCED —\n"
    "CORRECTED BOOTSTRAP COMPLETE"
)

FAILURE_STDOUT = (
    "XJTU A8 DETERMINISTIC RECONSTRUCTION FAILED —\n"
    "FROZEN SUMMARY NOT REPRODUCED"
)
