"""Frozen constants for SOH Primary ML Benchmark V1."""

from __future__ import annotations


PIPELINE_VERSION = "SOH_PRIMARY_BENCHMARK_V1"
RANDOM_SEED = 20260928

PRIMARY_FEATURES = (
    "CV_Current_Taper_Slope_A_per_s",
    "CV_Duration_s",
    "CV_End_Current_A",
    "CV_Median_Current_A",
    "Charge_End_V",
    "Charge_Max_V",
    "Charge_Start_V",
    "Charge_Terminal_Voltage_Std_V",
    "Pre_Discharge_Voltage_V",
)

PROTOCOL_SHORTCUT_FEATURES = (
    "Charge_Voltage_Rise_Slope_V_per_s",
    "Pre_CV_CC_Median_Current_A",
)

ALL_EXTRACTED_FEATURES = (
    "CV_Current_Taper_Slope_A_per_s",
    "CV_Duration_s",
    "CV_End_Current_A",
    "CV_Median_Current_A",
    "Charge_End_V",
    "Charge_Max_V",
    "Charge_Start_V",
    "Charge_Terminal_Voltage_Std_V",
    "Charge_Voltage_Rise_Slope_V_per_s",
    "Pre_CV_CC_Median_Current_A",
    "Pre_Discharge_Voltage_V",
)

COMMON_QREFS = (
    "nominal",
    "first_comparable",
    "early_median5_comparable",
)

SECONDARY_XJTU_QREFS = ("nominal", "initial_test")

MODEL_NAMES = (
    "Ridge Regression",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
)

QREF_PAIRS = (
    ("nominal", "first_comparable"),
    ("nominal", "early_median5_comparable"),
    ("first_comparable", "early_median5_comparable"),
)

XJTU_EXPECTED_MODEL_ROWS = 10427
HUST_EXPECTED_PREMASK_ROWS = 146122
N_FOLDS = 5
MOST_FOLDS_THRESHOLD = 3

CSV_OUTPUTS = (
    "soh_primary_feature_freeze_v1.csv",
    "soh_model_eligible_mask_xjtu.csv",
    "soh_model_eligible_mask_hust.csv",
    "soh_primary_predictions.csv",
    "soh_primary_cell_metrics.csv",
    "soh_primary_fold_metrics.csv",
    "soh_primary_summary_metrics.csv",
    "soh_primary_model_rankings.csv",
    "soh_primary_ranking_stability.csv",
    "soh_primary_qref_paired_effects.csv",
    "soh_primary_xjtu_batch_metrics.csv",
    "soh_secondary_xjtu_initial_test_metrics.csv",
    "soh_primary_regression_tests.csv",
)

