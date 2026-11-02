"""Centralized, frozen feature-extraction thresholds.

Every threshold used by the raw extractor is declared here.  No target
capacity, SOH, Qref, EOL, RUL, or future-life quantity is an extractor input.
"""

from __future__ import annotations


PIPELINE_VERSION = "SOH_FEATURE_EXTRACTION_V1"
RANDOM_SEED = 20260928

FEATURE_NAMES = (
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

FEATURE_UNITS = {
    "CV_Current_Taper_Slope_A_per_s": "A/s",
    "CV_Duration_s": "s",
    "CV_End_Current_A": "A",
    "CV_Median_Current_A": "A",
    "Charge_End_V": "V",
    "Charge_Max_V": "V",
    "Charge_Start_V": "V",
    "Charge_Terminal_Voltage_Std_V": "V",
    "Charge_Voltage_Rise_Slope_V_per_s": "V/s",
    "Pre_CV_CC_Median_Current_A": "A",
    "Pre_Discharge_Voltage_V": "V",
}

FEATURE_STAGE = {
    "CV_Current_Taper_Slope_A_per_s": "CHARGE_CV",
    "CV_Duration_s": "CHARGE_CV",
    "CV_End_Current_A": "CHARGE_CV",
    "CV_Median_Current_A": "CHARGE_CV",
    "Charge_End_V": "MAIN_CHARGE",
    "Charge_Max_V": "MAIN_CHARGE",
    "Charge_Start_V": "MAIN_CHARGE",
    "Charge_Terminal_Voltage_Std_V": "CHARGE_CV_TERMINAL_WINDOW",
    "Charge_Voltage_Rise_Slope_V_per_s": "FINAL_PRE_CV_CC",
    "Pre_CV_CC_Median_Current_A": "FINAL_PRE_CV_CC",
    "Pre_Discharge_Voltage_V": "PRE_DISCHARGE_POINT",
}

# Physical stage identification.  These values follow the previously
# validated XJTU/HUST protocol adapters.
POSITIVE_CHARGE_CURRENT_THRESHOLD_A = 0.01
NEGATIVE_DISCHARGE_CURRENT_THRESHOLD_A = -0.05
REST_CURRENT_ABS_THRESHOLD_A = 0.05
CV_TRANSITION_CURRENT_FRACTION = 0.90
CV_TRANSITION_VOLTAGE_BELOW_MAX_V = 0.03
HUST_PRE_CV_PLATEAU_MIN_CURRENT_A = 0.50
XJTU_CC_UPPER_QUANTILE = 0.75
CV_MAX_VOLTAGE_RANGE_V = 0.05

# Deterministic stage/slope requirements.
MIN_VALID_STAGE_SAMPLES = 5
ROBUST_SLOPE_CENTRAL_TIME_LOW = 0.20
ROBUST_SLOPE_CENTRAL_TIME_HIGH = 0.80
HUBER_TUNING_CONSTANT = 1.345
HUBER_MAX_ITERATIONS = 10
HUBER_CONVERGENCE_TOLERANCE = 1e-12
TERMINAL_CV_WINDOW_SECONDS = 30.0

# Review-only descriptive audits.  None of these thresholds removes a row.
OUTLIER_MAD_MULTIPLIER = 10.0
NEAR_TARGET_PROXY_ABS_SPEARMAN = 0.98
PROTOCOL_DOMINANCE_RATIO_THRESHOLD = 0.50
SYSTEMATIC_MISSINGNESS_RATE_THRESHOLD = 0.01
CONSTANT_ABS_TOLERANCE = 1e-12
CONSTANT_REL_TOLERANCE = 1e-9
NEAR_CONSTANT_ABS_TOLERANCE = 1e-9
NEAR_CONSTANT_REL_IQR_TOLERANCE = 1e-4
NEAR_CONSTANT_MODE_FRACTION = 0.99

XJTU_EXPECTED_OBSERVATIONS = 10427
HUST_EXPECTED_OBSERVATIONS = 146122

HUST_STATUS_CHARGE_HIGH = "Constant current charge"
HUST_STATUS_CHARGE_CCCV = "Constant current-constant voltage charge"
HUST_STATUS_DISCHARGE_FIRST = "Constant current discharge_0"

ALLOWED_EXTRACTION_STATUSES = ("PASS", "CHECK", "MISSING", "INVALID_EXTRACTION")
