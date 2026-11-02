from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_SCAN_CONFIG: dict[str, Any] = {
    "pipeline_phase": "SCAN_ONLY",
    "dataset": "HUST_77_LFP_GRAPHITE",
    "expected_canonical_cell_count": 77,
    "nominal_capacity_Ah": 1.1,
    "nominal_voltage_V": 3.3,
    "test_temperature_degC": 30.0,
    "expected_cycle_life_min": 1100,
    "expected_cycle_life_max": 2700,
    "fixed_absolute_EOL_threshold_Ah": 0.88,
    "expected_charge_protocol": {
        "stage1_CC_current_A": 5.5,
        "stage1_target_SOC_fraction": 0.80,
        "stage2_CC_current_A": 1.1,
        "stage2_target_voltage_V": 3.6,
        "stage3_CV_voltage_V": 3.6,
        "stage3_cutoff_current_A": 0.055,
    },
    "expected_discharge_protocol": {
        "stage_count": 4,
        "final_stage_current_A_approx": -1.1,
        "final_cutoff_voltage_V_approx": 2.0,
        "cell_specific_first_three_stages": True,
    },
    "raw_column_map": {
        "status": "Status",
        "cycle_index": "Cycle number",
        "current_mA": "Current (mA)",
        "voltage_V": "Voltage (V)",
        "capacity_mAh": "Capacity (mAh)",
        "time_s": "Time (s)",
    },
    "required_cell_keys": ["rul", "dq", "data"],
    "required_discharge_statuses": [
        "Constant current discharge_0",
        "Constant current discharge_1",
        "Constant current discharge_2",
        "Constant current discharge_3",
    ],
    "charge_stage1_status": "Constant current charge",
    "charge_cccv_status": "Constant current-constant voltage charge",
    "scan_thresholds": {
        "capacity_semantics_abs_tolerance_mAh": 1.0,
        "capacity_semantics_relative_tolerance": 0.005,
        "terminal_discharge_complete_max_min_voltage_V": 2.10,
        "terminal_discharge_end_max_voltage_V": 2.15,
        "charge_complete_min_max_voltage_V": 3.55,
        "cv_end_current_upper_A": 0.080,
        "severe_discharge_voltage_high_V": 4.00,
        "severe_discharge_voltage_low_V": 1.00,
        "capacity_jump_abs_threshold_Ah": 0.050,
        "capacity_jump_relative_threshold": 0.050,
        "protocol_current_abs_tolerance_A": 0.150,
        "protocol_current_relative_tolerance": 0.100,
        "protocol_change_min_consecutive_cycles": 3,
        "protocol_baseline_cycle_count": 20,
        "stopping_rule_near_0p88_abs_tolerance_Ah": 0.030,
        "capacity_semantics_required_match_fraction": 0.95,
        "full_discharge_required_traceable_fraction": 0.95,
        "protocol_change_hard_stop_cell_fraction": 0.10,
    },
    "representative_selection": {
        "count": 8,
        "method": "DETERMINISTIC_LIFE_CAPACITY_FINGERPRINT_COVERAGE",
        "life_capacity_quantile_targets": [
            [0.00, 0.00],
            [0.25, 0.50],
            [0.50, 1.00],
            [0.75, 0.00],
            [1.00, 0.50],
            [0.25, 1.00],
            [0.75, 1.00],
            [0.50, 0.50],
        ],
        "seen_fingerprint_penalty": 0.25,
    },
    "qref_feasibility_definitions": {
        "nominal": "1.100 Ah",
        "initial_test": "UNAVAILABLE_NO_SEPARATE_STANDARDIZED_BOL_TEST unless raw evidence proves otherwise",
        "first_comparable": "first valid full-discharge capacity under the cell-specific fixed protocol",
        "early_median5_comparable": "median of first five valid comparable full-discharge capacities sorted by numeric Raw_Cycle_Index",
    },
    "native_label_policy": "AUDIT_ONLY_NOT_GROUND_TRUTH",
    "raw_data_policy": {
        "modify": False,
        "delete": False,
        "move": False,
        "rename": False,
        "smooth": False,
        "interpolate": False,
        "capacity_correction": False,
    },
}


DEFAULT_PROTOCOL_VALIDATION_CONFIG: dict[str, Any] = {
    "pipeline_phase": "FOUR_CELL_RAW_PROTOCOL_VALIDATION_ONLY",
    "dataset": "HUST_77_LFP_GRAPHITE",
    "selected_cells": ["HUST_2-5", "HUST_9-3", "HUST_4-7", "HUST_10-3"],
    "cycles_per_cell": 4,
    "nominal_capacity_Ah": 1.1,
    "paper_reported_terminal_discharge_CRate": 1.0,
    "raw_column_map": deepcopy(DEFAULT_SCAN_CONFIG["raw_column_map"]),
    "charge_stage1_status": DEFAULT_SCAN_CONFIG["charge_stage1_status"],
    "charge_cccv_status": DEFAULT_SCAN_CONFIG["charge_cccv_status"],
    "validation_thresholds": {
        # Negative-current samples are considered physical discharge samples.
        "discharge_current_detection_threshold_A": -0.05,
        # The low-rate CC/CV transition is the first near-maximum-voltage sample
        # whose current has fallen below this fraction of the CC plateau.
        "cv_transition_current_fraction": 0.90,
        "cv_transition_voltage_below_max_V": 0.03,
        # Paper-vs-raw terminal-current classification bands.
        "terminal_1C_current_abs_min_A": 1.08,
        "terminal_1C_current_abs_max_A": 1.12,
        "terminal_2C_current_abs_min_A": 2.15,
        "terminal_2C_current_abs_max_A": 2.25,
        # A terminal stage must reach the low-voltage region.  The lower bound
        # is a plausibility guard, not a value to which measurements are clipped.
        "terminal_low_voltage_min_V": 1.85,
        "terminal_low_voltage_max_V": 2.05,
        # Exact small-scale capacity-semantics regression requested by the study.
        "capacity_match_abs_tolerance_mAh": 0.001,
        "stage_sum_match_abs_tolerance_mAh": 0.001,
        "within_cell_terminal_current_consistency_A": 0.05,
    },
    "raw_data_policy": deepcopy(DEFAULT_SCAN_CONFIG["raw_data_policy"]),
    "prohibited_analyses": [
        "QREF",
        "SOH",
        "EOL",
        "RUL",
        "MODEL_TRAINING",
        "FULL_77_CELL_ANALYSIS",
    ],
}


DEFAULT_8CELL_PILOT_CONFIG: dict[str, Any] = {
    "pipeline_phase": "EIGHT_CELL_QREF_EOL_PILOT_ONLY",
    "dataset": "HUST_77_LFP_GRAPHITE",
    "selected_cells": [
        "HUST_2-5",
        "HUST_7-1",
        "HUST_9-3",
        "HUST_1-5",
        "HUST_3-6",
        "HUST_4-5",
        "HUST_4-7",
        "HUST_10-3",
    ],
    "nominal_capacity_Ah": 1.1,
    "eol_soh_threshold": 0.80,
    "fixed_absolute_eol_threshold_Ah": 0.880000,
    "capacity_source": "NATIVE_DQ_MAH_DIVIDED_BY_1000",
    "initial_test_qref_status": "UNAVAILABLE_NO_SEPARATE_STANDARDIZED_BOL_TEST",
    "first_comparable_required_count": 1,
    "early_median5_required_count": 5,
    "paper_reported_terminal_discharge_CRate": 1.0,
    "raw_observed_terminal_discharge_CRate_approx": 2.0,
    "paper_vs_raw_protocol_discrepancy": "YES",
    "protocol_qc_basis": "RAW_OBSERVED_PROTOCOL",
    "qref_definitions": [
        "nominal",
        "first_comparable",
        "early_median5_comparable",
    ],
    "eol_definitions": [
        "nominal",
        "first_comparable",
        "early_median5_comparable",
        "fixed_absolute_0p88Ah",
    ],
    "pairwise_comparisons": [
        ["nominal", "first_comparable"],
        ["nominal", "early_median5_comparable"],
        ["first_comparable", "early_median5_comparable"],
    ],
    "pairwise_delta_sign_convention": (
        "EOL_B_MINUS_EOL_A; negative means definition B reaches its threshold earlier"
    ),
    "raw_data_policy": deepcopy(DEFAULT_SCAN_CONFIG["raw_data_policy"]),
    "prohibited_analyses": ["FULL_77_CELL_ANALYSIS", "MODEL_TRAINING"],
}


DEFAULT_77CELL_FULL_CONFIG: dict[str, Any] = {
    "pipeline_phase": "HUST_77CELL_FINAL_V1",
    "dataset": "HUST_77_LFP_GRAPHITE",
    "expected_canonical_cell_count": 77,
    "expected_raw_cycle_count": 146122,
    "nominal_capacity_Ah": 1.1,
    "eol_soh_threshold": 0.80,
    "fixed_absolute_eol_threshold_Ah": 0.880000,
    "capacity_source": "NATIVE_DQ_MAH_DIVIDED_BY_1000",
    "initial_test_qref_status": "UNAVAILABLE_NO_SEPARATE_STANDARDIZED_BOL_TEST",
    "paper_reported_terminal_discharge_CRate": 1.0,
    "raw_observed_terminal_discharge_CRate_approx": 2.0,
    "paper_vs_raw_protocol_discrepancy": "YES",
    "protocol_qc_basis": "RAW_OBSERVED_PROTOCOL",
    "validated_raw_protocol_limits": {
        "terminal_stage_abs_current_min_A": 2.15,
        "terminal_stage_abs_current_max_A": 2.25,
        "terminal_stage_CRate_min": 1.95,
        "terminal_stage_CRate_max": 2.05,
    },
    "qref_definitions": [
        "nominal",
        "first_comparable",
        "early_median5_comparable",
    ],
    "eol_definitions": [
        "nominal",
        "first_comparable",
        "early_median5_comparable",
        "fixed_absolute_0p88Ah",
    ],
    "pairwise_comparisons": [
        ["nominal", "first_comparable"],
        ["nominal", "early_median5_comparable"],
        ["first_comparable", "early_median5_comparable"],
    ],
    "pairwise_delta_sign_convention": (
        "EOL_B_MINUS_EOL_A; negative means definition B reaches its threshold earlier"
    ),
    "minimum_observed_earlier_by_cycles_definition": (
        "LOWER_BOUND_ONLY: Last_Valid_Raw_Cycle minus exact measured-Qref EOL cycle; "
        "not an exact EOL shift"
    ),
    "native_endpoint_policy": "AUDIT_ONLY_NOT_QREF_DERIVED_EOL_OR_RUL",
    "stopping_rule_near_0p88_abs_tolerance_Ah": 0.030,
    "frozen_pilot_cells": list(DEFAULT_8CELL_PILOT_CONFIG["selected_cells"]),
    "raw_data_policy": deepcopy(DEFAULT_SCAN_CONFIG["raw_data_policy"]),
    "prohibited_analyses": ["MODEL_TRAINING"],
}


def make_scan_config(workspace: Path, input_dir: Path, output_dir: Path) -> dict[str, Any]:
    config = deepcopy(DEFAULT_SCAN_CONFIG)
    config["workspace"] = str(workspace.resolve())
    config["input_directory"] = str(input_dir.resolve())
    config["output_directory"] = str(output_dir.resolve())
    return config


def make_protocol_validation_config(
    workspace: Path, input_dir: Path, scan_dir: Path, output_dir: Path
) -> dict[str, Any]:
    config = deepcopy(DEFAULT_PROTOCOL_VALIDATION_CONFIG)
    config["workspace"] = str(workspace.resolve())
    config["input_directory"] = str(input_dir.resolve())
    config["scan_directory"] = str(scan_dir.resolve())
    config["output_directory"] = str(output_dir.resolve())
    return config


def make_8cell_pilot_config(
    workspace: Path,
    input_dir: Path,
    scan_dir: Path,
    protocol_validation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = deepcopy(DEFAULT_8CELL_PILOT_CONFIG)
    config["workspace"] = str(workspace.resolve())
    config["input_directory"] = str(input_dir.resolve())
    config["scan_directory"] = str(scan_dir.resolve())
    config["protocol_validation_directory"] = str(
        protocol_validation_dir.resolve()
    )
    config["output_directory"] = str(output_dir.resolve())
    return config


def make_77cell_full_config(
    workspace: Path,
    input_dir: Path,
    scan_dir: Path,
    protocol_validation_dir: Path,
    pilot_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = deepcopy(DEFAULT_77CELL_FULL_CONFIG)
    config["workspace"] = str(workspace.resolve())
    config["input_directory"] = str(input_dir.resolve())
    config["scan_directory"] = str(scan_dir.resolve())
    config["protocol_validation_directory"] = str(
        protocol_validation_dir.resolve()
    )
    config["frozen_pilot_directory"] = str(pilot_dir.resolve())
    config["output_directory"] = str(output_dir.resolve())
    return config


__all__ = [
    "DEFAULT_77CELL_FULL_CONFIG",
    "DEFAULT_8CELL_PILOT_CONFIG",
    "DEFAULT_PROTOCOL_VALIDATION_CONFIG",
    "DEFAULT_SCAN_CONFIG",
    "make_77cell_full_config",
    "make_8cell_pilot_config",
    "make_protocol_validation_config",
    "make_scan_config",
]
