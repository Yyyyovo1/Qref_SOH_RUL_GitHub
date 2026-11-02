from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_CONFIG_KEYS = {
    "pipeline_version",
    "expected_battery_ids",
    "canonical_filename_template",
    "duplicate_filename_template",
    "nominal_capacity_Ah",
    "charge_cutoff_V",
    "discharge_cutoff_V",
    "cycle1_charge_current_A",
    "cycle1_cv_cutoff_A",
    "cycle1_cv_cutoff_upper_A",
    "cycle1_discharge_current_A",
    "cycle1_discharge_current_abs_tolerance_A",
    "standard_charge_current_A",
    "standard_cv_cutoff_A",
    "standard_cv_cutoff_upper_A",
    "standard_discharge_current_A",
    "standard_discharge_current_abs_tolerance_A",
    "charge_complete_min_V",
    "discharge_complete_max_V",
    "discharge_end_voltage_max_V",
    "charge_current_detection_threshold_A",
    "discharge_current_detection_threshold_A",
    "voltage_spike_current_threshold_A",
    "voltage_spike_threshold_V",
    "voltage_jump_threshold_V",
    "capacity_match_abs_tolerance_Ah",
    "capacity_match_relative_tolerance",
    "capacity_jump_abs_threshold_Ah",
    "capacity_jump_relative_threshold",
    "EOL_SOH_threshold",
    "fixed_absolute_EOThreshold_Ah",
    "battery1_capacity_regression_tolerance_Ah",
    "battery1_expected_cycle_count",
    "battery1_expected_capacities_Ah",
    "battery1_cycle1_expected_discharge_current_A",
    "battery1_cycle2_expected_discharge_current_A",
    "battery1_current_regression_tolerance_A",
    "battery1_cycle390_expected_min_voltage_V",
    "battery1_cycle390_voltage_tolerance_V",
    "battery1_qref_expected_Ah",
    "csv_float_format",
    "figure_dpi",
}


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the single authoritative JSON configuration."""

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    missing = sorted(REQUIRED_CONFIG_KEYS.difference(config))
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    expected_ids = [int(value) for value in config["expected_battery_ids"]]
    if expected_ids != list(range(1, 9)):
        raise ValueError("expected_battery_ids must be exactly [1, 2, 3, 4, 5, 6, 7, 8]")

    positive_keys = (
        "nominal_capacity_Ah",
        "charge_cutoff_V",
        "discharge_cutoff_V",
        "charge_complete_min_V",
        "discharge_complete_max_V",
        "discharge_end_voltage_max_V",
        "capacity_match_abs_tolerance_Ah",
        "capacity_match_relative_tolerance",
        "capacity_jump_abs_threshold_Ah",
        "capacity_jump_relative_threshold",
        "EOL_SOH_threshold",
        "fixed_absolute_EOThreshold_Ah",
    )
    for key in positive_keys:
        if float(config[key]) <= 0:
            raise ValueError(f"Config value {key} must be positive")

    if float(config["discharge_current_detection_threshold_A"]) >= 0:
        raise ValueError("discharge_current_detection_threshold_A must be negative")
    if float(config["voltage_spike_current_threshold_A"]) >= 0:
        raise ValueError("voltage_spike_current_threshold_A must be negative")
    return config


def canonical_config_text(path: Path) -> str:
    """Return normalized JSON text used in the processing log."""

    config = load_config(path)
    return json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True)

