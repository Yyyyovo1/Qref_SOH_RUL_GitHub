from __future__ import annotations

from copy import deepcopy
from typing import Any


EXPECTED_CANONICAL_COUNTS = {1: 8, 2: 15, 3: 8, 4: 8, 5: 8, 6: 8}
NOMINAL_CAPACITY_AH = 2.0


BASE_CONFIG: dict[str, Any] = {
    "config_status": "DRAFT_AWAITING_HUMAN_PROTOCOL_REVIEW",
    "nominal_capacity_Ah": NOMINAL_CAPACITY_AH,
    "charge_complete_min_V": 4.18,
    "full_discharge_min_voltage_max_V": 2.55,
    "full_discharge_end_voltage_max_V": 2.60,
    "partial_discharge_target_V": 3.00,
    "partial_discharge_voltage_tolerance_V": 0.18,
    "charge_current_detection_threshold_A": 0.05,
    "discharge_current_detection_threshold_A": -0.05,
    "rest_current_abs_threshold_A": 0.05,
    # The CC plateau can be very short when a partial-aging cycle ends at high SOC.
    # A near-maximum robust quantile captures the commanded current without using
    # a single unconstrained raw maximum.
    "charge_cc_estimation_quantile": 0.999,
    "voltage_spike_current_threshold_A": -0.10,
    "voltage_spike_threshold_V": 4.30,
    "voltage_jump_threshold_V": 0.50,
    "capacity_match_abs_tolerance_Ah": 0.005,
    "capacity_match_relative_tolerance": 0.01,
    "capacity_jump_abs_threshold_Ah": 0.08,
    "capacity_jump_relative_threshold": 0.05,
    "EOL_SOH_threshold": 0.80,
    "fixed_absolute_EOThreshold_Ah": 1.600,
    "minimum_protocol_confidence_for_auto_match": 0.75,
    "known_information_anomaly_policy": {
        "unknown_or_unexplained_anomalies_require_human_review": True,
        "known_information_never_changes_primary_eligibility": True,
    },
    "protocol_classification_weights": {
        "description": 0.35,
        "discharge_current": 0.25,
        "discharge_mode": 0.15,
        "charge_current": 0.10,
        "charge_voltage": 0.075,
        "cv_cutoff": 0.075,
    },
    "schema_field_aliases": {
        "description": ["description", "protocol_description", "step_description"],
        "voltage": ["voltage_V", "voltage", "Voltage_V", "Voltage"],
        "current": ["current_A", "current", "Current_A", "Current"],
        "capacity": ["capacity_Ah", "capacity", "Capacity_Ah", "Capacity"],
        "relative_time": [
            "relative_time_min",
            "relative_time_s",
            "time_min",
            "time_s",
            "test_time_s",
            "test_time_min",
        ],
        "summary_discharge_capacity": [
            "discharge_capacity_Ah",
            "discharge_capacity",
            "Discharge_Capacity_Ah",
        ],
        "summary_cycle_life": ["cycle_life", "cycle_count", "Cycle_Life"],
    },
    "approval_gate": {
        "required": True,
        "approval_file": "protocol_review_approval.json",
        "required_status": "APPROVED",
        "required_hash_fields": [
            "xjtu_protocol_inventory.csv",
            "batch2_config_draft.json",
            "batch3_config_draft.json",
            "batch4_config_draft.json",
            "batch5_config_draft.json",
            "batch6_config_draft.json",
        ],
    },
}


def _protocol(
    protocol_class: str,
    *,
    charge_current_A: float | None,
    charge_current_tolerance_A: float | None,
    cv_cutoff_upper_A: float | None,
    discharge_current_A: float | None,
    discharge_current_tolerance_A: float | None,
    discharge_mode: str,
    primary_eligible: str,
    primary_reason: str,
    counts_as_aging_operation: bool,
    recurring_comparable: bool,
    initial_test: bool = False,
    description_markers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "Protocol_Class": protocol_class,
        "Expected_Charge_Current_A": charge_current_A,
        "Charge_Current_Tolerance_A": charge_current_tolerance_A,
        "Expected_CV_Cutoff_Upper_A": cv_cutoff_upper_A,
        "Expected_Discharge_Current_A": discharge_current_A,
        "Discharge_Current_Tolerance_A": discharge_current_tolerance_A,
        "Discharge_Mode": discharge_mode,
        "Primary_Capacity_Eligible": primary_eligible,
        "Primary_Capacity_Reason": primary_reason,
        "Counts_As_Aging_Operation": counts_as_aging_operation,
        "Recurring_Comparable": recurring_comparable,
        "Initial_Test": initial_test,
        "Description_Markers": description_markers or [],
    }


def make_batch_config(batch_id: int) -> dict[str, Any]:
    if batch_id not in range(2, 7):
        raise ValueError(f"Batch config is defined only for Batch-2 through Batch-6: {batch_id}")
    config = deepcopy(BASE_CONFIG)
    config["batch_id"] = batch_id
    config["expected_canonical_count"] = EXPECTED_CANONICAL_COUNTS[batch_id]
    config["scientific_definitions_frozen_from_batch1"] = {
        "qref_definitions": [
            "nominal",
            "initial_test",
            "first_comparable",
            "early_median5_comparable",
        ],
        "EOL_SOH_threshold": 0.80,
        "fixed_absolute_EOThreshold_Ah": 1.600,
        "no_interpolation": True,
        "no_smoothing": True,
        "preserve_raw_cycle_index": True,
        "capacity_jump_alone_is_check_only": True,
    }

    initial = _protocol(
        "INITIAL_CAPACITY_TEST",
        charge_current_A=1.0,
        charge_current_tolerance_A=0.20,
        cv_cutoff_upper_A=0.06,
        discharge_current_A=-0.4,
        discharge_current_tolerance_A=0.10,
        discharge_mode="FULL_TO_2P5V",
        primary_eligible="NO",
        primary_reason="BOL_INITIAL_TEST_QREF_ONLY",
        counts_as_aging_operation=False,
        recurring_comparable=False,
        initial_test=True,
        description_markers=["initial", "0.2c discharge", "capacity test"],
    )

    if batch_id == 2:
        config["expected_protocol_summary"] = (
            "Cycle 1 BOL 0.2C full test; recurring 3C charge and 1C full discharge"
        )
        config["protocols"] = [
            initial,
            _protocol(
                "STANDARD_FULL_DISCHARGE_1C",
                charge_current_A=6.0,
                charge_current_tolerance_A=0.70,
                cv_cutoff_upper_A=0.13,
                discharge_current_A=-2.0,
                discharge_current_tolerance_A=0.25,
                discharge_mode="FULL_TO_2P5V",
                primary_eligible="YES",
                primary_reason="RECURRING_PROTOCOL_COMPARABLE_1C_FULL_DISCHARGE",
                counts_as_aging_operation=True,
                recurring_comparable=True,
                description_markers=["3c charge", "1c discharge"],
            ),
        ]
    elif batch_id == 3:
        config["expected_protocol_summary"] = (
            "Cycle 1 BOL test; recurring 2C charge with variable full-discharge rates"
        )
        rates = [
            ("0P5C", -1.0, 0.15, "NO", "NONPRIMARY_DISCHARGE_RATE"),
            ("1C", -2.0, 0.25, "YES", "PRIMARY_VARIABLE_SEQUENCE_1C_FULL_DISCHARGE"),
            ("2C", -4.0, 0.45, "NO", "NONPRIMARY_DISCHARGE_RATE"),
            ("3C", -6.0, 0.70, "NO", "NONPRIMARY_DISCHARGE_RATE"),
            ("5C", -10.0, 1.10, "NO", "NONPRIMARY_DISCHARGE_RATE"),
        ]
        config["protocols"] = [initial] + [
            _protocol(
                f"VARIABLE_FULL_DISCHARGE_{label}",
                charge_current_A=4.0,
                charge_current_tolerance_A=0.55,
                cv_cutoff_upper_A=0.13,
                discharge_current_A=current,
                discharge_current_tolerance_A=tolerance,
                discharge_mode="FULL_TO_2P5V",
                primary_eligible=eligible,
                primary_reason=reason,
                counts_as_aging_operation=True,
                recurring_comparable=eligible == "YES",
                description_markers=[
                    f"{('0.5c' if label == '0P5C' else label.lower())} discharge"
                ],
            )
            for label, current, tolerance, eligible, reason in rates
        ]
    elif batch_id == 4:
        config["expected_protocol_summary"] = (
            "Variable-rate partial aging discharge to 3.0 V plus separate 1C full diagnostics"
        )
        config["protocols"] = [
            initial,
            _protocol(
                "AGING_PARTIAL_DISCHARGE",
                charge_current_A=4.0,
                charge_current_tolerance_A=0.55,
                cv_cutoff_upper_A=0.13,
                discharge_current_A=None,
                discharge_current_tolerance_A=None,
                discharge_mode="PARTIAL_TO_3P0V_OR_TIME",
                primary_eligible="NO",
                primary_reason="PARTIAL_AGING_DISCHARGE_NOT_FULL_CAPACITY",
                counts_as_aging_operation=True,
                recurring_comparable=False,
                description_markers=["to 3v", "3.0v", "partial", "aging"],
            ),
            _protocol(
                "CAPACITY_DIAGNOSTIC_1C",
                charge_current_A=4.0,
                charge_current_tolerance_A=0.55,
                cv_cutoff_upper_A=0.13,
                discharge_current_A=-2.0,
                discharge_current_tolerance_A=0.25,
                discharge_mode="FULL_TO_2P5V",
                primary_eligible="YES",
                primary_reason="SEPARATE_1C_FULL_CAPACITY_DIAGNOSTIC",
                counts_as_aging_operation=False,
                recurring_comparable=True,
                description_markers=["test capacity", "capacity diagnostic", "to 2.5v"],
            ),
        ]
        config["allowed_partial_discharge_currents_A"] = [-1.0, -2.0, -4.0, -6.0, -10.0]
        config["allowed_partial_current_relative_tolerance"] = 0.14
    elif batch_id == 5:
        config["expected_protocol_summary"] = (
            "Random-walk partial aging with recurring 1C full capacity diagnostics; BOL initial test not assumed"
        )
        config["protocols"] = [
            initial,
            _protocol(
                "RANDOM_WALK_AGING",
                charge_current_A=None,
                charge_current_tolerance_A=None,
                cv_cutoff_upper_A=0.16,
                discharge_current_A=None,
                discharge_current_tolerance_A=None,
                discharge_mode="PARTIAL_TO_3P0V_OR_TIME",
                primary_eligible="NO",
                primary_reason="RANDOM_WALK_PARTIAL_DISCHARGE_NOT_FULL_CAPACITY",
                counts_as_aging_operation=True,
                recurring_comparable=False,
                description_markers=["random", "walking", "3.0v"],
            ),
            _protocol(
                "CAPACITY_DIAGNOSTIC_1C",
                charge_current_A=2.0,
                charge_current_tolerance_A=0.35,
                cv_cutoff_upper_A=0.13,
                discharge_current_A=-2.0,
                discharge_current_tolerance_A=0.25,
                discharge_mode="FULL_TO_2P5V",
                primary_eligible="YES",
                primary_reason="RECURRING_1C_FULL_CAPACITY_DIAGNOSTIC",
                counts_as_aging_operation=False,
                recurring_comparable=True,
                description_markers=["test capacity", "capacity diagnostic", "to 2.5v"],
            ),
        ]
        config["description_charge_expectations"] = [
            {
                "description_marker": "0.5c charge",
                "expected_current_A": 1.0,
                "tolerance_A": 0.25,
            },
            {
                "description_marker": "3c charge",
                "expected_current_A": 6.0,
                "tolerance_A": 0.75,
            },
        ]
        config["random_walk_discharge_current_abs_range_A"] = [2.0, 8.0]
        config["random_walk_discharge_duration_min_range"] = [2.0, 6.0]
        config["expected_initial_random_walk_cycles"] = 20
        config["expected_random_walk_cycles_between_diagnostics"] = 10
        config["qref_initial_test_policy"] = {
            "value": None,
            "status": "UNAVAILABLE_NO_BOL_INITIAL_TEST",
            "fallback_to_random_walk_cycle_1": False,
        }
        config["description_charge_current_mismatch_policy"] = {
            "applicable_protocol_class": "RANDOM_WALK_AGING",
            "severity": "INFO",
            "requires_human_review": False,
        }
    elif batch_id == 6:
        config["expected_protocol_summary"] = (
            "Cycle 1 BOL test; GEO partial aging plus periodic 0.5C full diagnostics"
        )
        config["protocols"] = [
            initial,
            _protocol(
                "GEO_AGING_PARTIAL",
                charge_current_A=4.0,
                charge_current_tolerance_A=0.55,
                cv_cutoff_upper_A=0.13,
                discharge_current_A=-1.334,
                discharge_current_tolerance_A=0.22,
                discharge_mode="PARTIAL_BY_PROTOCOL_DURATION",
                primary_eligible="NO",
                primary_reason="GEO_PARTIAL_AGING_DISCHARGE_NOT_FULL_CAPACITY",
                counts_as_aging_operation=True,
                recurring_comparable=False,
                description_markers=["geo", "0.67c discharge", "0.667c", "partial"],
            ),
            _protocol(
                "CAPACITY_DIAGNOSTIC_0P5C",
                charge_current_A=2.0,
                charge_current_tolerance_A=0.35,
                cv_cutoff_upper_A=0.13,
                discharge_current_A=-1.0,
                discharge_current_tolerance_A=0.16,
                discharge_mode="FULL_TO_2P5V",
                primary_eligible="YES",
                primary_reason="PERIODIC_0P5C_FULL_CAPACITY_DIAGNOSTIC",
                counts_as_aging_operation=False,
                recurring_comparable=True,
                description_markers=["test capacity", "capacity diagnostic", "to 2.5v"],
            ),
        ]
        config["expected_aging_operations_between_diagnostics"] = 5
        config["geo_high_soc_cv_terminal_current_upper_A"] = 0.13
        config["geo_cc_plateau_not_observed_policy"] = {
            "minimum_charge_max_V": 4.18,
            "maximum_terminal_current_A": 0.13,
            "severity": "INFO",
            "requires_human_review": False,
        }
        config["geo_full_cutoff_policy"] = {
            "retain_protocol_class": "GEO_AGING_PARTIAL",
            "retain_primary_capacity_eligible": "NO",
            "severity": "INFO",
            "requires_human_review": False,
        }
    return config


def all_batch_configs() -> dict[int, dict[str, Any]]:
    return {batch_id: make_batch_config(batch_id) for batch_id in range(2, 7)}
