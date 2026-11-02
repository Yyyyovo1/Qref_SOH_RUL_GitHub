from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


OUTPUT_DIR_NAME = "eol_threshold_robustness_v1"
THRESHOLD_FRACTIONS = (0.75, 0.80, 0.85)

COMMON_QREFS = (
    "nominal",
    "first_comparable",
    "early_median5_comparable",
)
MEASURED_COMMON_QREFS = (
    "first_comparable",
    "early_median5_comparable",
)
XJTU_SECONDARY_QREFS = ("initial_test",)
XJTU_QREFS = (*COMMON_QREFS, *XJTU_SECONDARY_QREFS)
HUST_QREFS = COMMON_QREFS

QREF_ORDER = {
    "nominal": 1,
    "first_comparable": 2,
    "early_median5_comparable": 3,
    "initial_test": 4,
}
THRESHOLD_ORDER = {value: index for index, value in enumerate(THRESHOLD_FRACTIONS, 1)}

XJTU_DENSE_BATCHES = frozenset({1, 2})
XJTU_NOMINAL_CAPACITY_AH = 2.0
HUST_NOMINAL_CAPACITY_AH = 1.1

EXACT = "EXACT_OBSERVED"
INTERVAL = "INTERVAL_CENSORED"
RIGHT = "RIGHT_CENSORED"
LEFT = "LEFT_CENSORED_OR_BASELINE_UNRESOLVED"
QREF_UNAVAILABLE = "QREF_UNAVAILABLE"

PRIMARY_STATUS_ORDER = {
    EXACT: 1,
    INTERVAL: 2,
    RIGHT: 3,
    LEFT: 4,
    QREF_UNAVAILABLE: 5,
}

STANDARD_TRANSITIONS = (
    "EXACT_TO_EXACT",
    "EXACT_TO_INTERVAL",
    "EXACT_TO_RIGHT",
    "INTERVAL_TO_EXACT",
    "INTERVAL_TO_INTERVAL",
    "INTERVAL_TO_RIGHT",
    "RIGHT_TO_EXACT",
    "RIGHT_TO_INTERVAL",
    "BOTH_RIGHT",
)

HUST_STOPPING_CONTEXT = {
    0.75: (
        "BELOW_RELEASE_STOPPING_REGION_FINITE_WINDOW_LIMITED",
        "The 75% threshold is below the nominal stopping region and is more strongly limited by the finite released observation window.",
    ),
    0.80: (
        "NEAR_RELEASED_DATASET_STOPPING_RULE",
        "The 80% threshold lies directly near the released dataset stopping rule; near-threshold endpoints remain censored unless an observed capacity is at or below the unrounded threshold.",
    ),
    0.85: (
        "ABOVE_NOMINAL_STOPPING_TARGET_EARLIER_OBSERVABILITY",
        "The 85% threshold is above the nominal stopping target and may be observed earlier in the recorded trajectory.",
    ),
}

OUTPUT_FILES = (
    "eol_threshold_cell_level.csv",
    "eol_threshold_status_summary.csv",
    "eol_threshold_transition_summary.csv",
    "eol_threshold_rul_availability.csv",
    "eol_threshold_effect_bounds.csv",
    "eol_threshold_qref_direction_audit.csv",
    "eol_threshold_hust_stopping_rule_audit.csv",
    "eol_threshold_cross_threshold_summary.csv",
    "eol_threshold_regression_tests.csv",
)


@dataclass(frozen=True)
class PipelinePaths:
    root: Path

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def output_dir(self) -> Path:
        return self.outputs / OUTPUT_DIR_NAME

    @property
    def xjtu_dir(self) -> Path:
        return self.outputs / "xjtu_55_final_v2"

    @property
    def hust_dir(self) -> Path:
        return self.outputs / "hust_77_final_v1"

    @property
    def cross_dir(self) -> Path:
        return self.outputs / "cross_dataset_summary_v1"

    @property
    def frozen_dirs(self) -> tuple[Path, ...]:
        return (
            self.xjtu_dir,
            self.hust_dir,
            self.cross_dir,
            self.outputs / "soh_ml_design_v1",
            self.outputs / "soh_feature_extraction_v1",
            self.outputs / "soh_primary_benchmark_v1",
            self.outputs / "soh_ml_robustness_v1",
        )

    @property
    def source_files(self) -> tuple[Path, ...]:
        return (
            self.xjtu_dir / "xjtu_55_soh_labels_long.csv",
            self.xjtu_dir / "xjtu_55_qref_summary.csv",
            self.xjtu_dir / "xjtu_55_eol_sensitivity.csv",
            self.hust_dir / "hust_77_primary_capacity.csv",
            self.hust_dir / "hust_77_qref_summary.csv",
            self.hust_dir / "hust_77_eol_sensitivity.csv",
            self.hust_dir / "hust_77_stopping_rule_audit.csv",
            self.cross_dir / "cross_dataset_cell_level.csv",
        )


XJTU_QREF_COLUMNS = {
    "nominal": ("Qref_nominal_Ah", "Qref_nominal_Status"),
    "first_comparable": (
        "Qref_first_comparable_Ah",
        "Qref_first_comparable_Status",
    ),
    "early_median5_comparable": (
        "Qref_early_median5_comparable_Ah",
        "Qref_early_median5_comparable_Status",
    ),
    "initial_test": ("Qref_initial_test_Ah", "Qref_initial_test_Status"),
}

HUST_QREF_COLUMNS = {
    "nominal": ("Qref_Nominal_Ah", "Qref_Nominal_Status"),
    "first_comparable": (
        "Qref_First_Comparable_Ah",
        "Qref_First_Comparable_Status",
    ),
    "early_median5_comparable": (
        "Qref_Early_Median5_Comparable_Ah",
        "Qref_Early_Median5_Comparable_Status",
    ),
}
