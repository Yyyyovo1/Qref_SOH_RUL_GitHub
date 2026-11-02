"""Central configuration for the CALCE CS2 processing pipeline.

All thresholds and cell-specific boundary decisions that can change QC
results live in this module. The pipeline never contains hidden deletion,
smoothing, interpolation, or value-replacement rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


PIPELINE_VERSION = "2.3.0"
NOMINAL_CAPACITY_AH = 1.1  # Descriptive metadata only; never used to compute SOH.

PROJECT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = PROJECT_DIR.parent


@dataclass(frozen=True)
class CellProfile:
    """Fixed paths, benchmark, duplicate, and boundary rules for one cell."""

    cell_id: str
    input_dir: Path
    output_root: Path
    benchmark_file: str
    benchmark_cycle: int
    benchmark_capacity_ah: float
    benchmark_tolerance_ah: float
    benchmark_basis: str
    duplicate_files: Mapping[str, Mapping[str, str]]
    known_boundary_tails: Mapping[tuple[str, int], str]


CS2_35_DUPLICATE_FILES = {
    "CS2_35_1_10_11(1).xlsx": {
        "role": "UNIT_TEST_REFERENCE_DUPLICATE",
        "duplicate_of": "CS2_35_1_10_11.xlsx",
    },
    "CS2_35_2_10_11.xlsx": {
        "role": "DUPLICATE_CANDIDATE",
        "duplicate_of": "CS2_35_2_4_11.xlsx",
    },
}

CS2_35_BOUNDARY_TAILS = {
    ("CS2_35_9_7_10.xlsx", 45): "file ends during charging; continuation not established",
    ("CS2_35_9_8_10.xlsx", 7): "file ends before normal discharge cutoff; continuation not established",
    ("CS2_35_11_01_10.xlsx", 10): "Statistics omits the final partial discharge cycle",
    ("CS2_35_11_24_10.xlsx", 9): "Statistics omits the final charge-only cycle",
    ("CS2_35_12_23_10.xlsx", 25): "Statistics omits the final charge-only cycle",
    ("CS2_35_1_28_11.xlsx", 37): "Statistics omits the final charge-only cycle",
}

# CS2-36 has no duplicate copy in the supplied directory. The six entries
# below end without a normal discharge completion. They remain as independent
# local cycles and are never stitched to the following workbook.
CS2_36_BOUNDARY_TAILS = {
    (
        "CS2_36_9_7_10.xlsx",
        44,
    ): "file ends during main discharge at 3.8957 V; continuation not established",
    (
        "CS2_36_10_04_10.xlsx",
        8,
    ): "file ends during main discharge at 3.8620 V; continuation not established",
    (
        "CS2_36_10_05_10.xlsx",
        9,
    ): "file ends during CC charging at 4.0734 V; continuation not established",
    (
        "CS2_36_11_01_10.xlsx",
        17,
    ): "Statistics omits the final charge-only cycle; continuation not established",
    (
        "CS2_36_11_24_10.xlsx",
        17,
    ): "Statistics omits the final partial discharge cycle ending at 3.6868 V",
    (
        "CS2_36_12_23_10.xlsx",
        28,
    ): "Statistics omits the final charge-only cycle; continuation not established",
}

# CS2-37 has no duplicate copy in the supplied directory. These seven tails
# end without a normal discharge completion and must not be auto-stitched.
CS2_37_BOUNDARY_TAILS = {
    (
        "CS2_37_9_7_10.xlsx",
        45,
    ): "file ends during main discharge at 3.9185 V; continuation not established",
    (
        "CS2_37_10_04_10.xlsx",
        25,
    ): "file ends after incomplete CV charging with no main discharge",
    (
        "CS2_37_10_05_10.xlsx",
        9,
    ): "file ends during main discharge at 3.2224 V; continuation not established",
    (
        "CS2_37_11_01_10.xlsx",
        17,
    ): "Statistics omits the final charge-only cycle; continuation not established",
    (
        "CS2_37_11_24_10.xlsx",
        17,
    ): "Statistics omits the final charge-only cycle with incomplete CV taper",
    (
        "CS2_37_12_23_10.xlsx",
        28,
    ): "Statistics omits the final charge-only cycle; continuation not established",
    (
        "CS2_37_1_28_11.xlsx",
        49,
    ): "Statistics omits the final charge-only cycle with incomplete CV taper",
}

# CS2-38 contains one byte-distinct export pair whose Channel and Statistics
# values are identical. Keep both files for audit, but index only the copy whose
# name matches the internally observed Jan 31--Feb 3 test interval.
CS2_38_DUPLICATE_FILES = {
    "CS2_38_2_10_11.xlsx": {
        "role": "DUPLICATE_CANDIDATE",
        "duplicate_of": "CS2_38_2_4_11.xlsx",
    },
}

# These seven CS2-38 tails end without a normal discharge completion. They are
# retained as independent local cycles and are never stitched automatically.
CS2_38_BOUNDARY_TAILS = {
    (
        "CS2_38_9_7_10.xlsx",
        43,
    ): "file ends during main discharge at 3.8261 V; continuation not established",
    (
        "CS2_38_10_04_10.xlsx",
        25,
    ): "file ends after incomplete CV charging with no main discharge",
    (
        "CS2_38_10_05_10.xlsx",
        9,
    ): "file ends during main discharge at 3.4996 V; continuation not established",
    (
        "CS2_38_11_01_10.xlsx",
        17,
    ): "Statistics omits the final charge-only cycle; continuation not established",
    (
        "CS2_38_11_24_10.xlsx",
        17,
    ): "Statistics omits the final charge-only cycle with incomplete CV taper",
    (
        "CS2_38_12_23_10.xlsx",
        26,
    ): "Statistics omits the final partial discharge cycle ending at 3.8860 V",
    (
        "CS2_38_1_28_11.xlsx",
        42,
    ): "Statistics omits the final partial CC-only cycle ending at 4.1362 V",
}


CELL_PROFILES: dict[str, CellProfile] = {
    "CS2-35": CellProfile(
        cell_id="CS2-35",
        input_dir=WORKSPACE_DIR / "CS2_35",
        output_root=PROJECT_DIR / "outputs",
        benchmark_file="CS2_35_1_10_11(1).xlsx",
        benchmark_cycle=2,
        benchmark_capacity_ah=0.8699526,
        benchmark_tolerance_ah=1.0e-6,
        benchmark_basis="human-verified Statistics and Channel cross-check",
        duplicate_files=CS2_35_DUPLICATE_FILES,
        known_boundary_tails=CS2_35_BOUNDARY_TAILS,
    ),
    "CS2-36": CellProfile(
        cell_id="CS2-36",
        input_dir=WORKSPACE_DIR / "CS2_36",
        output_root=PROJECT_DIR / "outputs" / "CS2_36",
        benchmark_file="CS2_36_10_28_10.xlsx",
        benchmark_cycle=2,
        benchmark_capacity_ah=0.9963362809876899,
        benchmark_tolerance_ah=1.0e-9,
        benchmark_basis=(
            "independent Statistics adjacent-delta and Channel within-cycle-delta cross-check"
        ),
        duplicate_files={},
        known_boundary_tails=CS2_36_BOUNDARY_TAILS,
    ),
    "CS2-37": CellProfile(
        cell_id="CS2-37",
        input_dir=WORKSPACE_DIR / "CS2_37",
        output_root=PROJECT_DIR / "outputs" / "CS2_37",
        benchmark_file="CS2_37_10_28_10.xlsx",
        benchmark_cycle=2,
        benchmark_capacity_ah=0.9691913622921109,
        benchmark_tolerance_ah=1.0e-9,
        benchmark_basis=(
            "independent Statistics adjacent-delta and Channel within-cycle-delta cross-check"
        ),
        duplicate_files={},
        known_boundary_tails=CS2_37_BOUNDARY_TAILS,
    ),
    "CS2-38": CellProfile(
        cell_id="CS2-38",
        input_dir=WORKSPACE_DIR / "CS2_38",
        output_root=PROJECT_DIR / "outputs" / "CS2_38",
        benchmark_file="CS2_38_10_28_10.xlsx",
        benchmark_cycle=2,
        benchmark_capacity_ah=0.96857087639073,
        benchmark_tolerance_ah=1.0e-9,
        benchmark_basis=(
            "independent Statistics adjacent-delta and Channel within-cycle-delta cross-check"
        ),
        duplicate_files=CS2_38_DUPLICATE_FILES,
        known_boundary_tails=CS2_38_BOUNDARY_TAILS,
    ),
}


DEFAULT_CELL_ID = "CS2-35"
DEFAULT_PROFILE = CELL_PROFILES[DEFAULT_CELL_ID]

# Backward-compatible aliases used by the original CS2-35 tests and callers.
CELL_ID = DEFAULT_PROFILE.cell_id
DEFAULT_INPUT_DIR = DEFAULT_PROFILE.input_dir
DEFAULT_OUTPUT_ROOT = DEFAULT_PROFILE.output_root
BENCHMARK_FILE = DEFAULT_PROFILE.benchmark_file
BENCHMARK_CYCLE = DEFAULT_PROFILE.benchmark_cycle
BENCHMARK_CAPACITY_AH = DEFAULT_PROFILE.benchmark_capacity_ah
BENCHMARK_TOLERANCE_AH = DEFAULT_PROFILE.benchmark_tolerance_ah
DUPLICATE_FILES = DEFAULT_PROFILE.duplicate_files
KNOWN_BOUNDARY_TAILS = DEFAULT_PROFILE.known_boundary_tails


def get_profile(cell_id: str) -> CellProfile:
    """Resolve a supported cell id and report allowed values on failure."""

    normalized = cell_id.strip().upper().replace("_", "-")
    try:
        return CELL_PROFILES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(CELL_PROFILES))
        raise ValueError(f"unsupported cell id {cell_id!r}; choose one of: {allowed}") from exc


@dataclass(frozen=True)
class QCThresholds:
    """Protocol and consistency thresholds, expressed in source units."""

    # Counter and cross-source checks.
    counter_zero_tolerance_ah: float = 0.005
    counter_decrease_tolerance_ah: float = 0.0001
    stats_channel_agreement_tolerance_ah: float = 0.001

    # CC charge: expected 0.5 C is approximately +0.55 A.
    cc_current_min_a: float = 0.40
    cc_current_max_a: float = 0.70
    cc_max_current_std_a: float = 0.08
    cc_complete_min_records: int = 5
    cc_complete_min_duration_s: float = 300.0
    cc_complete_min_charge_delta_ah: float = 0.02
    cc_expected_end_voltage_min_v: float = 4.15

    # CV charge: voltage should stay near 4.2 V and current should taper.
    cv_voltage_min_v: float = 4.15
    cv_voltage_max_v: float = 4.25
    cv_min_records: int = 5
    cv_min_duration_s: float = 300.0
    cv_min_charge_delta_ah: float = 0.005
    cv_min_start_current_a: float = 0.10
    cv_max_end_current_a: float = 0.08
    cv_min_current_drop_a: float = 0.05

    # Main discharge: expected current is approximately -1.1 A.
    discharge_candidate_current_max_a: float = -0.80
    discharge_min_records: int = 5
    discharge_min_duration_s: float = 300.0
    discharge_min_capacity_delta_ah: float = 0.05
    discharge_mean_current_min_a: float = -1.25
    discharge_mean_current_max_a: float = -0.95
    discharge_start_voltage_min_v: float = 3.50
    discharge_start_voltage_max_v: float = 4.25
    discharge_cutoff_voltage_min_v: float = 2.55
    discharge_cutoff_voltage_max_v: float = 2.85

    # Capacity jumps compare against the nearest preceding valid reference
    # cycle and remain review signals only. Both thresholds must be met.
    capacity_jump_abs_ah: float = 0.12
    capacity_jump_relative: float = 0.15


QC = QCThresholds()


def threshold_dict() -> dict[str, object]:
    """Return thresholds in a deterministic, log-friendly mapping."""

    return asdict(QC)
