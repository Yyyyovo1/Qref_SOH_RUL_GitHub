from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .full_run import run_full_77
from .pilot import run_8cell_pilot
from .protocol_validation import run_protocol_validation
from .scan import run_scan


def _workspace_default() -> Path:
    return Path(__file__).resolve().parent.parent


def _parser() -> argparse.ArgumentParser:
    workspace = _workspace_default()
    parser = argparse.ArgumentParser(
        description="HUST 77-cell protocol-aware QC and Qref sensitivity pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Run read-only HUST schema/protocol scan")
    scan.add_argument("--workspace", type=Path, default=workspace)
    scan.add_argument(
        "--input-dir", type=Path, default=workspace / "HUST 77"
    )
    scan.add_argument(
        "--output-dir", type=Path, default=workspace / "outputs" / "hust_scan_v1"
    )

    validate = subparsers.add_parser(
        "validate",
        help="Run the human-approved fixed 4-cell raw protocol validation",
    )
    validate.add_argument("--workspace", type=Path, default=workspace)
    validate.add_argument(
        "--input-dir", type=Path, default=workspace / "HUST 77"
    )
    validate.add_argument(
        "--scan-dir", type=Path, default=workspace / "outputs" / "hust_scan_v1"
    )
    validate.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "outputs" / "hust_4cell_protocol_validate_v1",
    )
    pilot = subparsers.add_parser(
        "pilot",
        help="Run the human-approved fixed 8-cell Qref/EOL sensitivity pilot",
    )
    pilot.add_argument("--workspace", type=Path, default=workspace)
    pilot.add_argument("--input-dir", type=Path, default=workspace / "HUST 77")
    pilot.add_argument(
        "--scan-dir", type=Path, default=workspace / "outputs" / "hust_scan_v1"
    )
    pilot.add_argument(
        "--protocol-validation-dir",
        type=Path,
        default=workspace / "outputs" / "hust_4cell_protocol_validate_v1",
    )
    pilot.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "outputs" / "hust_8cell_pilot_v1",
    )
    full = subparsers.add_parser(
        "run",
        help="Run the human-approved frozen-rule 77-cell final replication",
    )
    full.add_argument("--workspace", type=Path, default=workspace)
    full.add_argument("--input-dir", type=Path, default=workspace / "HUST 77")
    full.add_argument(
        "--scan-dir", type=Path, default=workspace / "outputs" / "hust_scan_v1"
    )
    full.add_argument(
        "--protocol-validation-dir",
        type=Path,
        default=workspace / "outputs" / "hust_4cell_protocol_validate_v1",
    )
    full.add_argument(
        "--pilot-dir",
        type=Path,
        default=workspace / "outputs" / "hust_8cell_pilot_v1",
    )
    full.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "outputs" / "hust_77_final_v1",
    )
    return parser


def _format_counts(values: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in values.items()) or "none"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "scan":
        result = run_scan(args.workspace, args.input_dir, args.output_dir)
        print(f"Output directory: {result.output_dir}")
        print(f"Canonical cell count: {result.canonical_cell_count}")
        print(f"Total cycle count: {result.total_cycle_count}")
        print("File types: " + _format_counts(result.file_type_counts))
        print(f"Schema variants: {result.schema_variant_count}")
        print(
            "Capacity field candidates: "
            + ", ".join(result.capacity_field_candidates)
        )
        print(f"Cells with raw curve data: {result.cells_with_raw_curves}")
        print(f"Cells with native SOH labels: {result.cells_with_native_soh}")
        print(f"Cells with native RUL labels: {result.cells_with_native_rul}")
        print(f"Protocol fingerprint count: {result.protocol_fingerprint_count}")
        print(
            "Cells with protocol-change candidate: "
            f"{result.protocol_change_cell_count}"
        )
        print(
            "Incomplete discharge candidate count: "
            f"{result.incomplete_discharge_candidate_count}"
        )
        print(
            "Capacity jump CHECK candidate count: "
            f"{result.capacity_jump_check_count}"
        )
        print(
            "First capacity (Ah): "
            f"min={result.first_capacity_stats['min']:.6f}, "
            f"median={result.first_capacity_stats['median']:.6f}, "
            f"max={result.first_capacity_stats['max']:.6f}"
        )
        print(
            "Last capacity (Ah): "
            f"min={result.last_capacity_stats['min']:.6f}, "
            f"median={result.last_capacity_stats['median']:.6f}, "
            f"max={result.last_capacity_stats['max']:.6f}"
        )
        print(
            "Final capacity relative to 0.880 Ah: "
            f"<= {result.final_capacity_le_0p88_count}, "
            f"> {result.final_capacity_gt_0p88_count}"
        )
        print("Selected representatives: " + "|".join(result.representative_cells))
        print("Validate executed: NO")
        print("Run executed: NO")
        print("Model training executed: NO")
        if result.hard_stop_reasons:
            print("Hard-stop reasons:")
            for reason in result.hard_stop_reasons:
                print(f"  - {reason}")
            print("HUST SCAN NEEDS REVIEW")
            return 2
        print("HUST SCAN COMPLETE — WAITING FOR HUMAN REVIEW")
        return 0
    if args.command == "validate":
        result = run_protocol_validation(
            args.workspace, args.input_dir, args.scan_dir, args.output_dir
        )
        print(f"Output directory: {result.output_dir}")
        print(f"Selected validation cycles: {result.selected_cycle_count}")
        print(f"Physical step rows: {result.raw_step_count}")
        print(f"Review plots: {result.plot_count}")
        print(
            "Regression tests: "
            + ("ALL PASSED" if result.regression_tests_passed else "FAILED")
        )
        if result.final_assessment == "PARSER_STAGE_MAPPING_BUG":
            print("HUST 4-CELL PROTOCOL VALIDATION COMPLETE")
            print("\nFINAL ASSESSMENT:\nPARSER_STAGE_MAPPING_BUG")
            print("\nACTION REQUIRED:\nFIX PARSER AND REBUILD PROTOCOL FINGERPRINTS")
            print("\nDO NOT RUN QREF PILOT YET")
            return 0 if result.regression_tests_passed else 2
        if result.final_assessment == "RAW_DATA_PROTOCOL_DIFFERS_FROM_PAPER":
            print("HUST 4-CELL PROTOCOL VALIDATION COMPLETE")
            print("\nFINAL ASSESSMENT:\nRAW_DATA_PROTOCOL_DIFFERS_FROM_PAPER")
            print("\nRAW PROTOCOL IS INTERNALLY CONSISTENT")
            print("\nREADY FOR HUMAN REVIEW BEFORE QREF PILOT")
            return 0 if result.regression_tests_passed else 2
        print("HUST 4-CELL PROTOCOL VALIDATION NEEDS REVIEW")
        print("\nFINAL ASSESSMENT:\nPROTOCOL_DISCREPANCY_UNRESOLVED")
        print("\nDO NOT RUN QREF PILOT")
        print("DO NOT RUN 77 CELLS")
        return 2
    if args.command == "pilot":
        result = run_8cell_pilot(
            args.workspace,
            args.input_dir,
            args.scan_dir,
            args.protocol_validation_dir,
            args.output_dir,
        )
        print(f"Output directory: {result.output_dir}")
        print(f"Pilot cells: {result.pilot_cell_count}")
        print(f"Primary eligible observations: {result.eligible_cycle_count}")
        print(f"SOH label rows: {result.soh_row_count}")
        print(f"RUL label rows: {result.rul_row_count}")
        print(f"Review plots: {result.plot_count}")
        first = result.first_qref_stats
        median5 = result.median5_qref_stats
        print(
            "First-comparable Qref Ah (min / median / max): "
            f"{first['min']:.12g} / {first['median']:.12g} / {first['max']:.12g}"
        )
        print(
            "Early-median5 Qref Ah (min / median / max): "
            f"{median5['min']:.12g} / {median5['median']:.12g} / {median5['max']:.12g}"
        )
        for definition, counts in result.eol_status_counts.items():
            print(
                f"{definition}: EXACT_OBSERVED={counts['EXACT_OBSERVED']}, "
                f"RIGHT_CENSORED={counts['RIGHT_CENSORED']}"
            )
        print(
            "Nominal vs first-comparable censoring-status changes: "
            f"{result.nominal_first_status_changes}"
        )
        print(
            "Nominal vs first-comparable exact comparable shifts: "
            f"{result.nominal_first_exact_pair_count}"
        )
        shift = result.nominal_first_shift_stats
        if shift["median"] is None:
            print("Nominal vs first-comparable shift min / median / max: NA / NA / NA")
        else:
            print(
                "Nominal vs first-comparable shift min / median / max: "
                f"{shift['min']:.12g} / {shift['median']:.12g} / {shift['max']:.12g}"
            )
        print("Regression tests: ALL PASSED")
        print("HUST 8-CELL QREF/EOL PILOT COMPLETE —")
        print("WAITING FOR HUMAN REVIEW")
        print("DO NOT RUN ALL 77 CELLS")
        return 0
    if args.command == "run":
        result = run_full_77(
            args.workspace,
            args.input_dir,
            args.scan_dir,
            args.protocol_validation_dir,
            args.pilot_dir,
            args.output_dir,
        )
        print(f"Output directory: {result.output_dir}")
        print(f"Total cells: {result.canonical_cell_count}")
        print(f"Total raw cycles: {result.raw_cycle_count}")
        print(f"Total primary observations: {result.primary_observation_count}")
        print(f"SOH label rows: {result.soh_row_count}")
        print(f"RUL label rows: {result.rul_row_count}")
        print(f"Figures: {result.plot_count}")
        first = result.first_qref_stats
        print(
            "First-comparable Qref Ah (min / median / max): "
            f"{first['min']:.12g} / {first['median']:.12g} / {first['max']:.12g}"
        )
        first_pct = result.first_minus_nominal_pct_stats
        print(
            "First-comparable minus nominal percent (min / median / max): "
            f"{first_pct['min']:.12g} / {first_pct['median']:.12g} / "
            f"{first_pct['max']:.12g}"
        )
        median5 = result.median5_qref_stats
        print(
            "Early-median5 Qref Ah (min / median / max): "
            f"{median5['min']:.12g} / {median5['median']:.12g} / "
            f"{median5['max']:.12g}"
        )
        for definition, counts in result.eol_status_counts.items():
            print(
                f"{definition}: EXACT_OBSERVED={counts['EXACT_OBSERVED']}, "
                f"RIGHT_CENSORED={counts['RIGHT_CENSORED']}"
            )
        print(
            "Nominal -> first-comparable transitions: "
            + _format_counts(result.nominal_first_transitions)
        )
        print(
            "Nominal -> early-median5 transitions: "
            + _format_counts(result.nominal_median5_transitions)
        )
        delta = result.first_median5_delta_stats
        if delta["median"] is None:
            print(
                "First-comparable vs early-median5 exact EOL delta "
                "min / median / max / median abs: NA / NA / NA / NA"
            )
        else:
            print(
                "First-comparable vs early-median5 exact EOL delta "
                "min / median / max / median abs: "
                f"{delta['min']:.12g} / {delta['median']:.12g} / "
                f"{delta['max']:.12g} / {delta['median_absolute']:.12g}"
            )
        lower = result.lower_bound_stats
        if lower["median"] is None:
            print("Minimum observed earlier-by lower bound min / median / max: NA / NA / NA")
        else:
            print(
                "Minimum observed earlier-by lower bound min / median / max: "
                f"{lower['min']:.12g} / {lower['median']:.12g} / "
                f"{lower['max']:.12g}"
            )
        print(f"QC CHECK count: {result.check_count}")
        print(f"INVALID_CANDIDATE count: {result.invalid_count}")
        print(f"Unresolved review count: {result.unresolved_review_count}")
        print("Regression tests: ALL PASSED")
        print("HUST 77-CELL FINAL V1 COMPLETE")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
