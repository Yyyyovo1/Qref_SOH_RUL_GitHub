from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .analysis import run_full_analysis
from .final_v2 import run_final_v2
from .hotfix import run_hotfix_validation
from .scan import run_scan
from .validation import validate_scan


def _workspace_default() -> Path:
    return Path(__file__).resolve().parent.parent


def _parser() -> argparse.ArgumentParser:
    workspace = _workspace_default()
    parser = argparse.ArgumentParser(
        description="XJTU 55-cell protocol-aware capacity/Qref sensitivity pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Read-only schema/protocol scan")
    scan.add_argument("--workspace", type=Path, default=workspace)
    scan.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "outputs" / "xjtu_multibatch_scan_v1",
    )
    scan.add_argument(
        "--frozen-batch1-dir",
        type=Path,
        default=(
            workspace
            / "xjtu_batch1_pipeline"
            / "outputs"
            / "xjtu_batch1_pilot_final_v1"
        ),
    )

    validate = subparsers.add_parser(
        "validate", help="Validate a human-approved protocol inventory"
    )
    validate.add_argument(
        "--scan-dir",
        type=Path,
        default=workspace / "outputs" / "xjtu_multibatch_scan_v2",
    )
    validate.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "outputs" / "xjtu_multibatch_validate_v1",
    )
    validate.add_argument(
        "--frozen-batch1-dir",
        type=Path,
        default=(
            workspace
            / "xjtu_batch1_pipeline"
            / "outputs"
            / "xjtu_batch1_pilot_final_v1"
        ),
    )
    validate.add_argument(
        "--review-file",
        type=Path,
        default=workspace / "xjtu_multibatch_manual_protocol_review_v1.json",
    )

    run = subparsers.add_parser("run", help="Run approved 55-cell analysis")
    run.add_argument(
        "--validation-dir",
        type=Path,
        default=workspace / "outputs" / "xjtu_multibatch_validate_v1",
    )
    run.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "outputs" / "xjtu_55_final_v1",
    )

    hotfix = subparsers.add_parser(
        "hotfix-validate",
        help="Validate numeric early-median5 sorting and export Batch-5 CHECK review material",
    )
    hotfix.add_argument("--workspace", type=Path, default=workspace)
    hotfix.add_argument(
        "--prehotfix-dir",
        type=Path,
        default=workspace / "outputs" / "xjtu_55_final_v1",
    )
    hotfix.add_argument(
        "--scan-dir",
        type=Path,
        default=workspace / "outputs" / "xjtu_multibatch_scan_v2",
    )
    hotfix.add_argument(
        "--validation-dir",
        type=Path,
        default=workspace / "outputs" / "xjtu_multibatch_validate_v1",
    )
    hotfix.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "outputs" / "xjtu_55_hotfix_validation_v1",
    )

    final_v2 = subparsers.add_parser(
        "final-v2",
        help="Apply approved Batch-5 manual KEEP decisions and regenerate derived outputs",
    )
    final_v2.add_argument("--workspace", type=Path, default=workspace)
    final_v2.add_argument(
        "--prehotfix-dir",
        type=Path,
        default=workspace / "outputs" / "xjtu_55_final_v1",
    )
    final_v2.add_argument(
        "--hotfix-dir",
        type=Path,
        default=workspace / "outputs" / "xjtu_55_hotfix_validation_v1",
    )
    final_v2.add_argument(
        "--validation-dir",
        type=Path,
        default=workspace / "outputs" / "xjtu_multibatch_validate_v1",
    )
    final_v2.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "outputs" / "xjtu_55_final_v2",
    )
    return parser


def _format_counts(values: dict[object, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in values.items()) or "none"


def _scan_report(output_dir: Path, result: object) -> None:
    protocol = pd.read_csv(
        output_dir / "xjtu_protocol_inventory.csv",
        keep_default_na=False,
        low_memory=False,
    )
    print(f"SCAN_STATUS: {result.scan_status}")
    print("Canonical files per batch: " + _format_counts(result.canonical_counts))
    print(f"Duplicate files: {result.duplicate_count}")
    print("Schema variants: " + _format_counts(result.schema_variant_counts))
    print("Cycle counts: " + _format_counts(result.cycle_counts))
    print("Detected protocol classes: " + _format_counts(result.protocol_counts))
    print(
        "Primary-capacity candidate counts: "
        + _format_counts(result.primary_candidate_counts)
    )
    print(f"Unknown protocol count: {result.unknown_protocol_count}")
    print(f"Protocol anomaly count: {result.anomaly_count}")
    if not protocol.empty:
        b3 = protocol.loc[protocol["Batch_ID"].eq(3)]
        b3_counts = b3["Protocol_Class"].value_counts().to_dict()
        print("Batch-3 discharge-rate distribution: " + _format_counts(b3_counts))
        for batch_id, classes in (
            (4, ["CAPACITY_DIAGNOSTIC_1C", "AGING_PARTIAL_DISCHARGE"]),
            (5, ["CAPACITY_DIAGNOSTIC_1C", "RANDOM_WALK_AGING"]),
            (6, ["CAPACITY_DIAGNOSTIC_0P5C", "GEO_AGING_PARTIAL"]),
        ):
            subset = protocol.loc[protocol["Batch_ID"].eq(batch_id)]
            counts = {name: int(subset["Protocol_Class"].eq(name).sum()) for name in classes}
            print(f"Batch-{batch_id} diagnostic/aging counts: " + _format_counts(counts))
    for batch_id in range(2, 7):
        print(
            f"Batch-{batch_id} representative battery: "
            f"{result.representatives.get(batch_id, '') or 'UNAVAILABLE'}"
        )
    if result.review_reasons:
        print("Review gates:")
        for reason in result.review_reasons:
            print(f"  - {reason}")
    print(f"Output directory: {output_dir}")
    print("VALIDATE EXECUTED: NO")
    print("RUN EXECUTED: NO")
    print("SCAN COMPLETE — WAITING FOR HUMAN PROTOCOL REVIEW")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "scan":
        result = run_scan(args.workspace, args.output_dir, args.frozen_batch1_dir)
        _scan_report(result.output_dir, result)
        return 0
    if args.command == "validate":
        result = validate_scan(
            args.scan_dir,
            args.output_dir,
            args.frozen_batch1_dir,
            args.review_file,
        )
        print(f"Validation output directory: {result.output_dir}")
        print(
            "Regression tests: "
            f"{result.regression_test_count - result.failed_test_count} passed, "
            f"{result.failed_test_count} failed"
        )
        print(
            f"Protocol information rows: {result.info_anomaly_count}; "
            f"unresolved review rows: {result.review_anomaly_count}"
        )
        if result.passed:
            print("VALIDATION PASSED — WAITING FOR HUMAN APPROVAL TO RUN")
            return 0
        print("VALIDATION FAILED — DO NOT RUN")
        return 1
    if args.command == "run":
        output = run_full_analysis(args.validation_dir, args.output_dir)
        cycle = pd.read_csv(output / "xjtu_55_cycle_qc.csv", low_memory=False)
        primary = pd.read_csv(
            output / "xjtu_55_primary_capacity.csv", low_memory=False
        )
        qref = pd.read_csv(output / "xjtu_55_qref_summary.csv", low_memory=False)
        eol = pd.read_csv(output / "xjtu_55_eol_sensitivity.csv", low_memory=False)
        pairwise = pd.read_csv(output / "xjtu_55_eol_pairwise.csv", low_memory=False)
        notes = pd.read_csv(
            output / "xjtu_55_known_data_quality_notes.csv", low_memory=False
        )
        tests = pd.read_csv(
            output / "xjtu_55_run_regression_tests.csv", low_memory=False
        )
        print(f"Output directory: {output}")
        print(f"Total cells: {qref['Cell_ID'].nunique()}")
        print(f"Total raw cycles: {len(cycle)}")
        print(f"Total primary capacity observations: {len(primary)}")
        print(
            "Primary observations by Batch: "
            + _format_counts(primary.groupby("Batch_ID").size().to_dict())
        )
        qref_columns = {
            "nominal": "Qref_nominal_Ah",
            "initial_test": "Qref_initial_test_Ah",
            "first_comparable": "Qref_first_comparable_Ah",
            "early_median5_comparable": "Qref_early_median5_comparable_Ah",
        }
        print(
            "Qref available cells: "
            + _format_counts(
                {
                    name: int(pd.to_numeric(qref[column], errors="coerce").notna().sum())
                    for name, column in qref_columns.items()
                }
            )
        )
        for definition, rows in eol.groupby("Qref_Definition", sort=True):
            print(
                f"EOL status [{definition}]: "
                + _format_counts(rows["EOL_Status"].value_counts().to_dict())
            )
        nominal_initial = pairwise.loc[
            pairwise["Definition_A"].eq("nominal")
            & pairwise["Definition_B"].eq("initial_test")
        ]
        exact_delta = pd.to_numeric(
            nominal_initial["Delta_EOL_Cycles"], errors="coerce"
        ).dropna()
        if exact_delta.empty:
            print("Nominal vs initial_test exact EOL shift: no directly comparable exact pairs")
        else:
            print(
                "Nominal vs initial_test exact EOL shift: "
                f"n={len(exact_delta)}, min={exact_delta.min():.0f}, "
                f"median={exact_delta.median():.1f}, max={exact_delta.max():.0f} cycles"
            )
        transitions = pairwise.loc[
            pairwise["EOL_Status_A"].ne(pairwise["EOL_Status_B"]),
            "Censoring_Status_Change",
        ].value_counts().to_dict()
        print("Censoring-status transitions: " + _format_counts(transitions))
        print(f"Known INFO anomalies: {int(notes['Severity'].eq('INFO').sum())}")
        print(
            "Run regression tests: "
            f"{int(tests['Status'].eq('PASS').sum())} passed, "
            f"{int(tests['Status'].eq('FAIL').sum())} failed"
        )
        print("XJTU 55-CELL RUN COMPLETE")
        return 0
    if args.command == "hotfix-validate":
        output = run_hotfix_validation(
            args.workspace,
            args.prehotfix_dir,
            args.scan_dir,
            args.validation_dir,
            args.output_dir,
        )
        regression = pd.read_csv(
            output / "early_median5_hotfix_regression.csv", keep_default_na=False
        )
        preview = pd.read_csv(
            output / "corrected_qref_preview.csv", keep_default_na=False
        )
        review = pd.read_csv(
            output / "batch5_check_manual_review.csv", keep_default_na=False
        )
        print(f"Output directory: {output}")
        print(
            f"Regression tests: {int(regression['Status'].eq('PASS').sum())} passed, "
            f"{int(regression['Status'].eq('FAIL').sum())} failed"
        )
        print(f"Corrected Qref preview cells: {len(preview)}")
        print(f"Batch-5 CHECK rows exported: {len(review)}")
        print("EOL/RUL regenerated: NO")
        print("HOTFIX VALIDATION COMPLETE — WAITING FOR MANUAL CHECK REVIEW")
        return 0
    if args.command == "final-v2":
        output = run_final_v2(
            args.workspace,
            args.prehotfix_dir,
            args.hotfix_dir,
            args.validation_dir,
            args.output_dir,
        )
        qref = pd.read_csv(
            output / "xjtu_55_qref_summary.csv",
            keep_default_na=False,
            low_memory=False,
        )
        batch = pd.read_csv(
            output / "xjtu_55_batch_summary.csv",
            keep_default_na=False,
            low_memory=False,
        )
        tests = pd.read_csv(
            output / "xjtu_55_run_regression_tests.csv",
            keep_default_na=False,
            low_memory=False,
        )
        audit = pd.read_csv(
            output / "xjtu_55_manual_qc_audit.csv",
            keep_default_na=False,
            low_memory=False,
        )
        batch5_primary = int(
            batch.loc[
                pd.to_numeric(batch["Batch_ID"], errors="raise").eq(5),
                "Primary_Capacity_Observation_Count",
            ].iloc[0]
        )
        batch5_manual = audit.loc[
            pd.to_numeric(audit["Batch_ID"], errors="raise").eq(5)
        ]
        print(f"Output directory: {output}")
        print(f"Canonical cells: {qref['Cell_ID'].nunique()}")
        print(f"Batch-5 primary observations: {batch5_primary}")
        print(
            "Batch-5 manual KEEP cycles included: "
            f"{int(batch5_manual['Final_Include'].eq('YES').sum())}/5"
        )
        print(
            "Regression tests: "
            f"{int(tests['Status'].eq('PASS').sum())} passed, "
            f"{int(tests['Status'].eq('FAIL').sum())} failed"
        )
        print("XJTU 55-CELL FINAL V2 COMPLETE")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
