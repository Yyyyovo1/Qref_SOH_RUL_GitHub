"""Command-line entry point for the frozen SOH ML design audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from soh_ml_design_repro.design import run_design


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the SOH ML design audit without training models."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Project workspace containing outputs/ (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory; defaults to outputs/soh_ml_design_v1.",
    )
    args = parser.parse_args()
    run_design(args.workspace.resolve(), args.output_dir)
    print("SOH ML DESIGN V1 COMPLETE —")
    print("NO FINAL MODELS TRAINED —")
    print("WAITING FOR HUMAN APPROVAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
