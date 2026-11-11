"""CLI for the retrospective protocol validation (protocol_validation/VALIDATION_PLAN.md)."""

from __future__ import annotations

import argparse
from pathlib import Path

from protocol_validation.pipeline import run
from rul_ranking_pilot.pipeline import compare_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("RUN_A", "RUN_B"), help="Check 5: compare two runs.")
    args = parser.parse_args()
    if args.compare:
        table = compare_runs(*args.compare)
        print(table.to_string(index=False))
        return 0 if table.Identical.all() else 1
    print(f"PROTOCOL VALIDATION COMPLETE: {run(args.workspace, args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
