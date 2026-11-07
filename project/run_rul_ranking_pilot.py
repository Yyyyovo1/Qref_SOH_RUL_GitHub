"""CLI for the preregistered RUL ranking-sensitivity pilot (rul_ranking_pilot/PROTOCOL.md)."""

from __future__ import annotations

import argparse
from pathlib import Path

from rul_ranking_pilot.config import THETAS
from rul_ranking_pilot.labels import PilotError
from rul_ranking_pilot.pipeline import compare_runs, run_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RUL ranking-sensitivity pilot.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--thetas", type=str, default=None,
                        help="Comma-separated subset of the theta grid, for development runs only.")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("RUN_A", "RUN_B"),
                        help="Check 6: compare two completed output directories and exit.")
    args = parser.parse_args()
    if args.compare:
        table = compare_runs(*args.compare)
        print(table.to_string(index=False))
        return 0 if table.Identical.all() else 1
    thetas = THETAS
    if args.thetas:
        thetas = tuple(t for t in THETAS if f"{t:.2f}" in args.thetas.split(","))
    try:
        target = run_pilot(args.workspace.resolve(), args.output_dir, thetas)
    except PilotError as exc:
        print("RUL RANKING PILOT FAILED")
        print(str(exc))
        return 1
    print(f"RUL RANKING PILOT COMPLETE: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
