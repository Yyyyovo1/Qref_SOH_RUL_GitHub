"""CLI for the RUL ranking-sensitivity v3 on TJU (rul_ranking_pilot/PROTOCOL_v3_TJU.md)."""

from __future__ import annotations

import argparse
from pathlib import Path

from rul_pilot_v3_tju.config import DATASETS, THETAS
from rul_pilot_v3_tju.pipeline import run
from rul_ranking_pilot.labels import PilotError
from rul_ranking_pilot.pipeline import compare_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dev", action="store_true", help="Development run: checks 0-4 and estimability only.")
    parser.add_argument("--thetas", default=None, help="Comma-separated theta subset (development runs only).")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset subset (development runs only).")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("RUN_A", "RUN_B"), help="Check 6: compare two runs.")
    args = parser.parse_args()
    if args.compare:
        table = compare_runs(*args.compare)
        print(table.to_string(index=False))
        return 0 if table.Identical.all() else 1
    if (args.thetas or args.datasets) and not args.dev:
        print("theta / dataset subsets are allowed only with --dev")
        return 2
    thetas = tuple(t for t in THETAS if not args.thetas or f"{t:.2f}" in args.thetas.split(","))
    datasets = tuple(d for d in DATASETS if not args.datasets or d in args.datasets.split(","))
    try:
        target = run(args.workspace.resolve(), args.output_dir, thetas, datasets, dev=args.dev)
    except PilotError as exc:
        print("RUL V3 TJU FAILED")
        print(str(exc))
        return 1
    print(f"RUL V3 TJU {'DEV RUN' if args.dev else 'RUN'} COMPLETE: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
