"""CLI for SOH ML Robustness + Statistical Uncertainty V1."""

from __future__ import annotations

import argparse
from pathlib import Path

from soh_ml_robustness_repro.pipeline import RobustnessError, run_robustness


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen SOH robustness experiments and cell-level bootstrap."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = run_robustness(args.workspace.resolve(), args.output_dir)
    except RobustnessError as exc:
        print("SOH ML ROBUSTNESS V1 FAILED")
        print(str(exc))
        return 1
    print(result["final_stdout"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
