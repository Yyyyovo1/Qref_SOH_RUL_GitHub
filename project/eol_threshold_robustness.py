from __future__ import annotations

import argparse
from pathlib import Path

from eol_threshold_robustness_repro import FrozenReproductionError, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EOL Threshold Robustness V1 from frozen XJTU/HUST artifacts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing the frozen outputs directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output_dir = run_pipeline(args.root)
    except FrozenReproductionError as exc:
        print("EOL THRESHOLD ROBUSTNESS V1 FAILED —")
        print("FROZEN 80PCT RESULT NOT REPRODUCED")
        print(str(exc))
        return 2
    print(f"Output directory: {output_dir}")
    print("EOL THRESHOLD ROBUSTNESS V1 COMPLETE —")
    print("WAITING FOR HUMAN REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
