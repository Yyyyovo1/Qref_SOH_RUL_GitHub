"""Command-line entry point for the CALCE CS2 extraction/QC pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calce_qc.config import DEFAULT_CELL_ID, get_profile  # noqa: E402
from calce_qc.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CALCE CS2 single-cycle discharge capacity and protocol QC."
    )
    parser.add_argument(
        "--cell-id",
        default=DEFAULT_CELL_ID,
        help="Cell profile to process, for example CS2-35, CS2-36, CS2-37, or CS2-38.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional source .xlsx directory override; otherwise uses the cell profile.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output-root override; otherwise uses the cell profile.",
    )
    parser.add_argument(
        "--run-prefix",
        default="run",
        help="Prefix for the new non-overwriting run directory, for example run_v2.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        profile = get_profile(args.cell_id)
        result = run_pipeline(
            profile,
            args.input_dir,
            args.output_root,
            run_prefix=args.run_prefix,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"OUTPUT_DIR={result['output_dir']}")
    print(f"CELL_ID={result['cell_id']}")
    print(
        "BENCHMARK="
        f"Cycle:{result['benchmark']['cycle']},"
        f"Statistics:{result['benchmark']['statistics_ah']:.12f}Ah,"
        f"Channel:{result['benchmark']['channel_ah']:.12f}Ah"
    )
    print(f"CANONICAL_CYCLES={result['canonical_cycle_count']}")
    print(f"QC_COUNTS={result['canonical_qc_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
