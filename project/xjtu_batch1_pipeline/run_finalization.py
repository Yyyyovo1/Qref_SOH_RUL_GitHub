from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xjtu_batch1.finalization import run_finalization  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize XJTU Batch-1 pilot v1 using authorized manual QC decisions"
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "xjtu_batch1_pilot_v1",
        help="Existing automatic pilot v1 output directory; read-only",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT.parent / "Batch-1",
        help="Raw MAT directory used only for SHA-256 verification",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "xjtu_batch1_pilot_final_v1",
        help="New final output directory; existing directories are never overwritten",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "xjtu_batch1_pilot_v1"
        / "xjtu_batch1_config.json",
        help="Configuration snapshot from the automatic pilot v1 run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_finalization(
            pilot_dir=args.pilot_dir,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            config_path=args.config,
        )
    except Exception as exc:
        print(f"FINALIZATION FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Final outputs saved to: {outputs.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
