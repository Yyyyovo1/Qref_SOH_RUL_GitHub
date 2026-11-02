from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xjtu_batch1.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XJTU Batch-1 cycle QC and Qref/SOH/EOL/RUL sensitivity pilot"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT.parent / "Batch-1",
        help="Directory containing canonical and optional byte-identical duplicate MAT files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "xjtu_batch1_pilot_v1",
        help="New output directory; the pipeline refuses to overwrite an existing directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "xjtu_batch1_config.json",
        help="JSON configuration containing every QC and sensitivity threshold",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config_path=args.config,
        )
    except Exception as exc:
        print(f"PIPELINE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Outputs saved to: {outputs.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

