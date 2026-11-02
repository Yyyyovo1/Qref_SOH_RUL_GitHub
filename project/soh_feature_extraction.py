"""CLI for the approved full feature-extraction and QC stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from soh_feature_extraction_repro.pipeline import run_feature_extraction


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract frozen SOH SAFE_CORE features without training models."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_feature_extraction(args.workspace.resolve(), args.output_dir)
    print(result["final_stdout"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
