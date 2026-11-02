"""CLI for the targeted XJTU A8 rank-flip verification."""

from __future__ import annotations

import argparse
from pathlib import Path

from xjtu_a8_rankflip_repro.pipeline import (
    RankFlipVerificationError,
    run_verification,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify XJTU A8 RF-versus-GB rank flips by cell bootstrap."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = run_verification(args.workspace.resolve(), args.output_dir)
    except RankFlipVerificationError as exc:
        print("XJTU A8 RANK-FLIP VERIFICATION FAILED")
        print(str(exc))
        return 1
    print(result["final_stdout"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
