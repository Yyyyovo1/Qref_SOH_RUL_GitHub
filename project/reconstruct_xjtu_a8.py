"""Command-line entry point for the targeted XJTU A8 reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

from xjtu_a8_deterministic_reconstruction_repro import (
    ReconstructionError,
    finalize_existing_staging,
    run_reconstruction,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct frozen XJTU ROBUSTNESS_A8 intermediate artifacts."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--resume-staging",
        type=Path,
        default=None,
        help="Validate and publish an existing completed staging directory without retraining.",
    )
    args = parser.parse_args()
    try:
        if args.resume_staging is None:
            result = run_reconstruction(args.workspace, args.output_dir)
        else:
            result = finalize_existing_staging(
                args.workspace, args.resume_staging, args.output_dir
            )
    except ReconstructionError as exc:
        message = str(exc)
        if "FROZEN SUMMARY NOT REPRODUCED" in message:
            print(
                "XJTU A8 DETERMINISTIC RECONSTRUCTION FAILED —\n"
                "FROZEN SUMMARY NOT REPRODUCED",
                flush=True,
            )
        else:
            print(f"XJTU A8 DETERMINISTIC RECONSTRUCTION FAILED —\n{message}", flush=True)
        return 1
    print(result["final_stdout"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
