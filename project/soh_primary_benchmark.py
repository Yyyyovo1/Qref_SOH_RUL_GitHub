"""CLI for the frozen SOH Primary ML Benchmark V1."""

from __future__ import annotations

import argparse
from pathlib import Path

from soh_primary_benchmark_repro.pipeline import (
    PrimaryBenchmarkError,
    run_primary_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the frozen Ridge/RF/GB SOH benchmark without sensitivity runs."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = run_primary_benchmark(args.workspace.resolve(), args.output_dir)
    except PrimaryBenchmarkError as exc:
        print("SOH PRIMARY BENCHMARK V1 FAILED")
        print(str(exc))
        return 1
    print(result["final_stdout"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
