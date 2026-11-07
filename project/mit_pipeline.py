"""CLI: freeze the MIT/Severson dataset for the RUL pilot v2 (rul_ranking_pilot/PROTOCOL_v2.md)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mit_repro.pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    target = run(args.workspace, args.output_dir)
    print(json.dumps(json.loads((target / "scan_summary.json").read_text(encoding="utf-8")), indent=2))
    print(f"MIT FREEZE COMPLETE: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
