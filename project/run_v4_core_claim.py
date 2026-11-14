"""CLI for the v4 core-claim test (v4_core_claim/PROTOCOL_v4.md and AMENDMENT_01.md).

  python run_v4_core_claim.py --dev [--experiments 1,3] [--datasets HUST,MIT,NCA] [--thetas 0.85,0.90] --output-dir ../reruns/v4/dev_run_1
  python run_v4_core_claim.py --output-dir ../reruns/v4/formal_run_1          (formal run: full grid only)
  python run_v4_core_claim.py --compare ../reruns/v4/formal_run_1 ../reruns/v4/formal_run_2   (check 6)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rul_ranking_pilot.labels import PilotError
from rul_v4_core_claim.config import EXP1_THETAS, EXP3_DATASETS
from rul_v4_core_claim.pipeline import compare_outputs, run


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent, help="project directory")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dev", action="store_true", help="development run: checks 0-4, S, exp1 leakage, estimability only")
    p.add_argument("--experiments", default=None, help="dev only, e.g. 1 or 3")
    p.add_argument("--datasets", default=None, help="dev only, subset of HUST,MIT,NCA for experiment 3")
    p.add_argument("--thetas", default=None, help="dev only, comma-separated theta subset (both experiments)")
    p.add_argument("--compare", nargs=2, type=Path, metavar=("RUN_A", "RUN_B"), help="check 6: compare two runs")
    a = p.parse_args()
    if a.compare:
        table = compare_outputs(*a.compare)
        print(table.to_string(index=False))
        return 0 if table.Identical.all() else 1
    if (a.experiments or a.datasets or a.thetas) and not a.dev:
        print("experiment / dataset / theta subsets are allowed only with --dev")
        return 2
    experiments = tuple(a.experiments.split(",")) if a.experiments else ("1", "3")
    datasets = tuple(d for d in EXP3_DATASETS if not a.datasets or d in a.datasets.split(","))
    th = {round(float(t), 2) for t in a.thetas.split(",")} if a.thetas else None
    exp1_thetas = tuple(t for t in EXP1_THETAS if th is None or t in th)
    try:
        target = run(a.workspace.resolve(), a.output_dir, dev=a.dev, experiments=experiments, datasets=datasets,
                     exp1_thetas=exp1_thetas, exp3_thetas=tuple(sorted(th)) if th else None)
    except PilotError as exc:
        print("RUL V4 FAILED")
        print(str(exc))
        return 1
    print(f"RUL V4 {'DEV RUN' if a.dev else 'FORMAL RUN'} COMPLETE: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
