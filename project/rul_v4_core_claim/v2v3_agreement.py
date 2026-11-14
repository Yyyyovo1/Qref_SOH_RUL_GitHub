"""PROTOCOL_v4 section 7 step 6: supported agreement share (section 3.1) on the existing v2 and v3 results, as a
descriptive supplement. Refuses to run unless a formal v4 decision.json with H1-H3 verdicts already exists.

  python -m rul_v4_core_claim.v2v3_agreement ../reruns/v4/formal_run_1/decision.json ../reruns/v4/v2v3_agreement
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from rul_ranking_pilot.labels import PilotError
from rul_ranking_pilot.pipeline import _write

from .blinding import allow_evaluation
from .config import QREF_SHORT
from .exp2 import agreement

V2_DIR = "outputs/rul_ranking_pilot_v2"
V3_DIR = "outputs/rul_ranking_v3_tju"
GROUPS = (["Source"], ["Source", "Dataset"], ["Source", "Dataset", "Eval_Set"], ["Source", "Qref_Pair"],
          ["Source", "Dataset", "Pair", "Eval_Set"])


def v2_comparisons(root: Path) -> pd.DataFrame:
    rev = pd.read_csv(root / V2_DIR / "reversals.csv", encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    ci = pd.read_csv(root / V2_DIR / "pair_ci.csv", encoding="utf-8-sig", usecols=["Dataset", "Theta", "Qref_Definition", "Eval_Set", "Pair", "Estimable"])
    est = ci.set_index(["Dataset", "Theta", "Qref_Definition", "Eval_Set", "Pair"]).Estimable.to_dict()
    rev["Estimable_Comparison"] = [bool(est[(r.Dataset, r.Theta, r.Setting_1, r.Eval_Set, r.Pair)]) and
                                   bool(est[(r.Dataset, r.Theta, r.Setting_2, r.Eval_Set, r.Pair)]) for r in rev.itertuples()]
    rev["Qref_Pair"] = rev.Setting_1.map(QREF_SHORT) + "-" + rev.Setting_2.map(QREF_SHORT)
    return rev.assign(Source="v2")


def v3_comparisons(root: Path) -> pd.DataFrame:
    comp = pd.read_csv(root / V3_DIR / "comparisons.csv", encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    return comp.assign(Source="v3")


def run(root: Path, decision_json: Path, out_dir: Path) -> Path:
    dec = json.loads(Path(decision_json).read_text(encoding="utf-8"))
    if not all(k in dec and "Verdict" in dec[k] for k in ("H1", "H2", "H3")):
        raise PilotError("V4_VERDICTS_NOT_FOUND_RUN_THIS_ONLY_AFTER_THE_FORMAL_DECISION")
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise PilotError(f"OUTPUT_DIRECTORY_EXISTS_REFUSING_TO_OVERWRITE:{out_dir}")
    out_dir.mkdir(parents=True)
    comp = pd.concat([v2_comparisons(root), v3_comparisons(root)], ignore_index=True)
    for c in ("Better_1", "Better_2"):
        comp[c] = comp[c].fillna("")
    with allow_evaluation():
        table = pd.concat([agreement(comp, g).assign(Grouping="+".join(g)) for g in GROUPS], ignore_index=True)
    _write(table, out_dir / "v2v3_supported_agreement.csv")
    return out_dir


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[1]
    print(run(here, Path(sys.argv[1]), Path(sys.argv[2])))
