"""Run the follow-up analyses in order. All outputs go to project/outputs/followup_analyses_v1/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs" / "followup_analyses_v1"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for args in (["followup.py"], ["followup2.py"],
                 ["plot_summary_annotated.py", str(HERE.parent / "outputs/protocol_validation_v1/summary_table.csv"),
                  str(OUT / "protocol_validation_summary_annotated")]):
        subprocess.run([sys.executable, str(HERE / args[0]), *args[1:]], check=True, stdout=subprocess.DEVNULL)
