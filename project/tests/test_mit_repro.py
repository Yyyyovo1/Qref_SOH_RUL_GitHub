from __future__ import annotations

import numpy as np
import pandas as pd

from mit_repro.freeze import assign_folds, qc_cycles, qref_table
from mit_repro.ingest import assemble


def _cycles(key: str, n: int, start_q: float = 1.07) -> pd.DataFrame:
    return pd.DataFrame({"Source_Key": key, "Source_Cycle": np.arange(1, n + 1, dtype=float),
                         "QDischarge_Ah": start_q - 0.001 * np.arange(n), "Discharge_Median_Current_C": -4.0,
                         "Source_Row": np.arange(n)})


def test_merge_offsets_cycles_and_applies_deletions() -> None:
    keys = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4", "b1c8", "b1c10", "b1c12", "b1c13", "b1c22",
            "b2c7", "b2c8", "b2c9", "b2c15", "b2c16", "b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"]
    cells = pd.DataFrame({"Source_Key": keys, "Batch": [int(k[1]) for k in keys], "Summary_Rows": 10,
                          "Author_Cycle_Life_Raw": 11.0})
    cycles = pd.concat([_cycles(k, 10) for k in keys], ignore_index=True)
    out_cells, out_cycles = assemble(cells, cycles)
    assert sorted(out_cells[out_cells.Status.eq("INCLUDED")].Source_Key) == ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]
    b1c0 = out_cycles[out_cycles.Cell_ID.eq("b1c0")].Cycle_Index
    assert list(b1c0) == list(range(1, 21)) and not out_cycles.duplicated(["Cell_ID", "Cycle_Index"]).any()
    assert out_cells.set_index("Source_Key").at["b1c0", "Author_Cycle_Life"] == 11.0 + 662


def test_qc_excludes_non_4c_and_nonpositive_but_only_flags_jumps() -> None:
    c = _cycles("b1c5", 8).rename(columns={"Source_Key": "Cell_ID", "Source_Cycle": "Cycle_Index"})
    c.loc[0, "QDischarge_Ah"] = 0.0
    c.loc[3, "Discharge_Median_Current_C"] = -0.1
    c.loc[5, "QDischarge_Ah"] = 0.90  # single-cycle dip
    out = qc_cycles(c)
    assert list(out.Primary_Capacity_Eligible) == ["NO", "YES", "YES", "NO", "YES", "YES", "YES", "YES"]
    assert out.loc[5, "Capacity_Jump_Flag"] == "YES" and out.loc[6, "Capacity_Jump_Flag"] == "YES"
    q = qref_table(out[out.Primary_Capacity_Eligible.eq("YES")])
    assert q.Qref_First_Comparable_Source_Cycle.iat[0] == 2
    assert q.Qref_Early_Median5_Comparable_Source_Cycles.iat[0] == "2|3|5|6|7"


def test_fold_assignment_is_deterministic_and_balanced() -> None:
    cells = [f"b3c{i}" for i in range(40)]
    a, b = assign_folds(cells), assign_folds(list(reversed(cells)))
    assert a.equals(b) and a.Fold_ID.value_counts().tolist() == [8] * 5
