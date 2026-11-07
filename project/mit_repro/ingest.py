"""Read the data.matr.io MATLAB structs (HDF5) the way the authors' Load Data.ipynb does."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import AUTHOR_DELETIONS, AUTHOR_MERGES

SUMMARY_FIELDS = {"QDischarge": "QDischarge_Ah", "QCharge": "QCharge_Ah", "IR": "IR_Ohm", "Tavg": "Tavg_C",
                  "Tmin": "Tmin_C", "Tmax": "Tmax_C", "chargetime": "Chargetime_min", "cycle": "Source_Cycle"}


def _chars(f, ref) -> str:
    """Decode a MATLAB char array (uint16 or uint32 code points)."""
    raw = np.asarray(f[ref][()]).ravel()
    return "".join(chr(int(c)) for c in raw if int(c) > 0).strip()


def _scalar(f, ref) -> float:
    value = np.asarray(f[ref][()], dtype=float).ravel()
    return float(value[0]) if value.size else np.nan


def read_batch(path: Path, batch: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cell metadata and per-cycle summary for one batch file.

    Current in the cycle records is in C-rate units (4C discharge = -4.0); the per-cycle
    discharge current is the median of the negative-current samples.
    """
    import h5py

    cells, cycles = [], []
    with h5py.File(path, "r") as f:
        b = f["batch"]
        for i in range(b["summary"].shape[0]):
            key = f"b{batch}c{i}"
            summary = f[b["summary"][i, 0]]
            frame = pd.DataFrame({out: np.hstack(summary[src][0, :].tolist()) for src, out in SUMMARY_FIELDS.items()})
            cyc = f[b["cycles"][i, 0]]
            n_cyc = cyc["I"].shape[0]
            if n_cyc != len(frame):
                raise ValueError(f"{key}: {n_cyc} cycle records but {len(frame)} summary rows")
            median_c = np.full(n_cyc, np.nan)
            for j in range(n_cyc):
                current = np.asarray(f[cyc["I"][j, 0]][()], dtype=float).ravel()
                negative = current[current < 0]
                median_c[j] = float(np.median(negative)) if negative.size else np.nan
            cells.append({"Source_Key": key, "Batch": batch, "Batch_Index": i,
                          "Author_Cycle_Life_Raw": _scalar(f, b["cycle_life"][i, 0]),
                          "Charge_Policy": _chars(f, b["policy_readable"][i, 0]),
                          # barcode/channel_id are MATLAB `string` objects that h5py cannot decode.
                          "Barcode": "NOT_DECODABLE_MATLAB_STRING_OBJECT",
                          "Summary_Rows": len(frame)})
            frame.insert(0, "Source_Key", key)
            frame["Source_Row"] = np.arange(len(frame))
            frame["Discharge_Median_Current_C"] = median_c
            cycles.append(frame)
    return pd.DataFrame(cells), pd.concat(cycles, ignore_index=True)


def assemble(cells: pd.DataFrame, cycles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the authors' merges (cycle offset = number of batch-1 summary rows) and deletions."""
    cells = cells.set_index("Source_Key")
    cycles = cycles.copy()
    cycles["Cell_ID"] = cycles["Source_Key"]
    cycles["Cycle_Index"] = cycles["Source_Cycle"]
    cells["Author_Cycle_Life"] = cells["Author_Cycle_Life_Raw"]
    cells["Merged_From"] = ""
    cells["Merge_Offset_Cycles"] = 0
    cells["Status"] = "INCLUDED"
    cells["Status_Reason"] = ""
    for b1, b2, add_len in AUTHOR_MERGES:
        offset = int(cells.at[b1, "Summary_Rows"])
        part = cycles["Source_Key"].eq(b2)
        cycles.loc[part, "Cell_ID"] = b1
        cycles.loc[part, "Cycle_Index"] = cycles.loc[part, "Source_Cycle"] + offset
        cells.at[b1, "Author_Cycle_Life"] = cells.at[b1, "Author_Cycle_Life_Raw"] + add_len
        cells.at[b1, "Merged_From"] = b2
        cells.at[b1, "Merge_Offset_Cycles"] = offset
        cells.at[b2, "Status"] = "MERGED_INTO_" + b1
        cells.at[b2, "Status_Reason"] = "AUTHOR_BATCH_CONTINUATION"
    for key, reason in AUTHOR_DELETIONS.items():
        cells.at[key, "Status"] = "EXCLUDED"
        cells.at[key, "Status_Reason"] = reason
    keep = cells.index[cells["Status"].eq("INCLUDED")]
    cycles = cycles[cycles["Cell_ID"].isin(keep)].copy()
    cycles["Cycle_Index"] = cycles["Cycle_Index"].astype(int)
    return cells.reset_index(), cycles.reset_index(drop=True)
