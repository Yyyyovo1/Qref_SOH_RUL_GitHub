"""Scan the TJU per-cell CSVs (read directly from the verified Zenodo zips) into a per-cycle inventory.

Each CSV is a raw time series: 'cycle number' groups the rows of one charge/rest/discharge/rest cycle,
'Q discharge/mA.h' accumulates within the cycle and peaks at the end of the discharge.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COLUMNS, CURRENT_SIGN_THRESHOLD_MA, FILES

NAME = re.compile(r"^CY(?P<temp>\d+)-(?P<chg>\d+)_(?P<dis>\d+)-#(?P<tag>\d+)$")


def _rate(code: str) -> float:
    """'025' -> 0.25, '05' -> 0.5, '1' -> 1.0 (leading zero means a decimal)."""
    return float(code) if not code.startswith("0") else float("0." + code[1:])


def parse_name(stem: str) -> dict:
    m = NAME.match(stem)
    if not m:
        raise ValueError(f"UNEXPECTED_FILE_NAME_{stem}")
    return {"Temperature_C": int(m["temp"]), "Charge_CRate_Name": _rate(m["chg"]), "Discharge_CRate_Name": _rate(m["dis"]),
            "Cell_Tag": int(m["tag"]), "Condition": f"CY{m['temp']}-{m['chg']}_{m['dis']}"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cycle_summary(df: pd.DataFrame) -> pd.DataFrame:
    t, v, i = (df[COLUMNS[k]].to_numpy(float) for k in ("time", "voltage", "current"))
    qd, qc, cyc = (df[COLUMNS[k]].to_numpy(float) for k in ("qd", "qc", "cycle"))
    rows = []
    bounds = np.flatnonzero(np.diff(cyc) != 0) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(cyc)]):
        ti, vi, ii, qdi, qci = t[a:b], v[a:b], i[a:b], qd[a:b], qc[a:b]
        dis = ii < -CURRENT_SIGN_THRESHOLD_MA
        chg = ii > CURRENT_SIGN_THRESHOLD_MA
        rec = {"Cycle_Number": float(cyc[a]), "Rows": int(b - a), "Start_Time_s": float(ti[0]), "End_Time_s": float(ti[-1]),
               "Discharge_Rows": int(dis.sum()), "Charge_Rows": int(chg.sum()),
               "QDischarge_Max_Ah": float(np.nanmax(qdi)) / 1000.0, "QCharge_Max_Ah": float(np.nanmax(qci)) / 1000.0}
        if chg.any():
            ci = ii[chg]
            cc = ci >= 0.9 * ci.max()
            rec.update({"Charge_CC_Median_Current_mA": float(np.median(ci[cc])), "Charge_Max_V": float(vi[chg].max()),
                        "Charge_Last_Current_mA": float(ci[-1]), "First_Charge_Row": int(np.flatnonzero(chg)[0])})
        if dis.any():
            di = np.flatnonzero(dis)
            first, last = di[0], di[-1]
            seg_t, seg_i = ti[dis], ii[dis]
            integrated = float(np.sum(-0.5 * (seg_i[1:] + seg_i[:-1]) * np.diff(seg_t) * ((np.diff(di) == 1)))) / 3.6e6
            rec.update({"Discharge_Median_Current_mA": float(np.median(seg_i)), "Discharge_Min_V": float(vi[dis].min()),
                        "Discharge_End_V": float(vi[last]), "Discharge_Start_V": float(vi[first]),
                        "Discharge_Duration_s": float(ti[last] - ti[first]),
                        "QDischarge_At_Discharge_Start_Ah": float(qdi[first]) / 1000.0,
                        "QDischarge_At_Discharge_End_Ah": float(qdi[last]) / 1000.0,
                        "QDischarge_Integrated_Ah": integrated,
                        "Max_V_Before_Discharge": float(vi[:first].max()) if first > 0 else np.nan,
                        "Discharge_Contiguous": bool(last - first + 1 == dis.sum())})
        rows.append(rec)
    return pd.DataFrame(rows)


def scan(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest, cycles = [], []
    for ds, info in FILES.items():
        path = raw_dir / info["zip"]
        if path.stat().st_size != info["bytes"] or sha256_file(path) != info["sha256"]:
            raise ValueError(f"{ds}_ZIP_DOES_NOT_MATCH_RECORDED_CHECKSUM")
        with zipfile.ZipFile(path) as zf:
            names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
            if len(names) != info["csv_count"]:
                raise ValueError(f"{ds}_CSV_COUNT_{len(names)}")
            for n in names:
                stem = Path(n).stem
                meta = parse_name(stem)
                with zf.open(n) as fh:
                    df = pd.read_csv(fh)
                missing = set(COLUMNS.values()) - set(df.columns)
                if missing:
                    raise ValueError(f"{stem}_MISSING_COLUMNS_{sorted(missing)}")
                monotone_cycle = bool((np.diff(df[COLUMNS["cycle"]].to_numpy(float)) >= 0).all())
                s = cycle_summary(df)
                cell = f"{ds}_{stem}"
                s.insert(0, "Cell_ID", cell)
                s.insert(0, "Dataset", ds)
                cycles.append(s)
                manifest.append({"Dataset": ds, "Cell_ID": cell, "Source_File": n, "Rows": len(df), "Cycles": len(s),
                                 "Cycle_Number_Min": float(s.Cycle_Number.min()), "Cycle_Number_Max": float(s.Cycle_Number.max()),
                                 "Cycle_Number_Monotone": monotone_cycle,
                                 "Cycle_Numbers_Unique": bool(s.Cycle_Number.is_unique), **meta})
                print(cell, len(s), flush=True)
    return pd.DataFrame(manifest), pd.concat(cycles, ignore_index=True)
