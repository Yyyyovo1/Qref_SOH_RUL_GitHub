"""Cycle QC, cell list, Qref table and folds for TJU NCA / NCM (same structure as the HUST / MIT freezes)."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import (
    CHARGE_CUTOFF_V,
    DISCHARGE_CUTOFF_V,
    EXPECTED_DISCHARGE_CURRENT_MA,
    N_FOLDS,
    NOMINAL_CAPACITY_AH,
    QC,
    RANDOM_SEED,
)

MIN_ELIGIBLE_CYCLES = 5  # needed for the early-median-5 reference


def qc_cycles(inv: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    out = inv.reset_index(drop=True).copy()
    out["File_Order"] = out.groupby("Cell_ID").cumcount()
    rate = manifest.set_index("Cell_ID").Charge_CRate_Name
    reasons = [[] for _ in range(len(out))]
    # A time reset inside a file starts a new run; blocks before the last reset are superseded.
    for cell, g in out.groupby("Cell_ID", sort=False):
        resets = g.index[g.Start_Time_s.diff() < 0]
        if len(resets):
            for i in g.index[g.index < resets.max()]:
                reasons[i].append("SUPERSEDED_BY_TIME_RESET")
    for i, r in out.iterrows():
        nom = NOMINAL_CAPACITY_AH[r.Dataset]
        if r.Discharge_Rows == 0:
            reasons[i].append("NO_DISCHARGE")
            continue
        q = r.QDischarge_Max_Ah
        if not np.isfinite(q) or q <= 0:
            reasons[i].append("CAPACITY_NONFINITE_OR_NONPOSITIVE")
        elif q > QC["max_capacity_factor_of_nominal"] * nom:
            reasons[i].append("CAPACITY_ABOVE_1P5_NOMINAL")
        tol = QC["protocol_discharge_current_relative_tolerance"]
        if abs(r.Discharge_Median_Current_mA - EXPECTED_DISCHARGE_CURRENT_MA) > tol * abs(EXPECTED_DISCHARGE_CURRENT_MA):
            reasons[i].append("DISCHARGE_CURRENT_PROTOCOL_INCONSISTENT")
        expected_chg = rate[r.Cell_ID] * nom * 1000.0
        if not np.isfinite(r.Charge_CC_Median_Current_mA) or abs(r.Charge_CC_Median_Current_mA - expected_chg) > tol * expected_chg:
            reasons[i].append("CHARGE_CURRENT_PROTOCOL_INCONSISTENT")
        if r.Discharge_Min_V > DISCHARGE_CUTOFF_V[r.Dataset] + QC["full_discharge_voltage_margin_V"]:
            reasons[i].append("INCOMPLETE_DISCHARGE")
        if not np.isfinite(r.Max_V_Before_Discharge) or r.Max_V_Before_Discharge < CHARGE_CUTOFF_V - QC["full_charge_voltage_margin_V"]:
            reasons[i].append("INCOMPLETE_CHARGE")
        if not np.isfinite(r.Discharge_Start_V) or r.Discharge_Start_V < QC["min_discharge_start_V"]:
            reasons[i].append("DISCHARGE_NOT_FROM_CHARGED_STATE")
    out["Exclusion_Reasons"] = ["|".join(x) for x in reasons]
    out["Primary_Capacity_Eligible"] = np.where(out.Exclusion_Reasons.eq(""), "YES", "NO")
    elig = out[out.Primary_Capacity_Eligible.eq("YES")]
    if elig.duplicated(["Cell_ID", "Cycle_Number"]).any():
        raise ValueError("DUPLICATE_ELIGIBLE_CELL_CYCLE")
    out["Discharge_Noncontiguous_Flag"] = np.where(out.Discharge_Contiguous.astype(str).isin(["False", "0"]), "YES", "NO")

    out["Capacity_Jump_Flag"] = "NO"
    out["Jump_Reference_Cycle"] = np.nan
    out["Capacity_Jump_Ah"] = np.nan
    for _, g in elig.sort_values(["Cell_ID", "Cycle_Number"]).groupby("Cell_ID", sort=False):
        ref = None
        for i in g.index:
            if ref is not None:
                delta = out.at[i, "QDischarge_Max_Ah"] - out.at[ref, "QDischarge_Max_Ah"]
                out.at[i, "Jump_Reference_Cycle"] = out.at[ref, "Cycle_Number"]
                out.at[i, "Capacity_Jump_Ah"] = delta
                if abs(delta) > max(QC["capacity_jump_abs_threshold_Ah"],
                                    QC["capacity_jump_relative_threshold"] * abs(out.at[ref, "QDischarge_Max_Ah"])):
                    out.at[i, "Capacity_Jump_Flag"] = "YES"
            ref = i
    return out


def primary_capacity(qc: pd.DataFrame) -> pd.DataFrame:
    p = qc[qc.Primary_Capacity_Eligible.eq("YES")].sort_values(["Dataset", "Cell_ID", "Cycle_Number"])
    return pd.DataFrame({"Dataset": p.Dataset, "Cell_ID": p.Cell_ID, "Cycle_Index": p.Cycle_Number.astype(int),
                         "Capacity_Ah": p.QDischarge_Max_Ah, "Capacity_Jump_Flag": p.Capacity_Jump_Flag}).reset_index(drop=True)


def cell_inventory(manifest: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    n = primary.groupby("Cell_ID").size()
    inv = manifest.copy()
    inv["Eligible_Cycles"] = inv.Cell_ID.map(n).fillna(0).astype(int)
    inv["Status"] = np.where(inv.Eligible_Cycles >= MIN_ELIGIBLE_CYCLES, "INCLUDED", "EXCLUDED")
    inv["Status_Reason"] = np.where(inv.Status.eq("INCLUDED"), "", f"FEWER_THAN_{MIN_ELIGIBLE_CYCLES}_ELIGIBLE_CYCLES")
    return inv


def qref_table(primary: pd.DataFrame, included: list[str]) -> pd.DataFrame:
    rows = []
    for (ds, cell), g in primary[primary.Cell_ID.isin(included)].groupby(["Dataset", "Cell_ID"], sort=True):
        g = g.sort_values("Cycle_Index")
        first5 = g.head(5)
        rows.append({"Dataset": ds, "Cell_ID": cell, "Qref_Nominal_Ah": NOMINAL_CAPACITY_AH[ds],
                     "Qref_First_Comparable_Ah": float(g.Capacity_Ah.iloc[0]),
                     "Qref_First_Comparable_Source_Cycle": int(g.Cycle_Index.iloc[0]),
                     "Qref_Early_Median5_Comparable_Ah": float(np.median(first5.Capacity_Ah)),
                     "Qref_Early_Median5_Comparable_Source_Cycles": "|".join(str(int(c)) for c in first5.Cycle_Index),
                     "Retained_Cycles": len(g), "Last_Retained_Cycle": int(g.Cycle_Index.iloc[-1]),
                     "Last_Retained_Capacity_Ah": float(g.Capacity_Ah.iloc[-1])})
    return pd.DataFrame(rows)


def assign_folds(cells: list[str], dataset: str) -> pd.DataFrame:
    """Same algorithm as soh_ml_design_v1 / MIT: SHA256-seeded cell order, round robin; one split per dataset."""
    order = sorted(set(cells), key=lambda c: hashlib.sha256(f"{RANDOM_SEED}|TJU_{dataset}|{c}".encode()).hexdigest())
    return pd.DataFrame([{"Dataset": dataset, "Cell_ID": c, "Fold_ID": i % N_FOLDS + 1} for i, c in enumerate(order)])
