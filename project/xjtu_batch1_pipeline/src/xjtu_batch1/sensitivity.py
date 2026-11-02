from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .qc import INITIAL_PROTOCOL, STANDARD_PROTOCOL


QREF_DEFINITIONS = (
    "nominal",
    "initial_test",
    "first_standard",
    "early_median5",
)

QREF_COLUMNS = {
    "nominal": "Qref_nominal_Ah",
    "initial_test": "Qref_initial_test_Ah",
    "first_standard": "Qref_first_standard_Ah",
    "early_median5": "Qref_early_median5_Ah",
}


def _safe_difference(value_a: float, value_b: float) -> float:
    return float(value_a - value_b) if np.isfinite(value_a) and np.isfinite(value_b) else np.nan


def _safe_percent_difference(value_a: float, value_b: float) -> float:
    if not np.isfinite(value_a) or not np.isfinite(value_b) or value_b == 0:
        return np.nan
    return float(100.0 * (value_a - value_b) / value_b)


def compute_qref_summary(qc: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell_id, cell in qc.groupby("Cell_ID", sort=True):
        ordered = cell.sort_values("Raw_Cycle_Index")
        nominal = float(config["nominal_capacity_Ah"])

        cycle1 = ordered.loc[ordered["Raw_Cycle_Index"] == 1]
        initial_available = bool(
            len(cycle1) == 1
            and cycle1.iloc[0]["Protocol_Class"] == INITIAL_PROTOCOL
            and cycle1.iloc[0]["Primary_Include_Status"] == "YES"
            and bool(cycle1.iloc[0]["Charge_Complete"])
            and bool(cycle1.iloc[0]["Discharge_Complete"])
        )
        initial = (
            float(cycle1.iloc[0]["Summary_Discharge_Capacity_Ah"])
            if initial_available
            else np.nan
        )

        standard_valid = ordered.loc[
            (ordered["Protocol_Class"] == STANDARD_PROTOCOL)
            & (ordered["Primary_Include_Status"] == "YES")
            & ordered["Charge_Complete"].astype(bool)
            & ordered["Discharge_Complete"].astype(bool)
            & (ordered["Summary_Discharge_Capacity_Ah"] > 0)
        ].sort_values("Raw_Cycle_Index")

        first_available = len(standard_valid) >= 1
        first_standard = (
            float(standard_valid.iloc[0]["Summary_Discharge_Capacity_Ah"])
            if first_available
            else np.nan
        )
        first_source = int(standard_valid.iloc[0]["Raw_Cycle_Index"]) if first_available else pd.NA

        median5_available = len(standard_valid) >= 5
        early = standard_valid.head(5)
        early_median5 = (
            float(early["Summary_Discharge_Capacity_Ah"].median())
            if median5_available
            else np.nan
        )
        early_sources = (
            "|".join(str(int(value)) for value in early["Raw_Cycle_Index"].tolist())
            if median5_available
            else ""
        )

        rows.append(
            {
                "Cell_ID": cell_id,
                "Qref_nominal_Ah": nominal,
                "Qref_nominal_Status": "AVAILABLE",
                "Qref_Nominal_Source": "CONFIG_MANUFACTURER_NOMINAL_CAPACITY",
                "Qref_initial_test_Ah": initial,
                "Qref_initial_test_Status": (
                    "AVAILABLE" if initial_available else "QREF_INITIAL_TEST_UNAVAILABLE"
                ),
                "Qref_Initial_Test_Source_Cycle": 1 if initial_available else pd.NA,
                "Qref_first_standard_Ah": first_standard,
                "Qref_first_standard_Status": (
                    "AVAILABLE" if first_available else "QREF_FIRST_STANDARD_UNAVAILABLE"
                ),
                "Qref_First_Standard_Source_Cycle": first_source,
                "Qref_early_median5_Ah": early_median5,
                "Qref_early_median5_Status": (
                    "AVAILABLE" if median5_available else "QREF_EARLY_MEDIAN5_UNAVAILABLE"
                ),
                "Qref_Early_Median5_Source_Cycles": early_sources,
                "Qref_initial_minus_nominal_Ah": _safe_difference(initial, nominal),
                "Qref_initial_minus_nominal_pct": _safe_percent_difference(initial, nominal),
                "Qref_firststandard_minus_initial_Ah": _safe_difference(first_standard, initial),
                "Qref_firststandard_minus_initial_pct": _safe_percent_difference(
                    first_standard, initial
                ),
                "Qref_median5_minus_initial_Ah": _safe_difference(early_median5, initial),
                "Qref_median5_minus_initial_pct": _safe_percent_difference(
                    early_median5, initial
                ),
            }
        )

    frame = pd.DataFrame(rows).sort_values("Cell_ID").reset_index(drop=True)
    for column in (
        "Qref_Initial_Test_Source_Cycle",
        "Qref_First_Standard_Source_Cycle",
    ):
        frame[column] = pd.array(frame[column], dtype="Int64")
    return frame


def _qref_value(row: pd.Series, definition: str) -> float:
    value = row[QREF_COLUMNS[definition]]
    return float(value) if pd.notna(value) else np.nan


def compute_soh_labels(qc: pd.DataFrame, qref: pd.DataFrame) -> pd.DataFrame:
    qref_by_cell = qref.set_index("Cell_ID")
    rows: list[dict[str, Any]] = []
    clean_standard = qc.loc[
        (qc["Protocol_Class"] == STANDARD_PROTOCOL)
        & (qc["Primary_Include_Status"] == "YES")
    ].sort_values(["Cell_ID", "Raw_Cycle_Index"])

    for cycle in clean_standard.itertuples(index=False):
        qref_row = qref_by_cell.loc[cycle.Cell_ID]
        capacity = float(cycle.Summary_Discharge_Capacity_Ah)
        for definition in QREF_DEFINITIONS:
            reference = _qref_value(qref_row, definition)
            soh_fraction = capacity / reference if np.isfinite(reference) and reference > 0 else np.nan
            rows.append(
                {
                    "Cell_ID": cycle.Cell_ID,
                    "Raw_Cycle_Index": int(cycle.Raw_Cycle_Index),
                    "Aging_Cycle_Index": int(cycle.Aging_Cycle_Index),
                    "Capacity_Ah": capacity,
                    "Auto_QC_Flag": cycle.Auto_QC_Flag,
                    "Primary_Include_Status": cycle.Primary_Include_Status,
                    "Qref_Definition": definition,
                    "Qref_Ah": reference,
                    "SOH_Fraction": soh_fraction,
                    "SOH_Percent": 100.0 * soh_fraction if np.isfinite(soh_fraction) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _detect_eol(
    cycles: pd.DataFrame,
    threshold_Ah: float,
) -> tuple[str, int | pd._libs.missing.NAType, int | pd._libs.missing.NAType, str]:
    if cycles.empty:
        return "RIGHT_CENSORED", pd.NA, pd.NA, "NO_PRIMARY_INCLUDED_STANDARD_CYCLES"
    crossing = cycles.loc[cycles["Summary_Discharge_Capacity_Ah"] <= threshold_Ah]
    if crossing.empty:
        return (
            "RIGHT_CENSORED",
            pd.NA,
            pd.NA,
            "NO_PRIMARY_INCLUDED_CAPACITY_AT_OR_BELOW_THRESHOLD",
        )
    first = crossing.sort_values("Raw_Cycle_Index").iloc[0]
    return (
        "OBSERVED",
        int(first["Raw_Cycle_Index"]),
        int(first["Aging_Cycle_Index"]),
        "",
    )


def compute_eol_sensitivity(
    qc: pd.DataFrame,
    qref: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    qref_by_cell = qref.set_index("Cell_ID")
    for cell_id, cell in qc.groupby("Cell_ID", sort=True):
        primary = cell.loc[
            (cell["Protocol_Class"] == STANDARD_PROTOCOL)
            & (cell["Primary_Include_Status"] == "YES")
        ].sort_values("Raw_Cycle_Index")
        provisional = cell.loc[
            (cell["Protocol_Class"] == STANDARD_PROTOCOL)
            & (cell["Provisional_Include"] == "YES")
        ].sort_values("Raw_Cycle_Index")
        last_valid_raw = int(primary.iloc[-1]["Raw_Cycle_Index"]) if not primary.empty else pd.NA
        last_valid_capacity = (
            float(primary.iloc[-1]["Summary_Discharge_Capacity_Ah"])
            if not primary.empty
            else np.nan
        )
        qref_row = qref_by_cell.loc[cell_id]

        definitions: list[tuple[str, float, float, str]] = []
        for definition in QREF_DEFINITIONS:
            reference = _qref_value(qref_row, definition)
            threshold = (
                float(config["EOL_SOH_threshold"]) * reference
                if np.isfinite(reference)
                else np.nan
            )
            definitions.append((definition, reference, threshold, "QREF_RELATIVE"))
        definitions.append(
            (
                "fixed_absolute_1p6Ah",
                np.nan,
                float(config["fixed_absolute_EOThreshold_Ah"]),
                "FIXED_ABSOLUTE_CONTROL",
            )
        )

        for definition, reference, threshold, threshold_type in definitions:
            if not np.isfinite(threshold):
                status, raw_cycle, aging_cycle, reason = (
                    "QREF_UNAVAILABLE",
                    pd.NA,
                    pd.NA,
                    "QREF_UNAVAILABLE",
                )
                provisional_status, provisional_raw, provisional_aging = (
                    "QREF_UNAVAILABLE",
                    pd.NA,
                    pd.NA,
                )
            else:
                status, raw_cycle, aging_cycle, reason = _detect_eol(primary, threshold)
                (
                    provisional_status,
                    provisional_raw,
                    provisional_aging,
                    _,
                ) = _detect_eol(provisional, threshold)
            rows.append(
                {
                    "Cell_ID": cell_id,
                    "Qref_Definition": definition,
                    "Threshold_Type": threshold_type,
                    "Qref_Ah": reference,
                    "EOThreshold_Ah": threshold,
                    "EOL_Observed": "YES" if status == "OBSERVED" else "NO",
                    "EOL_Status": status,
                    "EOL_Raw_Cycle": raw_cycle,
                    "EOL_Aging_Cycle": aging_cycle,
                    "Last_Valid_Raw_Cycle": last_valid_raw,
                    "Last_Valid_Capacity_Ah": last_valid_capacity,
                    "Censoring_Reason": reason,
                    "Provisional_EOL_Status": provisional_status,
                    "Provisional_EOL_Raw_Cycle": provisional_raw,
                    "Provisional_EOL_Aging_Cycle": provisional_aging,
                }
            )

    frame = pd.DataFrame(rows)
    for column in (
        "EOL_Raw_Cycle",
        "EOL_Aging_Cycle",
        "Last_Valid_Raw_Cycle",
        "Provisional_EOL_Raw_Cycle",
        "Provisional_EOL_Aging_Cycle",
    ):
        frame[column] = pd.array(frame[column], dtype="Int64")
    order = {definition: index for index, definition in enumerate((*QREF_DEFINITIONS, "fixed_absolute_1p6Ah"))}
    frame["_Definition_Order"] = frame["Qref_Definition"].map(order)
    return frame.sort_values(["Cell_ID", "_Definition_Order"]).drop(columns="_Definition_Order").reset_index(drop=True)


def compute_eol_pairwise(eol: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    relative = eol.loc[eol["Qref_Definition"].isin(QREF_DEFINITIONS)]
    for cell_id, cell in relative.groupby("Cell_ID", sort=True):
        lookup = cell.set_index("Qref_Definition")
        for definition_a, definition_b in combinations(QREF_DEFINITIONS, 2):
            a = lookup.loc[definition_a]
            b = lookup.loc[definition_b]
            status_a = str(a["EOL_Status"])
            status_b = str(b["EOL_Status"])
            if status_a == "OBSERVED" and status_b == "OBSERVED":
                delta = int(b["EOL_Aging_Cycle"]) - int(a["EOL_Aging_Cycle"])
                state_change = "BOTH_OBSERVED"
            else:
                delta = np.nan
                if status_a == "RIGHT_CENSORED" and status_b == "RIGHT_CENSORED":
                    state_change = "BOTH_CENSORED"
                elif status_a == "OBSERVED" and status_b == "RIGHT_CENSORED":
                    state_change = "OBSERVED_TO_CENSORED"
                elif status_a == "RIGHT_CENSORED" and status_b == "OBSERVED":
                    state_change = "CENSORED_TO_OBSERVED"
                else:
                    state_change = "UNAVAILABLE_COMPARISON"
            rows.append(
                {
                    "Cell_ID": cell_id,
                    "Definition_A": definition_a,
                    "Definition_B": definition_b,
                    "Qref_A_Ah": a["Qref_Ah"],
                    "Qref_B_Ah": b["Qref_Ah"],
                    "EOL_Status_A": status_a,
                    "EOL_Status_B": status_b,
                    "EOL_A": a["EOL_Aging_Cycle"],
                    "EOL_B": b["EOL_Aging_Cycle"],
                    "EOL_Raw_A": a["EOL_Raw_Cycle"],
                    "EOL_Raw_B": b["EOL_Raw_Cycle"],
                    "Delta_EOL_Cycles": delta,
                    "Censoring_Status_Change": state_change,
                }
            )
    frame = pd.DataFrame(rows)
    for column in ("EOL_A", "EOL_B", "EOL_Raw_A", "EOL_Raw_B"):
        frame[column] = pd.array(frame[column], dtype="Int64")
    return frame


def compute_rul_sensitivity(qc: pd.DataFrame, eol: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    clean_standard = qc.loc[
        (qc["Protocol_Class"] == STANDARD_PROTOCOL)
        & (qc["Primary_Include_Status"] == "YES")
    ].sort_values(["Cell_ID", "Raw_Cycle_Index"])
    eol_relative = eol.loc[eol["Qref_Definition"].isin(QREF_DEFINITIONS)]
    eol_lookup = eol_relative.set_index(["Cell_ID", "Qref_Definition"])

    for cycle in clean_standard.itertuples(index=False):
        for definition in QREF_DEFINITIONS:
            eol_row = eol_lookup.loc[(cycle.Cell_ID, definition)]
            status = str(eol_row["EOL_Status"])
            eol_aging = eol_row["EOL_Aging_Cycle"]
            if status == "OBSERVED" and pd.notna(eol_aging):
                if int(cycle.Aging_Cycle_Index) <= int(eol_aging):
                    rul = int(eol_aging) - int(cycle.Aging_Cycle_Index)
                    raw_rul = int(eol_row["EOL_Raw_Cycle"]) - int(cycle.Raw_Cycle_Index)
                    if rul != raw_rul:
                        raise AssertionError(
                            f"RUL index mismatch for {cycle.Cell_ID} Raw Cycle {cycle.Raw_Cycle_Index}"
                        )
                    rul_status = "DEFINED"
                else:
                    rul = np.nan
                    rul_status = "POST_EOL_EXCLUDED"
            elif status == "RIGHT_CENSORED":
                rul = np.nan
                rul_status = "CENSORED_EOL"
            else:
                rul = np.nan
                rul_status = "QREF_UNAVAILABLE"
            rows.append(
                {
                    "Cell_ID": cycle.Cell_ID,
                    "Raw_Cycle_Index": int(cycle.Raw_Cycle_Index),
                    "Aging_Cycle_Index": int(cycle.Aging_Cycle_Index),
                    "Qref_Definition": definition,
                    "EOL_Status": status,
                    "EOL_Aging_Cycle": eol_aging,
                    "RUL_Cycles": rul,
                    "RUL_Status": rul_status,
                }
            )
    frame = pd.DataFrame(rows)
    frame["EOL_Aging_Cycle"] = pd.array(frame["EOL_Aging_Cycle"], dtype="Int64")
    return frame


def compute_pilot_summary(
    qc: pd.DataFrame,
    qref: pd.DataFrame,
    eol: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    qref_lookup = qref.set_index("Cell_ID")
    for cell_id, cell in qc.groupby("Cell_ID", sort=True):
        qr = qref_lookup.loc[cell_id]
        eol_cell = eol.loc[eol["Cell_ID"] == cell_id].set_index("Qref_Definition")
        pair_cell = pairwise.loc[pairwise["Cell_ID"] == cell_id]
        observed_shifts = pd.to_numeric(pair_cell["Delta_EOL_Cycles"], errors="coerce").dropna()
        status_change = pair_cell["Censoring_Status_Change"].isin(
            ["OBSERVED_TO_CENSORED", "CENSORED_TO_OBSERVED"]
        ).any()

        def eol_value(definition: str, field: str) -> Any:
            return eol_cell.loc[definition, field]

        counts = cell["Auto_QC_Flag"].value_counts()
        primary = cell.loc[
            (cell["Protocol_Class"] == STANDARD_PROTOCOL)
            & (cell["Primary_Include_Status"] == "YES")
        ]
        rows.append(
            {
                "Cell_ID": cell_id,
                "Total_Raw_Cycles": int(len(cell)),
                "PASS_Count": int(counts.get("PASS", 0)),
                "CHECK_Count": int(counts.get("CHECK", 0)),
                "INVALID_Count": int(counts.get("INVALID_CANDIDATE", 0)),
                "Review_Count": int((cell["Primary_Include_Status"] == "REVIEW").sum()),
                "Flagged_Cycle_Count": int((cell["Auto_QC_Flag"] != "PASS").sum()),
                "Last_Valid_Cycle": (
                    int(primary["Raw_Cycle_Index"].max()) if not primary.empty else pd.NA
                ),
                "Qref_nominal": qr["Qref_nominal_Ah"],
                "Qref_initial_test": qr["Qref_initial_test_Ah"],
                "Qref_first_standard": qr["Qref_first_standard_Ah"],
                "Qref_early_median5": qr["Qref_early_median5_Ah"],
                "EOL_nominal_status": eol_value("nominal", "EOL_Status"),
                "EOL_nominal_cycle": eol_value("nominal", "EOL_Raw_Cycle"),
                "EOL_initial_status": eol_value("initial_test", "EOL_Status"),
                "EOL_initial_cycle": eol_value("initial_test", "EOL_Raw_Cycle"),
                "EOL_first_standard_status": eol_value("first_standard", "EOL_Status"),
                "EOL_first_standard_cycle": eol_value("first_standard", "EOL_Raw_Cycle"),
                "EOL_median5_status": eol_value("early_median5", "EOL_Status"),
                "EOL_median5_cycle": eol_value("early_median5", "EOL_Raw_Cycle"),
                "Fixed_1p6Ah_EOL_Status": eol_value(
                    "fixed_absolute_1p6Ah", "EOL_Status"
                ),
                "Fixed_1p6Ah_EOL_Cycle": eol_value(
                    "fixed_absolute_1p6Ah", "EOL_Raw_Cycle"
                ),
                "Max_Observed_Pairwise_EOL_Shift": (
                    float(observed_shifts.abs().max()) if not observed_shifts.empty else np.nan
                ),
                "Any_Censoring_Status_Change": bool(status_change),
            }
        )
    frame = pd.DataFrame(rows)
    for column in (
        "Last_Valid_Cycle",
        "EOL_nominal_cycle",
        "EOL_initial_cycle",
        "EOL_first_standard_cycle",
        "EOL_median5_cycle",
        "Fixed_1p6Ah_EOL_Cycle",
    ):
        frame[column] = pd.array(frame[column], dtype="Int64")
    return frame

