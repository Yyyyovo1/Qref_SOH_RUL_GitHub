from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .mat_io import SchemaResult
from .sensitivity import QREF_DEFINITIONS


class RegressionFailure(AssertionError):
    """Raised when the manually verified battery-1 baseline is not reproduced."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegressionFailure(message)


def run_battery1_regressions(
    schema: SchemaResult,
    qc: pd.DataFrame,
    qref: pd.DataFrame,
    eol: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, str]:
    results: dict[str, str] = {}
    expected_count = int(config["battery1_expected_cycle_count"])

    _require(schema.schema_status == "PASS", f"battery1 schema status={schema.schema_status}")
    _require(schema.data_cycle_count == expected_count, "battery1 data cycle count mismatch")
    _require(schema.summary_cycle_life == expected_count, "battery1 cycle_life mismatch")
    _require(len(qc) == expected_count, "battery1 QC row count mismatch")
    _require(qc["Raw_Cycle_Index"].tolist() == list(range(1, expected_count + 1)), "Raw cycles changed")
    results["battery1 schema test"] = "PASSED"

    tolerance = float(config["battery1_capacity_regression_tolerance_Ah"])
    indexed = qc.set_index("Raw_Cycle_Index")
    for raw_text, expected in config["battery1_expected_capacities_Ah"].items():
        raw = int(raw_text)
        actual = float(indexed.loc[raw, "Summary_Discharge_Capacity_Ah"])
        _require(
            abs(actual - float(expected)) <= tolerance,
            f"battery1 Raw Cycle {raw} capacity {actual:.6f} Ah != {float(expected):.6f} Ah",
        )
    results["battery1 capacity test"] = "PASSED"

    cycle1 = indexed.loc[1]
    cycle2 = indexed.loc[2]
    _require(
        "0.5c charge and 0.2c discharge" in str(cycle1["Description"]).lower(),
        "battery1 Cycle 1 description mismatch",
    )
    _require(
        "2c charge and 1c discharge" in str(cycle2["Description"]).lower(),
        "battery1 Cycle 2 description mismatch",
    )
    current_tolerance = float(config["battery1_current_regression_tolerance_A"])
    _require(
        abs(
            float(cycle1["Discharge_Median_Current_A"])
            - float(config["battery1_cycle1_expected_discharge_current_A"])
        )
        <= current_tolerance,
        "battery1 Cycle 1 discharge current mismatch",
    )
    _require(
        abs(
            float(cycle2["Discharge_Median_Current_A"])
            - float(config["battery1_cycle2_expected_discharge_current_A"])
        )
        <= current_tolerance,
        "battery1 Cycle 2 discharge current mismatch",
    )
    _require(abs(float(cycle1["Discharge_Min_V"]) - 2.5) <= 0.05, "Cycle 1 cutoff mismatch")
    _require(abs(float(cycle2["Discharge_Min_V"]) - 2.5) <= 0.05, "Cycle 2 cutoff mismatch")
    results["battery1 protocol test"] = "PASSED"

    cycle390 = indexed.loc[390]
    _require(
        abs(
            float(cycle390["Discharge_Min_V"])
            - float(config["battery1_cycle390_expected_min_voltage_V"])
        )
        <= float(config["battery1_cycle390_voltage_tolerance_V"]),
        "battery1 Cycle 390 minimum voltage mismatch",
    )
    codes390 = set(str(cycle390["Auto_QC_Reason_Codes"]).split("|"))
    _require(
        "INCOMPLETE_DISCHARGE_PROTOCOL" in codes390,
        "battery1 Cycle 390 did not trigger incomplete discharge",
    )
    _require(
        "VOLTAGE_SPIKE_DURING_DISCHARGE" in codes390
        or "ABNORMAL_VOLTAGE_JUMP" in codes390,
        "battery1 Cycle 390 did not trigger abnormal discharge voltage",
    )
    _require(
        cycle390["Auto_QC_Flag"] == "INVALID_CANDIDATE",
        "battery1 Cycle 390 must be INVALID_CANDIDATE",
    )
    results["battery1 cycle390 anomaly test"] = "PASSED"

    qref_row = qref.loc[qref["Cell_ID"] == "2C_battery-1"].iloc[0]
    expected_qrefs = config["battery1_qref_expected_Ah"]
    actual_qrefs = {
        "nominal": float(qref_row["Qref_nominal_Ah"]),
        "initial_test": float(qref_row["Qref_initial_test_Ah"]),
        "first_standard": float(qref_row["Qref_first_standard_Ah"]),
        "early_median5": float(qref_row["Qref_early_median5_Ah"]),
    }
    for definition, expected in expected_qrefs.items():
        _require(
            np.isfinite(actual_qrefs[definition])
            and abs(actual_qrefs[definition] - float(expected)) <= tolerance,
            f"battery1 Qref {definition}={actual_qrefs[definition]:.6f} Ah; expected {expected}",
        )
    _require(
        str(qref_row["Qref_Early_Median5_Source_Cycles"]) == "2|3|4|5|6",
        "battery1 median-5 source cycles mismatch",
    )
    results["battery1 Qref test"] = "PASSED"

    eol_battery1 = eol.loc[
        (eol["Cell_ID"] == "2C_battery-1")
        & (eol["Qref_Definition"].isin(QREF_DEFINITIONS))
    ]
    _require(len(eol_battery1) == 4, "battery1 EOL result count mismatch")
    _require(
        (eol_battery1["EOL_Status"] == "RIGHT_CENSORED").all(),
        "battery1 relative-Qref EOL results must all be right-censored",
    )
    _require(
        eol_battery1["EOL_Raw_Cycle"].isna().all(),
        "battery1 censored EOL must not contain a cycle number",
    )
    _require(
        int(eol_battery1["Last_Valid_Raw_Cycle"].max()) == 389,
        "battery1 last valid cycle must be 389",
    )
    fixed = eol.loc[
        (eol["Cell_ID"] == "2C_battery-1")
        & (eol["Qref_Definition"] == "fixed_absolute_1p6Ah")
    ].iloc[0]
    _require(fixed["EOL_Status"] == "RIGHT_CENSORED", "battery1 fixed 1.6 Ah EOL mismatch")
    results["battery1 EOL censoring test"] = "PASSED"
    return results

