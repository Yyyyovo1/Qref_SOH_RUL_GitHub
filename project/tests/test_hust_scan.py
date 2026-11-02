from __future__ import annotations

import io
import pickle
import unittest

import numpy as np
import pandas as pd

from hust_repro.config import DEFAULT_SCAN_CONFIG
from hust_repro.safe_pickle import restricted_load
from hust_repro.scan import (
    _apply_protocol_and_jump_qc,
    _cycle_index_status,
    _cycle_metrics,
    _select_representatives,
)


class UnsafeObject:
    pass


class SafePickleTests(unittest.TestCase):
    def test_plain_mapping_loads(self) -> None:
        payload = pickle.dumps({"cell": {"rul": {}, "dq": {}, "data": {}}})
        self.assertIn("cell", restricted_load(io.BytesIO(payload)))

    def test_unexpected_global_is_blocked(self) -> None:
        payload = pickle.dumps(UnsafeObject())
        with self.assertRaisesRegex(pickle.UnpicklingError, "Blocked"):
            restricted_load(io.BytesIO(payload))


class CycleMetricTests(unittest.TestCase):
    @staticmethod
    def raw_cycle(cycle: int = 1) -> pd.DataFrame:
        rows: list[dict[str, object]] = []

        def add(status: str, currents: list[float], voltages: list[float], capacities: list[float], times: list[float]) -> None:
            for current, voltage, capacity, time in zip(currents, voltages, capacities, times):
                rows.append(
                    {
                        "Status": status,
                        "Cycle number": cycle,
                        "Current (mA)": current,
                        "Voltage (V)": voltage,
                        "Capacity (mAh)": capacity,
                        "Time (s)": time,
                    }
                )

        add("Constant current charge", [5500, 5500], [2.8, 3.59], [0, 850], [0, 500])
        add(
            "Constant current-constant voltage charge",
            [1100, 1100, 55],
            [3.42, 3.60, 3.60],
            [850, 1100, 1200],
            [510, 1000, 1500],
        )
        add("Constant current discharge_0", [-5500, -5500], [3.30, 3.05], [1200, 800], [1510, 1800])
        add("Constant current discharge_1", [-3300, -3300], [3.15, 3.09], [800, 600], [1810, 2000])
        add("Constant current discharge_2", [-2200, -2200], [3.12, 3.04], [600, 400], [2010, 2200])
        add("Constant current discharge_3", [-1100, -1100], [3.10, 2.00], [400, 50], [2210, 2500])
        return pd.DataFrame(rows)

    def test_full_cycle_capacity_semantics(self) -> None:
        row = _cycle_metrics(
            "HUST_1-1",
            "1-1",
            "1-1.pkl",
            1,
            self.raw_cycle(),
            1150.0,
            10,
            DEFAULT_SCAN_CONFIG,
        )
        self.assertAlmostEqual(row["Raw_Full_Discharge_Capacity_mAh"], 1150.0)
        self.assertEqual(row["Capacity_Semantics_Match"], "YES")
        self.assertEqual(row["Full_Discharge_Traceable"], "YES")
        self.assertEqual(row["Full_Discharge_Complete_Scan_Candidate"], "YES")
        self.assertEqual(row["Auto_QC_Flag"], "PASS")

    def test_cycle_identity_gap_is_preserved(self) -> None:
        status = _cycle_index_status([1, 2, 4], [1, 2, 4])
        self.assertEqual(status["Cycle_Index_Status"], "MISSING_CYCLE_INDEX")
        self.assertEqual(status["Missing_Cycle_Indexes"], "3")

    def test_capacity_jump_is_check_only(self) -> None:
        base = {
            "Cell_ID": "HUST_1-1",
            "Native_Cell_ID": "1-1",
            "Source_File": "1-1.pkl",
            "Cycle_Number_Field_Status": "MATCH",
            "Raw_Record_Count": 1,
            "Raw_Status_Count": 6,
            "Raw_Status_Sequence": "",
            "Native_RUL_Cycles": 1,
            "Raw_Full_Discharge_Capacity_mAh": 1100,
            "Capacity_Difference_mAh": 0,
            "Capacity_Difference_Pct": 0,
            "Capacity_Semantics_Match": "YES",
            "Charge_Stage1_Median_Current_A": 5.5,
            "Charge_Stage2_Plateau_Median_Current_A": 1.1,
            "Charge_Max_V": 3.6,
            "CV_End_Current_A": 0.055,
            "Charge_Protocol_Traceable": "YES",
            "Charge_Complete_Scan_Candidate": "YES",
            "Discharge_Step1_Median_Current_A": -5.5,
            "Discharge_Step2_Median_Current_A": -3.3,
            "Discharge_Step3_Median_Current_A": -2.2,
            "Discharge_Step4_Median_Current_A": -1.1,
            "Discharge_Step1_Capacity_mAh": 400,
            "Discharge_Step2_Capacity_mAh": 200,
            "Discharge_Step3_Capacity_mAh": 200,
            "Discharge_Step4_Capacity_mAh": 300,
            "Discharge_Min_V": 2.0,
            "Discharge_End_V": 2.0,
            "Final_Discharge_Current_A": -1.1,
            "Full_Discharge_Traceable": "YES",
            "Full_Discharge_Complete_Scan_Candidate": "YES",
            "Severe_Voltage_Anomaly": "NO",
            "Protocol_Current_Deviation": "NO",
            "Protocol_Change_Candidate": "NO",
            "Capacity_Jump_Flag": "NO",
            "Jump_Reference_Raw_Cycle": np.nan,
            "Jump_Reference_Capacity_Ah": np.nan,
            "Capacity_Jump_Ah": np.nan,
            "Capacity_Jump_Pct": np.nan,
            "Auto_QC_Flag": "PASS",
            "Auto_QC_Reason_Codes": "PASS",
            "Primary_Capacity_Candidate": "YES",
        }
        rows = []
        for cycle, capacity in [(1, 1.10), (2, 0.90)]:
            row = dict(base)
            row.update(
                {
                    "Raw_Cycle_Index": cycle,
                    "Native_Discharge_Capacity_mAh": capacity * 1000,
                    "Primary_Capacity_Ah": capacity,
                }
            )
            rows.append(row)
        result, _ = _apply_protocol_and_jump_qc(
            pd.DataFrame(rows), DEFAULT_SCAN_CONFIG
        )
        self.assertEqual(result.iloc[1]["Capacity_Jump_Flag"], "YES")
        self.assertEqual(result.iloc[1]["Auto_QC_Flag"], "CHECK")
        self.assertNotEqual(result.iloc[1]["Auto_QC_Flag"], "INVALID_CANDIDATE")


class RepresentativeSelectionTests(unittest.TestCase):
    def test_deterministic_eight_cell_selection(self) -> None:
        rows = []
        for index in range(1, 13):
            rows.append(
                {
                    "Cell_ID": f"HUST_1-{index}",
                    "Native_Cell_ID": f"1-{index}",
                    "Source_File": f"1-{index}.pkl",
                    "Cycle_Count": 1000 + index * 100,
                    "First_Valid_Capacity_Ah": 1.0 + index / 100,
                    "Protocol_Fingerprint_ID": f"P{index % 4}",
                    "Processing_Status": "SCAN_PROCESSED",
                }
            )
        source = pd.DataFrame(rows)
        first = _select_representatives(source, DEFAULT_SCAN_CONFIG)
        second = _select_representatives(source, DEFAULT_SCAN_CONFIG)
        self.assertEqual(len(first), 8)
        self.assertEqual(first["Cell_ID"].tolist(), second["Cell_ID"].tolist())
        self.assertEqual(first["Cell_ID"].nunique(), 8)


if __name__ == "__main__":
    unittest.main()
