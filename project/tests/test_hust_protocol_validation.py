from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hust_repro.config import make_protocol_validation_config
from hust_repro.protocol_validation import (
    FINAL_ASSESSMENT_RAW_DIFFERS,
    analyze_raw_cycle,
    determine_final_assessment,
    select_validation_cycles,
)


class SelectionTests(unittest.TestCase):
    def test_middle_cycle_uses_half_up_rounding(self) -> None:
        selected = select_validation_cycles(list(range(1, 2217)))
        self.assertEqual(
            [row["Raw_Cycle_Index"] for row in selected], [1, 2, 1109, 2216]
        )

    def test_missing_middle_selects_nearest_actual_without_reindexing(self) -> None:
        cycles = [1, 2, 3, 5, 6, 7, 8]
        selected = select_validation_cycles(cycles)
        self.assertEqual(selected[2]["Target_Raw_Cycle_Index"], 5)
        self.assertEqual(selected[2]["Raw_Cycle_Index"], 5)


class RawCycleReconstructionTests(unittest.TestCase):
    @staticmethod
    def raw_cycle() -> pd.DataFrame:
        rows: list[dict[str, object]] = []

        def add(
            status: str,
            currents_mA: list[float],
            voltages: list[float],
            capacities: list[float],
            times: list[float],
        ) -> None:
            for current, voltage, capacity, time in zip(
                currents_mA, voltages, capacities, times
            ):
                rows.append(
                    {
                        "Status": status,
                        "Cycle number": 1,
                        "Current (mA)": current,
                        "Voltage (V)": voltage,
                        "Capacity (mAh)": capacity,
                        "Time (s)": time,
                    }
                )

        add("Constant current charge", [5500, 5500], [2.8, 3.55], [0, 850], [0, 500])
        add(
            "Constant current-constant voltage charge",
            [1100, 1100, 500, 55],
            [3.40, 3.58, 3.599, 3.60],
            [850, 1050, 1150, 1200],
            [530, 800, 1000, 1200],
        )
        add("Constant current discharge_0", [-5500, -5500], [3.35, 3.15], [1200, 900], [1210, 1400])
        add("Constant current discharge_1", [-3300, -3300], [3.20, 3.10], [900, 600], [1410, 1600])
        add("Constant current discharge_2", [-1100, -1100], [3.15, 3.05], [600, 300], [1610, 1800])
        add("Constant current discharge_3", [-2200, -2200], [3.10, 1.98], [300, 0], [1810, 2000])
        return pd.DataFrame(rows)

    def config(self) -> dict[str, object]:
        return make_protocol_validation_config(
            Path("."), Path("HUST 77"), Path("outputs/hust_scan_v1"), Path("outputs/test")
        )

    def test_raw_order_four_stage_and_capacity_semantics(self) -> None:
        steps, terminal, capacity, _ = analyze_raw_cycle(
            cell_id="HUST_2-5",
            source_file="2-5.pkl",
            raw_cycle=1,
            cycle_position="INITIAL",
            reason_selected="RAW_CYCLE_1",
            frame=self.raw_cycle(),
            native_dq_mAh=1200.0,
            config=self.config(),
        )
        self.assertEqual(terminal["Raw_Time_Order_Monotonic"], "YES")
        self.assertEqual(terminal["Four_Discharge_Stages_Identified"], "YES")
        self.assertEqual(terminal["Terminal_Current_Class"], "APPROX_2C")
        self.assertEqual(
            terminal["Mapping_Status"],
            "RAW_STATUS_INDEX_MATCHES_PHYSICAL_TIME_ORDER",
        )
        self.assertAlmostEqual(capacity["Raw_Reconstructed_Discharge_mAh"], 1200.0)
        self.assertAlmostEqual(capacity["Stage_Contribution_Sum_mAh"], 1200.0)
        self.assertEqual(capacity["Validation_Status"], "PASS")
        functions = [row["Physical_Function"] for row in steps]
        self.assertIn("CHARGE_CC_LOW_RATE", functions)
        self.assertIn("CHARGE_CV", functions)
        self.assertEqual(functions[-4:], [
            "DISCHARGE_STAGE_1",
            "DISCHARGE_STAGE_2",
            "DISCHARGE_STAGE_3",
            "DISCHARGE_STAGE_4",
        ])

    def test_uniform_two_c_result_classifies_as_raw_paper_difference(self) -> None:
        frame = pd.DataFrame(
            {
                "Protocol_Validation_Flag": ["PASS"] * 16,
                "Terminal_Current_Class": ["APPROX_2C"] * 16,
                "Mapping_Status": [
                    "RAW_STATUS_INDEX_MATCHES_PHYSICAL_TIME_ORDER"
                ]
                * 16,
            }
        )
        self.assertEqual(
            determine_final_assessment(frame), FINAL_ASSESSMENT_RAW_DIFFERS
        )


if __name__ == "__main__":
    unittest.main()
