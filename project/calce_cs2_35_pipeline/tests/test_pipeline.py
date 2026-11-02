from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calce_qc.config import (  # noqa: E402
    BENCHMARK_CAPACITY_AH,
    BENCHMARK_FILE,
    BENCHMARK_TOLERANCE_AH,
    DUPLICATE_FILES,
    get_profile,
)
from calce_qc.extract import extract_workbook  # noqa: E402
from calce_qc.pipeline import _assign_order_and_boundaries  # noqa: E402
from calce_qc.qc import finalize_qc  # noqa: E402


def build_full_profile_frame(cell_id: str):
    """Rebuild a full cell sequence from source workbooks without writing outputs."""

    profile = get_profile(cell_id)
    manifests = []
    rows = []
    for workbook in sorted(profile.input_dir.glob("*.xlsx")):
        manifest, workbook_rows = extract_workbook(
            workbook,
            cell_id=profile.cell_id,
        )
        manifests.append(manifest)
        rows.extend(workbook_rows)
    _manifests, ordered_rows = _assign_order_and_boundaries(manifests, rows, profile)
    return finalize_qc(
        ordered_rows,
        known_boundary_tails=profile.known_boundary_tails,
    )


def complete_synthetic_cycle(global_cycle: int, capacity_ah: float) -> dict[str, object]:
    """Create a protocol-complete row for isolated jump-rule tests."""

    return {
        "Cell_ID": "SYNTHETIC",
        "Source_File": "synthetic.xlsx",
        "File_Role": "CANONICAL",
        "Duplicate_Of": "",
        "Boundary_Status": "CONFIRMED",
        "Local_Cycle_Index": global_cycle,
        "Global_Cycle_Index": global_cycle,
        "Single_Cycle_Discharge_Capacity_Ah": capacity_ah,
        "CC_Stage_Present": True,
        "CC_Protocol_Complete": True,
        "CV_Stage_Present": True,
        "CV_Protocol_Complete": True,
        "Main_Discharge_Stage_Present": True,
        "Main_Discharge_Protocol_Complete": True,
        "Main_Discharge_Mean_Current_A": -1.1,
        "Discharge_Start_Voltage_V": 4.1,
        "Discharge_Cutoff_Voltage_V": 2.7,
        "Discharge_Counter_Decrease_Count": 0,
        "Capacity_Agreement_Delta_Ah": 0.0,
    }


class CS235PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        workbook = WORKSPACE_DIR / "CS2_35" / BENCHMARK_FILE
        _manifest, rows = extract_workbook(workbook)
        for row in rows:
            row["File_Role"] = "CANONICAL"
            row["Duplicate_Of"] = ""
            row["Boundary_Status"] = "CONFIRMED"
            row["Global_Cycle_Index"] = row["Local_Cycle_Index"]
        cls.frame = finalize_qc(rows)

    def test_cycle_2_capacity_matches_human_benchmark(self) -> None:
        cycle = self.frame[self.frame["Local_Cycle_Index"].eq(2)].iloc[0]
        statistics_capacity = float(cycle["Statistics_Single_Cycle_Discharge_Capacity_Ah"])
        channel_capacity = float(cycle["Channel_Single_Cycle_Discharge_Capacity_Ah"])
        self.assertLessEqual(
            abs(statistics_capacity - BENCHMARK_CAPACITY_AH), BENCHMARK_TOLERANCE_AH
        )
        self.assertLessEqual(
            abs(channel_capacity - BENCHMARK_CAPACITY_AH), BENCHMARK_TOLERANCE_AH
        )

    def test_cycle_9_is_flagged_for_incomplete_cv_protocol(self) -> None:
        cycle = self.frame[self.frame["Local_Cycle_Index"].eq(9)].iloc[0]
        self.assertFalse(bool(cycle["CV_Stage_Present"]))
        self.assertEqual(cycle["QC_Flag"], "INVALID_CANDIDATE")
        self.assertIn("incomplete CV charging protocol", cycle["QC_Reason"])
        self.assertNotEqual(cycle["QC_Reason"], "low capacity")

    def test_neighbor_cycles_have_complete_cv_protocol(self) -> None:
        for cycle_index in (8, 10):
            cycle = self.frame[self.frame["Local_Cycle_Index"].eq(cycle_index)].iloc[0]
            self.assertTrue(bool(cycle["CV_Stage_Present"]))
            self.assertTrue(bool(cycle["CV_Protocol_Complete"]))

    def test_duplicate_mapping_is_explicit(self) -> None:
        self.assertEqual(
            DUPLICATE_FILES["CS2_35_1_10_11(1).xlsx"]["duplicate_of"],
            "CS2_35_1_10_11.xlsx",
        )
        self.assertEqual(
            DUPLICATE_FILES["CS2_35_2_10_11.xlsx"]["duplicate_of"],
            "CS2_35_2_4_11.xlsx",
        )


class CS236PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = get_profile("CS2-36")
        workbook = cls.profile.input_dir / cls.profile.benchmark_file
        _manifest, rows = extract_workbook(workbook, cell_id=cls.profile.cell_id)
        for row in rows:
            row["File_Role"] = "CANONICAL"
            row["Duplicate_Of"] = ""
            row["Boundary_Status"] = "CONFIRMED"
            row["Global_Cycle_Index"] = row["Local_Cycle_Index"]
        cls.frame = finalize_qc(
            rows,
            known_boundary_tails=cls.profile.known_boundary_tails,
        )

    def test_cycle_2_cross_source_benchmark(self) -> None:
        cycle = self.frame[self.frame["Local_Cycle_Index"].eq(2)].iloc[0]
        statistics_capacity = float(cycle["Statistics_Single_Cycle_Discharge_Capacity_Ah"])
        channel_capacity = float(cycle["Channel_Single_Cycle_Discharge_Capacity_Ah"])
        self.assertLessEqual(
            abs(statistics_capacity - self.profile.benchmark_capacity_ah),
            self.profile.benchmark_tolerance_ah,
        )
        self.assertLessEqual(
            abs(channel_capacity - self.profile.benchmark_capacity_ah),
            self.profile.benchmark_tolerance_ah,
        )

    def test_boundary_rules_are_explicit_and_no_duplicates_are_assumed(self) -> None:
        self.assertEqual(len(self.profile.known_boundary_tails), 6)
        self.assertIn(("CS2_36_9_7_10.xlsx", 44), self.profile.known_boundary_tails)
        self.assertIn(("CS2_36_12_23_10.xlsx", 28), self.profile.known_boundary_tails)
        self.assertEqual(dict(self.profile.duplicate_files), {})


class CS237PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = get_profile("CS2-37")
        workbook = cls.profile.input_dir / cls.profile.benchmark_file
        _manifest, rows = extract_workbook(workbook, cell_id=cls.profile.cell_id)
        for row in rows:
            row["File_Role"] = "CANONICAL"
            row["Duplicate_Of"] = ""
            row["Boundary_Status"] = "CONFIRMED"
            row["Global_Cycle_Index"] = row["Local_Cycle_Index"]
        cls.frame = finalize_qc(
            rows,
            known_boundary_tails=cls.profile.known_boundary_tails,
        )

    def test_cycle_2_cross_source_benchmark(self) -> None:
        cycle = self.frame[self.frame["Local_Cycle_Index"].eq(2)].iloc[0]
        statistics_capacity = float(cycle["Statistics_Single_Cycle_Discharge_Capacity_Ah"])
        channel_capacity = float(cycle["Channel_Single_Cycle_Discharge_Capacity_Ah"])
        self.assertLessEqual(
            abs(statistics_capacity - self.profile.benchmark_capacity_ah),
            self.profile.benchmark_tolerance_ah,
        )
        self.assertLessEqual(
            abs(channel_capacity - self.profile.benchmark_capacity_ah),
            self.profile.benchmark_tolerance_ah,
        )

    def test_boundary_rules_are_explicit_and_no_duplicates_are_assumed(self) -> None:
        self.assertEqual(len(self.profile.known_boundary_tails), 7)
        self.assertIn(("CS2_37_9_7_10.xlsx", 45), self.profile.known_boundary_tails)
        self.assertIn(("CS2_37_1_28_11.xlsx", 49), self.profile.known_boundary_tails)
        self.assertEqual(dict(self.profile.duplicate_files), {})


class CS238PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = get_profile("CS2-38")
        workbook = cls.profile.input_dir / cls.profile.benchmark_file
        _manifest, rows = extract_workbook(workbook, cell_id=cls.profile.cell_id)
        for row in rows:
            row["File_Role"] = "CANONICAL"
            row["Duplicate_Of"] = ""
            row["Boundary_Status"] = "CONFIRMED"
            row["Global_Cycle_Index"] = row["Local_Cycle_Index"]
        cls.frame = finalize_qc(
            rows,
            known_boundary_tails=cls.profile.known_boundary_tails,
        )

    def test_cycle_2_cross_source_benchmark(self) -> None:
        cycle = self.frame[self.frame["Local_Cycle_Index"].eq(2)].iloc[0]
        statistics_capacity = float(cycle["Statistics_Single_Cycle_Discharge_Capacity_Ah"])
        channel_capacity = float(cycle["Channel_Single_Cycle_Discharge_Capacity_Ah"])
        self.assertLessEqual(
            abs(statistics_capacity - self.profile.benchmark_capacity_ah),
            self.profile.benchmark_tolerance_ah,
        )
        self.assertLessEqual(
            abs(channel_capacity - self.profile.benchmark_capacity_ah),
            self.profile.benchmark_tolerance_ah,
        )

    def test_boundary_and_duplicate_rules_are_explicit(self) -> None:
        self.assertEqual(len(self.profile.known_boundary_tails), 7)
        self.assertIn(("CS2_38_9_7_10.xlsx", 43), self.profile.known_boundary_tails)
        self.assertIn(("CS2_38_1_28_11.xlsx", 42), self.profile.known_boundary_tails)
        self.assertEqual(
            self.profile.duplicate_files["CS2_38_2_10_11.xlsx"]["duplicate_of"],
            "CS2_38_2_4_11.xlsx",
        )


class CapacityJumpRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cs235 = build_full_profile_frame("CS2-35")
        cls.cs236 = build_full_profile_frame("CS2-36")

    @staticmethod
    def global_cycle(frame, cycle_index: int):
        return frame[frame["Global_Cycle_Index"].eq(cycle_index)].iloc[0]

    def test_cs235_global_516_remains_invalid_candidate(self) -> None:
        cycle = self.global_cycle(self.cs235, 516)
        self.assertAlmostEqual(
            float(cycle["Single_Cycle_Discharge_Capacity_Ah"]),
            0.809483616105,
            places=9,
        )
        self.assertEqual(cycle["QC_Flag"], "INVALID_CANDIDATE")
        self.assertIn("INCOMPLETE_CV_CHARGING_PROTOCOL", cycle["QC_Reason_Codes"])

    def test_cs235_global_517_skips_invalid_516_as_reference(self) -> None:
        cycle = self.global_cycle(self.cs235, 517)
        self.assertFalse(bool(cycle["Capacity_Jump_Flag"]))
        self.assertNotIn("CAPACITY_JUMP", cycle["QC_Reason_Codes"])
        self.assertEqual(int(cycle["Jump_Reference_Global_Cycle"]), 515)
        self.assertAlmostEqual(float(cycle["Jump_Delta_Ah"]), -0.00052793245, places=9)

    def test_cs236_global_738_remains_invalid_candidate(self) -> None:
        cycle = self.global_cycle(self.cs236, 738)
        self.assertAlmostEqual(
            float(cycle["Single_Cycle_Discharge_Capacity_Ah"]),
            0.507458360383,
            places=9,
        )
        self.assertEqual(cycle["QC_Flag"], "INVALID_CANDIDATE")
        self.assertIn("INCOMPLETE_CV_CHARGING_PROTOCOL", cycle["QC_Reason_Codes"])

    def test_cs236_global_739_skips_invalid_738_as_reference(self) -> None:
        cycle = self.global_cycle(self.cs236, 739)
        self.assertFalse(bool(cycle["Capacity_Jump_Flag"]))
        self.assertNotIn("CAPACITY_JUMP", cycle["QC_Reason_Codes"])
        self.assertEqual(int(cycle["Jump_Reference_Global_Cycle"]), 737)
        self.assertAlmostEqual(float(cycle["Jump_Delta_Ah"]), -0.006497427167, places=9)

    def test_valid_to_valid_true_jump_is_still_check(self) -> None:
        frame = finalize_qc(
            [
                complete_synthetic_cycle(1, 1.0),
                complete_synthetic_cycle(2, 0.8),
            ],
            known_boundary_tails={},
        )
        first = self.global_cycle(frame, 1)
        second = self.global_cycle(frame, 2)
        self.assertEqual(first["Jump_Reference_Status"], "NO_VALID_JUMP_REFERENCE")
        self.assertIn("no earlier canonical cycle", first["Jump_Reference_Reason"])
        self.assertTrue(bool(second["Capacity_Jump_Flag"]))
        self.assertEqual(second["QC_Flag"], "CHECK")
        self.assertEqual(second["QC_Reason_Codes"], "CAPACITY_JUMP")
        self.assertEqual(int(second["Jump_Reference_Global_Cycle"]), 1)
        self.assertAlmostEqual(float(second["Jump_Reference_Capacity_Ah"]), 1.0)
        self.assertAlmostEqual(float(second["Jump_Delta_Ah"]), -0.2)


if __name__ == "__main__":
    unittest.main()
