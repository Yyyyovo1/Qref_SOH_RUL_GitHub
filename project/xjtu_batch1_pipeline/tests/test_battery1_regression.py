from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xjtu_batch1.config import load_config  # noqa: E402
from xjtu_batch1.mat_io import InventoryError, scan_inventory, validate_schema  # noqa: E402
from xjtu_batch1.qc import process_battery  # noqa: E402
from xjtu_batch1.regression import run_battery1_regressions  # noqa: E402
from xjtu_batch1.sensitivity import compute_eol_sensitivity, compute_qref_summary  # noqa: E402


class Battery1RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(PROJECT_ROOT / "xjtu_batch1_config.json")
        cls.input_dir = PROJECT_ROOT.parent / "Batch-1"
        cls.inventory = scan_inventory(cls.input_dir, cls.config)
        cls.schema = validate_schema(
            cls.inventory.canonical_by_id[1], "2C_battery-1"
        )
        cls.qc = process_battery(cls.inventory.canonical_by_id[1], 1, cls.config)
        cls.qref = compute_qref_summary(cls.qc, cls.config)
        cls.eol = compute_eol_sensitivity(cls.qc, cls.qref, cls.config)

    def test_authorized_inventory_roles_and_hashes(self) -> None:
        canonical = [row for row in self.inventory.records if row.file_role == "CANONICAL"]
        duplicates = [
            row for row in self.inventory.records if row.file_role == "DUPLICATE_CONFIRMED"
        ]
        self.assertEqual(len(canonical), 8)
        self.assertEqual(len(duplicates), 8)
        self.assertEqual(set(self.inventory.canonical_by_id), set(range(1, 9)))
        self.assertTrue(all(row.included_in_analysis == "YES" for row in canonical))
        self.assertTrue(all(row.included_in_analysis == "NO" for row in duplicates))
        self.assertTrue(all(row.hash_match is True for row in duplicates))

    def test_duplicate_content_mismatch_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for battery_id in range(1, 9):
                canonical = root / f"2C_battery-{battery_id}.mat"
                duplicate = root / f"2C_battery-{battery_id}(1).mat"
                canonical.write_bytes(f"canonical-{battery_id}".encode())
                duplicate.write_bytes(f"canonical-{battery_id}".encode())
            (root / "2C_battery-4(1).mat").write_bytes(b"different-content")
            with self.assertRaisesRegex(InventoryError, "DUPLICATE_CONTENT_MISMATCH"):
                scan_inventory(root, self.config)

    def test_schema_cycle_count(self) -> None:
        self.assertEqual(self.schema.schema_status, "PASS")
        self.assertEqual(self.schema.data_cycle_count, 390)
        self.assertEqual(self.schema.summary_cycle_life, 390)

    def test_capacity_baselines(self) -> None:
        indexed = self.qc.set_index("Raw_Cycle_Index")
        tolerance = float(self.config["battery1_capacity_regression_tolerance_Ah"])
        for raw_text, expected in self.config["battery1_expected_capacities_Ah"].items():
            actual = float(indexed.loc[int(raw_text), "Summary_Discharge_Capacity_Ah"])
            self.assertLessEqual(abs(actual - float(expected)), tolerance)

    def test_protocol_baselines(self) -> None:
        indexed = self.qc.set_index("Raw_Cycle_Index")
        cycle1 = indexed.loc[1]
        cycle2 = indexed.loc[2]
        self.assertIn("0.5C charge and 0.2C discharge", cycle1["Description"])
        self.assertIn("2C charge and 1C discharge", cycle2["Description"])
        self.assertAlmostEqual(float(cycle1["Discharge_Median_Current_A"]), -0.4, places=2)
        self.assertAlmostEqual(float(cycle2["Discharge_Median_Current_A"]), -2.0, places=2)
        self.assertAlmostEqual(float(cycle1["Discharge_Min_V"]), 2.5, places=2)
        self.assertAlmostEqual(float(cycle2["Discharge_Min_V"]), 2.5, places=2)

    def test_cycle390_is_invalid_protocol_anomaly(self) -> None:
        cycle390 = self.qc.loc[self.qc["Raw_Cycle_Index"] == 390].iloc[0]
        codes = set(str(cycle390["Auto_QC_Reason_Codes"]).split("|"))
        self.assertEqual(cycle390["Auto_QC_Flag"], "INVALID_CANDIDATE")
        self.assertIn("INCOMPLETE_DISCHARGE_PROTOCOL", codes)
        self.assertTrue(
            "VOLTAGE_SPIKE_DURING_DISCHARGE" in codes
            or "ABNORMAL_VOLTAGE_JUMP" in codes
        )
        self.assertAlmostEqual(float(cycle390["Discharge_Min_V"]), 2.8445, places=3)

    def test_qref_baselines(self) -> None:
        row = self.qref.iloc[0]
        tolerance = float(self.config["battery1_capacity_regression_tolerance_Ah"])
        expected = self.config["battery1_qref_expected_Ah"]
        actual = {
            "nominal": row["Qref_nominal_Ah"],
            "initial_test": row["Qref_initial_test_Ah"],
            "first_standard": row["Qref_first_standard_Ah"],
            "early_median5": row["Qref_early_median5_Ah"],
        }
        for definition, value in actual.items():
            self.assertTrue(np.isfinite(value))
            self.assertLessEqual(abs(float(value) - float(expected[definition])), tolerance)
        self.assertEqual(row["Qref_Early_Median5_Source_Cycles"], "2|3|4|5|6")

    def test_cycle390_does_not_trigger_eol(self) -> None:
        relative = self.eol.loc[
            self.eol["Qref_Definition"].isin(
                ["nominal", "initial_test", "first_standard", "early_median5"]
            )
        ]
        self.assertTrue((relative["EOL_Status"] == "RIGHT_CENSORED").all())
        self.assertTrue(relative["EOL_Raw_Cycle"].isna().all())
        self.assertTrue((relative["Last_Valid_Raw_Cycle"] == 389).all())

    def test_combined_regression_suite(self) -> None:
        results = run_battery1_regressions(
            self.schema, self.qc, self.qref, self.eol, self.config
        )
        self.assertEqual(len(results), 6)
        self.assertTrue(all(value == "PASSED" for value in results.values()))


if __name__ == "__main__":
    unittest.main()

