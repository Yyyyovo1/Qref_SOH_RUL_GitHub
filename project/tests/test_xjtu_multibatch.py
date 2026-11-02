from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from xjtu_multibatch.analysis import (
    _pairwise,
    _qref_row_from_observations,
    _validate_approval,
    detect_eol,
    rul_from_eol,
)
from xjtu_multibatch.config_templates import make_batch_config
from xjtu_multibatch.final_v2 import (
    EXPECTED_BATCH5_MANUAL_KEYS,
    MANUAL_REASON,
    _apply_batch5_manual_keep,
)
from xjtu_multibatch.inventory import scan_inventory
from xjtu_multibatch.protocols import Segment, classify_protocol
from xjtu_multibatch.scan import output_hash_manifest
from xjtu_multibatch.validation import resolve_known_protocol_information


def segment(current: float, minimum: float, end: float, maximum: float = 4.2) -> Segment:
    return Segment(
        start_index=0,
        end_index=99,
        record_count=100,
        median_current_A=current,
        mean_current_A=current,
        start_voltage_V=4.2 if current < 0 else 3.0,
        end_voltage_V=end,
        min_voltage_V=minimum,
        max_voltage_V=maximum,
        duration_s=1000.0,
        capacity_delta_Ah=1.8,
    )


class ProtocolClassificationTests(unittest.TestCase):
    def classify(
        self,
        batch: int,
        raw_cycle: int,
        description: str,
        charge_current: float,
        discharge_current: float,
        minimum: float,
        full: bool,
        charge_end: float = 0.1,
    ) -> dict[str, object]:
        return classify_protocol(
            batch,
            raw_cycle,
            description,
            segment(charge_current, 3.0, 4.2),
            charge_end,
            segment(discharge_current, minimum, minimum, 4.2),
            full,
            make_batch_config(batch),
        )

    def test_batch3_1c_is_primary_without_modulo_rule(self) -> None:
        result = self.classify(3, 17, "2C charge and 1C discharge", 4.0, -2.0, 2.5, True)
        self.assertEqual(result["Protocol_Class"], "VARIABLE_FULL_DISCHARGE_1C")
        self.assertEqual(result["Primary_Capacity_Eligible"], "YES")

    def test_batch4_partial_is_not_primary(self) -> None:
        result = self.classify(4, 22, "2C charge and 1C discharge to 3V", 4.0, -2.0, 3.0, False)
        self.assertEqual(result["Protocol_Class"], "AGING_PARTIAL_DISCHARGE")
        self.assertEqual(result["Primary_Capacity_Eligible"], "NO")

    def test_batch5_random_and_diagnostic_are_distinct(self) -> None:
        random = self.classify(5, 12, "0.5C charge and random discharge", 1.0, -5.0, 3.0, False, 0.05)
        diagnostic = self.classify(5, 21, "1C charge and 1C discharge [test capacity]", 2.0, -2.0, 2.5, True)
        self.assertEqual(random["Protocol_Class"], "RANDOM_WALK_AGING")
        self.assertEqual(random["Primary_Capacity_Eligible"], "NO")
        self.assertEqual(diagnostic["Protocol_Class"], "CAPACITY_DIAGNOSTIC_1C")
        self.assertEqual(diagnostic["Primary_Capacity_Eligible"], "YES")

    def test_batch6_geo_and_diagnostic_are_distinct(self) -> None:
        geo = self.classify(6, 2, "2.0C charge and 0.67C discharge", 4.0, -1.334, 3.5, False)
        diagnostic = self.classify(6, 7, "1.0C charge and 0.5C discharge [test capacity]", 2.0, -1.0, 2.5, True)
        self.assertEqual(geo["Protocol_Class"], "GEO_AGING_PARTIAL")
        self.assertEqual(geo["Primary_Capacity_Eligible"], "NO")
        self.assertEqual(diagnostic["Protocol_Class"], "CAPACITY_DIAGNOSTIC_0P5C")
        self.assertEqual(diagnostic["Primary_Capacity_Eligible"], "YES")

    def test_batch6_high_soc_geo_charge_is_info_not_review(self) -> None:
        result = self.classify(
            6,
            419,
            "2.0C charge and 0.67C discharge",
            3.30,
            -1.334,
            3.82,
            False,
            0.10,
        )
        self.assertEqual(result["Protocol_Class"], "GEO_AGING_PARTIAL")
        self.assertEqual(result["Primary_Capacity_Eligible"], "NO")
        self.assertEqual(result["Expected_Protocol_Match"], "YES")
        self.assertIn(
            "CC_PLATEAU_NOT_OBSERVED_HIGH_SOC_START",
            result["Classification_Issues"],
        )
        self.assertNotIn(
            "CHARGE_CURRENT_PROTOCOL_DEVIATION",
            result["Classification_Issues"],
        )


class CensoringTests(unittest.TestCase):
    @staticmethod
    def observations(ages: list[int], capacities: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Raw_Cycle_Index": np.arange(2, 2 + len(ages)),
                "Diagnostic_Index": np.arange(1, 1 + len(ages)),
                "Cumulative_Aging_Operation_Count": ages,
                "Capacity_Ah": capacities,
            }
        )

    def test_dense_crossing_is_exact(self) -> None:
        result = detect_eol(self.observations([1, 2], [1.7, 1.5]), 1.6, dense_measurement=True)
        self.assertEqual(result.status, "EXACT_OBSERVED")
        self.assertEqual(result.exact_aging_cycle, 2.0)

    def test_sparse_crossing_is_interval_censored(self) -> None:
        result = detect_eol(self.observations([100, 110], [1.7, 1.5]), 1.6, dense_measurement=False)
        self.assertEqual(result.status, "INTERVAL_CENSORED")
        self.assertEqual(result.lower_bound_aging_cycle, 100.0)
        self.assertEqual(result.upper_bound_aging_cycle, 110.0)
        rul = rul_from_eol(90.0, result)
        self.assertTrue(np.isnan(rul["RUL_Cycles"]))
        self.assertEqual(rul["RUL_Lower_Bound"], 10.0)
        self.assertEqual(rul["RUL_Upper_Bound"], 20.0)

    def test_right_censored_has_no_point_rul(self) -> None:
        result = detect_eol(self.observations([1, 2], [1.8, 1.7]), 1.6, dense_measurement=True)
        self.assertEqual(result.status, "RIGHT_CENSORED")
        self.assertTrue(np.isnan(rul_from_eol(1.0, result)["RUL_Cycles"]))

    def test_pairwise_keeps_interval_bounds_and_unavailable_pairs(self) -> None:
        qref = pd.DataFrame(
            [
                {
                    "Batch_ID": 6,
                    "Cell_ID": "cell-1",
                    "Qref_nominal_Ah": 2.0,
                    "Qref_initial_test_Ah": np.nan,
                    "Qref_first_comparable_Ah": 1.9,
                    "Qref_early_median5_comparable_Ah": 1.88,
                }
            ]
        )
        eol = pd.DataFrame(
            [
                {
                    "Batch_ID": 6,
                    "Cell_ID": "cell-1",
                    "Qref_Definition": "nominal",
                    "EOL_Status": "INTERVAL_CENSORED",
                    "EOL_Exact_Aging_Cycle": np.nan,
                    "EOL_Lower_Bound_Aging_Cycle": 100,
                    "EOL_Upper_Bound_Aging_Cycle": 110,
                },
                {
                    "Batch_ID": 6,
                    "Cell_ID": "cell-1",
                    "Qref_Definition": "first_comparable",
                    "EOL_Status": "EXACT_OBSERVED",
                    "EOL_Exact_Aging_Cycle": 120,
                    "EOL_Lower_Bound_Aging_Cycle": 120,
                    "EOL_Upper_Bound_Aging_Cycle": 120,
                },
                {
                    "Batch_ID": 6,
                    "Cell_ID": "cell-1",
                    "Qref_Definition": "early_median5_comparable",
                    "EOL_Status": "RIGHT_CENSORED",
                    "EOL_Exact_Aging_Cycle": np.nan,
                    "EOL_Lower_Bound_Aging_Cycle": 130,
                    "EOL_Upper_Bound_Aging_Cycle": np.nan,
                },
            ]
        )
        result = _pairwise(eol, qref)
        self.assertEqual(len(result), 6)
        nominal_first = result.loc[
            result["Definition_A"].eq("nominal")
            & result["Definition_B"].eq("first_comparable")
        ].iloc[0]
        self.assertEqual(
            nominal_first["Censoring_Status_Change"], "INTERVAL_TO_EXACT"
        )
        self.assertTrue(np.isnan(nominal_first["Delta_EOL_Cycles"]))
        self.assertEqual(nominal_first["Delta_EOL_Lower_Bound_Cycles"], 10)
        self.assertEqual(nominal_first["Delta_EOL_Upper_Bound_Cycles"], 20)
        unavailable = result.loc[
            result["Definition_A"].eq("nominal")
            & result["Definition_B"].eq("initial_test")
        ].iloc[0]
        self.assertEqual(unavailable["EOL_Status_B"], "QREF_UNAVAILABLE")


class IntegrityTests(unittest.TestCase):
    @staticmethod
    def frozen_reference(directory: Path) -> None:
        pd.DataFrame(
            [
                {
                    "Cell_ID": "2C_battery-1",
                    "Source_File": "2C_battery-1.mat",
                    "File_Size_Bytes": 1,
                    "File_Role": "CANONICAL",
                    "Duplicate_Of": "",
                    "SHA256": "x",
                    "Duplicate_SHA256": "",
                    "Hash_Match": True,
                    "Included_In_Analysis": "YES",
                }
            ]
        ).to_csv(directory / "xjtu_file_manifest.csv", index=False)
        pd.DataFrame([{"Cell_ID": "2C_battery-1"}]).to_csv(
            directory / "xjtu_cycle_capacity_final.csv", index=False
        )

    def test_duplicate_match_and_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root / "frozen"
            frozen.mkdir()
            self.frozen_reference(frozen)
            for batch in range(2, 7):
                (root / f"Batch-{batch}").mkdir()
            canonical = root / "Batch-2" / "3C_battery-1.mat"
            duplicate = root / "Batch-2" / "3C_battery-1(1).mat"
            canonical.write_bytes(b"identical")
            duplicate.write_bytes(b"identical")
            matched = scan_inventory(root, frozen)
            self.assertFalse(matched.hard_stop)
            self.assertEqual(matched.duplicates.iloc[0]["File_Role"], "DUPLICATE_CONFIRMED")
            duplicate.write_bytes(b"different")
            mismatched = scan_inventory(root, frozen)
            self.assertTrue(mismatched.hard_stop)
            self.assertIn(
                "DUPLICATE_CONTENT_MISMATCH", "|".join(mismatched.issues)
            )

    def test_approval_hash_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scan_dir = Path(temporary)
            names = [
                "xjtu_protocol_inventory.csv",
                *[f"batch{batch}_config_draft.json" for batch in range(2, 7)],
            ]
            for name in names:
                (scan_dir / name).write_text(name, encoding="utf-8")
            hashes = output_hash_manifest(scan_dir)
            approval = {
                "status": "APPROVED",
                "approved_by": "unit-test",
                "approved_batches": [2, 3, 4, 5, 6],
                "acknowledge_all_scan_anomalies": True,
                "file_hashes": hashes,
            }
            self.assertEqual(_validate_approval(scan_dir, approval), hashes)
            approval["file_hashes"]["batch2_config_draft.json"] = "bad"
            with self.assertRaisesRegex(RuntimeError, "HASH_MISMATCH"):
                _validate_approval(scan_dir, approval)

    def test_batch5_does_not_fallback_initial_qref(self) -> None:
        rows = pd.DataFrame(
            {
                "Batch_ID": [5] * 5,
                "Cell_ID": ["B5_RW_battery-1"] * 5,
                "Raw_Cycle_Index": [21, 32, 43, 54, 65],
                "Protocol_Class": ["CAPACITY_DIAGNOSTIC_1C"] * 5,
                "Primary_Capacity_Eligible": ["YES"] * 5,
                "Final_Include": ["YES"] * 5,
                "Diagnostic_Index": [1, 2, 3, 4, 5],
                "Cumulative_Aging_Operation_Count": [20, 30, 40, 50, 60],
                "Capacity_Ah": [1.9, 1.89, 1.88, 1.87, 1.86],
            }
        )
        result = _qref_row_from_observations("B5_RW_battery-1", 5, rows)
        self.assertTrue(np.isnan(result["Qref_initial_test_Ah"]))
        self.assertEqual(result["Qref_First_Comparable_Source_Cycle"], 21)

    def test_early_median5_uses_numeric_not_lexical_diagnostic_order(self) -> None:
        rows = pd.DataFrame(
            {
                "Batch_ID": [2] * 8,
                "Cell_ID": ["B2_test"] * 8,
                "Raw_Cycle_Index": ["2", "3", "4", "5", "6", "11", "101", "102"],
                "Protocol_Class": ["STANDARD_FULL_DISCHARGE_1C"] * 8,
                "Primary_Capacity_Eligible": ["YES"] * 8,
                "Final_Include": ["YES"] * 8,
                "Diagnostic_Index": ["1", "2", "3", "4", "5", "10", "100", "101"],
                "Cumulative_Aging_Operation_Count": ["1", "2", "3", "4", "5", "10", "100", "101"],
                "Capacity_Ah": [1.81, 1.82, 1.83, 1.84, 1.85, 1.50, 1.40, 1.30],
            }
        )
        result = _qref_row_from_observations("B2_test", 2, rows)
        self.assertEqual(
            result["Qref_Early_Median5_Comparable_Source_Cycles"],
            "2|3|4|5|6",
        )
        self.assertAlmostEqual(
            result["Qref_early_median5_comparable_Ah"], 1.83, places=12
        )

    def test_early_median5_skips_excluded_numeric_diagnostic(self) -> None:
        rows = pd.DataFrame(
            {
                "Batch_ID": [5] * 6,
                "Cell_ID": ["B5_test"] * 6,
                "Raw_Cycle_Index": ["21", "32", "43", "54", "65", "76"],
                "Protocol_Class": ["CAPACITY_DIAGNOSTIC_1C"] * 6,
                "Primary_Capacity_Eligible": ["YES"] * 6,
                "Final_Include": ["YES", "YES", "NO", "YES", "YES", "YES"],
                "Diagnostic_Index": ["1", "2", "3", "4", "5", "6"],
                "Cumulative_Aging_Operation_Count": ["20", "30", "40", "50", "60", "70"],
                "Capacity_Ah": [1.90, 1.89, 0.50, 1.88, 1.87, 1.86],
            }
        )
        result = _qref_row_from_observations("B5_test", 5, rows)
        self.assertEqual(
            result["Qref_Early_Median5_Comparable_Source_Cycles"],
            "21|32|54|65|76",
        )

    def test_batch5_manual_keep_preserves_automatic_qc(self) -> None:
        rows = []
        for cell, raw_cycle in sorted(EXPECTED_BATCH5_MANUAL_KEYS):
            rows.append(
                {
                    "Cell_ID": cell,
                    "Raw_Cycle_Index": raw_cycle,
                    "Auto_QC_Flag": "CHECK",
                    "Auto_QC_Reason_Codes": "CAPACITY_JUMP",
                    "Primary_Capacity_Eligible": "YES",
                    "Manual_Review_Status": "PENDING",
                    "Manual_Decision": "",
                    "Manual_Reason": "",
                    "Final_Include": "REVIEW",
                    "Final_Include_Basis": "AUTO_CHECK_REQUIRES_REVIEW",
                }
            )
        source = pd.DataFrame(rows)
        result, original = _apply_batch5_manual_keep(source)
        self.assertTrue(result["Auto_QC_Flag"].eq("CHECK").all())
        self.assertTrue(
            result["Auto_QC_Reason_Codes"].eq("CAPACITY_JUMP").all()
        )
        self.assertTrue(result["Manual_Decision"].eq("KEEP").all())
        self.assertTrue(result["Manual_Reason"].eq(MANUAL_REASON).all())
        self.assertTrue(result["Final_Include"].eq("YES").all())
        self.assertEqual(set(original.values()), {"REVIEW"})

    def test_known_metadata_mismatch_resolves_to_info_only(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "Batch_ID": 5,
                    "Cell_ID": "B5_RW_battery-1",
                    "Source_File": "RW_battery-1.mat",
                    "Raw_Cycle_Index": 1,
                    "Protocol_Class": "RANDOM_WALK_AGING",
                    "Primary_Capacity_Eligible": "NO",
                    "Expected_Protocol_Match": "REVIEW",
                    "Issues": "DESCRIPTION_CHARGE_CURRENT_MISMATCH",
                    "Observed_Charge_Max_V": 4.2,
                    "Observed_Charge_End_Current_A": 0.05,
                }
            ]
        )
        resolved = resolve_known_protocol_information(
            frame, {5: make_batch_config(5), 6: make_batch_config(6)}
        ).iloc[0]
        self.assertEqual(resolved["Validated_Expected_Protocol_Match"], "YES")
        self.assertEqual(resolved["Validation_Severity"], "INFO")
        self.assertEqual(resolved["Validation_Requires_Human_Review"], "NO")
        self.assertEqual(resolved["Validated_Primary_Capacity_Eligible"], "NO")


if __name__ == "__main__":
    unittest.main()
