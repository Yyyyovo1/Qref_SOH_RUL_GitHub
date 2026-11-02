from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hust_repro.config import DEFAULT_77CELL_FULL_CONFIG
from hust_repro.full_run import (
    _augment_cycle_inventory,
    _series_equal_csv12,
    add_pairwise_lower_bound,
    build_protocol_discrepancy_audit,
)
from hust_repro.pilot import EOL_EXACT, EOL_RIGHT_CENSORED, build_eol_pairwise


class PairwiseLowerBoundTests(unittest.TestCase):
    def test_frozen_csv_comparison_uses_delivery_precision(self) -> None:
        left = pd.Series([0.8800000000000001, 0.9272085333360001])
        right = pd.Series([0.88, 0.927208533336])
        self.assertTrue(_series_equal_csv12(left, right))

    def test_right_censored_nominal_to_exact_measured_has_lower_bound(self) -> None:
        eol = pd.DataFrame(
            [
                {
                    "Cell_Selection_Order": 1,
                    "Cell_ID": "HUST_TEST",
                    "Qref_Definition": "nominal",
                    "Qref_Ah": 1.1,
                    "EOThreshold_Ah": 0.88,
                    "EOL_Status": EOL_RIGHT_CENSORED,
                    "EOL_Raw_Cycle": np.nan,
                    "Last_Valid_Raw_Cycle": 100,
                },
                {
                    "Cell_Selection_Order": 1,
                    "Cell_ID": "HUST_TEST",
                    "Qref_Definition": "first_comparable",
                    "Qref_Ah": 1.2,
                    "EOThreshold_Ah": 0.96,
                    "EOL_Status": EOL_EXACT,
                    "EOL_Raw_Cycle": 80,
                    "Last_Valid_Raw_Cycle": 100,
                },
                {
                    "Cell_Selection_Order": 1,
                    "Cell_ID": "HUST_TEST",
                    "Qref_Definition": "early_median5_comparable",
                    "Qref_Ah": 1.19,
                    "EOThreshold_Ah": 0.952,
                    "EOL_Status": EOL_EXACT,
                    "EOL_Raw_Cycle": 75,
                    "Last_Valid_Raw_Cycle": 100,
                },
            ]
        )
        pairwise = build_eol_pairwise(
            eol,
            [
                ["nominal", "first_comparable"],
                ["nominal", "early_median5_comparable"],
                ["first_comparable", "early_median5_comparable"],
            ],
            "EOL_B_MINUS_EOL_A",
        )
        result = add_pairwise_lower_bound(pairwise, eol, "LOWER_BOUND_ONLY")
        nominal_first = result.iloc[0]
        nominal_median = result.iloc[1]
        measured_pair = result.iloc[2]
        self.assertTrue(np.isnan(nominal_first["Delta_EOL_Cycles"]))
        self.assertEqual(
            nominal_first["Minimum_Observed_Earlier_By_Cycles"], 20
        )
        self.assertEqual(
            nominal_median["Minimum_Observed_Earlier_By_Cycles"], 25
        )
        self.assertTrue(
            np.isnan(measured_pair["Minimum_Observed_Earlier_By_Cycles"])
        )
        self.assertEqual(measured_pair["Delta_EOL_Cycles"], -5)


class ProtocolAuditTests(unittest.TestCase):
    def test_validated_two_c_protocol_passes_and_one_c_requires_review(self) -> None:
        fingerprint = pd.DataFrame(
            {
                "Cell_ID": ["HUST_OK", "HUST_REVIEW"],
                "Discharge_Step4_Median_Current_A": [-2.2, -1.1],
                "Estimated_C_Rate_4": [2.0, 1.0],
                "Protocol_Fingerprint_ID": ["FP_OK", "FP_REVIEW"],
                "Protocol_Fingerprint_Status": [
                    "STABLE_SCAN_CANDIDATE",
                    "STABLE_SCAN_CANDIDATE",
                ],
            }
        )
        cells = pd.DataFrame(
            {
                "Cell_ID": ["HUST_OK", "HUST_REVIEW"],
                "Native_Cell_ID": ["OK", "REVIEW"],
                "Source_File": ["ok.pkl", "review.pkl"],
                "Protocol_Change_Candidate": ["NO", "NO"],
            }
        )
        result = build_protocol_discrepancy_audit(
            fingerprint,
            cells,
            ["HUST_OK", "HUST_REVIEW"],
            DEFAULT_77CELL_FULL_CONFIG,
        ).set_index("Cell_ID")
        self.assertEqual(result.loc["HUST_OK", "Protocol_Review_Required"], "NO")
        self.assertEqual(
            result.loc["HUST_REVIEW", "Protocol_Review_Required"], "YES"
        )
        self.assertTrue(
            result["Paper_vs_Raw_Protocol_Discrepancy"].eq("YES").all()
        )
        self.assertTrue(result["Protocol_QC_Basis"].eq("RAW_OBSERVED_PROTOCOL").all())

    def test_capacity_jump_check_is_not_excluded_when_protocol_is_consistent(self) -> None:
        cycles = pd.DataFrame(
            {
                "Cell_ID": ["HUST_OK"],
                "Auto_QC_Flag": ["CHECK"],
                "Auto_QC_Reason_Codes": ["CAPACITY_JUMP"],
                "Capacity_Jump_Flag": ["YES"],
                "Primary_Capacity_Eligible": ["YES"],
            }
        )
        protocol = pd.DataFrame(
            {
                "Cell_ID": ["HUST_OK"],
                "Raw_Observed_Final_Stage_CRate_Approx": [2.0],
                "Paper_vs_Raw_Protocol_Discrepancy": ["YES"],
                "Protocol_QC_Basis": ["RAW_OBSERVED_PROTOCOL"],
                "Protocol_Consistency_Status": [
                    "CONSISTENT_WITH_VALIDATED_RAW_OBSERVED_PROTOCOL"
                ],
                "Protocol_Review_Required": ["NO"],
                "Included_In_Final_Scientific_Aggregation": ["YES"],
            }
        )
        result = _augment_cycle_inventory(cycles, protocol)
        self.assertEqual(result.loc[0, "Auto_QC_Flag"], "CHECK")
        self.assertEqual(result.loc[0, "Primary_Capacity_Eligible"], "YES")
        self.assertEqual(result.loc[0, "Full_Run_Exclusion_Reason"], "")


if __name__ == "__main__":
    unittest.main()
