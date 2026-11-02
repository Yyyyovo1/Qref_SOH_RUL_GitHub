from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hust_repro.pilot import (
    EOL_EXACT,
    EOL_RIGHT_CENSORED,
    build_eol_pairwise,
    build_rul_labels,
    compute_qref_summary,
    detect_eol,
)


class QrefTests(unittest.TestCase):
    def test_qref_source_cycles_are_sorted_numerically(self) -> None:
        source = pd.DataFrame(
            {
                "Cell_ID": ["HUST_TEST"] * 6,
                "Source_File": ["test.pkl"] * 6,
                "Raw_Cycle_Index": ["1", "10", "2", "3", "4", "5"],
                "Primary_Capacity_Ah": [1.20, 1.10, 1.19, 1.18, 1.17, 1.16],
                "Primary_Capacity_Eligible": ["YES"] * 6,
            }
        )
        result = compute_qref_summary(source, ["HUST_TEST"])
        row = result.iloc[0]
        self.assertEqual(row["Qref_First_Comparable_Source_Cycle"], 1)
        self.assertEqual(
            row["Qref_Early_Median5_Comparable_Source_Cycles"], "1|2|3|4|5"
        )
        self.assertAlmostEqual(row["Qref_Early_Median5_Comparable_Ah"], 1.18)


class EOLTests(unittest.TestCase):
    @staticmethod
    def eligible(capacities: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Raw_Cycle_Index": list(range(1, len(capacities) + 1)),
                "Primary_Capacity_Ah": capacities,
            }
        )

    def test_eol_uses_strict_unrounded_less_than_or_equal(self) -> None:
        result = detect_eol(self.eligible([0.9000001, 0.88, 0.87]), 0.88)
        self.assertEqual(result["EOL_Status"], EOL_EXACT)
        self.assertEqual(result["EOL_Raw_Cycle"], 2)

    def test_right_censored_eol_has_no_point(self) -> None:
        result = detect_eol(self.eligible([0.91, 0.8800001]), 0.88)
        self.assertEqual(result["EOL_Status"], EOL_RIGHT_CENSORED)
        self.assertTrue(np.isnan(result["EOL_Raw_Cycle"]))
        self.assertTrue(np.isnan(result["EOL_Capacity_Ah"]))

    def test_pairwise_delta_is_b_minus_a_and_censored_is_na(self) -> None:
        rows = []
        for definition, status, cycle in [
            ("nominal", EOL_RIGHT_CENSORED, np.nan),
            ("first_comparable", EOL_EXACT, 80),
            ("early_median5_comparable", EOL_EXACT, 75),
        ]:
            rows.append(
                {
                    "Cell_Selection_Order": 1,
                    "Cell_ID": "HUST_TEST",
                    "Qref_Definition": definition,
                    "Qref_Ah": 1.1,
                    "EOThreshold_Ah": 0.88,
                    "EOL_Status": status,
                    "EOL_Raw_Cycle": cycle,
                }
            )
        result = build_eol_pairwise(
            pd.DataFrame(rows),
            [
                ["nominal", "first_comparable"],
                ["first_comparable", "early_median5_comparable"],
            ],
            "EOL_B_MINUS_EOL_A",
        )
        self.assertTrue(np.isnan(result.iloc[0]["Delta_EOL_Cycles"]))
        self.assertEqual(result.iloc[1]["Delta_EOL_Cycles"], -5)


class RULTests(unittest.TestCase):
    def test_censored_rul_is_na_and_exact_rul_uses_eol_minus_cycle(self) -> None:
        cycles = pd.DataFrame(
            {
                "Cell_Selection_Order": [1, 1, 1],
                "Cell_ID": ["HUST_TEST"] * 3,
                "Source_File": ["test.pkl"] * 3,
                "Raw_Cycle_Index": [1, 2, 3],
                "Primary_Capacity_Ah": [1.1, 1.0, 0.9],
                "Primary_Capacity_Eligible": ["YES"] * 3,
            }
        )
        eol = pd.DataFrame(
            [
                {
                    "Cell_Selection_Order": 1,
                    "Cell_ID": "HUST_TEST",
                    "Qref_Definition": "nominal",
                    "Definition_Order": 1,
                    "Definition_Type": "QREF_DERIVED",
                    "Qref_Ah": 1.1,
                    "EOThreshold_Ah": 0.88,
                    "EOL_Status": EOL_RIGHT_CENSORED,
                    "EOL_Raw_Cycle": np.nan,
                },
                {
                    "Cell_Selection_Order": 1,
                    "Cell_ID": "HUST_TEST",
                    "Qref_Definition": "first_comparable",
                    "Definition_Order": 2,
                    "Definition_Type": "QREF_DERIVED",
                    "Qref_Ah": 1.1,
                    "EOThreshold_Ah": 0.95,
                    "EOL_Status": EOL_EXACT,
                    "EOL_Raw_Cycle": 3,
                },
            ]
        )
        result = build_rul_labels(cycles, eol)
        censored = result[result["Qref_Definition"].eq("nominal")]
        exact = result[result["Qref_Definition"].eq("first_comparable")]
        self.assertTrue(censored["RUL_Cycles"].isna().all())
        self.assertEqual(exact["RUL_Cycles"].tolist(), [2, 1, 0])
        self.assertTrue(result["Native_RUL_Used"].eq("NO").all())


if __name__ == "__main__":
    unittest.main()
