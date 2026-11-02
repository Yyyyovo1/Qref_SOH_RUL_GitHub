from __future__ import annotations

import unittest

import pandas as pd

from soh_ml_design_repro.design import (
    COMMON_QREFS,
    N_FOLDS,
    OUTPUT_NAMES,
    _feature_specs,
    assign_hust_folds,
    assign_xjtu_folds,
    build_metric_plan,
    build_model_hyperparameter_plan,
)


class SOHMLDesignUnitTests(unittest.TestCase):
    def test_required_output_inventory_is_complete(self) -> None:
        self.assertEqual(len(OUTPUT_NAMES), 14)
        self.assertIn("ml_design_regression_tests.csv", OUTPUT_NAMES)

    def test_safe_core_has_no_target_or_future_features(self) -> None:
        core = [row for row in _feature_specs() if row["Decision"] == "SAFE_CORE"]
        self.assertGreater(len(core), 0)
        for row in core:
            self.assertEqual(row["Uses_Target_Capacity"], "NO")
            self.assertEqual(row["Uses_Future_Information"], "NO")
            upper = row["Feature_Name"].upper()
            for prohibited in ("QREF", "SOH", "RUL", "EOL", "CAPACITY_AH"):
                self.assertNotIn(prohibited, upper)

    def test_full_discharge_duration_is_not_safe_core(self) -> None:
        row = next(
            item
            for item in _feature_specs()
            if item["Feature_Name"] == "Full_Discharge_Duration_s"
        )
        self.assertEqual(row["Decision"], "PROHIBITED")
        self.assertEqual(row["Leakage_Risk"], "HIGH")
        self.assertIn("HIGH_LEAKAGE_RISK", row["Potential_Capacity_Proxy"])

    def test_xjtu_assignment_is_deterministic_and_cell_grouped(self) -> None:
        cells = pd.DataFrame(
            {
                "Batch_ID": [1] * 8 + [2] * 15 + [3] * 8 + [4] * 8 + [5] * 8 + [6] * 8,
                "Cell_ID": [f"cell_{index:02d}" for index in range(55)],
            }
        )
        first = assign_xjtu_folds(cells)
        second = assign_xjtu_folds(cells.sample(frac=1.0, random_state=7))
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first["Cell_ID"].duplicated().any())
        self.assertEqual(set(first["Fold_ID"]), set(range(1, N_FOLDS + 1)))
        counts = first.groupby("Fold_ID")["Cell_ID"].size()
        self.assertLessEqual(counts.max() - counts.min(), 1)

    def test_hust_assignment_is_deterministic_and_cell_grouped(self) -> None:
        cells = [f"HUST_{index:02d}" for index in range(1, 78)]
        first = assign_hust_folds(cells)
        second = assign_hust_folds(reversed(cells))
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first["Cell_ID"].duplicated().any())
        counts = first.groupby("Fold_ID")["Cell_ID"].size()
        self.assertLessEqual(counts.max() - counts.min(), 1)

    def test_model_plan_is_untrained_and_fixed(self) -> None:
        plan = build_model_hyperparameter_plan()
        self.assertEqual(len(plan), 3)
        self.assertTrue(plan["Training_Status"].eq("NOT_TRAINED_IN_DESIGN_RUN").all())
        self.assertTrue(plan["Hyperparameter_Mode"].eq("FIXED").all())
        self.assertTrue(plan["Test_Fold_Tuning"].eq("PROHIBITED").all())

    def test_primary_metric_is_cell_level_physical_mae(self) -> None:
        plan = build_metric_plan()
        primary = plan.loc[plan["Primary_Model_Ranking_Metric"].eq("YES")]
        self.assertEqual(primary["Metric_Name"].tolist(), ["MAE_Q_Ah"])
        self.assertEqual(primary.iloc[0]["Metric_Space"], "PHYSICAL_CAPACITY_SPACE")
        self.assertTrue(plan["First_Level_Aggregation_Unit"].eq("HELD_OUT_Cell_ID").all())

    def test_common_qrefs_are_exact(self) -> None:
        self.assertEqual(
            COMMON_QREFS,
            ("nominal", "first_comparable", "early_median5_comparable"),
        )


if __name__ == "__main__":
    unittest.main()
