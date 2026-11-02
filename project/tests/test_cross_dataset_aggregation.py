from __future__ import annotations

import math
import unittest

from cross_dataset_repro.aggregation import (
    ALLOWED_TRANSITIONS,
    classify_qref_direction,
    classify_transition,
    compute_effect_bounds,
    ordering_consistency,
)


class CrossDatasetAggregationTests(unittest.TestCase):
    def test_qref_direction(self) -> None:
        self.assertEqual(classify_qref_direction(1.2, 1.1), "QREF_ABOVE_NOMINAL")
        self.assertEqual(classify_qref_direction(1.1, 1.1), "QREF_EQUAL_NOMINAL")
        self.assertEqual(classify_qref_direction(1.0, 1.1), "QREF_BELOW_NOMINAL")

    def test_transition_categories_are_frozen(self) -> None:
        self.assertEqual(classify_transition("EXACT_OBSERVED", "RIGHT_CENSORED"), "EXACT_TO_RIGHT")
        self.assertEqual(classify_transition("INTERVAL_CENSORED", "EXACT_OBSERVED"), "INTERVAL_TO_EXACT")
        self.assertEqual(classify_transition("RIGHT_CENSORED", "RIGHT_CENSORED"), "BOTH_RIGHT")
        self.assertEqual(len(ALLOWED_TRANSITIONS), 9)

    def test_right_to_exact_is_only_a_lower_bound_on_earlier_eol(self) -> None:
        result = compute_effect_bounds(
            status_a="RIGHT_CENSORED",
            exact_a=math.nan,
            lower_a=1504.0,
            upper_a=math.nan,
            last_a=1504.0,
            status_b="EXACT_OBSERVED",
            exact_b=1347.0,
            lower_b=math.nan,
            upper_b=math.nan,
            last_b=1504.0,
        )
        self.assertTrue(math.isnan(result["Delta_EOL_Exact_Cycles"]))
        self.assertEqual(result["Delta_EOL_Upper_Bound_Cycles"], -157.0)
        self.assertEqual(result["Minimum_Observed_Earlier_By_Cycles"], 157.0)
        self.assertEqual(result["Effect_Estimate_Type"], "LOWER_BOUND_ON_EARLIER_EOL_ONLY")

    def test_interval_bounds_remain_bounds(self) -> None:
        result = compute_effect_bounds(
            status_a="INTERVAL_CENSORED",
            exact_a=math.nan,
            lower_a=100.0,
            upper_a=110.0,
            last_a=200.0,
            status_b="INTERVAL_CENSORED",
            exact_b=math.nan,
            lower_b=80.0,
            upper_b=90.0,
            last_b=200.0,
        )
        self.assertTrue(math.isnan(result["Delta_EOL_Exact_Cycles"]))
        self.assertEqual(result["Delta_EOL_Lower_Bound_Cycles"], -30.0)
        self.assertEqual(result["Delta_EOL_Upper_Bound_Cycles"], -10.0)

    def test_ordering_consistency_uses_censoring_bounds(self) -> None:
        result, _ = ordering_consistency(
            direction="QREF_ABOVE_NOMINAL",
            status_a="RIGHT_CENSORED",
            exact_a=math.nan,
            lower_a=1504.0,
            upper_a=math.nan,
            last_a=1504.0,
            status_b="EXACT_OBSERVED",
            exact_b=1347.0,
            lower_b=math.nan,
            upper_b=math.nan,
            last_b=1504.0,
        )
        self.assertEqual(result, "CONSISTENT")


if __name__ == "__main__":
    unittest.main()
