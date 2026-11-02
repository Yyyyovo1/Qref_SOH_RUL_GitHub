from __future__ import annotations

import inspect
import unittest

import numpy as np

from soh_feature_extraction_repro.config import FEATURE_NAMES
from soh_feature_extraction_repro.extractors import (
    _robust_huber_slope,
    extract_hust_features,
    extract_xjtu_features,
    invalid_extraction_result,
    matrix_row,
)


class SOHFeatureExtractionUnitTests(unittest.TestCase):
    def test_robust_slope_uses_physical_time_and_is_deterministic(self) -> None:
        time_s = np.asarray([0, 1, 2, 10, 20, 30, 40, 50, 60, 98, 99, 100], dtype=float)
        values = 1.0 + 2.0 * time_s
        slope_first, count_first, reason_first = _robust_huber_slope(time_s, values)
        slope_second, count_second, reason_second = _robust_huber_slope(time_s, values)
        self.assertEqual(reason_first, "PASS")
        self.assertEqual(reason_second, "PASS")
        self.assertEqual(count_first, count_second)
        self.assertAlmostEqual(slope_first, 2.0, places=12)
        self.assertEqual(slope_first, slope_second)

    def test_xjtu_extractor_returns_exact_feature_set_before_discharge(self) -> None:
        # XJTU raw time is minutes; use a 5-second sample interval here.
        cc_time = np.arange(11, dtype=float) / 12.0
        cv_time = np.arange(11, dtype=float) / 12.0
        rest_time = np.arange(2, dtype=float) / 12.0
        discharge_time = np.arange(10, dtype=float) / 12.0
        time_min = np.r_[cc_time, cv_time, rest_time, discharge_time]
        current = np.r_[
            np.full(11, 2.0),
            np.linspace(1.7, 0.1, 11),
            np.zeros(2),
            np.full(10, -1.0),
        ]
        voltage = np.r_[
            np.linspace(3.3, 4.17, 11),
            np.linspace(4.19, 4.20, 11),
            [4.18, 4.15],
            np.linspace(4.10, 2.50, 10),
        ]
        result = extract_xjtu_features(
            relative_time_min=time_min,
            current_A=current,
            voltage_V=voltage,
        )
        self.assertEqual(tuple(result.features), FEATURE_NAMES)
        self.assertTrue(all(np.isfinite(item.value) for item in result.features.values()))
        self.assertEqual(result.raw_time_unit, "MINUTES")
        self.assertEqual(result.time_conversion_factor_to_seconds, 60.0)
        self.assertEqual(result.pre_discharge_state_type, "AFTER_REST")
        discharge_start = result.boundaries["discharge_start_index"]
        self.assertIsNotNone(discharge_start)
        self.assertTrue(
            all(
                item.stage_end_index is None or item.stage_end_index < discharge_start
                for item in result.features.values()
            )
        )

    def test_hust_validated_status_mapping_and_cycle_index_sensitivity(self) -> None:
        status = np.asarray(
            ["Constant current charge"] * 5
            + ["Constant current-constant voltage charge"] * 22
            + ["Constant current discharge_0"] * 10,
            dtype=object,
        )
        time_s = np.arange(status.size, dtype=float) * 5.0
        current_mA = np.r_[
            np.full(5, 5500.0),
            np.full(11, 1100.0),
            np.linspace(900.0, 50.0, 11),
            np.full(10, -5500.0),
        ]
        voltage = np.r_[
            np.linspace(2.8, 3.4, 5),
            np.linspace(3.45, 4.17, 11),
            np.linspace(4.19, 4.20, 11),
            np.linspace(4.15, 2.0, 10),
        ]
        result = extract_hust_features(
            time_s=time_s,
            current_mA=current_mA,
            voltage_V=voltage,
            status=status,
        )
        self.assertEqual(tuple(result.features), FEATURE_NAMES)
        self.assertEqual(result.raw_time_unit, "SECONDS")
        self.assertEqual(result.time_conversion_factor_to_seconds, 1.0)
        self.assertEqual(result.pre_discharge_state_type, "AFTER_CHARGE")
        row = matrix_row(
            result=result,
            dataset="HUST",
            batch_id=None,
            cell_id="HUST_SYNTHETIC",
            raw_cycle_index=17,
            fold_id=3,
        )
        self.assertEqual(row["Feature_Set_B_Raw_Cycle_Index"], 17)
        self.assertEqual(row["Raw_Cycle_Index"], 17)
        self.assertTrue(all(name in row for name in FEATURE_NAMES))

    def test_short_cv_terminal_window_is_retained_and_flagged(self) -> None:
        cc_time = np.arange(11, dtype=float)
        cv_time = 11.0 + np.arange(11, dtype=float) * 2.0
        discharge_time = 32.0 + np.arange(10, dtype=float)
        result = extract_xjtu_features(
            relative_time_min=np.r_[cc_time, cv_time, discharge_time] / 60.0,
            current_A=np.r_[
                np.full(11, 2.0),
                np.linspace(1.7, 0.1, 11),
                np.full(10, -1.0),
            ],
            voltage_V=np.r_[
                np.linspace(3.3, 4.17, 11),
                np.linspace(4.19, 4.20, 11),
                np.linspace(4.1, 2.5, 10),
            ],
        )
        feature = result.features["Charge_Terminal_Voltage_Std_V"]
        self.assertTrue(np.isfinite(feature.value))
        self.assertEqual(feature.status, "CHECK")
        self.assertEqual(feature.reason_code, "SHORT_TERMINAL_WINDOW")

    def test_invalid_extraction_preserves_all_feature_rows_as_na(self) -> None:
        result = invalid_extraction_result(
            reason="SYNTHETIC_RAW_ERROR",
            raw_time_unit="SECONDS",
            time_conversion_factor_to_seconds=1.0,
        )
        self.assertEqual(tuple(result.features), FEATURE_NAMES)
        self.assertEqual(result.overall_flag, "INVALID_EXTRACTION")
        self.assertTrue(all(item.status == "INVALID_EXTRACTION" for item in result.features.values()))
        self.assertTrue(all(np.isnan(item.value) for item in result.features.values()))

    def test_extractor_signatures_exclude_targets_and_qref(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(extract_xjtu_features).parameters),
            ("relative_time_min", "current_A", "voltage_V"),
        )
        self.assertEqual(
            tuple(inspect.signature(extract_hust_features).parameters),
            ("time_s", "current_mA", "voltage_V", "status"),
        )


if __name__ == "__main__":
    unittest.main()
