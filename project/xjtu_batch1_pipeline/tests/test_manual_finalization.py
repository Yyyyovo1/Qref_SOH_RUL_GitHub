from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xjtu_batch1.config import load_config  # noqa: E402
from xjtu_batch1.finalization import (  # noqa: E402
    MANUAL_REVIEW_RECORDS,
    _read_sources,
    apply_manual_review,
    compute_final_tables,
    validate_final_tables,
)


class ManualFinalizationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pilot_dir = PROJECT_ROOT / "outputs" / "xjtu_batch1_pilot_v1"
        cls.config = load_config(cls.pilot_dir / "xjtu_batch1_config.json")
        cls.source = _read_sources(cls.pilot_dir)
        (
            cls.cycle_final,
            cls.cycle_final_text,
            cls.manual_review,
        ) = apply_manual_review(cls.source["qc"], cls.source["qc_text"])
        (
            cls.qref,
            cls.soh,
            cls.eol,
            cls.pairwise,
            cls.rul,
            cls.pilot,
        ) = compute_final_tables(cls.cycle_final, cls.config)

    def test_authorized_manual_exclusions(self) -> None:
        indexed = self.cycle_final.set_index(["Cell_ID", "Raw_Cycle_Index"])
        for key in (("2C_battery-1", 390), ("2C_battery-3", 393)):
            row = indexed.loc[key]
            self.assertEqual(row["Manual_Decision"], "EXCLUDE")
            self.assertEqual(row["Final_Include"], "NO")
            self.assertEqual(row["Auto_QC_Flag"], "INVALID_CANDIDATE")

    def test_auto_qc_capacity_and_indices_unchanged(self) -> None:
        source = self.source["qc"]
        self.assertTrue(self.cycle_final["Auto_QC_Flag"].equals(source["Auto_QC_Flag"]))
        for column in (
            "Raw_Cycle_Index",
            "Aging_Cycle_Index",
            "Summary_Discharge_Capacity_Ah",
        ):
            pd.testing.assert_series_equal(
                self.cycle_final[column], source[column], check_names=False, check_dtype=False
            )

    def test_battery1_and_battery3_nominal_eol_are_censored(self) -> None:
        indexed = self.eol.set_index(["Cell_ID", "Qref_Definition"])
        for cell_id in ("2C_battery-1", "2C_battery-3"):
            row = indexed.loc[(cell_id, "nominal")]
            self.assertEqual(row["EOL_Status"], "RIGHT_CENSORED")
            self.assertTrue(pd.isna(row["EOL_Raw_Cycle"]))

    def test_invalid_cycles_absent_from_final_analysis(self) -> None:
        invalid = self.cycle_final.loc[
            self.cycle_final["Auto_QC_Flag"].eq("INVALID_CANDIDATE")
        ]
        self.assertTrue(invalid["Final_Include"].eq("NO").all())
        excluded_keys = set(zip(invalid["Cell_ID"], invalid["Raw_Cycle_Index"].astype(int)))
        soh_keys = set(zip(self.soh["Cell_ID"], self.soh["Raw_Cycle_Index"].astype(int)))
        rul_keys = set(zip(self.rul["Cell_ID"], self.rul["Raw_Cycle_Index"].astype(int)))
        self.assertTrue(excluded_keys.isdisjoint(soh_keys))
        self.assertTrue(excluded_keys.isdisjoint(rul_keys))

    def test_censored_rul_is_na(self) -> None:
        censored = self.rul.loc[self.rul["EOL_Status"].eq("RIGHT_CENSORED")]
        self.assertTrue(censored["RUL_Cycles"].isna().all())
        self.assertTrue(censored["RUL_Status"].eq("CENSORED_EOL").all())

    def test_unreviewed_nonpass_cycle_stops(self) -> None:
        incomplete_records = MANUAL_REVIEW_RECORDS[:1]
        with self.assertRaisesRegex(AssertionError, "UNRESOLVED_NONPASS_MANUAL_REVIEW"):
            apply_manual_review(
                self.source["qc"],
                self.source["qc_text"],
                manual_records=incomplete_records,
            )

    def test_full_final_validation_suite(self) -> None:
        results = validate_final_tables(
            self.source,
            self.cycle_final,
            self.cycle_final_text,
            self.manual_review,
            self.qref,
            self.soh,
            self.eol,
            self.pairwise,
            self.rul,
            self.pilot,
        )
        self.assertTrue(results)
        self.assertTrue(all(value == "PASSED" for value in results.values()))


if __name__ == "__main__":
    unittest.main()
