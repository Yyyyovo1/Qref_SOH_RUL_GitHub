"""Frozen constants for the TJU (Tongji) ingest. Datasets 1 (NCA) and 2 (NCM) only; dataset 3 (NCM+NCA, 9 cells) excluded."""

from __future__ import annotations

PIPELINE_VERSION = "TJU_NCA_NCM_FINAL_V1"
OUTPUT_DIR_NAME = "tju_final_v1"
SCAN_DIR_NAME = "tju_scan_v1"
RAW_DIR = "raw_data/TJU/zenodo_6405084_v2"

ZENODO = {"concept_record": 6379165, "concept_doi": "10.5281/zenodo.6379165", "version_record": 6405084,
          "version_doi": "10.5281/zenodo.6405084", "version": "v2 (2022-04-01); v1 is record 6379166 (2022-03-25)",
          "license": "CC BY 4.0"}
CITATION = ("Zhu, J. et al. Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation. "
            "Nature Communications 13, 2261 (2022). doi:10.1038/s41467-022-29837-w")
FILES = {
    "NCA": {"zip": "Dataset_1_NCA_battery.zip", "bytes": 579629324, "md5": "cbd6d7eab2a296b52385e025e8d0f42a",
            "sha256": "0d58f6f09f4ada76c2c3ab038ce98926ebdba7d18aaa31a9d7bebdccc9a834ac", "csv_count": 66},
    "NCM": {"zip": "Dataset_2_NCM_battery.zip", "bytes": 737047539, "md5": "a05c485df92f4af560d1d8d7f659ce1b",
            "sha256": "942e1dc6cdaf900be350dd36dcb2b310cc49872775f1650e7b3306667e439d0f", "csv_count": 55},
}
# Zhu et al. 2022, Table 1 and Methods.
NOMINAL_CAPACITY_AH = {"NCA": 3.5, "NCM": 3.5}
CELL_MODEL = {"NCA": "LG INR18650-35E", "NCM": "Samsung INR18650-MJ1"}
DISCHARGE_CUTOFF_V = {"NCA": 2.65, "NCM": 2.5}
CHARGE_CUTOFF_V = 4.2
EXPECTED_DISCHARGE_CURRENT_MA = -3500.0  # 1C for both datasets (Table 1: all NCA / NCM conditions discharge at 1C)

COLUMNS = {"time": "time/s", "voltage": "Ecell/V", "current": "<I>/mA", "qd": "Q discharge/mA.h", "qc": "Q charge/mA.h",
           "cycle": "cycle number"}
CURRENT_SIGN_THRESHOLD_MA = 50.0

# QC rules (same structure as HUST / MIT). Frozen with this commit.
QC = {
    "max_capacity_factor_of_nominal": 1.5,
    "protocol_discharge_current_relative_tolerance": 0.10,
    "full_discharge_voltage_margin_V": 0.05,   # discharge must reach cutoff + margin
    "full_charge_voltage_margin_V": 0.02,      # charge must reach 4.2 V - margin before the discharge
    "capacity_jump_abs_threshold_Ah": 0.050,
    "capacity_jump_relative_threshold": 0.050,
    "capacity_jump_is_check_only": True,
}

QREFS = ("nominal", "first_comparable", "early_median5_comparable")
RANDOM_SEED = 20260928
N_FOLDS = 5
