"""Frozen constants for the MIT/Severson ingest (PROTOCOL_v2.md section 3.3, commit 4cdf0a2)."""

from __future__ import annotations

PIPELINE_VERSION = "MIT_124_FINAL_V1"
PROTOCOL_COMMIT = "4cdf0a2"
OUTPUT_DIR_NAME = "mit_124_final_v1"
RAW_DIR = "raw_data/MIT"

# data.matr.io project "Data-driven prediction of battery cycle life before capacity degradation".
SOURCE_PAGE = "https://data.matr.io/1/projects/5c48dd2bc625d700019f3204"
LICENSE_TEXT = "All data is released under CC BY 4"
CITATION = ("Severson, K. A. et al. Data-driven prediction of battery cycle life before capacity "
            "degradation. Nature Energy 4, 383-391 (2019). doi:10.1038/s41560-019-0356-8")
BATCHES = (
    {"batch": 1, "date": "2017-05-12", "file_id": "5c86c0b5fa2ede00015ddf66",
     "file": "2017-05-12_batchdata_updated_struct_errorcorrect.mat", "bytes": 3025320241},
    {"batch": 2, "date": "2017-06-30", "file_id": "5c86bf13fa2ede00015ddd82",
     "file": "2017-06-30_batchdata_updated_struct_errorcorrect.mat", "bytes": 2007331155},
    {"batch": 3, "date": "2018-04-12", "file_id": "5c86bd64fa2ede00015ddbb2",
     "file": "2018-04-12_batchdata_updated_struct_errorcorrect.mat", "bytes": 3236690412},
)

# Authors' processing (Load Data.ipynb in rdbraatz/data-driven-prediction-...).
AUTHOR_DELETIONS = {
    "b1c8": "AUTHOR_EXCLUDED_DID_NOT_REACH_80PCT", "b1c10": "AUTHOR_EXCLUDED_DID_NOT_REACH_80PCT",
    "b1c12": "AUTHOR_EXCLUDED_DID_NOT_REACH_80PCT", "b1c13": "AUTHOR_EXCLUDED_DID_NOT_REACH_80PCT",
    "b1c22": "AUTHOR_EXCLUDED_DID_NOT_REACH_80PCT",
    "b3c2": "AUTHOR_EXCLUDED_NOISY_CHANNEL", "b3c23": "AUTHOR_EXCLUDED_NOISY_CHANNEL",
    "b3c32": "AUTHOR_EXCLUDED_NOISY_CHANNEL", "b3c37": "AUTHOR_EXCLUDED_NOISY_CHANNEL",
    "b3c42": "AUTHOR_EXCLUDED_NOISY_CHANNEL", "b3c43": "AUTHOR_EXCLUDED_NOISY_CHANNEL",
}
AUTHOR_MERGES = (  # (batch-1 key, batch-2 continuation key, author add_len)
    ("b1c0", "b2c7", 662), ("b1c1", "b2c8", 981), ("b1c2", "b2c9", 1060),
    ("b1c3", "b2c15", 208), ("b1c4", "b2c16", 482),
)
EXPECTED_CELLS_PER_BATCH = {1: 41, 2: 43, 3: 40}

NOMINAL_CAPACITY_AH = 1.1
# Cycle-record current is stored in C-rate units; 4C discharge appears as -4.0.
EXPECTED_DISCHARGE_CURRENT_C = -4.0

# QC rules (PROTOCOL_v2 section 3.3); frozen with this commit.
QC = {
    "max_capacity_factor_of_nominal": 1.5,
    "exclude_cycle_index_zero": True,
    "protocol_discharge_current_relative_tolerance": 0.10,
    "capacity_jump_abs_threshold_Ah": 0.050,
    "capacity_jump_relative_threshold": 0.050,
    "capacity_jump_is_check_only": True,
}

QREFS = ("nominal", "first_comparable", "early_median5_comparable")
THETAS = tuple(round(0.75 + 0.01 * i, 2) for i in range(21))
REBOUND_THETAS = (0.75, 0.80, 0.85)
RANDOM_SEED = 20260928
N_FOLDS = 5
T0_FRACTIONS = (0.35, 0.50)
