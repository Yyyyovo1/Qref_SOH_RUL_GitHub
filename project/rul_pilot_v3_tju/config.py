"""Constants fixed by PROTOCOL_v3_TJU.md (commit 332dbf4) and the TJU freeze v2 (commit 22801aa).

Methods, hyperparameters, estimability, bootstrap and reversal rules are imported unchanged from v2 / v1.
"""

from __future__ import annotations

from rul_pilot_v2.config import (  # noqa: F401  (re-exported, unchanged from v2)
    BASELINE,
    CAPACITY_FEATURES,
    EVAL_SETS,
    EXACT,
    INTERVAL,
    MARGIN_FEATURE,
    METHODS,
    MIN_CELLS,
    QREF_PAIRS,
    QREFS,
    RIGHT,
)

PIPELINE_VERSION = "RUL_RANKING_V3_TJU"
PROTOCOL_COMMIT = "332dbf4"
TJU_FREEZE_COMMIT = "22801aa"
OUTPUT_DIR_NAME = "rul_ranking_v3_tju"
TJU_DIR = "outputs/tju_final_v2"
LABEL_REFERENCE = "../reruns/tju_label_structure/2_labels_per_cell_setting.csv"  # committed with the protocol (check 1)

DATASETS = ("NCA", "NCM")
NOMINAL_CAPACITY_AH = {"NCA": 3.5, "NCM": 3.5}
THETAS = tuple(round(0.60 + 0.01 * i, 2) for i in range(36))
T0 = {"NCA": (6, 77, 110), "NCM": (5, 142, 203)}  # PROTOCOL_v3_TJU section 2 (frozen)
STATIC_FEATURES = ("Temperature_C", "Charge_CRate")  # section 3.2: known from the test design

# Section 5 (predictions).
P0_MAX_REVERSALS = 1
P1_SEGMENTS = {"A": (0.30, 0.60), "B": (0.60, 0.90)}  # lower strict labeled share r: lo <= r < hi
P1_MIN_RATIO = 2.0
P1_MIN_RATE_A = 0.02
MIN_ESTIMABLE_COMPARISONS = 100
QREF_SHORT = {"nominal": "nominal", "first_comparable": "first", "early_median5_comparable": "first5"}
