"""Constants fixed by PROTOCOL_v4.md (commit 74f645a, Zenodo 10.5281/zenodo.24222904) and AMENDMENT_01.md
(commit 9e912a0, Zenodo 10.5281/zenodo.24223632). Nothing here is tuned on v4 results."""

from __future__ import annotations

from rul_ranking_pilot.config import (  # noqa: F401  (re-exported, unchanged from v1/v2)
    BASELINE,
    EXACT,
    INTERVAL,
    MAX_BOOT_MISSING_FRACTION,
    MIN_CELLS,
    N_BOOTSTRAP,
    N_FOLDS,
    QREF_PAIRS,
    QREF_SHORT,
    QREFS,
    RANDOM_SEED,
    RIGHT,
)

PIPELINE_VERSION = "RUL_V4_CORE_CLAIM"
PROTOCOL_COMMIT = "74f645a"
AMENDMENT_01_COMMIT = "9e912a0"
PROTOCOL_DOI = "10.5281/zenodo.24222904"
AMENDMENT_01_DOI = "10.5281/zenodo.24223632"

# ------------------------------------------------------------------ archived files (PREREG_RECORD.md, AMENDMENT_01_RECORD.md)
# Paths relative to the repository root. The run stops if any hash differs.
ARCHIVED_SHA256 = {
    "project/v4_core_claim/PROTOCOL_v4.md": "f14204f90df78a5cfacc024699b3d2b3399b536de99f1a6751fa5a7ec65ad9b9",
    "reruns/v4_prep/1_feature_comparison_summary.json": "d4c9f22a53ba073f7509d9023705aab85762151d91bb58da07acecde48a23746",
    "reruns/v4_prep/1_feature_comparison_vs_attia2021.csv": "ccd19bd69d77a52639544b48395292b43872d82275f7e86a988881ffa97149d1",
    "reruns/v4_prep/1_severson_features_mit124.csv": "8c5cbb8c3afe969bddf49acd8fc191c699cd05ee3f6ad98ec443a4dfa97e1ec5",
    "reruns/v4_prep/3_mit_common_windows.csv": "93ffac0316a990a1a9dd32a3848cc75b370d720c89b802f81f8f461170a2475c",
    "reruns/v4_prep/3_mit_estimability_by_pair_theta.csv": "c2141253d55ecde62a3abbfafdde52d670e11b703b7f2f7698717691be1c20c2",
    "reruns/v4_prep/3_mit_label_summary.json": "6bb5750305514893501a627eabbbca656a68834855676c32ef86e9d79bb48d04",
    "reruns/v4_prep/3_mit_labeled_share_eol_gt_100.csv": "df1a6ed2516a29a4c6419fedb81ac9620204d10b337b03832c6d88fec898d178",
    "reruns/v4_prep/3_mit_theta080_original_split_coverage.csv": "f568d1b75ac79111cccfadf208313151704a5d893b32c2d26c070fd9ec9a6f1c",
    "reruns/v4_prep/4_exp3_window_summary.csv": "aed142dbb980c3f1fb5977c670043b31f49b366e14eb8b19eb491943387ba2c0",
    "reruns/v4_prep/4_exp3_windows_estimability.csv": "b29b82dafb1d2082d9a1e121ed2923600f31b5874a18dc3edf5374c06cfde7e9",
    "reruns/v4_prep/V4_PREP_REPORT.md": "6ed7926ef48c4d317f92b6bbcabb8ba371ca8c1e97beeaa1e0d936d13a593eaf",
    "reruns/v4_prep/exp3_windows_estimability.py": "163505b2b4b64ce0c3433cca01c4d9033f169049cf023548c71aa889f827e7fc",
    "reruns/v4_prep/mit_label_windows.py": "c75b9022934abd05e6fb841fddf0cb1d4c027da999a7c8526d82cc6f345fbb0c",
    "reruns/v4_prep/severson_features.py": "81a817723239e13db53562084dc712f35f72dcc4b4f49a6d86cb742872fab209",
    "project/v4_core_claim/AMENDMENT_01.md": "1ce6d3a83dd8fa351688d8a860a131a308a145b78c0019f6fc90e6467e96d36e",
    "project/v4_core_claim/IMPLEMENTATION_PLAN_stage1.md": "08f48206391a7db7902a91002748c683570917da96f75cd44b7bd2fb92672679",
}
RECORD_FILES = ("project/v4_core_claim/PREREG_RECORD.md", "project/v4_core_claim/AMENDMENT_01_RECORD.md")

# ------------------------------------------------------------------ paths (relative to the project directory)
PREP_DIR = "../reruns/v4_prep"
SEVERSON_SCRIPT = f"{PREP_DIR}/severson_features.py"
SEVERSON_FEATURES_REF = f"{PREP_DIR}/1_severson_features_mit124.csv"
MIT_WINDOWS_REF = f"{PREP_DIR}/3_mit_common_windows.csv"
MIT_SHARES_REF = f"{PREP_DIR}/3_mit_labeled_share_eol_gt_100.csv"
EXP3_WINDOWS_REF = f"{PREP_DIR}/4_exp3_windows_estimability.csv"
MIT_RAW_DIR = "../raw_data/MIT"
ATTIA_DIR = "../raw_data/attia2021_revisit_severson"
MIT_DIR = "outputs/mit_124_final_v1"
MIT_INVENTORY = f"{MIT_DIR}/mit_cell_inventory.csv"
MIT_FOLDS = f"{MIT_DIR}/mit_cell_fold_assignments.csv"
TRAPD_LABELS = "outputs/protocol_validation_v1/trapD_labels.csv"
GB_PLAN = "outputs/soh_ml_design_v1/ml_model_hyperparameter_plan.csv"
V2_CHECK5 = "outputs/rul_ranking_pilot_v2/check5_baseline.csv"
V3_CHECK5 = "outputs/rul_ranking_v3_tju/check5_baseline.csv"
OUTPUT_ROOT = "../reruns/v4"

# ------------------------------------------------------------------ experiment 1 (protocol section 2)
EXP1_DATASET = "MIT"
EXP1_THETAS = tuple(round(0.75 + 0.01 * i, 2) for i in range(21))  # implementation notes 2.4 (as the prep script)
EXP1_C_THETA = 0.80
PREDICTION_CYCLE = 100
WINDOW_SHARE = 0.90
RULES = ("strict", "tolerance")
EXP1_MODELS = ("variance", "discharge", "full")
EXP1_PAIRS = (("variance", "discharge"), ("variance", "full"), ("discharge", "full"))
H1_PAIRS = ("variance-discharge", "variance-full")
ENET_L1_RATIOS = (0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0)
ENET_N_ALPHAS = 100
ENET_EPS = 1e-3
INNER_FOLDS = 4
INNER_REPEATS = 10
SEVERSON_SPLITS = ("train", "test1", "test2")
CHECK_S_REL_TOL = 1e-11   # amendment 01 section 1
CHECK_S_ABS_TOL = 1e-12
DESCRIPTIVE_EXCLUDED_CELL = "b1c18"

# ------------------------------------------------------------------ experiment 3 (protocol section 4)
EXP3_DATASETS = ("HUST", "MIT", "NCA")
EXP3_LABEL = {"HUST": "HUST", "MIT": "MIT", "NCA": "TJU_NCA"}
EXP3_METHODS = ("RF", "TRAJ", "WAFT", "GB", "BASE")  # amendment 01 section 6
V2_METHODS = ("RF", "TRAJ", "WAFT", "BASE")
CHECK5_PAIRS = ("RF-BASE", "TRAJ-BASE", "WAFT-BASE")  # best method chosen among v2 methods only
H3_PAIR = "RF-GB"
EXP3_C_THETA = 0.80
GB_PARAMS = {"learning_rate": 0.05, "loss": "squared_error", "max_depth": 3, "min_samples_leaf": 3,
             "n_estimators": 200, "random_state": RANDOM_SEED, "subsample": 1.0}
CHECK5_EXPECTED = {"HUST": (99, 102), "MIT": (82, 115), "NCA": (87, 105)}  # amendment 01 section 6

# ------------------------------------------------------------------ experiment 2 and verdicts (protocol sections 3, 5)
DELTA_FRACTION = 0.10
DETECTION_FRACTIONS = (0.10, 0.20)
DETECTION_SE_MULTIPLES = (1.0, 2.0)
DETECTION_Z = 1.96  # protocol section 3.3 writes 1.96
H2_NONEQUIVALENT_MAX = 0.05
H2_IMPRECISE_MAX = 0.50
H2_EQUIVALENT_MIN = 0.50
H1_SUPPORT_MIN = 0.50
H1_REFUTE_THETA_POINTS = 2
H3_MAX_REVERSALS = 1
H3_LOW_POWER_MEDIAN = 0.50
