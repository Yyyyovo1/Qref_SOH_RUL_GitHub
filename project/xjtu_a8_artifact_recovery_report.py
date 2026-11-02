from __future__ import annotations

import csv
from pathlib import Path


OUTPUT_DIR = Path("outputs/xjtu_a8_artifact_recovery_v1")
OUTPUT_PATH = OUTPUT_DIR / "xjtu_a8_artifact_recovery_report.csv"


ROWS = [
    {
        "Artifact_Type": "A8_CELL_LEVEL_METRICS_EXPECTED_PATH",
        "Candidate_Path": "outputs/soh_ml_robustness_v1/soh_robustness_a8_cell_metrics.csv",
        "Found": "NO",
        "Usable": "NO",
        "Reason": "Expected A8 cell-level file is absent.",
    },
    {
        "Artifact_Type": "A8_CELL_LEVEL_METRICS_EXPECTED_PATH",
        "Candidate_Path": "outputs/soh_ml_robustness_v1/soh_robustness_cell_metrics.csv",
        "Found": "NO",
        "Usable": "NO",
        "Reason": "Alternative robustness cell-level file is absent.",
    },
    {
        "Artifact_Type": "A8_CELL_LEVEL_METRICS_EXPECTED_PATH",
        "Candidate_Path": "outputs/soh_ml_robustness_v1/a8_cell_metrics.csv",
        "Found": "NO",
        "Usable": "NO",
        "Reason": "Alternative A8 cell-level file is absent.",
    },
    {
        "Artifact_Type": "A8_AGGREGATE_SUMMARY",
        "Candidate_Path": "outputs/soh_ml_robustness_v1/soh_robustness_a8_summary.csv",
        "Found": "YES",
        "Usable": "NO",
        "Reason": (
            "Contains 18 aggregate rows (9 XJTU rows) and reproduces all nine frozen "
            "XJTU medians, but has no Cell_ID-level MAE_Q_Ah rows."
        ),
    },
    {
        "Artifact_Type": "PRIMARY_CELL_LEVEL_METRICS",
        "Candidate_Path": "outputs/soh_primary_benchmark_v1/soh_primary_cell_metrics.csv",
        "Found": "YES",
        "Usable": "NO",
        "Reason": (
            "Contains 1,188 PRIMARY rows only; Analysis_ID values are "
            "XJTU_PRIMARY_COMMON_QREFS_55_CELL and HUST_PRIMARY_COMMON_QREFS_77_CELL, "
            "not ROBUSTNESS_A8."
        ),
    },
    {
        "Artifact_Type": "PRIMARY_HELD_OUT_PREDICTIONS",
        "Candidate_Path": "outputs/soh_primary_benchmark_v1/soh_primary_predictions.csv",
        "Found": "YES",
        "Usable": "NO",
        "Reason": (
            "Contains 1,408,932 held-out prediction rows for frozen PRIMARY analyses only; "
            "there are no ROBUSTNESS_A8 or A8_MINUS_CV_DURATION rows."
        ),
    },
    {
        "Artifact_Type": "PRIMARY_XJTU_BATCH_CELL_METRICS",
        "Candidate_Path": "outputs/soh_primary_benchmark_v1/soh_primary_xjtu_batch_metrics.csv",
        "Found": "YES",
        "Usable": "NO",
        "Reason": "Contains held-out PRIMARY cell metrics, not A8-minus-CV-duration model results.",
    },
    {
        "Artifact_Type": "CROSS_DATASET_CELL_LEVEL",
        "Candidate_Path": "outputs/cross_dataset_summary_v1/cross_dataset_cell_level.csv",
        "Found": "YES",
        "Usable": "NO",
        "Reason": "Contains Qref/EOL/RUL descriptive rows and no model-level MAE_Q_Ah results.",
    },
    {
        "Artifact_Type": "PRIOR_TARGETED_BOOTSTRAP_AGGREGATES",
        "Candidate_Path": "outputs/xjtu_a8_rankflip_verification_v1/",
        "Found": "YES",
        "Usable": "NO",
        "Reason": (
            "Contains four aggregate bootstrap/regression CSV files only; no 495-row "
            "Cell_ID x Qref x Model metric table and no held-out predictions."
        ),
    },
    {
        "Artifact_Type": "A8_HELD_OUT_PREDICTIONS_EXPECTED_PATH",
        "Candidate_Path": "outputs/soh_ml_robustness_v1/soh_robustness_a8_predictions.csv",
        "Found": "NO",
        "Usable": "NO",
        "Reason": "Expected A8 held-out prediction file is absent.",
    },
    {
        "Artifact_Type": "A8_SERIALIZED_MODELS_OR_PREPROCESSORS",
        "Candidate_Path": "workspace recursive search (*.joblib, *.pkl, *.pickle, *.sav, *.bin, *.onnx, *.pt, *.pth, *.model, *.skops)",
        "Found": "NO",
        "Usable": "NO",
        "Reason": (
            "No A8 estimator, scaler, imputer, or fitted pipeline artifact was found. "
            "The .pkl files present are raw HUST data or library test data."
        ),
    },
    {
        "Artifact_Type": "PARQUET_OR_FEATHER_CACHE",
        "Candidate_Path": "workspace recursive search (*.parquet, *.feather)",
        "Found": "NO",
        "Usable": "NO",
        "Reason": "No parquet or feather artifacts exist in the workspace.",
    },
    {
        "Artifact_Type": "ROBUSTNESS_STAGING_DIRECTORY",
        "Candidate_Path": "outputs/.soh_ml_robustness_v1_staging_8ou3dioi/",
        "Found": "YES",
        "Usable": "NO",
        "Reason": "Directory exists but contains zero files.",
    },
    {
        "Artifact_Type": "TASK_RELATED_TEMP_CACHE",
        "Candidate_Path": "LOCAL_HOME/AppData/Local/Temp/ recursive candidate search",
        "Found": "NO",
        "Usable": "NO",
        "Reason": (
            "No A8/robustness/rank-flip/cell-metric/prediction candidate with a supported "
            "data-artifact extension was found."
        ),
    },
    {
        "Artifact_Type": "ROBUSTNESS_PIPELINE_PERSISTENCE_AUDIT",
        "Candidate_Path": "soh_ml_robustness_repro/pipeline.py",
        "Found": "YES",
        "Usable": "NO",
        "Reason": (
            "Source audit shows experiment_cell_metrics is held in memory and only aggregate "
            "summaries/bootstrap tables are written; A8 cell metrics, predictions, and fitted "
            "models are not persisted."
        ),
    },
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    with OUTPUT_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Artifact_Type",
                "Candidate_Path",
                "Found",
                "Usable",
                "Reason",
            ],
        )
        writer.writeheader()
        writer.writerows(ROWS)
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
