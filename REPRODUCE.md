# Reproduction guide

## Prepare

Use Python 3.12. From the repository root:

```text
python tools/verify_release.py
python tools/restore_data.py
python -m pip install -r requirements.txt
python tools/check_supplement.py
python tools/reproduce_s6_s7.py
```

The last two commands read frozen data only; they do not train models or read raw cycles. New verification outputs go to `reruns/`. Original supplementary tables remain unchanged.

## Original processed-data analyses

Run from `project/` and select a fresh output directory (existing frozen directories must not be overwritten):

```text
python cross_dataset_summary.py
python eol_threshold_robustness.py --help
python soh_primary_benchmark.py --workspace . --output-dir ../reruns/primary_benchmark
python soh_ml_robustness.py --help
python reconstruct_xjtu_a8.py --help
```

The ML stages use frozen feature matrices, labels, eligibility masks and cell-level folds already included in this package. Feature extraction and raw-data QC require separately obtained original datasets under `Batch-1/` … `Batch-6/`, `HUST 77/`, and the CALCE cell folders. See project/HUST_PIPELINE_README.md, project/XJTU_MULTIBATCH_README.md and the pipeline-specific READMEs. Paths are relative to `project/` or the `--workspace` argument.

## Supplement versions

The authoritative published-material candidate is `supplementary/original/Supplementary_Tables_S1-S9.xlsx`. The workbook uses the display name `first5_comparable` for the frozen data key `early_median5_comparable`.

Historical `project/xjtu_a8_rankflip_repro/pipeline.py` and its historical tests calculate median of paired differences in the RF/GB interval function. Final Table S6 instead specifies the difference of medians. Those are different statistics. To reproduce final S6 and S7, use `tools/reproduce_s6_s7.py`, which uses the frozen A8 per-cell MAE_Q data, 5,000 paired Cell_ID resamples with seed 20260928, the same draws across Qref definitions and linear percentile intervals. The new helper checks its results against the unmodified workbook, including the bootstrap-draw hash, and saves new CSVs under `reruns/`. The old code is retained as historical evidence and does not reproduce final S6.

S9 is the final static summary, with its hidden audit detail retained. The corrected per-cell Qref summary is S3; similarly named old S3 audit files in historical directories need not refer to the same final table numbering.

## Environments and verification scope

requirements.txt specifies a practical pinned environment based on the original XJTU requirements and sklearn configuration; it is not claimed to be a historical lockfile. The original run logs record inconsistent NumPy/SciPy version strings and were not used to invent a verified historical environment. Packaging checks and S6/S7 verification versions are recorded in provenance/VALIDATION.json. Full primary ML retraining was not performed during packaging.

Core original-data pipeline order: CALCE QC; XJTU scan/validation/final_v1/final_v2; HUST scan/protocol validation/pilot/full run; cross-dataset endpoint summaries; threshold sensitivity; feature extraction; fixed fold design; primary benchmark; robustness; A8 deterministic reconstruction. Historical guards and no-overwrite protections remain in force.
