# Release notes

Prepared locally on 2026-11-02. Source artifacts: 566. Byte-identical copies: 523. Files with documented path/configuration/metadata changes: 43.

Scientific data and provisional status fields are retained. Source and release hashes are separate. Historical manifests remain historical. New helper scripts and README files are release additions, not evidence of a historical protocol freeze.

Internal run/access logs, raw data, local dependency installations, temporary files, private manuscripts and third-party articles are excluded. Selected original tests passed; full raw-data processing and ML training were not rerun. VALIDATION.json records the scope. Tests were run on the restored data where required; upload storage remains compressed.

## Documentation and licence update (2026-11-02)

Added `LICENSE` (MIT), `CITATION.cff`, `PAPER_MAP.md`, and `figures/IJPHM_TB/` (script and output for Figure 1 of the IJPHM technical brief, computed from `supplementary/csv/S3_PerCell_Qref.csv`). Rewrote `README.md` in English and updated `LICENSE_STATUS.md` and `DATA_SOURCES.md` (direct data locations for XJTU and HUST). No data, code, supplementary table, or historical output was changed. `provenance/SHA256SUMS.csv` was regenerated to include these files.

## Version 2.0.0 (prepared 2026-11-17)

Built from commit `ffdcb58` of branch `rul-pilot` (GitHub: Yyyyovo1/Qref_SOH_RUL_GitHub) with `git archive`, so every tracked file is byte-identical to that commit. Version 1.0.0 corresponds to commit `47c6f95` (tag `zenodo-archive-2026-11-02`); the 32 commits in between add the MIT and TJU pipelines, protocol validation, RUL tests v1–v4 with their protocols, amendments, reports and frozen outputs, and the v4 preregistration records.

Added from the working tree (not tracked in git because `reruns/` and `manuscript/` are excluded by `.gitignore`):

- `reruns/scaling_bootstrap/` (script, outputs and report of the SOH model-pair bootstrap, Supplementary Table S12a), unchanged.
- `reruns/retrain_report.md`, `reruns/primary_benchmark_comparison.csv`, `reruns/primary_benchmark_metric_diffs.csv` (2026-11-06 SOH retrain check). In `retrain_report.md` the local path of the virtual environment was replaced by `.venv`; nothing else was changed. The 395 MB retrain predictions are not included (see below).
- `manuscript/jes_v1/`: figure scripts, derived tables (`results_data/`), figure files and data, Supplementary Tables S1–S15 and Supplementary Notes S1–S4 of the JES manuscript. Manuscript drafts and the scripts that build them are not included.

Documentation changed in this version: `README.md`, `CITATION.cff`, `DATA_SOURCES.md`, `LICENSE_STATUS.md` (MIT and TJU data added), `requirements.txt` (`openpyxl==3.1.5` added; needed by `make_supplementary_tables.py`) and this file. Added: `PAPER_MAP_JES.md`, `tools/prepare_scaling_bootstrap_inputs.py`, `provenance/v1.0.0/SHA256SUMS.csv` (the integrity manifest of version 1.0.0, kept unchanged). `provenance/SHA256SUMS.csv` was regenerated and now lists every file of version 2.0.0 except itself. `provenance/SOURCE_MANIFEST.csv`, `DATA_CATALOG.csv` and the other provenance files describe version 1.0.0 and were not changed.

Checks run on the packaged copy before upload:

- `tools/verify_release.py` passed for all files of the new manifest, and every file listed in `provenance/v1.0.0/SHA256SUMS.csv` except the changed documentation files has the same SHA-256 as in version 1.0.0.
- `tools/restore_data.py` restored all large files from their gzip parts with matching SHA-256.
- `manuscript/jes_v1/scripts/results_tables.py` and the figure scripts, run on the packaged copy, reproduced every derived CSV (`results_data/*.csv`, `figures/*_data.csv`) byte for byte.
- `reruns/scaling_bootstrap/scaling_bootstrap.py`, run on the restored frozen predictions (the published run used the 2026-11-06 retrain predictions, which agree with the frozen ones to within 2e-14), reproduced all five output tables: identical row and text content, identical resampling hashes, numerical differences ≤ 3e-15 Ah.
- Every file path cited in `PAPER_MAP_JES.md` exists in the package.
