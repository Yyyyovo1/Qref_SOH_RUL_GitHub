# Reference capacity as a hidden threshold: SOH/RUL label-definition shift in public lithium-ion battery datasets

Code, processed data, and supplementary tables for a study of how the reference capacity (Q_ref) used to define capacity-based state of health (SOH) changes end-of-life (EOL) thresholds, endpoint censoring, exact point-RUL label availability, and SOH benchmark errors. Datasets: CALCE CS2 (quality-control development), XJTU (55 NCM cells, primary benchmark), and HUST (77 LFP/graphite cells, external replication).

**Associated manuscript.** J. Yang, *Reference Capacity as a Hidden Threshold: How State-of-Health Normalization Shifts End-of-Life Labels in Public Lithium-Ion Battery Datasets*, technical brief prepared for the *International Journal of Prognostics and Health Management*. Citation details will be added after publication. See [PAPER_MAP.md](PAPER_MAP.md) for where each figure, table, and number in the brief comes from.

## Repository layout

| Path | Content |
| --- | --- |
| `project/` | Analysis code, processed data, parameters, and frozen historical outputs. The relative directory layout is preserved because scripts resolve inputs relative to it. |
| `supplementary/original/` | Supplementary Tables S1–S9 (`.xlsx`) and Supplementary Notes S1–S2 (`.docx`/`.pdf`). Only the workbook's `absPath` metadata (a local save location) was removed; cells, formulas, formatting, and the hidden audit sheet are unchanged. |
| `supplementary/csv/` | CSV exports of the stored values in S1–S9 (no recalculation). |
| `figures/IJPHM_TB/` | Script and output for Figure 1 of the technical brief. |
| `figures/Figure1_two_branch/`, `figures/Figures_2-6_FINAL/` | Figures of the earlier full-length version of the manuscript (not used in the technical brief; kept for traceability). |
| `tools/` | Release verification, data restoration, and supplementary-table checks. |
| `provenance/` | Source manifest, SHA-256 checksums, conversion records, and validation results. |

## Quick start

Python 3.12. From the repository root:

```text
python tools/verify_release.py          # check every packaged file against provenance/SHA256SUMS.csv
python tools/restore_data.py            # rebuild large processed files from lossless gzip parts
python -m pip install -r requirements.txt
python tools/check_supplement.py        # compare frozen data with Supplementary Tables S1–S9
python tools/reproduce_s6_s7.py         # reproduce the bootstrap tables S6 and S7
python figures/IJPHM_TB/make_fig1_equivalent_threshold.py
```

Processed files larger than 20 MiB are stored as lossless gzip parts. The restore tool verifies the SHA-256 of every part and every restored file, skips files that already exist with identical content, and never overwrites different content. Restored large files are excluded from version control by `.gitignore`. Verification outputs are written to `reruns/`.

Full reproduction instructions and limitations are in [REPRODUCE.md](REPRODUCE.md); the data catalogue is [provenance/DATA_CATALOG.csv](provenance/DATA_CATALOG.csv).

## Data versions and conventions

- Labels, reference cycles, missingness, and censoring status are retained for the `nominal`, `first_comparable`, and `first5_comparable` references (`early_median5_comparable` is the frozen data key of `first5_comparable`).
- XJTU `final_v2` is authoritative; `final_v1` is kept as upstream history.
- Exact, interval-censored, and right-censored endpoints are never substituted for one another, and no censored endpoint is given a point EOL or RUL value.
- The primary feature set A9 and the sensitivity analyses A8, B10, and POSTCYCLE5 keep their own file names.
- The historical script in `project/xjtu_a8_rankflip_repro/` computes the median of paired differences, whereas final Table S6 uses the difference of medians. Use `tools/reproduce_s6_s7.py` to reproduce S6 and S7.
- Supplementary Tables S1–S9 and Notes S1–S2 were prepared for the earlier full-length version; table and figure numbers cited inside them refer to that version.

## What was and was not changed during packaging

No raw-data analysis or model training was re-run, and no scientific value was changed. Text files that contained local absolute paths were redacted; original and release hashes of every file are listed in `provenance/SOURCE_MANIFEST.csv`. Historical freeze manifests keep their original values and are not the integrity manifest of this release; use `provenance/SHA256SUMS.csv`.

The repository does not contain raw measurement files, private correspondence, manuscript drafts, third-party full-text articles, virtual environments, dependency folders, caches, or complete run-log directories.

Known limitation: the Node.js validation scripts (`*.mjs`) in `project/` import a spreadsheet library from a local runtime path that is not part of this repository, so they will not run as-is. They are kept as historical evidence only; the Python tools above cover the same checks.

## Data sources

Raw datasets are not redistributed. Obtain them from the original publishers and follow their terms (see [DATA_SOURCES.md](DATA_SOURCES.md)):

- CALCE CS2 cells: https://calce.umd.edu/data
- XJTU (55 cells): Zenodo record https://zenodo.org/records/10963339 — Wang et al., *Nature Communications* 15, 4332 (2024), https://doi.org/10.1038/s41467-024-48779-z
- HUST (77 cells): Mendeley Data https://doi.org/10.17632/nsc7hnsg4s (CC BY 4.0) — Ma et al., *Energy & Environmental Science* 15, 4083–4094 (2022), https://doi.org/10.1039/D2EE01676A

## License

Released under the [MIT License](LICENSE). The license covers the author's code, processed data, figures, and supplementary materials in this repository. Processed data derived from third-party datasets remain subject to the attribution requirements of those datasets; please cite the original data sources. See [LICENSE_STATUS.md](LICENSE_STATUS.md).

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the original dataset publications listed above.

## 中文简介

本仓库为参考容量（Q_ref）定义导致锂离子电池 SOH/RUL 标签偏移研究的代码、处理后数据与补充表，对应投稿 IJPHM 的技术简报。原始数据未包含，请按上方链接获取。以 MIT 许可证发布。
