# Reference capacity, end-of-life threshold and endpoint rule in lithium-ion battery SOH/RUL benchmarks

Code, processed data, frozen outputs and supplementary materials for a study of how the reference capacity (Q_ref) used to define capacity-based state of health (SOH), the end-of-life (EOL) threshold fraction θ and the endpoint rule change EOL labels, censoring, SOH benchmark errors and the conclusions of remaining-useful-life (RUL) method comparisons.

Datasets: CALCE CS2 (quality-control development), XJTU (55 NCM cells), HUST (77 LFP/graphite cells), MIT/Severson et al. (124 LFP/graphite cells) and TJU (66 NCA and 55 NCM cells).

## Versions

| Version | Associated manuscript | What it adds |
| --- | --- | --- |
| 2.0.0 (this version) | J. Yang, *Reference capacity preserves method rankings but not effect sizes in lithium-ion battery health benchmarks: a preregistered multi-dataset study*, prepared for the *Journal of Energy Storage* (under review; citation details will be added after publication) | MIT and TJU pipelines and frozen labels; RUL method-comparison tests v1–v4 with protocols, amendments, reports and frozen outputs; SOH model-pair bootstrap; scripts and data for Figures 1–6; Supplementary Tables S1–S15 and Notes S1–S4. See [PAPER_MAP_JES.md](PAPER_MAP_JES.md). |
| 1.0.0, doi:[10.5281/zenodo.23889698](https://doi.org/10.5281/zenodo.23889698) | J. Yang, *Reference Capacity as a Hidden Threshold*, technical brief prepared for the *International Journal of Prognostics and Health Management* | XJTU/HUST labels and SOH benchmark; Supplementary Tables S1–S9 and Notes S1–S2. See [PAPER_MAP.md](PAPER_MAP.md). |

Everything in version 1.0.0 is retained byte for byte except the documentation files listed in `provenance/RELEASE_NOTES.md`.

The preregistration of test v4 (protocol, preparatory outputs and amendments 01 and 02) is archived separately: concept doi:[10.5281/zenodo.24222903](https://doi.org/10.5281/zenodo.24222903). The archived files are also in this repository; their hashes are listed in `project/v4_core_claim/PREREG_RECORD.md` and the two `AMENDMENT_0*_RECORD.md` files.

## Repository layout

| Path | Content |
| --- | --- |
| `project/` | Analysis code, processed data, parameters and frozen outputs. The relative layout is preserved because scripts resolve inputs relative to it. |
| `project/rul_ranking_pilot/`, `project/v4_core_claim/` | Protocols and amendments of RUL tests v1–v3 and v4 (see Table 2 of the manuscript and Supplementary Note S3). |
| `project/outputs/mit_124_final_v1/`, `project/outputs/tju_final_v2/` | Frozen MIT and TJU labels, reference capacities and cycle QC (each with `DATA_LICENSE.md`). |
| `project/outputs/rul_ranking_pilot_v1/`, `rul_ranking_pilot_v2/`, `rul_ranking_v3_tju/`, `rul_v4_core_claim/` | Frozen outputs of RUL tests v1–v4. |
| `reruns/` | Test reports (`rul_pilot_report.md`, `rul_pilot_v2_report.md`, `protocol_validation_report.md`, `rul_v3_tju_report.md`, `rul_v4_report.md`), v4 preparatory outputs, TJU label structure, formal-run consoles, the SOH retrain check and the SOH model-pair bootstrap (`reruns/scaling_bootstrap/`). |
| `manuscript/jes_v1/` | Scripts, derived tables and output files for Figures 1–6 and Fig. S1, and Supplementary Tables S1–S15 and Notes S1–S4 of the JES manuscript. The manuscript text itself is not included. |
| `supplementary/` | Supplementary Tables S1–S9 and Notes S1–S2 of version 1.0.0 (unchanged). |
| `figures/` | Figures of the version 1.0.0 manuscripts (unchanged). |
| `tools/` | Release verification, data restoration and supplementary-table checks. |
| `provenance/` | Source manifest, SHA-256 checksums, conversion records, validation results and release notes. |

## Quick start

Python 3.12. From the repository root:

```text
python tools/verify_release.py          # check every packaged file against provenance/SHA256SUMS.csv
python tools/restore_data.py            # rebuild large processed files from lossless gzip parts
python -m pip install -r requirements.txt
python tools/check_supplement.py        # compare frozen data with Supplementary Tables S1–S9
python tools/reproduce_s6_s7.py         # reproduce the bootstrap tables S6 and S7
```

JES manuscript figures and tables (all read frozen outputs only):

```text
python manuscript/jes_v1/scripts/results_tables.py            # derived tables in manuscript/jes_v1/results_data/
python manuscript/jes_v1/scripts/make_fig1.py                 # Figure 1 (study design)
python manuscript/jes_v1/scripts/make_fig2_fig3.py            # Figures 2 and 3
python manuscript/jes_v1/scripts/make_fig4.py                 # Figure 4
python manuscript/jes_v1/scripts/make_fig6_fig7.py            # Figures 5 and 6 (file name kept from the draft numbering)
python manuscript/jes_v1/scripts/make_figS1.py                # Figure S1
python manuscript/jes_v1/scripts/make_supplementary_tables.py # Supplementary Tables S1–S15
```

SOH model-pair bootstrap (Table S12a): `python tools/prepare_scaling_bootstrap_inputs.py` copies the restored frozen predictions to `reruns/primary_benchmark/`; then run `reruns/scaling_bootstrap/scaling_bootstrap.py` in a copy of that folder without its output files (the script refuses to overwrite). With the frozen predictions the published values are reproduced to within 3e-15 Ah, with identical resampling hashes and identical rank and reversal tables (see `provenance/RELEASE_NOTES.md`).

Processed files larger than 20 MiB are stored as lossless gzip parts. The restore tool verifies the SHA-256 of every part and every restored file, skips files that already exist with identical content, and never overwrites different content. Full reproduction instructions and limitations are in [REPRODUCE.md](REPRODUCE.md); the data catalogue of version 1.0.0 is [provenance/DATA_CATALOG.csv](provenance/DATA_CATALOG.csv).

## Data versions and conventions

- Labels, reference cycles, missingness and censoring status are retained for the `nominal`, `first_comparable` and `first5_comparable` references (`early_median5_comparable` is the frozen data key of `first5_comparable`). XJTU also has `initial_test`.
- XJTU `final_v2` and TJU `final_v2` are authoritative; the `final_v1` folders are kept as upstream history.
- Exact, interval-censored and right-censored endpoints are never substituted for one another, and no censored endpoint is given a point EOL or RUL value.
- Frozen outputs and committed protocols were not modified after they were written. Deviations are recorded as dated amendments or corrections (for example `reruns/protocol_validation_followup/CORRECTION_01_MIT_batch2_stop_level.md`) and listed in Supplementary Note S3.
- Supplementary Tables S1–S9 and Notes S1–S2 were prepared for an earlier full-length version; table and figure numbers cited inside them refer to that version. S10–S15 and Notes S3–S4 belong to the JES manuscript.

## What is not included

Raw measurement files, private correspondence, manuscript drafts, third-party full-text articles, virtual environments, dependency folders, caches and complete run-log directories are not included. The 395 MB SOH predictions of the 2026-11-06 retrain check are not included because they agree with the frozen predictions to within 2e-14 (`reruns/retrain_report.md`).

Known limitation: the Node.js validation scripts (`*.mjs`) in `project/` import a spreadsheet library from a local runtime path that is not part of this repository, so they will not run as-is. They are kept as historical evidence only.

## Data sources

Raw datasets are not redistributed. Obtain them from the original publishers and follow their terms (see [DATA_SOURCES.md](DATA_SOURCES.md)):

- CALCE CS2 cells: https://calce.umd.edu/data
- XJTU (55 cells): https://zenodo.org/records/10963339 — Wang et al., *Nature Communications* 15, 4332 (2024)
- HUST (77 cells): https://doi.org/10.17632/nsc7hnsg4s (CC BY 4.0) — Ma et al., *Energy & Environmental Science* 15, 4083–4094 (2022)
- MIT (124 cells): https://data.matr.io/1/projects/5c48dd2bc625d700019f3204 (CC BY 4.0) — Severson et al., *Nature Energy* 4, 383–391 (2019)
- TJU (NCA and NCM cells): https://doi.org/10.5281/zenodo.6405084 (CC BY 4.0) — Zhu et al., *Nature Communications* 13, 2261 (2022)

## License

Released under the [MIT License](LICENSE). The license covers the author's code, processed data, figures and supplementary materials. Processed data derived from third-party datasets remain subject to the attribution requirements of those datasets; please cite the original data sources. See [LICENSE_STATUS.md](LICENSE_STATUS.md).

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the original dataset publications listed above.

## 中文简介

本仓库为参考容量（Q_ref）、寿命终止阈值和终点规则如何影响锂离子电池 SOH/RUL 标签与方法比较结论的研究提供代码、处理后数据、冻结输出和补充材料。2.0.0 版对应投稿 *Journal of Energy Storage* 的论文，新增 MIT、TJU 数据处理、RUL 方法比较检验 v1–v4（方案、修订、报告和冻结输出）、图 1–6 的脚本和数据、补充表 S1–S15 与补充说明 S1–S4。原始数据未包含，请按上方链接获取。以 MIT 许可证发布。
