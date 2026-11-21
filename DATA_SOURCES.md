# Data sources

Raw datasets are not included. The study uses CALCE CS2_35–CS2_38 for quality-control development, the XJTU multi-batch data (55 cells) and the HUST data (77 cells) for the label analysis and SOH benchmark, and, from version 2.0.0, the MIT/Severson et al. data (124 cells) and the TJU data (datasets 1 and 2: 66 NCA and 55 NCM cells) for the label analysis and the RUL method-comparison tests. Obtain the raw data from the original publishers and respect their distribution terms.

| Dataset | Data location | Reference publication |
| --- | --- | --- |
| CALCE CS2 | https://calce.umd.edu/data | He et al. (2011), https://doi.org/10.1016/j.jpowsour.2011.08.040 |
| XJTU (55 NCM cells) | https://zenodo.org/records/10963339 | Wang et al. (2024), *Nat. Commun.* 15, 4332, https://doi.org/10.1038/s41467-024-48779-z; Wang et al. (2024), *J. Energy Storage* 77, 109884, https://doi.org/10.1016/j.est.2023.109884 |
| HUST (77 LFP/graphite cells) | https://doi.org/10.17632/nsc7hnsg4s (Mendeley Data, CC BY 4.0) | Ma et al. (2022), *Energy Environ. Sci.* 15, 4083–4094, https://doi.org/10.1039/D2EE01676A |
| MIT/Severson et al. (124 LFP/graphite cells; batches 2017-05-12, 2017-06-30, 2018-04-12) | https://data.matr.io/1/projects/5c48dd2bc625d700019f3204 (CC BY 4.0 as stated on the project and batch pages, checked 2026-11-06; files downloaded 2026-11-07) | Severson et al. (2019), *Nat. Energy* 4, 383–391, https://doi.org/10.1038/s41560-019-0356-8 |
| TJU (datasets 1 and 2: 66 NCA and 55 NCM cells) | https://doi.org/10.5281/zenodo.6405084 (version 2 of concept record 10.5281/zenodo.6379165, CC BY 4.0) | Zhu et al. (2022), *Nat. Commun.* 13, 2261, https://doi.org/10.1038/s41467-022-29837-w |
| Attia et al. (2021) per-cell arrays (verification only, v4 preparation) | GitHub repository petermattia/revisit-severson-et-al, commit 3a77c7d513126e921896315038d0f8f2645e440d (MIT License; downloaded 2026-11-14) | Attia et al. (2021), *J. Electrochem. Soc.* 168, 090547, https://doi.org/10.1149/1945-7111/ac2704 |

Check each data location for the current licence before reuse; the entries above record the sources used in this study and are not a statement of current availability.

The source project preserves acquisition and raw-file identity in the scan manifests, processing-method READMEs and derived tables (for MIT: `project/outputs/mit_124_final_v1/mit_file_manifest.csv` with file sizes and SHA-256). Full dataset reference entries copied from the earlier manuscript are in `DATA_PAPER_REFERENCES.txt`. Expected raw-data folder names for the XJTU, HUST and CALCE pipelines are described in `REPRODUCE.md`; the MIT pipeline expects the three `*_batchdata_updated_struct_errorcorrect.mat` files in `raw_data/MIT/`, and the TJU pipeline the raw CSV folders of datasets 1 and 2 in `raw_data/TJU/`.
