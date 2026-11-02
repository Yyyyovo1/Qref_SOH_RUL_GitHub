# Data sources

Raw datasets are not included. The study uses CALCE CS2_35–CS2_38 for quality-control development, the XJTU multi-batch data (55 cells) as the primary benchmark, and the HUST data (77 cells) for external replication. Obtain the raw data from the original publishers and respect their distribution terms.

| Dataset | Data location | Reference publication |
| --- | --- | --- |
| CALCE CS2 | https://calce.umd.edu/data | He et al. (2011), https://doi.org/10.1016/j.jpowsour.2011.08.040 |
| XJTU (55 NCM cells) | https://zenodo.org/records/10963339 | Wang et al. (2024), *Nat. Commun.* 15, 4332, https://doi.org/10.1038/s41467-024-48779-z; Wang et al. (2024), *J. Energy Storage* 77, 109884, https://doi.org/10.1016/j.est.2023.109884 |
| HUST (77 LFP/graphite cells) | https://doi.org/10.17632/nsc7hnsg4s (Mendeley Data, CC BY 4.0) | Ma et al. (2022), *Energy Environ. Sci.* 15, 4083–4094, https://doi.org/10.1039/D2EE01676A |

Check each data location for the current licence before reuse; the entries above record the sources used in this study and are not a statement of current availability.

The source project preserves acquisition and raw-file identity in the scan manifests, processing-method READMEs, and derived tables. Full dataset reference entries copied from the earlier manuscript are in `DATA_PAPER_REFERENCES.txt`. Expected raw-data folder names for the pipelines (`Batch-1/` … `Batch-6/`, `HUST 77/`, CALCE cell folders) are described in `REPRODUCE.md`.
