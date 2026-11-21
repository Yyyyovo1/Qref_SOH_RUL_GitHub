# Manuscript-to-file map (Journal of Energy Storage manuscript, version 2.0.0)

J. Yang, *Reference capacity preserves method rankings but not effect sizes in lithium-ion battery health benchmarks: a preregistered multi-dataset study*. Paths are relative to the repository root. Scripts in `manuscript/jes_v1/scripts/` read frozen outputs only and are run from the repository root.

## Figures

| Item | Script | Input (frozen) | Output |
| --- | --- | --- | --- |
| Fig. 1 Study design | `make_fig1.py` | none (schematic) | `manuscript/jes_v1/figures/Figure_1.pdf/.svg/.png` |
| Fig. 2 Deviation of measured references from nominal | `make_fig2_fig3.py` | `manuscript/jes_v1/results_data/A_per_cell_qref.csv` (built by `results_tables.py` from `supplementary/csv/S3_PerCell_Qref.csv`, `project/outputs/mit_124_final_v1/mit_qref_summary.csv`, `project/outputs/tju_final_v2/tju_qref_summary.csv`) | `Figure_2.pdf/.png` |
| Fig. 3 Labelled share versus θ, strict and tolerance rules | `make_fig2_fig3.py` | `manuscript/jes_v1/results_data/C_labelled_share_by_theta.csv` (from `project/outputs/protocol_validation_v1/trapD_labels.csv`, `project/outputs/followup_analyses_v1/5_observable_windows.csv`, `reruns/tju_label_structure/2_labeled_share_by_theta.csv`) | `Figure_3.pdf/.png` |
| Fig. 4 SOH benchmark in capacity units | `make_fig4.py` | `project/outputs/soh_primary_benchmark_v1/soh_primary_cell_metrics.csv`, `soh_primary_summary_metrics.csv` | `Figure_4.pdf/.svg/.png`, `Figure_4_data.csv` |
| Fig. 5 Change in method difference, nominal vs first_comparable, MIT (v4 Exp. 1/2) | `make_fig6_fig7.py` | `project/outputs/rul_v4_core_claim/exp1_pair_ci.csv`, `exp2_equivalence_and_power.csv` | `Figure_5.pdf/.png`, `Figure_5_data.csv` |
| Fig. 6 Severson et al. models across θ, MIT (v4 Exp. 1) | `make_fig6_fig7.py` | as Fig. 5 | `Figure_6.pdf/.png`, `Figure_6_data.csv` |
| Fig. S1 As Fig. 5 for nominal vs first5_comparable | `make_figS1.py` | as Fig. 5 | `Figure_S1.pdf/.png`, `Figure_S1_data.csv` |

## Tables and sections

| Item | Source file(s) |
| --- | --- |
| Table 1 Datasets (cells, nominal capacity, retained observations, last retained capacity) | `project/outputs/final_results_synthesis_v1/table_dataset_overview.csv`; `project/outputs/mit_124_final_v1/mit_qref_summary.csv`; `project/outputs/tju_final_v2/tju_qref_summary.csv`; `project/outputs/followup_analyses_v1/4_position_distribution_by_dataset.csv` |
| Section 2.3 QC rules and exclusions (MIT, TJU) | `project/outputs/mit_124_final_v1/mit_cycle_qc.csv`; `project/outputs/tju_final_v2/tju_cycle_qc.csv`, `tju_qc_rules.json`; Supplementary Note S4 |
| Table 2 Sequence of RUL tests; Supplementary Note S3 | Protocols `project/rul_ranking_pilot/PROTOCOL.md`, `PROTOCOL_v2.md`, `PROTOCOL_v3_TJU.md`, `PROTOCOL_v3_TJU_AMENDMENT_01.md`, `project/v4_core_claim/PROTOCOL_v4.md`, `AMENDMENT_01.md`, `AMENDMENT_02.md` and their `*_RECORD.md` files; reports `reruns/rul_pilot_report.md`, `reruns/rul_pilot_v2_report.md`, `reruns/protocol_validation_report.md`, `reruns/rul_v3_tju_report.md`, `reruns/rul_v4_report.md`; preregistration archive doi:10.5281/zenodo.24222903 |
| Section 3.1 Reference deviations | `manuscript/jes_v1/results_data/A_qref_deviation_summary.csv`; Supplementary Tables S3 and S11 |
| Table 3 Endpoint labels at θ = 0.80; Supplementary Table S10 | `manuscript/jes_v1/results_data/B_endpoint_classes_075_080_085.csv` (from `project/outputs/protocol_validation_v1/trapD_labels.csv` and `reruns/tju_label_structure/2_labels_per_cell_setting.csv`) |
| Section 3.2 Labelled-share windows; Supplementary Table S13 | `manuscript/jes_v1/results_data/C_windows_90pct.csv` |
| Section 3.3 SOH benchmark; Supplementary Table S12a | `project/outputs/soh_primary_benchmark_v1/`; `reruns/scaling_bootstrap/model_pair_ci.csv`, `rank_frequency.csv`, `reversal_assessment.csv`; retrain check `reruns/retrain_report.md` |
| Section 3.4 Information condition (OWN vs TRAIN); Supplementary Table S12b | `project/outputs/protocol_validation_v1/trapB_settings.csv`; `reruns/scaling_bootstrap/measured_minus_nominal_ci.csv` |
| Section 3.5 RUL tests v1–v3 | `project/outputs/rul_ranking_pilot_v1/`, `project/outputs/rul_ranking_pilot_v2/`, `project/outputs/protocol_validation_v1/`, `project/outputs/rul_ranking_v3_tju/`; the reports listed for Table 2 |
| Section 3.6 and Table 4 v4 verdicts (H1 undecided, H2 refuted, H3 held) | `project/outputs/rul_v4_core_claim/decision.json`, `exp1_h1_units.csv`, `exp1_pair_ci.csv`, `exp1_reversals.csv`, `exp2_equivalence_and_power.csv`, `exp2_detection_summary.csv`, `exp3_comparisons.csv`, `exp3_pair_ci.csv`, `sanity_checks.csv`; formal-run consoles `reruns/v4/formal_console/` |
| Supplementary Table S14 v4 equivalence classification (247 rows) | `project/outputs/rul_v4_core_claim/exp2_equivalence_and_power.csv` |
| Supplementary Table S15 Severson et al. models at θ = 0.80 vs the published values | `project/outputs/rul_v4_core_claim/exp1_common_practice_theta080.csv` |
| v4 preparation (feature check against Attia et al. 2021, MIT windows) | `reruns/v4_prep/V4_PREP_REPORT.md` and the files listed in `project/v4_core_claim/PREREG_RECORD.md` |
| Supplementary Tables S1–S15 workbook | `manuscript/jes_v1/scripts/make_supplementary_tables.py` → `manuscript/jes_v1/supplementary/Supplementary_Tables_S1-S15.xlsx` (S1–S9 copied unchanged from `supplementary/original/Supplementary_Tables_S1-S9.xlsx`) |
| Supplementary Notes | `manuscript/jes_v1/supplementary/Supplementary_Notes_S1-S2.pdf` (unchanged from version 1.0.0) and `Supplementary_Notes_S3-S4_and_Fig_S1.pdf` |
