# Manuscript-to-file map (IJPHM technical brief)

Paths are relative to the repository root. Numbers in the brief can be checked against the files below.

| Item in the technical brief | Source file(s) |
| --- | --- |
| Section 2.1: retained observation counts per dataset | `supplementary/csv/S1_Dataset_Inventory.csv`; `project/outputs/final_results_synthesis_v1/table_dataset_overview.csv` |
| Section 2.2: reference definitions and availability (source cycles, retrospective first5) | `supplementary/csv/S3_PerCell_Qref.csv`; `supplementary/csv/S9_Reference_Availability.csv` |
| Section 3.1: per-cell reference differences from nominal | `supplementary/csv/S3_PerCell_Qref.csv`; `project/outputs/cross_dataset_summary_v1/cross_dataset_qref_summary.csv` |
| Figure 1: equivalent threshold θ′ on the nominal scale | `figures/IJPHM_TB/make_fig1_equivalent_threshold.py` (input: `supplementary/csv/S3_PerCell_Qref.csv`) → `figures/IJPHM_TB/Figure1_equivalent_threshold.png/.pdf` |
| Table 1: endpoint classes at θ = 0.75, 0.80, 0.85 (incl. XJTU initial-test note) | `supplementary/csv/S2_EOL_Censoring.csv`; `project/outputs/eol_threshold_robustness_v1/eol_threshold_status_summary.csv` |
| Section 3.2: exact point-RUL eligibility | `project/outputs/eol_threshold_robustness_v1/eol_threshold_rul_availability.csv` |
| Section 3.2: absolute-threshold shifts, exact-pair ΔN_EOL, lower bounds on earlier endpoints | `project/outputs/eol_threshold_robustness_v1/eol_threshold_transition_summary.csv`; `project/outputs/eol_threshold_robustness_v1/eol_threshold_effect_bounds.csv` |
| Section 3.2: interval-censoring gaps and later rebound counts | `supplementary/original/Supplementary_Notes_S1-S2.pdf` (Note S2.4) |
| Section 3.3: primary SOH benchmark (MAE in Ah and percentage points, rankings) | `supplementary/csv/S4_Primary_A9_Metrics.csv` |
| Section 3.3: sensitivity runs (A8, B10, POSTCYCLE5) | `supplementary/csv/S5_Robustness_Full.csv` |
| Section 3.3: XJTU A8 cell-bootstrap check | `supplementary/csv/S6_A8_Corrected_CI.csv`, `supplementary/csv/S7_A8_RankFlip.csv`; reproduce with `tools/reproduce_s6_s7.py` |
| Section 2.4: model, preprocessing, fold, and seed settings | `supplementary/csv/S8_ML_Config.csv` |

Figures in `figures/Figure1_two_branch/` and `figures/Figures_2-6_FINAL/` belong to the earlier full-length version and are not used in the technical brief.
