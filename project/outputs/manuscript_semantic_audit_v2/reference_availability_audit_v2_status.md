# Reference Availability Audit V2 — validation gate status

## Outcome

**STOPPED BEFORE OBSERVATION COUNTING.** The required cell-by-cell consistency gate against Supplementary Table S3 failed. No `reference_availability_audit_v2.csv` was generated, and no FROZEN_SOH_ELIGIBLE or POSTCYCLE5 counts were recomputed.

## Frozen authority

- XJTU: `.\outputs\xjtu_55_final_v2\xjtu_55_qref_summary.csv`
- HUST: `.\outputs\hust_77_final_v1\hust_77_qref_summary.csv`
- These frozen Qref summaries were treated as the sole Qref-value and source-cycle authority.

## S3 validation

- Keys checked: 132 cells (XJTU 55; HUST 77).
- Fully matching cells: 85.
- Failing cells: 47.
- `First5_Comparable_Ah` mismatches: 47.
- `First5_Source_Cycles` mismatches: 47.
- All failures are XJTU Batch 2–6. XJTU Batch 1 and all 77 HUST cells pass.
- Example: `B2_3C_battery-1` is `2|11|101|102|103` in S3 but `2|3|4|5|6` in frozen XJTU Final V2.

The mismatch pattern is consistent with the pre-hotfix string-sorting artifact in S3, whereas the frozen XJTU Final V2 summary contains the numeric-sort-corrected source cycles.

## V1 → V2 audit comparison

- V1 rows checked: 792.
- V1 Qref mismatches versus frozen authority: 0.
- V1 source-cycle mismatches versus frozen authority: 0.
- V1 latest-source-cycle mismatches versus frozen authority: 0.
- Count deltas are intentionally blank because the S3 gate failed before the counting stage.

## Required resolution

Either (1) create a separately versioned Supplementary Table S3 sourced from the frozen XJTU Final V2/HUST Final V1 Qref summaries, then rerun this audit, or (2) explicitly authorize proceeding despite S3 inconsistency. This run did not alter S3, frozen summaries, V1 audit, SOH targets, Qref, EOL, model files, or folds.
