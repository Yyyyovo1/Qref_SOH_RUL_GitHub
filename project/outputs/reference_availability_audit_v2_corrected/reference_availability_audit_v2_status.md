# Reference Availability Audit V2 Corrected

## Corrected S3 frozen-authority gate

**PASS.** The corrected S3 was copied from the frozen XJTU Final V2 and HUST Final V1 Qref summaries; no Qref was recomputed from capacity data.

- Cells validated: 132 (XJTU 55; HUST 77)
- Missing cell keys: 0
- Duplicate cell keys: 0
- Qref mismatches: 0
- Source-cycle mismatches: 0
- Latest-source-cycle mismatches: 0
- Explicit XJTU Batch 2–6 regression cells checked: 47; all passed
- `B2_3C_battery-1` corrected first5 sources: `2|3|4|5|6`

## Recomputed reference-availability counts

`FROZEN_SOH_ELIGIBLE` uses retained observations in the frozen SOH label outputs. `EXISTING_ROBUSTNESS_POSTCYCLE5` uses the frozen primary A9 feature matrices and eligibility masks, then applies the existing rule `Raw_Cycle_Index > 5`. POSTCYCLE5 was not redefined as post-reference availability.

| Dataset | Reference | Population | Total | Before | At | After | At or after | Changed vs V1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XJTU | nominal | FROZEN_SOH_ELIGIBLE | 10427 | 0 | 0 | 10427 | 10427 | NO |
| XJTU | nominal | EXISTING_ROBUSTNESS_POSTCYCLE5 | 10327 | 0 | 0 | 10327 | 10327 | NO |
| XJTU | first_comparable | FROZEN_SOH_ELIGIBLE | 10427 | 0 | 55 | 10372 | 10427 | NO |
| XJTU | first_comparable | EXISTING_ROBUSTNESS_POSTCYCLE5 | 10327 | 0 | 24 | 10303 | 10327 | NO |
| XJTU | first5_comparable | FROZEN_SOH_ELIGIBLE | 10427 | 220 | 55 | 10152 | 10207 | NO |
| XJTU | first5_comparable | EXISTING_ROBUSTNESS_POSTCYCLE5 | 10327 | 120 | 55 | 10152 | 10207 | NO |
| HUST | nominal | FROZEN_SOH_ELIGIBLE | 146122 | 0 | 0 | 146122 | 146122 | NO |
| HUST | nominal | EXISTING_ROBUSTNESS_POSTCYCLE5 | 145737 | 0 | 0 | 145737 | 145737 | NO |
| HUST | first_comparable | FROZEN_SOH_ELIGIBLE | 146122 | 0 | 77 | 146045 | 146122 | NO |
| HUST | first_comparable | EXISTING_ROBUSTNESS_POSTCYCLE5 | 145737 | 0 | 0 | 145737 | 145737 | NO |
| HUST | first5_comparable | FROZEN_SOH_ELIGIBLE | 146122 | 308 | 77 | 145737 | 145814 | NO |
| HUST | first5_comparable | EXISTING_ROBUSTNESS_POSTCYCLE5 | 145737 | 0 | 0 | 145737 | 145737 | NO |

## V1 → V2 result

- Aggregate rows changed: 0 of 12.
- Cell × reference × population count rows changed: 0 of 792.
- The XJTU first5 values remain 10,427 / 220 / 55 / 10,152 for FROZEN_SOH_ELIGIBLE and 10,327 / 120 / 55 / 10,152 for POSTCYCLE5 (total / before / at / after).
- Manuscript text that reports the old 220 and 120 counts does not need numerical modification.
- The obsolete S3 must be replaced because it contains pre-hotfix Batch 2–6 first5 Qref/source-cycle values.
- Inputs for the old Table S9 should be replaced by `reference_availability_audit_v2.csv` and `aggregate_reference_availability_v2.csv`, even though the reported counts are unchanged, so provenance points to the corrected frozen-authority mapping.

## Prohibited analyses

No Qref, SOH target, EOL, RUL, fold, feature, model fit, or prediction was recomputed or modified.
