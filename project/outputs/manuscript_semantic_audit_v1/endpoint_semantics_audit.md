# Endpoint semantics audit: frozen capacity, Qref, EOL, and RUL artifacts

- Audit timestamp (UTC): 2026-09-30T13:04:38.988912+00:00
- Project root: `.`
- Mode: read-only audit of frozen sources; no raw-data processing, threshold change, interpolation, endpoint repair, model training, fold change, or overwrite.
- New outputs are confined to `outputs/manuscript_semantic_audit_v1`.

## Concise determination

1. Frozen endpoint counts reproduce the manuscript tables exactly: zero compared-count mismatches, including Qref availability, in both the 21-row threshold table and 6-row 80% table.
2. The executable endpoint is the first retained eligible observation at or below the unrounded threshold. In the 75/80/85% classifier, `EXACT_OBSERVED` is restricted to adjacent positions on the declared endpoint index: XJTU dense batches 1/2 with cumulative-aging gap <= 1; HUST raw-cycle gap <= 1. It is exact on that observed grid, not a generic assertion that an unobserved physical crossing time was measured.
3. `INTERVAL_CENSORED` uses the immediately preceding retained observation and first retained below-threshold observation. Treating that as the latent cycle-level first-passage interval requires ruling out an earlier unobserved below-threshold excursion followed by rebound.
4. `RIGHT_CENSORED` means no retained eligible observation met the threshold. For sparse diagnostics, latent cycle-level right-censoring requires the same no-missed-excursion/persistence assumption.
5. The current manuscript already says “observed capacity sequence,” “retained observation,” and “first observed qualifying crossing.” Clarification is sufficient if the estimand remains observation-grid first crossing. A separate versioned analysis is required if the estimand is an unobserved cycle-level or persistent/irreversible first passage.

## VERIFIED FROM CODE

### Actual classifiers and frozen-version trace

| Dataset/component | Executed path and function | SHA-256 | Supporting lines | Frozen trace |
| --- | --- | --- | --- | --- |
| XJTU Final V2, batches 2-6 | `xjtu_multibatch/analysis.py::detect_eol`, invoked by `_new_eol` | `e428cef82e672b165af5c5a91e466aec90358f1ed959b87e555edae0fc646949` | analysis.py 190-288, 714-785; final_v2.py 208-291 | Reads pre-hotfix Final V1 plus approved hotfix/manual layer; writes Final V2 EOL |
| XJTU Final V2, Batch 1 preserved | `xjtu_batch1_pipeline/.../sensitivity.py::_detect_eol` | `db8b46c3e6e331b2d30e20076122c8722784d64dd4b6e245c435199b15f37a75` | sensitivity.py 165-185; final_v2.py 235-251 | Frozen Batch-1 output preserved through Final V1 into Final V2 |
| HUST Final V1 | `hust_repro/pilot.py::detect_eol`, invoked by `hust_repro/full_run.py` | `515a7ef5cb7f6c54c9f4e5ee8faaed9199d422f6495910cff9c310bfa2106e8f` | pilot.py 287-391; full_run.py 1446-1471 | Writes HUST Final V1 EOL |
| 75/80/85% manuscript source | `detect_xjtu_eol` and `detect_hust_eol` | `880a12834a427b85c3409d91a6503eeb4a0e3f20cfabd90a1821b6bcf03d483f` | pipeline.py 331-536, 566-750 | Reads XJTU Final V2 and HUST Final V1; writes threshold cell-level artifact |

### Executable predicates (quoted from current code)

XJTU multi-batch classifier (`xjtu_multibatch/analysis.py:226-288`):

```python
below = data["Capacity_Ah"].le(float(threshold_Ah))
if not below.any():
    return EOLResult("RIGHT_CENSORED", ...)
if position == 0:
    return EOLResult("LEFT_CENSORED_OR_BASELINE_UNRESOLVED", ...)
if dense_measurement and current_age - previous_age <= 1.0:
    return EOLResult("EXACT_OBSERVED", ...)
return EOLResult("INTERVAL_CENSORED", ...)
```

For Final V2 batches 2-6, `_new_eol` sets `dense = batch_id == 2` (`analysis.py:727`). Batch 1 is preserved from its legacy classifier. The threshold audit declares batches 1 and 2 dense (`config.py:31`) and reproduces all frozen common-Qref 80% statuses.

HUST Final V1 classifier (`hust_repro/pilot.py:302-330`):

```python
crossed = ordered[ordered["Primary_Capacity_Ah"] <= float(threshold_Ah)]
if crossed.empty:
    return {"EOL_Status": EOL_RIGHT_CENSORED, ...}
first = crossed.iloc[0]
return {"EOL_Status": EOL_EXACT, "EOL_Raw_Cycle": int(first["Raw_Cycle_Index"]), ...}
```

HUST Final V1 itself has no gap or left-censor branch. The manuscript threshold classifier adds position 0 -> left and raw gap > 1 -> interval (`eol_threshold_robustness_repro/pipeline.py:487-535`). No frozen HUST case exercised either added branch.

Unavailable Qref is handled before classification: XJTU missing Qref -> `QREF_UNAVAILABLE` with `QREF_UNAVAILABLE_NO_FALLBACK` (`pipeline.py:581-600`). HUST initial-test Qref is absent from the frozen HUST settings (`config.py:19-21`), not imputed.

Protocol/QC flags do not decide exactness inside the classifier. They define the upstream retained population. Exactness then depends on dense-batch/index-gap logic. A raw-cycle gap of one is decisive in HUST; XJTU uses cumulative aging-operation count.

### Current RUL implementation

- XJTU batches 2-6: `value = max(0.0, eol.exact_aging_cycle - current_aging_cycle)` (`xjtu_multibatch/analysis.py:299-305`), while `_rul` iterates every retained primary observation (`980-1008`). Post-EOL values can therefore appear and are clipped to zero, never negative.
- Frozen XJTU Batch 1: point RUL is emitted only when `Aging_Cycle_Index <= EOL_Aging_Cycle`; later rows are `POST_EOL_EXCLUDED` with blank RUL (`sensitivity.py:345-356`).
- HUST Final V1: `label_cycles = eligible[Raw_Cycle_Index <= eol_cycle]`, then `RUL = eol_cycle - raw_cycle` (`pilot.py:460-475`). Post-EOL rows are omitted.
- Cell endpoint availability is not the number of per-cycle labels. Threshold robustness reports a cell-setting as point-RUL-available only for `EXACT_OBSERVED` (`pipeline.py:550-559,1258-1297`).

## VERIFIED FROM FROZEN DATA

- Endpoint trace: 1,353 rows (XJTU 660; HUST 693).
- Status predicate reproduction mismatches: 0.
- Common-Qref 80% mismatches versus Final V2/Final V1: XJTU 0; HUST 0.
- Manuscript table count mismatches: threshold table 0; 80% table 0; threshold-table Qref-availability count mismatches 0.
- Left-censor/baseline-unresolved rows: 0.

### Reaggregated statuses

| Dataset | Threshold_Fraction | Qref_Definition | Cell_Count | EXACT_Count | INTERVAL_Count | RIGHT_Count | LEFT_Count | QREF_UNAVAILABLE_Count | Point_RUL_Available_Count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HUST | 0.75 | early_median5_comparable | 77 | 57 | 0 | 20 | 0 | 0 | 57 |
| HUST | 0.75 | first_comparable | 77 | 61 | 0 | 16 | 0 | 0 | 61 |
| HUST | 0.75 | nominal | 77 | 0 | 0 | 77 | 0 | 0 | 0 |
| HUST | 0.8 | early_median5_comparable | 77 | 77 | 0 | 0 | 0 | 0 | 77 |
| HUST | 0.8 | first_comparable | 77 | 77 | 0 | 0 | 0 | 0 | 77 |
| HUST | 0.8 | nominal | 77 | 0 | 0 | 77 | 0 | 0 | 0 |
| HUST | 0.85 | early_median5_comparable | 77 | 77 | 0 | 0 | 0 | 0 | 77 |
| HUST | 0.85 | first_comparable | 77 | 77 | 0 | 0 | 0 | 0 | 77 |
| HUST | 0.85 | nominal | 77 | 77 | 0 | 0 | 0 | 0 | 77 |
| XJTU | 0.75 | early_median5_comparable | 55 | 0 | 0 | 55 | 0 | 0 | 0 |
| XJTU | 0.75 | first_comparable | 55 | 0 | 0 | 55 | 0 | 0 | 0 |
| XJTU | 0.75 | initial_test | 55 | 0 | 0 | 47 | 0 | 8 | 0 |
| XJTU | 0.75 | nominal | 55 | 0 | 1 | 54 | 0 | 0 | 0 |
| XJTU | 0.8 | early_median5_comparable | 55 | 0 | 3 | 52 | 0 | 0 | 0 |
| XJTU | 0.8 | first_comparable | 55 | 0 | 1 | 54 | 0 | 0 | 0 |
| XJTU | 0.8 | initial_test | 55 | 6 | 14 | 27 | 0 | 8 | 6 |
| XJTU | 0.8 | nominal | 55 | 20 | 32 | 3 | 0 | 0 | 20 |
| XJTU | 0.85 | early_median5_comparable | 55 | 10 | 32 | 13 | 0 | 0 | 10 |
| XJTU | 0.85 | first_comparable | 55 | 5 | 31 | 19 | 0 | 0 | 5 |
| XJTU | 0.85 | initial_test | 55 | 23 | 24 | 0 | 0 | 8 | 23 |
| XJTU | 0.85 | nominal | 55 | 23 | 32 | 0 | 0 | 0 | 23 |

### Existing-data examples with exact source rows

Status rows are in `eol_threshold_cell_level.csv`; capacity row numbers refer to the per-row frozen capacity path in `endpoint_status_trace.csv`. Row numbers include the header as row 1.

| Status | Example | Status_Source_Row | Capacity_Source_Rows |
| --- | --- | --- | --- |
| EXACT_OBSERVED | HUST HUST_1-1; theta=0.75; first_comparable; threshold=0.8807244 Ah; prev=1502/0.8810301222 Ah; first_below=1503/0.8804143028 Ah; last=1504/0.8804153417 Ah | 3 | prev=1503;first_below=1504;last=1505 |
| INTERVAL_CENSORED | XJTU B3_R2.5_battery-1; theta=0.8; nominal; threshold=1.6 Ah; prev=573/1.607 Ah; first_below=578/1.599 Ah; last=588/1.581 Ah | 975 | prev=28202;first_below=28206;last=28214 |
| RIGHT_CENSORED | HUST HUST_1-1; theta=0.75; nominal; threshold=0.825 Ah; prev=1504/0.8804153417 Ah; first_below=/ Ah; last=1504/0.8804153417 Ah | 2 | prev=1505;first_below=;last=1505 |
| QREF_UNAVAILABLE | XJTU B5_RW_battery-1; theta=0.75; initial_test; threshold= Ah; prev=/ Ah; first_below=/ Ah; last=/ Ah | 1166 | prev=;first_below=;last= |
| LEFT_CENSORED_OR_BASELINE_UNRESOLVED | NOT FOUND IN FROZEN THRESHOLD ARTIFACT |  |  |

### Gap and rebound audit

| Dataset | Original_EOL_Status | Endpoint_Settings | Raw_Gap_Min | Raw_Gap_Median | Raw_Gap_Max | Index_Gap_Min | Index_Gap_Median | Index_Gap_Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HUST | EXACT_OBSERVED | 503 | 1 | 1 | 1 | 1 | 1 | 1 |
| XJTU | EXACT_OBSERVED | 87 | 1 | 1 | 1 | 1 | 1 | 1 |
| XJTU | INTERVAL_CENSORED | 170 | 5 | 6 | 11 | 4 | 5 | 10 |

| Dataset | Original_EOL_Status | Endpoint_Settings | Settings_With_Later_Above_Threshold_Rebound |
| --- | --- | --- | --- |
| HUST | EXACT_OBSERVED | 503 | 5 |
| XJTU | EXACT_OBSERVED | 87 | 2 |
| XJTU | INTERVAL_CENSORED | 170 | 3 |

Rebound counts are dataset x cell x Qref x theta settings, not unique cells. A later rebound does not undo a first hit, but shows threshold membership is not absorbing.

### Qref source-cycle availability

`reference_availability_audit.csv` records every source cycle, latest required source cycle, and counts before/at/after availability. Stored `early_median5_comparable` is displayed as `first5_comparable` without changing the frozen name.

| Dataset | Reference_Display_Name | Population_Scope | Cells | Eligible_Observations | Before | At | After |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HUST | first5_comparable | EXISTING_ROBUSTNESS_POSTCYCLE5 | 77 | 145737 | 0 | 0 | 145737 |
| HUST | first5_comparable | FROZEN_SOH_ELIGIBLE | 77 | 146122 | 308 | 77 | 145737 |
| HUST | first_comparable | EXISTING_ROBUSTNESS_POSTCYCLE5 | 77 | 145737 | 0 | 0 | 145737 |
| HUST | first_comparable | FROZEN_SOH_ELIGIBLE | 77 | 146122 | 0 | 77 | 146045 |
| HUST | nominal | EXISTING_ROBUSTNESS_POSTCYCLE5 | 77 | 145737 | 0 | 0 | 145737 |
| HUST | nominal | FROZEN_SOH_ELIGIBLE | 77 | 146122 | 0 | 0 | 146122 |
| XJTU | first5_comparable | EXISTING_ROBUSTNESS_POSTCYCLE5 | 55 | 10327 | 120 | 55 | 10152 |
| XJTU | first5_comparable | FROZEN_SOH_ELIGIBLE | 55 | 10427 | 220 | 55 | 10152 |
| XJTU | first_comparable | EXISTING_ROBUSTNESS_POSTCYCLE5 | 55 | 10327 | 0 | 24 | 10303 |
| XJTU | first_comparable | FROZEN_SOH_ELIGIBLE | 55 | 10427 | 0 | 55 | 10372 |
| XJTU | nominal | EXISTING_ROBUSTNESS_POSTCYCLE5 | 55 | 10327 | 0 | 0 | 10327 |
| XJTU | nominal | FROZEN_SOH_ELIGIBLE | 55 | 10427 | 0 | 0 | 10427 |

The existing `ROBUSTNESS_POSTCYCLE5` population is reconstructed from the frozen A9 matrix plus primary eligibility mask, then the exact existing `Raw_Cycle_Index > 5` filter (`soh_ml_robustness_repro/pipeline.py:1260-1263`). It is separate from reference availability.

### RUL rows versus endpoint availability

- Endpoint artifact: 590 exact cell-settings among 1,353 settings.
- XJTU Final V2 RUL: 41,708 rows, 7,956 numeric labels. Of 51 exact-endpoint rows after EOL, 44 Batch-1 rows are blank/POST_EOL_EXCLUDED and 7 Batch-2 rows are clipped to zero.
- HUST Final V1 RUL: 550,925 rows, 258,681 numeric labels, 0 rows after exact EOL.

## ASSUMPTION REQUIRING HUMAN APPROVAL

1. **Estimand wording.** Approve whether EOL is first threshold hit in retained eligible observations or latent first passage over every physical cycle. Frozen code implements the former, with adjacency-based exactness in the robustness layer.
2. **Interval plus rebound.** Above then below proves a hit by the later observation; locating the latent *first* hit to that interval assumes no earlier unobserved below-threshold excursion and rebound. No monotonicity/persistence rule is coded.
3. **Right censoring plus rebound.** Sparse XJTU diagnostics with no retained hit do not logically exclude an unobserved hit followed by recovery. Latent cycle-level right censoring requires a no-missed-excursion or persistent-failure assumption; observation-grid right censoring does not.
4. **Prospective Qref availability.** Frozen SOH labels are retrospective. Pre-source observations can be labeled retrospectively but were not prospectively available at that time. The audit counts them; exclusion for future prediction is a design decision, not a repair.
5. **Post-EOL RUL.** Final V2 mixes Batch-1 post-EOL exclusion with Batch-2 zero clipping. Approve one policy before claiming a manuscript-wide per-cycle RUL-label count. Cell-level endpoint counts are unaffected.

## NOT FOUND

- No left-censored/baseline-unresolved row among 1,353 frozen settings.
- No manuscript-table count mismatch for exact, interval, right, cell count, Qref availability, or point-RUL availability.
- No HUST frozen `initial_test` setting.
- No monotonicity, persistent-crossing, consecutive-below, or rebound-suppression predicate in the classifiers.
- No protocol flag used directly for exact-versus-interval after upstream eligibility selection.
- No model training, raw-data access, smoothing, interpolation, or endpoint substitution in this audit.
- No Qref source-cycle mapping failure and no source hash change.

## Manuscript action required

**Clarification is enough** if the manuscript defines `N(T)` over retained eligible observations and says `EXACT_OBSERVED` is exact on the declared observed index grid (adjacent cumulative aging operations for dense XJTU batches; adjacent raw cycles for HUST). Frozen numerical tables can remain unchanged.

**A separate, explicitly versioned EOL/RUL recalculation is needed** if `N(T)` is intended as an unobserved cycle-level first passage or persistent/irreversible EOL. That requires an approved rule for sparse gaps, transient rebounds, persistence, left censoring, and post-EOL RUL; this audit does not impose one.

## Audit outputs

- `endpoint_status_trace.csv`: one row per frozen dataset x cell x Qref x theta, with branch and exact source rows.
- `reference_availability_audit.csv`: source cycles and before/at/after counts for frozen SOH and existing POSTCYCLE5 populations.
- `source_manifest.csv`: paths, roles, source lines, file sizes and before/after SHA-256 checks.
