# XJTU 55-cell multi-batch pipeline

This pipeline extends the frozen Batch-1 pilot through protocol adapters for
Batch-2 through Batch-6. It never modifies MAT files, never smooths or
interpolates capacity, and preserves every raw cycle index.

## Commands

Use the same Python environment that contains NumPy, pandas, SciPy, h5py, and
Matplotlib.

```powershell
python xjtu_pipeline.py scan
python xjtu_pipeline.py validate
python xjtu_pipeline.py run
python xjtu_pipeline.py hotfix-validate
python xjtu_pipeline.py final-v2
```

`scan` is read-only and writes a new directory. Existing output directories are
never overwritten. Batch-1 is read from the frozen final outputs; its raw MAT
files and QC are not reprocessed.

`validate` consumes the read-only `xjtu_multibatch_scan_v2` inventory and the
versioned representative manual-review record
`xjtu_multibatch_manual_protocol_review_v1.json`. It writes final-candidate
configuration files, a validated protocol inventory, the regenerated anomaly
table, and all regression-test evidence. Passing validation does **not** grant
permission to run the 55-cell analysis.

`run` requires a separate `run_approval.json` in the validation directory. The
approval must name an approver and bind both the validation receipt and the
validated protocol inventory by SHA-256. The run also refuses to continue if
the scan artifacts, validation artifacts, frozen Batch-1 outputs, or canonical
MAT hashes change.

The approved full run writes a new, non-overwriting
`outputs/xjtu_55_final_v1/` directory. It contains the 55-cell manifest, complete
cycle QC, primary observations, Qref/SOH/EOL/RUL sensitivity tables, interval
bound-aware pairwise comparisons, descriptive summaries, known data-quality
notes, regression-test evidence, figures, a processing log, and a SHA-256 run
receipt. Batch-1 is mapped from its frozen final outputs and is never
reprocessed from raw MAT during aggregation.

`hotfix-validate` leaves `outputs/xjtu_55_final_v1/` unchanged. It strictly
converts `Diagnostic_Index`, `Raw_Cycle_Index`, and
`Cumulative_Aging_Operation_Count` with `pd.to_numeric(errors="raise")` before
selecting the first five final-included comparable observations. It writes only
the corrected Qref preview, regression evidence, the five Batch-5 CHECK rows,
their raw-trace review plots, and a validation log to the separate
`outputs/xjtu_55_hotfix_validation_v1/` directory. It does not regenerate EOL
or RUL and it does not change any manual inclusion decision.

`final-v2` consumes the frozen PRE_HOTFIX run and the completed hotfix-review
evidence. It records the five approved Batch-5 capacity-jump cycles as manual
`KEEP` while preserving their automatic `CHECK` / `CAPACITY_JUMP` audit fields,
applies numeric diagnostic ordering, and regenerates Qref, SOH, EOL,
censoring-aware RUL, and descriptive summaries in the non-overwriting
`outputs/xjtu_55_final_v2/` directory. It refuses to write final outputs unless
all source-cycle, median-capacity, EOL-bound, manual-review, frozen Batch-1, and
source-integrity regressions pass.

## Scientific definitions

- Qref: nominal, protocol-valid BOL initial test, first recurring comparable
  observation, and median of the first five recurring comparable observations.
- Batch-5 never substitutes its first later diagnostic for a missing BOL initial
  test.
- Primary capacity uses only the batch-specific comparable full-discharge
  protocol.
- EOL uses the fixed 80% SOH criterion and a separate 1.600 Ah control.
- Sparse diagnostic crossings are interval-censored; right-censored cells have
  no fabricated point EOL or RUL.
- Batch-5 description/current mismatches that are already explained by the
  validated random-walk metadata are retained as INFO and remain nonprimary.
- Batch-6 GEO cycles can lack an observable 4 A plateau after a high-SOC start
  when they still reach 4.18 V and complete CV taper; this is INFO only.
- A Batch-6 GEO cycle reaching 2.5 V remains GEO and never becomes a primary
  capacity diagnostic.
- Capacity jumps are CHECK-only and use the nearest preceding valid comparable
  reference.

## Scan outputs

The current validated scan directory is `outputs/xjtu_multibatch_scan_v2/`. It contains the
file and duplicate manifest, schema inventory, description and cycle protocol
catalogs, primary candidates, anomaly list, five draft batch configurations,
the representative human-validation plan, and a complete processing log.
