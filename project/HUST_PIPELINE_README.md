# HUST 77-cell reproducibility pipeline

This pipeline provides protocol-aware, traceable processing for the HUST
77-cell LFP/graphite dataset. The scan, fixed 4-cell raw protocol validation,
fixed 8-cell pilot, and human-approved frozen-rule 77-cell replication are
implemented.

## Scan command

Use the existing Python environment that contains NumPy and pandas:

```powershell
python hust_pipeline.py scan
```

The default input is the read-only `HUST 77/` directory. The scan creates a
new `outputs/hust_scan_v1/` directory and refuses to overwrite an existing
directory. It computes SHA-256 for every source file and uses a restricted
pickle loader that accepts only the NumPy and pandas object classes required
by the dataset.

The scan never trains a model.

## Fixed 4-cell raw protocol validation

After the frozen scan has completed, run the separately approved validation:

```powershell
python hust_pipeline.py validate
```

This reads only `HUST_2-5`, `HUST_9-3`, `HUST_4-7`, and `HUST_10-3`, and
checks raw cycles 1, 2, the half-up rounded middle-life cycle, and the final
raw cycle. It reconstructs charge and all four discharge stages from stored
row/time order and verifies native `dq` against the raw within-cycle capacity
channel. Evidence is written to `outputs/hust_4cell_protocol_validate_v1/`.

The validation does not calculate Qref, SOH, EOL, or RUL. It does not modify
the raw pickle files or the frozen scan. Its human-approved resolution is a
required input to later phases.

## Fixed 8-cell Qref/EOL pilot

After the 4-cell protocol result is frozen as
`RAW_DATA_PROTOCOL_DIFFERS_FROM_PAPER`, run the approved pilot with:

```powershell
python hust_pipeline.py pilot
```

The pilot uses exactly the eight configured representative cells and reads
capacity from verified native `dq / 1000`. It calculates nominal,
first-comparable, and numeric-order early-median5 Qref definitions; un-clipped
SOH; strict exact/right-censored EOL; and censoring-aware RUL labels. The
paper-reported 1C versus raw-observed 2C discrepancy remains an explicit audit
field. Results are written to `outputs/hust_8cell_pilot_v1/` without changing
the raw files, frozen scan, or 4-cell validation outputs.

The pilot output remains frozen and is used as an exact regression reference
for the full run.

## Full 77-cell replication

After the 8-cell pilot has passed human review, run:

```powershell
python hust_pipeline.py run
```

The command requires exactly 77 canonical files and 146,122 raw cycles. It
reproduces the frozen pilot rules without smoothing, interpolation, capacity
correction, threshold rounding, cycle renumbering, or native-RUL substitution.
It verifies every raw-file SHA-256, validates the raw-observed approximately
2C terminal-discharge fingerprint, and reproduces the eight pilot cells before
committing `outputs/hust_77_final_v1/` atomically. Existing output directories
are never overwritten.

The full run produces cycle QC, primary capacities, Qref, SOH, EOL, pairwise
censoring-aware comparisons, lower-bound shift metrics, RUL labels, native
endpoint and stopping-rule audits, review candidates, regression results, a
processing log, and one capacity/EOL review figure per canonical cell.

## Capacity semantics

The native `dq` label is treated as an audit candidate, not as unquestioned
ground truth. For representative first, middle, and last cycles of every cell,
the scan independently calculates full-discharge capacity from the raw
`Capacity (mAh)` channel as the value at the beginning of discharge stage 0
minus the value at the end of discharge stage 3. The output records the
difference and tolerance result without smoothing, interpolation, or capacity
replacement.

## Scientific safeguards

- Raw pickle files are opened read-only and are never modified, renamed,
  moved, deleted, or overwritten.
- Original cell and cycle identifiers are retained. `Cell_ID` is namespaced as
  `HUST_<native id>` while `Native_Cell_ID` preserves the source value.
- Native `rul` and `dq` labels are audit evidence only. They cannot overwrite
  future study-defined SOH, EOL, or RUL labels.
- Capacity jumps are CHECK-only. Missing or incomplete terminal discharge,
  nonpositive capacity, severe voltage anomalies, and schema corruption remain
  explicit INVALID_CANDIDATE evidence.
- The nominal 0.880 Ah end threshold is audited as a dataset stopping rule; it
  is not presented as an independent discovery.
- Numeric `Raw_Cycle_Index` ordering is mandatory. No cycle is renumbered.

## Scan outputs

The output directory contains the ten requested CSV audit tables, the draft
JSON configuration, and the processing log. The log records versions,
thresholds, scan status, safety checks, and the explicit block on validation,
analysis, and model training.
