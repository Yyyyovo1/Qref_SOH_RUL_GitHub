"""Freeze MIT/Severson for the RUL pilot v2: cell list, QC, Qref, folds, t0 (all datasets), check 1."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    AUTHOR_DELETIONS,
    BATCHES,
    CITATION,
    EXPECTED_CELLS_PER_BATCH,
    LICENSE_TEXT,
    OUTPUT_DIR_NAME,
    PIPELINE_VERSION,
    PROTOCOL_COMMIT,
    QC,
    RAW_DIR,
    REBOUND_THETAS,
    SOURCE_PAGE,
)
from .freeze import (
    assign_folds,
    check1_cycle_life,
    endpoint_labels,
    qc_cycles,
    qref_table,
    rebound_counts,
    t0_values,
)
from .ingest import assemble, read_batch


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def _other_dataset_t0(root: Path) -> list[dict]:
    """XJTU/HUST t0 under PROTOCOL_v2 section 4; t0_1 inputs come from the frozen v1 pilot output."""
    v1 = pd.read_csv(root / "outputs/rul_ranking_pilot_v1/t0.csv", encoding="utf-8-sig")
    xjtu = pd.read_csv(root / "outputs/xjtu_55_final_v2/xjtu_55_soh_labels_long.csv", encoding="utf-8-sig")
    hust = pd.read_csv(root / "outputs/hust_77_final_v1/hust_77_primary_capacity.csv", encoding="utf-8-sig", low_memory=False)
    last = {
        "XJTU": xjtu[xjtu.Qref_Definition.eq("nominal") & xjtu.Final_Include.eq("YES")].groupby("Cell_ID").Raw_Cycle_Index.max(),
        "HUST": hust[hust.Primary_Capacity_Eligible.eq("YES")].groupby("Cell_ID").Raw_Cycle_Index.max(),
    }
    rows = []
    for r in v1.itertuples(index=False):
        rows.append(t0_values(r.Dataset, int(r.T_avail), pd.Series([r.E_min]), last[r.Dataset]))
    return rows


def run(root: Path, output_dir: Path | None = None) -> Path:
    root = Path(root).resolve()
    target = Path(output_dir).resolve() if output_dir else root / "outputs" / OUTPUT_DIR_NAME
    if target.exists():
        raise FileExistsError(f"Output directory exists; refusing to overwrite: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}_staging_", dir=target.parent))
    log = [f"{PIPELINE_VERSION}; protocol commit {PROTOCOL_COMMIT}"]
    raw = root.parent / RAW_DIR

    manifest, cells, cycles = [], [], []
    for b in BATCHES:
        path = raw / b["file"]
        size = path.stat().st_size
        manifest.append({"Batch": b["batch"], "Batch_Date": b["date"], "File": b["file"], "Bytes": size,
                         "Expected_Bytes": b["bytes"], "Size_Match": size == b["bytes"], "SHA256": _sha(path),
                         "Source_URL": f"https://data.matr.io/1/api/v1/file/{b['file_id']}/download"})
        c, y = read_batch(path, b["batch"])
        cells.append(c)
        cycles.append(y)
        log.append(f"read {b['file']}: {len(c)} cells, {len(y)} cycle rows")
    manifest = pd.DataFrame(manifest)
    if not manifest.Size_Match.all():
        raise ValueError("RAW_FILE_SIZE_MISMATCH")
    cells, cycles = assemble(pd.concat(cells, ignore_index=True), pd.concat(cycles, ignore_index=True))
    included = cells[cells.Status.eq("INCLUDED")]
    per_batch = included.Batch.value_counts().sort_index().to_dict()
    if per_batch != EXPECTED_CELLS_PER_BATCH:
        raise ValueError(f"CELL_COUNT_MISMATCH:{per_batch}")

    qc = qc_cycles(cycles)
    primary = qc[qc.Primary_Capacity_Eligible.eq("YES")].copy()
    qref = qref_table(primary)
    labels = endpoint_labels(primary, qref)
    check1 = check1_cycle_life(labels, included.rename(columns={"Source_Key": "Cell_ID"}))
    folds = assign_folds(list(included.Source_Key))
    folds = folds.merge(included[["Source_Key", "Batch"]].rename(columns={"Source_Key": "Cell_ID"}), on="Cell_ID")

    # Merge continuity (QC rule 4): last batch-1 vs first batch-2 retained capacity.
    continuity = []
    for r in included[included.Merged_From.ne("")].itertuples(index=False):
        g = primary[primary.Cell_ID.eq(r.Source_Key)].sort_values("Cycle_Index")
        before = g[g.Cycle_Index <= r.Merge_Offset_Cycles].tail(1)
        after = g[g.Cycle_Index > r.Merge_Offset_Cycles].head(1)
        continuity.append({"Cell_ID": r.Source_Key, "Merged_From": r.Merged_From, "Merge_Offset_Cycles": r.Merge_Offset_Cycles,
                           "Last_Batch1_Cycle": int(before.Cycle_Index.iat[0]), "Last_Batch1_Capacity_Ah": float(before.QDischarge_Ah.iat[0]),
                           "First_Batch2_Cycle": int(after.Cycle_Index.iat[0]), "First_Batch2_Capacity_Ah": float(after.QDischarge_Ah.iat[0]),
                           "Delta_Ah": float(after.QDischarge_Ah.iat[0] - before.QDischarge_Ah.iat[0])})
    continuity = pd.DataFrame(continuity)

    # Rebound counts (same convention as XJTU/HUST: theta 0.75/0.80/0.85) and over the full grid.
    reb = rebound_counts(primary, labels, REBOUND_THETAS).assign(Theta_Set="0.75|0.80|0.85")
    reb_all = rebound_counts(primary, labels, tuple(labels.Theta.unique())).assign(Theta_Set="0.75-0.95 grid")
    rebound = pd.concat([reb, reb_all], ignore_index=True)

    status = labels.pivot_table(index=["Qref_Definition", "Theta"], columns="Status", values="Cell_ID", aggfunc="count", fill_value=0).reset_index()
    nominal80 = labels[(labels.Qref_Definition == "nominal") & np.isclose(labels.Theta, 0.80)]
    within_1pct = nominal80[nominal80.Status.eq("RIGHT") & (nominal80.Last_Capacity_Ah <= 0.88 * 1.01)]

    # t0 (PROTOCOL_v2 section 4) for MIT and, from frozen inputs, XJTU/HUST.
    t_avail = max(int(s.split("|")[-1]) for s in qref.Qref_Early_Median5_Comparable_Source_Cycles)
    hits = labels[labels.Status.ne("BASELINE")].First_Hit_Cycle.dropna()
    t0 = pd.DataFrame([t0_values("MIT", t_avail, hits, qref.Last_Retained_Cycle), *_other_dataset_t0(root)])

    exact_rows = check1[check1.Status.eq("EXACT")]
    summary = {
        "cells_included": int(len(included)), "cells_per_batch": {str(k): int(v) for k, v in per_batch.items()},
        "cycle_rows_after_merge": int(len(qc)), "cycles_retained": int(len(primary)),
        "cycles_excluded_by_reason": qc[qc.Primary_Capacity_Eligible.eq("NO")].Exclusion_Reasons.value_counts().to_dict(),
        "capacity_jump_flags": int(qc.Capacity_Jump_Flag.eq("YES").sum()),
        "cells_with_capacity_jump_flag": int(qc[qc.Capacity_Jump_Flag.eq("YES")].Cell_ID.nunique()),
        "check1_nominal80_status_counts": check1.Status.value_counts().to_dict(),
        "check1_exact_cells": int(len(exact_rows)),
        "check1_exact_within_1_cycle": int(exact_rows.Within_1_Cycle.astype(bool).sum()),
        "check1_pass": bool(len(exact_rows) and exact_rows.Within_1_Cycle.astype(bool).all()),
        "nominal80_right_censored_within_1pct_of_threshold": int(len(within_1pct)),
        "t0": t0.set_index("Dataset")[["t0_1", "t0_2", "t0_3"]].to_dict(orient="index"),
    }

    _write(manifest, staging / "mit_file_manifest.csv")
    _write(cells, staging / "mit_cell_inventory.csv")
    _write(qc, staging / "mit_cycle_qc.csv")
    _write(primary[["Cell_ID", "Cycle_Index", "QDischarge_Ah", "IR_Ohm", "Tavg_C", "Chargetime_min", "Capacity_Jump_Flag"]],
           staging / "mit_primary_capacity.csv")
    _write(qref, staging / "mit_qref_summary.csv")
    _write(labels, staging / "mit_endpoint_labels.csv")
    _write(status, staging / "mit_endpoint_status_summary.csv")
    _write(check1, staging / "mit_check1_cycle_life.csv")
    _write(continuity, staging / "mit_merge_continuity.csv")
    _write(rebound, staging / "mit_rebound_counts.csv")
    _write(folds, staging / "mit_cell_fold_assignments.csv")
    _write(t0, staging / "rul_pilot_v2_t0_all_datasets.csv")
    (staging / "mit_qc_rules.json").write_text(json.dumps({
        "protocol_commit": PROTOCOL_COMMIT, "qc": QC, "author_deletions": AUTHOR_DELETIONS,
        "expected_discharge_current_C": -4.0, "endpoint_rules": "frozen HUST detector: first retained cycle with capacity <= theta*Qref; "
        "EXACT if previous retained cycle is adjacent, INTERVAL otherwise; no smoothing, interpolation or tolerance"}, indent=2), encoding="utf-8")
    (staging / "DATA_LICENSE.md").write_text(
        "# MIT/Severson data licence\n\n"
        f"- Source: {SOURCE_PAGE}\n- Statement on the source pages (project and all three batch pages), "
        f"checked 2026-11-06: \"{LICENSE_TEXT}\" (Creative Commons Attribution 4.0).\n"
        f"- Required citation: {CITATION}\n- Processed files in this directory are derived from that data; "
        "raw files are not redistributed (see mit_file_manifest.csv for sources and SHA-256).\n", encoding="utf-8")
    (staging / "scan_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    import h5py
    info = {"pipeline": PIPELINE_VERSION, "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "h5py": h5py.__version__}
    (staging / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.append(time.strftime("%Y-%m-%dT%H:%M:%S") + " complete")
    (staging / "processing_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    staging.replace(target)
    return target
