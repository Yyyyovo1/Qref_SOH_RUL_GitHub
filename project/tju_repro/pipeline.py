"""Build the frozen TJU NCA / NCM outputs: scan -> cycle QC -> cell list -> Qref table -> folds.

Usage: python -m tju_repro.pipeline   (run from project/; reads raw_data/TJU, writes outputs/tju_scan_v1 and outputs/tju_final_v1)
No endpoint labels, t0 or model inputs are produced here.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CELL_MODEL,
    CITATION,
    DISCHARGE_CUTOFF_V,
    FILES,
    NOMINAL_CAPACITY_AH,
    OUTPUT_DIR_NAME,
    PIPELINE_VERSION,
    QC,
    QREFS,
    RAW_DIR,
    SCAN_DIR_NAME,
    ZENODO,
)
from .freeze import MIN_ELIGIBLE_CYCLES, assign_folds, cell_inventory, primary_capacity, qc_cycles, qref_table
from .ingest import scan


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def run(project: Path) -> Path:
    raw = project.parent / RAW_DIR
    scan_dir, out = project / "outputs" / SCAN_DIR_NAME, project / "outputs" / OUTPUT_DIR_NAME
    scan_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest, inv = scan(raw)
    _write(manifest, scan_dir / "tju_file_manifest.csv")
    _write(inv, scan_dir / "tju_cycle_inventory.csv")

    qc = qc_cycles(inv, manifest)
    primary = primary_capacity(qc)
    cells = cell_inventory(manifest, primary)
    included = cells[cells.Status.eq("INCLUDED")].Cell_ID.tolist()
    primary = primary[primary.Cell_ID.isin(included)].reset_index(drop=True)
    qref = qref_table(primary, included)
    folds = pd.concat([assign_folds(g.Cell_ID.tolist(), ds) for ds, g in cells[cells.Status.eq("INCLUDED")].groupby("Dataset")],
                      ignore_index=True).sort_values(["Dataset", "Fold_ID", "Cell_ID"])

    _write(qc, out / "tju_cycle_qc.csv")
    _write(primary, out / "tju_primary_capacity.csv")
    _write(cells, out / "tju_cell_inventory.csv")
    _write(qref, out / "tju_qref_summary.csv")
    _write(folds, out / "tju_cell_fold_assignments.csv")
    rules = {"pipeline_version": PIPELINE_VERSION, "qc": QC, "min_eligible_cycles_per_cell": MIN_ELIGIBLE_CYCLES,
             "nominal_capacity_Ah": NOMINAL_CAPACITY_AH, "discharge_cutoff_V": DISCHARGE_CUTOFF_V,
             "capacity_definition": "max of 'Q discharge/mA.h' within the cycle (equals its value at the end of the 1C discharge)",
             "cycle_index": "'cycle number' from the source file",
             "qref_definitions": {"nominal": "manufacturer nominal capacity (3.5 Ah, Zhu et al. 2022 Table 1)",
                                  "first_comparable": "capacity of the first eligible cycle",
                                  "early_median5_comparable": "median capacity of the first five eligible cycles"},
             "qrefs": list(QREFS), "excluded_dataset": "Dataset 3 (NCM+NCA blend, 9 cells): not downloaded, not used",
             "exclusion_reasons": ["SUPERSEDED_BY_TIME_RESET", "NO_DISCHARGE", "CAPACITY_NONFINITE_OR_NONPOSITIVE",
                                   "CAPACITY_ABOVE_1P5_NOMINAL", "DISCHARGE_CURRENT_PROTOCOL_INCONSISTENT",
                                   "CHARGE_CURRENT_PROTOCOL_INCONSISTENT", "INCOMPLETE_DISCHARGE", "INCOMPLETE_CHARGE"],
             "check_only_flags": ["Capacity_Jump_Flag", "Discharge_Noncontiguous_Flag"]}
    (out / "tju_qc_rules.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")
    reasons = qc.Exclusion_Reasons[qc.Exclusion_Reasons.ne("")].str.split("|").explode().value_counts().to_dict()
    info = {"pipeline_version": PIPELINE_VERSION, "zenodo": ZENODO, "citation": CITATION, "cell_models": CELL_MODEL,
            "files": FILES, "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "cells": cells.groupby(["Dataset", "Status"]).size().rename("n").reset_index().to_dict("records"),
            "cycles_scanned": int(len(qc)), "cycles_eligible": int(qc.Primary_Capacity_Eligible.eq("YES").sum()),
            "exclusion_reason_counts": reasons,
            "capacity_jump_flags": int(qc.Capacity_Jump_Flag.eq("YES").sum()),
            "discharge_noncontiguous_flags": int(qc.Discharge_Noncontiguous_Flag.eq("YES").sum())}
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out / "DATA_LICENSE.md").write_text(
        "# Data licence\n\nDerived from the TJU (Tongji) battery dataset, Zenodo concept record "
        f"{ZENODO['concept_doi']} (version {ZENODO['version_doi']}), released under CC BY 4.0.\n\n"
        f"Cite: {CITATION}\n\nChanges: per-cycle discharge capacity extracted from the raw CSVs of datasets 1 (NCA) and 2 (NCM); "
        "cycle QC applied (see tju_qc_rules.json). Dataset 3 is not included.\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(run(Path(__file__).resolve().parents[1]))
