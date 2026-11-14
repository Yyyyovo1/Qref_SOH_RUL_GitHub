"""Archived-file check: the 17 files archived on Zenodo must be byte-identical to the hashes in config (and the records)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from rul_ranking_pilot.labels import PilotError
from rul_ranking_pilot.pipeline import _file_sha

from .config import ARCHIVED_SHA256, RECORD_FILES

_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*\d+\s*\|\s*`([0-9a-f]{64})`\s*\|")


def record_hashes(repo: Path) -> dict[str, str]:
    """SHA-256 values as listed in PREREG_RECORD.md and AMENDMENT_01_RECORD.md."""
    out = {}
    for rec in RECORD_FILES:
        for line in (repo / rec).read_text(encoding="utf-8").splitlines():
            m = _ROW.match(line.strip())
            if m:
                out[m.group(1)] = m.group(2)
    return out


def check_archived(repo: Path) -> pd.DataFrame:
    listed = record_hashes(repo)
    if listed != ARCHIVED_SHA256:
        raise PilotError("ARCHIVE_RECORDS_DIFFER_FROM_CONFIG")
    rows = []
    for path, expected in ARCHIVED_SHA256.items():
        p = repo / path
        actual = _file_sha(p) if p.exists() else ""
        rows.append({"Path": path, "Expected_SHA256": expected, "Actual_SHA256": actual, "Match": actual == expected})
    table = pd.DataFrame(rows)
    if not table.Match.all():
        raise PilotError("ARCHIVED_FILE_CHANGED:" + ",".join(table.Path[~table.Match]))
    return table
