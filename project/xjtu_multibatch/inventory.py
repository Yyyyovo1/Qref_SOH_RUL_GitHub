from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config_templates import EXPECTED_CANONICAL_COUNTS


@dataclass(frozen=True)
class CanonicalFile:
    batch_id: int
    local_battery_id: int
    cell_id: str
    path: Path
    sha256: str
    duplicate_ambiguity: bool


@dataclass(frozen=True)
class InventoryResult:
    manifest: pd.DataFrame
    duplicates: pd.DataFrame
    canonical_by_batch: dict[int, tuple[CanonicalFile, ...]]
    issues: tuple[str, ...]
    hard_stop: bool


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _battery_id(stem: str) -> int | None:
    match = re.search(r"(?i)(?:^|[_\- ])battery[_\- ]?(\d+)$", stem)
    return int(match.group(1)) if match else None


def _cell_id(batch_id: int, canonical_stem: str) -> str:
    return f"B{batch_id}_{canonical_stem}"


def _empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Batch_ID",
            "Cell_ID",
            "Local_Battery_ID",
            "Source_File",
            "Source_Directory",
            "File_Size_Bytes",
            "SHA256",
            "File_Role",
            "Duplicate_Of",
            "Duplicate_SHA256",
            "Hash_Match",
            "Included_In_Analysis",
            "Reference_Status",
            "Canonical_Count_Status",
            "Inventory_Issues",
        ]
    )


def _load_frozen_batch1(frozen_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    manifest_path = frozen_dir / "xjtu_file_manifest.csv"
    cycle_path = frozen_dir / "xjtu_cycle_capacity_final.csv"
    if not manifest_path.is_file():
        manifest_path = frozen_dir.parent / "xjtu_batch1_pilot_v1" / "xjtu_file_manifest.csv"
    if not manifest_path.is_file() or not cycle_path.is_file():
        return [], [f"FROZEN_BATCH1_OUTPUT_MISSING:{frozen_dir}"]
    source = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for item in source.itertuples(index=False):
        role = str(item.File_Role)
        rows.append(
            {
                "Batch_ID": 1,
                "Cell_ID": str(item.Cell_ID),
                "Local_Battery_ID": int(str(item.Cell_ID).rsplit("-", 1)[-1]),
                "Source_File": str(item.Source_File),
                "Source_Directory": "FROZEN_BATCH1_REFERENCE",
                "File_Size_Bytes": int(item.File_Size_Bytes),
                "SHA256": str(item.SHA256),
                "File_Role": role,
                "Duplicate_Of": str(item.Duplicate_Of),
                "Duplicate_SHA256": str(item.Duplicate_SHA256),
                "Hash_Match": str(item.Hash_Match),
                "Included_In_Analysis": str(item.Included_In_Analysis),
                "Reference_Status": "FROZEN_REFERENCE",
                "Canonical_Count_Status": "EXPECTED_COUNT_MATCH",
                "Inventory_Issues": "",
            }
        )
    return rows, issues


def scan_inventory(workspace: Path, frozen_batch1_dir: Path) -> InventoryResult:
    manifest_rows, issues = _load_frozen_batch1(frozen_batch1_dir)
    duplicate_rows: list[dict[str, Any]] = [
        row.copy()
        for row in manifest_rows
        if str(row.get("File_Role", "")).startswith("DUPLICATE")
    ]
    canonical_by_batch: dict[int, tuple[CanonicalFile, ...]] = {1: tuple()}
    hard_stop = False

    frozen_canonical_count = sum(
        1 for row in manifest_rows if row.get("File_Role") == "CANONICAL"
    )
    if frozen_canonical_count != EXPECTED_CANONICAL_COUNTS[1]:
        issues.append(
            "CANONICAL_COUNT_MISMATCH:Batch-1:"
            f"found={frozen_canonical_count};expected={EXPECTED_CANONICAL_COUNTS[1]}"
        )
        for row in manifest_rows:
            row["Canonical_Count_Status"] = "CANONICAL_COUNT_MISMATCH"

    for batch_id in range(2, 7):
        batch_dir = workspace / f"Batch-{batch_id}"
        canonical_files: list[CanonicalFile] = []
        batch_rows: list[dict[str, Any]] = []
        if not batch_dir.is_dir():
            issues.append(f"BATCH_DIRECTORY_MISSING:Batch-{batch_id}")
            canonical_by_batch[batch_id] = tuple()
            continue

        mat_files = sorted(batch_dir.glob("*.mat"), key=lambda value: value.name.lower())
        grouped: dict[str, list[tuple[Path, int | None]]] = {}
        for path in mat_files:
            match = re.fullmatch(r"(.+)\((\d+)\)\.mat", path.name, flags=re.IGNORECASE)
            if match:
                base_name = f"{match.group(1)}.mat"
                copy_number = int(match.group(2))
            else:
                base_name = path.name
                copy_number = None
            grouped.setdefault(base_name.lower(), []).append((path, copy_number))

        hashes = {path.name: sha256_file(path) for path in mat_files}
        used_ids: dict[int, str] = {}
        for normalized_base, members in sorted(grouped.items()):
            canonical_candidates = [path for path, copy_no in members if copy_no is None]
            duplicate_candidates = [(path, copy_no) for path, copy_no in members if copy_no is not None]
            if len(canonical_candidates) != 1:
                code = "ORPHAN_DUPLICATE_NO_CANONICAL" if not canonical_candidates else "MULTIPLE_CANONICAL_BASE_FILES"
                issues.append(f"{code}:Batch-{batch_id}:{normalized_base}")
                for path, copy_no in members:
                    batch_rows.append(
                        {
                            "Batch_ID": batch_id,
                            "Cell_ID": "",
                            "Local_Battery_ID": pd.NA,
                            "Source_File": path.name,
                            "Source_Directory": str(batch_dir),
                            "File_Size_Bytes": path.stat().st_size,
                            "SHA256": hashes[path.name],
                            "File_Role": "UNRESOLVED_FILE_ROLE",
                            "Duplicate_Of": normalized_base,
                            "Duplicate_SHA256": "",
                            "Hash_Match": "",
                            "Included_In_Analysis": "NO",
                            "Reference_Status": "NEW_BATCH_SCAN",
                            "Canonical_Count_Status": "PENDING",
                            "Inventory_Issues": code,
                        }
                    )
                continue

            canonical = canonical_candidates[0]
            local_id = _battery_id(canonical.stem)
            canonical_issue_codes: list[str] = []
            if local_id is None:
                canonical_issue_codes.append("UNRECOGNIZED_BATTERY_FILENAME")
                issues.append(f"UNRECOGNIZED_BATTERY_FILENAME:Batch-{batch_id}:{canonical.name}")
            elif local_id in used_ids:
                canonical_issue_codes.append("DUPLICATE_LOCAL_BATTERY_ID")
                issues.append(
                    f"DUPLICATE_LOCAL_BATTERY_ID:Batch-{batch_id}:{local_id}:"
                    f"{used_ids[local_id]}|{canonical.name}"
                )
            else:
                used_ids[local_id] = canonical.name

            cell_id = _cell_id(batch_id, canonical.stem) if local_id is not None else ""
            canonical_hash = hashes[canonical.name]
            mismatch = False
            duplicate_hashes: list[str] = []
            for duplicate, copy_number in sorted(duplicate_candidates, key=lambda item: item[0].name):
                duplicate_hash = hashes[duplicate.name]
                duplicate_hashes.append(duplicate_hash)
                hash_match = duplicate_hash == canonical_hash
                if not hash_match:
                    mismatch = True
                    hard_stop = True
                    issues.append(
                        f"DUPLICATE_CONTENT_MISMATCH:Batch-{batch_id}:"
                        f"{duplicate.name}!={canonical.name}"
                    )
                duplicate_record = {
                    "Batch_ID": batch_id,
                    "Cell_ID": cell_id,
                    "Local_Battery_ID": local_id if local_id is not None else pd.NA,
                    "Source_File": duplicate.name,
                    "Source_Directory": str(batch_dir),
                    "File_Size_Bytes": duplicate.stat().st_size,
                    "SHA256": duplicate_hash,
                    "File_Role": "DUPLICATE_CONFIRMED" if hash_match else "DUPLICATE_CONTENT_MISMATCH",
                    "Duplicate_Of": canonical.name,
                    "Duplicate_SHA256": canonical_hash,
                    "Hash_Match": "TRUE" if hash_match else "FALSE",
                    "Included_In_Analysis": "NO",
                    "Reference_Status": "NEW_BATCH_SCAN",
                    "Canonical_Count_Status": "PENDING",
                    "Inventory_Issues": "" if hash_match else "DUPLICATE_CONTENT_MISMATCH",
                }
                batch_rows.append(duplicate_record)
                duplicate_rows.append(duplicate_record.copy())

            included = local_id is not None and not mismatch and not canonical_issue_codes
            batch_rows.append(
                {
                    "Batch_ID": batch_id,
                    "Cell_ID": cell_id,
                    "Local_Battery_ID": local_id if local_id is not None else pd.NA,
                    "Source_File": canonical.name,
                    "Source_Directory": str(batch_dir),
                    "File_Size_Bytes": canonical.stat().st_size,
                    "SHA256": canonical_hash,
                    "File_Role": "CANONICAL" if included else "CANONICAL_UNRESOLVED",
                    "Duplicate_Of": "",
                    "Duplicate_SHA256": "|".join(duplicate_hashes),
                    "Hash_Match": "TRUE" if duplicate_candidates and not mismatch else ("FALSE" if mismatch else ""),
                    "Included_In_Analysis": "YES" if included else "NO",
                    "Reference_Status": "NEW_BATCH_SCAN",
                    "Canonical_Count_Status": "PENDING",
                    "Inventory_Issues": "|".join(canonical_issue_codes),
                }
            )
            if included:
                canonical_files.append(
                    CanonicalFile(
                        batch_id=batch_id,
                        local_battery_id=int(local_id),
                        cell_id=cell_id,
                        path=canonical,
                        sha256=canonical_hash,
                        duplicate_ambiguity=mismatch,
                    )
                )

        expected = EXPECTED_CANONICAL_COUNTS[batch_id]
        count_status = (
            "EXPECTED_COUNT_MATCH"
            if len(canonical_files) == expected
            else "CANONICAL_COUNT_MISMATCH"
        )
        if count_status != "EXPECTED_COUNT_MATCH":
            issues.append(
                f"CANONICAL_COUNT_MISMATCH:Batch-{batch_id}:"
                f"found={len(canonical_files)};expected={expected}"
            )
        for row in batch_rows:
            row["Canonical_Count_Status"] = count_status
        manifest_rows.extend(batch_rows)
        canonical_by_batch[batch_id] = tuple(
            sorted(canonical_files, key=lambda item: item.local_battery_id)
        )

    manifest = pd.DataFrame(manifest_rows) if manifest_rows else _empty_manifest()
    if not manifest.empty:
        manifest = manifest.sort_values(
            ["Batch_ID", "Local_Battery_ID", "File_Role", "Source_File"],
            na_position="last",
        ).reset_index(drop=True)
        manifest["Local_Battery_ID"] = pd.array(manifest["Local_Battery_ID"], dtype="Int64")
    duplicates = pd.DataFrame(duplicate_rows, columns=manifest.columns)
    if duplicates.empty:
        duplicates = _empty_manifest()
    return InventoryResult(
        manifest=manifest,
        duplicates=duplicates,
        canonical_by_batch=canonical_by_batch,
        issues=tuple(issues),
        hard_stop=hard_stop,
    )


def verify_inventory_hashes(inventory: InventoryResult) -> None:
    changed: list[str] = []
    for files in inventory.canonical_by_batch.values():
        for record in files:
            if sha256_file(record.path) != record.sha256:
                changed.append(str(record.path))
    if changed:
        raise RuntimeError(f"RAW_INPUT_HASH_CHANGED:{'|'.join(changed)}")
