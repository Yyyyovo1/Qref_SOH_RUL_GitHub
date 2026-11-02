from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from scipy.io import loadmat


REQUIRED_DATA_FIELDS = (
    "system_time",
    "relative_time_min",
    "voltage_V",
    "current_A",
    "capacity_Ah",
    "power_Wh",
    "temperature_C",
    "description",
)

SUMMARY_ARRAY_FIELDS = (
    "charge_capacity_Ah",
    "discharge_capacity_Ah",
    "charge_power_Wh",
    "discharge_power_Wh",
    "charge_median_voltage",
    "discharge_median_voltage",
    "charge_mean_voltage",
    "discharge_mean_voltage",
)

REQUIRED_SUMMARY_FIELDS = SUMMARY_ARRAY_FIELDS + ("cycle_life", "description")


class InventoryError(RuntimeError):
    """Raised when canonical or duplicate input inventory is unsafe."""


class SchemaError(RuntimeError):
    """Raised when one or more canonical MAT files fail schema validation."""


@dataclass(frozen=True)
class InventoryRecord:
    cell_id: str
    battery_id: int
    path: Path
    file_role: str
    duplicate_of: str
    sha256: str
    duplicate_sha256: str
    hash_match: bool | None
    included_in_analysis: str


@dataclass(frozen=True)
class InputInventory:
    records: tuple[InventoryRecord, ...]
    canonical_by_id: dict[int, Path]
    hashes_by_name: dict[str, str]


@dataclass(frozen=True)
class SchemaResult:
    cell_id: str
    source_file: str
    file_size_bytes: int
    data_cycle_count: int | None
    summary_cycle_life: int | None
    schema_status: str
    schema_issues: tuple[str, ...]
    processing_status: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_id(battery_id: int) -> str:
    return f"2C_battery-{battery_id}"


def scan_inventory(input_dir: Path, config: dict[str, Any]) -> InputInventory:
    """Classify canonical and byte-identical duplicate MAT files.

    Any unknown extra file, missing canonical ID, or hash mismatch is fatal. Duplicate
    files are recorded but never returned as analysis inputs.
    """

    if not input_dir.is_dir():
        raise InventoryError(f"Input directory does not exist: {input_dir}")

    candidates = sorted(input_dir.glob("2C_battery-*.mat"), key=lambda path: path.name)
    canonical_pattern = re.compile(r"^2C_battery-(\d+)\.mat$")
    duplicate_pattern = re.compile(r"^2C_battery-(\d+)\(1\)\.mat$")
    canonical: dict[int, Path] = {}
    duplicates: dict[int, Path] = {}
    unknown: list[str] = []

    for path in candidates:
        canonical_match = canonical_pattern.fullmatch(path.name)
        duplicate_match = duplicate_pattern.fullmatch(path.name)
        if canonical_match:
            battery_id = int(canonical_match.group(1))
            if battery_id in canonical:
                raise InventoryError(f"Duplicate canonical battery ID {battery_id}: {path.name}")
            canonical[battery_id] = path
        elif duplicate_match:
            battery_id = int(duplicate_match.group(1))
            if battery_id in duplicates:
                raise InventoryError(f"Multiple duplicate files for battery ID {battery_id}")
            duplicates[battery_id] = path
        else:
            unknown.append(path.name)

    if unknown:
        raise InventoryError(f"Unrecognized extra MAT files: {', '.join(unknown)}")

    expected_ids = [int(value) for value in config["expected_battery_ids"]]
    expected_set = set(expected_ids)
    canonical_set = set(canonical)
    if canonical_set != expected_set:
        missing = sorted(expected_set - canonical_set)
        extra = sorted(canonical_set - expected_set)
        raise InventoryError(
            f"Canonical battery IDs must be exactly 1-8; missing={missing}, extra={extra}"
        )
    extra_duplicate_ids = sorted(set(duplicates) - expected_set)
    if extra_duplicate_ids:
        raise InventoryError(f"Unexpected duplicate battery IDs: {extra_duplicate_ids}")

    paths_to_hash = sorted([*canonical.values(), *duplicates.values()], key=lambda path: path.name)
    hashes = {path.name: sha256_file(path) for path in paths_to_hash}
    records: list[InventoryRecord] = []

    for battery_id in expected_ids:
        canonical_path = canonical[battery_id]
        duplicate_path = duplicates.get(battery_id)
        canonical_hash = hashes[canonical_path.name]
        duplicate_hash = hashes[duplicate_path.name] if duplicate_path else ""
        hash_match = canonical_hash == duplicate_hash if duplicate_path else None
        if duplicate_path and not hash_match:
            raise InventoryError(
                "DUPLICATE_CONTENT_MISMATCH: "
                f"{duplicate_path.name} does not match {canonical_path.name}; no file was selected"
            )

        records.append(
            InventoryRecord(
                cell_id=_cell_id(battery_id),
                battery_id=battery_id,
                path=canonical_path,
                file_role="CANONICAL",
                duplicate_of="",
                sha256=canonical_hash,
                duplicate_sha256=duplicate_hash,
                hash_match=hash_match,
                included_in_analysis="YES",
            )
        )
        if duplicate_path:
            records.append(
                InventoryRecord(
                    cell_id=_cell_id(battery_id),
                    battery_id=battery_id,
                    path=duplicate_path,
                    file_role="DUPLICATE_CONFIRMED",
                    duplicate_of=canonical_path.name,
                    sha256=duplicate_hash,
                    duplicate_sha256=canonical_hash,
                    hash_match=True,
                    included_in_analysis="NO",
                )
            )

    return InputInventory(
        records=tuple(sorted(records, key=lambda item: (item.battery_id, item.file_role != "CANONICAL"))),
        canonical_by_id={battery_id: canonical[battery_id] for battery_id in expected_ids},
        hashes_by_name=hashes,
    )


def _decode_matlab_char(values: np.ndarray) -> str:
    flat = np.asarray(values).ravel(order="F")
    return "".join(chr(int(code)) for code in flat if int(code) != 0)


def _decode_hdf5_node(node: h5py.Group | h5py.Dataset, root: h5py.File) -> Any:
    """Best-effort MATLAB v7.3 decoder used only when scipy cannot read a file."""

    if isinstance(node, h5py.Group):
        return {name: _decode_hdf5_node(child, root) for name, child in node.items()}

    matlab_class = node.attrs.get("MATLAB_class", b"")
    if isinstance(matlab_class, np.ndarray):
        matlab_class = matlab_class.tobytes()
    if isinstance(matlab_class, bytes):
        matlab_class = matlab_class.decode("ascii", errors="ignore")

    values = node[()]
    if matlab_class == "char":
        return _decode_matlab_char(values)
    if isinstance(values, np.ndarray) and values.dtype.kind == "O":
        decoded: list[Any] = []
        for reference in values.ravel(order="F"):
            if isinstance(reference, h5py.Reference) and reference:
                decoded.append(_decode_hdf5_node(root[reference], root))
            else:
                decoded.append(None)
        return np.asarray(decoded, dtype=object).reshape(values.shape, order="F")
    return np.asarray(values).squeeze()


def _load_mat_v73(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        return {
            name: _decode_hdf5_node(node, handle)
            for name, node in handle.items()
            if not name.startswith("#")
        }


def load_mat_payload(path: Path) -> dict[str, Any]:
    """Load a MATLAB file without mutating it, with a v7.3 h5py fallback."""

    try:
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        return {key: value for key, value in payload.items() if not key.startswith("__")}
    except (NotImplementedError, ValueError) as exc:
        if not h5py.is_hdf5(path):
            raise
        try:
            return _load_mat_v73(path)
        except Exception as h5_exc:  # pragma: no cover - current Batch-1 is MATLAB v5.
            raise RuntimeError(f"Failed to read MATLAB v7.3 file {path}: {h5_exc}") from exc


def field_names(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value)
    return set(getattr(value, "_fieldnames", ()) or ())


def get_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value[name]
    return getattr(value, name)


def cycle_sequence(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    array = np.atleast_1d(data)
    return list(array.ravel(order="C"))


def numeric_1d(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).ravel(order="C")


def scalar_int(value: Any) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar, found shape {array.shape}")
    numeric = float(array.item())
    integer = int(numeric)
    if not np.isfinite(numeric) or numeric != integer:
        raise ValueError(f"Expected finite integer scalar, found {numeric!r}")
    return integer


def text_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    array = np.asarray(value)
    if array.size == 1:
        return str(array.item()).strip()
    return " ".join(str(item) for item in array.ravel(order="C")).strip()


def _append_cycle_examples(issues: list[str], code: str, cycles: Iterable[int]) -> None:
    cycle_list = list(cycles)
    if not cycle_list:
        return
    examples = "|".join(str(value) for value in cycle_list[:20])
    suffix = "|..." if len(cycle_list) > 20 else ""
    issues.append(f"{code}[count={len(cycle_list)};cycles={examples}{suffix}]")


def validate_schema(path: Path, cell_id: str) -> SchemaResult:
    """Validate one canonical MAT independently and retain all issue reasons."""

    issues: list[str] = []
    data_count: int | None = None
    cycle_life: int | None = None
    try:
        payload = load_mat_payload(path)
    except Exception as exc:
        return SchemaResult(
            cell_id=cell_id,
            source_file=path.name,
            file_size_bytes=path.stat().st_size,
            data_cycle_count=None,
            summary_cycle_life=None,
            schema_status="ERROR",
            schema_issues=(f"MAT_READ_ERROR:{type(exc).__name__}:{exc}",),
            processing_status="STOPPED_SCHEMA_ERROR",
        )

    missing_top = [name for name in ("data", "summary") if name not in payload]
    if missing_top:
        issues.append(f"MISSING_TOP_LEVEL_VARIABLES:{'|'.join(missing_top)}")
    if missing_top:
        return SchemaResult(
            cell_id=cell_id,
            source_file=path.name,
            file_size_bytes=path.stat().st_size,
            data_cycle_count=None,
            summary_cycle_life=None,
            schema_status="ERROR",
            schema_issues=tuple(issues),
            processing_status="STOPPED_SCHEMA_ERROR",
        )

    cycles = cycle_sequence(payload["data"])
    data_count = len(cycles)
    summary = payload["summary"]
    missing_summary = sorted(set(REQUIRED_SUMMARY_FIELDS) - field_names(summary))
    if missing_summary:
        issues.append(f"MISSING_SUMMARY_FIELDS:{'|'.join(missing_summary)}")

    if "cycle_life" not in missing_summary:
        try:
            cycle_life = scalar_int(get_field(summary, "cycle_life"))
        except Exception as exc:
            issues.append(f"UNREADABLE_SUMMARY_CYCLE_LIFE:{type(exc).__name__}:{exc}")

    if cycle_life is not None and data_count != cycle_life:
        issues.append(f"DATA_CYCLE_COUNT_MISMATCH:data={data_count};cycle_life={cycle_life}")

    summary_lengths: dict[str, int] = {}
    summary_nonfinite: list[str] = []
    for name in SUMMARY_ARRAY_FIELDS:
        if name in missing_summary:
            continue
        try:
            values = numeric_1d(get_field(summary, name))
            summary_lengths[name] = int(values.size)
            if not np.isfinite(values).all():
                summary_nonfinite.append(name)
        except Exception as exc:
            issues.append(f"UNREADABLE_SUMMARY_ARRAY:{name}:{type(exc).__name__}:{exc}")
    if cycle_life is not None:
        mismatched_lengths = {
            name: length for name, length in summary_lengths.items() if length != cycle_life
        }
        if mismatched_lengths:
            details = "|".join(f"{name}={length}" for name, length in mismatched_lengths.items())
            issues.append(f"SUMMARY_ARRAY_LENGTH_MISMATCH:{details};expected={cycle_life}")
    if summary_nonfinite:
        issues.append(f"SUMMARY_NONFINITE:{'|'.join(summary_nonfinite)}")

    missing_cycle_fields: list[int] = []
    raw_length_mismatch: list[int] = []
    empty_arrays: list[int] = []
    nonfinite_cycles: list[int] = []
    for raw_index, cycle in enumerate(cycles, start=1):
        missing_fields = sorted(set(REQUIRED_DATA_FIELDS) - field_names(cycle))
        if missing_fields:
            missing_cycle_fields.append(raw_index)
            continue
        arrays: dict[str, np.ndarray] = {}
        unreadable = False
        for name in (
            "system_time",
            "relative_time_min",
            "voltage_V",
            "current_A",
            "capacity_Ah",
            "power_Wh",
            "temperature_C",
        ):
            try:
                arrays[name] = np.asarray(get_field(cycle, name)).ravel(order="C")
            except Exception:
                unreadable = True
        if unreadable:
            empty_arrays.append(raw_index)
            continue
        if any(values.size == 0 for values in arrays.values()):
            empty_arrays.append(raw_index)
        core_lengths = {
            name: int(arrays[name].size)
            for name in ("voltage_V", "current_A", "capacity_Ah")
        }
        if len(set(core_lengths.values())) != 1:
            raw_length_mismatch.append(raw_index)
        for name in (
            "relative_time_min",
            "voltage_V",
            "current_A",
            "capacity_Ah",
            "power_Wh",
            "temperature_C",
        ):
            try:
                if not np.isfinite(arrays[name].astype(float)).all():
                    nonfinite_cycles.append(raw_index)
                    break
            except Exception:
                nonfinite_cycles.append(raw_index)
                break

    _append_cycle_examples(issues, "MISSING_REQUIRED_DATA_FIELDS", missing_cycle_fields)
    _append_cycle_examples(issues, "RAW_CORE_LENGTH_MISMATCH", raw_length_mismatch)
    _append_cycle_examples(issues, "EMPTY_REQUIRED_ARRAY", empty_arrays)
    _append_cycle_examples(issues, "RAW_NONFINITE", nonfinite_cycles)

    status = "PASS" if not issues else "ERROR"
    return SchemaResult(
        cell_id=cell_id,
        source_file=path.name,
        file_size_bytes=path.stat().st_size,
        data_cycle_count=data_count,
        summary_cycle_life=cycle_life,
        schema_status=status,
        schema_issues=tuple(issues),
        processing_status="READY" if status == "PASS" else "STOPPED_SCHEMA_ERROR",
    )


def assert_input_hashes_unchanged(inventory: InputInventory) -> None:
    changed: list[str] = []
    for record in inventory.records:
        current_hash = sha256_file(record.path)
        if current_hash != inventory.hashes_by_name[record.path.name]:
            changed.append(record.path.name)
    if changed:
        raise RuntimeError(f"RAW_INPUT_HASH_CHANGED:{'|'.join(changed)}")

