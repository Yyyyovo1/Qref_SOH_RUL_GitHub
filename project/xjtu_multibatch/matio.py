from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


@dataclass(frozen=True)
class SchemaAdapter:
    description_field: str
    voltage_field: str
    current_field: str
    capacity_field: str
    time_field: str
    time_unit: str
    summary_discharge_capacity_field: str
    summary_cycle_life_field: str | None
    signature: str
    variant_id: str = "PENDING"


@dataclass(frozen=True)
class MatFileScan:
    batch_id: int
    cell_id: str
    source_file: str
    path: Path
    payload: Mapping[str, Any] | None
    cycles: tuple[Any, ...]
    summary_capacities_Ah: np.ndarray | None
    adapter: SchemaAdapter | None
    schema_row: dict[str, Any]


def _decode_matlab_char(values: np.ndarray) -> str:
    flat = np.asarray(values).ravel(order="F")
    return "".join(chr(int(code)) for code in flat if int(code) != 0)


def _decode_hdf5_node(node: h5py.Group | h5py.Dataset, root: h5py.File) -> Any:
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
    try:
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        return {key: value for key, value in payload.items() if not key.startswith("__")}
    except (NotImplementedError, ValueError) as exc:
        if not h5py.is_hdf5(path):
            raise
        try:
            return _load_mat_v73(path)
        except Exception as h5_exc:
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
    return list(np.atleast_1d(data).ravel(order="C"))


def numeric_1d(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).ravel(order="C")


def text_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    array = np.asarray(value)
    if array.size == 1:
        return str(array.item()).strip()
    return " ".join(str(item) for item in array.ravel(order="C")).strip()


def _first_alias(names: Iterable[str], aliases: Iterable[str]) -> str | None:
    actual = set(names)
    for candidate in aliases:
        if candidate in actual:
            return candidate
    lower_lookup = {name.lower(): name for name in actual}
    for candidate in aliases:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    return None


def _time_unit(field_name: str) -> str:
    lowered = field_name.lower()
    if "min" in lowered:
        return "MINUTES"
    if lowered.endswith("_s") or "second" in lowered:
        return "SECONDS"
    return "UNKNOWN"


def discover_adapter(
    cycles: list[Any],
    summary: Any,
    config: Mapping[str, Any],
) -> tuple[SchemaAdapter | None, list[str]]:
    issues: list[str] = []
    if not cycles:
        return None, ["EMPTY_DATA_CYCLE_ARRAY"]
    data_names = field_names(cycles[0])
    summary_names = field_names(summary)
    aliases = config["schema_field_aliases"]
    mappings: dict[str, str | None] = {
        logical: _first_alias(data_names, aliases[logical])
        for logical in ("description", "voltage", "current", "capacity", "relative_time")
    }
    summary_capacity = _first_alias(
        summary_names, aliases["summary_discharge_capacity"]
    )
    summary_life = _first_alias(summary_names, aliases["summary_cycle_life"])
    missing = [logical for logical, actual in mappings.items() if actual is None]
    if summary_capacity is None:
        missing.append("summary_discharge_capacity")
    if missing:
        issues.append(f"MISSING_LOGICAL_FIELDS:{'|'.join(missing)}")
        return None, issues
    assert summary_capacity is not None
    mapping_text = {
        **{key: str(value) for key, value in mappings.items()},
        "summary_discharge_capacity": summary_capacity,
        "summary_cycle_life": summary_life or "",
    }
    signature = json.dumps(mapping_text, sort_keys=True, separators=(",", ":"))
    adapter = SchemaAdapter(
        description_field=str(mappings["description"]),
        voltage_field=str(mappings["voltage"]),
        current_field=str(mappings["current"]),
        capacity_field=str(mappings["capacity"]),
        time_field=str(mappings["relative_time"]),
        time_unit=_time_unit(str(mappings["relative_time"])),
        summary_discharge_capacity_field=summary_capacity,
        summary_cycle_life_field=summary_life,
        signature=signature,
    )
    if adapter.time_unit == "UNKNOWN":
        issues.append(f"UNKNOWN_TIME_UNIT:{adapter.time_field}")
    return adapter, issues


def _scalar_int(value: Any) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar, found {array.shape}")
    number = float(array.item())
    if not np.isfinite(number) or number != int(number):
        raise ValueError(f"Expected finite integer, found {number}")
    return int(number)


def _example_issue(code: str, cycles: list[int]) -> str | None:
    if not cycles:
        return None
    sample = "|".join(str(value) for value in cycles[:20])
    suffix = "|..." if len(cycles) > 20 else ""
    return f"{code}[count={len(cycles)};cycles={sample}{suffix}]"


def scan_mat_schema(
    path: Path,
    batch_id: int,
    cell_id: str,
    config: Mapping[str, Any],
) -> MatFileScan:
    base_row: dict[str, Any] = {
        "Batch_ID": batch_id,
        "Cell_ID": cell_id,
        "Source_File": path.name,
        "File_Size_Bytes": path.stat().st_size,
        "Top_Level_Variables": "",
        "Data_Cycle_Count": pd.NA,
        "Summary_Cycle_Life": pd.NA,
        "Data_Fields": "",
        "Summary_Fields": "",
        "Description_Field": "",
        "Voltage_Field": "",
        "Current_Field": "",
        "Capacity_Field": "",
        "Time_Field": "",
        "Time_Unit": "",
        "Summary_Discharge_Capacity_Field": "",
        "Schema_Variant_Signature": "",
        "Schema_Variant_ID": "PENDING",
        "Schema_Status": "ERROR",
        "Schema_Issues": "",
        "Processing_Status": "STOPPED_SCHEMA_ERROR",
    }
    try:
        payload = load_mat_payload(path)
    except Exception as exc:
        base_row["Schema_Issues"] = f"MAT_READ_ERROR:{type(exc).__name__}:{exc}"
        return MatFileScan(
            batch_id, cell_id, path.name, path, None, tuple(), None, None, base_row
        )

    top_names = sorted(payload)
    base_row["Top_Level_Variables"] = "|".join(top_names)
    missing_top = [name for name in ("data", "summary") if name not in payload]
    if missing_top:
        base_row["Schema_Issues"] = f"MISSING_TOP_LEVEL_VARIABLES:{'|'.join(missing_top)}"
        return MatFileScan(
            batch_id, cell_id, path.name, path, payload, tuple(), None, None, base_row
        )

    cycles = cycle_sequence(payload["data"])
    summary = payload["summary"]
    base_row["Data_Cycle_Count"] = len(cycles)
    base_row["Data_Fields"] = (
        "|".join(sorted(field_names(cycles[0]))) if cycles else ""
    )
    base_row["Summary_Fields"] = "|".join(sorted(field_names(summary)))
    adapter, issues = discover_adapter(cycles, summary, config)
    if adapter is None:
        base_row["Schema_Issues"] = "|".join(issues)
        return MatFileScan(
            batch_id, cell_id, path.name, path, payload, tuple(cycles), None, None, base_row
        )

    base_row.update(
        {
            "Description_Field": adapter.description_field,
            "Voltage_Field": adapter.voltage_field,
            "Current_Field": adapter.current_field,
            "Capacity_Field": adapter.capacity_field,
            "Time_Field": adapter.time_field,
            "Time_Unit": adapter.time_unit,
            "Summary_Discharge_Capacity_Field": adapter.summary_discharge_capacity_field,
            "Schema_Variant_Signature": adapter.signature,
        }
    )
    cycle_life: int | None = None
    if adapter.summary_cycle_life_field:
        try:
            cycle_life = _scalar_int(get_field(summary, adapter.summary_cycle_life_field))
            base_row["Summary_Cycle_Life"] = cycle_life
        except Exception as exc:
            issues.append(f"UNREADABLE_SUMMARY_CYCLE_LIFE:{type(exc).__name__}:{exc}")
    else:
        issues.append("MISSING_SUMMARY_CYCLE_LIFE_FIELD")

    try:
        summary_capacity = numeric_1d(
            get_field(summary, adapter.summary_discharge_capacity_field)
        )
    except Exception as exc:
        summary_capacity = None
        issues.append(f"UNREADABLE_SUMMARY_DISCHARGE_CAPACITY:{type(exc).__name__}:{exc}")
    if cycle_life is not None and cycle_life != len(cycles):
        issues.append(
            f"DATA_CYCLE_COUNT_MISMATCH:data={len(cycles)};cycle_life={cycle_life}"
        )
    if summary_capacity is not None:
        if summary_capacity.size != len(cycles):
            issues.append(
                f"SUMMARY_CAPACITY_LENGTH_MISMATCH:capacity={summary_capacity.size};"
                f"data={len(cycles)}"
            )
        if not np.isfinite(summary_capacity).all():
            issues.append("SUMMARY_CAPACITY_NONFINITE")

    required_actual = {
        adapter.description_field,
        adapter.voltage_field,
        adapter.current_field,
        adapter.capacity_field,
        adapter.time_field,
    }
    missing_fields: list[int] = []
    length_mismatch: list[int] = []
    empty_arrays: list[int] = []
    nonfinite: list[int] = []
    unreadable: list[int] = []
    for raw_index, cycle in enumerate(cycles, start=1):
        if not required_actual.issubset(field_names(cycle)):
            missing_fields.append(raw_index)
            continue
        try:
            voltage = numeric_1d(get_field(cycle, adapter.voltage_field))
            current = numeric_1d(get_field(cycle, adapter.current_field))
            capacity = numeric_1d(get_field(cycle, adapter.capacity_field))
            time_values = numeric_1d(get_field(cycle, adapter.time_field))
            text_value(get_field(cycle, adapter.description_field))
        except Exception:
            unreadable.append(raw_index)
            continue
        arrays = (voltage, current, capacity, time_values)
        if any(values.size == 0 for values in arrays):
            empty_arrays.append(raw_index)
        if len({values.size for values in arrays}) != 1:
            length_mismatch.append(raw_index)
        if any(not np.isfinite(values).all() for values in arrays):
            nonfinite.append(raw_index)
    for code, values in (
        ("MISSING_REQUIRED_DATA_FIELDS", missing_fields),
        ("UNREADABLE_REQUIRED_DATA_FIELDS", unreadable),
        ("RAW_ARRAY_LENGTH_MISMATCH", length_mismatch),
        ("EMPTY_REQUIRED_ARRAY", empty_arrays),
        ("RAW_NONFINITE", nonfinite),
    ):
        issue = _example_issue(code, values)
        if issue:
            issues.append(issue)

    severe_prefixes = (
        "MISSING_SUMMARY_CYCLE_LIFE_FIELD",
        "UNREADABLE_SUMMARY_DISCHARGE_CAPACITY",
        "DATA_CYCLE_COUNT_MISMATCH",
        "SUMMARY_CAPACITY_LENGTH_MISMATCH",
        "SUMMARY_CAPACITY_NONFINITE",
        "MISSING_REQUIRED_DATA_FIELDS",
        "UNREADABLE_REQUIRED_DATA_FIELDS",
        "RAW_ARRAY_LENGTH_MISMATCH",
        "EMPTY_REQUIRED_ARRAY",
        "RAW_NONFINITE",
        "UNKNOWN_TIME_UNIT",
    )
    severe = any(issue.startswith(severe_prefixes) for issue in issues)
    base_row["Schema_Status"] = "ERROR" if severe else "PASS"
    base_row["Schema_Issues"] = "|".join(issues)
    base_row["Processing_Status"] = (
        "STOPPED_SCHEMA_ERROR" if severe else "READY_FOR_PROTOCOL_SCAN"
    )
    return MatFileScan(
        batch_id=batch_id,
        cell_id=cell_id,
        source_file=path.name,
        path=path,
        payload=payload,
        cycles=tuple(cycles),
        summary_capacities_Ah=summary_capacity,
        adapter=adapter,
        schema_row=base_row,
    )


def assign_schema_variants(scans: list[MatFileScan]) -> list[MatFileScan]:
    known_signature = json.dumps(
        {
            "capacity": "capacity_Ah",
            "current": "current_A",
            "description": "description",
            "relative_time": "relative_time_min",
            "summary_cycle_life": "cycle_life",
            "summary_discharge_capacity": "discharge_capacity_Ah",
            "voltage": "voltage_V",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    signatures = sorted(
        {scan.adapter.signature for scan in scans if scan.adapter is not None}
    )
    variant_by_signature: dict[str, str] = {}
    next_number = 1
    if known_signature in signatures:
        variant_by_signature[known_signature] = "XJTU_SCHEMA_V1"
        next_number = 2
    for signature in signatures:
        if signature not in variant_by_signature:
            variant_by_signature[signature] = f"XJTU_SCHEMA_V{next_number}"
            next_number += 1

    updated: list[MatFileScan] = []
    for scan in scans:
        if scan.adapter is None:
            updated.append(scan)
            continue
        variant = variant_by_signature[scan.adapter.signature]
        row = dict(scan.schema_row)
        row["Schema_Variant_ID"] = variant
        updated.append(
            replace(scan, adapter=replace(scan.adapter, variant_id=variant), schema_row=row)
        )
    return updated


def time_to_seconds(values: np.ndarray, unit: str) -> np.ndarray:
    if unit == "MINUTES":
        return values * 60.0
    if unit == "SECONDS":
        return values
    raise ValueError(f"Unsupported time unit: {unit}")


def adapter_mapping_text(adapter: SchemaAdapter) -> str:
    return json.dumps(
        {
            "description": adapter.description_field,
            "voltage": adapter.voltage_field,
            "current": adapter.current_field,
            "capacity": adapter.capacity_field,
            "relative_time": adapter.time_field,
            "time_unit": adapter.time_unit,
            "summary_discharge_capacity": adapter.summary_discharge_capacity_field,
            "summary_cycle_life": adapter.summary_cycle_life_field,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
