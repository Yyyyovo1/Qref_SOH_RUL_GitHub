"""End-to-end orchestration for reproducible CALCE CS2 QC runs."""

from __future__ import annotations

import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy
import openpyxl
import pandas

from .config import (
    DEFAULT_PROFILE,
    PIPELINE_VERSION,
    QC,
    CellProfile,
    threshold_dict,
)
from .extract import benchmark_capacities, extract_workbook, file_sha256
from .qc import finalize_qc
from .report import (
    CYCLE_COLUMNS,
    MANIFEST_COLUMNS,
    build_qc_summary,
    plot_capacity_curve,
    write_csv,
)


def _new_run_directory(output_root: Path, run_prefix: str = "run") -> Path:
    if not run_prefix or not run_prefix.replace("_", "").replace("-", "").isalnum():
        raise ValueError("run_prefix must contain only letters, numbers, underscores, or hyphens")
    output_root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime(f"{run_prefix}_%Y%m%dT%H%M%SZ")
    candidate = output_root / stem
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def _configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("calce_qc")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="x")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def validate_benchmark(input_dir: Path, profile: CellProfile) -> dict[str, Any]:
    benchmark_path = input_dir / profile.benchmark_file
    if not benchmark_path.exists():
        raise FileNotFoundError(f"benchmark workbook not found: {benchmark_path}")
    stats_capacity, channel_capacity = benchmark_capacities(
        benchmark_path,
        cycle=profile.benchmark_cycle,
        cell_id=profile.cell_id,
    )
    stats_error = abs(stats_capacity - profile.benchmark_capacity_ah)
    channel_error = abs(channel_capacity - profile.benchmark_capacity_ah)
    if (
        stats_error > profile.benchmark_tolerance_ah
        or channel_error > profile.benchmark_tolerance_ah
    ):
        raise RuntimeError(
            f"Cycle {profile.benchmark_cycle} benchmark failed; batch processing stopped. "
            f"expected={profile.benchmark_capacity_ah:.12f} Ah, "
            f"Statistics={stats_capacity:.12f} Ah, Channel={channel_capacity:.12f} Ah"
        )
    return {
        "file": profile.benchmark_file,
        "cycle": profile.benchmark_cycle,
        "basis": profile.benchmark_basis,
        "expected_ah": profile.benchmark_capacity_ah,
        "tolerance_ah": profile.benchmark_tolerance_ah,
        "statistics_ah": stats_capacity,
        "channel_ah": channel_capacity,
        "statistics_error_ah": stats_error,
        "channel_error_ah": channel_error,
    }


def _file_role(file_name: str, profile: CellProfile) -> tuple[str, str]:
    duplicate = profile.duplicate_files.get(file_name)
    if duplicate is None:
        return "CANONICAL", ""
    return str(duplicate["role"]), str(duplicate["duplicate_of"])


def _assign_order_and_boundaries(
    manifests: list[dict[str, Any]],
    cycle_rows: list[dict[str, Any]],
    profile: CellProfile,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for manifest in manifests:
        role, duplicate_of = _file_role(str(manifest["File_Name"]), profile)
        manifest["File_Role"] = role
        manifest["Duplicate_Of"] = duplicate_of

    manifests.sort(
        key=lambda row: (
            str(row.get("Start_Time") or "9999"),
            0 if row["File_Role"] == "CANONICAL" else 1,
            str(row["File_Name"]),
        )
    )

    run_group = 0
    previous_interval: tuple[str, str] | None = None
    for order, manifest in enumerate(manifests, start=1):
        interval = (str(manifest.get("Start_Time")), str(manifest.get("End_Time")))
        if interval != previous_interval:
            run_group += 1
            previous_interval = interval
        manifest["File_Time_Order"] = order
        manifest["Chronological_Run_Group"] = run_group

    canonical = [row for row in manifests if row["File_Role"] == "CANONICAL"]
    tail_files = {file_name for file_name, _cycle in profile.known_boundary_tails}
    for index, manifest in enumerate(canonical):
        if index == 0:
            incoming = "START_OF_SERIES"
        elif canonical[index - 1]["File_Name"] in tail_files:
            incoming = "NEEDS_BOUNDARY_CHECK"
        else:
            incoming = "COUNTER_RESET_CONFIRMED"

        if index == len(canonical) - 1:
            outgoing = "END_OF_SERIES"
        elif manifest["File_Name"] in tail_files:
            outgoing = "NEEDS_BOUNDARY_CHECK"
        else:
            outgoing = "COUNTER_RESET_CONFIRMED"
        manifest["Incoming_Boundary_Status"] = incoming
        manifest["Outgoing_Boundary_Status"] = outgoing
        manifest["Boundary_Status"] = (
            "NEEDS_BOUNDARY_CHECK"
            if "NEEDS_BOUNDARY_CHECK" in (incoming, outgoing)
            else outgoing
        )
        evidence = []
        if manifest["File_Name"] in tail_files:
            affected = [
                f"Cycle {cycle}: {reason}"
                for (file_name, cycle), reason in profile.known_boundary_tails.items()
                if file_name == manifest["File_Name"]
            ]
            evidence.extend(affected)
        if incoming == "NEEDS_BOUNDARY_CHECK":
            evidence.append("previous file ends with an unstitched incomplete cycle")
        if not evidence:
            evidence.append("Channel counter and Local Cycle_Index restart at the file boundary")
        manifest["Boundary_Evidence"] = "; ".join(evidence)

    for manifest in manifests:
        if manifest["File_Role"] != "CANONICAL":
            manifest["Incoming_Boundary_Status"] = "DUPLICATE_CANDIDATE"
            manifest["Outgoing_Boundary_Status"] = "DUPLICATE_CANDIDATE"
            manifest["Boundary_Status"] = "NEEDS_BOUNDARY_CHECK"
            manifest["Boundary_Evidence"] = (
                f"duplicate/reference copy of {manifest['Duplicate_Of']}; retained but not globally indexed"
            )

    first_canonical_name = canonical[0]["File_Name"]
    for manifest in manifests:
        first_capacity = manifest.get("First_Cumulative_Discharge_Capacity")
        reset_confirmed = (
            first_capacity is not None
            and abs(float(first_capacity)) <= QC.counter_zero_tolerance_ah
        )
        manifest["Counter_Reset_Confirmed_From_Channel"] = reset_confirmed
        manifest["Counter_Reset_Suspected"] = manifest["File_Name"] != first_canonical_name
        manifest["Cycle_Index_Reset_Suspected"] = manifest["File_Name"] != first_canonical_name

    manifest_by_file = {str(row["File_Name"]): row for row in manifests}
    rows_by_file: dict[str, list[dict[str, Any]]] = {}
    for row in cycle_rows:
        rows_by_file.setdefault(str(row["Source_File"]), []).append(row)

    global_cycle = 0
    for manifest in manifests:
        file_name = str(manifest["File_Name"])
        file_rows = sorted(rows_by_file[file_name], key=lambda row: int(row["Local_Cycle_Index"]))
        for row in file_rows:
            row["File_Time_Order"] = manifest["File_Time_Order"]
            row["Chronological_Run_Group"] = manifest["Chronological_Run_Group"]
            row["File_Role"] = manifest["File_Role"]
            row["Duplicate_Of"] = manifest["Duplicate_Of"]
            key = (file_name, int(row["Local_Cycle_Index"]))
            if (
                manifest["File_Role"] != "CANONICAL"
                or key in profile.known_boundary_tails
            ):
                row["Boundary_Status"] = "NEEDS_BOUNDARY_CHECK"
            else:
                row["Boundary_Status"] = "CONFIRMED"
            if manifest["File_Role"] == "CANONICAL":
                global_cycle += 1
                row["Global_Cycle_Index"] = global_cycle
            else:
                row["Global_Cycle_Index"] = None

    ordered_cycle_rows = sorted(
        cycle_rows,
        key=lambda row: (int(row["File_Time_Order"]), int(row["Local_Cycle_Index"])),
    )
    return manifests, ordered_cycle_rows


def run_pipeline(
    profile: CellProfile = DEFAULT_PROFILE,
    input_dir: Path | None = None,
    output_root: Path | None = None,
    run_prefix: str = "run",
) -> dict[str, Any]:
    input_dir = profile.input_dir if input_dir is None else input_dir
    output_root = profile.output_root if output_root is None else output_root
    input_dir = input_dir.resolve()
    output_root = output_root.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"no .xlsx files found in: {input_dir}")

    hashes_before = {path.name: file_sha256(path) for path in files}
    benchmark = validate_benchmark(input_dir, profile)

    run_dir = _new_run_directory(output_root, run_prefix=run_prefix)
    logger = _configure_logger(run_dir / "processing_log.txt")
    try:
        logger.info("CALCE %s pipeline version %s", profile.cell_id, PIPELINE_VERSION)
        logger.info("Input directory: %s", input_dir)
        logger.info("Output directory: %s", run_dir)
        logger.info("Workbook count: %d", len(files))
        logger.info(
            "Runtime: Python %s | %s | openpyxl %s | pandas %s | numpy %s | matplotlib %s",
            platform.python_version(),
            platform.platform(),
            openpyxl.__version__,
            pandas.__version__,
            numpy.__version__,
            matplotlib.__version__,
        )
        logger.info("QC thresholds: %s", threshold_dict())
        logger.info(
            "Cycle %d benchmark PASS: expected %.10f Ah; Statistics %.12f Ah; Channel %.12f Ah",
            profile.benchmark_cycle,
            benchmark["expected_ah"],
            benchmark["statistics_ah"],
            benchmark["channel_ah"],
        )
        logger.info("Benchmark file: %s | basis: %s", benchmark["file"], benchmark["basis"])

        manifests: list[dict[str, Any]] = []
        cycle_rows: list[dict[str, Any]] = []
        for path in files:
            try:
                manifest, rows = extract_workbook(path, cell_id=profile.cell_id)
            except Exception as exc:
                logger.exception("Failed while processing file %s", path.name)
                raise RuntimeError(f"failed processing file {path.name}: {exc}") from exc
            manifests.append(manifest)
            cycle_rows.extend(rows)
            logger.info(
                "Read %s: Channel cycles=%d; Statistics cycles=%d",
                path.name,
                manifest["Channel_Cycle_Count"],
                manifest["Statistics_Cycle_Count"],
            )

        manifests, cycle_rows = _assign_order_and_boundaries(
            manifests,
            cycle_rows,
            profile,
        )
        cycle_frame = finalize_qc(
            cycle_rows,
            known_boundary_tails=profile.known_boundary_tails,
        )
        manifest_frame = pandas.DataFrame(manifests)
        summary_frame = build_qc_summary(cycle_frame)
        flagged_frame = cycle_frame[
            cycle_frame["QC_Flag"].ne("PASS")
            | cycle_frame["Boundary_Status"].eq("NEEDS_BOUNDARY_CHECK")
        ].copy()

        hashes_after = {path.name: file_sha256(path) for path in files}
        if hashes_before != hashes_after:
            changed = sorted(name for name in hashes_before if hashes_before[name] != hashes_after.get(name))
            raise RuntimeError(f"source workbook checksum changed unexpectedly: {changed}")
        logger.info("Verified all source workbook SHA-256 checksums are unchanged")

        write_csv(manifest_frame, run_dir / "file_manifest.csv", MANIFEST_COLUMNS)
        write_csv(cycle_frame, run_dir / "cycle_capacity_qc.csv", CYCLE_COLUMNS)
        write_csv(summary_frame, run_dir / "qc_summary.csv", summary_frame.columns)
        write_csv(flagged_frame, run_dir / "qc_flagged_cycles.csv", CYCLE_COLUMNS)
        plot_capacity_curve(
            cycle_frame,
            run_dir / "cycle_capacity_curve.png",
            cell_id=profile.cell_id,
        )

        canonical = cycle_frame[cycle_frame["File_Role"].eq("CANONICAL")]
        logger.info("Cycle records: all=%d; canonical=%d", len(cycle_frame), len(canonical))
        logger.info("Canonical QC counts: %s", canonical["QC_Flag"].value_counts().to_dict())
        logger.info(
            "Canonical jump references: valid=%d; no_valid=%d; capacity_jumps=%d",
            int(canonical["Jump_Reference_Status"].eq("VALID_JUMP_REFERENCE").sum()),
            int(canonical["Jump_Reference_Status"].eq("NO_VALID_JUMP_REFERENCE").sum()),
            int(canonical["Capacity_Jump_Flag"].fillna(False).sum()),
        )
        logger.info("Flagged output rows: %d", len(flagged_frame))
        logger.info("Completed successfully; six requested outputs written")
        return {
            "output_dir": run_dir,
            "cell_id": profile.cell_id,
            "benchmark": benchmark,
            "all_cycle_count": int(len(cycle_frame)),
            "canonical_cycle_count": int(len(canonical)),
            "canonical_qc_counts": canonical["QC_Flag"].value_counts().to_dict(),
            "flagged_count": int(len(flagged_frame)),
        }
    finally:
        _close_logger(logger)
