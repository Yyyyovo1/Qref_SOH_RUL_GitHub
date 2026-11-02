"""Target-independent physical feature extractors for XJTU and HUST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import (
    CV_MAX_VOLTAGE_RANGE_V,
    CV_TRANSITION_CURRENT_FRACTION,
    CV_TRANSITION_VOLTAGE_BELOW_MAX_V,
    FEATURE_NAMES,
    FEATURE_STAGE,
    FEATURE_UNITS,
    HUBER_CONVERGENCE_TOLERANCE,
    HUBER_MAX_ITERATIONS,
    HUBER_TUNING_CONSTANT,
    HUST_PRE_CV_PLATEAU_MIN_CURRENT_A,
    HUST_STATUS_CHARGE_CCCV,
    HUST_STATUS_CHARGE_HIGH,
    HUST_STATUS_DISCHARGE_FIRST,
    MIN_VALID_STAGE_SAMPLES,
    NEGATIVE_DISCHARGE_CURRENT_THRESHOLD_A,
    POSITIVE_CHARGE_CURRENT_THRESHOLD_A,
    REST_CURRENT_ABS_THRESHOLD_A,
    ROBUST_SLOPE_CENTRAL_TIME_HIGH,
    ROBUST_SLOPE_CENTRAL_TIME_LOW,
    TERMINAL_CV_WINDOW_SECONDS,
    XJTU_CC_UPPER_QUANTILE,
)


@dataclass(frozen=True)
class Stage:
    name: str
    indices: np.ndarray
    raw_sample_count: int

    @property
    def start_index(self) -> int | None:
        return int(self.indices[0]) if self.indices.size else None

    @property
    def end_index(self) -> int | None:
        return int(self.indices[-1]) if self.indices.size else None


@dataclass(frozen=True)
class FeatureResult:
    name: str
    value: float
    unit: str
    status: str
    reason_code: str
    stage: str
    stage_start_index: int | None
    stage_end_index: int | None
    stage_start_time_s: float
    stage_end_time_s: float
    raw_sample_count: int
    valid_sample_count: int
    detail: str = ""


@dataclass
class ExtractionResult:
    features: dict[str, FeatureResult]
    overall_flag: str
    overall_reason_codes: str
    pre_discharge_state_type: str
    raw_time_unit: str
    time_conversion_factor_to_seconds: float
    boundaries: dict[str, int | None]
    arrays_for_plot: dict[str, np.ndarray]


def _as_float_1d(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel(order="C")


def _as_text_1d(values: Any) -> np.ndarray:
    return np.asarray(values).astype(str).ravel(order="C")


def _validate_lengths(*arrays: np.ndarray) -> None:
    lengths = {array.size for array in arrays}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError(f"RAW_ARRAY_LENGTH_OR_EMPTY_ERROR:{sorted(lengths)}")


def contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask.astype(bool), False]))
    return [(int(start), int(stop)) for start, stop in boundaries.reshape(-1, 2)]


def _duration(time_seconds: np.ndarray, indices: np.ndarray) -> float:
    if indices.size < 2:
        return 0.0
    return float(time_seconds[indices[-1]] - time_seconds[indices[0]])


def _stage_monotonic(time_seconds: np.ndarray, indices: np.ndarray) -> bool:
    return bool(
        indices.size >= 2
        and np.isfinite(time_seconds[indices]).all()
        and np.all(np.diff(time_seconds[indices]) >= 0)
    )


def _robust_huber_slope(
    time_seconds: np.ndarray, values: np.ndarray
) -> tuple[float, int, str]:
    """Deterministic Huber IRLS slope on the central 20-80% physical time."""
    finite = np.isfinite(time_seconds) & np.isfinite(values)
    time = np.asarray(time_seconds[finite], dtype=float)
    y = np.asarray(values[finite], dtype=float)
    if time.size < MIN_VALID_STAGE_SAMPLES:
        return np.nan, int(time.size), "INSUFFICIENT_VALID_STAGE_SAMPLES"
    start = float(time[0])
    stop = float(time[-1])
    duration = stop - start
    if not np.isfinite(duration) or duration <= 0:
        return np.nan, 0, "NONPOSITIVE_STAGE_DURATION"
    low = start + ROBUST_SLOPE_CENTRAL_TIME_LOW * duration
    high = start + ROBUST_SLOPE_CENTRAL_TIME_HIGH * duration
    central = (time >= low) & (time <= high)
    x = time[central]
    y = y[central]
    if x.size < MIN_VALID_STAGE_SAMPLES or np.unique(x).size < 2:
        return np.nan, int(x.size), "INSUFFICIENT_CENTRAL_TIME_SAMPLES"
    x_center = x - float(np.mean(x))
    design = np.column_stack([np.ones(x.size), x_center])
    try:
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan, int(x.size), "ROBUST_SLOPE_LINEAR_ALGEBRA_ERROR"
    for _ in range(HUBER_MAX_ITERATIONS):
        residual = y - design @ beta
        residual_center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - residual_center)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            break
        cutoff = HUBER_TUNING_CONSTANT * scale
        absolute = np.abs(residual)
        weights = np.ones_like(absolute)
        high_residual = absolute > cutoff
        weights[high_residual] = cutoff / absolute[high_residual]
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_y = y * np.sqrt(weights)
        try:
            updated = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.nan, int(x.size), "ROBUST_SLOPE_LINEAR_ALGEBRA_ERROR"
        if np.max(np.abs(updated - beta)) <= HUBER_CONVERGENCE_TOLERANCE:
            beta = updated
            break
        beta = updated
    slope = float(beta[1])
    if not np.isfinite(slope):
        return np.nan, int(x.size), "NONFINITE_ROBUST_SLOPE"
    return slope, int(x.size), "PASS"


def _sequential_elapsed(time_seconds: np.ndarray) -> np.ndarray:
    """Create a monotonic plotting axis without interpolating measurements."""
    elapsed = np.zeros(time_seconds.size, dtype=float)
    if time_seconds.size < 2:
        return elapsed
    differences = np.diff(time_seconds)
    positive = differences[np.isfinite(differences) & (differences >= 0)]
    default_step = float(np.median(positive[positive > 0])) if np.any(positive > 0) else 0.0
    increments = np.where(np.isfinite(differences) & (differences >= 0), differences, default_step)
    elapsed[1:] = np.cumsum(increments)
    return elapsed


def _missing_feature(name: str, status: str, reason: str) -> FeatureResult:
    return FeatureResult(
        name=name,
        value=np.nan,
        unit=FEATURE_UNITS[name],
        status=status,
        reason_code=reason,
        stage=FEATURE_STAGE[name],
        stage_start_index=None,
        stage_end_index=None,
        stage_start_time_s=np.nan,
        stage_end_time_s=np.nan,
        raw_sample_count=0,
        valid_sample_count=0,
    )


def invalid_extraction_result(
    *,
    reason: str,
    raw_time_unit: str,
    time_conversion_factor_to_seconds: float,
) -> ExtractionResult:
    """Preserve an observation when its raw curve cannot be extracted safely."""
    features = {
        name: _missing_feature(name, "INVALID_EXTRACTION", reason)
        for name in FEATURE_NAMES
    }
    return ExtractionResult(
        features=features,
        overall_flag="INVALID_EXTRACTION",
        overall_reason_codes=reason,
        pre_discharge_state_type="UNAVAILABLE",
        raw_time_unit=raw_time_unit,
        time_conversion_factor_to_seconds=time_conversion_factor_to_seconds,
        boundaries={
            "charge_start_index": None,
            "charge_end_index": None,
            "cc_start_index": None,
            "cc_end_index": None,
            "cv_start_index": None,
            "cv_end_index": None,
            "pre_discharge_index": None,
            "discharge_start_index": None,
        },
        arrays_for_plot={
            "time_seconds": np.asarray([], dtype=float),
            "plot_elapsed_seconds": np.asarray([], dtype=float),
            "current_A": np.asarray([], dtype=float),
            "voltage_V": np.asarray([], dtype=float),
        },
    )


def _feature(
    name: str,
    value: float,
    status: str,
    reason: str,
    stage: Stage,
    time_seconds: np.ndarray,
    *,
    valid_sample_count: int | None = None,
    detail: str = "",
) -> FeatureResult:
    count = int(stage.indices.size) if valid_sample_count is None else int(valid_sample_count)
    return FeatureResult(
        name=name,
        value=float(value) if np.isfinite(value) else np.nan,
        unit=FEATURE_UNITS[name],
        status=status,
        reason_code=reason,
        stage=FEATURE_STAGE[name],
        stage_start_index=stage.start_index,
        stage_end_index=stage.end_index,
        stage_start_time_s=(
            float(time_seconds[stage.start_index]) if stage.start_index is not None else np.nan
        ),
        stage_end_time_s=(
            float(time_seconds[stage.end_index]) if stage.end_index is not None else np.nan
        ),
        raw_sample_count=int(stage.raw_sample_count),
        valid_sample_count=count,
        detail=detail,
    )


def _find_cv_split(
    indices: np.ndarray,
    current_A: np.ndarray,
    voltage_V: np.ndarray,
    *,
    dataset: str,
) -> tuple[Stage | None, Stage | None, str]:
    if indices.size < 2:
        return None, None, "NO_VALID_CHARGE_STAGE_FOR_CV_SPLIT"
    stage_current = current_A[indices]
    stage_voltage = voltage_V[indices]
    finite_positive = (
        np.isfinite(stage_current)
        & np.isfinite(stage_voltage)
        & (stage_current > POSITIVE_CHARGE_CURRENT_THRESHOLD_A)
    )
    indices = indices[finite_positive]
    if indices.size < 2:
        return None, None, "NO_VALID_POSITIVE_CHARGE_SAMPLES"
    stage_current = current_A[indices]
    stage_voltage = voltage_V[indices]
    maximum_voltage = float(np.max(stage_voltage))
    if dataset == "HUST":
        plateau_values = stage_current[stage_current >= HUST_PRE_CV_PLATEAU_MIN_CURRENT_A]
        plateau = float(np.median(plateau_values)) if plateau_values.size else np.nan
    else:
        quantile = float(np.quantile(stage_current, XJTU_CC_UPPER_QUANTILE))
        plateau_values = stage_current[stage_current >= quantile]
        plateau = float(np.median(plateau_values)) if plateau_values.size else np.nan
    if not np.isfinite(plateau) or plateau <= 0:
        return None, None, "NO_VALID_PRE_CV_CC_PLATEAU"
    candidates = np.flatnonzero(
        (stage_current <= plateau * CV_TRANSITION_CURRENT_FRACTION)
        & (stage_voltage >= maximum_voltage - CV_TRANSITION_VOLTAGE_BELOW_MAX_V)
    )
    if not candidates.size or int(candidates[0]) == 0:
        return None, None, "NO_PHYSICALLY_PLAUSIBLE_CV_TRANSITION"
    split = int(candidates[0])
    cc = Stage("FINAL_PRE_CV_CC", indices[:split], split)
    cv = Stage("CHARGE_CV", indices[split:], indices.size - split)
    return cc, cv, "PASS"


def _finalize(
    *,
    dataset: str,
    time_seconds: np.ndarray,
    current_A: np.ndarray,
    voltage_V: np.ndarray,
    main_charge: Stage | None,
    final_cc: Stage | None,
    cv: Stage | None,
    pre_discharge_index: int | None,
    discharge_start_index: int | None,
    cv_split_reason: str,
    raw_time_unit: str,
    time_factor: float,
) -> ExtractionResult:
    results: dict[str, FeatureResult] = {}

    if main_charge is None or main_charge.indices.size < 2:
        for name in ("Charge_Start_V", "Charge_End_V", "Charge_Max_V"):
            results[name] = _missing_feature(name, "MISSING", "NO_VALID_MAIN_CHARGE")
    else:
        charge_voltage = voltage_V[main_charge.indices]
        results["Charge_Start_V"] = _feature(
            "Charge_Start_V", charge_voltage[0], "PASS", "PASS", main_charge, time_seconds
        )
        results["Charge_End_V"] = _feature(
            "Charge_End_V", charge_voltage[-1], "PASS", "PASS", main_charge, time_seconds
        )
        results["Charge_Max_V"] = _feature(
            "Charge_Max_V", np.max(charge_voltage), "PASS", "PASS", main_charge, time_seconds
        )

    cc_valid = bool(
        final_cc is not None
        and final_cc.indices.size >= MIN_VALID_STAGE_SAMPLES
        and _stage_monotonic(time_seconds, final_cc.indices)
        and _duration(time_seconds, final_cc.indices) > 0
    )
    if not cc_valid:
        reason = (
            "NO_VALID_FINAL_PRE_CV_CC"
            if final_cc is None
            else "FINAL_PRE_CV_CC_INSUFFICIENT_SAMPLES_OR_TIME"
        )
        results["Pre_CV_CC_Median_Current_A"] = _missing_feature(
            "Pre_CV_CC_Median_Current_A", "MISSING", reason
        )
        results["Charge_Voltage_Rise_Slope_V_per_s"] = _missing_feature(
            "Charge_Voltage_Rise_Slope_V_per_s", "MISSING", reason
        )
    else:
        assert final_cc is not None
        results["Pre_CV_CC_Median_Current_A"] = _feature(
            "Pre_CV_CC_Median_Current_A",
            np.median(current_A[final_cc.indices]),
            "PASS",
            "PASS",
            final_cc,
            time_seconds,
        )
        slope, central_count, slope_reason = _robust_huber_slope(
            time_seconds[final_cc.indices], voltage_V[final_cc.indices]
        )
        results["Charge_Voltage_Rise_Slope_V_per_s"] = _feature(
            "Charge_Voltage_Rise_Slope_V_per_s",
            slope,
            "PASS" if slope_reason == "PASS" else "CHECK",
            slope_reason,
            final_cc,
            time_seconds,
            valid_sample_count=central_count,
            detail=(
                f"central_time_fraction={ROBUST_SLOPE_CENTRAL_TIME_LOW:.2f}-"
                f"{ROBUST_SLOPE_CENTRAL_TIME_HIGH:.2f};robust_method=HUBER_IRLS"
            ),
        )

    cv_valid = bool(
        cv is not None
        and cv.indices.size >= MIN_VALID_STAGE_SAMPLES
        and _stage_monotonic(time_seconds, cv.indices)
        and _duration(time_seconds, cv.indices) > 0
    )
    cv_reason = cv_split_reason
    cv_slope = np.nan
    cv_central_count = 0
    if cv_valid:
        assert cv is not None
        cv_voltage = voltage_V[cv.indices]
        cv_current = current_A[cv.indices]
        cv_slope, cv_central_count, cv_slope_reason = _robust_huber_slope(
            time_seconds[cv.indices], cv_current
        )
        voltage_range = float(np.max(cv_voltage) - np.min(cv_voltage))
        taper_valid = bool(
            cv_slope_reason == "PASS"
            and cv_slope < 0
            and cv_current[-1] < cv_current[0]
            and voltage_range <= CV_MAX_VOLTAGE_RANGE_V
        )
        if not taper_valid:
            cv_valid = False
            cv_reason = (
                "CV_PHYSICAL_PLAUSIBILITY_FAILED:"
                f"slope_reason={cv_slope_reason};slope={cv_slope};"
                f"start_I={cv_current[0]};end_I={cv_current[-1]};"
                f"voltage_range={voltage_range}"
            )
    if not cv_valid:
        status = "MISSING" if cv is None else "CHECK"
        for name in (
            "CV_Current_Taper_Slope_A_per_s",
            "CV_Duration_s",
            "CV_End_Current_A",
            "CV_Median_Current_A",
            "Charge_Terminal_Voltage_Std_V",
        ):
            results[name] = _missing_feature(name, status, cv_reason)
    else:
        assert cv is not None
        duration = _duration(time_seconds, cv.indices)
        results["CV_Current_Taper_Slope_A_per_s"] = _feature(
            "CV_Current_Taper_Slope_A_per_s",
            cv_slope,
            "PASS",
            "PASS",
            cv,
            time_seconds,
            valid_sample_count=cv_central_count,
            detail=(
                f"central_time_fraction={ROBUST_SLOPE_CENTRAL_TIME_LOW:.2f}-"
                f"{ROBUST_SLOPE_CENTRAL_TIME_HIGH:.2f};robust_method=HUBER_IRLS"
            ),
        )
        results["CV_Duration_s"] = _feature(
            "CV_Duration_s", duration, "PASS", "PASS", cv, time_seconds
        )
        results["CV_End_Current_A"] = _feature(
            "CV_End_Current_A",
            current_A[cv.indices[-1]],
            "PASS",
            "PASS",
            cv,
            time_seconds,
        )
        results["CV_Median_Current_A"] = _feature(
            "CV_Median_Current_A",
            np.median(current_A[cv.indices]),
            "PASS",
            "PASS",
            cv,
            time_seconds,
        )
        terminal_start_time = float(time_seconds[cv.indices[-1]]) - TERMINAL_CV_WINDOW_SECONDS
        terminal_indices = cv.indices[time_seconds[cv.indices] >= terminal_start_time]
        terminal_stage = Stage(
            "CHARGE_CV_TERMINAL_WINDOW",
            terminal_indices,
            int(terminal_indices.size),
        )
        terminal_span = _duration(time_seconds, terminal_indices)
        short = duration < TERMINAL_CV_WINDOW_SECONDS
        if terminal_indices.size < 2:
            results["Charge_Terminal_Voltage_Std_V"] = _missing_feature(
                "Charge_Terminal_Voltage_Std_V",
                "CHECK",
                "TERMINAL_CV_WINDOW_HAS_FEWER_THAN_2_SAMPLES",
            )
        else:
            results["Charge_Terminal_Voltage_Std_V"] = _feature(
                "Charge_Terminal_Voltage_Std_V",
                np.std(voltage_V[terminal_indices], ddof=1),
                "CHECK" if short else "PASS",
                "SHORT_TERMINAL_WINDOW" if short else "PASS",
                terminal_stage,
                time_seconds,
                detail=(
                    f"requested_window_s={TERMINAL_CV_WINDOW_SECONDS};"
                    f"actual_window_s={terminal_span};SHORT_TERMINAL_WINDOW={'YES' if short else 'NO'}"
                ),
            )

    if pre_discharge_index is None:
        pre_state = "UNAVAILABLE"
        results["Pre_Discharge_Voltage_V"] = _missing_feature(
            "Pre_Discharge_Voltage_V", "MISSING", "PRE_DISCHARGE_POINT_NOT_FOUND"
        )
    else:
        previous_current = float(current_A[pre_discharge_index])
        if previous_current > POSITIVE_CHARGE_CURRENT_THRESHOLD_A:
            pre_state = "AFTER_CHARGE"
        elif abs(previous_current) <= REST_CURRENT_ABS_THRESHOLD_A:
            pre_state = "AFTER_REST"
        else:
            pre_state = "OTHER_VALID_PRE_DISCHARGE_STATE"
        point = Stage(
            "PRE_DISCHARGE_POINT",
            np.asarray([pre_discharge_index], dtype=int),
            1,
        )
        results["Pre_Discharge_Voltage_V"] = _feature(
            "Pre_Discharge_Voltage_V",
            voltage_V[pre_discharge_index],
            "PASS",
            "PASS",
            point,
            time_seconds,
            detail=f"Pre_Discharge_State_Type={pre_state}",
        )

    missing_names = [name for name in FEATURE_NAMES if name not in results]
    if missing_names:
        raise RuntimeError(f"INTERNAL_FEATURE_RESULT_MISSING:{missing_names}")
    statuses = [results[name].status for name in FEATURE_NAMES]
    reasons = sorted(
        {results[name].reason_code for name in FEATURE_NAMES if results[name].reason_code != "PASS"}
    )
    return ExtractionResult(
        features={name: results[name] for name in FEATURE_NAMES},
        overall_flag="PASS" if all(status == "PASS" for status in statuses) else "CHECK",
        overall_reason_codes="PASS" if not reasons else "|".join(reasons),
        pre_discharge_state_type=pre_state,
        raw_time_unit=raw_time_unit,
        time_conversion_factor_to_seconds=time_factor,
        boundaries={
            "charge_start_index": main_charge.start_index if main_charge else None,
            "charge_end_index": main_charge.end_index if main_charge else None,
            "cc_start_index": final_cc.start_index if final_cc else None,
            "cc_end_index": final_cc.end_index if final_cc else None,
            "cv_start_index": cv.start_index if cv else None,
            "cv_end_index": cv.end_index if cv else None,
            "pre_discharge_index": pre_discharge_index,
            "discharge_start_index": discharge_start_index,
        },
        arrays_for_plot={
            "time_seconds": time_seconds,
            "plot_elapsed_seconds": _sequential_elapsed(time_seconds),
            "current_A": current_A,
            "voltage_V": voltage_V,
        },
    )


def extract_xjtu_features(
    *,
    relative_time_min: Any,
    current_A: Any,
    voltage_V: Any,
) -> ExtractionResult:
    """Extract XJTU features from time/current/voltage only."""
    time_raw = _as_float_1d(relative_time_min)
    current = _as_float_1d(current_A)
    voltage = _as_float_1d(voltage_V)
    _validate_lengths(time_raw, current, voltage)
    time_seconds = time_raw * 60.0
    finite = np.isfinite(time_seconds) & np.isfinite(current) & np.isfinite(voltage)

    discharge_mask = finite & (current < NEGATIVE_DISCHARGE_CURRENT_THRESHOLD_A)
    discharge_spans = contiguous_spans(discharge_mask)
    discharge_span = max(
        discharge_spans,
        key=lambda span: (span[1] - span[0], _duration(time_seconds, np.arange(span[0], span[1]))),
        default=None,
    )
    discharge_start = int(discharge_span[0]) if discharge_span else None

    positive_mask = finite & (current > POSITIVE_CHARGE_CURRENT_THRESHOLD_A)
    charge_spans = contiguous_spans(positive_mask)
    if discharge_start is not None:
        charge_spans = [span for span in charge_spans if span[1] <= discharge_start]
    charge_span = max(
        charge_spans,
        key=lambda span: (span[1] - span[0], _duration(time_seconds, np.arange(span[0], span[1]))),
        default=None,
    )
    main_charge = (
        Stage("MAIN_CHARGE", np.arange(charge_span[0], charge_span[1]), charge_span[1] - charge_span[0])
        if charge_span
        else None
    )
    if main_charge is not None:
        final_cc, cv, cv_reason = _find_cv_split(
            main_charge.indices, current, voltage, dataset="XJTU"
        )
    else:
        final_cc, cv, cv_reason = None, None, "NO_VALID_MAIN_CHARGE"
    pre_index = None
    if discharge_start is not None:
        candidates = np.flatnonzero(finite & (np.arange(current.size) < discharge_start))
        pre_index = int(candidates[-1]) if candidates.size else None
    return _finalize(
        dataset="XJTU",
        time_seconds=time_seconds,
        current_A=current,
        voltage_V=voltage,
        main_charge=main_charge,
        final_cc=final_cc,
        cv=cv,
        pre_discharge_index=pre_index,
        discharge_start_index=discharge_start,
        cv_split_reason=cv_reason,
        raw_time_unit="MINUTES",
        time_factor=60.0,
    )


def extract_hust_features(
    *,
    time_s: Any,
    current_mA: Any,
    voltage_V: Any,
    status: Any,
) -> ExtractionResult:
    """Extract HUST features using validated status-to-physical-stage mapping."""
    time_seconds = _as_float_1d(time_s)
    current = _as_float_1d(current_mA) / 1000.0
    voltage = _as_float_1d(voltage_V)
    statuses = _as_text_1d(status)
    _validate_lengths(time_seconds, current, voltage, statuses)
    finite = np.isfinite(time_seconds) & np.isfinite(current) & np.isfinite(voltage)
    high = np.flatnonzero(
        finite
        & (statuses == HUST_STATUS_CHARGE_HIGH)
        & (current > POSITIVE_CHARGE_CURRENT_THRESHOLD_A)
    )
    cccv = np.flatnonzero(
        finite
        & (statuses == HUST_STATUS_CHARGE_CCCV)
        & (current > POSITIVE_CHARGE_CURRENT_THRESHOLD_A)
    )
    combined = np.sort(np.concatenate([high, cccv])) if high.size or cccv.size else np.asarray([], dtype=int)
    main_charge = Stage("MAIN_CHARGE", combined, int(high.size + cccv.size)) if combined.size else None
    if cccv.size:
        final_cc, cv, cv_reason = _find_cv_split(cccv, current, voltage, dataset="HUST")
    else:
        final_cc, cv, cv_reason = None, None, "NO_VALID_HUST_CCCV_STATUS"
    discharge_candidates = np.flatnonzero(
        finite
        & (statuses == HUST_STATUS_DISCHARGE_FIRST)
        & (current < NEGATIVE_DISCHARGE_CURRENT_THRESHOLD_A)
    )
    discharge_start = int(discharge_candidates[0]) if discharge_candidates.size else None
    pre_index = None
    if discharge_start is not None:
        candidates = np.flatnonzero(finite & (np.arange(current.size) < discharge_start))
        pre_index = int(candidates[-1]) if candidates.size else None
    return _finalize(
        dataset="HUST",
        time_seconds=time_seconds,
        current_A=current,
        voltage_V=voltage,
        main_charge=main_charge,
        final_cc=final_cc,
        cv=cv,
        pre_discharge_index=pre_index,
        discharge_start_index=discharge_start,
        cv_split_reason=cv_reason,
        raw_time_unit="SECONDS",
        time_factor=1.0,
    )


def provenance_rows(
    *,
    result: ExtractionResult,
    dataset: str,
    batch_id: int | str | None,
    cell_id: str,
    raw_cycle_index: int,
    raw_source_file: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in FEATURE_NAMES:
        feature = result.features[name]
        rows.append(
            {
                "Dataset": dataset,
                "Batch_ID": batch_id if batch_id is not None else "",
                "Cell_ID": cell_id,
                "Raw_Cycle_Index": int(raw_cycle_index),
                "Feature_Name": name,
                "Raw_Source_File": raw_source_file,
                "Physical_Stage": feature.stage,
                "Stage_Start_Index": feature.stage_start_index,
                "Stage_End_Index": feature.stage_end_index,
                "Stage_Start_Time_s": feature.stage_start_time_s,
                "Stage_End_Time_s": feature.stage_end_time_s,
                "Raw_Sample_Count": feature.raw_sample_count,
                "Valid_Sample_Count": feature.valid_sample_count,
                "Feature_Value": feature.value,
                "Feature_Unit": feature.unit,
                "Extraction_Status": feature.status,
                "Extraction_Reason_Code": feature.reason_code,
                "Raw_Time_Unit": result.raw_time_unit,
                "Time_Conversion_Factor_To_Seconds": result.time_conversion_factor_to_seconds,
                "Pre_Discharge_State_Type": result.pre_discharge_state_type,
                "Extraction_Detail": feature.detail,
            }
        )
    return rows


def matrix_row(
    *,
    result: ExtractionResult,
    dataset: str,
    batch_id: int | str | None,
    cell_id: str,
    raw_cycle_index: int,
    fold_id: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "Dataset": dataset,
        "Cell_ID": cell_id,
        "Raw_Cycle_Index": int(raw_cycle_index),
        "Fold_ID": int(fold_id),
        "Physical_Observation_Key": (
            f"{dataset}|{batch_id}|{cell_id}|{int(raw_cycle_index)}"
            if batch_id is not None
            else f"{dataset}|{cell_id}|{int(raw_cycle_index)}"
        ),
    }
    if batch_id is not None:
        row["Batch_ID"] = int(batch_id)
    for name in FEATURE_NAMES:
        row[name] = result.features[name].value
    valid_count = sum(np.isfinite(row[name]) for name in FEATURE_NAMES)
    row["Feature_Set_B_Raw_Cycle_Index"] = int(raw_cycle_index)
    row["Feature_Extraction_Flag"] = result.overall_flag
    row["Feature_Extraction_Reason_Codes"] = result.overall_reason_codes
    row["Feature_Set_A_Valid_Count"] = valid_count
    row["Feature_Set_A_Missing_Count"] = len(FEATURE_NAMES) - valid_count
    row["Feature_Set_A_Complete"] = "YES" if valid_count == len(FEATURE_NAMES) else "NO"
    row["Pre_Discharge_State_Type"] = result.pre_discharge_state_type
    return row


__all__ = [
    "ExtractionResult",
    "FeatureResult",
    "Stage",
    "contiguous_spans",
    "extract_hust_features",
    "extract_xjtu_features",
    "invalid_extraction_result",
    "matrix_row",
    "provenance_rows",
]
