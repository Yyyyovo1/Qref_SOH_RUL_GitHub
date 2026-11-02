from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .sensitivity import QREF_DEFINITIONS


PASS_COLOR = "#2C7FB8"
CHECK_COLOR = "#E69F00"
INVALID_COLOR = "#D73027"
RAW_LINE_COLOR = "#B8C2CC"
QREF_COLORS = {
    "nominal": "#1B9E77",
    "initial_test": "#7570B3",
    "first_standard": "#D95F02",
    "early_median5": "#E7298A",
}
QREF_LABELS = {
    "nominal": "Nominal",
    "initial_test": "Initial test",
    "first_standard": "First standard",
    "early_median5": "Early median-5",
}


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_capacity_curves(
    qc: pd.DataFrame,
    qref: pd.DataFrame,
    figures_dir: Path,
    config: dict[str, Any],
) -> list[Path]:
    paths: list[Path] = []
    qref_lookup = qref.set_index("Cell_ID")
    dpi = int(config["figure_dpi"])
    for cell_id, cell in qc.groupby("Cell_ID", sort=True):
        ordered = cell.sort_values("Raw_Cycle_Index")
        x = ordered["Raw_Cycle_Index"].to_numpy(dtype=float)
        y = ordered["Summary_Discharge_Capacity_Ah"].to_numpy(dtype=float)
        flags = ordered["Auto_QC_Flag"].astype(str).to_numpy()
        line_y = y.copy()
        line_y[flags == "INVALID_CANDIDATE"] = np.nan

        fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
        ax.plot(x, line_y, color=RAW_LINE_COLOR, linewidth=0.9, label="Observed non-invalid sequence")
        for flag, color, marker, size in (
            ("PASS", PASS_COLOR, "o", 13),
            ("CHECK", CHECK_COLOR, "^", 45),
            ("INVALID_CANDIDATE", INVALID_COLOR, "x", 40),
        ):
            mask = flags == flag
            if np.any(mask):
                ax.scatter(
                    x[mask],
                    y[mask],
                    s=size,
                    marker=marker,
                    color=color,
                    linewidths=1.1,
                    label=flag,
                    zorder=3,
                )

        qr = qref_lookup.loc[cell_id]
        threshold_specs = (
            ("Qref_nominal_Ah", "0.8 × nominal Qref", "#1B9E77", "--"),
            ("Qref_initial_test_Ah", "0.8 × initial-test Qref", "#7570B3", ":"),
            ("Qref_early_median5_Ah", "0.8 × early median-5 Qref", "#E7298A", "-."),
        )
        for column, label, color, style in threshold_specs:
            reference = float(qr[column]) if pd.notna(qr[column]) else np.nan
            if np.isfinite(reference):
                ax.axhline(
                    float(config["EOL_SOH_threshold"]) * reference,
                    color=color,
                    linestyle=style,
                    linewidth=1.25,
                    label=label,
                )

        ax.set_title(f"XJTU Batch-1 {cell_id} discharge capacity with protocol QC")
        ax.set_xlabel("Raw Cycle Index")
        ax.set_ylabel("Summary Discharge Capacity (Ah)")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", fontsize=8.5, frameon=False, ncol=2)
        path = figures_dir / f"capacity_curve_{cell_id}.png"
        _save_figure(fig, path, dpi)
        paths.append(path)
    return paths


def plot_qref_comparison(
    qref: pd.DataFrame,
    figures_dir: Path,
    config: dict[str, Any],
) -> Path:
    ordered = qref.sort_values("Cell_ID")
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    columns = {
        "nominal": "Qref_nominal_Ah",
        "initial_test": "Qref_initial_test_Ah",
        "first_standard": "Qref_first_standard_Ah",
        "early_median5": "Qref_early_median5_Ah",
    }
    for definition in QREF_DEFINITIONS:
        ax.plot(
            positions,
            ordered[columns[definition]].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
            color=QREF_COLORS[definition],
            label=QREF_LABELS[definition],
        )
    ax.set_xticks(positions, ordered["Cell_ID"].tolist(), rotation=25, ha="right")
    ax.set_ylabel("Reference Capacity (Ah)")
    ax.set_title("XJTU Batch-1 reference-capacity definitions")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", frameon=False, ncol=2)
    path = figures_dir / "xjtu_qref_comparison.png"
    _save_figure(fig, path, int(config["figure_dpi"]))
    return path


def plot_eol_overview(
    eol: pd.DataFrame,
    figures_dir: Path,
    config: dict[str, Any],
) -> Path:
    relative = eol.loc[eol["Qref_Definition"].isin(QREF_DEFINITIONS)].copy()
    cells = sorted(relative["Cell_ID"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.8), sharex=False, constrained_layout=True)
    y_positions = {definition: index for index, definition in enumerate(QREF_DEFINITIONS)}

    for ax, cell_id in zip(axes.ravel(), cells, strict=True):
        cell = relative.loc[relative["Cell_ID"] == cell_id]
        last_valid = int(cell["Last_Valid_Raw_Cycle"].dropna().max())
        for row in cell.itertuples(index=False):
            y = y_positions[row.Qref_Definition]
            color = QREF_COLORS[row.Qref_Definition]
            if row.EOL_Status == "OBSERVED":
                ax.scatter(
                    int(row.EOL_Raw_Cycle),
                    y,
                    marker="o",
                    s=55,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
            elif row.EOL_Status == "RIGHT_CENSORED":
                ax.scatter(
                    last_valid,
                    y,
                    marker=">",
                    s=70,
                    facecolors="white",
                    edgecolors=color,
                    linewidth=1.6,
                )
        ax.axvline(last_valid, color="#9CA3AF", linestyle=":", linewidth=0.8)
        ax.set_title(cell_id, fontsize=10)
        ax.set_xlim(0, last_valid + max(15, int(0.04 * last_valid)))
        ax.set_yticks(
            list(y_positions.values()),
            [QREF_LABELS[definition] for definition in QREF_DEFINITIONS],
            fontsize=8,
        )
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Raw Cycle Index")

    fig.suptitle(
        "XJTU Batch-1 EOL sensitivity\n"
        "Filled circles: observed EOL; open triangles: right-censored at last valid observation (not EOL)",
        fontsize=13,
    )
    path = figures_dir / "xjtu_eol_sensitivity_overview.png"
    _save_figure(fig, path, int(config["figure_dpi"]))
    return path


def generate_all_plots(
    qc: pd.DataFrame,
    qref: pd.DataFrame,
    eol: pd.DataFrame,
    figures_dir: Path,
    config: dict[str, Any],
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=False)
    paths = plot_capacity_curves(qc, qref, figures_dir, config)
    paths.append(plot_qref_comparison(qref, figures_dir, config))
    paths.append(plot_eol_overview(eol, figures_dir, config))
    return paths

