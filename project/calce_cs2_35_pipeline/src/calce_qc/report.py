"""Deterministic CSV summaries and the raw capacity/QC plot."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CSV_FLOAT_FORMAT = "%.12g"


MANIFEST_COLUMNS = [
    "Cell_ID",
    "File_Time_Order",
    "Chronological_Run_Group",
    "File_Name",
    "File_Role",
    "Duplicate_Of",
    "Start_Time",
    "End_Time",
    "First_Cycle_Index",
    "Last_Cycle_Index",
    "Channel_Cycle_Count",
    "Statistics_Cycle_Count",
    "First_Cumulative_Discharge_Capacity",
    "Last_Cumulative_Discharge_Capacity",
    "First_Statistics_Cumulative_Discharge_Capacity_Ah",
    "Last_Statistics_Cumulative_Discharge_Capacity_Ah",
    "Statistics_First_Cycle_Index",
    "Statistics_Last_Cycle_Index",
    "Counter_Reset_Suspected",
    "Counter_Reset_Confirmed_From_Channel",
    "Cycle_Index_Reset_Suspected",
    "Incoming_Boundary_Status",
    "Outgoing_Boundary_Status",
    "Boundary_Status",
    "Boundary_Evidence",
    "Statistics_Sheet_Present",
    "Sheet_Names",
    "Sheet_Shapes",
    "Channel_Repeated_Cycle_Transitions",
    "Statistics_Duplicate_Cycles",
    "Source_File_SHA256",
]


CYCLE_COLUMNS = [
    "Cell_ID",
    "Source_File",
    "File_Time_Order",
    "Chronological_Run_Group",
    "File_Role",
    "Duplicate_Of",
    "Boundary_Status",
    "Local_Cycle_Index",
    "Global_Cycle_Index",
    "Cycle_Start_Time",
    "Cycle_End_Time",
    "Single_Cycle_Discharge_Capacity_Ah",
    "Single_Cycle_Discharge_Capacity_mAh",
    "Capacity_Source",
    "Statistics_Cumulative_Discharge_Capacity_Ah",
    "Statistics_Previous_Cumulative_Discharge_Capacity_Ah",
    "Statistics_Single_Cycle_Discharge_Capacity_Ah",
    "Channel_Discharge_Capacity_Start_Ah",
    "Channel_Discharge_Capacity_End_Ah",
    "Channel_Single_Cycle_Discharge_Capacity_Ah",
    "Capacity_Agreement_Delta_Ah",
    "Discharge_Counter_Decrease_Count",
    "Channel_Record_Count",
    "Observed_Step_Indices",
    "CC_Stage_Present",
    "CC_Protocol_Complete",
    "CC_Step_Index",
    "CC_Record_Count",
    "CC_Duration_s",
    "CC_Mean_Current_A",
    "CC_Start_Voltage_V",
    "CC_End_Voltage_V",
    "CC_Charge_Delta_Ah",
    "CV_Stage_Present",
    "CV_Protocol_Complete",
    "CV_Selection_Basis",
    "CV_Step_Index",
    "CV_Record_Count",
    "CV_Duration_s",
    "CV_Average_Voltage_V",
    "CV_Start_Voltage_V",
    "CV_End_Voltage_V",
    "CV_Start_Current_A",
    "CV_End_Current_A",
    "CV_Charge_Delta_Ah",
    "Main_Discharge_Stage_Present",
    "Main_Discharge_Protocol_Complete",
    "Discharge_Step_Index",
    "Discharge_Record_Count",
    "Main_Discharge_Mean_Current_A",
    "Discharge_Start_Voltage_V",
    "Discharge_Cutoff_Voltage_V",
    "Discharge_Duration_s",
    "Main_Discharge_Segment_Delta_Ah",
    "Incomplete_CV_Flag",
    "Incomplete_Discharge_Flag",
    "Nonpositive_Capacity_Flag",
    "Capacity_Jump_Flag",
    "Jump_Reference_Status",
    "Jump_Reference_Global_Cycle",
    "Jump_Reference_Capacity_Ah",
    "Jump_Delta_Ah",
    "Jump_Reference_Reason",
    "QC_Flag",
    "QC_Reason_Codes",
    "QC_Reason",
]


def ordered_columns(frame: pd.DataFrame, preferred: Iterable[str]) -> pd.DataFrame:
    leading = [column for column in preferred if column in frame.columns]
    remaining = [column for column in frame.columns if column not in leading]
    return frame.loc[:, leading + remaining]


def write_csv(frame: pd.DataFrame, path: Path, preferred_columns: Iterable[str]) -> None:
    ordered = ordered_columns(frame, preferred_columns)
    ordered.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        float_format=CSV_FLOAT_FORMAT,
    )


def build_qc_summary(cycles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def summarize(scope: str, subset: pd.DataFrame) -> dict[str, object]:
        flags = subset["QC_Flag"].value_counts()
        return {
            "Scope": scope,
            "Source_File": "",
            "File_Role": "",
            "Total_Cycles": int(len(subset)),
            "PASS_Count": int(flags.get("PASS", 0)),
            "CHECK_Count": int(flags.get("CHECK", 0)),
            "INVALID_CANDIDATE_Count": int(flags.get("INVALID_CANDIDATE", 0)),
            "Needs_Boundary_Check_Count": int(
                subset["Boundary_Status"].eq("NEEDS_BOUNDARY_CHECK").sum()
            ),
            "Incomplete_CV_Count": int(subset["Incomplete_CV_Flag"].fillna(False).sum()),
            "Incomplete_Discharge_Count": int(
                subset["Incomplete_Discharge_Flag"].fillna(False).sum()
            ),
            "Nonpositive_Capacity_Count": int(
                subset["Nonpositive_Capacity_Flag"].fillna(False).sum()
            ),
            "Capacity_Jump_Count": int(subset["Capacity_Jump_Flag"].fillna(False).sum()),
            "Statistics_Capacity_Count": int(
                subset["Capacity_Source"].str.startswith("STATISTICS", na=False).sum()
            ),
            "Channel_Capacity_Count": int(
                subset["Capacity_Source"].eq("CHANNEL_CYCLE_DELTA").sum()
            ),
        }

    rows.append(summarize("TOTAL_ALL_RECORDS", cycles))
    canonical = cycles[cycles["File_Role"].eq("CANONICAL")]
    rows.append(summarize("TOTAL_CANONICAL", canonical))

    for source_file, subset in cycles.groupby("Source_File", sort=False):
        row = summarize("FILE", subset)
        row["Source_File"] = source_file
        row["File_Role"] = str(subset["File_Role"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_capacity_curve(
    cycles: pd.DataFrame,
    output_path: Path,
    cell_id: str = "CS2-35",
) -> None:
    canonical = cycles[
        cycles["File_Role"].eq("CANONICAL")
        & cycles["Global_Cycle_Index"].notna()
        & cycles["Single_Cycle_Discharge_Capacity_Ah"].notna()
    ].copy()
    canonical = canonical.sort_values("Global_Cycle_Index")

    fig, axis = plt.subplots(figsize=(12, 6.5), dpi=180)
    axis.plot(
        canonical["Global_Cycle_Index"],
        canonical["Single_Cycle_Discharge_Capacity_Ah"],
        color="#A7AFB8",
        linewidth=0.7,
        alpha=0.75,
        zorder=1,
        label="Raw observed sequence",
    )

    styles = {
        "PASS": {"color": "#1F77B4", "marker": "o", "size": 13, "label": "PASS"},
        "CHECK": {"color": "#E69F00", "marker": "^", "size": 31, "label": "CHECK"},
        "INVALID_CANDIDATE": {
            "color": "#D62728",
            "marker": "x",
            "size": 37,
            "label": "INVALID_CANDIDATE",
        },
    }
    for flag, style in styles.items():
        subset = canonical[canonical["QC_Flag"].eq(flag)]
        axis.scatter(
            subset["Global_Cycle_Index"],
            subset["Single_Cycle_Discharge_Capacity_Ah"],
            c=style["color"],
            marker=style["marker"],
            s=style["size"],
            linewidths=0.9,
            alpha=0.9,
            label=style["label"],
            zorder=3,
        )

    boundary = canonical[canonical["Boundary_Status"].eq("NEEDS_BOUNDARY_CHECK")]
    if not boundary.empty:
        axis.scatter(
            boundary["Global_Cycle_Index"],
            boundary["Single_Cycle_Discharge_Capacity_Ah"],
            facecolors="none",
            edgecolors="#7B2CBF",
            marker="s",
            s=72,
            linewidths=1.2,
            label="NEEDS_BOUNDARY_CHECK",
            zorder=4,
        )

    axis.set_title(f"CALCE {cell_id} Discharge Capacity with Protocol QC", fontsize=13)
    axis.set_xlabel("Global Cycle")
    axis.set_ylabel("Single Cycle Discharge Capacity (Ah)")
    axis.grid(axis="y", color="#D9DDE2", linewidth=0.6, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    axis.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", metadata={"Software": "calce_qc"})
    plt.close(fig)
