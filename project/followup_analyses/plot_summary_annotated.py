"""Annotated summary figure for the protocol validation, drawn directly from summary_table.csv.

Every number on the figure is read from the committed results file; nothing is typed in by hand.
Copy of project/protocol_validation/plot_summary_annotated.py (commit 885d9e6); only change: PDF CreationDate omitted.
Usage: python plot_summary_annotated.py <summary_table.csv> <output_stem>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

COMMON, PROTOCOL = "#eb6834", "#2a78d6"     # categorical slots 2 and 1 (validated pair)
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

GROUP_LABELS = {
    "SOH main (OWN) XJTU": "SOH, OWN · XJTU", "SOH main (OWN) HUST": "SOH, OWN · HUST",
    "RUL v2 E1+E2 XJTU": "RUL v2 E1+E2 · XJTU", "RUL v2 E1+E2 HUST": "RUL v2 E1+E2 · HUST",
    "RUL v2 E1+E2 MIT": "RUL v2 E1+E2 · MIT",
    "SOH supplementary (TRAIN)": "SOH, TRAIN (suppl.)", "SOH supplementary (NOM)": "SOH, NOM (suppl.)",
    "SOH supplementary (A8, OWN)": "SOH, A8 OWN (suppl.)",
    "SOH 12 settings": "SOH, 12 settings",
    "RUL v2 E3 XJTU": "XJTU", "RUL v2 E3 HUST": "HUST", "RUL v2 E3 MIT": "MIT",
    "Endpoint labels XJTU": "XJTU", "Endpoint labels HUST": "HUST", "Endpoint labels MIT": "MIT",
}
PANELS = {
    "A": ("A  Noise read as a ranking change", "surface ranking changes vs. supported reversals"),
    "B": ("B  Undeclared information condition", "settings vs. settings whose conclusion does not depend on it"),
    "C": ("C  Biased censored metric (E3)", "one-sided-error reversals vs. those also supported by the C index"),
    "D": ("D  Endpoint rule", "tolerance-rule point labels vs. those the strict rule also gives"),
}


def _panel(ax, s: pd.DataFrame, trap: str) -> None:
    h = 0.36
    n = len(s)
    xmax = max(s.Common_Practice_Results.max(), 1)
    for i, r in enumerate(s.itertuples(index=False)):
        y = n - 1 - i
        suppl = "suppl." in GROUP_LABELS[r.Group]
        hatch = "////" if suppl else None
        for value, color, dy in ((r.Common_Practice_Results, COMMON, h / 2 + 0.02),
                                 (r.Still_Holding_Under_Protocol, PROTOCOL, -h / 2 - 0.02)):
            ax.barh(y + dy, value, height=h, color=color, hatch=hatch, edgecolor="white", linewidth=0)
            label = f"{value:,}"
            if trap == "D" and color == PROTOCOL:
                label += f"  ({value / r.Common_Practice_Results:.0%})"
            ax.text(value + xmax * 0.012, y + dy, label, va="center", ha="left", fontsize=8.5, color=TEXT)
    if trap == "A":
        sep = n - 1 - list(s.Group).index("SOH supplementary (TRAIN)") + 0.5
        ax.axhline(sep, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(xmax * 1.25, sep - 0.12, "supplementary scope, not pooled", ha="right", va="top", fontsize=7.5, color=MUTED)
    title, subtitle = PANELS[trap]
    if trap == "B":
        k = int(s.Common_Practice_Results.iat[0] - s.Still_Holding_Under_Protocol.iat[0])
        subtitle = f"{k} of {int(s.Common_Practice_Results.iat[0])} conclusions depend on the information condition"
    ax.set_yticks(range(n), [GROUP_LABELS[g] for g in s.Group[::-1]], fontsize=8.5, color=TEXT)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, xmax * 1.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=8, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.set_title(f"{title}\n", loc="left", fontsize=10.5, color=TEXT, fontweight="bold")
    # Fixed offset in points, so the subtitle sits the same distance below the title in tall and short panels.
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                fontsize=8, color=MUTED, va="bottom")


def main(table: Path, stem: Path) -> None:
    summary = pd.read_csv(table, encoding="utf-8-sig")
    fig = plt.figure(figsize=(12.5, 8.6))
    # A has eight groups and takes the left column; B (1 group), C and D (3 groups each) stack on the right.
    gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 3, 3])
    axes = {"A": fig.add_subplot(gs[:, 0]), "B": fig.add_subplot(gs[0, 1]),
            "C": fig.add_subplot(gs[1, 1]), "D": fig.add_subplot(gs[2, 1])}
    for trap, ax in axes.items():
        _panel(ax, summary[summary.Trap == trap].reset_index(drop=True), trap)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COMMON), plt.Rectangle((0, 0), 1, 1, color=PROTOCOL)]
    fig.legend(handles, ["Common practice", "Still holds under the protocol"], loc="upper center", ncol=2,
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, "Counts in different panels have different units and are not comparable across panels.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965), h_pad=2.5, w_pad=3.0)
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, metadata={"Software": None} if ext == "png" else {"CreationDate": None},
                    facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
