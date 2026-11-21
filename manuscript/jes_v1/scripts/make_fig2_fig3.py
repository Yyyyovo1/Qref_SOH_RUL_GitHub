"""Figure 2 (reference-capacity deviation from nominal) and Figure 3 (labelled share vs theta).
Inputs: manuscript/jes_v1/results_data/ (written by results_tables.py from frozen outputs)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "manuscript" / "jes_v1" / "results_data"
FIG = ROOT / "manuscript" / "jes_v1" / "figures"
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.5, "legend.frameon": False})
INK, INK2 = "#0b0b0b", "#52514e"
QCOL = {"nominal": "#2a78d6", "first": "#eb6834", "first5": "#1baf7a"}
DS = ["XJTU", "HUST", "MIT", "TJU-NCA", "TJU-NCM"]
W_IN = 190 / 25.4

# ---------------- Figure 2
cells = pd.read_csv(DATA / "A_per_cell_qref.csv")
fig, ax = plt.subplots(figsize=(W_IN, 2.8))
rng = np.random.default_rng(20260928)
for i, ds in enumerate(DS):
    g = cells[cells.Dataset == ds]
    for q, dx, mk in (("first", -0.18, "o"), ("first5", 0.18, "^")):
        y = 100 * (g[q] - g.nominal) / g.nominal
        x = i + dx + rng.uniform(-0.09, 0.09, len(y))
        ax.scatter(x, y, s=7, color=QCOL[q], alpha=0.55, linewidth=0, marker=mk, zorder=3)
        ax.hlines(np.median(y), i + dx - 0.14, i + dx + 0.14, color=INK, linewidth=1.6, zorder=4)
ax.axhline(0, color=INK2, linewidth=0.8, zorder=2)
ax.set_xticks(range(len(DS)), [f"{d}\n(n = {int((cells.Dataset == d).sum())})" for d in DS])
ax.set_ylabel("(Q$_{ref}$ − nominal) / nominal (%)")
ax.grid(axis="x", visible=False)
h = [Line2D([], [], marker="o", color=QCOL["first"], linestyle="", markersize=4, label="first_comparable"),
     Line2D([], [], marker="^", color=QCOL["first5"], linestyle="", markersize=4, label="first5_comparable"),
     Line2D([], [], color=INK, linewidth=1.6, label="median")]
ax.legend(handles=h, loc="upper right", ncol=3, fontsize=7)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"Figure_2.{ext}", dpi=300)
plt.close(fig)

# ---------------- Figure 3
sh = pd.read_csv(DATA / "C_labelled_share_by_theta.csv")
fig, axes = plt.subplots(2, 5, figsize=(W_IN, 3.9), sharey=True)
for r, (rule, col) in enumerate((("Strict rule", "Strict_Share"), ("Tolerance rule", "Tolerance_Share"))):
    for c, ds in enumerate(DS):
        ax = axes[r, c]
        g = sh[sh.Dataset == ds]
        for q, ls in (("nominal", "-"), ("first", "--"), ("first5", ":")):
            s = g[g.Qref == q].sort_values("Theta")
            ax.plot(s.Theta, 100 * s[col], color=QCOL[q], linestyle=ls, linewidth=1.5)
        ax.axhline(90, color=INK2, linewidth=0.7, linestyle=(0, (1, 2)))
        lo, hi = g.Theta.min(), g.Theta.max()
        ax.set_xlim(lo - 0.005, hi + 0.005)
        ax.set_xticks([0.60, 0.70, 0.80, 0.90] if lo < 0.7 else [0.75, 0.80, 0.85, 0.90, 0.95])
        ax.tick_params(labelsize=6.5)
        if r == 0:
            ax.set_title(ds, fontsize=8, color=INK)
        if c == 0:
            ax.set_ylabel(f"{rule}\nlabelled cells (%)")
        if r == 1:
            ax.set_xlabel("θ", fontsize=8)
h = [Line2D([], [], color=QCOL[q], linestyle=ls, linewidth=1.5, label=lab) for q, ls, lab in
     (("nominal", "-", "nominal"), ("first", "--", "first_comparable"), ("first5", ":", "first5_comparable"))]
h.append(Line2D([], [], color=INK2, linewidth=0.7, linestyle=(0, (1, 2)), label="90% (window criterion)"))
fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.06, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"Figure_3.{ext}", dpi=300)
plt.close(fig)
print("ok")
