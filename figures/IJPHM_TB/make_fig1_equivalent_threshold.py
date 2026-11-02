"""Figure 1 of the IJPHM technical brief: equivalent EOL threshold on the nominal scale.

For an EOL rule at fraction theta applied with a measured reference Q_ref', the
equivalent fraction on the nominal scale is theta' = theta * Q_ref' / Q_nominal (Eq. 3).
Input: supplementary/csv/S3_PerCell_Qref.csv (per-cell reference capacities).
Output: Figure1_equivalent_threshold.png and .pdf in this folder.

Run from the repository root:  python figures/IJPHM_TB/make_fig1_equivalent_threshold.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
THETA = 0.80

s3 = pd.read_csv(ROOT / "supplementary" / "csv" / "S3_PerCell_Qref.csv")
rows = []
for dataset in ["XJTU", "HUST"]:
    g = s3[s3["Dataset"] == dataset]
    for col, label in [("First_Comparable_Ah", "first"), ("First5_Comparable_Ah", "first5")]:
        tp = 100 * THETA * g[col] / g["Nominal_Qref_Ah"]
        rows.append((f"{dataset}, {label}", dataset, tp.min(), tp.median(), tp.max(), len(tp)))
        print(f"{dataset:5s} {label:7s} n={len(tp):3d}  min={tp.min():.2f}  median={tp.median():.2f}  max={tp.max():.2f}")

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6, "mathtext.fontset": "stix"})
fig, ax = plt.subplots(figsize=(3.38, 2.0), dpi=600)
ax.axvspan(70, 80, color="0.88", lw=0, zorder=0)
ax.axvline(100 * THETA, color="0.2", ls="--", lw=0.8, zorder=1)
ys = [3, 2, 1, 0]
for y, (label, dataset, lo, mid, hi, n) in zip(ys, rows):
    marker = "o" if dataset == "XJTU" else "s"
    ax.errorbar(mid, y, xerr=[[mid - lo], [hi - mid]], fmt=marker, color="black", mfc="white",
                mew=0.9, ms=4.5, elinewidth=0.9, capsize=2.5, zorder=3)
    ax.text(mid, y + 0.32, f"{mid:.1f}%", va="bottom", ha="center", fontsize=7)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(68, 93)
ax.set_ylim(-0.7, 4.2)
ax.set_xlabel(r"Equivalent threshold $\theta'$ (% of nominal capacity)")
ax.text(75, 4.1, "conventional 70–80% range", ha="center", va="top", fontsize=6.5, color="0.25")
ax.text(80.5, 4.1, "nominal 80% rule", ha="left", va="top", fontsize=6.5, color="0.25")
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "Figure1_equivalent_threshold.png", dpi=600)
fig.savefig(OUT / "Figure1_equivalent_threshold.pdf")
