"""Figure 4: SOH benchmark, held-out-cell MAE_Q by model and reference (XJTU, HUST).
Input: frozen project/outputs/soh_primary_benchmark_v1/soh_primary_cell_metrics.csv (per-cell MAE_Q).
Medians are checked against soh_primary_summary_metrics.csv. Outputs PDF, SVG and 450-dpi PNG."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "project" / "outputs" / "soh_primary_benchmark_v1"
FIG = ROOT / "manuscript" / "jes_v1" / "figures"
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.5, "legend.frameon": False})
INK, INK2 = "#0b0b0b", "#52514e"
MODELS = [("Random Forest Regressor", "RF", "#2a78d6", "o"), ("Gradient Boosting Regressor", "GB", "#eb6834", "s"),
          ("Ridge Regression", "Ridge", "#1baf7a", "^")]
QREFS = [("nominal", "nominal"), ("first_comparable", "first"), ("early_median5_comparable", "first5")]

cm = pd.read_csv(SRC / "soh_primary_cell_metrics.csv", encoding="utf-8-sig")
cm = cm[cm.Analysis_ID.str.contains("PRIMARY_COMMON")]
summ = pd.read_csv(SRC / "soh_primary_summary_metrics.csv", encoding="utf-8-sig")
summ = summ[summ.Analysis_ID.str.contains("PRIMARY_COMMON")]
agg = cm.groupby(["Dataset", "Qref_Definition", "Model"]).MAE_Q_Ah.agg(
    Cells="count", Median="median", Q1=lambda x: x.quantile(0.25), Q3=lambda x: x.quantile(0.75)).reset_index()
chk = agg.merge(summ[["Dataset", "Qref_Definition", "Model", "Median_MAE_Q_Ah", "Q1_MAE_Q_Ah", "Q3_MAE_Q_Ah"]],
                on=["Dataset", "Qref_Definition", "Model"])
assert len(chk) == 18 and np.allclose(chk.Median, chk.Median_MAE_Q_Ah, atol=1e-12) \
    and np.allclose(chk.Q1, chk.Q1_MAE_Q_Ah, atol=1e-12) and np.allclose(chk.Q3, chk.Q3_MAE_Q_Ah, atol=1e-12)
agg.to_csv(FIG / "Figure_4_data.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(190 / 25.4, 3.0), sharey=True)
rng = np.random.default_rng(20260928)
for ax, ds in zip(axes, ["XJTU", "HUST"]):
    for qi, (qk, ql) in enumerate(QREFS):
        for mi, (mk, ml, col, mrk) in enumerate(MODELS):
            x0 = qi + (mi - 1) * 0.26
            v = cm[(cm.Dataset == ds) & (cm.Qref_Definition == qk) & (cm.Model == mk)].MAE_Q_Ah.to_numpy()
            ax.scatter(x0 + rng.uniform(-0.07, 0.07, len(v)), 1000 * v, s=5, color=col, alpha=0.30, linewidth=0, zorder=2)
            a = agg[(agg.Dataset == ds) & (agg.Qref_Definition == qk) & (agg.Model == mk)].iloc[0]
            ax.vlines(x0, 1000 * a.Q1, 1000 * a.Q3, color=col, linewidth=2.2, zorder=3)
            ax.scatter([x0], [1000 * a.Median], s=34, color=col, marker=mrk, edgecolor="white", linewidth=0.9, zorder=4)
    n = int(agg[agg.Dataset == ds].Cells.iloc[0])
    ax.set_title(f"{ds} ({n} cells)", fontsize=8.5, loc="left", color=INK)
    ax.set_xticks(range(3), [q[1] for q in QREFS])
    ax.set_xlabel("Reference capacity")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Held-out-cell MAE$_Q$ (mAh)")
h = [Line2D([], [], color=c, marker=m, linestyle="", markersize=5, label=l) for _, l, c, m in MODELS]
h += [Line2D([], [], color=INK2, linewidth=2.2, label="interquartile range"),
      Line2D([], [], color=INK2, marker="o", linestyle="", alpha=0.35, markersize=3, label="individual cells")]
fig.legend(handles=h, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("pdf", "svg"):
    fig.savefig(FIG / f"Figure_4.{ext}")
fig.savefig(FIG / "Figure_4.png", dpi=450)
print(agg.round(5).to_string())
