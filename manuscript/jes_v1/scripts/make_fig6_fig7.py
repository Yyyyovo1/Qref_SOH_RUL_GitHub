"""Figures 5 and 6 of the JES manuscript (file name kept from the draft numbering) (v4 experiment 1, MIT).

Reads only the frozen v4 outputs in project/outputs/rul_v4_core_claim/.
Writes PDF (vector, fonts embedded) and 300-dpi PNG to manuscript/jes_v1/figures/.
Usage (from repo root):  python manuscript/jes_v1/scripts/make_fig6_fig7.py
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "project" / "outputs" / "rul_v4_core_claim"
FIG = ROOT / "manuscript" / "jes_v1" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "legend.frameon": False,
})
# Categorical slots 1-3 of the validated reference palette (all-pairs pass).
COL = {"variance": "#2a78d6", "discharge": "#eb6834", "full": "#1baf7a"}
INK2 = "#52514e"
RULES = [("strict", "Strict rule"), ("tolerance", "Tolerance rule")]
PAIR = "nominal-first"
W_IN = 190 / 25.4  # full-page width

pair_ci = pd.read_csv(OUT / "exp1_pair_ci.csv", encoding="utf-8-sig")
equiv = pd.read_csv(OUT / "exp2_equivalence_and_power.csv", encoding="utf-8-sig")
for df in (pair_ci, equiv):
    df["Theta"] = df["Theta"].round(2)

# ---------- Figure 6: change in method difference, Delta = D(first) - D(nominal)
d = pair_ci[(pair_ci.Pair == PAIR)]
piv = d.pivot_table(index=["Rule", "Theta", "Method_Pair"], columns="Qref_Definition",
                    values="Observed_Difference")
piv["Delta_point"] = piv["first_comparable"] - piv["nominal"]
e = equiv[(equiv.Experiment == 1) & (equiv.Pair == PAIR)].set_index(["Rule", "Theta", "Method_Pair"])
f6 = e.join(piv["Delta_point"]).reset_index()
f6 = f6.rename(columns={"Delta": "delta_margin"})
f6.to_csv(FIG / "Figure_5_data.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(W_IN, 3.2), sharey=True)
series6 = [("variance-discharge", "discharge", "o", -0.0018, "variance − discharge"),
           ("variance-full", "full", "s", 0.0018, "variance − full")]
for ax, (rule, title) in zip(axes, RULES):
    r = f6[f6.Rule == rule]
    band = r.drop_duplicates("Theta").sort_values("Theta")
    ax.fill_between(band.Theta, -band.delta_margin, band.delta_margin, step="mid",
                    color="#d9d8d3", alpha=0.7, linewidth=0, zorder=1)
    ax.axhline(0, color=INK2, linewidth=0.6, zorder=2)
    for mp, key, mk, dx, _ in series6:
        s = r[r.Method_Pair == mp].sort_values("Theta")
        x = s.Theta + dx
        ax.vlines(x, s.CI90_Lower, s.CI90_Upper, color=COL[key], linewidth=1.2, zorder=3)
        ne = s.Category == "NOT_EQUIVALENT"
        ax.scatter(x[ne], s.Delta_point[ne], marker=mk, s=22, color=COL[key],
                   edgecolor="#fcfcfb", linewidth=0.8, zorder=4)
        ax.scatter(x[~ne], s.Delta_point[~ne], marker=mk, s=22, facecolor="#fcfcfb",
                   edgecolor=COL[key], linewidth=1.0, zorder=4)
    ax.set_title(title, fontsize=8.5, loc="left", color="#0b0b0b")
    ax.set_xlabel("EOL threshold θ (fraction of reference capacity)")
    ax.set_xticks([0.84, 0.86, 0.88, 0.90, 0.92, 0.94])
axes[0].set_ylabel("Δ = D(first) − D(nominal) (pp)")
handles = [Line2D([], [], color=COL[k], marker=m, linestyle="-", linewidth=1.2, markersize=5,
                  label=lab) for _, k, m, _, lab in series6]
handles += [Line2D([], [], color=INK2, marker="o", linestyle="", markerfacecolor=INK2,
                   markersize=5, label="not equivalent (90% CI outside ±δ)"),
            Line2D([], [], color=INK2, marker="o", linestyle="", markerfacecolor="#fcfcfb",
                   markersize=5, label="equivalent or imprecise"),
            Patch(facecolor="#d9d8d3", label="equivalence margin ±δ")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.15, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"Figure_5.{ext}", dpi=300)
plt.close(fig)

# ---------- Figure 7: mean APE of the three models under nominal vs first
rows = []
for (rule, theta, q), g in d.groupby(["Rule", "Theta", "Qref_Definition"]):
    vd = g[g.Method_Pair == "variance-discharge"].iloc[0]
    vf = g[g.Method_Pair == "variance-full"].iloc[0]
    assert abs(vd.Observed_A - vf.Observed_A) < 1e-12 and vd.Cells == vf.Cells
    rows.append(dict(Rule=rule, Theta=theta, Qref=q, Cells=vd.Cells,
                     variance=vd.Observed_A, discharge=vd.Observed_B, full=vf.Observed_B))
f7 = pd.DataFrame(rows)
f7.to_csv(FIG / "Figure_6_data.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(W_IN, 2.9), sharey=True)
LS = {"nominal": "-", "first_comparable": "--"}
MK = {"variance": "o", "discharge": "^", "full": "s"}
for ax, (rule, title) in zip(axes, RULES):
    r = f7[f7.Rule == rule]
    for q in ("nominal", "first_comparable"):
        s = r[r.Qref == q].sort_values("Theta")
        for m in ("variance", "discharge", "full"):
            ax.plot(s.Theta, s[m], color=COL[m], linestyle=LS[q], linewidth=1.6,
                    marker=MK[m], markersize=3.5,
                    markerfacecolor=COL[m] if q == "nominal" else "#fcfcfb", zorder=3)
    s = r[r.Qref == "nominal"].sort_values("Theta").iloc[-1]
    # direct labels at the right end of the nominal lines
    for m, dy in (("variance", 0.0), ("discharge", 0.6), ("full", -0.9)):
        ax.annotate(m, (s.Theta, s[m]), xytext=(4, dy * 6), textcoords="offset points",
                    fontsize=7, color="#0b0b0b", va="center")
    ax.set_title(title, fontsize=8.5, loc="left", color="#0b0b0b")
    ax.set_xlabel("EOL threshold θ (fraction of reference capacity)")
    ax.set_xticks([0.84, 0.86, 0.88, 0.90, 0.92, 0.94])
    ax.set_xlim(r.Theta.min() - 0.005, r.Theta.max() + 0.02)
axes[0].set_ylabel("Mean absolute percentage error (%)")
handles = [Line2D([], [], color=COL[m], marker=MK[m], linewidth=1.6, markersize=4, label=m)
           for m in ("variance", "discharge", "full")]
handles += [Line2D([], [], color=INK2, linestyle="-", linewidth=1.6, label="nominal reference"),
            Line2D([], [], color=INK2, linestyle="--", linewidth=1.6,
                   label="first reference (first_comparable)")]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.1, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"Figure_6.{ext}", dpi=300)
plt.close(fig)
print("ok", len(f6), len(f7))
