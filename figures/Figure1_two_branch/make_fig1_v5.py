"""Figure 1 (v4): schematic only, no data. Two analysis branches sharing one Q_ref definition."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
fig, ax = plt.subplots(figsize=(7.0, 4.25))
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
EDGE = "#3B4F63"

def box(x, y, w, h, title, body, fc, tsize=7.2, bsize=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0",
                                fc=fc, ec=EDGE, lw=0.8))
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=tsize, weight="bold", color="#1E2A36")
    ax.text(x + w / 2, y + h - (5.6 if "\n" not in title else 8.6), body, ha="center", va="top",
            fontsize=bsize, linespacing=1.4, color="#2F3B47")

def arrow(x0, y0, x1, y1, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, lw=0.9, color=EDGE, mutation_scale=8, shrinkA=0, shrinkB=0))

grey, blue, amber, rose, green = "#EEF2F6", "#E3EEF8", "#FFF3D6", "#FBE4DC", "#E2F1E4"

# shared root
box(0.5, 22, 14.5, 18, "Retained\nobservations", "$Q_k$, $k\\in\\mathcal{K}$\nidentical under\nevery definition", grey)
box(18, 22, 14.5, 18, "$Q_{\\mathrm{ref}}$ definition", "nominal\nfirst_comparable\nfirst5_comparable\ninitial_test (XJTU)", blue)
arrow(15.4, 31, 17.6, 31)

# Branch A (top)
ax.text(36, 60.5, "Branch A: endpoint labels", fontsize=7.4, weight="bold", color="#8A5A00", va="top")
yA, hA = 41, 16.5
box(36, yA, 15, hA, "Absolute\nthreshold", "$T=\\theta\\,Q_{\\mathrm{ref}}$\n$\\theta$ = 0.75, 0.80,\nor 0.85", amber)
box(53.5, yA, 15.5, hA, "Observed\nfirst hit", "$N_{\\mathrm{obs}}(T)=$\n$\\min\\{k\\in\\mathcal{K}:Q_k\\leq T\\}$\nno retained hit\n$\\rightarrow$ right-censored", amber, bsize=6.0)
box(71.5, yA, 13, hA, "Endpoint\nclass", "EXACT\nINTERVAL\nRIGHT", rose)
box(87, yA, 12.6, hA, "Point-RUL\neligibility", "cell-level;\nexact only", rose)
for x0, x1 in [(51.4, 53.1), (69.4, 71.1), (84.9, 86.6)]:
    arrow(x0, yA + hA / 2, x1, yA + hA / 2)

# Branch B (bottom)
ax.text(36, 19.6, "Branch B: SOH benchmark", fontsize=7.4, weight="bold", color="#2E6B3A", va="top")
yB, hB = 1.5, 15.5
box(36, yB, 15, hB, "SOH target", "$\\mathrm{SOH}_k=Q_k/Q_{\\mathrm{ref}}$\n(not clipped)", blue)
box(53.5, yB, 17, hB, "SOH models", "Ridge, RF, GB\nfixed A9 mask,\nfeatures, cell folds", green)
box(73, yB, 26.6, hB, "Two error spaces", "SOH (percentage points)\ncapacity $\\widehat{Q}_k=\\widehat{\\mathrm{SOH}}_k\\,Q_{\\mathrm{ref}}$ (Ah)", green)
for x0, x1 in [(51.4, 53.1), (70.9, 72.6)]:
    arrow(x0, yB + hB / 2, x1, yB + hB / 2)

# split from Q_ref to both branches
arrow(32.9, 34, 35.6, yA + hA / 2)
arrow(32.9, 28, 35.6, yB + hB / 2)

# note: no data flow from A to B
ax.text(67.8, 29.6, "Endpoint class and RUL eligibility are not used to\nselect the SOH training or evaluation population.",
        ha="center", va="center", fontsize=6.2, style="italic", color="#4A5560",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#B8C2CC", lw=0.6, ls="--"))
fig.savefig("Figure1_two_branch_v5.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
fig.savefig("Figure1_two_branch_v5.pdf", bbox_inches="tight", pad_inches=0.03)
