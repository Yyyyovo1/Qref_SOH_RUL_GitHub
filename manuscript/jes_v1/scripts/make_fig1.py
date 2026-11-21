"""Figure 1: study design (vector redraw of the approved AI design draft, img_1794845398.png).
Outputs PDF and SVG (vector) and a 1000-dpi PNG (JES minimum for bitmapped line drawings)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

FIG = Path(__file__).resolve().parents[1] / "figures"
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                            "font.family": "DejaVu Sans", "mathtext.fontset": "dejavuserif"})
INK, INK2 = "#0b0b0b", "#52514e"
BLUE, BLUE_F = "#2a78d6", "#eaf2fc"
ORG, ORG_F = "#eb6834", "#fdeee7"
GRN, GRN_F = "#1baf7a", "#e6f6f0"
GREY_F, GREY_E = "#f0efec", "#8a8986"

W, H = 190, 100            # mm, drawing units
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

def box(x, y, w, h, ec, fc, lw=1.2, r=3.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", ec=ec, fc=fc, lw=lw))

def text(x, y, s, size=7, weight="normal", color=INK, **kw):
    ax.text(x, y, s, ha="center", va="center", fontsize=size, fontweight=weight, color=color, **kw)

def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.9, color=INK2))

# ---- left column: data
lx, lw_ = 3, 48
box(lx, 17, lw_, 78, BLUE, BLUE_F, lw=1.3)
text(lx + lw_ / 2, 88.5, "Fixed capacity\nobservations", 8.5, "bold", linespacing=1.15)
for i in range(5):                                  # battery icons
    cx = lx + 9 + i * 7.6
    ax.add_patch(FancyBboxPatch((cx - 2.6, 63), 5.2, 12, boxstyle="round,pad=0,rounding_size=1.1", ec=INK2, fc="white", lw=0.7))
    ax.add_patch(Rectangle((cx - 0.9, 75), 1.8, 1.1, ec=INK2, fc="white", lw=0.7))
for j, d in enumerate(["XJTU", "HUST", "MIT", "TJU-NCA", "TJU-NCM"]):
    text(lx + lw_ / 2, 56 - j * 5.6, d, 7.2, "bold")
box(lx + 11, 22, lw_ - 22, 6.5, INK2, "white", lw=0.8, r=1.5)
text(lx + lw_ / 2, 25.25, "377 cells", 7.2, "bold")
text(lx + lw_ / 2, 12.5, "observations never change", 6.6, color=INK2)

# ---- middle column: label definition
mx, mw = 62, 62
box(mx, 17, mw, 78, ORG, ORG_F, lw=2.0)
text(mx + mw / 2, 88.5, "Label definition", 8.5, "bold")
chips = [("Reference capacity", "nominal  |  first  |  first5"),
         ("Threshold fraction  $\\theta$", "0.60 – 0.95"),
         ("Endpoint rule", "strict  |  tolerance")]
for k, (t1, t2) in enumerate(chips):
    y = 70 - k * 15.5
    box(mx + 4, y, mw - 8, 11.5, INK2, "#fbfbfa", lw=0.8, r=5.0)
    text(mx + mw / 2, y + 7.6, t1, 7, "bold")
    text(mx + mw / 2, y + 3.4, t2, 6.6)
box(mx + 13, 22, mw - 26, 8.5, INK2, "white", lw=0.8, r=1.2)
text(mx + mw / 2, 26.25, "$T = \\theta \\times Q_{\\mathrm{ref}}$", 9)

# ---- right column: analysis branches
rx, rw = 135, 52
branches = [("Endpoint labels", "exact or censored\nlabelled-share window", "Sections 3.1–3.3"),
            ("SOH benchmark", "RF · GB · Ridge\non XJTU and HUST", "Section 3.4"),
            ("RUL method comparison", "four preregistered tests\n(v1–v4)", "Sections 3.5–3.6")]
ys = [74, 48.5, 23]
for (t1, t2, t3), y in zip(branches, ys):
    box(rx, y, rw, 21, GRN, GRN_F, lw=1.3)
    text(rx + rw / 2, y + 15.5, t1, 8, "bold")
    text(rx + rw / 2, y + 8.6, t2, 6.4, linespacing=1.2)
    text(rx + rw / 2, y + 2.6, t3, 5.8, color=INK2, style="italic")
    arrow(mx + mw + 0.8, y + 10.5, rx - 0.8, y + 10.5)
arrow(lx + lw_ + 0.8, 56, mx - 0.8, 56)

# ---- evaluation band (applies to the SOH and RUL branches)
box(rx, 3, rw, 15.5, GREY_E, GREY_F, lw=0.8, r=2)
text(rx + rw / 2, 14.2, "Evaluation in the SOH and RUL branches", 5.8, "bold", color=INK2)
text(rx + rw / 2, 8.4, "cell-level cross-validation · paired bootstrap\nequivalence test (v4)", 6.2, linespacing=1.25)

for ext in ("pdf", "svg"):
    fig.savefig(FIG / f"Figure_1.{ext}")
fig.savefig(FIG / "Figure_1.png", dpi=1000)
print("ok")
