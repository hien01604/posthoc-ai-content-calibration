"""Ve Figure 2: so do HAU XU LY.

Muc dich cua hinh nay khong phai minh hoa kien truc mang. Nguoc lai: no phai lam ro
ngay tu cai nhin dau tien rang **bo phat hien la DAU VAO cho san**, con dong gop cua
bai nam TOAN BO o phia sau no -- hai nhanh hau xu ly chay tren diem so tho.

    python3 make_fig2.py   ->  paper/figures/fig2_pipeline.pdf / .png  (mot cot IEEE)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

BASE = Path(__file__).resolve().parent
FIGDIR = BASE / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Bang mau da qua validator (xem make_fig1.py). Nhanh trai/phai tach nhau ve VI TRI,
# nen in den trang van doc duoc; mau chi la ho tro.
BLUE, AQUA = "#2a78d6", "#1baf7a"
INK, MUTED, GIVEN = "#0b0b0b", "#52514e", "#8a8a86"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.4, 3.15))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10.4)
ax.axis("off")


def box(x, y, w, h, text, ec, fc="white", fs=6.2, weight="normal", ls="-", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                linewidth=0.9, edgecolor=ec, facecolor=fc,
                                linestyle=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, color=tc, zorder=4, linespacing=1.35)


def arrow(x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=0.9, color=color,
                                 zorder=2, shrinkA=0, shrinkB=0))


# ---------------- phan CHO SAN: bo phat hien ----------------
box(1.4, 9.05, 7.2, 1.0,
    "Any trained segment detector\n(16 checkpoints, §III-B)",
    ec=GIVEN, fc="#f2f2f0", fs=6.4, tc=MUTED, ls=(0, (3, 2)))
ax.text(9.95, 9.55, "given", ha="right", va="center", fontsize=5.8,
        style="italic", color=GIVEN, rotation=90)

arrow(5.0, 9.05, 5.0, 8.45)
ax.text(5.15, 8.72, "raw segment scores $s_1,\\dots,s_k$",
        ha="left", va="center", fontsize=6.0, color=INK)

# ---------------- vung DONG GOP: hau xu ly ----------------
ax.add_patch(Rectangle((0.15, 0.15), 9.7, 8.30, linewidth=0.8,
                       edgecolor=INK, facecolor="none", zorder=1))
ax.text(0.45, 8.14, "post-processing (this paper)",
        ha="left", va="center", fontsize=6.0, style="italic", color=INK)

arrow(5.0, 8.45, 2.55, 7.35)
arrow(5.0, 8.45, 7.45, 7.35)

# ----- nhanh trai: uoc luong ti le -----
box(0.55, 6.35, 4.0, 1.0, "anchors $\\mu_0,\\mu_1$\nfrom val split only",
    ec=BLUE, fs=6.2)
arrow(2.55, 6.35, 2.55, 5.75)
box(0.55, 4.75, 4.0, 1.0, "$\\hat{\\pi}=\\dfrac{\\bar{s}-\\mu_0}{\\mu_1-\\mu_0}$",
    ec=BLUE, fs=7.4)
arrow(2.55, 4.75, 2.55, 4.15)
box(0.55, 3.05, 4.0, 1.1,
    "empirical percentile\ninterval, recalibrated\nat $\\hat{\\pi}$ (§IV-B)", ec=BLUE, fs=6.2)
arrow(2.55, 3.05, 2.55, 2.45)
box(0.55, 1.30, 4.0, 1.15, "reported fraction\n$\\hat{\\pi}\\pm0.15$",
    ec=BLUE, fc="#e8f1fb", fs=6.6, weight="bold")

# ----- nhanh phai: quyet dinh -----
box(5.45, 6.35, 4.0, 1.0, "rank vs. human pool\n$\\to$ mid-$p$ $\\to z_i$",
    ec=AQUA, fs=6.2)
arrow(7.45, 6.35, 7.45, 5.75)
box(5.45, 4.75, 4.0, 1.0, "$Z_k=k^{-1/2}\\sum_i z_i$", ec=AQUA, fs=7.0)
arrow(7.45, 4.75, 7.45, 4.15)
box(5.45, 3.05, 4.0, 1.1,
    "compare to\n$z_{1-\\alpha}\\sqrt{1{+}(k{-}1)\\rho_z}$\n(§V-B)", ec=AQUA, fs=6.2)
arrow(7.45, 3.05, 7.45, 2.45)
box(5.45, 1.30, 4.0, 1.15, "flag / no flag\nat controlled FPR",
    ec=AQUA, fc="#e6f6ef", fs=6.6, weight="bold")

ax.text(5.0, 0.62,
        "Both branches read the same raw scores; neither feeds the other.",
        ha="center", va="center", fontsize=5.9, style="italic", color=MUTED)

fig.tight_layout(pad=0.12)
fig.savefig(FIGDIR / "fig2_pipeline.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(FIGDIR / "fig2_pipeline.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
print("-> %s" % (FIGDIR / "fig2_pipeline.pdf"))
print("-> %s" % (FIGDIR / "fig2_pipeline.png"))
