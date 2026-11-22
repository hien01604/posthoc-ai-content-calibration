"""Ve Figure 1: FPR muc tai lieu theo k -- hai luat tren thang tho that bai theo hai
kieu khac nhau, hai luat con lai bam sat muc tieu 1%.

    python3 make_fig1.py

Hinh nay KHONG lap lai Table III: bang bo k=3 va k=8 cho du cho, hinh ve du ca 8 diem k
va cho thay DANG cua duong cong (Gauss trot len don dieu; nguong tinh cat qua muc tieu
theo ca hai huong) -- thu ma bang khong the hien duoc.

DAU VAO
-------
  final_experiments_rivf2026/results/f7_cluster_marginal_ci.csv
      static / emp_deff / zscale -- DUNG nguon da sinh ra Table III, nen so khop tuyet doi.
  output/pihat_static_vs_corrected_by_k.csv
      fpr_corrected = duong "thang tho + phan vi Gauss" (cot Gauss cua Table III).

DAU RA
------
  paper/figures/fig1_fpr_curve.pdf / .png   (mot cot IEEE, rong 3.4in)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
F7 = BASE / "final_experiments_rivf2026" / "results" / "f7_cluster_marginal_ci.csv"
FIGDIR = BASE / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

KS = [1, 2, 3, 5, 8, 10, 15, 20]

# Bang mau da qua validator cua skill dataviz (light, --pairs all):
#   CVD dE 9.2 / normal-vision dE 16.3, ca hai deu dat nguong.
# Ma hoa thu cap (marker + linestyle) la BAT BUOC: bai in tren giay den trang,
# va cam orange/aqua gan nhu trung nhau khi chuyen sang thang xam.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED = "#0b0b0b", "#52514e"


def load():
    f7 = pd.read_csv(F7)
    s = {m: f7[f7.method == m].set_index("k")["fpr"] for m in ("static", "emp_deff", "zscale")}
    p = pd.read_csv(OUT / "pihat_static_vs_corrected_by_k.csv")
    s["raw_gauss"] = p.groupby("k")["fpr_corrected"].median() * 100
    return s


def main():
    s = load()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    })

    fig, ax = plt.subplots(figsize=(3.4, 2.55))

    # Duong muc tieu ve TRUOC de nam duoi cac duong du lieu.
    ax.axhline(1.0, color=INK, linestyle=(0, (4, 3)), linewidth=0.8, zorder=2)
    ax.annotate("target 1%", xy=(1.06, 1.0), xytext=(1.06, 1.34),
                fontsize=6.2, color=MUTED, ha="left")

    series = [
        ("raw_gauss", ORANGE, "o", (0, (5, 2)), "Raw scale, Gaussian", 1.5),
        ("static",    VIOLET, "s", (0, (1, 1.6)), "Static threshold ($k$=20)", 1.5),
        ("emp_deff",  AQUA,   "^", (0, (4, 1.5, 1, 1.5)), "Raw scale, empirical", 1.4),
        ("zscale",    BLUE,   "D", "-", "$z$-scale (proposed)", 1.9),
    ]
    for key, color, marker, ls, label, lw in series:
        v = s[key]
        ax.plot(KS, [v[k] for k in KS], color=color, marker=marker, linestyle=ls,
                linewidth=lw, markersize=3.4, markeredgewidth=0.0,
                label=label, zorder=4, clip_on=False)

    ax.set_xscale("log")
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS], fontsize=7)
    ax.minorticks_off()
    ax.set_xlim(0.95, 21.5)
    ax.set_ylim(0, 9.7)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(length=2.5, width=0.6, colors=MUTED)

    ax.set_xlabel("Segments per document, $k$ (log scale)", fontsize=7.6, color=INK)
    ax.set_ylabel("Document-level FPR (%)", fontsize=7.6, color=INK)

    ax.grid(axis="y", color="#d8d7d2", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)

    leg = ax.legend(fontsize=6.3, loc="center right", bbox_to_anchor=(1.005, 0.45),
                    frameon=True, framealpha=0.94, edgecolor="#d8d7d2",
                    handlelength=2.4, handletextpad=0.45, borderpad=0.4,
                    labelspacing=0.34)
    leg.get_frame().set_linewidth(0.5)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout(pad=0.25)
    fig.savefig(FIGDIR / "fig1_fpr_curve.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGDIR / "fig1_fpr_curve.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
    print("-> %s" % (FIGDIR / "fig1_fpr_curve.pdf"))
    print("-> %s" % (FIGDIR / "fig1_fpr_curve.png"))

    print("\nSo ve trong hinh (doi chieu Table III):")
    print("  k   static  rawGauss  rawEmp  zscale")
    for k in KS:
        print("  %2d   %5.2f    %5.2f    %5.2f   %5.2f"
              % (k, s["static"][k], s["raw_gauss"][k], s["emp_deff"][k], s["zscale"][k]))


if __name__ == "__main__":
    main()
