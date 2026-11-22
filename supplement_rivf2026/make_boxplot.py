"""Hinh S1: boxplot phan bo FPR qua 16 checkpoint tai moi k -- Revision Brief muc 4.

    python supplement_rivf2026/make_boxplot.py

Table III cua bai chi bao trung vi + khoang tin cay bootstrap. Bootstrap do bat dinh
doi voi QUAN THE BAI BAO; no KHONG cho biet 16 checkpoint tan mac ra sao. Hinh nay bo
sung dung phan con thieu do: moi hop = phan bo FPR cua 16 checkpoint tai mot k.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "figures"

KS = [1, 2, 3, 5, 8, 10, 15, 20]
METHODS = [("static", "Static (pinned at k=20)", "#8c8c8c"),
           ("emp_deff", "Raw scale, empirical quantile", "#c44e52"),
           ("zscale", "z-scale (proposed)", "#4c72b0")]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(ROOT / "revision_experiments_rivf2026" / "results"
                    / "rev12_per_checkpoint.csv")
    # mot gia tri / checkpoint: trung binh qua 5 lan chia truoc, roi moi lay phan bo
    per = D.groupby(["method", "k", "model"]).fpr.mean().reset_index()

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.24
    for j, (meth, lab, col) in enumerate(METHODS):
        data = [100 * per[(per.method == meth) & (per.k == k)].fpr.to_numpy() for k in KS]
        pos = np.arange(len(KS)) + (j - 1) * w
        bp = ax.boxplot(data, positions=pos, widths=w * 0.85, patch_artist=True,
                        showfliers=True, medianprops=dict(color="black", lw=1.2),
                        flierprops=dict(marker=".", markersize=3, alpha=0.6))
        for b in bp["boxes"]:
            b.set(facecolor=col, alpha=0.65, edgecolor=col)
        for part in ("whiskers", "caps"):
            for it in bp[part]:
                it.set(color=col)
        ax.plot([], [], color=col, lw=6, alpha=0.65, label=lab)

    ax.axhline(1.0, color="black", ls=":", lw=1.1, zorder=0)
    ax.text(len(KS) - 0.45, 1.06, "1% target", fontsize=8, va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.5, 1, 2, 5, 10])
    ax.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10"])
    ax.set_xticks(np.arange(len(KS)))
    ax.set_xticklabels(KS)
    ax.set_xlabel("Chunks per document, $k$")
    ax.set_ylabel("Document-level FPR (%)")
    ax.set_title("Spread across the 16 checkpoints at each $k$", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", color="0.88", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("figS1_fpr_boxplot.%s" % ext), dpi=200)

    print("=" * 80)
    print("Bien do giua 16 checkpoint (max - min, diem %)")
    print("=" * 80)
    print("%4s %12s %12s %12s" % ("k", "static", "emp", "z-scale"))
    for k in KS:
        r = ["%12.2f" % (100 * (lambda s: s.max() - s.min())(
            per[(per.method == m) & (per.k == k)].fpr)) for m, _, _ in METHODS]
        print("%4d %s" % (k, "".join(r)))
    print("\n-> supplement_rivf2026/figures/figS1_fpr_boxplot.{pdf,png}")


if __name__ == "__main__":
    main()
