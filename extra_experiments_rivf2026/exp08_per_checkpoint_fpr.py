"""EXP-08 -- Phan bo FPR theo TUNG checkpoint (khong chi trung vi).

    python3 exp08_per_checkpoint_fpr.py

VAN DE
------
Table III/IV chi bao trung vi qua 16 checkpoint. Trung vi che duoc nhung checkpoint
hieu chuan te: neu 4/16 checkpoint co FPR 2-3%% thi trung vi 0,8%% van dep. Reviewer
co quyen doi thay phan bo.

CACH LAM
--------
Thuan gop, khong chay lai gi: `fpr_law_nocorr` trong tau_k_mean_16models.csv da co
san theo (model, split, k). Trung binh qua 5 lan chia -> 1 so/checkpoint/k.

DAU RA
------
  results/exp08_per_checkpoint_fpr.csv   checkpoint, k, fpr
  figures/exp08_fpr_boxplot.pdf/.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # extra_experiments_rivf2026/
ROOT = HERE.parent                              # goc repo
OUT = ROOT / "output"
RES = HERE / "results"
FIG = HERE / "figures"

KS_SHOW = [1, 5, 10, 20]
BAND = (0.005, 0.015)          # dai dung sai 0,5%-1,5%

NICE = {
    "deberta_v3_large_seed42": "DeBERTa-v3-large (s42)",
    "deberta_v3_large_seed43": "DeBERTa-v3-large (s43)",
    "deberta_v3_large_seed44": "DeBERTa-v3-large (s44)",
    "greater_roberta_large_seed42": "GREATER + RoBERTa-large (s42)",
    "greater_xlm_roberta_large_seed42": "GREATER + XLM-R-large (s42)",
    "modernbert_large_seed42": "ModernBERT-large (s42)",
    "modernbert_large_seed43": "ModernBERT-large (s43)",
    "modernbert_large_seed44": "ModernBERT-large (s44)",
    "radar_seed42": "RADAR (s42)",
    "radar_xlm_seed42": "RADAR + XLM-R-large (s42)",
    "roberta_large_seed42": "RoBERTa-large (s42)",
    "roberta_large_seed43": "RoBERTa-large (s43)",
    "roberta_large_seed44": "RoBERTa-large (s44)",
    "xlm_roberta_large_seed42": "XLM-RoBERTa-large (s42)",
    "xlm_roberta_large_seed43": "XLM-RoBERTa-large (s43)",
    "xlm_roberta_large_seed44": "XLM-RoBERTa-large (s44)",
}


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    T = pd.read_csv(OUT / "tau_k_mean_16models.csv")
    per = (T[T.k.isin(KS_SHOW)]
           .groupby(["model", "k"]).agg(fpr=("fpr_law_nocorr", "mean")).reset_index())
    per["checkpoint"] = per.model.map(NICE).fillna(per.model)
    per[["checkpoint", "k", "fpr"]].to_csv(RES / "exp08_per_checkpoint_fpr.csv",
                                           index=False)

    print("Phan bo FPR (%) cua hieu chinh thang z, qua 16 checkpoint")
    print("  k   median     min     max     IQR          trong dai 0,5-1,5%")
    summ = []
    for k in KS_SHOW:
        v = per[per.k == k].fpr.to_numpy() * 100
        q1, q3 = np.percentile(v, [25, 75])
        inb = int(((v >= BAND[0] * 100) & (v <= BAND[1] * 100)).sum())
        summ.append(dict(k=k, median=np.median(v), min=v.min(), max=v.max(),
                         q1=q1, q3=q3, n_in_band=inb, n=len(v)))
        print("%3d   %6.2f  %6.2f  %6.2f  %5.2f-%-5.2f   %2d/%d"
              % (k, np.median(v), v.min(), v.max(), q1, q3, inb, len(v)))
    pd.DataFrame(summ).to_csv(RES / "exp08_summary.csv", index=False)

    worst = per[per.k == 20].nlargest(3, "fpr")[["checkpoint", "fpr"]]
    best = per[per.k == 20].nsmallest(3, "fpr")[["checkpoint", "fpr"]]
    print("\nTai k=20 -- 3 checkpoint FPR CAO nhat:")
    for _, r in worst.iterrows():
        print("   %-32s %.2f%%" % (r.checkpoint, 100 * r.fpr))
    print("3 checkpoint FPR THAP nhat (qua bao thu):")
    for _, r in best.iterrows():
        print("   %-32s %.2f%%" % (r.checkpoint, 100 * r.fpr))

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    data = [per[per.k == k].fpr.to_numpy() * 100 for k in KS_SHOW]
    bp = ax.boxplot(data, labels=["$k=%d$" % k for k in KS_SHOW], widths=0.55,
                    patch_artist=True, medianprops=dict(color="black", lw=1.6))
    for b in bp["boxes"]:
        b.set(facecolor="#9ecae1", alpha=0.75, edgecolor="#3182bd")
    rng = np.random.default_rng(0)
    for i, v in enumerate(data):
        ax.scatter(np.full(len(v), i + 1) + rng.uniform(-0.13, 0.13, len(v)), v,
                   s=11, color="#08519c", alpha=0.65, zorder=3)
    ax.axhline(1.0, color="#d62728", ls="--", lw=1.4, label="1% target")
    ax.axhspan(BAND[0] * 100, BAND[1] * 100, color="#d62728", alpha=0.07, zorder=0)
    ax.set_ylabel("Document-level FPR (%)")
    ax.set_ylim(0, max(max(d) for d in data) * 1.15)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Per-checkpoint FPR of the $z$-scale correction", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("exp08_fpr_boxplot.%s" % ext), dpi=200)

    print("\n-> results/exp08_per_checkpoint_fpr.csv, exp08_summary.csv")
    print("-> figures/exp08_fpr_boxplot.pdf/.png")


if __name__ == "__main__":
    main()
