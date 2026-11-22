"""BREAKDOWN -- tach EXP-01 va EXP-06 theo TUNG checkpoint + tach MAE thanh bias/spread.

    python3 extra_experiments_rivf2026/breakdown_per_checkpoint.py

VAN DE 1 -- trung vi co dang giau checkpoint hong khong?
--------------------------------------------------------
EXP-08 da hoi dung cau nay cho FPR va tra loi duoc. Nhung MAE ~ 0,039 (EXP-01) va
khoang cach oracle 0,14 diem % (EXP-06) van dang la SO GOP qua 16 checkpoint. Reviewer
se hoi lai y het cau do cho hai thi nghiem moi. Script nay gop lai so DA CO trong
exp01/exp02/exp06_per_checkpoint.csv -- khong tinh lai gi, khong chay lai detector.

VAN DE 2 -- "MAE nho" nghia la gi?
-----------------------------------
MAE nho co the den tu hai cau chuyen RAT khac nhau:
  (a) uoc luong TOT   -> bias nho VA do tan nho;
  (b) uoc luong TAM THUONG nhung khong lech -> bias nho, do tan LON, trung binh van trung.
Phan biet bang cach tach:
      MAE  ~  |bias|  (thanh phan he thong)   +   do tan  (thanh phan ngau nhien)
Voi phan phoi doi xung va bias = 0 thi MAE = sqrt(2/pi) * SD ~ 0,798 * SD. Neu ty so
MAE / (0,798*SD) ~ 1 va |bias|/SD << 1 -> toan bo MAE la DO TAN, khong phai thien lech.
Do la ket luan phai noi ro khi viet bai, vi no quyet dinh cach dien giai con so %.

DAU VAO (deu la ket qua da sinh san, chi doc lai)
------------------------------------------------
  results/exp01_per_checkpoint.csv   design, model, true_pi, mae, bias, sd, p5, p95
  results/exp02_per_checkpoint.csv   model, k, true_pi, mae, bias, sd, flag_rate
  results/exp06_per_checkpoint.csv   method, model, k, fpr, miss

DAU RA
------
  results/breakdown_exp01_by_checkpoint.csv   MAE tung checkpoint x tung pi
  results/breakdown_exp06_by_checkpoint.csv   khoang cach oracle tung checkpoint x k
  results/breakdown_mae_decomposition.csv     bias / SD / MAE cho EXP-01 va EXP-02
  figures/breakdown_boxplots.pdf/.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

BAND01 = 0.06        # nguong "chap nhan duoc" cho MAE cua pi_hat
BAND06 = 0.005       # nguong "bam sat oracle": lech < 0,5 diem %
SQ = np.sqrt(2.0 / np.pi)      # MAE = SQ * SD khi bias=0, phan phoi chuan

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


def nm(s):
    return NICE.get(s, s)


def dist(v):
    q1, q3 = np.percentile(v, [25, 75])
    return np.median(v), v.min(), v.max(), q1, q3


# ===================================================================== EXP-01
def exp01():
    D = pd.read_csv(RES / "exp01_per_checkpoint.csv")
    D["checkpoint"] = D.model.map(nm)
    D[["design", "checkpoint", "true_pi", "mae", "bias", "sd", "p5", "p95"]].to_csv(
        RES / "breakdown_exp01_by_checkpoint.csv", index=False)

    print("=" * 78)
    print("EXP-01 -- MAE cua pi_hat theo TUNG checkpoint (thiet ke ghep cap)")
    print("=" * 78)
    print("   pi   trung vi     min     max        IQR      so ckpt MAE<%.2f" % BAND01)
    P = D[D.design == "paired"]
    for pi in sorted(P.true_pi.unique()):
        v = P[P.true_pi == pi].mae.to_numpy()
        med, lo, hi, q1, q3 = dist(v)
        print("%5.2f   %8.4f %7.4f %7.4f  %.4f-%.4f   %2d/%d"
              % (pi, med, lo, hi, q1, q3, int((v < BAND01).sum()), len(v)))

    worst = (P.groupby("checkpoint").mae.mean().sort_values(ascending=False))
    print("\nMAE trung binh qua moi muc pi -- 3 checkpoint TE nhat / TOT nhat:")
    for c, v in worst.head(3).items():
        print("   te   %-32s %.4f" % (c, v))
    for c, v in worst.tail(3)[::-1].items():
        print("   tot  %-32s %.4f" % (c, v))
    print("\n-> chenh giua checkpoint te nhat va tot nhat: %.4f (%.1fx)"
          % (worst.max() - worst.min(), worst.max() / worst.min()))
    return P


# ===================================================================== EXP-06
def exp06():
    D = pd.read_csv(RES / "exp06_per_checkpoint.csv")
    piv = (D[D.method.isin(["oracle", "zscale"])]
           .pivot_table(index=["model", "k"], columns="method", values="fpr")
           .reset_index())
    piv["gap"] = piv.zscale - piv.oracle
    piv["abs_gap"] = piv.gap.abs()
    piv["checkpoint"] = piv.model.map(nm)
    piv[["checkpoint", "k", "oracle", "zscale", "gap", "abs_gap"]].to_csv(
        RES / "breakdown_exp06_by_checkpoint.csv", index=False)

    print("\n" + "=" * 78)
    print("EXP-06 -- |FPR(de xuat) - FPR(oracle)| theo TUNG checkpoint, diem %")
    print("=" * 78)
    print("   k   trung vi     min     max        IQR      so ckpt < %.1f diem"
          % (BAND06 * 100))
    for k in sorted(piv.k.unique()):
        v = piv[piv.k == k].abs_gap.to_numpy() * 100
        med, lo, hi, q1, q3 = dist(v)
        print("%4d   %8.2f %7.2f %7.2f  %.2f-%.2f    %2d/%d"
              % (k, med, lo, hi, q1, q3,
                 int((v < BAND06 * 100).sum()), len(v)))

    w = piv.groupby("checkpoint").abs_gap.mean().sort_values(ascending=False)
    print("\nKhoang cach TB qua moi k -- 3 checkpoint xa oracle nhat:")
    for c, v in w.head(3).items():
        print("   %-32s %.2f diem %%" % (c, 100 * v))
    mx = piv.loc[piv.abs_gap.idxmax()]
    print("\nTruong hop xau nhat tuyet doi: %s tai k=%d -- oracle %.2f%%, de xuat %.2f%%"
          " (lech %.2f diem)"
          % (mx.checkpoint, mx.k, 100 * mx.oracle, 100 * mx.zscale, 100 * mx.abs_gap))
    return piv


# ============================================================ tach MAE
def decomposition():
    rows = []
    for tag, f, keys in (("EXP-01 (k=20)", "exp01_per_checkpoint.csv",
                          ["design", "true_pi"]),
                         ("EXP-02", "exp02_per_checkpoint.csv", ["k", "true_pi"])):
        D = pd.read_csv(RES / f)
        if "sd" not in D.columns:
            print("  (!) %s thieu cot sd -- chay lai script sinh ra no truoc" % f)
            continue
        g = D.groupby(keys).agg(mae=("mae", "median"), bias=("bias", "mean"),
                                sd=("sd", "median")).reset_index()
        g["exp"] = tag
        g["abs_bias"] = g.bias.abs()
        g["mae_pred_from_sd"] = SQ * g.sd          # MAE ky vong neu bias=0
        g["ratio"] = g.mae / g.mae_pred_from_sd
        g["bias_over_sd"] = g.abs_bias / g.sd
        rows.append(g)
    A = pd.concat(rows, ignore_index=True)
    A.to_csv(RES / "breakdown_mae_decomposition.csv", index=False)

    print("\n" + "=" * 78)
    print("TACH MAE: he thong (bias) hay ngau nhien (do tan)?")
    print("=" * 78)
    print("MAE = 0,798*SD neu bias=0 va phan phoi doi xung.")
    print("  ty so MAE/(0,798*SD) ~ 1  VA  |bias|/SD << 1  =>  MAE hoan toan la DO TAN\n")
    sub = A[(A.exp == "EXP-01 (k=20)") & (A.design == "paired")]
    print("EXP-01, thiet ke ghep cap, k=20:")
    print("   pi      MAE      SD   |bias|   MAE/(0,798*SD)   |bias|/SD")
    for _, r in sub.iterrows():
        print("%5.2f   %6.4f  %6.4f  %6.4f        %8.2f      %8.2f"
              % (r.true_pi, r.mae, r.sd, r.abs_bias, r.ratio, r.bias_over_sd))
    print("\n   -> |bias|/SD trung vi = %.2f ; MAE/(0,798*SD) trung vi = %.2f"
          % (sub.bias_over_sd.median(), sub.ratio.median()))

    sub2 = A[A.exp == "EXP-02"]
    if len(sub2):
        print("\nEXP-02 -- do tan giam theo k (SD trung vi qua cac muc pi):")
        for k in sorted(sub2.k.unique()):
            s = sub2[sub2.k == k]
            print("   k=%2d   SD %.4f   MAE %.4f   |bias|/SD %.2f"
                  % (k, s.sd.median(), s.mae.median(), s.bias_over_sd.median()))
    return A, sub


# ===================================================================== figure
def figure(P, piv, sub):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))

    ax = axes[0]
    pis = sorted(P.true_pi.unique())
    data = [P[P.true_pi == p].mae.to_numpy() for p in pis]
    bp = ax.boxplot(data, tick_labels=["%.2f" % p for p in pis], widths=0.6,
                    patch_artist=True, medianprops=dict(color="black", lw=1.5))
    for b in bp["boxes"]:
        b.set(facecolor="#9ecae1", alpha=0.75, edgecolor="#3182bd")
    ax.set_xlabel("True AI fraction $\\pi$")
    ax.set_ylabel("Per-checkpoint MAE of $\\hat{\\pi}$")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("(a) EXP-01: MAE across 16 checkpoints", fontsize=10)

    ax = axes[1]
    ks = sorted(piv.k.unique())
    data = [piv[piv.k == k].abs_gap.to_numpy() * 100 for k in ks]
    bp = ax.boxplot(data, tick_labels=["%d" % k for k in ks], widths=0.6,
                    patch_artist=True, medianprops=dict(color="black", lw=1.5))
    for b in bp["boxes"]:
        b.set(facecolor="#a1d99b", alpha=0.75, edgecolor="#31a354")
    ax.axhline(0.5, color="#d62728", ls="--", lw=1.3, label="0.5 pp")
    ax.set_xlabel("Chunks per document, $k$")
    ax.set_ylabel("$|$FPR$_{\\rm proposed}-$FPR$_{\\rm oracle}|$ (pp)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("(b) EXP-06: distance to oracle", fontsize=10)

    ax = axes[2]
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub.abs_bias, 0.4, label="$|$bias$|$", color="#fdae6b")
    ax.bar(x + 0.2, sub.sd, 0.4, label="SD", color="#6baed6")
    ax.plot(x, sub.mae, "k^--", ms=6, lw=1.4, label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(["%.2f" % p for p in sub.true_pi])
    ax.set_xlabel("True AI fraction $\\pi$")
    ax.set_ylabel("Error component")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("(c) MAE is spread, not bias", fontsize=10)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("breakdown_boxplots.%s" % ext), dpi=200)


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    P = exp01()
    piv = exp06()
    _, sub = decomposition()
    figure(P, piv, sub)
    print("\n-> results/breakdown_exp01_by_checkpoint.csv")
    print("-> results/breakdown_exp06_by_checkpoint.csv")
    print("-> results/breakdown_mae_decomposition.csv")
    print("-> figures/breakdown_boxplots.pdf/.png")


if __name__ == "__main__":
    main()
