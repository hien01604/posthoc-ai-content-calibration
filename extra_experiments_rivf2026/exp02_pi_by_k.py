"""EXP-02 -- Do ben vung KEP cua pi_hat theo ca ti le pi LAN do dai k.

    python3 exp02_pi_by_k.py

VAN DE
------
Bai tu nhan la "length-robust", nhung dieu do moi duoc kiem cho QUYET DINH gan co
(Z_k), chua bao gio kiem cho chinh UOC LUONG TI LE pi_hat. Neu pi_hat troi theo k thi
con so % bao cho nguoi dung phu thuoc do dai tai lieu -- hong y nghia dien giai.

CACH LAM
--------
Dung LAI y nguyen may sinh tai lieu tong hop cua exp01_calibration.py (import truc
tiep, khong chep lai), chi doi k. Do:

    Delta_k(pi) = max_k E[pi_hat | pi, k] - min_k E[pi_hat | pi, k]

Mot con so duy nhat luong hoa do nhay theo do dai. Can so sanh voi khoang cach giua
cac muc pi (0,25) -- Delta_k phai nho hon HAN moi goi la ben vung.

BO SUNG (re, va tra loi cau hoi van hanh that)
----------------------------------------------
Ty le GAN CO theo pi: dung quy tac Z_k (Eq. 4, rho rieng tung checkpoint) tren chinh
cac tai lieu tong hop nay -> duong "bao nhieu %% AI thi he thong bat dau gan co".
EXP-06 cho thay tai lieu TOAN AI bi bat gan 100%% voi moi k>=5, nen truc thong tin
that nam o cac muc pi TRUNG GIAN.

DAU RA
------
  results/exp02_pi_by_k.csv        true_pi, k, mean_pi_hat, mae (+ p5/p95, flag rate)
  results/exp02_delta_k.csv        true_pi, delta_k
  figures/exp02_pi_by_k_lines.pdf/.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from exp01_calibration import NDOC, SEED, anchors, paired_pools, mask_with_exactly

HERE = Path(__file__).resolve().parent          # extra_experiments_rivf2026/
ROOT = HERE.parent                              # goc repo
ART = ROOT / "artifacts" / "kaggle"
FPF = ROOT / "artifacts" / "kaggle_testfp_full" / "pred_testfp_full"
OUT = ROOT / "output"
RES = HERE / "results"
FIG = HERE / "figures"

KS = [5, 10, 20]

# LUOI pi PHAI DAT CHINH XAC O CA BA k -- neu khong, Delta_k do LOI LAM TRON chu khong
# do do nhay theo do dai. Lan chay dau dung luoi cua EXP-01 {0; 0,1; 0,25; 0,5; 0,75;
# 0,9; 1} va ra Delta_k = 0,101; truy ra toan bo la vi round(pi*k) o k=5 bien pi=0,10
# thanh 0/5 = 0,00 va pi=0,50 thanh 2/5 = 0,40 -- pi_hat uoc DUNG cac ti le thuc do
# (0,0043 va 0,3945). Boi so cua 1/5 thi k=5, 10, 20 deu bieu dien duoc chinh xac.
PIS = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
ALPHA = 0.01
Z1A = float(norm.isf(ALPHA))


def paired_docs(sH, sA, pi, k, rng):
    """Nhu design_paired cua EXP-01 nhung k tuy y. Tra ve (NDOC, k) diem doan."""
    n = len(sH)
    slots = np.argsort(rng.random((NDOC, n)), axis=1)[:, :k]
    isai = mask_with_exactly(NDOC, k, int(round(pi * k)), rng)
    return np.where(isai, sA[slots], sH[slots])


def midp_pool(model):
    """Be mid-p: doan human cua test_fp_full (cung nguon voi §V-A/C)."""
    f = FPF / ("pred_test_fp_full_artifacts_%s.parquet" % model)
    d = pd.read_parquet(f).dropna(subset=["score"])
    return np.sort(d.score.to_numpy())


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    anc = anchors()
    rho_by_model = (pd.read_csv(OUT / "tau_k_mean_16models.csv")
                      .groupby("model")["rho"].median())

    rows = []
    for model, r in anc.iterrows():
        rng = np.random.default_rng(SEED)
        sH, sA = paired_pools(model)
        pool = midp_pool(model)
        nc = len(pool)
        rho = float(rho_by_model.loc[model])
        denom = r.mu1 - r.mu0
        for k in KS:
            tau_z = Z1A * np.sqrt(1 + (k - 1) * rho)
            for pi in PIS:
                S = paired_docs(sH, sA, pi, k, rng)
                ph = (S.mean(axis=1) - r.mu0) / denom
                pv = (0.5 + nc - np.searchsorted(pool, S, side="left")) / (nc + 1.0)
                Zk = norm.isf(pv).sum(axis=1) / np.sqrt(k)
                assert int(round(pi * k)) == pi * k, \
                    "pi=%s khong dat chinh xac o k=%d" % (pi, k)
                rows.append(dict(
                    model=model, k=k, true_pi=pi, n=NDOC,
                    mean_pi_hat=float(ph.mean()), median_pi_hat=float(np.median(ph)),
                    mae=float(np.abs(ph - pi).mean()),
                    bias=float(ph.mean() - pi), sd=float(ph.std(ddof=1)),
                    p5=float(np.percentile(ph, 5)), p95=float(np.percentile(ph, 95)),
                    flag_rate=float((Zk > tau_z).mean()),
                ))
        print("  xong %s" % model)

    D = pd.DataFrame(rows)
    D.to_csv(RES / "exp02_per_checkpoint.csv", index=False)

    agg = (D.groupby(["k", "true_pi"])
             .agg(mean_pi_hat=("mean_pi_hat", "mean"),
                  median_pi_hat=("median_pi_hat", "median"),
                  mae=("mae", "median"), p5=("p5", "median"), p95=("p95", "median"),
                  flag_rate=("flag_rate", "median"), n=("n", "sum")).reset_index())
    agg.to_csv(RES / "exp02_pi_by_k.csv", index=False)

    dk = (agg.groupby("true_pi")
             .agg(hi=("mean_pi_hat", "max"), lo=("mean_pi_hat", "min")).reset_index())
    dk["delta_k"] = dk.hi - dk.lo
    dk[["true_pi", "delta_k"]].to_csv(RES / "exp02_delta_k.csv", index=False)

    print("\nE[pi_hat | pi, k] -- trung binh qua 16 checkpoint")
    print("   pi  " + "".join("%12s" % ("k=%d" % k) for k in KS) + "%12s" % "Delta_k")
    for pi in PIS:
        line = "%5.2f " % pi
        for k in KS:
            line += "%12.4f" % agg[(agg.k == k) & (agg.true_pi == pi)].mean_pi_hat.iloc[0]
        line += "%12.4f" % dk[dk.true_pi == pi].delta_k.iloc[0]
        print(line)
    gap = PIS[1] - PIS[0]
    print("\n  Delta_k lon nhat = %.4f  (khoang cach giua cac muc pi = %.2f)"
          % (dk.delta_k.max(), gap))
    print("  -> ty so Delta_k / %.2f = %.1f%%" % (gap, 100 * dk.delta_k.max() / gap))

    print("\nMAE theo (pi, k)")
    print("   pi  " + "".join("%12s" % ("k=%d" % k) for k in KS))
    for pi in PIS:
        print("%5.2f " % pi + "".join(
            "%12.4f" % agg[(agg.k == k) & (agg.true_pi == pi)].mae.iloc[0] for k in KS))

    print("\nTy le GAN CO (%) theo quy tac Z_k, muc tieu FPR 1%% tai pi=0")
    print("   pi  " + "".join("%12s" % ("k=%d" % k) for k in KS))
    for pi in PIS:
        print("%5.2f " % pi + "".join(
            "%12.2f" % (100 * agg[(agg.k == k) & (agg.true_pi == pi)].flag_rate.iloc[0])
            for k in KS))

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    cols = {5: "#1f77b4", 10: "#2ca02c", 20: "#d62728"}
    mks = {5: "o", 10: "^", 20: "s"}

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="0.55", ls=":", lw=1.4, label="ideal", zorder=1)
    for k in KS:
        a = agg[agg.k == k].set_index("true_pi")
        x = np.array(PIS)
        ax.fill_between(x, a.loc[x, "p5"], a.loc[x, "p95"], color=cols[k],
                        alpha=0.10, zorder=2)
        ax.plot(x, a.loc[x, "mean_pi_hat"], color=cols[k], marker=mks[k], lw=1.8,
                ms=5, label="$k=%d$" % k, zorder=3)
    ax.set_xlabel("True AI fraction $\\pi$")
    ax.set_ylabel("Estimated $\\hat{\\pi}$")
    ax.set_xticks(PIS); ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("(a) $\\hat{\\pi}$ is invariant to document length", fontsize=10)

    ax = axes[1]
    for k in KS:
        a = agg[agg.k == k].set_index("true_pi")
        ax.plot(PIS, [100 * a.loc[p, "flag_rate"] for p in PIS], color=cols[k],
                marker=mks[k], lw=1.8, ms=5, label="$k=%d$" % k)
    ax.axhline(1.0, color="0.55", ls=":", lw=1.4)
    ax.set_xlabel("True AI fraction $\\pi$")
    ax.set_ylabel("Flagged by $Z_k$ rule (%)")
    ax.set_xticks(PIS); ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("(b) Detection rate vs. AI fraction", fontsize=10)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("exp02_pi_by_k_lines.%s" % ext), dpi=200)
    print("\n-> results/exp02_pi_by_k.csv, exp02_delta_k.csv, exp02_per_checkpoint.csv")
    print("-> figures/exp02_pi_by_k_lines.pdf/.png")


if __name__ == "__main__":
    main()
