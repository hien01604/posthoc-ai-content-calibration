"""F4 -- DANG THUC cua phan phoi null cua Z_k: PIT chuan hoa BIEN, khong chuan hoa DONG THOI.

    py f4_null_shape.py

VAN DE (phan bien M6)
---------------------
Bai lap luan: "Under H0, p_i is approximately uniform and z_i is therefore approximately
N(0,1) by the probability integral transform, regardless of the skewness of the original
score distribution." Dung -- nhung chi dung theo BIEN.

Quy tac quyet dinh lai can Z_k = k^{-1/2} sum(z_i) ~ N(0, 1+(k-1)rho). Tong cua cac
bien co PHAN PHOI BIEN chuan chi chuan neu chung DONG THOI Gauss (tuc la can gia thiet
ve copula). PIT khong cho dieu do. Neu cac doan trong mot bai co phu thuoc DUOI (bai
"kho" thi MOI doan deu diem cao) thi duoi cua Z_k nang hon Gauss dung o vung dat nguong.

Script nay do truc tiep dang thuc do:

    T_k = Z_k / sqrt(1 + (k-1)*rho)        neu gia thiet dung -> T_k ~ N(0,1)

roi so phan vi thuc nghiem cua T_k voi phan vi Gauss tuong ung, tren TAI LIEU THAT.
Do lech o phan vi 99% chinh la phan FPR bi lech ma hieu ung thiet ke KHONG giai thich
duoc -- tach bach voi phan do rho uoc sai.

DAU RA
------
  results/f4_null_shape.csv       phan vi thuc nghiem vs Gauss, skew/kurtosis, theo k
  figures/f4_qq_null.pdf/.png     QQ plot cua T_k o k = 5, 10, 20
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import paper_level_tau_k as TK   # noqa: E402

RES = HERE / "results"
FIG = HERE / "figures"
KS_SHOW = [1, 2, 5, 10, 20]
KS_QQ = [5, 10, 20]
QUANTS = [0.90, 0.95, 0.99, 0.999]
SEED = 20260829


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    files = sorted(TK.FP_FULL.glob("pred_test_fp_full_artifacts_*.parquet"))
    KS = np.arange(1, TK.KMAX + 1)

    rows = []
    qq_store = {k: [] for k in KS_QQ}

    for f in files:
        model = f.stem.replace("pred_test_fp_full_artifacts_", "")
        rng = np.random.default_rng(TK.SEED)
        d = pd.read_parquet(f).dropna(subset=["score"])
        by_paper = {p: g.score.to_numpy() for p, g in d.groupby("paper_id")}
        ids = np.array(sorted(p for p, v in by_paper.items() if len(v) >= TK.KMAX))
        papers = [by_paper[p] for p in ids]

        M = TK.draw(papers, rng)
        for sp in range(TK.S):
            perm = rng.permutation(len(ids))
            cal, ev = perm[: len(ids) // 2], perm[len(ids) // 2:]
            pool = np.sort(np.concatenate([by_paper[p] for p in ids[cal]]))
            nc = len(pool)
            pv = (0.5 + nc - np.searchsorted(pool, M, side="left")) / (nc + 1.0)
            Cz = np.cumsum(norm.isf(pv), axis=2) / np.sqrt(KS)
            rho = TK.rho_hartung(norm.isf(pv[0][cal][:, : TK.K0_RHO]))

            for k in KS_SHOW:
                T = (Cz[:, ev, k - 1] / np.sqrt(1 + (k - 1) * rho)).ravel()
                r = dict(model=model, split=sp, k=k, rho=rho, n=T.size,
                         mean=float(T.mean()), sd=float(T.std(ddof=1)),
                         skew=float(skew(T)), exkurt=float(kurtosis(T)))
                for q in QUANTS:
                    r["q%.3f" % q] = float(np.quantile(T, q))
                    r["gauss%.3f" % q] = float(norm.ppf(q))
                # FPR thuc te tai nguong Gauss danh nghia 1%
                r["fpr_at_z99"] = float((T >= norm.isf(0.01)).mean())
                rows.append(r)
                if k in KS_QQ and sp == 0:
                    qq_store[k].append(T)
        print("  xong %s" % model)

    D = pd.DataFrame(rows)
    D.to_csv(RES / "f4_null_shape.csv", index=False)

    per = D.groupby(["model", "k"]).mean(numeric_only=True).reset_index()
    g = per.groupby("k").median(numeric_only=True)

    print("\nDANG THUC cua T_k = Z_k / sqrt(1+(k-1)rho) duoi H0, tren bai THAT")
    print("Neu gia thiet Gauss dong thoi dung: sd=1, skew=0, ex.kurt=0,")
    print("q99 = %.3f, va FPR tai nguong Gauss 1%% phai = 1,00%%\n" % norm.ppf(0.99))
    print("  k     sd    skew  ex.kurt     q95     q99    q99.9  | Gauss q99  FPR@z99")
    for k in KS_SHOW:
        r = g.loc[k]
        print("%3d  %5.3f  %+5.2f   %+5.2f   %6.3f  %6.3f  %6.3f  |   %5.3f    %5.2f%%"
              % (k, r["sd"], r["skew"], r["exkurt"], r["q0.950"], r["q0.990"],
                 r["q0.999"], norm.ppf(0.99), 100 * r["fpr_at_z99"]))

    # ------------------------------------------------------------------ QQ plot
    fig, axes = plt.subplots(1, len(KS_QQ), figsize=(9.6, 3.4), sharey=True)
    for ax, k in zip(axes, KS_QQ):
        T = np.concatenate(qq_store[k])
        T = np.sort(rng_sample(T, 40000))
        n = len(T)
        theo = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        ax.plot([-4, 5], [-4, 5], color="0.6", lw=0.9, ls="--", zorder=1)
        ax.plot(theo, T, lw=1.4, color="#2a4b8d", zorder=2)
        ax.axhline(norm.ppf(0.99), color="#a3392a", lw=0.8, ls=":", zorder=1)
        ax.axvline(norm.ppf(0.99), color="#a3392a", lw=0.8, ls=":", zorder=1)
        ax.set_title("$k=%d$" % k, fontsize=10)
        ax.set_xlabel("Gaussian quantile", fontsize=9)
        ax.set_xlim(-4, 4.6)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("$T_k$ quantile", fontsize=9)
    axes[0].set_ylim(-4, 4.6)
    fig.tight_layout()
    fig.savefig(FIG / "f4_qq_null.pdf")
    fig.savefig(FIG / "f4_qq_null.png", dpi=180)
    print("\n-> results/f4_null_shape.csv, figures/f4_qq_null.pdf/.png")


def rng_sample(x, n):
    """Lay mau ngau nhien co dinh seed de QQ plot khong qua nang."""
    if len(x) <= n:
        return x
    r = np.random.default_rng(SEED)
    return x[r.choice(len(x), n, replace=False)]


if __name__ == "__main__":
    main()
