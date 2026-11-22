"""F8 -- Can bao nhieu tai lieu hieu chuan de uoc rho_z du chinh xac?

    py final_experiments_rivf2026/f8_rho_sample_size.py

DONG CO
-------
Bai da bao rho_z dao dong 0.204-0.401 qua checkpoint, va nhieu +-20% lam miss rate o
tai lieu nua-AI nhay 17.3% -> 48.5%. Nhung nguoi trien khai can biet dieu khac: *ho*
phai gom bao nhieu tai lieu human de uoc rho_z cho rieng detector cua ho.

Cau tra loi suy duoc tu du lieu da co: lay lai mau n_cal tai lieu tu nua hieu chuan,
uoc rho_z, lap lai, do do lech chuan. Khong can chay model.

NGUONG THUC HANH
----------------
+-20% quanh rho_z=0.303 tuong duong +-0.061. De khoang 95% nam trong do can
SE(rho_z) <= 0.061/1.96 = 0.031.

DAU RA: results/f8_rho_sample_size.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import paper_level_tau_k as TK  # noqa: E402

RES = HERE / "results"
N_CAL = [25, 50, 100, 200, 288]
B = 200
SEED = 20260829
TARGET_SE = 0.031          # xem doc-string


def rho_z_from(papers, rng):
    """PIT tren chinh pool cua cac bai nay (dung nhu khi trien khai), roi Hartung."""
    pool = np.sort(np.concatenate(papers))
    m = len(pool)
    G = np.array([p[:TK.K0_RHO] for p in papers], dtype=float)
    pv = (0.5 + m - np.searchsorted(pool, G, side="left")) / (m + 1.0)
    return TK.rho_hartung(norm.isf(pv))


def main():
    RES.mkdir(exist_ok=True)
    files = sorted(TK.FP_FULL.glob("pred_test_fp_full_artifacts_*.parquet"))
    rows = []

    for f in files:
        model = f.stem.replace("pred_test_fp_full_artifacts_", "")
        d = pd.read_parquet(f).dropna(subset=["score"])
        by_paper = {p: g.score.to_numpy() for p, g in d.groupby("paper_id")}
        papers = [v for v in by_paper.values() if len(v) >= TK.K0_RHO]
        rng = np.random.default_rng(SEED)

        for n in N_CAL:
            if n > len(papers):
                continue
            est = []
            for _ in range(B):
                idx = rng.choice(len(papers), size=n, replace=False)
                est.append(rho_z_from([papers[i] for i in idx], rng))
            est = np.asarray(est)
            rows.append(dict(model=model, n_cal=n, rho_mean=est.mean(),
                             rho_sd=est.std(ddof=1),
                             lo=np.percentile(est, 2.5), hi=np.percentile(est, 97.5)))
        print("  xong %s" % model)

    D = pd.DataFrame(rows)
    D.to_csv(RES / "f8_rho_sample_size.csv", index=False)

    g = D.groupby("n_cal").agg(sd=("rho_sd", "median"),
                               rho=("rho_mean", "median")).reset_index()
    g["halfwidth"] = 1.96 * g.sd
    g["pct_of_rho"] = 100 * g.halfwidth / g.rho

    print("\nSE va nua-rong khoang 95%% cua rho_z theo so tai lieu hieu chuan")
    print("(trung vi qua 16 checkpoint, %d lan lay lai mau)\n" % B)
    print("  n_cal    SE     +-95%%     = +-%% cua rho_z")
    for r in g.itertuples():
        flag = "  <- dat nguong +-20%" if r.pct_of_rho <= 20 else ""
        print("  %5d  %.4f   %.3f      %5.1f%%%s"
              % (r.n_cal, r.sd, r.halfwidth, r.pct_of_rho, flag))

    ok = g[g.pct_of_rho <= 20]
    if len(ok):
        print("\n=> can khoang %d tai lieu human de rho_z chinh xac trong +-20%%"
              % int(ok.n_cal.min()))
    print("\n-> results/f8_rho_sample_size.csv")


if __name__ == "__main__":
    main()
