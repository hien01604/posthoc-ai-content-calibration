"""F5 -- HIEU UNG THIET KE CO CAN THIET KHONG? Luoi 2x2: (thang do) x (rho vs rho=0).

    py f5_deff_necessity.py

VAN DE (phan bien A1/B5 cua reviewer)
-------------------------------------
Bai ten la "design-effect corrected" nhung KHONG BAO GIO chung minh phan hieu chinh
tuong quan la can thiet. Bang III chi so: raw+Gauss (hong), raw+emp (duoc), z+Hartung
(duoc), oracle. Ca ba duong "duoc" deu CO hieu ung thiet ke, nen khong the biet no
dong gop gi. Doi chung con thieu la chinh no: BO rho di (dat rho = 0).

Nhan xet giai tich quan trong (giai thich tai sao k=1 hai luat gan trung nhau):
  p_i la conformal p-value tinh tren chinh pool cal -> p_i ~ Uniform tren pool theo
  DINH NGHIA -> phan vi 99% thuc nghiem cua z_i BANG Phi^-1(0,99) = 2,326.
  Vi Phi^-1(1-p) don dieu theo s, phan vi 99% thuc nghiem cua MOT doan tren thang tho
  anh xa DUNG sang 2,326 tren thang z. Nghia la o k=1, "raw+quantile thuc nghiem" va
  "z+quantile Gauss" LA CUNG MOT LUAT; chenh lech quan sat duoc chi la nhieu uoc luong.
  => Moi khac biet o k>1 quy ve DUY NHAT mot dieu: lay trung binh tren thang nao.

LUOI 2x2 (quantile giu dung o muc "PIT-chuan" cho ca 4 o)
---------------------------------------------------------
                    co hieu ung thiet ke            rho = 0 (bo qua tuong quan)
  thang z      z_hartung: z_a*sqrt(1+(k-1)rho_z)    z_norho: z_a
  thang tho    raw_emp:   mu0+(q-mu0)*sqrt((1+(k-1)rho_r)/k)
                                                    raw_norho: mu0+(q-mu0)/sqrt(k)

Neu cot phai hong nang -> hieu chinh tuong quan LA can thiet (ket qua khang dinh ma
bai dang thieu). Neu cot phai van tot -> phai ha claim trong tieu de bai.

THIET KE do y het f2_paired_bootstrap.py (cung TK, cung seed, cung cach chia cal/eval,
cung quan the 576 bai >= 20 doan) de so sanh truc tiep duoc voi Bang III.

DAU RA (results/)
-----------------
  f5_deff_necessity.csv   FPR(%) theo k cho 4 o luoi + oracle, 16 checkpoint x 5 split
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from icc import icc1_and_var

import paper_level_tau_k as TK   # noqa: E402

RES = HERE / "results"
KS_SHOW = [1, 2, 3, 5, 8, 10, 15, 20]
Z1A = float(norm.isf(TK.ALPHA))

METHODS = ["z_hartung", "z_norho", "raw_emp", "raw_norho", "oracle"]
NAMES = {"z_hartung": "z-scale, design effect (proposed)",
         "z_norho":   "z-scale, rho = 0 (Stouffer, no correction)",
         "raw_emp":   "raw scale, empirical quantile, design effect",
         "raw_norho": "raw scale, empirical quantile, rho = 0",
         "oracle":    "per-k oracle"}


def rho_raw_icc(cal_papers):
    """ICC(1) mot chieu tren thang THO -- xem icc.py."""
    return icc1_and_var(cal_papers, TK.K0_RHO)


def main():
    RES.mkdir(exist_ok=True)
    files = sorted(TK.FP_FULL.glob("pred_test_fp_full_artifacts_*.parquet"))
    KS = np.arange(1, TK.KMAX + 1)
    rows = []

    for f in files:
        model = f.stem.replace("pred_test_fp_full_artifacts_", "")
        rng = np.random.default_rng(TK.SEED)
        d = pd.read_parquet(f).dropna(subset=["score"])
        by_paper = {p: g.score.to_numpy() for p, g in d.groupby("paper_id")}
        ids = np.array(sorted(p for p, v in by_paper.items() if len(v) >= TK.KMAX))
        papers = [by_paper[p] for p in ids]

        M = TK.draw(papers, rng)                    # (R, n, KMAX)
        Craw = np.cumsum(M, axis=2) / KS            # trung binh k doan, thang tho

        for sp in range(TK.S):
            perm = rng.permutation(len(ids))
            cal, ev = perm[: len(ids) // 2], perm[len(ids) // 2:]

            pool = np.sort(np.concatenate([by_paper[p] for p in ids[cal]]))
            nc = len(pool)
            pv = (0.5 + nc - np.searchsorted(pool, M, side="left")) / (nc + 1.0)
            Cz = np.cumsum(norm.isf(pv), axis=2) / np.sqrt(KS)
            rho_z = TK.rho_hartung(norm.isf(pv[0][cal][:, : TK.K0_RHO]))

            cal_papers = [by_paper[p] for p in ids[cal]]
            rho_r, _ = rho_raw_icc(cal_papers)
            mu0 = np.concatenate(cal_papers).mean()

            # phan vi 99% thuc nghiem tren nua cal, tai tung k (oracle)
            Q = np.quantile(Craw[:, cal, :], 1 - TK.ALPHA, axis=1,
                            method="higher").mean(axis=0)
            # phan vi 99% thuc nghiem cua MOT doan, tren thang z (kiem chung giai tich)
            Qz1 = float(np.quantile(Cz[:, cal, 0], 1 - TK.ALPHA, method="higher").mean())

            deff = np.sqrt(1 + (KS - 1) * rho_z)
            tau = {
                "z_hartung": Z1A * deff,
                "z_norho":   np.full(TK.KMAX, Z1A),
                "raw_emp":   mu0 + (Q[0] - mu0) * np.sqrt((1 + (KS - 1) * rho_r) / KS),
                "raw_norho": mu0 + (Q[0] - mu0) / np.sqrt(KS),
                "oracle":    Q,
            }
            for meth in METHODS:
                src = Cz if meth.startswith("z_") else Craw
                fpr = (src[:, ev, :] >= tau[meth][None, None, :]).mean(axis=(0, 1))
                for k in KS_SHOW:
                    rows.append(dict(model=model, split=sp, k=int(k), method=meth,
                                     fpr_pct=100 * float(fpr[k - 1]),
                                     rho_z=rho_z, rho_raw=rho_r,
                                     q_raw_k1=float(Q[0]), q_z_k1=Qz1, z_alpha=Z1A))
        print("  xong %s" % model)

    D = pd.DataFrame(rows)
    D.to_csv(RES / "f5_deff_necessity.csv", index=False)

    per = D.groupby(["method", "model", "k"]).fpr_pct.mean().reset_index()
    piv = per.groupby(["method", "k"]).fpr_pct.median().unstack("method")

    print("\nFPR (%%) so voi muc tieu 1,0%% -- trung vi qua 16 checkpoint")
    print("  k  " + "".join("%12s" % m for m in METHODS))
    for k in KS_SHOW:
        print("%3d  " % k + "".join("%12.2f" % piv.loc[k, m] for m in METHODS))

    print("\nKIEM CHUNG GIAI TICH o k=1: phan vi 99%% thuc nghiem tren thang z")
    q = D[D.k == 1].groupby("model")[["q_z_k1"]].first()
    print("  trung vi %.4f  dai [%.4f, %.4f]   so voi Phi^-1(0,99) = %.4f"
          % (q.q_z_k1.median(), q.q_z_k1.min(), q.q_z_k1.max(), Z1A))

    print("\n-> results/f5_deff_necessity.csv")


if __name__ == "__main__":
    main()
