"""So 2 cach uoc rho: ICC kieu Kish (dang dung trong paper_level_tau_k.py) vs
uoc luong rieng cua Hartung (1999), tren CUNG mot du lieu that.

    python3 paper_level_rho_compare.py

HAI CONG THUC
-------------
  Kish/ANOVA (dang dung):
      ms_b = k0 * Var(trung binh moi bai)
      ms_w = trung binh cua Var(trong moi bai)
      rho_kish = (ms_b - ms_w) / (ms_b + (k0-1)*ms_w)

  Hartung (1999), gia dinh phuong sai marginal duoi H0 CHINH XAC = 1:
      rho_hartung = 1 - E[ 1/(k0-1) * Sum (z_ij - mean_i(z))^2 ]
                  = 1 - trung binh cua phuong sai TRONG moi bai (ddof=1)

CAU HOI SO SANH
---------------
  Voi moi rho (kish/hartung), tinh lai tau_law0(k) = z_alpha*sqrt(1+(k-1)*rho),
  khop he so c rieng cho tung duong, roi so fpr_law voi fpr_emp (do tay) o
  tung k=1..20. Duong nao khop fpr_emp sat hon -> uoc luong do "phu hop" hon
  VOI DU LIEU NAY (khong phai dung tuyet doi ve mat ly thuyet).

DAU VAO / DAU RA giong paper_level_tau_k.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = Path(__file__).resolve().parent
FP_FULL = BASE / "artifacts" / "kaggle_testfp_full" / "pred_testfp_full"
OUT = BASE / "output"

ALPHA = 0.01
KMAX = 20
R = 200
S = 5
K0_RHO = 20
SEED = 42
KS = np.arange(1, KMAX + 1)
Z1A = float(norm.isf(ALPHA))


def draw(papers, rng):
    n = len(papers)
    L = max(len(a) for a in papers)
    pad = np.full((n, L), np.nan)
    for i, a in enumerate(papers):
        pad[i, : len(a)] = a
    invalid = np.isnan(pad)
    out = np.empty((R, n, KMAX))
    for r in range(R):
        key = rng.random((n, L))
        key[invalid] = np.inf
        idx = np.argsort(key, axis=1)[:, :KMAX]
        out[r] = np.take_along_axis(pad, idx, axis=1)
    return out


def icc_kish(Z):
    """ICC mot chieu tren ma tran (n bai x k0 doan) thang z -- dang dung trong code."""
    n, k0 = Z.shape
    ms_b = k0 * Z.mean(1).var(ddof=1)
    ms_w = Z.var(1, ddof=1).mean()
    v = (ms_b - ms_w) / (ms_b + (k0 - 1) * ms_w)
    return float(max(0.0, v)) if np.isfinite(v) else 0.0


def rho_hartung(Z):
    """Uoc luong rieng cua Hartung (1999): 1 - trung binh phuong sai TRONG moi bai."""
    within_var = Z.var(axis=1, ddof=1)  # (n,) -- dung ddof=1 = chia (k0-1)
    v = 1.0 - float(within_var.mean())
    return max(0.0, v)


def tau_curve(C, sel):
    Q = np.quantile(C[:, sel, :], 1 - ALPHA, axis=1, method="higher")
    return Q.mean(axis=0)


def main():
    rows = []
    files = sorted(FP_FULL.glob("pred_test_fp_full_artifacts_*.parquet"))
    if not files:
        raise SystemExit("Khong thay pred_test_fp_full_*.parquet")

    for f in files:
        model = f.stem.replace("pred_test_fp_full_artifacts_", "")
        rng = np.random.default_rng(SEED)
        df = pd.read_parquet(f).dropna(subset=["score"])
        by_paper = {p: g.score.to_numpy() for p, g in df.groupby("paper_id")}
        ids = np.array(sorted(p for p, v in by_paper.items() if len(v) >= KMAX))
        papers = [by_paper[p] for p in ids]

        M = draw(papers, rng)
        Craw = np.cumsum(M, axis=2) / KS

        for sp in range(S):
            perm = rng.permutation(len(ids))
            cal, ev = perm[: len(ids) // 2], perm[len(ids) // 2:]

            pool = np.sort(np.concatenate([by_paper[p] for p in ids[cal]]))
            nc = len(pool)
            pv = (0.5 + nc - np.searchsorted(pool, M, side="left")) / (nc + 1.0)
            Cz = np.cumsum(norm.isf(pv), axis=2) / np.sqrt(KS)

            Z_cal_k0 = norm.isf(pv[0][cal][:, :K0_RHO])
            rk = icc_kish(Z_cal_k0)
            rh = rho_hartung(Z_cal_k0)

            tau_emp = tau_curve(Craw, cal)
            tau_z_emp = tau_curve(Cz, cal)

            for tag, rho in (("kish", rk), ("hartung", rh)):
                tau_law0 = Z1A * np.sqrt(1 + (KS - 1) * rho)
                c = float(np.median(tau_z_emp / tau_law0))
                tau_law = c * tau_law0
                for j, k in enumerate(KS):
                    rows.append(dict(
                        model=model, split=sp, estimator=tag, k=int(k),
                        rho=rho, c=c,
                        fpr_emp=float((Craw[:, ev, j] >= tau_emp[j]).mean()),
                        fpr_law=float((Cz[:, ev, j] >= tau_law[j]).mean()),
                    ))
        print(f"  xong {model}")

    T = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    T.to_csv(OUT / "rho_compare_16models.csv", index=False)

    u = T.drop_duplicates(["model", "split", "estimator"])
    print("\n--- rho va c, theo estimator (trung vi qua 16 model x 5 split) ---")
    print(u.groupby("estimator")[["rho", "c"]].median().round(4))

    T["err"] = (T.fpr_law - T.fpr_emp).abs()
    print("\n--- sai so |fpr_law - fpr_emp|, trung vi qua tat ca (model,split,k) ---")
    print(T.groupby("estimator")["err"].median().round(5))
    print("\n--- sai so trung binh (mean), nhay cam hon voi cac k le/nho ---")
    print(T.groupby("estimator")["err"].mean().round(5))

    print("\n--- theo tung k (trung vi err qua model x split) ---")
    piv = T.pivot_table(index="k", columns="estimator", values="err", aggfunc="median")
    print(piv.round(5))

    print("\n-> output/rho_compare_16models.csv")


if __name__ == "__main__":
    main()
