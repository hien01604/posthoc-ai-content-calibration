"""Nhanh `mean`: nguong tau(k) theo HAI duong, tren 16 checkpoint.

    python3 paper_level_tau_k.py

VAN DE
------
Voi thong ke `mean`, nguong tau KHONG phai hang so: trung binh cua nhieu doan bi san
phang, nen bai human hiem khi dat diem cao khi k lon -> tau phai GIAM theo k. Mot tau
duy nhat chot o mot k roi dung cho moi k thi hong (xem Sec VI.2 outline). Script nay
dung duong cong tau(k) va do no giu duoc muc tieu FPR toi dau.

HAI DUONG LAY tau(k)
--------------------
  (A) THUC NGHIEM  - dem thuan tuy, khong gia thiet gi:
      moi bai cal rut ngau nhien k doan -> diem gop `mean` -> lay phan vi 99%.
      Chinh xac nhat, nhung chi chay duoc toi k ma corpus con du bai dai.

  (B) DESIGN EFFECT - suy ca duong cong tu MOT tham so rho, dung cong thuc gop dependent
      p-value cua Hartung (1999, Biometrical Journal 41(7):849-855) cho ca phan gop VA
      phan uoc rho (khong con muon ICC cua Kish 1965 nhu ban truoc):
          p_i  = (0,5 + #{diem cal >= s_i}) / (n_cal + 1)      conformal mid-p
          z_i  = Phi^-1(1 - p_i)                               null ~ N(0,1)
          Z_k  = sum(z_i) / sqrt(k)                            neu DOC LAP: Z_k ~ N(0,1)
          rho_hat = 1 - E[ Var_trong_nhom(z_i, ddof=1) ]       uoc luong RIENG cua Hartung
          Var(Z_k) = 1 + (k-1)*rho                             hieu ung cum trong bai (= DEFF)
          tau(k)   = c * z_{1-alpha} * sqrt(1 + (k-1)*rho)
      Da so truc tiep tren du lieu that (paper_level_rho_compare.py, 16 model): rho_hartung
      (0,300) va rho_kish/ICC (0,295) cho fpr_law sai khac fpr_emp <0,01 diem % -- khong
      phan biet duoc ro ret, nhung rho_hartung can he so va c nho hon (1,0014 so 1,0076),
      dung ly thuyet hon (tu-nhat-quan voi chinh cong thuc gop cua Hartung) -> chon dung.
      c   = he so hieu chinh khop tren vung chong lan. Do lech cua cong thuc la ~6% va
            KHONG doi theo k, nen mot he so la du -> cho phep ngoai suy k > KMAX.

  (C) DOI CHUNG SAI: tau hang so chot o k=KMAX roi ap cho moi k -> hong IM LANG o k nho.

BAY DA GAP (giu nguyen, dung sua)
---------------------------------
  * QUAN THE PHAI CO DINH theo k (chi bai >= KMAX doan). Neu moi k lai lay "bai >= k doan"
    thi quan the doi theo k -> lan "bai khac nhau" vao "k khac nhau" (dung loi da ha
    bang FPR-theo-dai-k trong report truoc).
  * mid-p (+0,5) chu khong phai (+1): p = 1 lam Phi^-1(0) = -inf, hong uoc luong rho.
  * rho phai uoc tren nhom BANG NHAU (dung k0 doan/bai) neu dung cong thuc ICC nay.
  * Lap R lan chi khu nhieu RUT MAU CON. Nhieu do "chi co ngan nay bai" phai doi seed
    chia cal/eval (S lan) moi thay -> bao ca hai.

DAU VAO
-------
  artifacts/kaggle_testfp_full/pred_testfp_full/pred_test_fp_full_<artifact>.parquet
      1.200 bai toan human, k that ~22,6. Chi co lop human -> do duoc FPR, khong do TPR.

DAU RA (output/)
----------------
  tau_k_mean_16models.csv   tau(k) va FPR dat duoc cua ca 3 duong, moi (model, split, k)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = Path(__file__).resolve().parent
FP_FULL = BASE / "artifacts" / "kaggle_testfp_full" / "pred_testfp_full"
OUT = BASE / "output"

ALPHA = 0.01          # muc tieu FPR muc tai lieu
KMAX = 20             # k lon nhat trong luoi; cung la nguong loc quan the
R = 200               # so lan rut mau con (khu nhieu rut mau)
S = 5                 # so lan chia cal/eval (khu nhieu chia tap)
K0_RHO = 20           # so doan/bai dung de uoc rho (nhom bang nhau)
SEED = 42
KS = np.arange(1, KMAX + 1)
Z1A = float(norm.isf(ALPHA))


def draw(papers, rng):
    """(R, n, KMAX) diem doan rut ngau nhien khong hoan lai, vector hoa."""
    n = len(papers)
    L = max(len(a) for a in papers)
    pad = np.full((n, L), np.nan)
    for i, a in enumerate(papers):
        pad[i, : len(a)] = a
    invalid = np.isnan(pad)
    out = np.empty((R, n, KMAX))
    for r in range(R):
        key = rng.random((n, L))
        key[invalid] = np.inf                     # o trong luon xep cuoi
        idx = np.argsort(key, axis=1)[:, :KMAX]
        out[r] = np.take_along_axis(pad, idx, axis=1)
    return out


def rho_hartung(Z):
    """Uoc luong rho RIENG cua Hartung (1999): 1 - trung binh phuong sai TRONG moi bai.

    Gia dinh duoi H0, moi z_i ~ N(0,1) doc lap-marginal (phuong sai=1 dung); tuong quan
    lam phuong sai QUAN SAT trong 1 nhom bi "nen" lai duoi 1 -> 1-E[Var trong nhom] > 0.
    Da so voi ICC kieu Kish tren du lieu that: 2 uoc luong lech nhau <2%, khong phan biet
    duoc ro ret ve do khop fpr_emp -- chon Hartung vi he so va c can it hon (tu nhat quan
    hon voi chinh cong thuc gop dung o duoi).
    """
    within_var = Z.var(axis=1, ddof=1)   # (n,) phuong sai TRONG moi bai, ddof=1 = chia (k0-1)
    v = 1.0 - float(within_var.mean())
    return max(0.0, v)


def tau_curve(C, sel):
    """Phan vi (1-ALPHA) tren tap `sel`, trung binh qua R lan rut mau. Tra ve vector theo k."""
    Q = np.quantile(C[:, sel, :], 1 - ALPHA, axis=1, method="higher")   # (R, KMAX)
    return Q.mean(axis=0)


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    files = sorted(FP_FULL.glob("pred_test_fp_full_artifacts_*.parquet"))
    if not files:
        raise SystemExit("Khong thay pred_test_fp_full_*.parquet — chay fetch_kaggle_preds.py truoc.")

    for f in files:
        model = f.stem.replace("pred_test_fp_full_artifacts_", "")
        rng = np.random.default_rng(SEED)
        df = pd.read_parquet(f).dropna(subset=["score"])
        by_paper = {p: g.score.to_numpy() for p, g in df.groupby("paper_id")}
        ids = np.array(sorted(p for p, v in by_paper.items() if len(v) >= KMAX))
        papers = [by_paper[p] for p in ids]

        M = draw(papers, rng)                            # (R, n, KMAX) diem doan tho
        Craw = np.cumsum(M, axis=2) / KS                 # mean cua k doan dau

        for sp in range(S):
            perm = rng.permutation(len(ids))
            cal, ev = perm[: len(ids) // 2], perm[len(ids) // 2:]

            # thang z: be hieu chuan muc doan CHI tu bai cal
            pool = np.sort(np.concatenate([by_paper[p] for p in ids[cal]]))
            nc = len(pool)
            pv = (0.5 + nc - np.searchsorted(pool, M, side="left")) / (nc + 1.0)
            Cz = np.cumsum(norm.isf(pv), axis=2) / np.sqrt(KS)

            rho = rho_hartung(norm.isf(pv[0][cal][:, :K0_RHO]))
            tau_law0 = Z1A * np.sqrt(1 + (KS - 1) * rho)  # cong thuc, chua hieu chinh

            tau_emp = tau_curve(Craw, cal)                # (A) thang diem tho
            tau_z_emp = tau_curve(Cz, cal)                # chi de khop he so c
            c = float(np.median(tau_z_emp / tau_law0))
            tau_law = c * tau_law0                        # (B)
            tau_const = tau_emp[-1]                       # (C) hang so chot o KMAX

            for j, k in enumerate(KS):
                rows.append(dict(
                    model=model, split=sp, k=int(k),
                    n_cal=len(cal), n_eval=len(ev), n_cal_chunk=nc, rho=rho, c=c,
                    tau_emp=tau_emp[j],
                    fpr_emp=float((Craw[:, ev, j] >= tau_emp[j]).mean()),
                    tau_z_emp=tau_z_emp[j], tau_law=tau_law[j],
                    fpr_law=float((Cz[:, ev, j] >= tau_law[j]).mean()),
                    fpr_law_nocorr=float((Cz[:, ev, j] >= tau_law0[j]).mean()),
                    fpr_const=float((Craw[:, ev, j] >= tau_const).mean()),
                ))
        print(f"  xong {model}")

    T = pd.DataFrame(rows)
    T.to_csv(OUT / "tau_k_mean_16models.csv", index=False)

    u = T.drop_duplicates(["model", "split"])
    print(f"\nrho : trung vi {u.rho.median():.3f}  dai [{u.rho.min():.3f}-{u.rho.max():.3f}]")
    print(f"c   : trung vi {u.c.median():.3f}  dai [{u.c.min():.3f}-{u.c.max():.3f}]")
    print(f"\nFPR tren nua eval, trung vi qua {T.model.nunique()} model x {S} split "
          f"(muc tieu {ALPHA:.0%}):\n")
    g = T.groupby("k").agg(tau_emp=("tau_emp", "median"),
                           fpr_emp=("fpr_emp", "median"), sd_emp=("fpr_emp", "std"),
                           fpr_law=("fpr_law", "median"),
                           fpr_law0=("fpr_law_nocorr", "median"),
                           fpr_const=("fpr_const", "median"))
    hdr = ("  k   tau_emp   (A) thuc nghiem   (B) design eff   B chua h/chinh   "
           "(C) tau hang so")
    print(hdr)
    for k, r in g.iterrows():
        print("%3d %9.4f %13.2f%%+-%.2f %14.2f%% %16.2f%% %16.2f%%"
              % (k, r.tau_emp, 100 * r.fpr_emp, 100 * r.sd_emp,
                 100 * r.fpr_law, 100 * r.fpr_law0, 100 * r.fpr_const))
    print("\n-> output/tau_k_mean_16models.csv")


if __name__ == "__main__":
    main()
