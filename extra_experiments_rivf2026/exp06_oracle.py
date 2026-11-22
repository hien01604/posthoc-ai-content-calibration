"""EXP-06 -- Duong co so "oracle": hieu chuan TRUC TIEP tai tung k.

    python3 exp06_oracle.py

VAN DE
------
Table II moi co MOT hang doi chung "chot dung k" tai k~2,38. Can he thong hoa no
tren TOAN BO luoi k da dung o Table III/IV -> bien mot phep so sanh tinh co thanh
mot duong co so oracle ro rang: "neu ban CO du lieu hieu chuan tai dung k trien khai
thi lam duoc toi dau" -- roi cho thay phuong phap de xuat bam sat no MA KHONG can
du lieu do.

BON PHUONG PHAP
---------------
  (A) Nguong tinh chot tai k=20        -- DA CO: tau_k_mean_16models.csv `tau_const`
  (B) Hieu chinh design-effect thang tho -- DA CO: pihat_static_vs_corrected_by_k.csv
  (C) ORACLE: hieu chuan truc tiep tai tung k -- DA CO nhung CHUA TUNG DUOC BAO CAO:
      cot `tau_emp`/`fpr_emp` trong tau_k_mean_16models.csv chinh la phan vi 99% cua
      trung binh k doan tren nua CAL, do tren nua EVAL, tinh RIENG cho tung k.
  (D) z-scale Hartung (de xuat)        -- DA CO: `fpr_law_nocorr`

TY LE BO LOT
------------
`test_fp_full` chi co lop human -> khong do duoc bo lot trong quan the co kiem soat.
Nen ty le bo lot o day do tren TAI LIEU TOAN AI do dai k, ghep tu pool AI cua
`test_ood_model` (cung ky thuat ghep doan cua exp01_calibration.py).
  - (A)(C) so tren thang diem TRUNG BINH THO, dung dung tau da luu.
  - (D) so tren thang z, dung dung cong thuc va be mid-p cua paper_level_tau_k.py.
  - (B) CHI bao cao FPR: no da truot muc tieu FPR toi 4-9 lan, nen ty le bo lot cua no
    khong mang thong tin (nguong long thi bo lot thap la duong nhien).

DAU RA
------
  results/exp06_oracle_comparison.csv   k, method, fpr, miss_rate (+ dai qua ckpt)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent          # extra_experiments_rivf2026/
ROOT = HERE.parent                              # goc repo
ART = ROOT / "artifacts" / "kaggle"
FPF = ROOT / "artifacts" / "kaggle_testfp_full" / "pred_testfp_full"
OUT = ROOT / "output"
RES = HERE / "results"

KS_SHOW = [1, 2, 3, 5, 8, 10, 15, 20]
KMAX, ALPHA, SEED = 20, 0.01, 20260825
NDOC = 2000                     # tai lieu toan AI moi (model, split, k)
Z1A = float(norm.isf(ALPHA))

NAMES = {"static": "(A) Static threshold pinned at k=20",
         "raw": "(B) Raw-scale design-effect adjustment",
         "oracle": "(C) Direct per-k calibration (oracle)",
         "zscale": "(D) z-scale Hartung correction (proposed)"}


def ai_pool(model):
    d = pd.read_parquet(ART / ("artifacts_%s" % model) / "pred_test_ood_model.parquet")
    return d.loc[d.label == 1, "score"].to_numpy()


def fpfull_by_paper(model):
    f = FPF / ("pred_test_fp_full_artifacts_%s.parquet" % model)
    d = pd.read_parquet(f).dropna(subset=["score"])
    return {p: g.score.to_numpy() for p, g in d.groupby("paper_id")}


def main():
    RES.mkdir(exist_ok=True)
    T = pd.read_csv(OUT / "tau_k_mean_16models.csv")
    P = pd.read_csv(OUT / "pihat_static_vs_corrected_by_k.csv")

    rows = []
    for model, g in T.groupby("model"):
        rng = np.random.default_rng(SEED)
        sA = ai_pool(model)
        by_paper = fpfull_by_paper(model)
        ids = np.array(sorted(p for p, v in by_paper.items() if len(v) >= KMAX))

        # tai tao dung cach chia cal/eval cua paper_level_tau_k.py de lay be mid-p
        rr = np.random.default_rng(42)
        rr.random((len(ids), max(len(by_paper[p]) for p in ids)))   # nuot draw() cua M
        for sp, sub in g.groupby("split"):
            sub = sub.set_index("k")
            # tai lieu toan AI: (NDOC, KMAX) diem doan rut khong hoan lai
            idx = np.argsort(rng.random((NDOC, len(sA))), axis=1)[:, :KMAX]
            S = sA[idx]
            cmean = np.cumsum(S, axis=1) / np.arange(1, KMAX + 1)

            # thang z: dung dung cong thuc cua paper_level_tau_k.py
            # be = doan human cua nua CAL trong test_fp_full (xem exp: pool cal)
            rho = float(sub.rho.iloc[0])
            perm = np.random.default_rng(SEED + sp).permutation(len(ids))
            cal = perm[: len(ids) // 2]
            pool = np.sort(np.concatenate([by_paper[p] for p in ids[cal]]))
            nc = len(pool)
            pv = (0.5 + nc - np.searchsorted(pool, S, side="left")) / (nc + 1.0)
            Z = np.cumsum(norm.isf(pv), axis=1) / np.sqrt(np.arange(1, KMAX + 1))

            # tau tinh KHONG duoc luu rieng; theo paper_level_tau_k.py no dung bang
            # tau_emp tai k=KMAX ("tau_const = tau_emp[-1]").
            tau_c = float(sub.loc[KMAX, "tau_emp"])
            for k in KS_SHOW:
                j = k - 1
                tau_e = float(sub.loc[k, "tau_emp"])
                tau_z = Z1A * np.sqrt(1 + (k - 1) * rho)
                rows.append(dict(model=model, split=sp, k=k, method="static",
                                 fpr=float(sub.loc[k, "fpr_const"]),
                                 miss=float((cmean[:, j] < tau_c).mean())))
                rows.append(dict(model=model, split=sp, k=k, method="oracle",
                                 fpr=float(sub.loc[k, "fpr_emp"]),
                                 miss=float((cmean[:, j] < tau_e).mean())))
                rows.append(dict(model=model, split=sp, k=k, method="zscale",
                                 fpr=float(sub.loc[k, "fpr_law_nocorr"]),
                                 miss=float((Z[:, j] < tau_z).mean())))
        print("  xong %s" % model)

    D = pd.DataFrame(rows)
    # (B) chi co FPR, tu pipeline pi_hat (1 gia tri/model/k)
    b = P.rename(columns={"fpr_corrected": "fpr"})[["model", "k", "fpr"]].copy()
    b["method"], b["miss"] = "raw", np.nan

    per_mk = (D.groupby(["method", "model", "k"])
                .agg(fpr=("fpr", "mean"), miss=("miss", "mean")).reset_index())
    per_mk = pd.concat([per_mk, b], ignore_index=True)
    per_mk.to_csv(RES / "exp06_per_checkpoint.csv", index=False)

    agg = (per_mk.groupby(["method", "k"])
                 .agg(fpr=("fpr", "median"), fpr_lo=("fpr", "min"), fpr_hi=("fpr", "max"),
                      miss=("miss", "median"), miss_lo=("miss", "min"),
                      miss_hi=("miss", "max"), n_ckpt=("model", "nunique"))
                 .reset_index())
    agg["method_name"] = agg.method.map(NAMES)
    agg = agg[agg.k.isin(KS_SHOW)].sort_values(
        ["method", "k"], key=lambda c: c.map({"static": 0, "raw": 1, "oracle": 2,
                                              "zscale": 3}) if c.name == "method" else c)
    agg.to_csv(RES / "exp06_oracle_comparison.csv", index=False)

    print("\nFPR (%, muc tieu 1,00) -- trung vi qua 16 checkpoint")
    print("  k  " + "".join("%22s" % NAMES[m][:21] for m in ("static", "raw", "oracle", "zscale")))
    for k in KS_SHOW:
        line = "%3d " % k
        for m in ("static", "raw", "oracle", "zscale"):
            r = agg[(agg.method == m) & (agg.k == k)]
            line += "%22s" % ("%.2f" % (r.fpr.iloc[0] * 100) if len(r) else "-")
        print(line)

    print("\nTy le BO LOT (%) tren tai lieu toan AI do dai k")
    print("  k  " + "".join("%22s" % NAMES[m][:21] for m in ("static", "oracle", "zscale")))
    for k in KS_SHOW:
        line = "%3d " % k
        for m in ("static", "oracle", "zscale"):
            r = agg[(agg.method == m) & (agg.k == k)]
            line += "%22s" % ("%.2f" % (r.miss.iloc[0] * 100) if len(r) else "-")
        print(line)

    z = agg[agg.method == "zscale"].set_index("k")
    o = agg[agg.method == "oracle"].set_index("k")
    d = np.abs(z.fpr - o.fpr) * 100
    print("\n|FPR(de xuat) - FPR(oracle)|: trung binh %.2f diem %%, toi da %.2f diem %%"
          % (d.mean(), d.max()))
    print("-> results/exp06_oracle_comparison.csv")
    print("-> results/exp06_per_checkpoint.csv")


if __name__ == "__main__":
    main()
