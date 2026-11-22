"""EXP-04 -- FPR nhay den dau khi uoc SAI rho.

    python3 exp04_rho_sensitivity.py

VAN DE
------
Quy tac quyet dinh Z_k > z_{1-a} * sqrt(1 + (k-1)*rho) chi co MOT tham so phai uoc:
rho. Cau hoi hien nhien cua reviewer: "neu uoc lech thi hong bao nhieu?" Neu FPR sup
do khi rho lech 10-20% thi phuong phap khong dung duoc thuc te.

CACH LAM
--------
Thuan hau ky: KHONG chay lai detector. Dung lai dung may cua paper_level_tau_k.py
(import truc tiep `draw`, cac hang so R/S/KMAX/SEED) de tai tao y het population co
kiem soat va cach chia cal/eval, roi chi doi rho -> {0,8; 0,9; 1,0; 1,1; 1,2} x rho_uoc.

KIEM TRA TU DONG
----------------
Cot scale=1,0 phai TRUNG voi `fpr_law_nocorr` da cong bo (Table IV). Script tu doi
chieu va bao neu lech -- neu lech tuc la da tai tao sai pipeline, khong duoc tin so.

DAU RA
------
  results/exp04_rho_sensitivity.csv   rho_scale, k, fpr (+ dai qua checkpoint)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent          # extra_experiments_rivf2026/
ROOT = HERE.parent                              # goc repo
sys.path.insert(0, str(ROOT))                   # de import duoc paper_level_tau_k.py

import paper_level_tau_k as TK                  # noqa: E402 -- phai sau sys.path
from exp01_calibration import paired_pools, mask_with_exactly  # noqa: E402

# TK.FP_FULL tu tinh theo __file__ CUA NO (goc repo) nen van tro dung artifacts/
ART = ROOT / "artifacts" / "kaggle"
OUT = ROOT / "output"
RES = HERE / "results"

SCALES = [0.8, 0.9, 1.0, 1.1, 1.2]
KS_SHOW = [1, 5, 10, 20]
Z1A = float(norm.isf(TK.ALPHA))

NDOC_AI = 2000        # tai lieu moi (model, split) de do ty le bo lot
SEED_AI = 20260825    # RNG RIENG -- tuyet doi khong dung `rng` cua vong tai tao

# Bo lot tren tai lieu TOAN AI bao hoa ve 0 voi k>=10, nen cot do gan nhu khong nhuc
# nhich va khong tra loi duoc cau hoi "nguong chat hon thi phai tra gia gi". Do them o
# pi=0,5 -- vung ma quyet dinh THUC SU can (xem duong ty le gan co cua EXP-02).
KS_HALF = [10, 20]    # chi k chia het cho 2 -> pi=0,5 dat CHINH XAC


def ai_pool(model):
    """Diem doan AI that tu test_ood_model (co nhan AI thuc)."""
    d = pd.read_parquet(ART / ("artifacts_%s" % model) / "pred_test_ood_model.parquet")
    return d.loc[d.label == 1, "score"].to_numpy()


def main():
    RES.mkdir(exist_ok=True)
    ref = pd.read_csv(OUT / "tau_k_mean_16models.csv")
    files = sorted(TK.FP_FULL.glob("pred_test_fp_full_artifacts_*.parquet"))
    rows = []

    for f in files:
        model = f.stem.replace("pred_test_fp_full_artifacts_", "")
        rng = np.random.default_rng(TK.SEED)          # y het paper_level_tau_k.py
        d = pd.read_parquet(f).dropna(subset=["score"])
        by_paper = {p: g.score.to_numpy() for p, g in d.groupby("paper_id")}
        ids = np.array(sorted(p for p, v in by_paper.items() if len(v) >= TK.KMAX))
        papers = [by_paper[p] for p in ids]

        M = TK.draw(papers, rng)                      # (R, n, KMAX)
        sA = ai_pool(model)
        sHp, sAp = paired_pools(model)                # 1.514 cap human/AI khop nhau
        for sp in range(TK.S):
            perm = rng.permutation(len(ids))
            cal, ev = perm[: len(ids) // 2], perm[len(ids) // 2:]
            pool = np.sort(np.concatenate([by_paper[p] for p in ids[cal]]))
            nc = len(pool)
            pv = (0.5 + nc - np.searchsorted(pool, M, side="left")) / (nc + 1.0)
            Cz = np.cumsum(norm.isf(pv), axis=2) / np.sqrt(TK.KS)
            rho = TK.rho_hartung(norm.isf(pv[0][cal][:, : TK.K0_RHO]))

            # --- tai lieu TOAN AI de do bo lot; RNG rieng, khong dung `rng` ---
            rng_ai = np.random.default_rng(SEED_AI + sp)
            iA = np.argsort(rng_ai.random((NDOC_AI, len(sA))), axis=1)[:, : TK.KMAX]
            pvA = (0.5 + nc - np.searchsorted(pool, sA[iA], side="left")) / (nc + 1.0)
            ZA = np.cumsum(norm.isf(pvA), axis=1) / np.sqrt(TK.KS)

            # --- tai lieu pi=0,5 (ghep cap, dung may cua EXP-01) ---
            Zhalf = {}
            for k in KS_HALF:
                slots = np.argsort(rng_ai.random((NDOC_AI, len(sHp))), axis=1)[:, :k]
                isai = mask_with_exactly(NDOC_AI, k, k // 2, rng_ai)
                S = np.where(isai, sAp[slots], sHp[slots])
                pvh = (0.5 + nc - np.searchsorted(pool, S, side="left")) / (nc + 1.0)
                Zhalf[k] = norm.isf(pvh).sum(axis=1) / np.sqrt(k)

            for sc in SCALES:
                tau = Z1A * np.sqrt(1 + (TK.KS - 1) * (sc * rho))
                for k in KS_SHOW:
                    j = k - 1
                    rows.append(dict(
                        model=model, split=sp, rho_scale=sc, k=k,
                        rho=rho, rho_used=sc * rho,
                        fpr=float((Cz[:, ev, j] >= tau[j]).mean()),
                        miss=float((ZA[:, j] < tau[j]).mean()),
                        miss_half=(float((Zhalf[k] < tau[j]).mean())
                                   if k in KS_HALF else np.nan)))
        print("  xong %s" % model)

    D = pd.DataFrame(rows)
    per_mk = (D.groupby(["rho_scale", "model", "k"])
                .agg(fpr=("fpr", "mean"), miss=("miss", "mean"),
                     miss_half=("miss_half", "mean"),
                     rho_used=("rho_used", "mean")).reset_index())
    agg = (per_mk.groupby(["rho_scale", "k"])
                 .agg(fpr=("fpr", "median"), fpr_lo=("fpr", "min"),
                      fpr_hi=("fpr", "max"),
                      miss=("miss", "median"), miss_lo=("miss", "min"),
                      miss_hi=("miss", "max"),
                      miss_half=("miss_half", "median"),
                      rho_used=("rho_used", "median"))
                 .reset_index())
    agg.to_csv(RES / "exp04_rho_sensitivity.csv", index=False)
    per_mk.to_csv(RES / "exp04_per_checkpoint.csv", index=False)

    # --- kiem tra tai tao: scale=1,0 phai trung fpr_law_nocorr da cong bo ---
    pub = (ref.groupby(["model", "k"]).agg(fpr=("fpr_law_nocorr", "mean")).reset_index())
    pub = pub[pub.k.isin(KS_SHOW)].groupby("k").fpr.median()
    mine = agg[agg.rho_scale == 1.0].set_index("k").fpr
    dev = (mine - pub).abs().max()
    print("\nKiem tra tai tao (scale=1,0 vs Table IV da cong bo): lech toi da %.5f diem"
          % (dev * 100))
    if dev > 5e-4:
        print("  <<< CANH BAO: lech qua lon, pipeline tai tao KHONG khop -- dung tin so")
    else:
        print("  OK -- tai tao khop, ket qua ben duoi dang tin")

    print("\nFPR (%, muc tieu 1,00) -- trung vi qua 16 checkpoint")
    print("  k   " + "".join("%12s" % ("rho x%.1f" % s) for s in SCALES))
    for k in KS_SHOW:
        print("%3d   " % k + "".join(
            "%12.2f" % (100 * agg[(agg.rho_scale == s) & (agg.k == k)].fpr.iloc[0])
            for s in SCALES))
    print("\nTy le BO LOT (%) tren tai lieu TOAN AI -- nua con lai cua cau chuyen")
    print("  k   " + "".join("%12s" % ("rho x%.1f" % s) for s in SCALES))
    for k in KS_SHOW:
        print("%3d   " % k + "".join(
            "%12.2f" % (100 * agg[(agg.rho_scale == s) & (agg.k == k)].miss.iloc[0])
            for s in SCALES))

    print("\nTy le BO LOT (%) tren tai lieu pi=0,5 -- VUNG QUYET DINH THUC SU CAN")
    print("  k   " + "".join("%12s" % ("rho x%.1f" % s) for s in SCALES))
    for k in KS_HALF:
        print("%3d   " % k + "".join(
            "%12.2f" % (100 * agg[(agg.rho_scale == s) & (agg.k == k)].miss_half.iloc[0])
            for s in SCALES))

    w = agg[agg.k > 1]
    print("\nVoi k>=5, FPR nam trong dai %.2f%% - %.2f%% qua toan bo +-20%% nhieu loan rho"
          % (100 * w.fpr.min(), 100 * w.fpr.max()))
    print("             bo lot nam trong dai %.2f%% - %.2f%%"
          % (100 * w.miss.min(), 100 * w.miss.max()))
    # danh doi that su: chenh giua hai dau cua dai rho
    lo = agg[(agg.rho_scale == 0.8)].set_index("k")
    hi = agg[(agg.rho_scale == 1.2)].set_index("k")
    print("\nDanh doi khi di tu rho x0,8 -> x1,2 (nguong chat dan):")
    print("  k    dFPR (diem %)   dMISS toan-AI   dMISS pi=0,5")
    for k in KS_SHOW:
        dh = (100 * (hi.loc[k, "miss_half"] - lo.loc[k, "miss_half"])
              if k in KS_HALF else float("nan"))
        print("%3d   %10.2f      %10.2f      %10s"
              % (k, 100 * (hi.loc[k, "fpr"] - lo.loc[k, "fpr"]),
                 100 * (hi.loc[k, "miss"] - lo.loc[k, "miss"]),
                 "-" if k not in KS_HALF else "%.2f" % dh))
    print("rho uoc (trung vi) = %.3f -> quet tu %.3f den %.3f"
          % (agg[agg.rho_scale == 1.0].rho_used.median(),
             agg[agg.rho_scale == 0.8].rho_used.median(),
             agg[agg.rho_scale == 1.2].rho_used.median()))
    print("-> results/exp04_rho_sensitivity.csv")


if __name__ == "__main__":
    main()
