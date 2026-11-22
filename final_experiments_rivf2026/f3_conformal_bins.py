"""F3 -- BASELINE TU CONG TRINH TRUOC: nguong conformal theo BIN DO DAI (kieu MCP).

    py f3_conformal_bins.py

VAN DE (phan bien M8)
---------------------
Moi baseline trong bai deu la baseline nha trong (nguong tinh, thang tho Gauss, thang
tho Emp, oracle). Khong co phuong phap nao tu cong trinh truoc duoc CHAY. Bang
`tab:related` con ghi "Doc. FPR = yes" cho ca MCP lan bai nay, trong khi MCP cho mot
BAO DAM conformal huu han mau con bai nay chi cho mot TY LE THUC NGHIEM BIEN.

Script nay cai dat dung y tuong cua MCP tren chinh diem so cua bai -- chia bai theo
bin do dai, chot nguong conformal rieng cho tung bin -- roi so voi quy tac de xuat.

DIEM QUAN TRONG NHAT: DO PHAN GIAI DUOI
---------------------------------------
Nguong conformal muc alpha can it nhat ceil(1/alpha) - 1 = 99 mau hieu chuan trong
MOI bin thi phan vi 1-alpha moi TON TAI. Voi ~600 bai hieu chuan chia thanh B bin:
  B = 5  -> ~120 bai/bin  -> vua du, nhung phan vi nam o dung mau lon nhat
  B = 10 -> ~60 bai/bin   -> KHONG the chot nguong 1% mot cach hop le
Day chinh la cai gia cua viec binning ma quy tac de xuat khong phai tra: no dung
dung k cua tung tai lieu, khong can bin nao ca.

THIET LAP -- KHAC §V-A: DUNG DO DAI TU NHIEN
--------------------------------------------
§V-A ep moi tai lieu ve cung mot k bang lay mau con. O day dung do dai THAT cua
1.200 bai test_fp_full (trung vi 19, dai 1..262) -- tuc dung boi canh trien khai.

BA PHUONG PHAP
--------------
  (A) conformal-bin : nguong conformal cua trung binh tai lieu, rieng tung bin do dai
  (B) zscale        : Z_k > z_{1-a} * sqrt(1 + (k-1)*rho), dung k rieng cua tung bai
  (C) static        : mot nguong duy nhat tren trung binh tai lieu, khong dem xia k

DAU RA (results/)
-----------------
  f3_conformal_overall.csv   FPR tong the cua 3 phuong phap, theo so bin
  f3_conformal_by_bin.csv    FPR theo tung bin + so bai hieu chuan moi bin
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import paper_level_tau_k as TK   # noqa: E402

RES = HERE / "results"
ALPHA = TK.ALPHA
Z1A = float(norm.isf(ALPHA))
S = 5                       # so lan chia cal/eval
K0_RHO = 20
NBINS = [5, 10]             # so bin do dai (chia deu so bai)
SEED = 20260829
MIN_CONF = int(np.ceil(1 / ALPHA)) - 1     # 99 bai/bin thi phan vi 1% moi ton tai


def conformal_tau(x, alpha):
    """Nguong conformal chia doi: phan vi ceil((n+1)(1-alpha))/n cua mau hieu chuan.

    Tra ve (tau, hop_le). hop_le=False khi n < ceil(1/alpha)-1: luc do rank can lay
    vuot qua mau lon nhat -> khong co nguong huu han nao dat duoc bao dam.
    """
    n = len(x)
    r = int(np.ceil((n + 1) * (1 - alpha)))
    if r > n:
        return np.inf, False                 # khong the -> khong bao gio gan co
    return float(np.sort(x)[r - 1]), True


def main():
    RES.mkdir(exist_ok=True)
    files = sorted(TK.FP_FULL.glob("pred_test_fp_full_artifacts_*.parquet"))
    rows_ov, rows_bin = [], []

    for f in files:
        model = f.stem.replace("pred_test_fp_full_artifacts_", "")
        rng = np.random.default_rng(SEED)
        d = pd.read_parquet(f).dropna(subset=["score"])
        by_paper = {p: g.score.to_numpy() for p, g in d.groupby("paper_id")}
        ids = np.array(sorted(by_paper))                    # TAT CA 1.200 bai
        kk = np.array([len(by_paper[p]) for p in ids])
        mean_raw = np.array([by_paper[p].mean() for p in ids])

        for sp in range(S):
            perm = rng.permutation(len(ids))
            cal, ev = perm[: len(ids) // 2], perm[len(ids) // 2:]

            # --- thang z: pool mid-p tu doan human cua bai CAL ---
            pool = np.sort(np.concatenate([by_paper[ids[i]] for i in cal]))
            nc = len(pool)

            def zk_of(i):
                s = by_paper[ids[i]]
                pv = (0.5 + nc - np.searchsorted(pool, s, side="left")) / (nc + 1.0)
                return float(norm.isf(pv).sum() / np.sqrt(len(s)))

            Zev = np.array([zk_of(i) for i in ev])

            # rho tren thang z, uoc tu bai CAL co >= K0_RHO doan (nhom bang nhau)
            long_cal = [i for i in cal if len(by_paper[ids[i]]) >= K0_RHO]
            Zmat = np.empty((len(long_cal), K0_RHO))
            for r_, i in enumerate(long_cal):
                s = by_paper[ids[i]][:K0_RHO]
                pv = (0.5 + nc - np.searchsorted(pool, s, side="left")) / (nc + 1.0)
                Zmat[r_] = norm.isf(pv)
            rho = TK.rho_hartung(Zmat)

            tau_z = Z1A * np.sqrt(1 + (kk[ev] - 1) * rho)
            fpr_z = float((Zev >= tau_z).mean())

            # --- (C) static: mot nguong duy nhat tren trung binh tho ---
            tau_s, _ = conformal_tau(mean_raw[cal], ALPHA)
            fpr_s = float((mean_raw[ev] >= tau_s).mean())

            # --- (A) conformal theo bin do dai (chia deu so bai theo phan vi cua k) ---
            for nb in NBINS:
                edges = np.quantile(kk[cal], np.linspace(0, 1, nb + 1))
                edges[0], edges[-1] = -np.inf, np.inf
                bc = np.digitize(kk[cal], edges[1:-1])
                be = np.digitize(kk[ev], edges[1:-1])
                flag = np.zeros(len(ev), dtype=bool)
                n_infeasible = 0
                for b in range(nb):
                    xc = mean_raw[cal][bc == b]
                    m = be == b
                    if len(xc) == 0 or m.sum() == 0:
                        continue
                    t, ok = conformal_tau(xc, ALPHA)
                    if not ok:
                        n_infeasible += 1
                    flag[m] = mean_raw[ev][m] >= t
                    rows_bin.append(dict(
                        model=model, split=sp, n_bins=nb, bin=b,
                        k_lo=float(kk[cal][bc == b].min()) if len(xc) else np.nan,
                        k_hi=float(kk[cal][bc == b].max()) if len(xc) else np.nan,
                        n_cal=len(xc), n_eval=int(m.sum()), feasible=ok,
                        tau=t if np.isfinite(t) else np.nan,
                        fpr_conformal=float((mean_raw[ev][m] >= t).mean()),
                        fpr_zscale=float((Zev[m] >= tau_z[m]).mean())))
                rows_ov.append(dict(
                    model=model, split=sp, n_bins=nb, rho=rho,
                    n_cal=len(cal), n_eval=len(ev),
                    bins_infeasible=n_infeasible,
                    fpr_conformal=float(flag.mean()),
                    fpr_zscale=fpr_z, fpr_static=fpr_s))
        print("  xong %s" % model)

    O = pd.DataFrame(rows_ov)
    Bn = pd.DataFrame(rows_bin)
    O.to_csv(RES / "f3_conformal_overall.csv", index=False)
    Bn.to_csv(RES / "f3_conformal_by_bin.csv", index=False)

    print("\nDO DAI TU NHIEN cua test_fp_full: %d bai, k trung vi %d, dai %d-%d"
          % (len(ids), int(np.median(kk)), kk.min(), kk.max()))
    print("Can >= %d bai hieu chuan/bin thi nguong conformal 1%% moi TON TAI.\n"
          % MIN_CONF)

    per = O.groupby(["model", "n_bins"]).mean(numeric_only=True).reset_index()
    print("FPR (%) o do dai TU NHIEN -- trung vi qua 16 checkpoint, [min,max]")
    print("  so bin   conformal-bin        z-scale (de xuat)     static")
    for nb in NBINS:
        q = per[per.n_bins == nb]
        print("  %5d   %5.2f [%.2f,%.2f]   %5.2f [%.2f,%.2f]   %5.2f [%.2f,%.2f]"
              % (nb,
                 100 * q.fpr_conformal.median(), 100 * q.fpr_conformal.min(),
                 100 * q.fpr_conformal.max(),
                 100 * q.fpr_zscale.median(), 100 * q.fpr_zscale.min(),
                 100 * q.fpr_zscale.max(),
                 100 * q.fpr_static.median(), 100 * q.fpr_static.min(),
                 100 * q.fpr_static.max()))
        print("           bin khong du mau de chot nguong 1%%: %.1f / %d"
              % (q.bins_infeasible.mean(), nb))

    print("\nFPR theo tung bin (nb=5), trung vi qua checkpoint x split")
    g = (Bn[Bn.n_bins == 5].groupby("bin")
         .agg(k_lo=("k_lo", "median"), k_hi=("k_hi", "median"),
              n_cal=("n_cal", "median"),
              conf=("fpr_conformal", "median"), z=("fpr_zscale", "median")))
    print("  bin   k range      n_cal   conformal   z-scale")
    for b, r in g.iterrows():
        print("  %3d   %4d-%-6d %6d   %6.2f%%    %6.2f%%"
              % (b, r.k_lo, r.k_hi, r.n_cal, 100 * r.conf, 100 * r.z))
    print("\n-> results/f3_conformal_overall.csv, f3_conformal_by_bin.csv")


if __name__ == "__main__":
    main()
