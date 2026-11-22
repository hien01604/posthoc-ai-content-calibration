"""F1 -- KHOANG TIN CAY cho pi_hat, co hieu chinh hieu ung thiet ke, va DO PHU that.

    py f1_pihat_ci.py

VAN DE (phan bien M1)
---------------------
Bai ten la "Estimating the AI-Generated Content Fraction" nhung khong bao gio dua ra
mot khoang tin cay nao cho pi_hat. Toan bo may moc can thiet (rho, 1+(k-1)rho) da duoc
dung san trong paper_level_pihat.py (dong 140) nhung chi de dung lam NGUONG QUYET DINH
mot phia, khong bao gio duoc bao cao nhu mot KHOANG va khong bao gio do do phu.

Script nay lam dung mot viec: dung lai chinh cong thuc do, bien no thanh khoang HAI PHIA,
roi do xem no co phu dung 95% tren TAI LIEU THAT hay khong.

    SE(pi_hat) = sqrt( Var_TONG / k * (1 + (k-1)*rho_raw) ) / (mu1 - mu0)
    CI         = pi_hat +- z_{0,975} * SE

BA BIEN THE DUOC SO SANH (cung mot cau chuyen 3 nhanh nhu §V-A)
---------------------------------------------------------------
  (1) naive : bo hieu ung thiet ke  -> SE = sqrt(Var_TONG/k)/(mu1-mu0)
  (2) deff  : co hieu ung thiet ke  -> nhan them sqrt(1+(k-1)*rho_raw)
  (3) emp   : phan vi 2,5/97,5 THUC NGHIEM cua pi_hat do tren nua CAL o dung k do
              (khong doi xung, khong gia thiet Gauss) -- chuan muc tren

DU LIEU
-------
  (A) do phu tai pi = 0, tren TAI LIEU THAT: 576 bai cua test_fp_full co >= 20 doan,
      dung dung quan the co dinh cua §V-A. Day la phep do sach nhat vi tuong quan
      noi tai lieu la THAT, khong phai dung.
  (B) do phu tai pi > 0, tren tai lieu BAN TONG HOP: k doan human lay tu MOT bai that
      (giu tuong quan that o phia human), round(pi*k) doan duoc thay bang diem AI rut
      tu pool test_ood_model. Phia AI van co rho = 0 -> ghi ro la gioi han tren lac quan.

mu0/mu1 doc tu output/pihat_sanity_16models.csv (uoc CHI tu `val`, khong uoc lai).
rho_raw va Var_TONG uoc tren nua CAL cua test_fp_full, tach khoi nua EVAL dung de do phu.

DAU RA (results/)
-----------------
  f1_coverage_null.csv        do phu tai pi=0 theo k, 3 bien the, 16 checkpoint
  f1_halfwidth.csv            do rong khoang (don vi pi) theo k -- con so dung duoc
  f1_coverage_mixed.csv       do phu tai pi thuoc {0; 0,25; 0,5; 0,75; 1} (ban tong hop)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from icc import icc1  # noqa: E402
ART = ROOT / "artifacts" / "kaggle"
FPF = ROOT / "artifacts" / "kaggle_testfp_full" / "pred_testfp_full"
OUT = ROOT / "output"
RES = HERE / "results"

KMAX = 20
KS = np.arange(1, KMAX + 1)
KS_SHOW = [1, 2, 5, 10, 20]
R = 200                 # so lan rut mau con trong tai lieu
S = 5                   # so lan chia cal/eval
K0_RHO = 20             # nhom bang nhau de uoc rho
SEED = 42
CONF = 0.95
ZC = float(norm.isf((1 - CONF) / 2))          # 1,96
PIS = [0.00, 0.25, 0.50, 0.75, 1.00]
NDOC = 500                                     # tai lieu ban tong hop moi (model, pi)


def draw(papers, rng, n_rep=R):
    """(n_rep, n_paper, KMAX) diem doan rut ngau nhien khong hoan lai trong tung bai."""
    n = len(papers)
    L = max(len(a) for a in papers)
    pad = np.full((n, L), np.nan)
    for i, a in enumerate(papers):
        pad[i, : len(a)] = a
    invalid = np.isnan(pad)
    out = np.empty((n_rep, n, KMAX))
    for r in range(n_rep):
        key = rng.random((n, L))
        key[invalid] = np.inf
        idx = np.argsort(key, axis=1)[:, :KMAX]
        out[r] = np.take_along_axis(pad, idx, axis=1)
    return out


def rho_raw_icc(papers_cal, var_total=None):
    """ICC(1) mot chieu tren thang THO -- xem icc.py.

    (var_total giu lai cho tuong thich chu ky ham cu; ICC(1) khong can no.)
    """
    return icc1(papers_cal, K0_RHO)


def mixed_docs(paper_pool, sA, pi, k, rng):
    """(NDOC, k): k doan human tu MOT bai that, round(pi*k) doan thay bang diem AI."""
    pick = rng.integers(0, len(paper_pool), NDOC)
    S_ = np.empty((NDOC, k))
    for r in range(NDOC):
        a = paper_pool[pick[r]]
        S_[r] = rng.choice(a, size=k, replace=False)
    n_ai = int(round(pi * k))
    if n_ai:
        m = np.zeros((NDOC, k), dtype=bool)
        idx = np.argsort(rng.random((NDOC, k)), axis=1)[:, :n_ai]
        np.put_along_axis(m, idx, True, axis=1)
        S_ = np.where(m, sA[rng.integers(0, len(sA), (NDOC, k))], S_)
    return S_


def main():
    RES.mkdir(exist_ok=True)
    anc = pd.read_csv(OUT / "pihat_sanity_16models.csv").set_index("model")

    rows_null, rows_hw, rows_mix = [], [], []
    files = sorted(FPF.glob("pred_test_fp_full_artifacts_*.parquet"))

    for f in files:
        model = f.stem.replace("pred_test_fp_full_artifacts_", "")
        mu0, mu1 = float(anc.loc[model, "mu0"]), float(anc.loc[model, "mu1"])
        den = mu1 - mu0

        rng = np.random.default_rng(SEED)
        df = pd.read_parquet(f).dropna(subset=["score"])
        by_paper = {p: g.score.to_numpy() for p, g in df.groupby("paper_id")}
        ids = np.array(sorted(p for p, v in by_paper.items() if len(v) >= KMAX))
        papers = [by_paper[p] for p in ids]

        M = draw(papers, rng)                              # (R, n, KMAX)
        Craw = np.cumsum(M, axis=2) / KS                   # trung binh k doan dau
        PH = (Craw - mu0) / den                            # pi_hat

        ai = pd.read_parquet(ART / ("artifacts_%s" % model) / "pred_test_ood_model.parquet")
        sA = ai.loc[ai.label == 1, "score"].to_numpy()

        for sp in range(S):
            perm = rng.permutation(len(ids))
            cal, ev = perm[: len(ids) // 2], perm[len(ids) // 2:]

            pool_cal = np.concatenate([papers[i] for i in cal])
            var_tot = float(np.var(pool_cal, ddof=1))
            rho = rho_raw_icc([papers[i] for i in cal], var_tot)

            for j, k in enumerate(KS):
                hw_naive = ZC * np.sqrt(var_tot / k) / den
                hw_deff = ZC * np.sqrt(var_tot / k * (1 + (k - 1) * rho)) / den
                lo_emp, hi_emp = np.quantile(PH[:, cal, j], [(1 - CONF) / 2,
                                                             1 - (1 - CONF) / 2])
                e = PH[:, ev, j]
                rows_null.append(dict(
                    model=model, split=sp, k=int(k), rho_raw=rho, var_total=var_tot,
                    cov_naive=float((np.abs(e) <= hw_naive).mean()),
                    cov_deff=float((np.abs(e) <= hw_deff).mean()),
                    cov_emp=float(((e >= lo_emp) & (e <= hi_emp)).mean()),
                    hw_naive=hw_naive, hw_deff=hw_deff,
                    hw_emp=float((hi_emp - lo_emp) / 2),
                    bias=float(e.mean())))
                rows_hw.append(dict(model=model, split=sp, k=int(k),
                                    hw_naive=hw_naive, hw_deff=hw_deff))

            # --- (B) do phu tai pi > 0, tai lieu ban tong hop ---
            if sp == 0:
                pool_cal_papers = [papers[i] for i in cal]
                eval_papers = [papers[i] for i in ev]
                for k in [5, 10, 20]:
                    hw_deff = ZC * np.sqrt(var_tot / k * (1 + (k - 1) * rho)) / den
                    hw_naive = ZC * np.sqrt(var_tot / k) / den
                    for pi in PIS:
                        if abs(round(pi * k) - pi * k) > 1e-9:
                            continue
                        Sm = mixed_docs(eval_papers, sA, pi, k, rng)
                        ph = (Sm.mean(axis=1) - mu0) / den
                        rows_mix.append(dict(
                            model=model, k=k, true_pi=pi,
                            mean_pi_hat=float(ph.mean()),
                            cov_naive=float((np.abs(ph - pi) <= hw_naive).mean()),
                            cov_deff=float((np.abs(ph - pi) <= hw_deff).mean()),
                            hw_deff=hw_deff))
                del pool_cal_papers
        print("  xong %s" % model)

    N = pd.DataFrame(rows_null)
    N.to_csv(RES / "f1_coverage_null.csv", index=False)
    pd.DataFrame(rows_hw).to_csv(RES / "f1_halfwidth.csv", index=False)
    Mx = pd.DataFrame(rows_mix)
    Mx.to_csv(RES / "f1_coverage_mixed.csv", index=False)

    per_mk = N.groupby(["model", "k"]).mean(numeric_only=True).reset_index()
    g = per_mk.groupby("k").agg(
        rho=("rho_raw", "median"),
        naive=("cov_naive", "median"), naive_lo=("cov_naive", "min"),
        deff=("cov_deff", "median"), deff_lo=("cov_deff", "min"),
        emp=("cov_emp", "median"),
        hw_naive=("hw_naive", "median"), hw_deff=("hw_deff", "median"))

    print("\nrho_raw (ICC thang tho): trung vi %.4f  dai [%.4f-%.4f]"
          % (per_mk.rho_raw.median(), per_mk.rho_raw.min(), per_mk.rho_raw.max()))
    print("\nDO PHU cua khoang %.0f%% cho pi_hat tai pi=0, tren bai THAT" % (100 * CONF))
    print("  k   naive    deff     emp   | nua rong (naive / deff), don vi pi")
    for k in KS_SHOW:
        r = g.loc[k]
        print("%3d  %5.1f%%  %5.1f%%  %5.1f%%  |  %.4f / %.4f"
              % (k, 100 * r.naive, 100 * r.deff, 100 * r.emp, r.hw_naive, r.hw_deff))

    if len(Mx):
        print("\nDO PHU tai pi > 0 (tai lieu ban tong hop, phia AI co rho=0)")
        q = Mx.groupby(["k", "true_pi"]).median(numeric_only=True).reset_index()
        print("  k    pi   naive    deff   E[pi_hat]")
        for _, r in q.iterrows():
            print("%3d  %.2f  %5.1f%%  %5.1f%%   %.4f"
                  % (r.k, r.true_pi, 100 * r.cov_naive, 100 * r.cov_deff, r.mean_pi_hat))

    print("\n-> results/f1_coverage_null.csv, f1_halfwidth.csv, f1_coverage_mixed.csv")


if __name__ == "__main__":
    main()
