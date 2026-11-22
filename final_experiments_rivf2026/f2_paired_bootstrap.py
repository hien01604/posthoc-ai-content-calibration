"""F2 -- BOOTSTRAP GHEP CAP tren HIEU SO FPR, thay vi so hai khoang tin cay bien.

    py f2_paired_bootstrap.py

VAN DE (phan bien M3)
---------------------
Bai ket luan thang z "on dinh nhat" vi dai 0,76-0,97 hep hon khoang ba lan dai
0,48-1,12 cua phan vi thuc nghiem. Nhung REV-1 chi bao KHOANG TIN CAY BIEN cua tung
phuong phap, va chung chong nhau o 7/8 gia tri k. So hai khoang bien chong nhau roi
ket luan "khac nhau" la sai; phai bootstrap chinh HIEU SO.

Diem mau chot: hai phuong phap cham CUNG mot tap bai, nen nhieu o cap bai la nhieu
CHUNG va bi triet tieu trong hieu so. Khoang tin cay cua hieu so vi vay HEP HON
nhieu so voi khoang cua tung phuong phap -- day chinh la ly do phai lam ghep cap.

DO THEM MOT DAI LUONG VO HUONG
------------------------------
Ca bang III chi la 8 con so cho moi phuong phap; can mot dai luong duy nhat de so:

    maxdev = max_k | FPR(k) - 1% |        (do lech xau nhat tren luoi k)
    rmsdev = sqrt( mean_k (FPR(k) - 1%)^2 )

roi bootstrap hieu so cua chinh dai luong do. Neu khoang tin cay cua hieu so khong
chua 0 thi moi duoc noi mot phuong phap on dinh hon phuong phap kia.

May tinh dung y het rev12_bootstrap_and_baseline.py (cung seed, cung cach chia,
cung nguong) -- chi khac o cho luu lai phan phoi HIEU SO.

DAU RA (results/)
-----------------
  f2_paired_diff_by_k.csv    hieu so FPR theo tung k, kem CI 95% va p-value bootstrap
  f2_scalar_stability.csv    maxdev/rmsdev cua tung phuong phap + hieu so co CI
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
NBOOT = 2000
SEED_BOOT = 20260826
Z1A = float(norm.isf(TK.ALPHA))
TARGET = 100 * TK.ALPHA          # 1,0 (don vi %)

METHODS = ["static", "oracle", "zscale", "emp_deff"]
NAMES = {"static": "Static threshold (pinned at k=20)",
         "oracle": "Per-k oracle calibration",
         "zscale": "z-scale Hartung (proposed)",
         "emp_deff": "Raw scale, empirical quantile x design effect"}
PAIRS = [("zscale", "emp_deff"), ("zscale", "oracle"), ("zscale", "static")]


def rho_raw_icc(cal_papers):
    """ICC(1) mot chieu tren thang THO -- xem icc.py."""
    return icc1_and_var(cal_papers, TK.K0_RHO)


def main():
    RES.mkdir(exist_ok=True)
    files = sorted(TK.FP_FULL.glob("pred_test_fp_full_artifacts_*.parquet"))
    KS = np.arange(1, TK.KMAX + 1)
    models = [f.stem.replace("pred_test_fp_full_artifacts_", "") for f in files]
    nM, nK = len(models), len(KS_SHOW)

    P = None      # (n_meth, n_model, S, n_k, n_ev) ty le gan co cua TUNG bai eval

    for mi, f in enumerate(files):
        rng = np.random.default_rng(TK.SEED)
        d = pd.read_parquet(f).dropna(subset=["score"])
        by_paper = {p: g.score.to_numpy() for p, g in d.groupby("paper_id")}
        ids = np.array(sorted(p for p, v in by_paper.items() if len(v) >= TK.KMAX))
        papers = [by_paper[p] for p in ids]

        M = TK.draw(papers, rng)
        Craw = np.cumsum(M, axis=2) / KS

        for sp in range(TK.S):
            perm = rng.permutation(len(ids))
            cal, ev = perm[: len(ids) // 2], perm[len(ids) // 2:]
            if P is None:
                P = np.empty((len(METHODS), nM, TK.S, nK, len(ev)))

            pool = np.sort(np.concatenate([by_paper[p] for p in ids[cal]]))
            nc = len(pool)
            pv = (0.5 + nc - np.searchsorted(pool, M, side="left")) / (nc + 1.0)
            Cz = np.cumsum(norm.isf(pv), axis=2) / np.sqrt(KS)
            rho_z = TK.rho_hartung(norm.isf(pv[0][cal][:, : TK.K0_RHO]))

            Q = np.quantile(Craw[:, cal, :], 1 - TK.ALPHA, axis=1,
                            method="higher").mean(axis=0)
            tau = {"oracle": Q, "static": np.full(TK.KMAX, Q[-1]),
                   "zscale": Z1A * np.sqrt(1 + (KS - 1) * rho_z)}
            cal_papers = [by_paper[p] for p in ids[cal]]
            rho_r, _ = rho_raw_icc(cal_papers)
            mu0 = np.concatenate(cal_papers).mean()
            tau["emp_deff"] = mu0 + (Q[0] - mu0) * np.sqrt((1 + (KS - 1) * rho_r) / KS)

            for ti, meth in enumerate(METHODS):
                src = Cz if meth == "zscale" else Craw
                fl = (src[:, ev, :] >= tau[meth][None, None, :]).mean(axis=0)
                P[ti, mi, sp] = fl[:, [k - 1 for k in KS_SHOW]].T
        print("  xong %s" % models[mi])

    n_ev = P.shape[-1]

    def summarise(A):
        """A: (n_meth, n_model, S, n_k) -> (n_meth, n_k) trung vi qua model, %."""
        return np.median(A.mean(axis=2), axis=1) * 100

    point = summarise(P.mean(axis=-1))                       # (n_meth, n_k)

    rng = np.random.default_rng(SEED_BOOT)
    B = np.empty((NBOOT, len(METHODS), nK))
    for b in range(NBOOT):
        acc = np.empty((len(METHODS), nM, TK.S, nK))
        for sp in range(TK.S):
            idx = rng.integers(0, n_ev, n_ev)                # CUNG bo bai cho moi ph.phap
            acc[:, :, sp, :] = P[:, :, sp, :, :][..., idx].mean(axis=-1)
        B[b] = summarise(acc)

    # ------------------------------------------------ hieu so theo tung k
    rows = []
    for a, c in PAIRS:
        ia, ic = METHODS.index(a), METHODS.index(c)
        for j, k in enumerate(KS_SHOW):
            d = B[:, ia, j] - B[:, ic, j]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            rows.append(dict(k=k, method_a=a, method_b=c,
                             fpr_a=point[ia, j], fpr_b=point[ic, j],
                             diff=point[ia, j] - point[ic, j],
                             ci_lo=lo, ci_hi=hi, p_boot=max(p, 1.0 / NBOOT),
                             excludes_zero=bool(lo > 0 or hi < 0)))
    D = pd.DataFrame(rows)
    D.to_csv(RES / "f2_paired_diff_by_k.csv", index=False)

    # ------------------------------------------------ dai luong vo huong
    def maxdev(X):   return np.abs(X - TARGET).max(axis=-1)
    def rmsdev(X):   return np.sqrt(((X - TARGET) ** 2).mean(axis=-1))

    srow = []
    for ti, meth in enumerate(METHODS):
        for nm, fn in (("maxdev", maxdev), ("rmsdev", rmsdev)):
            bs = fn(B[:, ti, :])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            srow.append(dict(method=meth, method_name=NAMES[meth], stat=nm,
                             value=float(fn(point[ti])), ci_lo=lo, ci_hi=hi))
    for a, c in PAIRS:
        ia, ic = METHODS.index(a), METHODS.index(c)
        for nm, fn in (("maxdev", maxdev), ("rmsdev", rmsdev)):
            d = fn(B[:, ia, :]) - fn(B[:, ic, :])
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            srow.append(dict(method="%s - %s" % (a, c), method_name="hieu so", stat=nm,
                             value=float(fn(point[ia]) - fn(point[ic])),
                             ci_lo=lo, ci_hi=hi, p_boot=max(p, 1.0 / NBOOT),
                             excludes_zero=bool(lo > 0 or hi < 0)))
    Sc = pd.DataFrame(srow)
    Sc.to_csv(RES / "f2_scalar_stability.csv", index=False)

    # ------------------------------------------------ in ket qua
    print("\nHIEU SO FPR (diem %%), bootstrap GHEP CAP theo bai (%d bai, B=%d)"
          % (n_ev, NBOOT))
    for a, c in PAIRS:
        s = D[(D.method_a == a) & (D.method_b == c)]
        print("\n  %s  -  %s" % (NAMES[a], NAMES[c]))
        print("    k    A      B     hieu so   CI 95%% cua hieu so     p")
        for _, r in s.iterrows():
            print("  %3d  %5.2f  %5.2f   %+6.2f    [%+6.2f, %+6.2f]  %s %.3f"
                  % (r.k, r.fpr_a, r.fpr_b, r["diff"], r.ci_lo, r.ci_hi,
                     "*" if r.excludes_zero else " ", r.p_boot))

    print("\nDO ON DINH TREN CA LUOI k (muc tieu %.1f%%)" % TARGET)
    print("  phuong phap                                    maxdev            rmsdev")
    for meth in METHODS:
        a = Sc[(Sc.method == meth) & (Sc.stat == "maxdev")].iloc[0]
        b = Sc[(Sc.method == meth) & (Sc.stat == "rmsdev")].iloc[0]
        print("  %-42s %.2f [%.2f,%.2f]  %.2f [%.2f,%.2f]"
              % (NAMES[meth], a.value, a.ci_lo, a.ci_hi, b.value, b.ci_lo, b.ci_hi))
    print("\n  hieu so (am = phuong phap dau on dinh hon):")
    for a, c in PAIRS:
        for nm in ("maxdev", "rmsdev"):
            r = Sc[(Sc.method == "%s - %s" % (a, c)) & (Sc.stat == nm)].iloc[0]
            print("    %-22s %-7s %+.2f  [%+.2f, %+.2f]  %s p=%.3f"
                  % ("%s - %s" % (a, c), nm, r.value, r.ci_lo, r.ci_hi,
                     "*" if r.excludes_zero else " ", r.p_boot))
    print("\n-> results/f2_paired_diff_by_k.csv, f2_scalar_stability.csv")


if __name__ == "__main__":
    main()
