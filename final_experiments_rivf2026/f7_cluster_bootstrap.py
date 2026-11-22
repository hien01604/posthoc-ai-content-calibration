"""F7 -- BOOTSTRAP THEO CUM `paper_id`, sua loi don vi resample cua f2.

    py f7_cluster_bootstrap.py

LOI DANG SUA (rui ro #1 trong docs/PAPER_CODE_AUDIT_2026-08-28.md)
------------------------------------------------------------------
f2_paper_bootstrap.py rut mot bo chi so bootstrap DOC LAP cho TUNG split:

    for sp in range(TK.S):
        idx = rng.integers(0, n_ev, n_ev)      # <-- doc lap moi split

Nam evaluation half deu rut tu CUNG 576 paper va chong lan nhau. Rut doc lap coi 5
split nhu 5 replicate doc lap, khong bao toan viec MOT paper xuat hien o nhieu split
-> CI hep hon thuc te, va buoc lay trung vi qua model con nen them.

CACH SUA
--------
Mot vector trong so bootstrap DUY NHAT tren 576 paper_id, dung CHUNG cho moi split,
checkpoint va phuong phap:

    cnt = rng.multinomial(n_paper, [1/n_paper]*n_paper)     # so lan moi paper duoc rut
    FPR(split) = sum_p cnt[p] * flag[p] / sum_p cnt[p]      # chi tren p thuoc eval(split)

Paper nao duoc rut nhieu lan thi duoc dem nhieu lan o MOI split ma no xuat hien
-> phu thuoc giua cac split duoc bao toan.

Moi thu khac (seed, cach chia cal/eval, nguong, thu tu model) giu y het f2 de so sanh
duoc truc tiep: chi doi DUY NHAT don vi resample.

DAU RA (results/)
-----------------
  f7_cluster_scalar.csv    maxdev/rmsdev + hieu so, CI cum, doi chieu voi f2
  f7_cluster_diff_by_k.csv hieu so FPR theo tung k voi CI cum
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
TARGET = 100 * TK.ALPHA

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
    nM, nK = len(files), len(KS_SHOW)

    P = None        # (n_meth, n_model, S, n_k, n_paper) ty le gan co, NaN neu khong o eval
    EVMASK = None   # (S, n_paper) True neu paper thuoc eval cua split do
    n_paper = None

    for mi, f in enumerate(files):
        rng = np.random.default_rng(TK.SEED)
        d = pd.read_parquet(f).dropna(subset=["score"])
        by_paper = {p: g.score.to_numpy() for p, g in d.groupby("paper_id")}
        ids = np.array(sorted(p for p, v in by_paper.items() if len(v) >= TK.KMAX))
        papers = [by_paper[p] for p in ids]

        if n_paper is None:
            n_paper = len(ids)
            P = np.full((len(METHODS), nM, TK.S, nK, n_paper), np.nan)
            EVMASK = np.zeros((TK.S, n_paper), dtype=bool)
            ids0 = ids.copy()
        else:
            assert len(ids) == n_paper and (ids == ids0).all(), "paper set khac nhau giua model"

        M = TK.draw(papers, rng)
        Craw = np.cumsum(M, axis=2) / KS

        for sp in range(TK.S):
            perm = rng.permutation(len(ids))
            cal, ev = perm[: len(ids) // 2], perm[len(ids) // 2:]
            if mi == 0:
                EVMASK[sp, ev] = True
            else:
                assert EVMASK[sp, ev].all(), "cach chia eval khac nhau giua model"

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
                P[ti, mi, sp, :, ev] = fl[:, [k - 1 for k in KS_SHOW]]
        print("  xong %s" % f.stem.replace("pred_test_fp_full_artifacts_", ""))

    def summarise(cnt):
        """cnt: (n_paper,) so lan moi paper duoc rut -> (n_meth, n_k) trung vi qua model."""
        w = cnt[None, None, None, None, :] * EVMASK[None, None, :, None, :]
        num = np.nansum(np.where(np.isnan(P), 0.0, P) * w, axis=-1)
        den = w.sum(axis=-1)
        rate = num / den                       # (n_meth, n_model, S, n_k)
        return np.median(rate.mean(axis=2), axis=1) * 100

    ones = np.ones(n_paper)
    point = summarise(ones)

    rng = np.random.default_rng(SEED_BOOT)
    B = np.empty((NBOOT, len(METHODS), nK))
    for b in range(NBOOT):
        # MOT vector trong so tren paper_id, dung chung cho moi split/model/phuong phap
        B[b] = summarise(rng.multinomial(n_paper, np.full(n_paper, 1.0 / n_paper)))
    print("\nbootstrap cum xong (B=%d, %d paper)" % (NBOOT, n_paper))

    # --- CI bien cho tung phuong phap tai tung k (dung cho ngoac vuong o Bang III)
    mrows = []
    for ti, meth in enumerate(METHODS):
        for j, k in enumerate(KS_SHOW):
            lo, hi = np.percentile(B[:, ti, j], [2.5, 97.5])
            mrows.append(dict(method=meth, k=k, fpr=point[ti, j], ci_lo=lo, ci_hi=hi))
    Mg = pd.DataFrame(mrows)
    Mg.to_csv(RES / "f7_cluster_marginal_ci.csv", index=False)
    print("\nCI BIEN theo cum (cho ngoac vuong Bang III):")
    print("  k        emp_deff                 zscale")
    for k in KS_SHOW:
        a = Mg[(Mg.method == "emp_deff") & (Mg.k == k)].iloc[0]
        b = Mg[(Mg.method == "zscale") & (Mg.k == k)].iloc[0]
        print("%3d  %.2f [%.2f,%.2f]      %.2f [%.2f,%.2f]"
              % (k, a.fpr, a.ci_lo, a.ci_hi, b.fpr, b.ci_lo, b.ci_hi))

    rows = []
    for a, c in PAIRS:
        ia, ic = METHODS.index(a), METHODS.index(c)
        for j, k in enumerate(KS_SHOW):
            dd = B[:, ia, j] - B[:, ic, j]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            pv_ = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
            rows.append(dict(k=k, method_a=a, method_b=c, fpr_a=point[ia, j],
                             fpr_b=point[ic, j], diff=point[ia, j] - point[ic, j],
                             ci_lo=lo, ci_hi=hi, p_boot=max(pv_, 1.0 / NBOOT),
                             excludes_zero=bool(lo > 0 or hi < 0)))
    pd.DataFrame(rows).to_csv(RES / "f7_cluster_diff_by_k.csv", index=False)

    def maxdev(X): return np.abs(X - TARGET).max(axis=-1)
    def rmsdev(X): return np.sqrt(((X - TARGET) ** 2).mean(axis=-1))

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
            dd = fn(B[:, ia, :]) - fn(B[:, ic, :])
            lo, hi = np.percentile(dd, [2.5, 97.5])
            pv_ = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
            srow.append(dict(method="%s - %s" % (a, c), method_name="hieu so", stat=nm,
                             value=float(fn(point[ia]) - fn(point[ic])),
                             ci_lo=lo, ci_hi=hi, p_boot=max(pv_, 1.0 / NBOOT),
                             excludes_zero=bool(lo > 0 or hi < 0)))
    Sc = pd.DataFrame(srow)
    Sc.to_csv(RES / "f7_cluster_scalar.csv", index=False)

    print("\nFPR diem (%) theo k -- phai TRUNG voi f2 (chi CI doi):")
    print("  k  " + "".join("%11s" % m for m in METHODS))
    for j, k in enumerate(KS_SHOW):
        print("%3d  " % k + "".join("%11.2f" % point[ti, j] for ti in range(len(METHODS))))

    print("\nSO SANH CI: f2 (doc lap moi split)  ->  f7 (bootstrap cum theo paper)")
    try:
        old = pd.read_csv(RES / "f2_scalar_stability.csv")
    except FileNotFoundError:
        old = None
    for a, c in PAIRS:
        for nm in ("rmsdev", "maxdev"):
            r = Sc[(Sc.method == "%s - %s" % (a, c)) & (Sc.stat == nm)].iloc[0]
            line = "  %-20s %-7s %+.2f  cum [%+.2f, %+.2f] p=%.3f" % (
                "%s-%s" % (a, c), nm, r.value, r.ci_lo, r.ci_hi, r.p_boot)
            if old is not None:
                o = old[(old.method == "%s - %s" % (a, c)) & (old.stat == nm)]
                if len(o):
                    o = o.iloc[0]
                    line += "   | f2 [%+.2f, %+.2f] p=%.3f" % (o.ci_lo, o.ci_hi, o.p_boot)
            print(line + ("  *" if r.excludes_zero else ""))
    print("\n-> results/f7_cluster_scalar.csv, f7_cluster_diff_by_k.csv")


if __name__ == "__main__":
    main()
