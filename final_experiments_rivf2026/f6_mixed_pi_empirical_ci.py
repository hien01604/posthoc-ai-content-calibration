"""F6 -- KHOANG DISTRIBUTION-FREE cho pi_hat tai pi > 0 (chu khong chi pi = 0).

    py f6_mixed_pi_empirical_ci.py

VAN DE (phan bien B3 cua reviewer)
----------------------------------
f1_pihat_ci.py do do phu cua khoang THUC NGHIEM (cov_emp) CHI tai pi = 0. O pi > 0 no
chi do cov_naive/cov_deff (deu hong: 83-92%). Nhung bai lai phat bieu tong quat
"a distribution-free interval holds 94.8% throughout" va "+-0,15 cho mot bai day du".
Ket luan rut ra tu pi = 0 khong duoc phep mo rong cho moi pi neu chua do.

HAI BIEN THE PHAI TACH BACH
---------------------------
  (1) emp_pi   : phan vi 2,5/97,5 hieu chuan tai DUNG pi that.
                 -> gioi han tren LAC QUAN: tra loi "khoang distribution-free CO THE
                    dat 95% o pi>0 khong?", nhung trien khai that khong biet pi.
  (2) emp_null : phan vi hieu chuan tai pi = 0 roi ap nguyen cho moi pi.
                 -> chinh la thu DUNG DUOC trong trien khai (khong can biet pi).
                    Neu (2) giu duoc 95% thi claim cua bai dung o dang manh.

TAI LIEU BAN TONG HOP (giong f1, giu nguyen han che)
----------------------------------------------------
  k doan human rut tu MOT bai that (giu tuong quan noi tai lieu THAT o phia human),
  round(pi*k) doan thay bang diem AI rut tu pool test_ood_model. Phia AI van co rho = 0
  -> ghi ro trong bai la gioi han lac quan, khong duoc im lang.

TACH CAL/EVAL O CA HAI PHIA
---------------------------
  - bai human: 576 bai >= 20 doan, chia doi theo paper_id (giong f1/f2).
  - diem AI: pool test_ood_model cung chia doi, nua cal chi dung de hieu chuan,
    nua eval chi dung de do -> khong tai su dung diem AI giua hai buoc.

DAU RA (results/)
-----------------
  f6_coverage_mixed_emp.csv   do phu 4 bien the x (k, pi) x 16 checkpoint
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from icc import icc1

ART = ROOT / "artifacts" / "kaggle"
FPF = ROOT / "artifacts" / "kaggle_testfp_full" / "pred_testfp_full"
OUT = ROOT / "output"
RES = HERE / "results"

KMAX = 20
KS_SHOW = [5, 10, 20]
PIS = [0.00, 0.25, 0.50, 0.75, 1.00]
NDOC = 4000          # tai lieu ban tong hop moi (model, split, k, pi), moi phia cal/eval
S = 5                # so lan chia cal/eval
SEED = 42
CONF = 0.95
ZC = float(norm.isf((1 - CONF) / 2))
QLO, QHI = (1 - CONF) / 2, 1 - (1 - CONF) / 2
K0_RHO = 20


def rho_raw_icc(papers_cal, var_total=None):
    """ICC(1) mot chieu tren thang THO -- xem icc.py."""
    return icc1(papers_cal, K0_RHO)


def mixed_docs(paper_pool, sA, pi, k, rng, ndoc=NDOC):
    """(ndoc, k): k doan human tu MOT bai that, round(pi*k) doan thay bang diem AI."""
    pick = rng.integers(0, len(paper_pool), ndoc)
    S_ = np.empty((ndoc, k))
    for r in range(ndoc):
        a = paper_pool[pick[r]]
        S_[r] = rng.choice(a, size=k, replace=False)
    n_ai = int(round(pi * k))
    if n_ai:
        m = np.zeros((ndoc, k), dtype=bool)
        idx = np.argsort(rng.random((ndoc, k)), axis=1)[:, :n_ai]
        np.put_along_axis(m, idx, True, axis=1)
        S_ = np.where(m, sA[rng.integers(0, len(sA), (ndoc, k))], S_)
    return S_


def main():
    RES.mkdir(exist_ok=True)
    anc = pd.read_csv(OUT / "pihat_sanity_16models.csv").set_index("model")
    rows = []

    for f in sorted(FPF.glob("pred_test_fp_full_artifacts_*.parquet")):
        model = f.stem.replace("pred_test_fp_full_artifacts_", "")
        mu0, mu1 = float(anc.loc[model, "mu0"]), float(anc.loc[model, "mu1"])
        den = mu1 - mu0

        rng = np.random.default_rng(SEED)
        df = pd.read_parquet(f).dropna(subset=["score"])
        by_paper = {p: g.score.to_numpy() for p, g in df.groupby("paper_id")}
        ids = np.array(sorted(p for p, v in by_paper.items() if len(v) >= KMAX))
        papers = [by_paper[p] for p in ids]

        ai = pd.read_parquet(ART / ("artifacts_%s" % model) / "pred_test_ood_model.parquet")
        sA_all = ai.loc[ai.label == 1, "score"].to_numpy()

        for sp in range(S):
            perm = rng.permutation(len(ids))
            cal, ev = perm[: len(ids) // 2], perm[len(ids) // 2:]
            cal_papers = [papers[i] for i in cal]
            ev_papers = [papers[i] for i in ev]

            permA = rng.permutation(len(sA_all))
            sA_cal = sA_all[permA[: len(sA_all) // 2]]
            sA_ev = sA_all[permA[len(sA_all) // 2:]]

            pool_cal = np.concatenate(cal_papers)
            var_tot = float(np.var(pool_cal, ddof=1))
            rho = rho_raw_icc(cal_papers, var_tot)

            for k in KS_SHOW:
                hw_naive = ZC * np.sqrt(var_tot / k) / den
                hw_deff = ZC * np.sqrt(var_tot / k * (1 + (k - 1) * rho)) / den

                # LUOI hieu chuan: phan vi cua sai so (pi_hat - pi) tai tung pi luoi
                grid = [q for q in PIS if abs(round(q * k) - q * k) <= 1e-9]
                gq = {}
                for q in grid:
                    cc = mixed_docs(cal_papers, sA_cal, q, k, rng)
                    gq[q] = np.quantile((cc.mean(axis=1) - mu0) / den - q, [QLO, QHI])
                gpi = np.array(grid)
                glo = np.array([gq[q][0] for q in grid])
                ghi = np.array([gq[q][1] for q in grid])
                lo0, hi0 = gq[0.0]              # hieu chuan tai pi = 0

                for pi in grid:
                    lo_pi, hi_pi = gq[pi]       # hieu chuan tai DUNG pi (lac quan)

                    ee = mixed_docs(ev_papers, sA_ev, pi, k, rng)
                    ph = (ee.mean(axis=1) - mu0) / den
                    e = ph - pi
                    # PLUG-IN: moi tai lieu tu chon o luoi gan pi_hat cua CHINH no
                    j = np.abs(ph[:, None] - gpi[None, :]).argmin(axis=1)
                    lo_pl, hi_pl = glo[j], ghi[j]
                    rows.append(dict(
                        model=model, split=sp, k=k, true_pi=pi,
                        mean_pi_hat=float(ph.mean()), rho_raw=rho,
                        cov_naive=float((np.abs(e) <= hw_naive).mean()),
                        cov_deff=float((np.abs(e) <= hw_deff).mean()),
                        cov_emp_pi=float(((e >= lo_pi) & (e <= hi_pi)).mean()),
                        cov_emp_null=float(((e >= lo0) & (e <= hi0)).mean()),
                        cov_emp_plugin=float(((e >= lo_pl) & (e <= hi_pl)).mean()),
                        hw_naive=hw_naive, hw_deff=hw_deff,
                        hw_emp_pi=float((hi_pi - lo_pi) / 2),
                        hw_emp_null=float((hi0 - lo0) / 2),
                        hw_emp_plugin=float(np.mean(hi_pl - lo_pl) / 2)))
        print("  xong %s" % model)

    D = pd.DataFrame(rows)
    D.to_csv(RES / "f6_coverage_mixed_emp.csv", index=False)

    per = D.groupby(["model", "k", "true_pi"]).mean(numeric_only=True).reset_index()
    g = per.groupby(["k", "true_pi"]).median(numeric_only=True)

    print("\nDO PHU cua khoang danh nghia 95%% cho pi_hat, tai lieu ban tong hop")
    print("  k    pi   naive    deff   emp@pi  emp@0  plugin | nua rong: deff  emp@0  plugin")
    for (k, pi), r in g.iterrows():
        print("%3d  %.2f  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%% |  %.3f  %.3f  %.3f"
              % (k, pi, 100 * r.cov_naive, 100 * r.cov_deff,
                 100 * r.cov_emp_pi, 100 * r.cov_emp_null, 100 * r.cov_emp_plugin,
                 r.hw_deff, r.hw_emp_null, r.hw_emp_plugin))

    print("\nTOM TAT tai k=20 (trung vi qua checkpoint, min qua pi):")
    s = per[per.k == 20].groupby("model")[
        ["cov_emp_pi", "cov_emp_null", "cov_emp_plugin", "cov_deff"]].min()
    for c in ("cov_deff", "cov_emp_null", "cov_emp_plugin", "cov_emp_pi"):
        print("  %-13s do phu THAP NHAT qua pi: trung vi %5.1f%%  dai [%.1f%%, %.1f%%]"
              % (c, 100 * s[c].median(), 100 * s[c].min(), 100 * s[c].max()))

    print("\n-> results/f6_coverage_mixed_emp.csv")


if __name__ == "__main__":
    main()
