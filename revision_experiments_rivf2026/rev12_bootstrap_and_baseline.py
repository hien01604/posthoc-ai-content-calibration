"""REV-1 + REV-2 -- khoang tin cay bootstrap THEO BAI, va baseline con thieu.

    python revision_experiments_rivf2026/rev12_bootstrap_and_baseline.py

Hai viec dung chung 90% khoi luong tinh (nap du lieu, rut mau con, z-hoa) nen gop
vao mot script.

REV-1 -- BOOTSTRAP THEO BAI
---------------------------
Bai dang bao FPR toi hai chu so thap phan va lap luan tren chenh lech 0,14-0,24 diem %.
Nhung FPR do tren 288 bai eval, nen sai so chuan nhi thuc o muc 1% la
sqrt(0,01*0,99/288) ~ 0,5 diem % -- LON HON moi chenh lech do.

Dai min-max qua 16 checkpoint ma bai dang dung nhu thanh sai so KHONG phai sai so
lay mau: 16 checkpoint cham Y HET 576 bai do, va 5 lan chia dung lai cung quan the
voi hai nua chong lan. Chung do DO ON DINH NOI BO, khong do bat dinh doi voi quan the
bai bao.

Cach lam dung: lay ty le gan co CUA TUNG BAI, roi bootstrap chinh cac BAI do.
Trong moi lan lap, cung mot bo chi so bai duoc dung cho MOI checkpoint va MOI phuong
phap -- vi chung deu cham cung nhung bai ay, nen nhieu o cap bai la nhieu CHUNG.

REV-2 -- BASELINE CON THIEU
---------------------------
Bai doi lap (a) nguong Gauss suy tu phuong sai tren thang tho voi (b) bien doi hang
co hieu chuan thuc nghiem. Nhung (b) duoc NHIN THAY mot pool hieu chuan con (a) thi
khong -- khong phai tran dau can suc, nen chua tach duoc hai gia thuyet:
    H1: hieu ung thiet ke sai vi phan phoi null lech  (luan diem cua bai)
    H2: hieu ung thiet ke dung, chi phan vi Gauss sai; hieu chuan thuc nghiem o
        THANG NAO CUNG chua duoc
Baseline tach duoc chung: lay phan vi 99% THUC NGHIEM cua diem mot doan tren nua cal
(khong dung phan vi Gauss), roi co gian theo dung hieu ung thiet ke:

    tau(k) = mu0_cal + (q99_cal - mu0_cal) * sqrt( (1 + (k-1)*rho_raw) / k )

Dung cung luong du lieu hieu chuan nhu phuong phap de xuat, chi khac DUY NHAT o cho
khong co phep bien doi phan phoi. Neu baseline nay dat muc tieu -> H2 dung, dong gop
cua bai yeu di nhieu. Neu no van hong -> H1 dung, va luc do moi ket luan duoc.

DAU RA
------
  results/rev1_bootstrap_ci.csv     k, method, fpr, ci_lo, ci_hi
  results/rev2_baseline_by_k.csv    k, method, fpr (+ CI)
  results/rev12_per_checkpoint.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import paper_level_tau_k as TK   # noqa: E402  -- tai su dung y het may cua bai

RES = HERE / "results"
KS_SHOW = [1, 2, 3, 5, 8, 10, 15, 20]
NBOOT = 2000
SEED_BOOT = 20260826
Z1A = float(norm.isf(TK.ALPHA))

METHODS = ["static", "oracle", "zscale", "emp_deff"]
NAMES = {"static": "Static threshold (pinned at k=20)",
         "oracle": "Per-k oracle calibration",
         "zscale": "z-scale Hartung (proposed)",
         "emp_deff": "REV-2 baseline: empirical q99 at k=1 x design effect"}


def rho_raw_icc(cal_papers):
    """ICC tren THANG THO, dung cong thuc cua paper_level_pihat.py."""
    within = np.mean([v[: TK.K0_RHO].var(ddof=1) for v in cal_papers])
    total = np.concatenate(cal_papers).var(ddof=1)
    return max(0.0, 1.0 - within / total), total


def main():
    RES.mkdir(exist_ok=True)
    files = sorted(TK.FP_FULL.glob("pred_test_fp_full_artifacts_*.parquet"))
    KS = np.arange(1, TK.KMAX + 1)

    # per_paper[(model, split, k, method)] = mang ty le gan co cua tung bai eval
    per_paper = {}
    rows = []

    for f in files:
        model = f.stem.replace("pred_test_fp_full_artifacts_", "")
        rng = np.random.default_rng(TK.SEED)          # y het paper_level_tau_k.py
        d = pd.read_parquet(f).dropna(subset=["score"])
        by_paper = {p: g.score.to_numpy() for p, g in d.groupby("paper_id")}
        ids = np.array(sorted(p for p, v in by_paper.items() if len(v) >= TK.KMAX))
        papers = [by_paper[p] for p in ids]

        M = TK.draw(papers, rng)                      # (R, n, KMAX)
        Craw = np.cumsum(M, axis=2) / KS

        for sp in range(TK.S):
            perm = rng.permutation(len(ids))
            cal, ev = perm[: len(ids) // 2], perm[len(ids) // 2:]
            pool = np.sort(np.concatenate([by_paper[p] for p in ids[cal]]))
            nc = len(pool)
            pv = (0.5 + nc - np.searchsorted(pool, M, side="left")) / (nc + 1.0)
            Cz = np.cumsum(norm.isf(pv), axis=2) / np.sqrt(KS)
            rho_z = TK.rho_hartung(norm.isf(pv[0][cal][:, : TK.K0_RHO]))

            # --- nguong cua tung phuong phap ---
            Q = np.quantile(Craw[:, cal, :], 1 - TK.ALPHA, axis=1,
                            method="higher").mean(axis=0)      # tau_emp(k)
            tau = {"oracle": Q, "static": np.full(TK.KMAX, Q[-1]),
                   "zscale": Z1A * np.sqrt(1 + (KS - 1) * rho_z)}

            # --- REV-2: phan vi THUC NGHIEM tai k=1, co gian theo hieu ung thiet ke
            cal_papers = [by_paper[p] for p in ids[cal]]
            rho_r, _ = rho_raw_icc(cal_papers)
            mu0 = np.concatenate(cal_papers).mean()
            q99 = Q[0]                                  # phan vi 99% cua MOT doan
            tau["emp_deff"] = mu0 + (q99 - mu0) * np.sqrt((1 + (KS - 1) * rho_r) / KS)

            for meth in METHODS:
                src = Cz if meth == "zscale" else Craw
                # ty le gan co cua TUNG bai eval (trung binh qua R lan rut mau con)
                fl = (src[:, ev, :] >= tau[meth][None, None, :]).mean(axis=0)  # (n_ev,KMAX)
                for k in KS_SHOW:
                    per_paper[(model, sp, k, meth)] = fl[:, k - 1]
                    rows.append(dict(model=model, split=sp, k=k, method=meth,
                                     fpr=float(fl[:, k - 1].mean()),
                                     rho_raw=rho_r, rho_z=rho_z))
        print("  xong %s" % model)

    D = pd.DataFrame(rows)
    D.to_csv(RES / "rev12_per_checkpoint.csv", index=False)

    models = sorted(D.model.unique())
    n_ev = len(per_paper[(models[0], 0, 1, "static")])

    # ---------------------------------------------------------------- bootstrap
    rng = np.random.default_rng(SEED_BOOT)
    boot = {(k, m): np.empty(NBOOT) for k in KS_SHOW for m in METHODS}
    for b in range(NBOOT):
        # CUNG bo chi so bai cho moi checkpoint/phuong phap trong mot lan lap
        idx = {sp: rng.integers(0, n_ev, n_ev) for sp in range(TK.S)}
        for k in KS_SHOW:
            for meth in METHODS:
                per_model = [np.mean([per_paper[(mo, sp, k, meth)][idx[sp]].mean()
                                      for sp in range(TK.S)]) for mo in models]
                boot[(k, meth)][b] = np.median(per_model)

    out = []
    for k in KS_SHOW:
        for meth in METHODS:
            pt = np.median([np.mean([per_paper[(mo, sp, k, meth)].mean()
                                     for sp in range(TK.S)]) for mo in models])
            lo, hi = np.percentile(boot[(k, meth)], [2.5, 97.5])
            out.append(dict(k=k, method=meth, method_name=NAMES[meth],
                            fpr=pt * 100, ci_lo=lo * 100, ci_hi=hi * 100,
                            ci_width=(hi - lo) * 100, n_eval_papers=n_ev,
                            n_boot=NBOOT))
    B = pd.DataFrame(out)
    B.to_csv(RES / "rev1_bootstrap_ci.csv", index=False)
    B[B.method == "emp_deff"].to_csv(RES / "rev2_baseline_by_k.csv", index=False)

    # ---------------------------------------------------------------- bao cao
    print("\n" + "=" * 78)
    print("REV-1  FPR voi KHOANG TIN CAY 95%% bootstrap theo bai (%d bai, B=%d)"
          % (n_ev, NBOOT))
    print("=" * 78)
    print("  k  " + "".join("%26s" % NAMES[m][:25] for m in
                            ["static", "oracle", "zscale"]))
    for k in KS_SHOW:
        line = "%3d  " % k
        for m in ["static", "oracle", "zscale"]:
            r = B[(B.k == k) & (B.method == m)].iloc[0]
            line += "%26s" % ("%.2f [%.2f, %.2f]" % (r.fpr, r.ci_lo, r.ci_hi))
        print(line)

    z = B[B.method == "zscale"]
    print("\nBe rong CI cua cot z-scale: trung vi %.2f diem %% (min %.2f, max %.2f)"
          % (z.ci_width.median(), z.ci_width.min(), z.ci_width.max()))
    print("Muc tieu 1,00%% co nam trong CI khong?")
    inside = [(int(r.k), r.ci_lo <= 1.0 <= r.ci_hi) for _, r in z.iterrows()]
    print("   " + ", ".join("k=%d:%s" % (k, "CO" if v else "KHONG") for k, v in inside))

    o = B[B.method == "oracle"].set_index("k")
    zz = z.set_index("k")
    print("\nChenh de xuat - oracle so voi be rong CI:")
    for k in KS_SHOW:
        gap = abs(zz.loc[k, "fpr"] - o.loc[k, "fpr"])
        print("   k=%2d  |gap| = %.2f diem %%   (CI cua z rong %.2f diem %%)  -> %s"
              % (k, gap, zz.loc[k, "ci_width"],
                 "DUOI nguong phan giai" if gap < zz.loc[k, "ci_width"] / 2
                 else "phan biet duoc"))

    print("\n" + "=" * 78)
    print("REV-2  BASELINE: phan vi 99%% THUC NGHIEM tai k=1 x hieu ung thiet ke")
    print("=" * 78)
    print("  k     baseline REV-2        z-scale (de xuat)        muc tieu")
    for k in KS_SHOW:
        e = B[(B.k == k) & (B.method == "emp_deff")].iloc[0]
        zr = B[(B.k == k) & (B.method == "zscale")].iloc[0]
        print("%3d  %8s  %8s   %s"
              % (k, "%.2f" % e.fpr, "[%.2f,%.2f]" % (e.ci_lo, e.ci_hi),
                 "%6.2f [%.2f,%.2f]        1.00" % (zr.fpr, zr.ci_lo, zr.ci_hi)))
    e = B[B.method == "emp_deff"]
    print("\nrho_raw dung cho baseline: trung vi %.4f" % D.rho_raw.median())
    print("Baseline REV-2 chay tu %.2f%% (k=1) den %.2f%% (k=20)"
          % (e[e.k == 1].fpr.iloc[0], e[e.k == 20].fpr.iloc[0]))
    print("-> results/rev1_bootstrap_ci.csv, rev2_baseline_by_k.csv")


if __name__ == "__main__":
    main()
