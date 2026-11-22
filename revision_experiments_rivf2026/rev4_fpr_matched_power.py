"""REV-4 -- so bo lot tai FPR KHOP CHINH XAC, de loai tru gia thuyet "chi vi nguong long hon".

    python revision_experiments_rivf2026/rev4_fpr_matched_power.py

VAN DE
------
REV-3 cho thay Z_k bo lot it hon nguong tho tren 13/16 checkpoint. Nhung Z_k dang chay
o FPR 1,26% con doi chung o 1,10% -- cao hon 15% tuong doi. Ha nguong thi LUON lam
bo lot giam va FPR tang: do la danh doi co hoc, khong can "thong ke tot hon" nao ca.
Nen 13/16 tu no KHONG loai tru duoc gia thuyet:

    H_loose: toan bo loi the cua Z_k den tu viec no dang o diem van hanh long hon.

Bai hien chi co mot cau tu ve bang loi ("establishes disagreement, not power dominance
at exactly equal FPR"). Day la thu nghiem loai tru H_loose bang so.

CACH LAM
--------
Voi TUNG checkpoint, tren test_ood_model (637 bai human + 637 bai AI):
  1. Chay quy tac de xuat -> dem duoc m bai human bi gan co (FPR = m/637).
  2. Chon nguong raw-mean tau* sao cho no gan co DUNG m bai human -- tuc lay thong ke
     thu tu thu m tu tren xuong cua trung binh diem cac bai human.
     => FPR cua hai quy tac BANG NHAU TUYET DOI theo cau tao, tren cung 637 bai.
  3. Chi luc do moi so bo lot tren 637 bai AI, va McNemar tren cac bai bat dong.

Neu Z_k van bo lot it hon o FPR khop chinh xac -> H_loose bi loai, loi the la cua
THONG KE. Neu khong -> phai ha claim xuong.

DAU RA
------
  results/rev4_fpr_matched.csv
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

ART = ROOT / "artifacts" / "kaggle"
OUT = ROOT / "output"
RES = HERE / "results"
ALPHA = 0.01
Z1A = float(norm.isf(ALPHA))


def main():
    RES.mkdir(exist_ok=True)
    rho_by_model = (pd.read_csv(OUT / "tau_k_mean_16models.csv")
                      .groupby("model")["rho"].median())

    rows = []
    for f in sorted(glob.glob(str(ART / "artifacts_*" / "pred_test_ood_model.parquet"))):
        model = Path(f).parent.name.replace("artifacts_", "")
        rho = float(rho_by_model.loc[model])

        val = pd.read_parquet(ART / ("artifacts_%s" % model) / "pred_val.parquet")
        pool = np.sort(val.loc[val.label == 0, "score"].to_numpy())
        m_pool = len(pool)

        ood = pd.read_parquet(f)
        pv = (0.5 + m_pool
              - np.searchsorted(pool, ood.score.to_numpy(), side="left")) / (m_pool + 1.0)
        ood = ood.assign(z=norm.isf(pv))

        g = ood.groupby(["paper_id", "label"])
        k = g.size().to_numpy()
        Zk = (g["z"].sum() / np.sqrt(k)).to_numpy()
        mean_raw = g["score"].mean().to_numpy()
        lbl = g.size().index.get_level_values("label").to_numpy()
        hu, ai = lbl == 0, lbl == 1

        # --- quy tac de xuat ---
        flag_z = Zk > Z1A * np.sqrt(1 + (k - 1) * rho)
        n_flag_h = int(flag_z[hu].sum())          # so bai human bi gan co
        n_h, n_a = int(hu.sum()), int(ai.sum())

        # --- doi chung KHOP FPR TUYET DOI: chon tau* de cung gan co dung n_flag_h bai ---
        h_scores = np.sort(mean_raw[hu])[::-1]     # giam dan
        if n_flag_h == 0:
            tau_m = h_scores[0] + 1e-12            # khong gan co bai human nao
        elif n_flag_h >= n_h:
            tau_m = -np.inf
        else:
            # nguong nam giua thong ke thu tu thu n_flag_h va thu (n_flag_h+1)
            tau_m = 0.5 * (h_scores[n_flag_h - 1] + h_scores[n_flag_h])
        flag_c = mean_raw >= tau_m

        fpr_z, fpr_c = flag_z[hu].mean(), flag_c[hu].mean()
        miss_z, miss_c = (~flag_z[ai]).mean(), (~flag_c[ai]).mean()

        # McNemar tren lop AI, chi dem bai bat dong
        b = int((flag_z[ai] & ~flag_c[ai]).sum())   # chi z bat duoc
        c = int((~flag_z[ai] & flag_c[ai]).sum())   # chi doi chung bat duoc
        p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0

        rows.append(dict(model=model, n_human=n_h, n_ai=n_a,
                         n_flagged_human=n_flag_h, tau_matched=float(tau_m),
                         fpr_z=float(fpr_z), fpr_ctrl=float(fpr_c),
                         fpr_gap=float(abs(fpr_z - fpr_c)),
                         miss_z=float(miss_z), miss_ctrl=float(miss_c),
                         delta_miss=float(miss_c - miss_z),
                         b_only_z=b, c_only_ctrl=c, p_mcnemar=p))
        print("  xong %s" % model)

    D = pd.DataFrame(rows)
    D.to_csv(RES / "rev4_fpr_matched.csv", index=False)

    assert D.fpr_gap.max() < 1e-12, "FPR chua khop tuyet doi: %g" % D.fpr_gap.max()

    print("\n" + "=" * 86)
    print("REV-4  Bo lot tai FPR KHOP CHINH XAC (cung 637 bai human, cung so bai bi gan co)")
    print("=" * 86)
    print("%-32s %8s %9s %9s %8s %6s %6s %9s"
          % ("checkpoint", "FPR", "miss z", "miss ctrl", "chenh", "b", "c", "p"))
    for _, r in D.sort_values("delta_miss", ascending=False).iterrows():
        print("%-32s %7.2f%% %8.2f%% %8.2f%% %7.2f%% %6d %6d %9.2e"
              % (r.model, 100 * r.fpr_z, 100 * r.miss_z, 100 * r.miss_ctrl,
                 100 * r.delta_miss, r.b_only_z, r.c_only_ctrl, r.p_mcnemar))

    better = int((D.miss_z < D.miss_ctrl).sum())
    worse = int((D.miss_z > D.miss_ctrl).sum())
    sig = int((D.p_mcnemar < 0.05).sum())
    print("\nFPR khop tuyet doi o moi checkpoint (chenh toi da %.1e)" % D.fpr_gap.max())
    print("  FPR trung vi          : %.2f%%" % (100 * D.fpr_z.median()))
    print("  bo lot z    trung vi  : %.2f%%" % (100 * D.miss_z.median()))
    print("  bo lot ctrl trung vi  : %.2f%%" % (100 * D.miss_ctrl.median()))
    print("  chenh trung vi        : %.2f diem %%" % (100 * D.delta_miss.median()))
    print("  z TOT hon o %d/16 | te hon o %d/16 | McNemar p<0,05 o %d/16"
          % (better, worse, sig))
    B, C = int(D.b_only_z.sum()), int(D.c_only_ctrl.sum())
    print("  gop: b=%d (chi z bat) vs c=%d (chi ctrl bat), p=%.2e"
          % (B, C, binomtest(B, B + C, 0.5).pvalue))
    print("\n-> results/rev4_fpr_matched.csv")


if __name__ == "__main__":
    main()
