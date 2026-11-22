"""REV-3 -- kiem dinh GHEP CAP cho chenh lech bo lot o Table IV.

    python revision_experiments_rivf2026/rev3_paired_miss_test.py

VAN DE
------
Table IV: o cung muc FPR (1,26% vs 1,31%), quy tac z-scale bo lot 3,77% con nguong
doi chung bo lot 6,16%. O CUNG FPR, hieu chuan KHONG the mua them nang luc -- nen
chenh lech nay co nghia Z_k la mot THONG KE GOP tot hon, khong chi duoc dat nguong
tot hon. Bai khong noi gi ve dieu do.

Ly do co that: trung binh cua cac gia tri da bien doi don dieu KHONG phai phep bien
doi don dieu cua trung binh. Z-hoa lam phang duoi phai, nen mot doan AI diem 0,99
khong con keo lech ca tai lieu, va tin hieu yeu trai deu nhieu doan duoc cong don tot
hon.

Nhung 2,39 diem % tren 637 bai chi dat z ~ 1,97 neu tinh KHONG ghep cap. Day la so
sanh GHEP CAP (hai quy tac cham CUNG mot tap bai), nen phai dung McNemar -- chi dem
cac bai ma hai quy tac BAT DONG y. Manh hon nhieu.

DAU RA
------
  results/rev3_paired_miss.csv   moi checkpoint: b, c, p (McNemar), delta
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
TARGET = 0.01
Z1A = float(norm.isf(TARGET))


def main():
    RES.mkdir(exist_ok=True)
    rho_by_model = (pd.read_csv(OUT / "tau_k_mean_16models.csv")
                      .groupby("model")["rho"].median())
    tau_ood = (pd.read_csv(OUT / "mean_ktransfer_16models.csv")
                 .set_index("model")["tau_ood"])

    rows = []
    for f in sorted(glob.glob(str(ART / "artifacts_*" / "pred_test_ood_model.parquet"))):
        model = Path(f).parent.name.replace("artifacts_", "")
        rho = float(rho_by_model.loc[model])

        val = pd.read_parquet(ART / ("artifacts_%s" % model) / "pred_val.parquet")
        pool = np.sort(val.loc[val.label == 0, "score"].to_numpy())
        nc = len(pool)

        ood = pd.read_parquet(f)
        pv = (0.5 + nc - np.searchsorted(pool, ood.score.to_numpy(), side="left")) / (nc + 1.0)
        ood = ood.assign(z=norm.isf(pv))

        g = ood.groupby(["paper_id", "label"])
        k = g.size().to_numpy()
        Zk = (g["z"].sum() / np.sqrt(k)).to_numpy()
        mean_raw = g["score"].mean().to_numpy()
        lbl = g.size().index.get_level_values("label").to_numpy()

        flag_z = Zk > Z1A * np.sqrt(1 + (k - 1) * rho)      # quy tac de xuat
        flag_c = mean_raw >= float(tau_ood.loc[model])       # nguong doi chung tai k~2,38

        ai = lbl == 1
        hu = lbl == 0
        # McNemar tren lop AI: chi dem cac bai hai quy tac BAT DONG
        b = int((flag_z[ai] & ~flag_c[ai]).sum())   # z BAT, doi chung BO LOT
        c = int((~flag_z[ai] & flag_c[ai]).sum())   # z BO LOT, doi chung BAT
        p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
        rows.append(dict(
            model=model, n_ai=int(ai.sum()), n_human=int(hu.sum()),
            miss_z=float((~flag_z[ai]).mean()), miss_ctrl=float((~flag_c[ai]).mean()),
            fpr_z=float(flag_z[hu].mean()), fpr_ctrl=float(flag_c[hu].mean()),
            b_only_z=b, c_only_ctrl=c, n_discordant=b + c, p_mcnemar=p))
        print("  xong %s" % model)

    D = pd.DataFrame(rows)
    D.to_csv(RES / "rev3_paired_miss.csv", index=False)

    print("\n" + "=" * 78)
    print("REV-3  McNemar ghep cap tren lop AI (moi checkpoint, n_ai = %d bai)"
          % D.n_ai.iloc[0])
    print("=" * 78)
    print("%-32s %7s %7s %6s %6s %10s" %
          ("checkpoint", "miss_z", "miss_c", "b", "c", "p"))
    for _, r in D.sort_values("p_mcnemar").iterrows():
        print("%-32s %6.2f%% %6.2f%% %6d %6d %10.2e"
              % (r.model, 100 * r.miss_z, 100 * r.miss_ctrl,
                 r.b_only_z, r.c_only_ctrl, r.p_mcnemar))

    sig = int((D.p_mcnemar < 0.05).sum())
    better = int((D.miss_z < D.miss_ctrl).sum())
    print("\nTrung vi: miss_z %.2f%%  vs  miss_ctrl %.2f%%   (chenh %.2f diem %%)"
          % (100 * D.miss_z.median(), 100 * D.miss_ctrl.median(),
             100 * (D.miss_ctrl.median() - D.miss_z.median())))
    print("FPR    : z %.2f%%  vs  doi chung %.2f%%  (phai TUONG DUONG moi so sanh duoc)"
          % (100 * D.fpr_z.median(), 100 * D.fpr_ctrl.median()))
    print("z-scale bo lot IT HON o %d/16 checkpoint; McNemar p<0,05 o %d/16"
          % (better, sig))

    # gop 16 checkpoint: dem discordant tong
    B, C = int(D.b_only_z.sum()), int(D.c_only_ctrl.sum())
    print("\nGop toan bo 16 checkpoint: b=%d (chi z bat), c=%d (chi doi chung bat)"
          % (B, C))
    print("   McNemar gop: p = %.3e" % binomtest(B, B + C, 0.5).pvalue)
    print("   (luu y: 16 checkpoint KHONG doc lap -- cham cung 637 bai;"
          " so gop chi de tham khao)")
    print("-> results/rev3_paired_miss.csv")


if __name__ == "__main__":
    main()
