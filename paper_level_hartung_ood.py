"""Hartung tau(k) danh gia tren test_ood_model (co CA HAI lop) -- lay TPR/bo lot that.

    python3 paper_level_hartung_ood.py

VAN DE
------
tau_k_mean_16models.csv (paper_level_tau_k.py) chi do duoc FPR cua tau(k) Hartung, vi
test_fp_full toan human. Day la manh ghep con thieu: FPR VA ty le bo lot AI that cua chinh
cong thuc tau(k), tren mot dataset khac (test_ood_model, k~2-3, rat khac k=1..20 dung de
uoc rho) -- vua la "cai gia phai tra" cho nhanh mean/median, vua la kiem tra tau(k) co
tong quat hoa duoc sang mot dataset/k-range hoan toan khac hay khong.

THIET KE
--------
  - rho: DUNG LAI dung gia tri da uoc cho tung checkpoint trong tau_k_mean_16models.csv
    (trung vi qua 5 lan chia, tren population 576 bai test_fp_full >=20 doan) -- KHONG
    uoc lai tren test_ood_model, vi moi bai o do chi co ~2-3 doan, khong du de uoc phuong
    sai trong-bai (can >=20 doan/bai nhu K0_RHO trong paper_level_tau_k.py).
  - be mid-p: diem doan human trong `val` (dung nguon hieu chuan chuan cua pipeline, giong
    tau_chunk trong paper_level_mean_ktransfer.py) -- KHONG dung pool tu test_fp_full, de
    tau(k) o day chi dua tren du lieu se dung that khi trien khai (val), rho la tham so
    rieng uoc truoc do.
  - tau(k) = z_{1-alpha} * sqrt(1 + (k-1)*rho)  -- KHONG he so vay c (dung quyet dinh da
    chot trong outline: 1 tham so, khong them c).
  - Ap dung THEO TUNG BAI (moi bai/moi lop co k rieng, k thuc rat khac nhau giua human/AI
    va giua cac bai) -- khac voi 3 nguon nguong trong paper_level_mean_ktransfer.py, von
    dung MOT so tinh cho ca population.

DAU VAO
-------
  artifacts/kaggle/<artifact>/pred_val.parquet            be mid-p (diem doan human)
  artifacts/kaggle/<artifact>/pred_test_ood_model.parquet tap danh gia, CA HAI lop, mUC DOAN
  output/tau_k_mean_16models.csv                          nguon rho da uoc san moi checkpoint

DAU RA (output/)
----------------
  hartung_ood_16models.csv   FPR va ty le bo lot AI cua tau(k) Hartung tren test_ood_model,
                              16 checkpoint -- de so sanh truc tiep voi Table III
                              (mean_ktransfer_16models.csv, hang "tau chot DUNG o k~2,38")
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts" / "kaggle"
OUT = BASE / "output"
TARGET = 0.01
Z1A = float(norm.isf(TARGET))


def artifacts():
    for f in sorted(glob.glob(str(ART / "artifacts_*" / "pred_test_ood_model.parquet"))):
        art = Path(f).parent.name.replace("artifacts_", "")
        yield art, f


def main():
    rho_by_model = (
        pd.read_csv(OUT / "tau_k_mean_16models.csv")
        .groupby("model")["rho"].median()
    )

    rows = []
    for art, f_ood in artifacts():
        rho = float(rho_by_model.loc[art])

        val = pd.read_parquet(ART / f"artifacts_{art}" / "pred_val.parquet")
        pool = np.sort(val.loc[val.label == 0, "score"].to_numpy())
        nc = len(pool)

        ood = pd.read_parquet(f_ood)
        # p-value mid-p tung DOAN (khong gop truoc), roi z-hoa, roi cong don theo bai
        pv = (0.5 + nc - np.searchsorted(pool, ood.score.to_numpy(), side="left")) / (nc + 1.0)
        ood = ood.assign(z=norm.isf(pv))

        g = ood.groupby(["paper_id", "label"])["z"]
        k = g.size().to_numpy()
        Zk = (g.sum() / np.sqrt(k)).to_numpy()
        lbl = g.size().index.get_level_values("label").to_numpy()
        tau_k = Z1A * np.sqrt(1 + (k - 1) * rho)
        flagged = Zk > tau_k

        human, ai = lbl == 0, lbl == 1
        rows.append(dict(
            model=art, rho=rho,
            fpr_hartung=float(flagged[human].mean()),
            miss_hartung=float((~flagged[ai]).mean()),
            k_mean_human=float(k[human].mean()), k_mean_ai=float(k[ai].mean()),
            n_human=int(human.sum()), n_ai=int(ai.sum()),
        ))

    df = pd.DataFrame(rows).sort_values("fpr_hartung")
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "hartung_ood_16models.csv", index=False)
    print(df.round(4).to_string(index=False))

    f, m = df.fpr_hartung * 100, df.miss_hartung * 100
    print(f"\nTrung vi qua {len(df)} checkpoint (muc tieu FPR 1,00%):")
    print(f"  tau(k) Hartung (rho tu test_fp_full, danh gia tren test_ood_model)"
          f"   FPR {f.median():5.2f}% [{f.min():.2f}-{f.max():.2f}]"
          f"   bo lot {m.median():6.2f}% [{m.min():.2f}-{m.max():.2f}]")
    print("\n-> output/hartung_ood_16models.csv")


if __name__ == "__main__":
    main()
