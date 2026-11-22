"""S8 -- pi-hat tren van ban tron o muc CAU (duoi muc doan).

    python supplement_rivf2026/submix_pihat.py

VI SAO
------
§VI cua bai noi -- bang loi, khong co so -- rang mo hinh tron 2 thanh phan gia dinh moi
doan thuoc TRON VEN mot trong hai cuc, nen pi-hat phai doc la "muc do giong van phong
AI" chu khong phai "ti le ky tu do AI go". `artifacts/pred_mix/scores_all.parquet` cho
phep DO chinh dieu do: 325.728 dong da cham diem, 609 bai, 16 checkpoint, moi dong la
MOT doan (~270 tu) bi thay mot so CAU bang ban AI tuong ung, kem ti le tu that
(`mix_ratio_actual`).

PHAM VI -- doc ky truoc khi trich dan
------------------------------------
Day KHONG phai phan bac §IV-A. §IV-A tron o muc DOAN (thay tron ven tung doan trong mot
tai lieu k=20) -- o do pi-hat affine theo dung dinh nghia. Bang duoi tron o muc CAU
BEN TRONG mot doan, tuc k=1 va nam NGOAI pham vi ma bai tuyen bo. Vi vay ket qua o day
la SO DO cho han che da neu o §VI, khong phai mau thuan voi bat ky claim nao.

Hai dieu can tach bach:
  - DO TAN MAC giam theo 1/sqrt(k) neu gop nhieu doan -> co the chua bang tai lieu dai.
  - THIEN LECH khong giam theo k. Neu pi-hat doc thap 0,12 o ti le that 0,25 thi gop
    bao nhieu doan cung van thap 0,12.

DAU RA
------
  results/s8_submix_pihat.csv            theo bin ti le
  results/s8_submix_by_checkpoint.csv    theo bin x checkpoint
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "results"
MIX = ROOT / "artifacts" / "pred_mix" / "scores_all.parquet"
BINS = np.arange(0, 1.01, 0.1)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    anchors = pd.read_csv(ROOT / "output" / "pihat_sanity_16models.csv").set_index("model")

    d = pd.read_parquet(MIX, columns=["paper_id", "artifact", "mix_ratio_actual",
                                      "mix_structure", "n_slots", "score", "ai_degree"])
    d["ck"] = d.artifact.str.replace("artifacts_", "", regex=False)
    miss = set(d.ck.unique()) - set(anchors.index)
    if miss:
        raise SystemExit("checkpoint khong co neo mu0/mu1: %s" % sorted(miss))

    # neo mu0/mu1 uoc CHI tu val -- y het paper_level_pihat.py, khong dung du lieu mix
    d["pi"] = ((d.score - d.ck.map(anchors.mu0))
               / (d.ck.map(anchors.mu1) - d.ck.map(anchors.mu0)))
    d["bin"] = pd.cut(d.mix_ratio_actual, BINS, include_lowest=True)

    def summarize(g):
        return pd.Series(dict(
            n=len(g), true_mean=g.mix_ratio_actual.mean(),
            pi_mean=g.pi.mean(), pi_median=g.pi.median(), pi_sd=g.pi.std(),
            bias=g.pi.mean() - g.mix_ratio_actual.mean(),
            mae=(g.pi - g.mix_ratio_actual).abs().mean()))

    B = d.groupby("bin", observed=True).apply(summarize, include_groups=False).reset_index()
    B.to_csv(RES / "s8_submix_pihat.csv", index=False)

    C = (d.groupby(["ck", "bin"], observed=True)
           .apply(summarize, include_groups=False).reset_index())
    C.to_csv(RES / "s8_submix_by_checkpoint.csv", index=False)

    print("=" * 92)
    print("S8  pi-hat tren doan tron o muc CAU  (%d dong, %d bai, %d checkpoint)"
          % (len(d), d.paper_id.nunique(), d.ck.nunique()))
    print("=" * 92)
    print("%-16s %8s %9s %9s %9s %8s %8s"
          % ("ti le that", "n", "TB that", "pi TB", "pi trung vi", "lech", "SD"))
    for _, r in B.iterrows():
        print("%-16s %8d %9.3f %9.3f %11.3f %8.3f %8.3f"
              % (str(r["bin"]), r.n, r.true_mean, r.pi_mean, r.pi_median, r.bias, r.pi_sd))

    mid = B[(B.true_mean > 0.1) & (B.true_mean < 0.55)]
    print("\nThien lech lon nhat o giua dai: %.3f (tai ti le that %.3f)"
          % (mid.bias.min(), mid.loc[mid.bias.idxmin(), "true_mean"]))
    print("MAE gop tren moi ti le  : %.3f" % (d.pi - d.mix_ratio_actual).abs().mean())
    print("Tuong quan Pearson      : %.3f" % d.pi.corr(d.mix_ratio_actual))
    print("\nDoi chieu: §IV-A (tron o muc DOAN, k=20, tong hop) bao MAE = 0,039.")
    print("Chenh lech nay KHONG phai mau thuan -- hai thiet ke tron o hai muc khac nhau.")

    # thien lech co on dinh qua checkpoint khong?
    piv = C.pivot_table(index="bin", columns="ck", values="bias", observed=True)
    print("\nThien lech theo checkpoint (min/max qua 16 ckpt) tai moi bin:")
    for b in piv.index:
        print("  %-16s  %+.3f .. %+.3f" % (str(b), piv.loc[b].min(), piv.loc[b].max()))

    # tach theo cach dat cau AI
    print("\nTheo cach dat cau AI:")
    for st, g in d.groupby("mix_structure"):
        m = g[(g.mix_ratio_actual > 0.2) & (g.mix_ratio_actual < 0.3)]
        print("  %-12s lech tai ti le 0,2-0,3 = %+.3f (n=%d)"
              % (st, (m.pi - m.mix_ratio_actual).mean(), len(m)))
    print("\n-> results/s8_submix_pihat.csv, s8_submix_by_checkpoint.csv")


if __name__ == "__main__":
    main()
