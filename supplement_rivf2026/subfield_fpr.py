"""FPR muc tai lieu theo TIEU LINH VUC arXiv -- Revision Brief muc 4, dong cuoi.

    python supplement_rivf2026/subfield_fpr.py

VI SAO CAN
----------
§VII cua bai hien chi neu HAI diem cuc tri (cs.CV 3,90%, cs.DS 23,53%) va tu ghi chu
rang chung "duoc do TRUOC khi chuyen sang khung pi-hat". Nghia la con so dang nam
trong bai KHONG phai FPR cua phuong phap dang de xuat -- no la FPR cua mot quy tac cu.
Brief doi bang day du. Thay vi chep lai so cu, script nay do lai TAT CA tieu linh vuc
duoi DUNG quy tac trien khai cua bai:

    z_i = Phi^-1(1 - p_i),  p_i doi voi pool human cua `val`   (giong het §V-D)
    Z_k = sum(z_i)/sqrt(k)  voi k = so doan THAT cua bai do
    gan co khi  Z_k > z_{1-alpha} * sqrt(1 + (k-1)*rho),  rho rieng tung checkpoint

De so sanh, cung do lai quy tac NGUONG TINH tren thang tho (tau chot o k=22,6, tuc cot
"k~22.6" cua Table II) tren cung nhung bai do. Chenh lech giua hai cot cho biet phep
bien doi co lam giam thien lech theo tieu linh vuc hay khong.

LUU Y VE CO MAU: nhieu tieu linh vuc chi co vai chuc bai, nen FPR cua chung co sai so
nhi thuc rat lon. Cot ci_lo/ci_hi la khoang Wilson 95% -- doc no truoc khi ket luan bat
ky "thien lech theo linh vuc" nao.

DAU RA
------
  results/s7_subfield_fpr.csv        moi tieu linh vuc x moi phuong phap
  results/s7_subfield_summary.csv    gop qua 16 checkpoint, kem khoang Wilson
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "artifacts" / "kaggle"
FP_FULL = ROOT / "artifacts" / "kaggle_testfp_full" / "pred_testfp_full"
OUT = ROOT / "output"
RES = HERE / "results"

ALPHA = 0.01
Z1A = float(norm.isf(ALPHA))
MIN_N = 20          # duoi nguong nay thi bao cao nhung danh dau la khong dien giai duoc


def wilson(k, n, z=1.96):
    """Khoang tin cay Wilson -- dung cho ty le nho tren n nho, khong dung Wald."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    RES.mkdir(parents=True, exist_ok=True)

    # --- tieu linh vuc chinh cua tung bai (danh muc arXiv dau tien) ---
    meta = pd.read_parquet(OUT / "dataset_test_fp_full_deconf2.parquet",
                           columns=["paper_id", "categories"])
    cat = (meta.groupby("paper_id").categories.first().astype(str)
               .str.split().str[0].str.strip("[]'\""))

    rho_by_model = (pd.read_csv(OUT / "tau_k_mean_16models.csv")
                      .groupby("model")["rho"].median())
    tau_mean_k = (pd.read_csv(OUT / "mean_ktransfer_16models.csv")
                    .set_index("model")["tau_mean_k"])

    rows = []
    for f in sorted(glob.glob(str(FP_FULL / "pred_test_fp_full_artifacts_*.parquet"))):
        model = Path(f).stem.replace("pred_test_fp_full_artifacts_", "")
        rho = float(rho_by_model.loc[model])

        val = pd.read_parquet(ART / ("artifacts_%s" % model) / "pred_val.parquet")
        pool = np.sort(val.loc[val.label == 0, "score"].to_numpy())
        m = len(pool)

        d = pd.read_parquet(f).dropna(subset=["score"])
        pv = (0.5 + m - np.searchsorted(pool, d.score.to_numpy(), side="left")) / (m + 1.0)
        d = d.assign(z=norm.isf(pv))

        g = d.groupby("paper_id")
        k = g.size()
        Zk = g["z"].sum() / np.sqrt(k)
        mean_raw = g["score"].mean()

        flag_z = (Zk > Z1A * np.sqrt(1 + (k - 1) * rho))
        flag_static = (mean_raw >= float(tau_mean_k.loc[model]))

        sub = cat.reindex(k.index)
        for name, fl in [("z-scale (proposed)", flag_z), ("static (raw scale)", flag_static)]:
            t = pd.DataFrame({"subfield": sub.to_numpy(), "flag": fl.to_numpy(),
                              "k": k.to_numpy()})
            for s, gg in t.groupby("subfield"):
                rows.append(dict(model=model, method=name, subfield=s,
                                 n_papers=len(gg), n_flagged=int(gg.flag.sum()),
                                 fpr=float(gg.flag.mean()), k_median=float(gg.k.median())))
        print("  xong %s" % model)

    D = pd.DataFrame(rows)
    D.to_csv(RES / "s7_subfield_fpr.csv", index=False)

    # --- gop qua 16 checkpoint: cong don co gan co roi tinh Wilson tren tong ---
    out = []
    for (meth, s), gg in D.groupby(["method", "subfield"]):
        n_tot, k_tot = int(gg.n_papers.sum()), int(gg.n_flagged.sum())
        lo, hi = wilson(k_tot, n_tot)
        out.append(dict(method=meth, subfield=s,
                        n_papers=int(gg.n_papers.iloc[0]), n_ckpt=gg.model.nunique(),
                        k_median=float(gg.k_median.iloc[0]),
                        fpr_median_over_ckpt=100 * gg.fpr.median(),
                        fpr_min=100 * gg.fpr.min(), fpr_max=100 * gg.fpr.max(),
                        fpr_pooled=100 * k_tot / n_tot,
                        ci_lo=100 * lo, ci_hi=100 * hi,
                        interpretable=int(gg.n_papers.iloc[0]) >= MIN_N))
    S = pd.DataFrame(out).sort_values(["method", "n_papers"], ascending=[True, False])
    S.to_csv(RES / "s7_subfield_summary.csv", index=False)

    for meth in sorted(set(S.method)):
        s = S[S.method == meth]
        print("\n" + "=" * 96)
        print("FPR muc tai lieu theo tieu linh vuc -- %s" % meth)
        print("=" * 96)
        print("%-12s %7s %7s %9s %22s %s"
              % ("subfield", "n_bai", "k_tv", "FPR gop", "Wilson 95%", "dien giai duoc?"))
        for _, r in s.iterrows():
            print("%-12s %7d %7.0f %8.2f%%   [%6.2f%%, %6.2f%%]   %s"
                  % (r.subfield, r.n_papers, r.k_median, r.fpr_pooled,
                     r.ci_lo, r.ci_hi, "co" if r.interpretable else "KHONG (n<%d)" % MIN_N))
        ok = s[s.interpretable]
        print("  -> chi tinh cac tieu linh vuc n>=%d (%d/%d): FPR tu %.2f%% den %.2f%%"
              % (MIN_N, len(ok), len(s), ok.fpr_pooled.min(), ok.fpr_pooled.max()))
        print("     bien do = %.1f diem %%" % (ok.fpr_pooled.max() - ok.fpr_pooled.min()))
    print("\n-> supplement_rivf2026/results/s7_subfield_fpr.csv, s7_subfield_summary.csv")


if __name__ == "__main__":
    main()
