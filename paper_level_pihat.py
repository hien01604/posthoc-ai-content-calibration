"""pi_hat = (diem trung binh doan - mu0) / (mu1 - mu0) -- uoc luong % AI truc tiep tren
thang diem tho (khong qua mid-p/z-hoa), cong voi khoang tin cay hieu chinh tuong quan
(ICC/design effect chuan, ap dung thang tho -- khong phai ban Hartung tren z-score).

    python3 paper_level_pihat.py

VI SAO SCRIPT NAY CAN THIET
---------------------------
paper_level_tau_k.py (va paper_level_hartung_ood.py) tinh tau(k)/Z_k tren THANG Z (mid-p
roi doi sang z-score). pi_hat lai la ham TUYEN TINH cua trung binh diem THO. Hai thang do
khong phai bien doi tuyen tinh cua nhau (mid-p la bien doi theo RANK, khong phai theo gia
tri), nen ket qua "generalize tot" cua Z_k KHONG tu dong dung cho pi_hat. Script nay tinh
lai moi thu tren dung thang tho ma pi_hat dang song, dung ICC chuan (Var trong-nhom / Var
tong) thay vi uoc luong rieng cua Hartung (vi khong con gia dinh z_i ~ N(0,1)).

CONG THUC
---------
  mu0, mu1        = trung binh diem doan human / AI that (uoc tu val hoac test_ood_model)
  rho_raw          = 1 - E[phuong sai TRONG mot bai] / Var_TONG(diem doan human)
                     (ICC chuan -- Var_TONG do tren pool nhieu bai, KHONG phai 1 bai)
  Var(S_k)         = Var_TONG/k * (1 + (k-1)*rho_raw)      design effect tren thang tho
  can_tin_cay(k)   = z_{1-alpha} * sqrt(Var(S_k)) / (mu1-mu0)   do rong CI cho pi_hat
  quyet dinh       : pi_hat > can_tin_cay(k)  <=>  can duoi CI vuot 0 (loai duoc gia thuyet
                     "tai lieu nay 100% human" o do tin cay 1-alpha)

DAU VAO
-------
  artifacts/kaggle/<artifact>/pred_val.parquet                        uoc mu0 (rieng cho tung ckpt)
  artifacts/kaggle/<artifact>/pred_test_ood_model.parquet              uoc mu0, mu1, VA la tap
                                                                        danh gia ngoai-mau (k~2,38)
  artifacts/kaggle_testfp_full/pred_testfp_full/pred_test_fp_full_*.parquet
                                                                        population co kiem soat
                                                                        (576 bai >=20 doan, giong Hinh 1)

DAU RA (output/)
----------------
  pihat_static_vs_corrected_by_k.csv   FPR (ty le loai-sai-0) theo k, tinh tinh vs hieu chinh,
                                        tren population co kiem soat -- Table II moi
  pihat_ood_16models.csv               pi_hat + FPR/bo lot tren test_ood_model, dung ngoai mau
                                        rho_raw da uoc tu test_fp_full -- Table IV moi
  pihat_sanity_16models.csv            pi_hat trung binh tren bai human that (ky vong 0) va
                                        bai AI that (ky vong 1) -- kiem tinh dung dan cua pi_hat
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from icc import icc1

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts" / "kaggle"
FP_FULL = BASE / "artifacts" / "kaggle_testfp_full" / "pred_testfp_full"
OUT = BASE / "output"
ALPHA = 0.01
KMAX = 20
K0_RHO = 20          # so doan/bai dung de uoc rho_raw (nhom bang nhau)
R = 20               # so lan lay mau con (population co kiem soat)
S = 5                # so lan chia cal/eval
SEED = 0
from scipy.stats import norm
Z1A = float(norm.isf(ALPHA))
KS_SHOW = [1, 2, 3, 5, 8, 10, 15, 20]


def artifacts_ood():
    for f in sorted(glob.glob(str(ART / "artifacts_*" / "pred_test_ood_model.parquet"))):
        yield Path(f).parent.name.replace("artifacts_", ""), f


def sanity_check(model, ood, mu0, mu1):
    """pi_hat trung binh tren bai human that (ky vong 0) va bai AI that (ky vong 1).

    mu0/mu1 PHAI duoc uoc tu mot split KHAC ood (vd. val) va truyen vao tu ngoai --
    neu uoc ngay tren chinh ood roi danh gia lai tren ood thi ket qua gan nhu dam bao
    theo construction (mu0 la trung binh co trong so cua chinh cac diem dang duoc so
    sanh voi no). Xem paper_level_pihat.py.bak_before_leakfix cho ban cu (leaky)."""
    g = ood.groupby(["paper_id", "label"])["score"].mean().reset_index()
    g["pi_hat"] = (g["score"] - mu0) / (mu1 - mu0)
    row = dict(model=model, mu0=mu0, mu1=mu1, sep=mu1 - mu0)
    for lbl, tag in ((0, "human"), (1, "ai")):
        v = g.loc[g.label == lbl, "pi_hat"]
        row[f"pihat_mean_{tag}"] = float(v.mean())
        row[f"pihat_median_{tag}"] = float(v.median())
    return row


def rho_raw_and_var(papers_by_id, mu0, var_total, rng):
    """rho_raw = ICC(1) mot chieu tren K0_RHO doan dau moi bai -- xem icc.py."""
    return icc1(papers_by_id, K0_RHO)


def controlled_curve(model, mu0, mu1, var_total):
    """Table II moi: FPR(ty le loai-sai-0) theo k, pi_hat CO HIEU CHINH tren thang tho,
    cung population 576 bai >=20 doan nhu Hinh 1.

    Cot 'static' (tau tinh chot o k=KMAX) KHONG con tinh o day. Truoc kia ham nay tu
    rut mau rieng (SEED=0, R=20) de uoc no, SONG SONG voi paper_level_tau_k.py cung tinh
    dai luong nay tren CUNG population nhung SEED=42, R=200 -- ra hai con so khac nhau
    cho cung mot thu (vd. k=1: 4,76% vs 4,19%). Xem ghi chu duoi Table A2 trong draft:
    "khi viet ban chinh thuc, chi chay mot lan". Fix: static lay TU
    tau_k_mean_16models.csv (paper_level_tau_k.py, R=200 on dinh hon), merge o main().
    """
    f = FP_FULL / f"pred_test_fp_full_artifacts_{model}.parquet"
    df = pd.read_parquet(f)
    by_paper = {p: g.score.to_numpy() for p, g in df.groupby("paper_id")}
    ids = np.array(sorted(p for p, v in by_paper.items() if len(v) >= KMAX))
    papers = [by_paper[p] for p in ids]
    n = len(papers)
    L = max(len(a) for a in papers)
    pad = np.full((n, L), np.nan)
    for i, a in enumerate(papers):
        pad[i, :len(a)] = a
    invalid = np.isnan(pad)

    rng = np.random.default_rng(SEED)
    rows_by_k = {k: {"corrected": []} for k in KS_SHOW}
    for sp in range(S):
        perm = rng.permutation(n)
        cal, ev = perm[:n // 2], perm[n // 2:]
        rho = rho_raw_and_var([papers[i] for i in cal], mu0, var_total, rng)

        for _ in range(R):
            key = rng.random((n, L))
            key[invalid] = np.inf
            idx = np.argsort(key, axis=1)[:, :KMAX]
            M = np.take_along_axis(pad, idx, axis=1)          # (n, KMAX) diem tho
            Craw = np.cumsum(M, axis=1) / np.arange(1, KMAX + 1)   # mean chay theo k
            pihat = (Craw - mu0) / (mu1 - mu0)                # (n, KMAX)

            for k in KS_SHOW:
                j = k - 1
                # corrected: bien do CI theo cong thuc design-effect, dung rho da uoc
                half_width = Z1A * np.sqrt(var_total / k * (1 + (k - 1) * rho)) / (mu1 - mu0)
                fpr_corr = float((pihat[ev, j] >= half_width).mean())
                rows_by_k[k]["corrected"].append(fpr_corr)

    return rows_by_k


def static_fpr_from_tau_k():
    """Doc cot fpr_const (tau tinh chot o k=KMAX) da tinh san boi paper_level_tau_k.py --
    NGUON DUY NHAT cho 'Ngưỡng tĩnh', dung chung cho ca Table II/III (script nay) va
    Table IV (paper_level_static_vs_hartung.py), thay vi moi ben tu rut mau rieng."""
    p = OUT / "tau_k_mean_16models.csv"
    if not p.exists():
        raise SystemExit(
            f"Khong thay {p} -- chay `python3 paper_level_tau_k.py` truoc (day la nguon "
            f"duy nhat cho cot 'Ngưỡng tĩnh', xem ghi chu trong controlled_curve()).")
    T = pd.read_csv(p)
    return T.groupby(["model", "k"])["fpr_const"].mean()   # trung binh qua 5 lan chia cal/eval


def main():
    OUT.mkdir(exist_ok=True)
    static_fpr = static_fpr_from_tau_k()
    sanity_rows, curve_rows, ood_rows = [], [], []

    for model, f_ood in artifacts_ood():
        ood = pd.read_parquet(f_ood)

        # mu0/mu1 UOC TU VAL, khong tu ood/fp -- val la split doc lap voi ca test_ood_model
        # lan test_fp_full (xem SPLITS trong build_testfp_full.py), nen day la neo "khong
        # nhin thay du lieu dang danh gia", tranh leakage da phat hien (mu0 tu chinh tap
        # dang do pi_hat tren no).
        val = pd.read_parquet(ART / f"artifacts_{model}" / "pred_val.parquet")
        mu0_val = float(val.loc[val.label == 0, "score"].mean())
        mu1_val = float(val.loc[val.label == 1, "score"].mean())

        srow = sanity_check(model, ood, mu0_val, mu1_val)
        sanity_rows.append(srow)

        # var_total van uoc TREN test_fp_full (can phuong sai cua CHINH population dang
        # subsample de tinh design effect) -- chi mu0/mu1 (neo) doi sang val.
        fp = pd.read_parquet(FP_FULL / f"pred_test_fp_full_artifacts_{model}.parquet")
        var_fp = float(fp.score.var(ddof=1))

        curve = controlled_curve(model, mu0_val, mu1_val, var_fp)
        for k, d in curve.items():
            curve_rows.append(dict(model=model, k=k,
                                    fpr_static=float(static_fpr.loc[(model, k)]),
                                    fpr_corrected=float(np.mean(d["corrected"]))))

        # --- ngoai mau: rho uoc mot lan tren nua CAL cua test_fp_full (k=1..20), mu0/mu1
        # TU VAL (khong tu test_ood_model dang danh gia, khong tu test_fp_full dung de uoc
        # rho) -- neo va tap danh gia hoan toan tach biet o ca ba nguon dữ liệu. ---
        rng = np.random.default_rng(SEED)
        by_paper_fp = {p: g.score.to_numpy() for p, g in fp.groupby("paper_id")}
        ids_fp = [p for p, v in by_paper_fp.items() if len(v) >= K0_RHO]
        cal_ids = rng.choice(ids_fp, size=len(ids_fp) // 2, replace=False)
        rho = rho_raw_and_var([by_paper_fp[p] for p in cal_ids], mu0_val, var_fp, rng)
        var_ood = float(ood.loc[ood.label == 0, "score"].var(ddof=1))

        g = ood.groupby(["paper_id", "label"])["score"]
        k_ood = g.size().to_numpy()
        mean_ood = g.mean().to_numpy()
        lbl = g.size().index.get_level_values("label").to_numpy()
        pihat_ood = (mean_ood - mu0_val) / (mu1_val - mu0_val)
        half = Z1A * np.sqrt(var_ood / k_ood * (1 + (k_ood - 1) * rho)) / (mu1_val - mu0_val)
        flagged = pihat_ood >= half
        human, ai = lbl == 0, lbl == 1
        ood_rows.append(dict(
            model=model, rho=rho, k_ood_mean=float(k_ood.mean()),
            fpr=float(flagged[human].mean()), miss=float((~flagged[ai]).mean()),
            pihat_mean_human=float(pihat_ood[human].mean()),
            pihat_mean_ai=float(pihat_ood[ai].mean()),
        ))
        print(f"  xong {model}")

    pd.DataFrame(sanity_rows).to_csv(OUT / "pihat_sanity_16models.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUT / "pihat_static_vs_corrected_by_k.csv", index=False)
    pd.DataFrame(ood_rows).to_csv(OUT / "pihat_ood_16models.csv", index=False)

    C = pd.DataFrame(curve_rows).groupby("k").agg(
        static_median=("fpr_static", "median"), static_min=("fpr_static", "min"), static_max=("fpr_static", "max"),
        corr_median=("fpr_corrected", "median"), corr_min=("fpr_corrected", "min"), corr_max=("fpr_corrected", "max"),
    ).reset_index()
    print("\n=== Table II moi: pi_hat tinh vs hieu chinh, theo k ===")
    for _, r in C.iterrows():
        print(f"k={int(r.k):>3}  tinh {r.static_median*100:5.2f}% [{r.static_min*100:.2f}-{r.static_max*100:.2f}]   "
              f"hieu chinh {r.corr_median*100:5.2f}% [{r.corr_min*100:.2f}-{r.corr_max*100:.2f}]")

    O = pd.DataFrame(ood_rows)
    print("\n=== Table IV moi: pi_hat ngoai mau tren test_ood_model ===")
    print(f"rho: trung vi {O.rho.median():.3f}  dai [{O.rho.min():.3f}-{O.rho.max():.3f}]")
    print(f"FPR: trung vi {O.fpr.median()*100:.2f}%  [{O.fpr.min()*100:.2f}-{O.fpr.max()*100:.2f}]")
    print(f"bo lot: trung vi {O.miss.median()*100:.2f}%  [{O.miss.min()*100:.2f}-{O.miss.max()*100:.2f}]")
    print(f"pi_hat TB (human): {O.pihat_mean_human.mean():.3f}   pi_hat TB (AI): {O.pihat_mean_ai.mean():.3f}")

    print("\n-> output/pihat_sanity_16models.csv, pihat_static_vs_corrected_by_k.csv, pihat_ood_16models.csv")


if __name__ == "__main__":
    main()
