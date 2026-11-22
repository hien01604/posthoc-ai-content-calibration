"""S9 -- cac bang tu REPORT_2026-08-15.md, DA VERIFY lai tung so tu CSV.

    python supplement_rivf2026/report_tables.py

VI SAO CO MUC NAY
-----------------
`REPORT_2026-08-15.md` la nhat ky cua mot khung bai CU (chan doan lech DON VI, roi chua
bang Sidak/Bonferroni tren luat `any`). Bai hien tai da chuyen sang khung khac (pi-hat
+ Z_k tren luat `mean`). Nhung so lieu cua no VAN DUNG va van tra loi duoc nhung cau
reviewer chac chan se hoi, nen giu lai o supplement thay vi vut di.

CANH BAO PHAI DOC TRUOC KHI TRICH
---------------------------------
Bang o day do luat gan co **`any`** (gan co bai neu CO doan nao vuot nguong). Table III
cua bai do luat **`mean`** (so trung binh diem cua bai voi mot nguong). HAI LUAT KHAC
NHAU -- khong duoc dat so cua bang nay canh so cua Table III nhu the chung so sanh duoc.

Vi du de thay: cung `test_fp_full`, cung k=20
  luat `any`  -> 10,11% oan   (bang S9.2 duoi day)
  luat `mean` ->  0,71% oan   (cot Static cua Table III)
Chenh 14 lan, va ca hai deu dung -- vi chung tra loi hai cau hoi khac nhau.

DAU RA
------
  results/s9_unit_mismatch.csv     chan doan 16 checkpoint (doan vs bai, luat any)
  results/s9_k_curve_any.csv       duong cong co kiem soat theo k, luat any
  results/s9_pool_size.csv         FPR theo co tap hieu chuan + san ly thuyet
  results/s9_tpr_cost.csv          gia phai tra khi doi diem van hanh
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "output"
RES = HERE / "results"
TARGET = 0.01


def main():
    RES.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------- S9.1 lech don vi (luat any)
    D = pd.read_csv(OUT / "diagnosis_16models.csv")
    U = D[["model", "fpr_chunk", "fpr_paper", "fpr_if_independent",
           "formula_error", "ratio", "k_mean", "n_papers"]].copy()
    U = U.sort_values("fpr_paper")
    U.to_csv(RES / "s9_unit_mismatch.csv", index=False)

    # ---------------------------------------------------- S9.2 duong cong theo k
    C = pd.read_csv(OUT / "testfp_full_curve.csv").rename(
        columns={"n_bài": "n_papers", "oan_any_TB": "fpr_any", "oan_any_std": "fpr_any_sd"})
    C.to_csv(RES / "s9_k_curve_any.csv", index=False)

    # ---------------------------------------------------- S9.3 co tap hieu chuan
    S = pd.read_csv(OUT / "sidak_calibration_size.csv")
    rows = []
    for col, lab, n, pipe in [
            ("val_438", "val", 438, "different from deployment"),
            ("cal_438", "cal subsample", 438, "same as deployment"),
            ("cal_2000", "cal subsample", 2000, "same as deployment"),
            ("cal_full", "cal full", int(S.n_cal_full.median()), "same as deployment")]:
        rows.append(dict(pool=lab, n_chunks=n, pipeline=pipe,
                         fpr_median=100 * S[col].median(),
                         fpr_min=100 * S[col].min(), fpr_max=100 * S[col].max(),
                         spread=100 * (S[col].max() - S[col].min())))
    P = pd.DataFrame(rows)
    # san ly thuyet: Sidak doi phan vi 1 - alpha_k, alpha_k = 1-(1-alpha)^(1/k)
    for k in [19, 20, 22, 23]:
        a = 1 - (1 - TARGET) ** (1.0 / k)
        P.loc[len(P)] = dict(pool="theoretical floor (k=%d)" % k, n_chunks=int(np.ceil(1 / a)),
                             pipeline="quantile %.6f" % (1 - a), fpr_median=np.nan,
                             fpr_min=np.nan, fpr_max=np.nan, spread=np.nan)
    P.to_csv(RES / "s9_pool_size.csv", index=False)

    # ---------------------------------------------------- S9.4 gia phai tra
    T = pd.read_csv(OUT / "tpr_cost_16models.csv")
    T2 = T.copy()
    T2["loss_points"] = 100 * (T2.tpr_at_1pct - T2.tpr_at_sidak)
    T2 = T2.sort_values("loss_points", ascending=False)
    T2.to_csv(RES / "s9_tpr_cost.csv", index=False)

    # ---------------------------------------------------------------- bao cao
    print("=" * 90)
    print("S9.1  LECH DON VI (luat `any`) -- 16 checkpoint tren test_fp_full")
    print("=" * 90)
    print("  FPR muc doan : %.2f%% - %.2f%%  (trung vi %.2f%%)"
          % (100 * U.fpr_chunk.min(), 100 * U.fpr_chunk.max(), 100 * U.fpr_chunk.median()))
    print("  FPR muc bai  : %.2f%% - %.2f%%  (trung vi %.2f%%)"
          % (100 * U.fpr_paper.min(), 100 * U.fpr_paper.max(), 100 * U.fpr_paper.median()))
    print("  ty so        : %.1fx - %.1fx    (trung vi %.1fx)  -- %d/%d checkpoint, khong ngoai le"
          % (U.ratio.min(), U.ratio.max(), U.ratio.median(), (U.ratio > 1).sum(), len(U)))
    print("  xap xi doc lap vong len %+.0f%% .. %+.0f%% khi dung de DU DOAN"
          % (100 * U.formula_error.min(), 100 * U.formula_error.max()))

    print("\n" + "=" * 90)
    print("S9.2  DUONG CONG CO KIEM SOAT (cung 576 bai, chi doi so doan nhin vao)")
    print("=" * 90)
    print("%4s %8s %10s" % ("k", "n bai", "FPR (any)"))
    for _, r in C.iterrows():
        print("%4d %8d %9.2f%%" % (r.k, r.n_papers, 100 * r.fpr_any))
    print("  -> tang don dieu, khong bac nhay. k=3 (co dataset cu) %.2f%% vs k=20 (bai that) %.2f%%"
          % (100 * C[C.k == 3].fpr_any.iloc[0], 100 * C[C.k == 20].fpr_any.iloc[0]))

    print("\n" + "=" * 90)
    print("S9.3  CO TAP HIEU CHUAN quyet dinh Sidak co chay duoc khong")
    print("=" * 90)
    for _, r in P.iterrows():
        if np.isnan(r.fpr_median):
            print("  %-26s can >= %5d manh  (%s)" % (r["pool"], r.n_chunks, r.pipeline))
        else:
            print("  %-14s n=%-6d %-26s FPR %.2f%%  [%.2f, %.2f]  bien do %.2f"
                  % (r["pool"], r.n_chunks, r.pipeline, r.fpr_median, r.fpr_min,
                     r.fpr_max, r.spread))
    print("  -> Cung phan phoi ma chi 438 manh van hong y het (3,90%% vs 3,98%%): la CO MAU.")
    print("  -> Lech phan phoi gay benh KHAC: bien do 9,29 diem thay vi 0,66 diem.")

    print("\n" + "=" * 90)
    print("S9.4  GIA PHAI TRA khi doi diem van hanh (TPR muc DOAN)")
    print("=" * 90)
    for c, lab in [("tpr_at_1pct", "alpha = 1%"), ("tpr_at_01pct", "alpha = 0,1%"),
                   ("tpr_at_sidak", "alpha Sidak = %.4f%%" % (100 * T.alpha_sidak.median()))]:
        print("  %-24s %.2f%% - %.2f%%   trung vi %.2f%%"
              % (lab, 100 * T[c].min(), 100 * T[c].max(), 100 * T[c].median()))
    w = T2.iloc[0]
    print("  mat trung vi %.2f diem | ca xau nhat %s: %.2f%% -> %.2f%% (mat %.1f diem)"
          % (100 * (T.tpr_at_1pct.median() - T.tpr_at_sidak.median()),
             w.model, 100 * w.tpr_at_1pct, 100 * w.tpr_at_sidak, w.loss_points))
    print("\n-> results/s9_*.csv")


def construction_recipe():
    """Cong thuc dung test_fp_full -- doc tu manifest, khong go tay."""
    import json
    m = json.load(open(OUT / "testfp_full_manifest.json", encoding="utf-8"))
    cp = m["chunks_per_paper"]
    rows = [
        ("source corpus", str(m["built_from"])),
        ("chunker", str(m["chunker"])),
        ("target words / chunk", str(m["chunker_params"]["target_words"])),
        ("min words / chunk", str(m["chunker_params"]["min_words"])),
        ("max words / chunk", str(m["chunker_params"]["max_words"])),
        ("median words / chunk", str(m["words_per_chunk_median"])),
        ("drop first chunk", str(m["drop_first_chunk"])),
        ("canonicalized", str(m["canonicalized"])),
        ("paper_ids excluded from", ", ".join(m["excluded_paper_ids_from"])),
        ("published year range", "%d-%d" % tuple(m["published_year_range"])),
        ("n papers", str(m["n_papers"])),
        ("n chunks", str(m["n_chunks"])),
        ("chunks/paper min", str(cp["min"])),
        ("chunks/paper p25", str(cp["p25"])),
        ("chunks/paper median", str(cp["median"])),
        ("chunks/paper p75", str(cp["p75"])),
        ("chunks/paper mean", str(cp["mean"])),
        ("chunks/paper max", str(cp["max"])),
        ("label", str(m["label"])),
    ]
    R = pd.DataFrame(rows, columns=["property", "value"])
    R.to_csv(RES / "s2f_construction_recipe.csv", index=False)
    print("\n" + "=" * 90)
    print("S2f  CONG THUC DUNG test_fp_full (tu output/testfp_full_manifest.json)")
    print("=" * 90)
    print(R.to_string(index=False))
    return R


if __name__ == "__main__":
    main()
    construction_recipe()
