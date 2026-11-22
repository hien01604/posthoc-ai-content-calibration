"""Sinh SUPPLEMENT.md tu cac CSV da tao -- KHONG go tay so nao.

    python supplement_rivf2026/render_supplement.py

Chay SAU build_supplement.py va subfield_fpr.py. Moi bang trong SUPPLEMENT.md deu
duoc render tu results/*.csv, nen khong the lech voi bai (yeu cau so 7 cua brief).
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DOC = HERE / "SUPPLEMENT.md"


def md(df, cols=None, headers=None, fmt=None):
    d = df[cols] if cols else df
    h = headers or list(d.columns)
    out = ["| " + " | ".join(h) + " |", "|" + "|".join(["---"] * len(h)) + "|"]
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v = r[c]
            f = (fmt or {}).get(c)
            cells.append(f(v) if f else ("" if pd.isna(v) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def p2(x):
    return "" if pd.isna(x) else "%.2f" % x


def p3(x):
    return "" if pd.isna(x) else "%.3f" % x


def p4(x):
    return "" if pd.isna(x) else "%.4f" % x


def pct2(x):
    return "" if pd.isna(x) else "%.2f" % (100 * x)


def main():
    S1 = pd.read_csv(RES / "s1_checkpoints.csv")
    S1b = pd.read_csv(RES / "s1b_document_results.csv")
    S2b = pd.read_csv(RES / "s2b_splits.csv")
    S2c = pd.read_csv(RES / "s2c_generators.csv")
    S2a = pd.read_csv(RES / "s2a_schema.csv")
    S4 = pd.read_csv(RES / "s4_ranges_main_tables.csv")
    S6p = pd.read_csv(RES / "s6_calibration_pool.csv")
    S6m = pd.read_csv(RES / "s6_multiple_testing.csv")
    S7 = pd.read_csv(RES / "s7_subfield_summary.csv")
    S2d = pd.read_csv(RES / "s2d_generator_models.csv")
    S2e = pd.read_csv(RES / "s2e_generation_config.csv")
    S8 = pd.read_csv(RES / "s8_submix_pihat.csv")
    S2f = pd.read_csv(RES / "s2f_construction_recipe.csv")
    S9u = pd.read_csv(RES / "s9_unit_mismatch.csv")
    S9k = pd.read_csv(RES / "s9_k_curve_any.csv")
    S9p = pd.read_csv(RES / "s9_pool_size.csv")
    S9t = pd.read_csv(RES / "s9_tpr_cost.csv")

    L = []
    A = L.append

    A("# Supplementary Material\n")
    A("**Estimating the AI-Generated Content Fraction of a Document: "
      "Length-Robust Document-Level Calibration via Distributional Transformation**\n")
    A("> Moi bang duoi day duoc **sinh tu dong** tu `supplement_rivf2026/results/*.csv` "
      "bang `render_supplement.py`. Khong con so nao duoc go tay, nen chung khong the "
      "lech voi bang trong bai bao.\n")
    A("Tai lieu nay chua nhung bang ma ban chinh (gioi han 6 trang IEEE) khong du cho. "
      "Moi muc ghi ro no tra loi phan nao cua bai.\n")
    A("---\n")

    # ---------------------------------------------------------------- S1
    A("## S1. Muoi sau checkpoint\n")
    A("### S1a. Hieu nang muc doan va tham so hieu chuan\n")
    A("Bo sung cho **§III-B**, cho hien tung checkpoint dang sau cac dai gop nhu "
      "\"AUROC 0.9746--0.9875\".\n")
    A("`rho` la uoc luong Hartung tren thang z. Bai bao cao **trung binh qua 5 lan chia** "
      "(cot `rho (mean)`; trung vi qua 16 checkpoint = 0.303). Code trien khai "
      "`paper_level_hartung_ood.py` lai dung **trung vi qua 5 lan chia** (cot "
      "`rho (median)`); chenh lech khong dang ke nhung duoc ghi ra day cho minh bach.\n")
    d = S1.copy()
    d["seed"] = d.seed.astype(str)
    A(md(d, ["id", "architecture", "base_model", "seed", "type", "auroc_ood",
             "tpr_at_fpr1_ood", "tpr_at_tau_ood", "fpr_at_tau_ood", "mu0_val", "mu1_val",
             "rho_z_mean", "rho_z_median"],
         ["ID", "Architecture", "HF base model", "Seed", "Type", "AUROC (OOD)",
          "TPR@FPR1% (OOD)", "TPR@tau (OOD)", "FPR@tau (OOD)", "mu0 (val)", "mu1 (val)",
          "rho (mean)", "rho (median)"],
         {"auroc_ood": p4, "tpr_at_fpr1_ood": p4, "tpr_at_tau_ood": p4,
          "fpr_at_tau_ood": p4, "mu0_val": p4, "mu1_val": p4,
          "rho_z_mean": p4, "rho_z_median": p4}))
    A("")
    A("**Doc luu y.** `TPR@tau` la TPR tai nguong `tau` duoc hieu chuan o FPR = 1%% tren "
      "`val`; FPR thuc te cua nguong do tren `test_ood_model` cao hon 1%% o mot so "
      "checkpoint. `TPR@FPR1%` la TPR tai dung FPR 1%% do **tren chinh** "
      "`test_ood_model`. Bai bao cao cot `TPR@tau` (0.8435--0.9254).\n")
    A("Nguon: `results/s1_checkpoints.csv`\n")

    A("### S1b. Ket qua thong ke muc tai lieu theo tung checkpoint\n")
    A("Bang nay noi truc tiep model voi hai ket qua chinh cua bai: do dung tam cua "
      "`pi-hat` va quy tac `Z_k` tren `test_ood_model` (637 bai human + 637 bai AI). "
      "Cac cot FPR/miss deu la phan tram.\n")
    d = S1b.copy()
    d["seed"] = d.seed.astype(str)
    A(md(d, ["id", "architecture", "seed", "pihat_mean_human", "pihat_mean_ai",
             "rho_z_median", "fpr_z", "miss_z", "fpr_ctrl", "miss_ctrl",
             "p_mcnemar", "sig_bh_q05"],
         ["ID", "Architecture", "Seed", "pi-hat H", "pi-hat AI", "rho",
          "FPR z %", "Miss z %", "FPR raw %", "Miss raw %", "McNemar p", "BH q<.05"],
         {"pihat_mean_human": p3, "pihat_mean_ai": p3, "rho_z_median": p3,
          "fpr_z": pct2, "miss_z": pct2, "fpr_ctrl": pct2, "miss_ctrl": pct2,
          "p_mcnemar": lambda x: "%.2e" % x}))
    A("")
    A("**Pham vi so sanh raw.** De ghep cap tren cung 637 bai, cot raw dung trung binh "
      "cua 200 nguong duoc chot tren cac nua calibration, roi ap mot lan cho toan bo "
      "tap. Vi vay miss raw trung vi la **6,04%**, khac nhe uoc luong split-evaluation "
      "**6,16%** o Table IV. FPR cung khong khop hoan toan theo checkpoint; McNemar o "
      "day do bat dong tai cac diem van hanh quan sat, khong chung minh uu the power o "
      "FPR bang nhau tuyet doi.\n")
    A("Nguon: `results/s1b_document_results.csv`, "
      "`revision_experiments_rivf2026/results/rev3_paired_miss.csv`\n")

    # ---------------------------------------------------------------- S2
    A("---\n")
    A("## S2. Dataset\n")
    A("### S2a. Thong ke split\n")
    A("Bo sung cho **§III-A** va **§III-C**. So o day la so **thuc te da cham diem** "
      "(sau khi loc khu nhieu be mat), khong phai so truoc khi loc. `k` tinh theo "
      "(paper_id, label) -- dung dinh nghia cua bai.\n")
    A(md(S2b, ["split", "n_papers", "n_segments", "n_human", "n_ai",
               "k_median", "k_mean", "k_p75", "k_max", "source"],
         ["Split", "# papers", "# segments", "# human", "# AI",
          "Median k", "Mean k", "P75 k", "Max k", "Source"],
         {"k_median": p2, "k_mean": p2, "k_p75": p2}))
    A("")
    A("Nguon: `results/s2b_splits.csv`\n")

    A("### S2b. Model sinh van ban AI: 21 model tren 7 ho\n")
    A("Bo sung cho **§III-A**. Diem thiet ke quan trong: **cac ho generator roi nhau "
      "hoan toan** giua tap huan luyen/thuc va tap OOD -- day chinh la nghia cua ten "
      "`test_ood_model`.\n")
    A(md(S2c, ["split", "generator_family", "n_segments", "pct_of_ai"],
         ["Split", "Generator family", "# AI segments", "% of AI in split"],
         {"pct_of_ai": p2}))
    A("")
    A("Nguon: `results/s2c_generators.csv`\n")

    A("#### Danh sach day du -- bai bao tro toi day tu §III-A\n")
    A("Parse truc tiep tu `FAMILY_BY_KEY` trong `genai.ipynb`. Cot cuoi doi chieu cheo "
      "voi `generator_family` co that trong parquet da cham diem.\n")
    A(md(S2d, ["model_id", "family", "split_pool", "family_present_in_scored_data"],
         ["OpenRouter model ID", "Family", "Split pool", "Seen in scored data"]))
    A("")
    A("**15 model / 5 ho** cho train+val; **6 model / 2 ho** (`google`, `deepseek`) giu "
      "rieng cho `test_ood_model` va **khong xuat hien trong train duoi bat ky hinh thuc "
      "nao** -- day la nghia chinh xac cua \"OOD-model\".\n")

    A("#### Cau hinh sinh\n")
    A(md(S2e, ["setting", "key", "value", "detail", "share_pct"],
         ["Setting", "Key", "Meaning", "Detail", "Share %"], {"share_pct": p2}))
    A("")
    A("Mode, template va temperature deu duoc gan bang **hash tat dinh cua "
      "(paper_id, source)** -- khong dung random, nen chay lai cho ket qua y het.\n")
    A("Nguon: `results/s2d_generator_models.csv`, `results/s2e_generation_config.csv`\n")

    A("### S2c. Cong thuc dung `test_fp_full`\n")
    A("Bo sung cho **§III-C**. Doc tu `output/testfp_full_manifest.json` -- day la recipe "
      "day du de dung lai tap nay tu dau.\n")
    A(md(S2f, ["property", "value"], ["Property", "Value"]))
    A("")
    A("Diem quan trong: dung **dung ham** `split_into_sentence_chunks` cua "
      "`get-dataset.ipynb`, chi bo buoc `random.sample(k=3)` -- nen `test_fp_full` va cac "
      "split khac cat doan **giong het nhau**, chi khac so doan giu lai moi bai. "
      "**5.274** `paper_id` xuat hien o split khac da bi loai truoc khi dung.\n")
    A("Nguon: `results/s2f_construction_recipe.csv`\n")

    A("### S2d. Schema record-level\n")
    A("Bo sung cho **§III-A**. Doc truc tiep tu parquet.\n")
    for f in S2a.file.unique():
        d = S2a[S2a.file == f]
        A("**`%s`**\n" % f)
        A(md(d, ["field", "dtype", "n_unique", "example", "description"],
             ["Field", "Type", "# unique", "Example", "Description"]))
        A("")
    A("Nguon: `results/s2a_schema.csv`\n")

    # ---------------------------------------------------------------- S3
    A("---\n")
    A("## S3. Do tan mac qua 16 checkpoint cho moi bang chinh\n")
    A("Tra loi **Revision Brief muc 4**. Khoang tin cay bootstrap trong Table III cua bai "
      "do **bat dinh doi voi quan the bai bao**; no khong noi gi ve viec 16 checkpoint "
      "tan mac ra sao. Bang nay bo sung dung phan do.\n")
    for t in ["I", "II", "III", "IV"]:
        d = S4[S4.table == t]
        if not len(d):
            continue
        A("### Table %s\n" % t)
        A(md(d, ["quantity", "median", "lo", "hi", "n_ckpt"],
             ["Quantity", "Median", "Min", "Max", "# ckpt"],
             {"median": (p3 if t == "I" else p2),
              "lo": (p3 if t == "I" else p2), "hi": (p3 if t == "I" else p2)}))
        A("")
    A("Xem them `figures/figS1_fpr_boxplot.pdf`: boxplot phan bo FPR cua 16 checkpoint "
      "tai moi `k`, ba phuong phap canh nhau.\n")
    A("Nguon: `results/s4_ranges_main_tables.csv`\n")

    # ---------------------------------------------------------------- S4
    A("---\n")
    A("## S4. Pool hieu chuan `m` va do phan giai p-value\n")
    A("Tra loi **Revision Brief muc 6.2**, va la co so cho cau canh bao them vao §V-B.\n")
    A(md(S6p, ["used_in", "pool", "m_median", "m_min", "m_max", "p_min",
               "n_steps_below_alpha", "z_max"],
         ["Dung o", "Pool", "m (median)", "m min", "m max", "p nho nhat",
          "# muc p duoi 1%", "z lon nhat"],
         {"m_median": lambda x: "%.0f" % x, "p_min": lambda x: "%.2e" % x,
          "n_steps_below_alpha": p2, "z_max": p3}))
    A("")
    A("Trong phep chuyen ngoai mau (§V-D), pool chi co **438** doan human, nen `z_i` bi "
      "chan tren o **3.05** va chi co **4** muc p roi rac nam duoi `alpha = 1%`. Do la "
      "mot gioi han that ve do phan giai duoi phai, khong chi la van de co mau.\n")
    A("Nguon: `results/s6_calibration_pool.csv`\n")

    # ---------------------------------------------------------------- S5
    A("---\n")
    A("## S5. Da kiem dinh cho McNemar\n")
    A("Tra loi **Revision Brief muc 6.1**. McNemar chay 16 lan (mot lan / checkpoint), "
      "nen p tho phai duoc hieu chinh.\n")
    A("| Tieu chi | So checkpoint co y nghia |\n|---|---|")
    A("| Tho (p < 0.05) | **%d** / 16 |" % S6m.sig_raw.sum())
    A("| Benjamini--Hochberg (q = 0.05) | **%d** / 16 |" % S6m.sig_bh_q05.sum())
    A("| Bonferroni (p < 0.05/16) | **%d** / 16 |" % S6m.sig_bonferroni.sum())
    A("")
    A(md(S6m, ["model", "miss_z", "miss_ctrl", "b_only_z", "c_only_ctrl",
               "p_mcnemar", "sig_raw", "sig_bh_q05", "sig_bonferroni"],
         ["Checkpoint", "miss (z)", "miss (control)", "b: only z", "c: only control",
          "p", "raw", "BH", "Bonf."],
         {"miss_z": p4, "miss_ctrl": p4, "p_mcnemar": lambda x: "%.2e" % x}))
    A("")
    A("> 16 checkpoint **khong doc lap** -- chung cham cung 637 bai AI -- nen ca "
      "Bonferroni lan BH deu bao thu qua muc o day. Con so dang tin la so checkpoint "
      "rieng le vuot nguong, khong phai mot p gop.\n")
    A("Nguon: `results/s6_multiple_testing.csv`\n")

    # ---------------------------------------------------------------- S6
    A("---\n")
    A("## S6. FPR muc tai lieu theo tieu linh vuc arXiv\n")
    A("Tra loi **Revision Brief muc 4** (dong cuoi). Ban truoc cua bai chi neu hai diem "
      "cuc tri (`cs.CV` 3.90%, `cs.DS` 23.53%) va tu ghi chu rang chung duoc do **truoc** "
      "khi chuyen sang khung pi-hat -- tuc chung khong phai FPR cua phuong phap dang de "
      "xuat. Bang duoi do lai **tat ca** tieu linh vuc duoi dung quy tac trien khai "
      "(z-scale, `k` that cua tung bai), va kem quy tac nguong tinh de so sanh.\n")
    for meth in ["z-scale (proposed)", "static (raw scale)"]:
        d = S7[S7.method == meth].sort_values("n_papers", ascending=False)
        A("### %s\n" % meth)
        A(md(d, ["subfield", "n_papers", "k_median", "fpr_pooled", "ci_lo", "ci_hi",
                 "fpr_min", "fpr_max", "interpretable"],
             ["Subfield", "# papers", "Median k", "FPR (pooled) %", "Wilson lo",
              "Wilson hi", "min over ckpt", "max over ckpt", "n>=20?"],
             {"k_median": lambda x: "%.0f" % x, "fpr_pooled": p2, "ci_lo": p2,
              "ci_hi": p2, "fpr_min": p2, "fpr_max": p2}))
        A("")
    z = S7[(S7.method == "z-scale (proposed)") & S7.interpretable]
    A("Voi 19 tieu linh vuc co it nhat 20 bai, FPR di tu **%.2f%%** den **%.2f%%**, "
      "trung vi **%.2f%%**. Khoang Wilson cua `cs.PL` la [%.2f%%, %.2f%%], loai tru "
      "muc tieu 1%% -- nen thien lech theo tieu linh vuc la **that**, khong bi phep bien "
      "doi xoa di. Nhung 30/38 tieu linh vuc co duoi 40 bai, nen tung uoc luong rieng le "
      "rat yeu.\n"
      % (z.fpr_pooled.min(), z.fpr_pooled.max(), z.fpr_pooled.median(),
         z[z.subfield == "cs.PL"].ci_lo.iloc[0], z[z.subfield == "cs.PL"].ci_hi.iloc[0]))
    A("Nguon: `results/s7_subfield_summary.csv`, `results/s7_subfield_fpr.csv`\n")

    A("---\n")
    A("## S7. pi-hat tren van ban tron o muc CAU\n")
    A("Dat so cho han che dinh tinh o **§VI**: mo hinh tron 2 thanh phan gia dinh moi doan "
      "thuoc **tron ven** mot trong hai cuc, nen pi-hat phai doc la \"muc do giong van "
      "phong AI\", khong phai ti le ky tu do AI go.\n")
    A("`artifacts/pred_mix/scores_all.parquet`: **325.728** dong da cham diem, **609** bai, "
      "**16** checkpoint. Moi dong la MOT doan (~270 tu) bi thay mot so **cau** bang ban AI "
      "tuong ung, kem ti le tu that.\n")
    A("> **Pham vi -- doc truoc khi trich dan.** Day **khong** phai phan bac §IV-A. §IV-A "
      "tron o muc **doan** (thay tron ven tung doan trong tai lieu k=20), noi pi-hat affine "
      "theo dung dinh nghia. Bang duoi tron o muc **cau ben trong mot doan** (k=1), nam "
      "**ngoai** pham vi bai tuyen bo. Do tan mac giam theo 1/sqrt(k) khi gop nhieu doan, "
      "nhung **thien lech thi khong** -- do la diem dang luu y.\n")
    A(md(S8, ["bin", "n", "true_mean", "pi_mean", "pi_median", "bias", "pi_sd"],
         ["True ratio bin", "n", "True mean", "pi-hat mean", "pi-hat median",
          "Bias", "SD"],
         {"true_mean": p3, "pi_mean": p3, "pi_median": p3, "bias": p3, "pi_sd": p3}))
    A("")
    mid = S8[(S8.true_mean > 0.1) & (S8.true_mean < 0.55)]
    A("pi-hat **doc hut** cac ti le trung gian, manh nhat la **%.3f** tai ti le that "
      "%.3f. Trung vi gan nhu **nhi thuc** -- am o ti le thap roi nhay len ~1,04 tu bin "
      "0,6 tro di -- nen mot doan tron nua-nua thuong bi doc thanh \"toan AI\" hoac "
      "\"toan nguoi\", hiem khi thanh \"mot nua\". Dat cau AI **lien khoi** bi doc hut it "
      "hon (-0,067) so voi **rai rac** (-0,144).\n"
      % (mid.bias.min(), mid.loc[mid.bias.idxmin(), "true_mean"]))
    A("Nguon: `results/s8_submix_pihat.csv`, `results/s8_submix_by_checkpoint.csv`\n")

    A("---\n")
    A("## S8. Ket qua tu khung bai TRUOC (`REPORT_2026-08-15.md`)\n")
    A("Bao cao ngay 15/08 ghi lai mot khung bai **cu**: chan doan lech **don vi** (nguong "
      "hieu chuan cho *doan*, quyet dinh dua ra o *bai*) roi chua bang Sidak/Bonferroni. "
      "Bai hien tai da chuyen sang khung khac (pi-hat + `Z_k`). So lieu cu **van dung** -- "
      "moi con so duoi day da duoc do lai tu CSV -- va van tra loi duoc nhung cau reviewer "
      "se hoi, nen giu o day thay vi vut di.\n")
    A("> **CANH BAO -- doc truoc khi trich.** Bang trong muc nay do luat gan co **`any`** "
      "(gan co bai neu **co doan nao** vuot nguong). Table III cua bai do luat **`mean`** "
      "(so trung binh diem cua bai voi mot nguong). **Hai luat khac nhau, khong dat canh "
      "nhau duoc.** Cung `test_fp_full`, cung k=20: luat `any` cho **10,11%** con luat "
      "`mean` cho **0,71%** -- chenh 14 lan, va ca hai deu dung, vi chung tra loi hai cau "
      "hoi khac nhau.\n")

    A("### S8a. Lech don vi duoi luat `any`\n")
    A("Nguong chot tren `val` cho 1%% **muc doan**, ap nguyen si, gan co bai neu co doan "
      "nao vuot:\n")
    d = S9u.copy()
    for c in ["fpr_chunk", "fpr_paper", "fpr_if_independent"]:
        d[c] = 100 * d[c]
    d["formula_error"] = 100 * d["formula_error"]
    A(md(d, ["model", "fpr_chunk", "fpr_paper", "fpr_if_independent", "formula_error",
             "ratio"],
         ["Checkpoint", "FPR segment %", "FPR paper %", "If independent %",
          "Formula error %", "Ratio"],
         {"fpr_chunk": p2, "fpr_paper": p2, "fpr_if_independent": p2,
          "formula_error": p2, "ratio": p2}))
    A("")
    A("FPR muc doan **%.2f--%.2f%%** giu dung cam ket; cung nguong do o muc bai cho "
      "**%.2f--%.2f%%**, tuc **%.1f--%.1f lan**, tren **%d/%d checkpoint khong ngoai le**. "
      "Huan luyen doi khang (GREATER, RADAR) **khong** cuu duoc -- chung nam rai khap dai.\n"
      % (100 * S9u.fpr_chunk.min(), 100 * S9u.fpr_chunk.max(),
         100 * S9u.fpr_paper.min(), 100 * S9u.fpr_paper.max(),
         S9u.ratio.min(), S9u.ratio.max(), (S9u.ratio > 1).sum(), len(S9u)))
    A("Nguon: `results/s9_unit_mismatch.csv`\n")

    A("### S8b. Duong cong co kiem soat theo `k` (luat `any`)\n")
    A("**Cung 576 bai, cung checkpoint, cung nguong -- chi doi so doan nhin vao.** Nen "
      "quan the khong doi theo `k`.\n")
    d = S9k.copy()
    d["fpr_any"] = 100 * d.fpr_any
    d["fpr_any_sd"] = 100 * d.fpr_any_sd
    A(md(d, ["k", "n_papers", "fpr_any", "fpr_any_sd"],
         ["k", "# papers", "FPR (any) %", "SD %"], {"fpr_any": p2, "fpr_any_sd": p2}))
    A("")
    A("Tang **don dieu, khong bac nhay**. O `k=3` (kich thuoc cua cac benchmark ngan) FPR "
      "chi **1,92%**; o `k=20` (bai khoa hoc that) la **10,11%** -- benchmark ngan lam van "
      "de trong **nho hon ~5 lan** thuc te.\n")
    A("Nguon: `results/s9_k_curve_any.csv`\n")

    A("### S8c. Co tap hieu chuan quyet dinh nguong co dung duoc khong\n")
    A("Lien quan truc tiep **§V-B** cua bai (pool `m=438`). Sidak doi phan vi "
      "$1-\\alpha/k$; voi `k=19` va muc tieu 1% thi can phan vi **0,999471**, ma 438 doan "
      "chi phan giai toi **0,99772**.\n")
    A(md(S9p, ["pool", "n_chunks", "pipeline", "fpr_median", "fpr_min", "fpr_max",
               "spread"],
         ["Pool", "# chunks", "Pipeline / note", "FPR median %", "min %", "max %",
          "Spread"],
         {"fpr_median": p2, "fpr_min": p2, "fpr_max": p2, "spread": p2}))
    A("")
    A("**Hai chan doan tach bach:**\n")
    A("1. **La CO MAU, khong phai lech phan phoi.** Cung pipeline ma chi 438 doan van "
      "hong y het (3,90% vs 3,98%).\n")
    A("2. **Lech phan phoi gay benh KHAC.** Cung 438 doan: cung pipeline lech **deu** "
      "(bien do 0,66 diem), khac pipeline lech **vo huong** (bien do 9,30 diem). Co mau "
      "gay sai lech co he thong; lech phan phoi gay sai lech **khong doan duoc huong**.\n")
    A("San ly thuyet la **>=1.891 doan** cho `k=19`; do thuc nghiem thi 2.000 doan cho "
      "1,40% va ~13.960 doan (≈7x san) cho 1,05%. `val` co **438** -- thieu khoang "
      "**4,3 lan**.\n")
    A("Nguon: `results/s9_pool_size.csv`\n")

    A("### S8d. Gia phai tra khi doi diem van hanh\n")
    A("Sidak ha `alpha` muc doan tu 1% xuong **0,0529%** (voi `k=19`), nen moi con so TPR "
      "do o 1% phai do lai:\n")
    d = S9t.copy()
    for c in ["tpr_at_1pct", "tpr_at_01pct", "tpr_at_sidak"]:
        d[c] = 100 * d[c]
    A(md(d, ["model", "k_median", "tpr_at_1pct", "tpr_at_01pct", "tpr_at_sidak",
             "loss_points"],
         ["Checkpoint", "k", "TPR @1% ", "TPR @0.1%", "TPR @Sidak", "Loss (pts)"],
         {"tpr_at_1pct": p2, "tpr_at_01pct": p2, "tpr_at_sidak": p2,
          "loss_points": p2}))
    A("")
    A("Trung vi mat **6,97 diem** (90,79% -> 83,82%), nhung **ca xau nhat mat 24,1 diem** "
      "(`radar_xlm` s42: 76,42% -> 52,31%). Phai bao ca dai, khong duoc dung trung vi che "
      "ca xau nhat. Day la TPR muc **doan**; muc bai chua do duoc vi `test_fp_full` chi co "
      "lop human.\n")
    A("Nguon: `results/s9_tpr_cost.csv`\n")

    A("---\n")
    A("## Cach tai tao\n")
    A("```bash\npython supplement_rivf2026/build_supplement.py   # S1, S2, S3, S4, S5\n"
      "python supplement_rivf2026/generator_matrix.py   # S2d, S2e\n"
      "python supplement_rivf2026/subfield_fpr.py       # S6\n"
      "python supplement_rivf2026/submix_pihat.py       # S7\n"
      "python supplement_rivf2026/report_tables.py      # S8, S2c\n"
      "python supplement_rivf2026/make_boxplot.py       # figS1\n"
      "python supplement_rivf2026/render_supplement.py  # tai sinh chinh file nay\n```\n")
    A("Khong buoc nao chay lai detector; tat ca deu la hau xu ly tren "
      "`artifacts/` va `output/` co san.\n")

    DOC.write_text("\n".join(L), encoding="utf-8")
    print("-> %s (%d dong, %d ky tu)" % (DOC, len("\n".join(L).split("\n")),
                                         len("\n".join(L))))


if __name__ == "__main__":
    main()
