"""Phan BO SUNG cho verify_paper_numbers.py: kiem not cac con so trong paper/rivf2026.tex
ma script goc chua kiem (so lieu dataset, §IV-A trung vi/phan vi, Table III nua rong,
§V-B do nhay rho_z, §VI ...).

    py verify_paper_numbers_ext.py          # chay CA script goc lan phan bo sung

Moi so deu tinh lai tu artifact / CSV goc, khong go tay. Dung chung chk()/summary()
va bo dem voi verify_paper_numbers.py.
"""
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import verify_paper_numbers as V
from verify_paper_numbers import ART, BASE, FEX, FP_FULL, OUT, chk

EXR = BASE / "extra_experiments_rivf2026" / "results"
SUP = BASE / "supplement_rivf2026" / "results"
DAG = BASE / "external" / "dagpap24"
RAW_AI = BASE / "results" / "human_ai_multi_free_models_v2.jsonl"
RAW_HU = BASE / "results" / "human_samples_2018_clean.jsonl"


def _models():
    return sorted(p.name.replace("artifacts_", "") for p in ART.glob("artifacts_*"))


def _ok(flag, name, got, paper):
    print("  [%s] %-42s %s | bai: %s" % ("OK" if flag else "!!", name, got, paper))
    V._tally["ok" if flag else "bad"] += 1


def _skip(name, why):
    """Phep kiem can du lieu khong phat tan (van ban goc, DAGPap24): bo qua, khong tinh."""
    print("  [--] %-42s SKIP: %s" % (name, why))
    V._tally["skip"] = V._tally.get("skip", 0) + 1


def _find(*rel):
    """Tim file o bo cuc repo lam viec hoac bo cuc repo cong khai."""
    for r in rel:
        p = BASE / r
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------- §I + §III-C
def sec1_corpus():
    print("\n### I / III-C  do dai corpus day du, DAGPap24")
    d = pd.read_parquet(OUT / "dataset_test_fp_full_deconf2.parquet")
    # repo cong khai: van ban da bo, chi con cot n_words (dem bang cung cach split())
    nw = d.n_words if "n_words" in d else d.human_text.str.split().str.len()
    words = nw.groupby(d.paper_id).sum()
    k = d.groupby("paper_id").size()
    chk("so tu / bai (trung vi)", words.median(), 5560, 5)
    chk("so doan: trung vi, p75, mean, max, min",
        [k.median(), k.quantile(.75), k.mean(), k.max(), k.min()], [19, 27, 22.6, 262, 10], 0.05)
    chk("so bai, so doan", [len(k), len(d)], [1200, 27119], 0.5)
    chk("nam xuat ban", [d.published_year.min(), d.published_year.max()], [2010, 2015], 0.5)
    chk("bai >=20 doan: so bai, ti le", [(k >= 20).sum(), (k >= 20).mean()], [576, 0.48], 0.005)
    if (DAG / "train_data.parquet").exists():
        g = pd.read_parquet(DAG / "train_data.parquet", columns=["text"])
        chk("DAGPap24 so tu / bai (trung vi)", g.text.str.split().str.len().median(), 5400, 50)
    else:
        _skip("DAGPap24 so tu / bai (trung vi)", "can external/dagpap24/train_data.parquet")
    t = pd.read_csv(OUT / "shortcut_audit_dagpap24" / "totals_auc.csv")
    a = t[(t.combo == "all") & (t.model == "GBM") & t.dataset.str.contains("3-nh")
          & t.dataset.str.endswith("/raw")].auc_mean
    chk("DAGPap24 22 dac trung, AUROC (AI gop, raw)", a.iloc[0], 0.9991, 0.0001)
    print("  (OpAI-Bench 244--367 tu: so lieu cua bai goc, khong tinh lai duoc o day;"
          " 5560/367=%.1fx, 5560/244=%.1fx -> '15--23x')" % (5560 / 367, 5560 / 244))


# ---------------------------------------------------------------- §III-A
def sec3a_dataset():
    print("\n### III-A  dataset va khu shortcut")
    ai = pd.read_parquet(OUT / "ai_clean.parquet")
    chk("so model sinh, so ho model", [ai.model_actual.nunique(), ai.generator_family.nunique()],
        [21, 7], 0.5)
    chk("temperature min / max", [ai.temperature.min(), ai.temperature.max()], [0.3, 1.1], 0.001)
    m = json.loads((OUT / "deconf2_manifest.json").read_text(encoding="utf-8"))
    chk("ti le cap bi bo (notation gap)", 1 - m["notation_keep_frac"], 0.15, 0.001)
    au = {x["variant"].split()[0]: x["total"] for x in m["audit"]}
    chk("surface AUROC truoc -> sau", [au["normalized"], au["deconf2"]], [0.9343, 0.8909], 0.0001)
    s = pd.read_csv(OUT / "surface_baseline_deconf2.csv").set_index(["baseline", "split"])
    r = s.loc[("surface", "test_ood_model")]
    chk("surface AUROC tap matched, n", [r.auroc_matched, r.n_matched], [0.545, 1206], 0.001)
    mm = [json.loads((ART / ("artifacts_" + x) / "metrics.json").read_text()) for x in _models()]
    mm = [{e["split"]: e for e in j["metrics"]}["test_ood_model_matched"]["auroc"] for j in mm]
    chk("16 detector tren tap matched: min / max", [min(mm), max(mm)], [0.927, 0.965], 0.001)
    if not RAW_AI.exists():
        _skip("dau cau kieu in: % mau AI, % mau human", "can van ban goc (khong phat tan)")
    else:
        # dung DUNG bang fold dau cau cua 01_clean_qa.ipynb, sau NFKC nhu pipeline
        nb = json.loads(_find("01_clean_qa.ipynb", "dataset_construction/01_clean_qa.ipynb")
                        .read_text(encoding="utf-8"))
        src = "".join(nb["cells"][0]["source"])
        i = src.index("_PUNCT_FOLD = str.maketrans({")
        ns = {}
        exec(src[i:src.index("})", i) + 2], ns)
        # bo dau phay tren (U+2032/2033): la ky hieu toan, co trong 4,2% mau human LaTeX,
        # nen khong phai "dau cau kieu in" phan biet hai lop
        chars = "".join(chr(c) for c in ns["_PUNCT_FOLD"] if c not in (0x2032, 0x2033))
        has = lambda t: any(ch in unicodedata.normalize("NFKC", t or "") for ch in chars)
        raw = pd.read_json(RAW_AI, lines=True)
        hu = pd.read_json(RAW_HU, lines=True)
        chk("dau cau kieu in: % mau AI, % mau human",
            [100 * raw[raw.text_type == "ai"].text.map(has).mean(), 100 * hu.text.map(has).mean()],
            [37, 0], 0.5)


# ---------------------------------------------------------------- §III-B + Table II
def sec3b_baselines():
    print("\n### III-B / Table II  baseline, zero-shot, cac split")
    s = pd.read_csv(OUT / "surface_baseline_deconf2.csv").set_index(["baseline", "split"])
    chk("TF-IDF / surface AUROC (held-out generator)",
        [s.loc[("lexical", "test_ood_model")].auroc_full, s.loc[("surface", "test_ood_model")].auroc_full],
        [0.9491, 0.8715], 0.0001)
    zs = {}
    for f in list((BASE / "detectgpt_models").glob("*/artifacts_*/metrics.json")) + \
            list((BASE / "statistical_models").glob("*/artifacts_*/metrics.json")) + \
            list((BASE / "detectors" / "zero_shot").glob("*/metrics.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        zs[j["run"]] = {e["split"]: e for e in j["metrics"]}["test_ood_model"]["auroc"]
    print("  zero-shot:", {k: round(v, 4) for k, v in zs.items()})
    v = list(zs.values())
    chk("zero-shot: so luong, min, max", [len(v), min(v), max(v)], [4, 0.40, 0.53], 0.005)
    chk("zero-shot nguoc dau (AUROC<0.5)", sum(x < 0.5 for x in v), 2, 0.5)
    fp = pd.read_parquet(ART / "artifacts_roberta_large_seed42" / "pred_test_fp.parquet")
    chk("false positive split: human / lightly edited",
        [(fp.ai_degree == 0).sum(), (fp.ai_degree == 1).sum()], [3335, 955], 0.5)
    ai = pd.read_parquet(OUT / "ai_clean.parquet")
    sh = ai.groupby("split_pool").paper_id.nunique()
    sh = 100 * sh / sh.sum()
    print("  Table II 'Share' = trong so hash gan bai (genai.ipynb SPLIT_POOL_WEIGHTS 70/15/15);"
          " thuc te tren bai co ban AI: train_val %.0f%%, ood %.0f%%, test_fp %.0f%%"
          % (sh["train_val"], sh["ood_model"], sh["test_fp"]))


# ---------------------------------------------------------------- §III-D + §IV-A
def sec4a_extra():
    print("\n### III-D / IV-A  neo, trung vi, phan vi, tai lieu tron")
    s = pd.read_csv(OUT / "pihat_sanity_16models.csv").set_index("model")
    chk("neo RoBERTa-large (s42): mu0, mu1",
        [s.loc["roberta_large_seed42", "mu0"], s.loc["roberta_large_seed42", "mu1"]],
        [0.0658, 0.9643], 0.0001)
    H, A, n_doc = [], [], set()
    for m in _models():
        o = pd.read_parquet(ART / ("artifacts_" + m) / "pred_test_ood_model.parquet")
        g = o.groupby(["paper_id", "label"]).score.mean().reset_index()
        g["pi"] = (g.score - s.loc[m, "mu0"]) / (s.loc[m, "mu1"] - s.loc[m, "mu0"])
        H.append(g[g.label == 0].pi)
        A.append(g[g.label == 1].pi)
        n_doc.add(((g.label == 0).sum(), (o.label == 0).sum()))
    H, A = pd.concat(H), pd.concat(A)
    print("  (trung vi / phan vi GOP qua 10.192 du doan checkpoint--tai lieu; trung binh thi"
          " tinh theo checkpoint roi lay trung binh)")
    chk("trung vi gop: human, AI", [H.median(), A.median()], [-0.038, 1.045], 0.001)
    chk("p95 human, p5 AI (gop)", [H.quantile(.95), A.quantile(.05)], [0.325, 0.531], 0.001)
    chk("ti le human co pi-hat > 0.32 (~5%)", (H > 0.32).mean(), 0.05, 0.006)
    chk("so bai / lop, so slot human", list(n_doc.pop()), [637, 1514], 0.5)
    e = pd.read_csv(EXR / "exp01_calibration_table.csv")
    e = e[e.design == "paired"].set_index("true_pi")
    chk("tron k=20: mean pi-hat tai 7 muc pi", list(e.mean_pi_hat),
        [0.007, 0.104, 0.246, 0.492, 0.737, 0.878, 0.977], 0.0006)
    chk("tron: trung vi MAE", e.mae.median(), 0.039, 0.0006)
    b = pd.read_csv(EXR / "breakdown_mae_decomposition.csv")
    chk("tron: trung vi |bias|/SD", b[b.design == "paired"].bias_over_sd.median(), 0.15, 0.006)


# ---------------------------------------------------------------- §IV-B
def sec4b_extra():
    print("\n### IV-B  rho_raw, nua rong Table III, khoang plug-in")
    f1 = pd.read_csv(FEX / "f1_coverage_null.csv")
    pm = f1.groupby(["model", "k"]).mean(numeric_only=True).reset_index()
    r = pm[pm.k == 20]
    chk("rho_raw: trung vi, min, max", [r.rho_raw.median(), r.rho_raw.min(), r.rho_raw.max()],
        [0.132, 0.087, 0.167], 0.0006)
    chk("do phu Empirical qua checkpoint, k=20", [100 * r.cov_emp.min(), 100 * r.cov_emp.max()],
        [93.5, 95.6], 0.06)
    hw = pm.groupby("k")[["hw_naive", "hw_emp"]].median()
    chk("Table III nua rong Gauss k=1,5,10,20", list(hw.loc[[1, 5, 10, 20], "hw_naive"]),
        [0.368, 0.165, 0.116, 0.082], 0.0006)
    chk("Table III nua rong Emp.  k=1,5,10,20", list(hw.loc[[1, 5, 10, 20], "hw_emp"]),
        [0.493, 0.198, 0.163, 0.152], 0.0006)
    f6 = pd.read_csv(FEX / "f6_coverage_mixed_emp.csv")
    g = f6.groupby(["model", "k", "true_pi"]).mean(numeric_only=True).reset_index()
    g = g[g.k == 20].groupby("true_pi")[["hw_emp_null", "hw_emp_plugin"]].median()
    chk("nua rong hieu chuan tai pi=0", g.loc[0.0, "hw_emp_null"], 0.147, 0.0006)
    chk("nua rong plug-in pi=0 -> 1", [g.loc[0.0, "hw_emp_plugin"], g.loc[1.0, "hw_emp_plugin"]],
        [0.147, 0.101], 0.0006)


# ---------------------------------------------------------------- §V-A / §V-B
def sec5_extra():
    print("\n### V-A / V-B  phan bo sung")
    p = pd.read_csv(OUT / "pihat_static_vs_corrected_by_k.csv").groupby("k").fpr_corrected.median() * 100
    chk("Gauss tho: k=2, trung vi k>=3", [p.loc[2], p[p.index >= 3].median()], [6.78, 5.4], 0.03)
    f4 = pd.read_csv(FEX / "f4_null_shape.csv")
    sd = f4[f4.k == 20].groupby("model").sd.mean().median()
    chk("sd cua T_k tai k=20", sd, 0.965, 0.0006)
    t = pd.read_csv(OUT / "tau_k_mean_16models.csv")
    rz = t.groupby("model").rho.mean()
    rs = t.groupby(["model", "split"]).rho.first()
    true = ((1 + 19 * rz.median()) * sd ** 2 - 1) / 19
    chk("Hartung danh gia cao rho_z (~9%)", 100 * (rz.median() / true - 1), 9, 0.6)
    chk("rho_z theo checkpoint: min, max", [rz.min(), rz.max()], [0.204, 0.401], 0.0006)
    chk("rho_z 80 resample: n, min, max", [len(rs), rs.min(), rs.max()], [80, 0.183, 0.424], 0.0006)
    chk("san sqrt(rho): raw 36%, z 55%", [100 * np.sqrt(0.132), 100 * np.sqrt(rz.median())], [36, 55], 0.6)
    rc = pd.read_csv(OUT / "rho_compare_16models.csv")
    chk("ICC mot chieu tren thang z", rc[rc.estimator == "kish"].groupby("model").rho.mean().median(),
        0.296, 0.0006)
    e4 = pd.read_csv(EXR / "exp04_rho_sensitivity.csv").set_index(["rho_scale", "k"])
    dfpr = max(abs(e4.loc[(sc, 20), "fpr"] - e4.loc[(1.0, 20), "fpr"]) for sc in (0.8, 1.2)) * 100
    _ok(dfpr < 1, "rho_z +-20%: FPR doi < 1 diem", "%.2f diem" % dfpr, "< 1")
    chk("rho_z +-20%: miss nua-AI k=20", [100 * e4.loc[(0.8, 20), "miss_half"],
                                         100 * e4.loc[(1.2, 20), "miss_half"]], [17.3, 48.5], 0.06)
    e8 = pd.read_csv(EXR / "exp08_per_checkpoint_fpr.csv")
    chk("FPR tung checkpoint (moi k): min, max", [100 * e8.fpr.min(), 100 * e8.fpr.max()],
        [0.31, 1.24], 0.006)
    c = pd.read_csv(SUP / "s6_calibration_pool.csv")
    chk("pool hieu chuan: m (V-A), m (V-C), so p<1%",
        [c.m_median.iloc[0], c.m_median.iloc[1], np.floor(c.n_steps_below_alpha.iloc[1])],
        [9000, 438, 4], 50)
    h = pd.read_csv(OUT / "hartung_ood_16models.csv")
    m = pd.read_csv(OUT / "mean_ktransfer_16models.csv")
    chk("Table VI khoang [min,max] dong 1-2",
        [100 * h.fpr_hartung.min(), 100 * h.fpr_hartung.max(), 100 * h.miss_hartung.min(),
         100 * h.miss_hartung.max(), 100 * m.fpr_tau_ood.min(), 100 * m.fpr_tau_ood.max()],
        [0.47, 1.88, 2.67, 8.16, 1.23, 1.40], 0.006)


# ---------------------------------------------------------------- §VI
def sec6_discussion():
    print("\n### VI  thao luan va han che")
    s = pd.read_csv(OUT / "pihat_sanity_16models.csv").set_index("model")
    ratios, cc = [], []
    for m in _models():
        tau = json.loads((ART / ("artifacts_" + m) / "metrics.json").read_text())["tau"]
        f = pd.read_parquet(ART / ("artifacts_" + m) / "pred_test_fp.parquet")
        ratios.append((f[f.ai_degree == 1].score >= tau).mean() / (f[f.ai_degree == 0].score >= tau).mean())
        o = pd.read_parquet(ART / ("artifacts_" + m) / "pred_test_ood_model.parquet")
        g = o.groupby(["paper_id", "label"]).size().reset_index(name="n")
        cc.append(roc_auc_score(g.label, g.n))
    chk("FPR doan polish / human: min, max", [min(ratios), max(ratios)], [4.7, 9.8], 0.06)
    chk("AUROC chi doc so doan (moi checkpoint)", [min(cc), max(cc)], [0.5, 0.5], 0.00005)
    e6 = pd.read_csv(EXR / "exp06_oracle_comparison.csv")
    mx = e6[(e6.k >= 8)].miss.max()
    _ok(mx < 0.001, "tai lieu AI hoan toan, k>=8: miss lon nhat", "%.4f" % mx, "~0")
    s7 = pd.read_csv(SUP / "s7_subfield_summary.csv")
    z = s7[(s7.method.str.startswith("z-scale")) & (s7.n_papers >= 20)]
    lo, hi = z.loc[z.fpr_pooled.idxmin()], z.loc[z.fpr_pooled.idxmax()]
    chk("tieu linh vuc >=20 bai: so luong", len(z), 19, 0.5)
    chk("FPR tieu linh vuc (z, gop): min, max", [lo.fpr_pooled, hi.fpr_pooled], [0.00, 7.21], 0.006)
    _ok(hi.subfield == "cs.PL" and "cs.LG" in set(z[z.fpr_pooled == lo.fpr_pooled].subfield),
        "ten tieu linh vuc min / max", "%s / %s" % (lo.subfield, hi.subfield), "cs.LG / cs.PL")
    miss = pd.read_csv(OUT / "hartung_ood_16models.csv").miss_hartung.median()
    n = lambda e: 1.96 ** 2 * miss * (1 - miss) / e ** 2
    chk("so tai lieu AI can cho +-2 / +-1 diem", [n(0.02), n(0.01)], [350, 1400], 10)


EXT = (sec1_corpus, sec3a_dataset, sec3b_baselines, sec4a_extra, sec4b_extra, sec5_extra,
       sec6_discussion)


def run_ext():
    for fn in EXT:
        try:
            fn()
        except Exception as e:
            print("\n### %s -- LOI: %s: %s" % (fn.__name__, type(e).__name__, e))
            V._tally["bad"] += 1


if __name__ == "__main__":
    for fn in (V.sec3b, V.sec4a, V.sec4b, V.sec4c, V.sec4c_cells, V.sec5a, V.sec5a_boot,
               V.sec5b, V.sec5b_budget, V.sec5c, V.sec5d, V.sec5d_full):
        try:
            fn()
        except Exception as e:
            print("\n### %s -- LOI: %s: %s" % (fn.__name__, type(e).__name__, e))
    run_ext()
    V.summary()
    if V._tally.get("skip"):
        print("(%d phep kiem bi bo qua vi can du lieu khong phat tan -- xem DATA.md)"
              % V._tally["skip"])
    raise SystemExit(1 if V._tally["bad"] else 0)
