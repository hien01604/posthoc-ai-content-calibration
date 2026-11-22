"""Dung TAT CA bang cho supplementary material -- tra loi Revision Brief muc 1, 2, 4.

    python supplement_rivf2026/build_supplement.py

KHONG chay lai detector. Moi so deu doc tu artifact/CSV da co san o goc repo, nen
bang trong supplement bao dam KHOP voi bang trong bai (yeu cau so 7 cua brief).

DAU RA (supplement_rivf2026/results/)
------------------------------------
  s1_checkpoints.csv        Bang 16 checkpoint: kien truc, seed, AUROC, TPR, mu0/mu1, rho
  s1b_document_results.csv  Ket qua paper-level theo tung checkpoint
  s2a_schema.csv            Schema record-level (ten cot THAT trong parquet)
  s2b_splits.csv            Thong ke tung split (so bai / doan / k)
  s2c_generators.csv        Ho model sinh AI theo split
  s4_ranges_main_tables.csv Range per-checkpoint cho Table I-IV cua bai
  s6_calibration_pool.csv   Kich thuoc pool hieu chuan m + do phan giai p-value
  s6_multiple_testing.csv   McNemar tho vs Bonferroni vs Benjamini-Hochberg
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "artifacts" / "kaggle"
FP_FULL = ROOT / "artifacts" / "kaggle_testfp_full" / "pred_testfp_full"
OUT = ROOT / "output"
RES = HERE / "results"

# ten hien thi -- khop voi cach bai goi ten trong text
NICE = {
    "deberta_v3_large": "DeBERTa-v3-large",
    "modernbert_large": "ModernBERT-large",
    "roberta_large": "RoBERTa-large",
    "xlm_roberta_large": "XLM-RoBERTa-large",
    "greater_roberta_large": "GREATER + RoBERTa-large",
    "greater_xlm_roberta_large": "GREATER + XLM-RoBERTa-large",
    "radar": "RADAR + RoBERTa-large",
    "radar_xlm": "RADAR + XLM-RoBERTa-large",
}
ADVERSARIAL = {"greater_roberta_large", "greater_xlm_roberta_large", "radar", "radar_xlm"}


def s1_checkpoints():
    """Muc 1 cua brief: bang 16 checkpoint day du."""
    sanity = pd.read_csv(OUT / "pihat_sanity_16models.csv").set_index("model")
    tau = pd.read_csv(OUT / "tau_k_mean_16models.csv").drop_duplicates(["model", "split"])
    # rho BAO CAO trong bai = trung binh 5 lan chia, roi lay trung vi qua 16 checkpoint
    rho_mean = tau.groupby("model").rho.mean()
    # rho THUC SU AP DUNG trong paper_level_hartung_ood.py = trung vi 5 lan chia
    rho_med = tau.groupby("model").rho.median()

    rows = []
    for f in sorted(glob.glob(str(ART / "artifacts_*" / "metrics.json"))):
        model = Path(f).parent.name.replace("artifacts_", "")
        m = json.load(open(f, encoding="utf-8"))
        ood = [x for x in m["metrics"] if x["split"] == "test_ood_model"][0]
        rows.append(dict(
            model=model, run=m["run"], base_model=m["model"], seed=m["seed"],
            type="Adversarial" if m["run"] in ADVERSARIAL else "Supervised",
            architecture=NICE.get(m["run"], m["run"]),
            tau_chunk=m["tau"],
            auroc_ood=ood["auroc"],
            tpr_at_fpr1_ood=ood["tpr@fpr0.01"],
            tpr_at_tau_ood=ood["tpr@tau"],
            fpr_at_tau_ood=ood["fpr@tau"],
            mu0_val=sanity.loc[model, "mu0"], mu1_val=sanity.loc[model, "mu1"],
            sep=sanity.loc[model, "sep"],
            rho_z_mean=rho_mean.loc[model], rho_z_median=rho_med.loc[model],
            pihat_mean_human=sanity.loc[model, "pihat_mean_human"],
            pihat_mean_ai=sanity.loc[model, "pihat_mean_ai"],
        ))
    D = pd.DataFrame(rows).sort_values(["type", "architecture", "seed"],
                                       ascending=[False, True, True]).reset_index(drop=True)
    D.insert(0, "id", ["C%02d" % (i + 1) for i in range(len(D))])
    D.to_csv(RES / "s1_checkpoints.csv", index=False)
    return D


def s1b_document_results(checkpoints, multiple_testing):
    """Noi chunk metrics voi ket qua paper-level cua tung checkpoint.

    Cot raw-control lay tu REV-3: nguong raw mean la trung binh cua 200 nguong duoc
    chot tren cac nua calibration, sau do ap cho ca 637 cap bai de McNemar co cung don
    vi ghep cap. Vi vay no khac nhe uoc luong split-evaluation trong Table IV (6.04%
    thay vi 6.16% miss); ghi chu nay phai di kem bang.
    """
    paired = pd.read_csv(
        ROOT / "revision_experiments_rivf2026" / "results" / "rev3_paired_miss.csv")
    cols = ["id", "model", "architecture", "seed", "type", "auroc_ood",
            "tpr_at_tau_ood", "fpr_at_tau_ood", "pihat_mean_human",
            "pihat_mean_ai", "rho_z_median"]
    D = checkpoints[cols].merge(paired, on="model", validate="one_to_one")
    D = D.merge(
        multiple_testing[["model", "sig_bh_q05", "sig_bonferroni"]],
        on="model", validate="one_to_one")
    D.to_csv(RES / "s1b_document_results.csv", index=False)
    return D


def s2a_schema():
    """Muc 2a: schema THAT, doc truc tiep tu parquet chu khong bia."""
    src = {
        "pred_val.parquet": ART / "artifacts_roberta_large_seed42" / "pred_val.parquet",
        "pred_test_ood_model.parquet": ART / "artifacts_roberta_large_seed42" / "pred_test_ood_model.parquet",
        "pred_test_fp.parquet": ART / "artifacts_roberta_large_seed42" / "pred_test_fp.parquet",
        "pred_test_fp_full.parquet": FP_FULL / "pred_test_fp_full_artifacts_roberta_large_seed42.parquet",
        "dataset_test_fp_full_deconf2.parquet": OUT / "dataset_test_fp_full_deconf2.parquet",
    }
    DESC = {
        "sample_id": "dinh danh doan", "paper_id": "arXiv ID cua bai",
        "label": "0 = human, 1 = AI", "score": "diem detector (mot checkpoint / file)",
        "ai_degree": "0 thuan human | 1 AI-polish | 3 AI viet lai | 4 AI viet moi",
        "generator_family": "ho model sinh AI, hoac 'human'",
        "source": "vi tri doan trong bai (full_chunk_NNN)",
        "split": "train / val / test_ood_model / test_fp / test_fp_full",
        "human_text": "van ban nguoi viet", "ai_text": "van ban AI ghep cap",
        "categories": "danh muc arXiv", "published_year": "nam cong bo",
        "sample_weight": "trong so can bang", "text_role": "vai tro trong cap khop",
        "surface_propensity": "diem xu huong be mat (dung de khop cap)",
        "artifact": "ten artifact checkpoint", "branch": "nhanh huan luyen",
        "model": "ten run", "seed": "seed", "notation_gap": "chenh lech ky hieu",
        "notation_ok": "co qua bo loc ky hieu khong",
    }
    rows = []
    for name, p in src.items():
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        for c in d.columns:
            v = d[c].dropna()
            rows.append(dict(file=name, field=c, dtype=str(d[c].dtype),
                             n_unique=int(v.nunique()),
                             example=str(v.iloc[0])[:60] if len(v) else "",
                             description=DESC.get(c, "")))
    D = pd.DataFrame(rows)
    D.to_csv(RES / "s2a_schema.csv", index=False)
    return D


def s2b_splits():
    """Muc 2b: thong ke split -- dung so THAT da cham diem, khong dung so trong doc cu."""
    rows = []
    base = ART / "artifacts_roberta_large_seed42"
    for name, p in [("val", base / "pred_val.parquet"),
                    ("test_ood_model", base / "pred_test_ood_model.parquet"),
                    ("test_fp", base / "pred_test_fp.parquet"),
                    ("test_fp_full",
                     FP_FULL / "pred_test_fp_full_artifacts_roberta_large_seed42.parquet")]:
        d = pd.read_parquet(p)
        # k phai tinh theo (paper_id, label) -- dung dinh nghia cua bai. Cac split ghep
        # cap co CA hai ban human va AI cua cung mot bai, gop lai se lam k tang gap doi
        # (vd test_ood_model se ra 4.75 thay vi k~2.38 nhu bai bao).
        key = ["paper_id", "label"] if d.label.nunique() > 1 else ["paper_id"]
        g = d.groupby(key).size()
        rows.append(dict(
            split=name, n_papers=int(d.paper_id.nunique()), n_segments=int(len(d)),
            n_human=int((d.label == 0).sum()), n_ai=int((d.label == 1).sum()),
            k_median=float(g.median()), k_mean=float(g.mean()), k_max=int(g.max()),
            k_p25=float(g.quantile(.25)), k_p75=float(g.quantile(.75)),
        ))
    for r in rows:
        r["source"] = "scored parquet (measured here)"
    # `train` khong co artifact da cham diem trong repo nay (chi cac split danh gia moi
    # duoc cham). So lay tu REPORT_2026-08-13.md §3.4, cot `deconf2` -- cung bang do cho
    # val/test_ood_model/test_fp KHOP y het so do duoc o tren, nen dang tin.
    # So paper_id sau deconf2 khong duoc ghi trong report (truoc deconf2 la 2.975).
    rows.append(dict(split="train", n_papers=np.nan, n_segments=14050,
                     n_human=7025, n_ai=7025, k_median=np.nan, k_mean=np.nan,
                     k_max=np.nan, k_p25=np.nan, k_p75=np.nan,
                     source="REPORT_2026-08-13.md Sec 3.4 (no scored artifact)"))
    D = pd.DataFrame(rows)
    D.to_csv(RES / "s2b_splits.csv", index=False)
    return D


def s2c_generators():
    """Muc 2c: ho model sinh AI. LUU Y -- artifact chi luu HO (provider), khong luu
    ten model + temperature cua tung dong, nen bang nay dung o muc ho."""
    base = ART / "artifacts_roberta_large_seed42"
    rows = []
    for name in ["val", "test_ood_model"]:
        d = pd.read_parquet(base / ("pred_%s.parquet" % name))
        ai = d[d.label == 1]
        for fam, n in ai.generator_family.value_counts().items():
            rows.append(dict(split=name, generator_family=fam, n_segments=int(n),
                             pct_of_ai=100.0 * n / len(ai)))
    D = pd.DataFrame(rows)
    D.to_csv(RES / "s2c_generators.csv", index=False)
    return D


def s4_ranges():
    """Muc 4: range per-checkpoint cho MOI bang chinh cua bai."""
    rows = []
    S = pd.read_csv(OUT / "pihat_sanity_16models.csv")
    for lab, s in [("Table I: mean pi-hat, human", S.pihat_mean_human),
                   ("Table I: mean pi-hat, AI", S.pihat_mean_ai),
                   ("Table I: median pi-hat, human", S.pihat_median_human),
                   ("Table I: median pi-hat, AI", S.pihat_median_ai)]:
        rows.append(dict(table="I", quantity=lab, unit="pi",
                         median=s.median(), lo=s.min(), hi=s.max(), n_ckpt=len(s)))

    M = pd.read_csv(OUT / "mean_ktransfer_16models.csv")
    for lab, s in [("Segment level (val, k~1): FPR", M.fpr_tau_chunk),
                   ("Segment level (val, k~1): miss", M.miss_tau_chunk),
                   ("k~22.6 (test_fp_full): FPR", M.fpr_tau_mean_k),
                   ("k~22.6 (test_fp_full): miss", M.miss_tau_mean_k),
                   ("Correct k~2.38 (control): FPR", M.fpr_tau_ood),
                   ("Correct k~2.38 (control): miss", M.miss_tau_ood)]:
        rows.append(dict(table="II", quantity=lab, unit="%",
                         median=100 * s.median(), lo=100 * s.min(), hi=100 * s.max(),
                         n_ckpt=len(s)))

    H = pd.read_csv(OUT / "static_vs_hartung_by_k.csv")
    # Main Table III's "Emp." is REV-2 (empirical q99 at k=1, then design effect),
    # NOT fpr_emp from paper_level_tau_k.py, which is a separately calibrated per-k
    # oracle. The previous supplement mislabeled that oracle as the REV-2 baseline.
    E = pd.read_csv(
        ROOT / "revision_experiments_rivf2026" / "results" / "rev12_per_checkpoint.csv")
    E = (E[E.method == "emp_deff"]
           .groupby(["model", "k"], as_index=False).fpr.mean())
    for _, r in H.iterrows():
        for meth, mid, lo, hi in [("Static", "fpr_const_median_pct", "fpr_const_min_pct", "fpr_const_max_pct"),
                                  ("z-scale (proposed)", "fpr_hartung_median_pct", "fpr_hartung_min_pct", "fpr_hartung_max_pct")]:
            rows.append(dict(table="III", quantity="k=%d, %s: FPR" % (r.k, meth), unit="%",
                             median=r[mid], lo=r[lo], hi=r[hi], n_ckpt=int(r.n_checkpoints)))
        emp = 100 * E[E.k == r.k].fpr
        rows.append(dict(table="III",
                         quantity="k=%d, Raw scale, empirical quantile: FPR" % r.k,
                         unit="%", median=emp.median(), lo=emp.min(), hi=emp.max(),
                         n_ckpt=len(emp)))

    O = pd.read_csv(OUT / "hartung_ood_16models.csv")
    for lab, s in [("z-score (out of sample): FPR", O.fpr_hartung),
                   ("z-score (out of sample): miss", O.miss_hartung)]:
        rows.append(dict(table="IV", quantity=lab, unit="%",
                         median=100 * s.median(), lo=100 * s.min(), hi=100 * s.max(),
                         n_ckpt=len(s)))
    for lab, s in [("Correct k~2.38 (control): FPR", M.fpr_tau_ood),
                   ("Correct k~2.38 (control): miss", M.miss_tau_ood)]:
        rows.append(dict(table="IV", quantity=lab, unit="%",
                         median=100 * s.median(), lo=100 * s.min(), hi=100 * s.max(),
                         n_ckpt=len(s)))

    D = pd.DataFrame(rows)
    D.to_csv(RES / "s4_ranges_main_tables.csv", index=False)
    return D


def s6_pool():
    """Muc 6.2: kich thuoc pool hieu chuan m va do phan giai p-value o duoi phai."""
    T = pd.read_csv(OUT / "tau_k_mean_16models.csv")
    val = pd.read_parquet(ART / "artifacts_roberta_large_seed42" / "pred_val.parquet")
    m_val = int((val.label == 0).sum())
    rows = [
        dict(used_in="Sec V-A / V-C (controlled experiment)",
             pool="human segments in the calibration half of test_fp_full",
             m_median=float(T.n_cal_chunk.median()), m_min=int(T.n_cal_chunk.min()),
             m_max=int(T.n_cal_chunk.max())),
        dict(used_in="Sec V-D (out-of-sample transfer)",
             pool="human segments of val", m_median=float(m_val),
             m_min=m_val, m_max=m_val),
    ]
    for r in rows:
        m = r["m_median"]
        r["p_min"] = 0.5 / (m + 1.0)                 # p nho nhat co the dat
        r["n_steps_below_alpha"] = 0.01 * (m + 1.0)  # so buoc p roi rac trong (0, 0.01]
        r["z_max"] = float(__import__("scipy.stats", fromlist=["norm"]).norm.isf(r["p_min"]))
    D = pd.DataFrame(rows)
    D.to_csv(RES / "s6_calibration_pool.csv", index=False)
    return D


def s6_multiple_testing():
    """Muc 6.1: McNemar x16 -- tho vs Bonferroni vs Benjamini-Hochberg."""
    M = pd.read_csv(ROOT / "revision_experiments_rivf2026" / "results" / "rev3_paired_miss.csv")
    p = M.p_mcnemar.to_numpy()
    n = len(p)
    order = np.argsort(p)
    bh_thresh = (np.arange(1, n + 1) / n) * 0.05
    passes = p[order] <= bh_thresh
    kmax = int(np.max(np.where(passes)[0]) + 1) if passes.any() else 0
    bh_reject = np.zeros(n, bool)
    bh_reject[order[:kmax]] = True

    D = M[["model", "miss_z", "miss_ctrl", "b_only_z", "c_only_ctrl", "p_mcnemar"]].copy()
    D["sig_raw"] = D.p_mcnemar < 0.05
    D["sig_bonferroni"] = D.p_mcnemar < 0.05 / n
    D["sig_bh_q05"] = bh_reject
    D = D.sort_values("p_mcnemar")
    D.to_csv(RES / "s6_multiple_testing.csv", index=False)
    return D


def main():
    RES.mkdir(parents=True, exist_ok=True)
    D6b = s6_multiple_testing()
    D1 = s1_checkpoints()
    D1b = s1b_document_results(D1, D6b)
    print("=" * 88)
    print("S1  16 CHECKPOINT")
    print("=" * 88)
    print(D1[["id", "architecture", "seed", "type", "auroc_ood", "tpr_at_tau_ood",
              "mu0_val", "mu1_val", "rho_z_mean"]].to_string(index=False,
                                                             float_format=lambda x: "%.4f" % x))
    print("\nS1b  DOCUMENT-LEVEL RESULTS")
    print(D1b[["id", "architecture", "seed", "pihat_mean_human", "pihat_mean_ai",
               "rho_z_median", "fpr_z", "miss_z", "fpr_ctrl", "miss_ctrl",
               "p_mcnemar"]].to_string(index=False,
                                        float_format=lambda x: "%.4f" % x))

    D2b = s2b_splits()
    print("\n" + "=" * 88); print("S2b  SPLIT"); print("=" * 88)
    print(D2b.to_string(index=False))

    D2c = s2c_generators()
    print("\n" + "=" * 88); print("S2c  HO MODEL SINH AI"); print("=" * 88)
    print(D2c.to_string(index=False, float_format=lambda x: "%.1f" % x))

    s2a_schema()
    D4 = s4_ranges()
    print("\n" + "=" * 88); print("S4  RANGE PER-CHECKPOINT (trich)"); print("=" * 88)
    print(D4[D4.table.isin(["II", "IV"])].to_string(index=False,
                                                    float_format=lambda x: "%.2f" % x))

    D6 = s6_pool()
    print("\n" + "=" * 88); print("S6  POOL HIEU CHUAN m"); print("=" * 88)
    print(D6.to_string(index=False))

    print("\n" + "=" * 88); print("S6  DA KIEM DINH"); print("=" * 88)
    print("  tho p<0.05      : %2d/16" % D6b.sig_raw.sum())
    print("  Bonferroni      : %2d/16" % D6b.sig_bonferroni.sum())
    print("  Benjamini-Hochb.: %2d/16" % D6b.sig_bh_q05.sum())
    print("\n-> supplement_rivf2026/results/")


if __name__ == "__main__":
    main()
