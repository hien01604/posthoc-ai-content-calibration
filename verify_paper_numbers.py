"""Tinh lai MOI con so trong paper/rivf2026.tex tu artifact goc, roi doi chieu.

Dung chung cho `verify_paper_numbers.ipynb` (notebook goi tung ham) va cho chay
thang: `py verify_paper_numbers.py`.

Moi ham tra ve list cac dong ket qua; `chk()` in ra dang
    [OK ] ten .............. tinh duoc | trong bai
    [!! ] ...
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
FEX = BASE / "final_experiments_rivf2026" / "results"
REV = BASE / "revision_experiments_rivf2026" / "results"
FP_FULL = BASE / "artifacts" / "kaggle_testfp_full" / "pred_testfp_full"
ART = BASE / "artifacts" / "kaggle"

KS = [1, 2, 3, 5, 8, 10, 15, 20]
_tally = {"ok": 0, "bad": 0}


def chk(name, got, paper, tol=0.005):
    """So sanh mot dai luong voi gia tri in trong bai."""
    import math
    def _bad(a, b):
        try:
            a = float(a)
        except (TypeError, ValueError):
            return True
        return math.isnan(a) or abs(a - b) > tol
    if isinstance(got, (list, tuple)):
        bad = any(_bad(a, b) for a, b in zip(got, paper))
        gs = " ".join("%.3f" % v for v in got)
        ps = " ".join("%.3f" % v for v in paper)
    else:
        bad = _bad(got, paper)
        gs, ps = "%.4f" % got, "%.4f" % paper
    _tally["bad" if bad else "ok"] += 1
    print("  [%s] %-42s %s | bai: %s" % ("!!" if bad else "OK", name, gs, ps))


def summary():
    n = _tally["ok"] + _tally["bad"]
    print("\n" + "=" * 72)
    print("TONG: %d dai luong, %d khop, %d LECH" % (n, _tally["ok"], _tally["bad"]))
    if _tally["bad"]:
        print("=> con so co dau [!!] phai sua trong bai hoac trong code")
    print("=" * 72)


# ---------------------------------------------------------------- §III-B
def sec3b():
    print("\n### III-B  16 checkpoint")
    rows = []
    for d in sorted(ART.glob("artifacts_*")):
        j = d / "metrics.json"
        if j.exists():
            import json
            m = json.loads(j.read_text())
            for e in m["metrics"]:
                if e.get("split") == "test_ood_model":
                    rows.append((d.name, e["auroc"], e["tpr@tau"], e["fpr@tau"]))
    R = pd.DataFrame(rows, columns=["model", "auroc", "tpr", "fpr"]).dropna()
    print("  n checkpoint = %d" % len(R))
    chk("AUROC min", R.auroc.min(), 0.9746, 0.0005)
    chk("AUROC max", R.auroc.max(), 0.9875, 0.0005)
    chk("TPR@tau min", R.tpr.min(), 0.8435, 0.0005)
    chk("TPR@tau max", R.tpr.max(), 0.9254, 0.0005)
    print("  FPR@tau thuc te (OOD): %.2f%% - %.2f%%  (nguong chot o 1%% tren val)"
          % (100 * R.fpr.min(), 100 * R.fpr.max()))


# ---------------------------------------------------------------- §IV-A
def sec4a():
    print("\n### IV-A  pi-hat co tam dung")
    d = pd.read_csv(OUT / "pihat_sanity_16models.csv")
    chk("mean pi-hat, human", d.pihat_mean_human.mean(), 0.007, 0.001)
    chk("mean pi-hat, AI", d.pihat_mean_ai.mean(), 0.982, 0.001)
    chk("range human lo", d.pihat_mean_human.min(), -0.009, 0.001)
    chk("range human hi", d.pihat_mean_human.max(), 0.021, 0.001)
    chk("range AI lo", d.pihat_mean_ai.min(), 0.966, 0.001)
    chk("range AI hi", d.pihat_mean_ai.max(), 0.995, 0.001)
    f = sorted(ART.glob("artifacts_*/pred_test_ood_model.parquet"))[0]
    g = pd.read_parquet(f).groupby(["paper_id", "label"]).size()
    lv = g.index.get_level_values("label")
    print("  don vi phan tich: %d bai/lop, %d checkpoint-document predictions"
          % ((lv == 0).sum(), (lv == 0).sum() * 16))


# ---------------------------------------------------------------- §IV-B
def sec4b():
    print("\n### IV-B  khoang tin cay cho pi-hat (Table I)")
    c = pd.read_csv(FEX / "f1_coverage_null.csv")
    pm = c.groupby(["model", "k"])[["cov_naive", "cov_deff", "cov_emp"]].mean().reset_index()
    m = pm.groupby("k")[["cov_naive", "cov_deff", "cov_emp"]].median() * 100
    for k, want in [(1, (95.7, 95.7, 94.8)), (5, (90.5, 93.3, 94.8)),
                    (10, (89.5, 93.5, 94.8)), (20, (86.7, 93.8, 94.8))]:
        chk("k=%-2d Gauss/+deff/emp" % k,
            [m.loc[k, "cov_naive"], m.loc[k, "cov_deff"], m.loc[k, "cov_emp"]],
            list(want), 0.15)
    f = pd.read_csv(FEX / "f6_coverage_mixed_emp.csv")
    pm = f.groupby(["model", "k", "true_pi"])[["cov_emp_null"]].mean().reset_index()
    m20 = pm[pm.k == 20].groupby("true_pi").cov_emp_null.median() * 100
    chk("chuyen sang pi>0: pi=0/0.25/1",
        [m20[0.0], m20[0.25], m20[1.0]], [94.6, 79.6, 67.2], 0.15)
    # Table IV in ca 5 cot tai k=20 -- kiem het, vi day chinh la cho tung truot
    # pham vi (dai qua MOI k khac dai chi tai k=20).
    cols = ["cov_emp_null", "cov_emp_plugin", "cov_emp_pi", "cov_deff", "cov_naive"]
    t4 = f.groupby(["model", "k", "true_pi"])[cols].mean().reset_index()
    t4 = t4[t4.k == 20].groupby("true_pi")[cols].median() * 100
    for pi, want in [(0.00, [94.6, 93.0, 94.6, 94.0, 87.7]),
                     (0.25, [79.6, 93.2, 95.1, 95.9, 85.6]),
                     (0.50, [72.5, 94.1, 94.9, 97.2, 84.7]),
                     (0.75, [76.4, 93.7, 94.9, 98.7, 86.9]),
                     (1.00, [67.2, 93.8, 94.7, 97.4, 86.4])]:
        chk("Table IV, pi=%.2f" % pi, list(t4.loc[pi]), want, 0.15)


# ---------------------------------------------------------------- §IV-C
def sec4c():
    print("\n### IV-C  ba nguon nguong (Table II)")
    d = pd.read_csv(OUT / "mean_ktransfer_16models.csv")
    med = d[["fpr_tau_chunk", "miss_tau_chunk", "fpr_tau_mean_k",
             "miss_tau_mean_k", "fpr_tau_ood", "miss_tau_ood"]].median() * 100
    chk("median 6 o cua hang Median", list(med),
        [0.24, 17.82, 5.10, 1.10, 1.31, 6.16], 0.02)
    chk("miss control: min", 100 * d.miss_tau_ood.min(), 3.85, 0.02)
    chk("miss control: max", 100 * d.miss_tau_ood.max(), 20.50, 0.02)


# ---------------------------------------------------------------- §V-A
def sec5a():
    print("\n### V-A  FPR theo k")
    f5 = pd.read_csv(FEX / "f5_deff_necessity.csv")
    m = (f5.groupby(["method", "model", "k"]).fpr_pct.mean().reset_index()
         .groupby(["method", "k"]).fpr_pct.median().unstack("method"))
    print("  k    static  rawGauss  rawEmp  zscale  oracle  z_norho")
    p = pd.read_csv(OUT / "pihat_static_vs_corrected_by_k.csv").groupby("k")[
        ["fpr_static", "fpr_corrected"]].median() * 100
    for k in KS:
        print("  %-4d %6.2f %9.2f %7.2f %7.2f %7.2f %8.2f"
              % (k, p.loc[k, "fpr_static"], p.loc[k, "fpr_corrected"],
                 m.loc[k, "raw_emp"], m.loc[k, "z_hartung"],
                 m.loc[k, "oracle"], m.loc[k, "z_norho"]))
    chk('cot "Emp." dai [min,max]',
        [m.raw_emp.min(), m.raw_emp.max()], [0.33, 1.10], 0.02)
    chk("Gauss thang tho k=1 -> k=20",
        [p.loc[1, "fpr_corrected"], p.loc[20, "fpr_corrected"]], [4.24, 5.35], 0.02)
    chk("z-scale k=1 -> k=20",
        [m.loc[1, "z_hartung"], m.loc[20, "z_hartung"]], [0.97, 0.76], 0.02)
    chk("bo rho: z-scale k=20", m.loc[20, "z_norho"], 18.28, 0.05)
    chk("bo rho: raw k=20", m.loc[20, "raw_norho"], 2.54, 0.05)


# ---------------------------------------------------------------- §V-B  (rho)
def icc_variants(papers, k0=20):
    G = np.array([p[:k0] for p in papers if len(p) >= k0], dtype=float)
    n, k = G.shape
    msw = float(np.mean(G.var(axis=1, ddof=1)))
    msb = float(k * np.sum((G.mean(1) - G.mean()) ** 2) / (n - 1))
    var_trunc = float(G.ravel().var(ddof=1))
    var_pool = float(np.concatenate([p for p in papers if len(p) >= k0]).var(ddof=1))
    return (max(0.0, 1 - msw / var_pool),                       # codebase (LOI)
            max(0.0, (msb - msw) / (msb + (k - 1) * msw)))      # ICC(1) chuan


def sec5b(n_models=16):
    print("\n### V-B  rho -- kiem tra loi estimator")
    a, c = [], []
    for f in sorted(FP_FULL.glob("pred_test_fp_full_artifacts_*.parquet"))[:n_models]:
        d = pd.read_parquet(f).dropna(subset=["score"])
        papers = [g.score.to_numpy() for _, g in d.groupby("paper_id")]
        x, y = icc_variants(papers)
        a.append(x)
        c.append(y)
    print("  rho_raw, codebase (1-MSW/Var_pool) : median %.4f  [%.4f - %.4f]"
          % (np.median(a), min(a), max(a)))
    print("  rho_raw, ICC(1) ANOVA chuan        : median %.4f  [%.4f - %.4f]"
          % (np.median(c), min(c), max(c)))
    chk("rho_raw (ICC(1)) nhu bai dang viet", float(np.median(c)), 0.132, 0.01)
    t = pd.read_csv(OUT / "tau_k_mean_16models.csv")
    rz = t.groupby("model").rho.mean()
    chk("rho_z (Hartung) median", float(rz.median()), 0.303, 0.005)
    print("  -> ti so dung mot estimator (ICC(1)): %.4f / %.4f = %.2fx"
          % (rz.median(), np.median(c), rz.median() / np.median(c)))
    print("     bai dang noi 'factor of seven' (0.296/0.044 = 6.7x)")


# ---------------------------------------------------------------- §V-C
def sec5c():
    print("\n### V-C  chuyen ra ngoai mau + power")
    o = pd.read_csv(OUT / "hartung_ood_16models.csv")
    chk("FPR z-scale OOD (median)", 100 * o.fpr_hartung.median(), 1.26, 0.02)
    chk("miss z-scale OOD (median)", 100 * o.miss_hartung.median(), 3.77, 0.02)
    r3 = pd.read_csv(REV / "rev3_paired_miss.csv")
    print("  rev3 (khong khop FPR): z thang %d/16, miss %.2f%% vs %.2f%%"
          % ((r3.miss_z < r3.miss_ctrl).sum(), 100 * r3.miss_z.median(),
             100 * r3.miss_ctrl.median()))
    r4 = pd.read_csv(REV / "rev4_fpr_matched.csv")
    print("  rev4 (khop FPR tuyet doi): z thang %d/16, %d test p<0.05 nghieng ve z, "
          "%d nghieng ve control"
          % ((r4.delta_miss > 0).sum(),
             ((r4.p_mcnemar < 0.05) & (r4.delta_miss > 0)).sum(),
             ((r4.p_mcnemar < 0.05) & (r4.delta_miss < 0)).sum()))
    chk("miss khop FPR: z / control",
        [100 * r4.miss_z.median(), 100 * r4.miss_ctrl.median()], [3.77, 5.02], 0.02)
    chk("discordant b / c", [r4.b_only_z.sum(), r4.c_only_ctrl.sum()], [419, 90], 0.5)


# ---------------------------------------------------------------- §V-D
def sec5d():
    print("\n### V-D  conformal theo bin do dai")
    c = pd.read_csv(FEX / "f3_conformal_overall.csv")
    pm = c.groupby(["model", "n_bins"])[
        ["fpr_conformal", "fpr_zscale", "fpr_static"]].mean().reset_index()
    for nb in sorted(pm.n_bins.unique()):
        g = pm[pm.n_bins == nb][["fpr_conformal", "fpr_zscale", "fpr_static"]].median() * 100
        print("  %2d bin: conformal %.2f%%  z-scale %.2f%%  static %.2f%%"
              % (nb, g.fpr_conformal, g.fpr_zscale, g.fpr_static))
    g5 = pm[pm.n_bins == 5][["fpr_conformal", "fpr_zscale", "fpr_static"]].median() * 100
    chk("5 bin: conformal / z / static",
        [g5.fpr_conformal, g5.fpr_zscale, g5.fpr_static], [0.97, 0.77, 1.25], 0.03)
    inf10 = c[c.n_bins == 10].bins_infeasible.mean()
    print("  10 bin: trung binh %.1f/10 bin khong kha thi" % inf10)


# ------------------------------------------------- V-A: p-value bootstrap
def sec5a_boot():
    """Cac p-value va do lech RMS/worst-case cua so sanh ghep cap (f7)."""
    print("\n### V-A  bootstrap ghep cap (p-value)")
    d = pd.read_csv(FEX / "f7_cluster_scalar.csv")
    d = d[d.method.str.contains("-")].set_index(["method", "stat"])

    def g(m, st):
        r = d.loc[(m, st)]
        return float(r.value), float(r.ci_lo), float(r.ci_hi), float(r.p_boot)

    v, lo, hi, p = g("zscale - emp_deff", "rmsdev")
    chk("z vs emp: rmsdev, CI, p", [v, lo, hi, p], [-0.44, -0.53, -0.02, 0.044], 0.01)
    v, lo, hi, p = g("zscale - emp_deff", "maxdev")
    chk("z vs emp: maxdev, CI, p", [v, lo, hi, p], [-0.43, -0.58, -0.02, 0.047], 0.01)
    v, lo, hi, p = g("zscale - oracle", "rmsdev")
    chk("z vs oracle: rmsdev, CI", [v, lo, hi], [-0.12, -0.20, 0.14], 0.01)
    v, lo, hi, p = g("zscale - oracle", "maxdev")
    chk("z vs oracle: maxdev, CI", [v, lo, hi], [-0.08, -0.26, 0.26], 0.01)
    _, _, _, p = g("zscale - static", "rmsdev")
    print("  [%s] %-42s %.4f | bai: p<0.001"
          % ("OK" if p < 0.001 else "!!", "z vs static: p", p))
    _tally["ok" if p < 0.001 else "bad"] += 1


# ------------------------------------------------- V-B: ngan sach hieu chuan rho_z
def sec5b_budget():
    """Do chinh xac cua rho_z theo so tai lieu hieu chuan (f8) -- so moi nhat trong bai."""
    print("\n### V-B  ngan sach hieu chuan cho rho_z")
    f = FEX / "f8_rho_sample_size.csv"
    if not f.exists():
        print("  !! chua co f8_rho_sample_size.csv -- chay f8_rho_sample_size.py truoc")
        _tally["bad"] += 1
        return
    d = pd.read_csv(f)
    g = d.groupby("n_cal").agg(sd=("rho_sd", "median"), rho=("rho_mean", "median"))
    g["pct"] = 100 * 1.96 * g.sd / g.rho
    for n, want in [(25, 54.0), (100, 27.0), (200, 17.0)]:
        chk("+-95%% cua rho_z tai n_cal=%d" % n, float(g.loc[n, "pct"]), want, 1.0)
    ok = g[g.pct <= 20]
    n_min = int(ok.index.min()) if len(ok) else -1
    print("  [%s] %-42s %d | bai: 200"
          % ("OK" if n_min == 200 else "!!", "n_cal nho nhat dat +-20%", n_min))
    _tally["ok" if n_min == 200 else "bad"] += 1


# ------------------------------------------------- IV-C: TOAN BO 96 o cua Table V
def sec4c_cells():
    """Kiem tung o cua Table V, khong chi hang Median."""
    print("\n### IV-C  Table V, toan bo 96 o")
    d = pd.read_csv(OUT / "mean_ktransfer_16models.csv")
    cols = ["fpr_tau_chunk", "miss_tau_chunk", "fpr_tau_mean_k",
            "miss_tau_mean_k", "fpr_tau_ood", "miss_tau_ood"]
    tex = (BASE / "paper" / "rivf2026.tex").read_text(encoding="utf-8")
    import re
    body = tex[tex.index("label{tab:ktransfer}"):]
    body = body[:body.index("bottomrule")]
    rows = re.findall(r"^([A-Z][^&\n]*?) & ([\d.]+) & ([\d.]+) & ([\d.]+) & "
                      r"([\d.]+) & ([\d.]+) & ([\d.]+) \\\\$", body, re.M)
    if len(rows) != 16:
        print("  !! doc duoc %d dong tu bang, mong doi 16" % len(rows))
        _tally["bad"] += 1
        return
    csv_vals = sorted(tuple(round(100 * v, 2) for v in r)
                      for r in d[cols].to_numpy())
    tex_vals = sorted(tuple(float(x) for x in r[1:]) for r in rows)
    bad = sum(1 for a, b in zip(csv_vals, tex_vals)
              if any(abs(x - y) > 0.005 for x, y in zip(a, b)))
    print("  [%s] %-42s %d o lech / 96" % ("OK" if bad == 0 else "!!",
                                           "moi o cua Table V khop CSV", bad * 6))
    _tally["ok" if bad == 0 else "bad"] += 1


# ------------------------------------------------- V-D: conformal
def sec5d_full():
    """Cac con so §V-D: FPR 5 bin, tran ngan sach bin, 10 bin bat kha thi."""
    print("\n### V-D  conformal, day du")
    c = pd.read_csv(FEX / "f3_conformal_overall.csv")
    pm = c.groupby(["model", "n_bins"])[
        ["fpr_conformal", "fpr_zscale", "fpr_static"]].mean().reset_index()
    g5 = pm[pm.n_bins == 5][["fpr_conformal", "fpr_zscale", "fpr_static"]].median() * 100
    chk("5 bin: conformal / z / static",
        [g5.fpr_conformal, g5.fpr_zscale, g5.fpr_static], [0.97, 0.77, 1.25], 0.03)
    r5 = pm[pm.n_bins == 5]
    chk("5 bin: dai conformal", [100 * r5.fpr_conformal.min(), 100 * r5.fpr_conformal.max()],
        [0.70, 1.10], 0.03)
    rz = pm[pm.n_bins == 5]
    chk("5 bin: dai z-scale", [100 * rz.fpr_zscale.min(), 100 * rz.fpr_zscale.max()],
        [0.27, 1.77], 0.03)
    inf10 = c[c.n_bins == 10].bins_infeasible.mean()
    print("  [%s] %-42s %.1f/10 | bai: moi bin deu thieu"
          % ("OK" if inf10 >= 9.99 else "!!", "10 bin: so bin bat kha thi", inf10))
    _tally["ok" if inf10 >= 9.99 else "bad"] += 1
    import math
    need = math.ceil(1 / 0.01) - 1
    chk("so tai lieu can moi bin", need, 99, 0.5)


def run_all():
    for fn in (sec3b, sec4a, sec4b, sec4c, sec4c_cells, sec5a, sec5a_boot,
               sec5b, sec5b_budget, sec5c, sec5d, sec5d_full):
        try:
            fn()
        except Exception as e:
            print("\n### %s -- LOI: %s: %s" % (fn.__name__, type(e).__name__, e))
    summary()


if __name__ == "__main__":
    run_all()
