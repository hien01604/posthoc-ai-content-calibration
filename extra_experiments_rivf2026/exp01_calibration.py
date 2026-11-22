"""EXP-01 -- Kiem chung pi_hat o cac ti le TRUNG GIAN (duong hieu chuan).

    python3 exp01_calibration.py

VAN DE
------
pi_hat moi duoc kiem o hai muc pi=0 va pi=1 (Table I). Tieu de bai ("AI-generated
content fraction") chua co bang chung nao cho 0 < pi < 1. Day la lo hong lon nhat.

CACH LAM -- HAU KY HOAN TOAN, KHONG CHAY LAI DETECTOR
-----------------------------------------------------
Dung lai diem so DA CO tren dia. Tai lieu tong hop duoc dung bang cach ghep lai cac
doan human/AI da duoc cham diem san.

HAI THIET KE (bao cao ca hai -- chung tra loi hai cau hoi khac nhau)
-------------------------------------------------------------------
  (P) GHEP CAP -- `test_ood_model` co 1.514 SLOT, moi slot co CA hai phien ban cua
      cung mot doan: ban human va ban AI (khoi label=0 dong 0..1513 ghep 1-1 voi khoi
      label=1 dong 1514..3027; moi paper co #human == #ai). Mot tai lieu pi gom k slot,
      trong do round(pi*k) slot dung PHIEN BAN AI, con lai dung PHIEN BAN HUMAN.
      -> Chi co MOT thu thay doi giua cac muc pi: bao nhieu slot bi thay bang ban AI.
      Day la thiet ke co kiem soat chat nhat.

  (R) CHUYEN QUAN THE -- doan human lay tu MOT bai THAT trong `test_fp_full` (>=20
      doan/bai, giu nguyen cau truc trong-bai), doan AI lay tu pool AI cua
      `test_ood_model`. Hai nua den tu hai split khac nhau -> kiem duoc ca viec neo
      mu0/mu1 co chuyen quan the duoc khong.

CANH BAO PHAI DOC (dung overclaim khi viet bai)
-----------------------------------------------
pi_hat = (s_tb - mu0)/(mu1 - mu0) la ham AFFINE cua trung binh diem. Voi thiet ke (P),
so doan AI duoc CHOT bang round(pi*k), nen ky vong

    E[pi_hat | pi] = pi * E[pi_hat | pi=1] + (1-pi) * E[pi_hat | pi=0]

la dung CHINH XAC ve mat toan hoc, khong phai phat hien thuc nghiem. Vi vay:
  - Cot `mean_pi_hat` gan nhu tuyen tinh la TAT YEU, KHONG duoc ban nhu bang chung moi.
  - Thu THUC SU moi va dang bao cao la: (a) do TAN (p5-p95, MAE) o cac muc pi trung
    gian -- khong suy ra duoc tu hai dau mut; (b) TRUNG VI, vi trung vi khong tuyen
    tinh; (c) sai so do LAM TRON round(pi*k); (d) thiet ke (R), noi hai nua den tu hai
    quan the khac nhau nen tinh tuyen tinh KHONG con duoc dam bao.

DAU VAO
-------
  artifacts/kaggle/artifacts_<ckpt>/pred_test_ood_model.parquet   1.514 cap human/AI
  artifacts/kaggle_testfp_full/pred_testfp_full/...parquet        bai human day du
  output/pihat_sanity_16models.csv                                mu0/mu1 uoc TU `val`

DAU RA
------
  results/exp01_calibration_table.csv    true_pi, mean_pi_hat, median_pi_hat, mae,
                                         bias, p5, p95, n  (moi thiet ke)
  results/exp01_per_checkpoint.csv       tach theo tung checkpoint
  figures/exp01_calibration_curve.pdf/.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # extra_experiments_rivf2026/
ROOT = HERE.parent                              # goc repo
ART = ROOT / "artifacts" / "kaggle"
FPF = ROOT / "artifacts" / "kaggle_testfp_full" / "pred_testfp_full"
OUT = ROOT / "output"
RES = HERE / "results"
FIG = HERE / "figures"

PIS = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
K = 20                # so doan/tai lieu -- khop do dai cua tap 576 bai co kiem soat
NDOC = 500            # tai lieu tong hop moi (checkpoint, pi, thiet ke); yeu cau >=200
SEED = 20260825


def anchors():
    """mu0/mu1 uoc CHI tu `val` -- doc lai, khong uoc lai."""
    return pd.read_csv(OUT / "pihat_sanity_16models.csv").set_index("model")


def paired_pools(model):
    """(sH, sA): diem human va diem AI cua CUNG 1.514 slot, da ghep cap."""
    d = pd.read_parquet(ART / ("artifacts_%s" % model) / "pred_test_ood_model.parquet")
    h = d[d.label == 0].reset_index(drop=True)
    a = d[d.label == 1].reset_index(drop=True)
    assert len(h) == len(a), "so doan human != so doan AI"
    # kiem tra cap dung: paper_id phai trung nhau theo tung vi tri
    assert (h.paper_id.to_numpy() == a.paper_id.to_numpy()).all(), "cap bi lech paper_id"
    return h.score.to_numpy(), a.score.to_numpy()


def real_human_papers(model, kmin=K):
    """List mang diem cua tung bai THAT trong test_fp_full co >= kmin doan."""
    f = FPF / ("pred_test_fp_full_artifacts_%s.parquet" % model)
    d = pd.read_parquet(f).dropna(subset=["score"])
    return [g.score.to_numpy() for _, g in d.groupby("paper_id") if len(g) >= kmin]


def mask_with_exactly(n_rows, k, n_true, rng):
    """(n_rows, k) bool, moi hang co DUNG n_true o True, vi tri ngau nhien."""
    m = np.zeros((n_rows, k), dtype=bool)
    if n_true:
        idx = np.argsort(rng.random((n_rows, k)), axis=1)[:, :n_true]
        np.put_along_axis(m, idx, True, axis=1)
    return m


def design_paired(sH, sA, pi, rng):
    """(P) k slot rut khong hoan lai; round(pi*k) slot dung ban AI."""
    n = len(sH)
    slots = np.argsort(rng.random((NDOC, n)), axis=1)[:, :K]      # khong hoan lai
    isai = mask_with_exactly(NDOC, K, int(round(pi * K)), rng)
    s = np.where(isai, sA[slots], sH[slots])
    return s.mean(axis=1)


def design_transfer(papers, sA, pi, rng):
    """(R) doan human tu MOT bai that; doan AI tu pool AI cua ood."""
    nai = int(round(pi * K))
    nhu = K - nai
    tot = np.zeros(NDOC)
    if nhu:
        pick = rng.integers(0, len(papers), NDOC)
        for i, p in enumerate(pick):
            v = papers[p]
            tot[i] += v[rng.choice(len(v), nhu, replace=False)].sum()
    if nai:
        idx = np.argsort(rng.random((NDOC, len(sA))), axis=1)[:, :nai]
        tot += sA[idx].sum(axis=1)
    return tot / K


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    anc = anchors()
    rows = []

    for model, r in anc.iterrows():
        rng = np.random.default_rng(SEED)
        sH, sA = paired_pools(model)
        papers = real_human_papers(model)
        denom = r.mu1 - r.mu0
        for pi in PIS:
            # Neu round(pi*K) != pi*K thi tai lieu KHONG dat dung ti le pi va moi so do
            # se lech co he thong. Da vap dung loi nay o EXP-02 voi k=5 (pi=0,10 -> 0/5).
            assert int(round(pi * K)) == pi * K, \
                "pi=%s khong dat chinh xac o k=%d" % (pi, K)
            for tag, mean_s in (("paired", design_paired(sH, sA, pi, rng)),
                                ("transfer", design_transfer(papers, sA, pi, rng))):
                ph = (mean_s - r.mu0) / denom
                rows.append(dict(
                    design=tag, model=model, true_pi=pi, n=len(ph),
                    mean_pi_hat=ph.mean(), median_pi_hat=float(np.median(ph)),
                    mae=float(np.abs(ph - pi).mean()), bias=float(ph.mean() - pi),
                    p5=float(np.percentile(ph, 5)), p95=float(np.percentile(ph, 95)),
                    sd=float(ph.std(ddof=1)),
                ))
        print("  xong %s (n_pair=%d, n_paper>=%d=%d)" % (model, len(sH), K, len(papers)))

    D = pd.DataFrame(rows)
    D.to_csv(RES / "exp01_per_checkpoint.csv", index=False)

    # gop qua 16 checkpoint: trung vi cua thong ke tung checkpoint
    agg = (D.groupby(["design", "true_pi"])
             .agg(mean_pi_hat=("mean_pi_hat", "mean"),
                  median_pi_hat=("median_pi_hat", "median"),
                  mae=("mae", "median"),
                  bias=("bias", "mean"),
                  p5=("p5", "median"), p95=("p95", "median"),
                  sd=("sd", "median"),
                  n=("n", "sum"))
             .reset_index())
    agg.to_csv(RES / "exp01_calibration_table.csv", index=False)

    # do lech so voi duong noi HAI DAU MUT (cau hoi thuc su cua bai)
    print("\n%-9s %5s %8s %8s %7s %8s %8s %8s" %
          ("design", "pi", "mean", "median", "MAE", "bias", "p5", "p95"))
    for tag in ("paired", "transfer"):
        a = agg[agg.design == tag].set_index("true_pi")
        lo, hi = a.loc[0.0, "mean_pi_hat"], a.loc[1.0, "mean_pi_hat"]
        for pi in PIS:
            r = a.loc[pi]
            print("%-9s %5.2f %8.4f %8.4f %7.4f %8.4f %8.4f %8.4f" %
                  (tag, pi, r.mean_pi_hat, r.median_pi_hat, r.mae, r.bias, r.p5, r.p95))
        dev = max(abs(a.loc[pi, "mean_pi_hat"] - (lo + pi * (hi - lo))) for pi in PIS)
        print("   -> lech toi da so voi duong noi hai dau mut: %.4f" % dev)
        print("   -> MAE trung vi qua moi muc pi: %.4f\n" % a.mae.median())

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot([0, 1], [0, 1], color="0.55", linestyle=":", linewidth=1.4,
            label="ideal ($\\hat{\\pi}=\\pi$)", zorder=1)
    style = {"paired": ("#1f77b4", "o", "-", "Paired design (same slots)"),
             "transfer": ("#d62728", "s", "--", "Transfer design (human from full papers)")}
    for tag, (c, mk, ls, lab) in style.items():
        a = agg[agg.design == tag].set_index("true_pi")
        x = np.array(PIS)
        y = a.loc[x, "mean_pi_hat"].to_numpy()
        ax.fill_between(x, a.loc[x, "p5"], a.loc[x, "p95"], color=c, alpha=0.13, zorder=2)
        ax.plot(x, y, color=c, marker=mk, linestyle=ls, linewidth=1.8,
                markersize=5, label=lab, zorder=3)
    ax.set_xlabel("True AI fraction $\\pi$")
    ax.set_ylabel("Estimated $\\hat{\\pi}$")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks(PIS)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Calibration of $\\hat{\\pi}$ at intermediate mixtures "
                 "($k=%d$, 16 checkpoints)" % K, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("exp01_calibration_curve.%s" % ext), dpi=200)

    print("-> results/exp01_calibration_table.csv")
    print("-> results/exp01_per_checkpoint.csv")
    print("-> figures/exp01_calibration_curve.pdf/.png")


if __name__ == "__main__":
    main()
