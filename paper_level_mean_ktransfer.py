"""Bai toan 2 (mean/median): BA nguon nguong cho cung thong ke `mean`, cung tap danh gia
test_ood_model (co ca 2 lop). Tai lap Table III / Table A3 o Sec VI.2 outline_rivf2026_v2.md.

    python3 paper_level_mean_ktransfer.py

BA NGUON NGUONG (chi khac nhau o CHO CHOT tau, moi thu khac giu nguyen)
----------------------------------------------------------------------
  tau_chunk  : phan vi 99% cua diem TUNG DOAN human trong `val`  (hieu chuan o k~1)
  tau_mean_k : phan vi 99% cua MEAN moi bai trong test_fp_full   (hieu chuan o k~22,6)
  tau_ood    : phan vi 99% cua MEAN moi bai human trong CHINH test_ood_model (k~2,38)
               -> HANG DOI CHUNG: chot dung tai k cua tap danh gia.
               Chia doi theo paper_id (chot tau tren nua cal, do tren nua eval), lap 200 lan
               -> khong bao gio do tren chinh bai da dung de chot nguong.

DAU VAO
-------
  artifacts/kaggle/<artifact>/pred_val.parquet                       nguon tau_chunk
  artifacts/kaggle_testfp_full/pred_testfp_full/pred_test_fp_full_<artifact>.parquet
                                                                     nguon tau_mean_k
  artifacts/kaggle/<artifact>/pred_test_ood_model.parquet            tap danh gia + tau_ood

DAU RA (output/)
----------------
  mean_ktransfer_16models.csv   FPR va ty le bo lot AI, ba nguon nguong, 16 checkpoint

QUY UOC
-------
  gop muc bai dung `mean`; gop theo (paper_id, label) vi test_ood_model co ca hai lop
  duoi CUNG mot paper_id (matched pair) - group theo paper_id KHONG label se lam le mean.
  phan vi `method="higher"`, muc tieu FPR = 1%.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts" / "kaggle"
FP_FULL = BASE / "artifacts" / "kaggle_testfp_full" / "pred_testfp_full"
OUT = BASE / "output"
TARGET = 0.01
N_SPLIT = 200
SEED = 42


def qhi(a, q):
    return float(np.quantile(a, q, method="higher"))


def artifacts():
    for f in sorted(glob.glob(str(ART / "artifacts_*" / "pred_test_ood_model.parquet"))):
        art = Path(f).parent.name.replace("artifacts_", "")
        yield art, f


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for art, f_ood in artifacts():
        rng = np.random.default_rng(SEED)

        val = pd.read_parquet(ART / f"artifacts_{art}" / "pred_val.parquet")
        tau_chunk = qhi(val.loc[val.label == 0, "score"].to_numpy(), 1 - TARGET)

        fp_full = pd.read_parquet(FP_FULL / f"pred_test_fp_full_artifacts_{art}.parquet")
        tau_mean_k = qhi(fp_full.groupby("paper_id")["score"].mean().to_numpy(), 1 - TARGET)

        ood = pd.read_parquet(f_ood)
        g = ood.groupby(["paper_id", "label"])["score"].mean()
        pid = g.index.get_level_values("paper_id").to_numpy()
        lbl = g.index.get_level_values("label").to_numpy()
        mean_ood = g.to_numpy()
        human, ai = lbl == 0, lbl == 1

        # --- hang doi chung: tau chot DUNG tai k cua tap nay, chia doi theo paper_id ---
        ids = np.unique(pid)
        taus, fprs, misses = [], [], []
        for _ in range(N_SPLIT):
            perm = rng.permutation(len(ids))
            cal_ids = set(ids[perm[: len(ids) // 2]])
            in_cal = np.fromiter((p in cal_ids for p in pid), bool, len(pid))
            t = qhi(mean_ood[human & in_cal], 1 - TARGET)
            taus.append(t)
            fprs.append(float((mean_ood[human & ~in_cal] >= t).mean()))
            misses.append(float((mean_ood[ai & ~in_cal] < t).mean()))

        rows.append(dict(
            model=art,
            tau_chunk=tau_chunk, tau_mean_k=tau_mean_k, tau_ood=float(np.mean(taus)),
            fpr_tau_chunk=float((mean_ood[human] >= tau_chunk).mean()),
            miss_tau_chunk=float((mean_ood[ai] < tau_chunk).mean()),
            fpr_tau_mean_k=float((mean_ood[human] >= tau_mean_k).mean()),
            miss_tau_mean_k=float((mean_ood[ai] < tau_mean_k).mean()),
            fpr_tau_ood=float(np.mean(fprs)), miss_tau_ood=float(np.mean(misses)),
            n_human=int(human.sum()), n_ai=int(ai.sum()), n_split=N_SPLIT,
        ))

    df = pd.DataFrame(rows).sort_values("fpr_tau_chunk")
    df.to_csv(OUT / "mean_ktransfer_16models.csv", index=False)
    print(df.round(4).to_string(index=False))

    def line(tag, fc, mc):
        f, m = df[fc] * 100, df[mc] * 100
        print(f"  {tag:34s} FPR {f.median():5.2f}% [{f.min():.2f}-{f.max():.2f}]"
              f"   bo lot {m.median():6.2f}% [{m.min():.2f}-{m.max():.2f}]")

    print(f"\nTrung vi qua {len(df)} checkpoint (muc tieu FPR 1,00%):")
    line("tau muc CHUNK (tu val, k~1)", "fpr_tau_chunk", "miss_tau_chunk")
    line("tau chot o k~22,6 (test_fp_full)", "fpr_tau_mean_k", "miss_tau_mean_k")
    line("tau chot DUNG o k~2,38 [doi chung]", "fpr_tau_ood", "miss_tau_ood")
    print("\n-> output/mean_ktransfer_16models.csv")


if __name__ == "__main__":
    main()
