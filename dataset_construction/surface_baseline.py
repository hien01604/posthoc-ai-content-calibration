"""SÀN chính thức — con số mà mọi detector phải vượt để có ý nghĩa.

    python3 surface_baseline.py                    # dùng biến thể trong pipeline_config
    python3 surface_baseline.py --tag deconf2 --leaderboard
    python3 surface_baseline.py --ngram --leaderboard   # thêm sàn TỪ VỰNG

HAI SÀN, KHÁC NHAU VỀ BẢN CHẤT — đừng gộp:

  sàn BỀ MẶT   GBM trên 22 đặc trưng đếm được, KHÔNG đọc một từ nào.
               Vượt nó = detector dùng nội dung chứ không chỉ đếm chữ.

  sàn TỪ VỰNG  TF-IDF char 3-5gram + LogReg (--ngram). Nó ĐỌC chữ, nên nó KHÔNG
               phải phép đo shortcut — bắt được khác biệt từ vựng là bắt đúng hiện
               tượng. Nó là baseline cạnh tranh: vượt nó = transformer đáng giá hơn
               một mô hình tuyến tính trên n-gram.
               Đo được grouped-CV trên train: 0,9676 — sát 0,9898 của deberta. Nên
               con số PHẢI có trong KLTN, nếu không phản biện đầu tiên sẽ là
               "sao không so với TF-IDF?". Chỗ n-gram thường sụp là OOD generator,
               và đó chính là điều bảng dưới đo.

VÌ SAO PHẢI CÓ HÀNG NÀY TRONG LEADERBOARD
-----------------------------------------
DeBERTa-v3 đạt AUROC 0,9898 trên test_ood_model. Câu hỏi đầu tiên của người phản biện
là "bao nhiêu trong đó chỉ là đếm chữ?". Nếu trong leaderboard không có sẵn một dòng
trả lời, câu hỏi đó thành một lỗ hổng.

Nên: huấn luyện đúng một mô hình chỉ-đặc-trưng-bề-mặt (GBM, 22 đặc trưng của
shortcut_audit.py — KHÔNG dùng một từ nào của văn bản), báo cáo nó như một detector
thật, cùng đúng những chỉ số ấy, trên đúng những split ấy. Khoảng cách giữa detector
và dòng này chính là phần detector đóng góp thêm.

Ba cột phải đọc CÙNG NHAU:
  auroc_full     trên split đầy đủ — sàn cho con số headline
  auroc_matched  trên tập matched (make_matched_eval.py) — phải ~0,5 theo thiết kế;
                 nếu KHÔNG ~0,5 thì tập matched dựng sai, đừng tin phần còn lại
  fpr@tau        dương tính giả tại ngưỡng lấy từ val — sàn cho test_fp

Mô hình bề mặt được huấn luyện trên TRAIN của cùng biến thể, không CV: đây là một
detector thật đem đi đo trên split khác, không phải một phép đo chẩn đoán.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import pipeline_config as cfg
import shortcut_audit as SA
from make_matched_eval import long_with_ids, surface_matrix

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
ARTIFACT_DIR = BASE_DIR / "artifacts"
RANDOM_STATE = 42
TARGET_FPR = 0.01


def tpr_at_fpr(y, s, fpr):
    y, s = np.asarray(y), np.asarray(s)
    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        return np.nan, np.nan
    thr = np.quantile(s[y == 0], 1 - fpr, method="higher")
    return float((s[y == 1] >= thr).mean()), float(thr)


def load_long(tag, split):
    p = OUTPUT_DIR / f"dataset_{split}_{tag}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if split == "test_fp":
        # test_fp đo dương tính giả: mọi dòng là human (ai_text = bản human đã được AI
        # gọt). Không melt thành hai lớp, nếu không là tự tạo ra lớp AI không có thật.
        has_ai = df["ai_text"].notna()
        out = df[[c for c in ("sample_id", "paper_id") if c in df.columns]].copy()
        out["text"] = df["ai_text"].where(has_ai, df["human_text"]).values
        out["label"] = 0
        out["text_role"] = np.where(has_ai, "ai_polished_human", "human")
        out["split"] = split
        return out[out["text"].notna()].reset_index(drop=True)
    return long_with_ids(df, split)


def fit_surface(train):
    """GBM trên 22 đặc trưng đếm được. Trả hàm chấm điểm nhận bảng long."""
    model = HistGradientBoostingClassifier(max_depth=4, random_state=RANDOM_STATE)
    model.fit(surface_matrix(train).to_numpy(), train["label"].to_numpy())
    return lambda d: model.predict_proba(surface_matrix(d).to_numpy())[:, 1]


def fit_lexical(train):
    """TF-IDF char 3-5gram + LogReg. CÙNG siêu tham số với shortcut_audit --ngram.

    char_wb chứ không phải char: nó không bắt n-gram vắt qua ranh giới từ, nên đặc
    trưng là hình thái TỪ chứ không phải trật tự từ. Đúng thứ cần cho một baseline
    "từ vựng", và giữ so sánh với con số 0,9676 của audit là so cùng một thứ.
    """
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                          max_features=200_000, sublinear_tf=True)
    Xtr = vec.fit_transform(train["text"].astype(str))
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, train["label"].to_numpy())
    return lambda d: clf.predict_proba(vec.transform(d["text"].astype(str)))[:, 1]


BASELINES = {
    "surface": {
        "title": "SÀN BỀ MẶT — GBM 22 đặc trưng đếm được, KHÔNG đọc chữ",
        "branch": "surface-floor",
        "fit": fit_surface,
        "model_name": lambda tag: f"GBM {len(SA.FEATURE_COLS)} surface feats ({tag})",
        "note": ("auroc_matched của CHÍNH mô hình này phải ~0,5 — nếu không thì tập\n"
                 "  matched dựng sai và mọi con số matched khác đều vô nghĩa."),
    },
    "lexical": {
        "title": "SÀN TỪ VỰNG — TF-IDF char 3-5gram + LogReg, CÓ đọc chữ",
        "branch": "lexical-floor",
        "fit": fit_lexical,
        "model_name": lambda tag: f"TF-IDF char3-5 + LogReg ({tag})",
        "note": ("Đây KHÔNG phải phép đo shortcut — bắt khác biệt từ vựng là bắt đúng\n"
                 "  hiện tượng. Nó là baseline cạnh tranh với transformer. Chỗ đáng xem là\n"
                 "  mức tụt từ val (cùng generator) sang test_ood_model (generator lạ)."),
    },
}


def run_baseline(kind, tag):
    spec = BASELINES[kind]
    train = load_long(tag, "train")
    assert train is not None, f"Thiếu dataset_train_{tag}.parquet"
    print(f"\n{'=' * 84}\n{spec['title']}\n{'=' * 84}")
    print(f"fit trên train_{tag} ({len(train):,} dòng)")
    score = spec["fit"](train)

    # τ lấy từ val, giống mọi detector khác trong leaderboard
    val = load_long(tag, "val")
    assert val is not None, f"Thiếu dataset_val_{tag}.parquet"
    _, tau = tpr_at_fpr(val["label"].to_numpy(), score(val), TARGET_FPR)
    print(f"τ tại FPR={TARGET_FPR:.0%} trên val = {tau:.4f}")

    rows = []
    for split in ("val", "test_ood_model", "test_fp"):
        d = load_long(tag, split)
        if d is None:
            print(f"[skip] {split}")
            continue
        y = d["label"].to_numpy()
        s = score(d)
        two_class = len(np.unique(y)) == 2
        row = {
            "baseline": kind, "split": split, "n": len(d),
            "auroc_full": roc_auc_score(y, s) if two_class else np.nan,
            "auprc_full": average_precision_score(y, s) if two_class else np.nan,
            "tpr@fpr1%": tpr_at_fpr(y, s, 0.01)[0],
            "tpr@fpr0.1%": tpr_at_fpr(y, s, 0.001)[0],
            "fpr@tau": float((s[y == 0] >= tau).mean()) if (y == 0).any() else np.nan,
        }
        mp = OUTPUT_DIR / f"matched_eval_{split}_{tag}.parquet"
        if mp.exists():
            m = pd.read_parquet(mp)
            my = m["label"].to_numpy()
            row["n_matched"] = len(m)
            row["auroc_matched"] = (roc_auc_score(my, score(m))
                                    if len(np.unique(my)) == 2 else np.nan)
        rows.append(row)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n" + res.round(4).to_string(index=False))
    print(f"\n  {spec['note']}")
    return res, tau


def write_leaderboard(kind, tag, res):
    spec = BASELINES[kind]
    lb_path = ARTIFACT_DIR / "leaderboard.csv"
    ood = res[res.split == "test_ood_model"]
    if ood.empty:
        print(f"Không có test_ood_model -> không ghi leaderboard cho {kind}")
        return
    r = ood.iloc[0]
    fp = res[res.split == "test_fp"]
    new = pd.DataFrame([{
        "branch": spec["branch"],
        "model": spec["model_name"](tag),
        "n_seed": 1,
        "auroc": round(float(r.auroc_full), 4),
        "auroc_std": np.nan,
        "tpr1": round(float(r["tpr@fpr1%"]), 4),
        "tpr1_std": np.nan,
        "tpr01": round(float(r["tpr@fpr0.1%"]), 4),
        "fp_tau": round(float(fp.iloc[0]["fpr@tau"]), 4) if not fp.empty else np.nan,
        "fp_tau_std": np.nan,
    }])
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        lb = lb[lb["branch"] != spec["branch"]]
        lb = pd.concat([lb, new], ignore_index=True)
    else:
        lb = new
    lb.to_csv(lb_path, index=False)
    print(f"-> cập nhật {lb_path.relative_to(BASE_DIR)} (hàng branch={spec['branch']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=cfg.DATASET_TAG)
    ap.add_argument("--ngram", action="store_true",
                    help="thêm sàn TỪ VỰNG (TF-IDF char 3-5gram + LogReg)")
    ap.add_argument("--leaderboard", action="store_true",
                    help="ghi/cập nhật hàng trong artifacts/leaderboard.csv")
    args = ap.parse_args()
    tag = args.tag

    kinds = ["surface"] + (["lexical"] if args.ngram else [])
    all_res, meta = [], {}
    for kind in kinds:
        res, tau = run_baseline(kind, tag)
        all_res.append(res)
        meta[kind] = {"tau": tau, "metrics": res.to_dict("records")}
        if args.leaderboard:
            write_leaderboard(kind, tag, res)

    combined = pd.concat(all_res, ignore_index=True)
    out = OUTPUT_DIR / f"surface_baseline_{tag}.csv"
    combined.to_csv(out, index=False)
    (OUTPUT_DIR / f"surface_baseline_{tag}.json").write_text(json.dumps({
        "tag": tag, "n_features": len(SA.FEATURE_COLS), "features": SA.FEATURE_COLS,
        "target_fpr": TARGET_FPR, "random_state": RANDOM_STATE, "baselines": meta,
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n-> {out.name}")

    if len(kinds) > 1:
        print(f"\n{'=' * 84}\nHAI SÀN CẠNH NHAU (test_ood_model)\n{'=' * 84}")
        piv = combined[combined.split == "test_ood_model"].set_index("baseline")
        for kind in kinds:
            r = piv.loc[kind]
            print(f"  {kind:<9} AUROC đầy đủ {r.auroc_full:.4f} | matched "
                  f"{r.get('auroc_matched', float('nan')):.4f} | TPR@FPR1% {r['tpr@fpr1%']:.4f}")
        print("\n  Transformer phải vượt CẢ HAI để có ý nghĩa: vượt sàn bề mặt = không chỉ")
        print("  đếm chữ; vượt sàn từ vựng = đáng giá hơn một mô hình tuyến tính n-gram.")


if __name__ == "__main__":
    main()
