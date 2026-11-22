"""Dựng TẬP EVAL MATCHED — nơi mô hình bề mặt về mức ngẫu nhiên.

    python3 make_matched_eval.py                 # dựng cho val + test_ood_model
    python3 make_matched_eval.py --tag deconf2 --splits val test_ood_model

VÌ SAO CẦN
----------
Sau 02c, dataset đã sạch dấu vết QUY TRÌNH (process 0,5055) nhưng mô hình bề mặt 22
đặc trưng vẫn đạt 0,8909 — vì phần còn lại là VĂN PHONG LLM thật: từ dài hơn (4,94 vs
5,82 ký tự/từ), từ vựng đa dạng hơn, câu dài hơn. Không thể xoá nó khỏi dữ liệu mà
không xoá luôn bài toán.

Nên câu hỏi "detector có thực sự học nội dung, hay chỉ đếm chữ" KHÔNG trả lời được
bằng cách làm dataset sạch hơn. Nó trả lời được bằng cách đo trên TẬP CON mà mô hình
bề mặt bất lực:

    Nếu trên tập con đó mô hình bề mặt = 0,5x mà detector vẫn giữ AUROC cao,
    thì phần AUROC đó KHÔNG đến từ đặc trưng bề mặt.

Đây là con số để trả lời phản biện "0,99 của anh chỉ là độ dài với độ dài từ", và nó
mạnh hơn mọi lập luận về cách dựng dữ liệu.

CÁCH LÀM
--------
1. Huấn luyện mô hình bề mặt (GBM, 22 đặc trưng) trên TRAIN -> điểm xu hướng
   p(AI | đặc trưng bề mặt). Huấn luyện trên train, KHÔNG trên split đang dựng: dùng
   chính split đó là để tiêu chí chọn mẫu phụ thuộc vào dữ liệu test.
2. Chia p thành N_BINS bin theo phân vị. Trong mỗi bin, giữ min(n_human, n_ai) mẫu
   mỗi lớp, rút ngẫu nhiên có seed -> phân phối p của hai lớp trở nên giống nhau.
3. LẶP LẠI bước 2 với mô hình bề mặt được HUẤN LUYỆN LẠI trên chính tập con hiện tại
   (điểm out-of-fold, CV theo paper_id), cho tới khi nó cũng về mức ngẫu nhiên.

Vì sao phải lặp — đo được, không phải đề phòng lý thuyết: ghép MỘT LẦN theo p của mô
hình fit-trên-train đưa p đó về 0,5011 trên test_ood_model, nhưng một mô hình bề mặt
huấn luyện LẠI trên chính tập con ấy vẫn đạt **0,5799**. Vì p là bản rút gọn MỘT
chiều: cân bằng nó không cân bằng phân phối 22 chiều, và mô hình mới tìm được phần dư
(ở đây là avg_sentence_len). Nếu chỉ báo con số 0,5011 thì người phản biện chỉ cần
huấn luyện lại một GBM là lật được.

Điều kiện dừng là CẢ HAI mô hình bề mặt (fit-trên-train và fit-lại-OOF) đều nằm trong
|AUC - 0,5| <= MATCH_TARGET_AUC - 0,5. Chỉ chặn một cái là vá một lỗ để hở lỗ kia:
ghép theo riêng p_refit thì p_train vẫn còn 0,6527 trên val. Nên tập matched không
phải "đã ghép theo p" mà là "đã ghép cho tới khi mô hình bề mặt hết đường".

Kết quả thực tế (deconf2, 2 vòng):
    val            876 -> 296 dòng (33,8%)  fit-trên-train 0,9049 -> 0,5027
                                            fit-lại        0,8786 -> 0,5475
    test_ood_model 3.028 -> 1.206 (39,8%)   fit-trên-train 0,8715 -> 0,5446
                                            fit-lại        0,8758 -> 0,4663

Vì sao chia bin chứ không ghép cặp 1-1 theo caliper: chia bin là tất định với một
seed, giải thích được trong một câu, và không phụ thuộc thứ tự duyệt.

LƯU Ý KHI ĐỌC KẾT QUẢ
---------------------
Tập matched là tập STRESS TEST, không phải tập test chính. Nó lệch phân phối có chủ ý
(giữ những văn bản human dùng từ dài bất thường và những văn bản AI dùng từ ngắn bất
thường), và nó nhỏ hơn nhiều. Báo cáo LUÔN đi kèm: AUROC trên tập đầy đủ, AUROC trên
tập matched, và tỉ lệ giữ lại.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import shortcut_audit as SA

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
RANDOM_STATE = 42
N_BINS = 20
# Dừng khi mô hình bề mặt fit-LẠI (out-of-fold) không còn vượt mức này. 0,55 chứ không
# 0,50: với ~1.200 dòng, std của AUROC out-of-fold đã ~0,015, nên đòi 0,50 là đòi
# thuật toán đuổi theo nhiễu và nó sẽ cắt tập cho tới khi rỗng.
MATCH_TARGET_AUC = 0.55
MAX_MATCH_ITERS = 8
# Dưới ngưỡng này thì tập matched quá nhỏ để kết luận -> dừng và báo, không cắt tiếp.
MIN_MATCHED_ROWS = 200


def long_with_ids(df, split):
    """Bảng long có sample_id + text_role để join được với pred_*.parquet."""
    keep = [c for c in ("sample_id", "paper_id", "ai_degree", "generator_family",
                        "source", "model_actual") if c in df.columns]
    human = df[keep].copy()
    human["text"] = df["human_text"].values
    human["label"] = 0
    human["text_role"] = "human"
    ai = df[keep].copy()
    ai["text"] = df["ai_text"].values
    ai["label"] = 1
    ai["text_role"] = "ai"
    long = pd.concat([human, ai], ignore_index=True)
    long = long[long["text"].notna() & long["text"].astype(str).str.strip().ne("")]
    long["split"] = split
    return long.reset_index(drop=True)


def surface_matrix(long):
    F = pd.DataFrame([SA.surface_features(t) for t in long["text"]])[SA.FEATURE_COLS]
    return F.replace([np.inf, -np.inf], np.nan)


def bin_match(propensity, labels, n_bins=N_BINS, seed=RANDOM_STATE):
    """Giữ min(n_human, n_ai) mỗi bin phân vị -> hai lớp cùng phân phối propensity."""
    rng = np.random.default_rng(seed)
    edges = np.unique(np.quantile(propensity, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.digitize(propensity, edges[1:-1]), 0, len(edges) - 2)
    keep = np.zeros(len(propensity), dtype=bool)
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        h = idx[labels[idx] == 0]
        a = idx[labels[idx] == 1]
        k = min(len(h), len(a))
        if k:
            keep[rng.choice(h, k, replace=False)] = True
            keep[rng.choice(a, k, replace=False)] = True
    return keep


def oof_surface_propensity(X, y, g, n_splits=5):
    """Điểm bề mặt OUT-OF-FOLD, CV theo paper_id. Đây là thứ phép kiểm chặt dùng.

    Phải là out-of-fold: điểm in-sample thì mô hình đã nhớ từng dòng, ghép theo nó
    không nói gì về khả năng tổng quát hoá của shortcut.
    """
    n_splits = min(n_splits, *[len(np.unique(g[y == c])) for c in np.unique(y)])
    if n_splits < 2:
        return np.full(len(y), 0.5)
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y, g):
        model = HistGradientBoostingClassifier(max_depth=4, random_state=RANDOM_STATE)
        oof[te] = model.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def iterative_match(X, y, g, train_model, n_bins=N_BINS):
    """Ghép lặp cho tới khi CẢ HAI mô hình bề mặt hết vượt ngưỡng.

    Hai mô hình, vì người phản biện có thể dùng bất kỳ cái nào:
      p_train : mô hình fit trên TRAIN, chấm sang tập này (kịch bản "detector bề mặt
                được huấn luyện sẵn rồi đem đi đo")
      p_refit : mô hình fit LẠI out-of-fold trên chính tập con (kịch bản "người phản
                biện tự train một GBM trên đúng tập con của anh")

    Mỗi vòng ghép theo cái đang CAO HƠN. Ghép chỉ theo p_refit thì đo được p_train vẫn
    còn 0,6527 trên val — tức vá một lỗ và để hở lỗ kia.

    |AUC - 0,5| chứ không phải AUC: ghép có thể ĐẢO chiều lệch (đo được uppercase_ratio
    xuống 0,3958 trên val). Lệch đảo chiều vẫn là lệch.

    Trả (mặt nạ giữ lại, lịch sử từng vòng).
    """
    keep = np.ones(len(y), dtype=bool)
    history = []
    for it in range(1, MAX_MATCH_ITERS + 1):
        idx = np.flatnonzero(keep)
        p_train = train_model.predict_proba(X[idx])[:, 1]
        p_refit = oof_surface_propensity(X[idx], y[idx], g[idx])
        auc_train = roc_auc_score(y[idx], p_train)
        auc_refit = roc_auc_score(y[idx], p_refit)
        dev_train, dev_refit = abs(auc_train - 0.5), abs(auc_refit - 0.5)
        history.append({"iter": it, "n_rows": int(keep.sum()),
                        "surface_auc_trainfit": float(auc_train),
                        "surface_auc_refit": float(auc_refit)})
        if max(dev_train, dev_refit) <= MATCH_TARGET_AUC - 0.5:
            break
        worse = p_train if dev_train >= dev_refit else p_refit
        history[-1]["matched_on"] = "trainfit" if dev_train >= dev_refit else "refit"
        sub_keep = bin_match(worse, y[idx], n_bins, seed=RANDOM_STATE + it)
        if sub_keep.sum() < MIN_MATCHED_ROWS:
            history[-1]["stopped"] = "tập con sẽ nhỏ hơn ngưỡng tối thiểu -> dừng"
            break
        new = np.zeros_like(keep)
        new[idx[sub_keep]] = True
        keep = new
    return keep, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="deconf2")
    ap.add_argument("--splits", nargs="+", default=["val", "test_ood_model"])
    ap.add_argument("--n-bins", type=int, default=N_BINS)
    args = ap.parse_args()

    train_path = OUTPUT_DIR / f"dataset_train_{args.tag}.parquet"
    assert train_path.exists(), f"Thiếu {train_path}"

    # ── mô hình xu hướng bề mặt, huấn luyện MỘT LẦN trên train ──────
    train_long = long_with_ids(pd.read_parquet(train_path), "train")
    Xtr = surface_matrix(train_long).to_numpy()
    ytr = train_long["label"].to_numpy()
    propensity_model = HistGradientBoostingClassifier(
        max_depth=4, random_state=RANDOM_STATE).fit(Xtr, ytr)
    print(f"mô hình xu hướng bề mặt: GBM 22 đặc trưng, fit trên train "
          f"({len(train_long):,} dòng)")
    print(f"  AUROC in-sample trên train: "
          f"{roc_auc_score(ytr, propensity_model.predict_proba(Xtr)[:, 1]):.4f}"
          "  (in-sample nên cao, chỉ để biết nó có học được không)")

    report = []
    for split in args.splits:
        p = OUTPUT_DIR / f"dataset_{split}_{args.tag}.parquet"
        if not p.exists():
            print(f"[skip] {p.name} không tồn tại")
            continue
        long = long_with_ids(pd.read_parquet(p), split)
        F = surface_matrix(long)
        prop = propensity_model.predict_proba(F.to_numpy())[:, 1]
        y = long["label"].to_numpy()
        g = long["paper_id"].astype(str).to_numpy()

        auc_full_train_fitted = roc_auc_score(y, prop)
        auc_full_refit, _ = SA.grouped_auc(
            HistGradientBoostingClassifier(max_depth=4, random_state=RANDOM_STATE),
            F.to_numpy(), y, g)

        keep, history = iterative_match(F.to_numpy(), y, g, propensity_model, args.n_bins)
        sub = long[keep].reset_index(drop=True)
        sub["surface_propensity"] = prop[keep]
        Fs = F[keep]
        ys = y[keep]
        gs = g[keep]

        auc_matched_train_fitted = roc_auc_score(ys, prop[keep])
        auc_matched_refit, _ = SA.grouped_auc(
            HistGradientBoostingClassifier(max_depth=4, random_state=RANDOM_STATE),
            Fs.to_numpy(), ys, gs)

        # kênh nào còn sống sót trên tập matched?
        worst = ("-", 0.5)
        for c in SA.FEATURE_COLS:
            v = Fs[[c]].to_numpy()
            if len(np.unique(v[np.isfinite(v)])) < 2:
                continue
            m, _ = SA.grouped_auc(
                HistGradientBoostingClassifier(max_depth=3, random_state=RANDOM_STATE),
                v, ys, gs)
            if abs(m - 0.5) > abs(worst[1] - 0.5):
                worst = (c, m)

        out = OUTPUT_DIR / f"matched_eval_{split}_{args.tag}.parquet"
        sub.to_parquet(out, index=False)

        print(f"\n{'=' * 78}\n{split}  ({args.tag})\n{'=' * 78}")
        print("  vòng ghép (điều kiện dừng: mô hình bề mặt fit-lại OOF "
              f"<= {MATCH_TARGET_AUC}):")
        for h in history:
            print(f"    vòng {h['iter']}: {h['n_rows']:>6,} dòng  "
                  f"fit-trên-train = {h['surface_auc_trainfit']:.4f}  "
                  f"fit-lại = {h['surface_auc_refit']:.4f}"
                  + (f"  -> ghép theo {h['matched_on']}" if "matched_on" in h else "  -> DỪNG")
                  + (f"   [{h['stopped']}]" if "stopped" in h else ""))
        print(f"  dòng: {len(long):,} -> {len(sub):,}  (giữ {len(sub) / len(long):.1%})"
              f"  | cân bằng lớp: {dict(pd.Series(ys).value_counts())}")
        print(f"  mô hình bề mặt fit-trên-train : đầy đủ {auc_full_train_fitted:.4f}"
              f" -> matched {auc_matched_train_fitted:.4f}")
        print(f"  mô hình bề mặt fit-LẠI (CV)   : đầy đủ {auc_full_refit:.4f}"
              f" -> matched {auc_matched_refit:.4f}   <- phép kiểm chặt hơn")
        print(f"  đặc trưng lệch nhất còn lại   : {worst[0]} = {worst[1]:.4f}")
        print(f"  -> {out.name}")

        report.append({
            "split": split, "tag": args.tag, "n_full": len(long), "n_matched": len(sub),
            "retention": len(sub) / len(long),
            "surface_auc_full_trainfit": auc_full_train_fitted,
            "surface_auc_matched_trainfit": auc_matched_train_fitted,
            "surface_auc_full_refit": auc_full_refit,
            "surface_auc_matched_refit": auc_matched_refit,
            "worst_feature_matched": worst[0], "worst_feature_auc": worst[1],
            "n_iters": len(history), "match_history": history,
            "n_bins": args.n_bins, "random_state": RANDOM_STATE,
            "match_target_auc": MATCH_TARGET_AUC,
        })

    if report:
        rdf = pd.DataFrame(report).drop(columns=["match_history"])
        rdf.to_csv(OUTPUT_DIR / f"matched_eval_report_{args.tag}.csv", index=False)
        (OUTPUT_DIR / f"matched_eval_report_{args.tag}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n{rdf.to_string(index=False)}")
        print(f"\nBáo cáo -> matched_eval_report_{args.tag}.csv")


if __name__ == "__main__":
    main()
