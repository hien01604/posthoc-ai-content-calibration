"""Chấm bộ 22 đặc trưng bề mặt đã khoá lên DAGPap24 — benchmark của shared task SDP@ACL 2024.

    python3 audit_dagpap24.py                      # audit bản RAW (như ban tổ chức phát)
    python3 audit_dagpap24.py --deconf2            # thêm bản đã áp recipe deconf2
    python3 audit_dagpap24.py --split dev          # audit tập dev thay vì train
    python3 audit_dagpap24.py --ngram              # thêm trần char 3-5gram (CHẬM, tốn RAM)

CÂU HỎI
-------
DAGPap24 là benchmark chính thức cho bài toán "detect AI trong paper khoa học ở mức
token/span" (457 submission / 28 team). Câu hỏi: một GBM **không đọc một từ nào**, chỉ
đọc 22 con số bề mặt, phân biệt human/AI ở mức nào trên benchmark đó?

VÌ SAO GHÉP ĐƯỢC THÀNH "CẶP"
----------------------------
`shortcut_audit.audit_frame` nhận dataframe dạng cặp (human_text, ai_text, paper_id) và
CV theo group=paper_id. DAGPap24 khớp tự nhiên: mỗi văn bản là MỘT bài, trong đó 25% số
token bị thay bằng AI. Nên lấy phần human và phần AI CỦA CÙNG MỘT BÀI làm một cặp:

    tokens:      [w w w w w w w w w w w w w w w w]
    labels:      [0 0 0 0 0 2 2 2 2 0 0 3 3 0 0 0]
                  └── human ──┘ └chatgpt┘ └sum┘

    -> human_text = các token nhãn 0, ai_text = các token nhãn AI, paper_id = index bài

Chủ đề được kiểm soát vì hai bên đến từ cùng một bài — đúng tinh thần matched pair của
dataset gốc, và group theo bài nên không rò chủ đề giữa các fold.

BỐN BIẾN THỂ, KHÔNG GỘP LÀM MỘT
-------------------------------
DAGPap24 có 4 nhãn, mà chỉ MỘT là LLM sinh văn bản thật:

    0 human · 1 NLTK_synonym_replacement · 2 chatgpt · 3 summarized

Nhãn 1 là script thay từ đồng nghĩa (không phải LLM), nhãn 3 là một model summarisation
không nêu tên. Gộp cả 3 vào "AI" thì con số bị nhãn dễ nhất kéo lên. Nên audit riêng
từng nhãn, và `chatgpt` mới là con số so được với dataset của mình.

BỘ ĐẶC TRƯNG
------------
Import từ `shortcut_audit.py`, KHÔNG khai lại. Bộ 22 đặc trưng + phân tầng + ngưỡng cổng
được khoá ở đúng một chỗ; khai lại ở đây là mở đường cho việc hai bảng số không so được
với nhau (xem docs/SHORTCUT_DEFENSE.md §2).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import (canonicalize, notation_gap, notation_profile,
                        split_sentences, truncate_pair_sentences)
from shortcut_audit import audit_frame, print_audit

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "external" / "dagpap24"
REPORT_DIR = BASE_DIR / "output" / "shortcut_audit_dagpap24"

# id nhãn -> tên, đúng theo trang Data của competition.
LABELS = {0: "human", 1: "nltk_synonym", 2: "chatgpt", 3: "summarized"}
AI_LABELS = {1: "nltk_synonym", 2: "chatgpt", 3: "summarized"}

# Cặp nào ngắn quá thì đặc trưng bề mặt chỉ là nhiễu (ttr_alpha_w200 cần 200 token).
MIN_WORDS = 50

# Giữ 85% cặp theo notation_gap — ĐÚNG tham số của deconf2 (output/deconf2_manifest.json).
NOTATION_KEEP_FRAC = 0.85


def _parse(cell):
    """Các cột list của DAGPap24 lưu dạng JSON string, không phải list thật."""
    if isinstance(cell, str):
        return json.loads(cell)
    return list(cell)


def load_split(split):
    """Trả về DataFrame [index, tokens(list), token_label_ids(list)]."""
    if split == "train":
        df = pd.read_parquet(DATA_DIR / "train_data.parquet")
        idx = df["index"].to_numpy() if "index" in df.columns else df.index.to_numpy()
        labels = df["token_label_ids"]
    else:
        df = pd.read_parquet(DATA_DIR / f"{split}_data.parquet")
        lab = pd.read_parquet(DATA_DIR / f"{split}_labels.parquet")
        # reindex theo index của data: hai file cùng thứ tự nhưng không dựa vào đó.
        lab = lab.reindex(df.index)
        idx = df.index.to_numpy()
        labels = lab["token_label_ids"]

    out = pd.DataFrame({
        "paper_id": [str(i) for i in idx],
        "tokens": [_parse(t) for t in df["tokens"]],
        "labels": [_parse(t) for t in labels],
    })
    bad = out.apply(lambda r: len(r["tokens"]) != len(r["labels"]), axis=1)
    if bad.any():
        print(f"  [!] bỏ {int(bad.sum())} bài lệch độ dài tokens/labels")
        out = out[~bad].reset_index(drop=True)
    return out


def span_stats(df):
    """Độ dài trung bình theo nhãn — bằng chứng shortcut độ dài, đọc TRƯỚC mọi AUROC."""
    rows = []
    for lid, name in LABELS.items():
        wc = [sum(1 for l in r.labels if l == lid) for r in df.itertuples()]
        wc = np.array([w for w in wc if w > 0])
        rows.append({
            "nhãn": name,
            "số bài có nhãn này": len(wc),
            "số từ TB": round(float(wc.mean()), 1) if len(wc) else 0.0,
            "trung vị": int(np.median(wc)) if len(wc) else 0,
        })
    return pd.DataFrame(rows)


def build_pairs(df, ai_label, recipe=None):
    """Ghép (human_text, ai_text) cho MỘT nhãn AI. ai_label=None -> gộp cả 3 nhãn AI."""
    want = set(AI_LABELS) if ai_label is None else {ai_label}
    rows = []
    for r in df.itertuples():
        h, a = [], []
        for tok, lab in zip(r.tokens, r.labels):
            if lab == 0:
                h.append(tok)
            elif lab in want:
                a.append(tok)
        if len(h) < MIN_WORDS or len(a) < MIN_WORDS:
            continue
        rows.append({"paper_id": r.paper_id,
                     "human_text": " ".join(h), "ai_text": " ".join(a)})
    pairs = pd.DataFrame(rows)
    if pairs.empty or recipe is None:
        return pairs
    return recipe(pairs)


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?)\]])")


def redetokenize(text):
    """Gộp lại khoảng trắng bị chèn TRƯỚC dấu câu: " . the" -> ". the".

    Sửa đúng lỗi của lớp `summarized`: output model summarisation không được
    detokenize nên dấu câu đứng rời thành token 1 ký tự -> single_char_frac = 1,0000.
    """
    return _SPACE_BEFORE_PUNCT.sub(r"\1", str(text))


def drop_partial_sentences(text):
    """Bỏ câu đầu nếu nó bắt đầu bằng chữ thường, bỏ câu cuối nếu không kết thúc bằng dấu câu.

    Sửa đúng lỗi biên span của DAGPap24: đoạn AI được ghép vào GIỮA CÂU, nên câu đầu và
    câu cuối của đoạn đều là mảnh vụn. `truncate_pair_sentences` không sửa được vì nó lấy
    TIỀN TỐ câu — mảnh vụn đầu đoạn luôn nằm trong tiền tố.
    """
    sents = [s for s in split_sentences(text) if s.strip()]
    if len(sents) > 2 and sents[0][:1].islower():
        sents = sents[1:]
    if len(sents) > 2 and not sents[-1].rstrip().endswith((".", "!", "?")):
        sents = sents[:-1]
    return " ".join(sents)


def apply_repair(pairs):
    """Hai bước sửa NHẮM ĐÚNG artifact của DAGPap24, chạy TRƯỚC recipe deconf2."""
    out = pairs.copy()
    for col in ("human_text", "ai_text"):
        out[col] = [drop_partial_sentences(redetokenize(t)) for t in out[col]]
    keep = (out.human_text.str.split().str.len() >= MIN_WORDS) & \
           (out.ai_text.str.split().str.len() >= MIN_WORDS)
    print(f"    sửa biên span + detokenize: {len(out):,} -> {int(keep.sum()):,} cặp")
    return apply_deconf2_recipe(out[keep].reset_index(drop=True))


def apply_deconf2_recipe(pairs):
    """Áp ĐÚNG recipe deconf2: canonicalize -> truncate_pair_sentences -> lọc notation_gap.

    Mục đích không phải "sửa DAGPap24" mà để trả lời: sau khi bỏ shortcut độ dài và lọc
    cặp không trung thực nội dung, còn lại bao nhiêu tín hiệu bề mặt? Đó là phép so
    trực tiếp với hành trình normalized -> deconf2 của dataset mình.
    """
    h = [canonicalize(t) for t in pairs["human_text"]]
    a = [canonicalize(t) for t in pairs["ai_text"]]
    cut = [truncate_pair_sentences(x, y) for x, y in zip(h, a)]
    out = pd.DataFrame({
        "paper_id": pairs["paper_id"].to_numpy(),
        "human_text": [c[0] for c in cut],
        "ai_text": [c[1] for c in cut],
    })
    gap = np.array([notation_gap(notation_profile(x), notation_profile(y))
                    for x, y in zip(out["human_text"], out["ai_text"])])
    thr = float(np.quantile(gap, NOTATION_KEEP_FRAC))
    kept = out[gap <= thr].reset_index(drop=True)
    print(f"    recipe deconf2: {len(out):,} -> {len(kept):,} cặp "
          f"(ngưỡng notation_gap = {thr:.4f}, giữ {NOTATION_KEEP_FRAC:.0%})")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--deconf2", action="store_true",
                    help="audit thêm bản đã áp recipe deconf2")
    ap.add_argument("--repair", action="store_true",
                    help="thêm biến thể sửa biên span + detokenize RỒI mới áp deconf2")
    ap.add_argument("--ngram", action="store_true", help="thêm trần char 3-5gram (chậm)")
    ap.add_argument("--max-docs", type=int, default=None, help="lấy mẫu để chạy nhanh")
    ap.add_argument("--report-dir", default=None,
                    help="ghi CSV vào thư mục khác (để chạy song song, khỏi ghi đè)")
    args = ap.parse_args()
    report_dir = Path(args.report_dir) if args.report_dir else REPORT_DIR

    print(f"đọc DAGPap24 / {args.split} từ {DATA_DIR}")
    df = load_split(args.split)
    if args.max_docs:
        df = df.sample(min(args.max_docs, len(df)), random_state=42).reset_index(drop=True)
    ntok = sum(len(t) for t in df["tokens"])
    print(f"  {len(df):,} bài · {ntok:,} token · TB {ntok/max(1,len(df)):,.0f} token/bài\n")

    print("=" * 86)
    print("ĐỘ DÀI THEO NHÃN — đọc cái này TRƯỚC mọi con số AUROC")
    print("=" * 86)
    st = span_stats(df)
    print(st.to_string(index=False))
    hm = st.loc[st["nhãn"] == "human", "số từ TB"].iloc[0]
    cg = st.loc[st["nhãn"] == "chatgpt", "số từ TB"].iloc[0]
    if cg:
        print(f"\n  -> phần human dài gấp {hm/cg:.1f}x phần chatgpt "
              f"({hm:,.0f} vs {cg:,.0f} từ)")
        print("     Chỉ riêng độ dài đã là kênh phân biệt. So sánh: dataset của mình có")
        print("     word_count human/AI = 253,75 / 253,49 (gần bằng nhau) mà VẪN rò 0,7201.")

    results, per_feature, totals = [], [], []
    variants = [("raw", None)]
    if args.deconf2:
        variants.append(("deconf2", apply_deconf2_recipe))
    if args.repair:
        variants.append(("repair", apply_repair))

    for vname, recipe in variants:
        for ai_label, ai_name in [(None, "AI-gộp-3-nhãn"), (2, "chatgpt"),
                                  (3, "summarized"), (1, "nltk_synonym")]:
            tag = f"dagpap24-{args.split}/{ai_name}/{vname}"
            print(f"\n{'#' * 86}\n# {tag}\n{'#' * 86}")
            pairs = build_pairs(df, ai_label, recipe=recipe)
            if len(pairs) < 50:
                print(f"  bỏ qua — chỉ {len(pairs)} cặp đủ điều kiện (cần >= 50)")
                continue
            res = audit_frame(pairs, tag, ngram_baseline=args.ngram)
            print_audit(res)
            results.append(res)
            per_feature.extend(res["per_feature"])
            for k, (m, s) in res["totals"].items():
                combo, model = k.split("|")
                totals.append({"dataset": tag, "combo": combo, "model": model,
                               "auc_mean": m, "auc_std": s,
                               "verdict": res["verdict"], "n_pairs": len(pairs)})

    if not results:
        raise SystemExit("Không audit được biến thể nào.")

    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_feature).to_csv(report_dir / "per_feature_auc.csv", index=False)
    T = pd.DataFrame(totals)
    T.to_csv(report_dir / "totals_auc.csv", index=False)
    st.to_csv(report_dir / "span_length.csv", index=False)

    print("\n" + "=" * 86)
    print("TỔNG KẾT — GBM trên 22 đặc trưng bề mặt, KHÔNG đọc một từ nào")
    print("=" * 86)
    piv = (T[T.model == "GBM"]
           .pivot_table(index="dataset", columns="combo", values="auc_mean")
           .reindex(columns=["process_only", "fidelity_only", "style_only",
                             "all", "legacy_g6_6feat"]))
    piv.insert(0, "n_pairs", T.groupby("dataset").n_pairs.first())
    piv["phán quyết"] = T.groupby("dataset").verdict.first()
    print(piv.round(4).to_string())
    print("\ncổng CỨNG process <= 0,55 · cổng MỀM fidelity <= 0,60 · style cao là ĐÚNG")
    print(f"\nBáo cáo -> {report_dir}")


if __name__ == "__main__":
    main()
