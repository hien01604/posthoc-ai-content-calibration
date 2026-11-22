"""Dựng `test_fp_full` — bài báo NGUYÊN VẸN, chưa từng xuất hiện ở bất kỳ split nào.

    python3 build_testfp_full.py                 # 1.200 bài
    python3 build_testfp_full.py --n-papers 600  # nhẹ hơn để chạy thử trên Kaggle

MỤC ĐÍCH
--------
`test_fp` hiện tại lấy 3 mảnh/bài (~700 từ = 11% một bài thật), nên FPR mức bài đo được
(5,2%) là GIỚI HẠN DƯỚI. Bộ này giữ MỌI mảnh của bài để k ≈ 20–27, trả lời:

    FPR mức bài ở độ dài bài THẬT là bao nhiêu — ĐO, không chiếu?
    Công thức 1-(1-p)^k có còn đúng ở k lớn không? (mới kiểm được ở k≈3)

BA ĐIỀU KIỆN SẠCH
-----------------
1. `paper_id` KHÔNG nằm trong train / val / test_ood_model / test_fp.
   -> không mảnh nào từng được train, cũng không dùng để chốt ngưỡng hay báo số.
2. `published_year <= 2015` (dữ liệu nguồn chỉ tới 2018; thực tế ra 2010–2015)
   -> trước ChatGPT 7 năm, nên nhãn human là CHẮC CHẮN, không phải suy đoán.
3. Chỉ có lớp human -> KHÔNG cần sinh `ai_text`, không tốn API, không thêm biến.

CẮT MẢNH: SAO CHÉP NGUYÊN VĂN, KHÔNG VIẾT LẠI
---------------------------------------------
`split_into_sentence_chunks` và `is_prose_chunk` dưới đây copy nguyên từ
`get-dataset.ipynb` — chính hàm đã dựng `body_chunk_1/2` của dataset gốc. Viết lại dù chỉ
một tham số (target_words=275, min_words=200, max_words=350) là bộ này KHÔNG so được với
`test_fp` nữa, và toàn bộ phép so k nhỏ vs k lớn mất nghĩa.

Khác duy nhất so với bản gốc: bỏ `random.sample(chunks, k=3)`. Vẫn giữ "bỏ mảnh đầu"
(mảnh đầu hay dính tiêu đề/header lặp).

ĐẦU RA
------
`output/dataset_test_fp_full_deconf2.parquet` — ĐÚNG schema của
`dataset_test_fp_deconf2.parquet` (`ai_text = None` -> mọi dòng label 0), nên code eval
sẵn có đọc được, chỉ đổi tên split.
"""
import argparse
import json
import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize

from preprocess import canonicalize

BASE_DIR = Path(__file__).resolve().parent
SRC = BASE_DIR / "results" / "with_doi_le_2018.jsonl"
OUT = BASE_DIR / "output" / "dataset_test_fp_full_deconf2.parquet"
MANIFEST = BASE_DIR / "output" / "testfp_full_manifest.json"
SPLITS = ("train", "val", "test_ood_model", "test_fp")

for pkg, res in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")]:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(pkg, quiet=True)

# ─── copy nguyên văn từ get-dataset.ipynb ────────────────────────────────────
_MATH_TOKEN_RE = re.compile(r"[0-9=+\-*/^_{}\\<>|~$]")


def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def is_prose_chunk(text, min_words=100, max_symbol_ratio=0.15, max_non_ascii_ratio=0.05):
    if not text or not text[0].isupper():
        return False
    tokens = text.split()
    if len(tokens) < min_words:
        return False
    if sum(1 for t in tokens if _MATH_TOKEN_RE.search(t)) / len(tokens) > max_symbol_ratio:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if len(text) > 0 and non_ascii / len(text) > max_non_ascii_ratio:
        return False
    return True


def split_into_sentence_chunks(text, target_words=275, min_words=200, max_words=350):
    sentences = sent_tokenize(clean_text(text))
    chunks, current, current_wc = [], [], 0
    for sent in sentences:
        sent_wc = len(sent.split())
        if current and current_wc + sent_wc > max_words:
            chunks.append(" ".join(current))
            current, current_wc = [], 0
        current.append(sent)
        current_wc += sent_wc
        if current_wc >= target_words:
            chunks.append(" ".join(current))
            current, current_wc = [], 0
    if current and current_wc >= min_words:
        chunks.append(" ".join(current))
    return [c for c in chunks if is_prose_chunk(c, min_words=min_words)]
# ─── hết phần copy ───────────────────────────────────────────────────────────


def used_paper_ids():
    ids = set()
    for s in SPLITS:
        p = BASE_DIR / "output" / f"dataset_{s}_deconf2.parquet"
        if p.exists():
            ids |= set(pd.read_parquet(p, columns=["paper_id"]).paper_id.astype(str))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-papers", type=int, default=1200)
    ap.add_argument("--min-chunks", type=int, default=10,
                    help="bỏ bài ít mảnh hơn mức này — cần k lớn mới trả lời được câu hỏi")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    used = used_paper_ids()
    print(f"{len(used):,} paper_id đã dùng ở 4 split -> loại hết")

    rows, kept, seen = [], 0, set()
    rng = np.random.default_rng(args.seed)
    cand = []
    with open(SRC) as fh:
        for line in fh:
            r = json.loads(line)
            pid = str(r["id"])
            if pid in used or pid in seen:
                continue
            seen.add(pid)
            cand.append((pid, r["full_text"], r.get("published_year"), r.get("categories", "")))
    print(f"{len(cand):,} bài sạch (chưa dùng ở split nào)")

    # xáo trước rồi cắt cho tới đủ số bài -> không thiên về bài dài
    order = rng.permutation(len(cand))
    for i in order:
        if kept >= args.n_papers:
            break
        pid, full, year, cats = cand[i]
        chunks = split_into_sentence_chunks(full)
        if len(chunks) > 3:            # giữ nguyên logic gốc: bỏ mảnh đầu
            chunks = chunks[1:]
        if len(chunks) < args.min_chunks:
            continue
        kept += 1
        for j, c in enumerate(chunks):
            rows.append({
                "sample_id": f"{pid}__full_{j:03d}",
                "paper_id": pid,
                "source": f"full_chunk_{j:03d}",
                "human_text": canonicalize(c),   # PHẢI canonicalize — model train trên bản đã canonicalize
                "ai_text": None,
                "ai_degree": 0,
                "sample_weight": 1.0,
                "split": "test_fp_full",
                "published_year": year,
                "categories": cats,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Không dựng được dòng nào — kiểm --min-chunks / đường dẫn nguồn.")
    df["notation_gap"] = np.nan          # không lọc: chỉ có lớp human, không có cặp để so
    df["notation_ok"] = True
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    k = df.groupby("paper_id").size()
    wc = df.human_text.str.split().str.len()
    man = {
        "built_from": str(SRC.relative_to(BASE_DIR)),
        "chunker": "get-dataset.ipynb split_into_sentence_chunks (copy nguyên văn)",
        "chunker_params": {"target_words": 275, "min_words": 200, "max_words": 350},
        "drop_first_chunk": True,
        "canonicalized": True,
        "excluded_paper_ids_from": list(SPLITS),
        "n_papers": int(len(k)), "n_chunks": int(len(df)),
        "chunks_per_paper": {"min": int(k.min()), "p25": int(k.quantile(.25)),
                             "median": int(k.median()), "p75": int(k.quantile(.75)),
                             "max": int(k.max()), "mean": round(float(k.mean()), 2)},
        "words_per_chunk_median": int(wc.median()),
        "published_year_range": [int(df.published_year.min()), int(df.published_year.max())],
        "label": "0 (human) cho MỌI dòng — bài xuất bản trước ChatGPT",
    }
    MANIFEST.write_text(json.dumps(man, indent=2, ensure_ascii=False))

    print(f"\n{len(k):,} bài · {len(df):,} mảnh")
    print(f"  mảnh/bài : min {k.min()} · p25 {k.quantile(.25):.0f} · trung vị {k.median():.0f} "
          f"· p75 {k.quantile(.75):.0f} · max {k.max()} · TB {k.mean():.2f}")
    print(f"  từ/mảnh  : trung vị {wc.median():.0f}  (test_fp cũ: ~284)")
    print(f"  năm      : {df.published_year.min()}–{df.published_year.max()}")
    print(f"\n-> {OUT}\n-> {MANIFEST}")
    print("\nSo với test_fp cũ: 1.529 bài · 4.290 mảnh · 2,81 mảnh/bài")
    print("Bộ này dùng ĐÚNG hàm cắt đó, chỉ không lấy mẫu 3 mảnh.")


if __name__ == "__main__":
    main()
