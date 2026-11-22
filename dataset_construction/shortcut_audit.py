"""Audit shortcut bề mặt — NGUỒN CHÂN LÝ DUY NHẤT cho cổng G6.

    OMP_NUM_THREADS=2 python3 shortcut_audit.py            # mọi biến thể trong output/
    OMP_NUM_THREADS=2 python3 shortcut_audit.py deconf2     # chỉ một biến thể
    OMP_NUM_THREADS=2 python3 shortcut_audit.py deconf2 --split train val

ĐẶT OMP_NUM_THREADS=2. Audit này chạy vài trăm lần fit GBM trên ma trận rất nhỏ
(<=16k dòng x 22 cột). Để OpenMP tự lấy hết 8 lõi thì phần lớn thời gian là đồng bộ
hoá thread, không phải tính: đo được 66 phút CPU / 15 phút treo với mặc định, so với
vài chục giây khi ép 2 thread.

VÌ SAO FILE NÀY TỒN TẠI
-----------------------
Trước đây 02b_deconfound và 03_diagnostics mỗi nơi tự khai một `FEATURE_COLS` sáu
phần tử, và bộ sáu đó KHÔNG chứa `avg_word_len` lẫn `type_token_ratio` — đúng hai
đặc trưng bề mặt mạnh nhất. Kết quả: cùng một tập train,

    6 đặc trưng  (02b tự chấm)  -> 0,6867  "G6 PASS"
    15 đặc trưng (check_shortcut) -> 0,9012  WARN

Con số PASS đó là hệ quả của việc CHỌN đặc trưng, không phải của việc shortcut đã
chết. Người phản biện chỉ cần thêm một đặc trưng là lật được kết luận. Nên bộ đặc
trưng phải được KHOÁ ở một chỗ, khai báo trước, và mọi báo cáo đọc từ đây.

BỘ ĐẶC TRƯNG (khoá, chỉ THÊM chứ không bớt — bớt là cherry-pick)
---------------------------------------------------------------
15 đặc trưng của check_shortcut_dataset.ipynb, cộng 7 đặc trưng phát sinh từ chẩn
đoán 2026-08-12 (docs/SHORTCUT_DEFENSE.md).

Mỗi đặc trưng được gán một trong BỐN tầng, phân loại theo *lý do đặc trưng đó được
phép lệch giữa hai lớp* — chứ không theo việc nó có làm con số đẹp hay không:

  process  Dấu vết QUY TRÌNH. Không có lý do chính đáng nào để lệch: độ dài tài
           liệu, mã hoá ký tự, dấu vết bóc file, chỗ ta cắt văn bản. -> phải ~0,50.
           Đây là cổng CỨNG. Lệch ở đây là LỖI dataset, không phải hiện tượng.

  fidelity Trung thực nội dung. human_text là đoạn bài báo THẬT nên còn biến toán,
           số liệu, tên riêng; bản AI viết lại thì lược đi. Lệch ở đây một phần là
           đặc điểm thật của việc viết lại, một phần là artifact của prompt sinh dữ
           liệu — không tách được sạch, và không sửa được bằng phẫu thuật văn bản
           (xoá số = xoá nội dung). -> ép xuống bằng LỌC CẶP đến mức <= 0,60, và
           báo cáo cả cái giá phải trả (số cặp mất).

  style    Văn phong LLM thật: từ dài hơn, câu dài hơn, từ vựng đa dạng hơn. -> KHÔNG
           được diệt. Diệt là xoá bài toán. Cao ở đây là ĐÚNG, và audit sẽ CẢNH BÁO
           nếu nó về gần 0,5 (dấu hiệu đã chuẩn hoá quá tay).

  derived  Hàm xác định của đặc trưng khác đã kiểm soát: char_count = word_count x
           avg_word_len, sentence_count = word_count / avg_sentence_len. Báo cáo để
           minh bạch, không tính là kênh rò rỉ độc lập.

Vì sao KHÔNG lấy con số TỔNG làm cổng: một dataset phát hiện AI lành mạnh không thể
có AUC bề mặt tổng bằng 0,5 (xem docs/DATASET_BUILD_FOR_PAPER.md §6.3). Đặt cổng
"tổng <= 0,70" tạo ra một cái cổng không ai qua được, và áp lực phải chọn bộ đặc
trưng cho nó qua — đúng thứ đã xảy ra với bộ 6 đặc trưng. Tổng vẫn được in ra, nhưng
để BÁO CÁO chứ không để phán quyết.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
REPORT_DIR = OUTPUT_DIR / "shortcut_audit"

RANDOM_STATE = 42
N_SPLITS = 5

# Cổng CỨNG cho tầng process: đúng ra phải bằng 0,50 vì kênh này không mang thông tin
# về việc ai viết. 0,55 là biên độ cho nhiễu hữu hạn mẫu (std của AUC lẻ ~0,006 trên
# 16,5k dòng, nên 0,55 là ~8 sigma — vượt là tín hiệu thật, không phải nhiễu).
PROCESS_MAX_AUC = 0.55
# Cổng MỀM cho tầng fidelity: không thể về 0,50 mà không xoá nội dung. 0,60 = "một
# kênh đơn lẻ chỉ còn lợi thế biên".
FIDELITY_MAX_AUC = 0.60
# Tầng style mà TỤT xuống đây thì đã chuẩn hoá quá tay -> cảnh báo, không phải cổng.
STYLE_MIN_AUC = 0.55

_MATH_TOKEN_RE = re.compile(r"[0-9=+\-*/^_{}\\<>|~$]")
_PUNCT_RE = re.compile(r"[.,;:!?\"'()\[\]{}]")
_CITATION_RE = re.compile(r"\[[0-9,;\-\s]+\]|\([A-Z][A-Za-z-]+(?: et al\.)?,? \d{4}[a-z]?\)")
_URL_RE = re.compile(r"https?://|www\.|\bdoi\s*:")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_ALPHA_RE = re.compile(r"^[A-Za-z]+$")
_ALLCAPS_RE = re.compile(r"^[A-Z]{2,}$")

# đặc trưng -> tầng. Thứ tự này là thứ tự cột trong mọi báo cáo.
CHANNEL = {
    # ── process: dấu vết quy trình, cổng CỨNG, phải về ~0,50 ────────────
    "word_count": "process",              # độ dài tài liệu
    "non_ascii_ratio": "process",         # mã hoá / ký hiệu chưa phiên chuyển
    "citation_ratio": "process",           # dấu vết trích dẫn của bài gốc
    "url_ratio": "process",
    "newline_ratio": "process",            # dấu vết bóc file
    "starts_lowercase": "process",         # cắt đầu đoạn
    "ends_terminal": "process",            # cắt giữa câu — do CHÍNH ta cắt
    # ── fidelity: bản AI viết lại lược ký hiệu/số/tên riêng của bài gốc ─
    # Xếp vào đây, KHÔNG xếp vào style, dù một phần là đặc điểm thật của việc viết
    # lại: xếp vào fidelity thì phải ép nó xuống, và ép xuống chỉ làm dataset khó bắt
    # bẻ hơn. Xếp vào style thì được miễn ép — đó mới là hướng sai.
    "single_char_frac": "fidelity",        # biến toán (x, k, n) của bài gốc
    "titlecase_token_frac": "fidelity",    # tên riêng, tên phương pháp
    "numeric_token_frac": "fidelity",      # số liệu
    "digit_ratio": "fidelity",             # bản mức ký tự của numeric_token_frac
    "allcaps_token_frac": "fidelity",      # acronym (đo được chỉ 0,551)
    "uppercase_ratio": "fidelity",         # đo được: bỏ chữ đầu câu vẫn 0,636 nên
                                           # phần lớn là tên riêng, không phải số câu
    "math_token_ratio": "fidelity",        # "token có chữ số hoặc ký hiệu"
    # ── style: văn phong LLM thật, KHÔNG được diệt ──────────────────────
    "avg_word_len": "style",               # 4,94 vs 5,82 ký tự/từ
    "avg_word_len_alpha": "style",         # vẫn 0,809 khi chỉ lấy token thuần chữ
    "type_token_ratio": "style",
    "ttr_alpha_w200": "style",             # cùng 200 token thuần chữ = so công bằng
    "avg_sentence_len": "style",
    "punct_ratio": "style",
    # ── derived: hàm xác định của đặc trưng đã kiểm soát ────────────────
    "char_count": "derived",               # = word_count x avg_word_len
    "sentence_count": "derived",           # = word_count / avg_sentence_len
}
FEATURE_COLS = list(CHANNEL)
PROCESS_COLS = [c for c, k in CHANNEL.items() if k == "process"]
FIDELITY_COLS = [c for c, k in CHANNEL.items() if k == "fidelity"]
STYLE_COLS = [c for c, k in CHANNEL.items() if k == "style"]
TIER_MAX_AUC = {"process": PROCESS_MAX_AUC, "fidelity": FIDELITY_MAX_AUC}

# Bộ 6 đặc trưng của cổng G6 CŨ. Giữ lại CHỈ để in ra cạnh con số thật, cho thấy
# khoảng cách giữa hai cách đo. Không dùng để phán quyết.
LEGACY_G6_COLS = ["non_ascii_ratio", "math_token_ratio", "starts_lowercase",
                  "word_count", "avg_sentence_len", "punct_ratio"]


def surface_features(text):
    """21 đặc trưng bề mặt + 1 derived. Không dùng bất kỳ thông tin ngữ nghĩa nào."""
    text = str(text)
    tokens = text.split()
    n_chars = max(1, len(text))
    n_tokens = max(1, len(tokens))
    alpha_chars = [c for c in text if c.isalpha()]
    alpha_tok = [t for t in tokens if _ALPHA_RE.match(t)]
    sentences = [s for s in _SENT_RE.split(text.strip()) if s]
    n_sentences = max(1, len(sentences))
    return {
        "word_count": len(tokens),
        "sentence_count": len(sentences),
        "non_ascii_ratio": sum(ord(c) > 127 for c in text) / n_chars,
        "citation_ratio": len(_CITATION_RE.findall(text)) / n_sentences,
        "url_ratio": len(_URL_RE.findall(text)) / n_sentences,
        "newline_ratio": text.count("\n") / n_chars,
        "starts_lowercase": int(bool(text) and text[0].islower()),
        "ends_terminal": int(text.rstrip().endswith((".", "!", "?"))),
        "single_char_frac": sum(1 for t in tokens if len(t) == 1) / n_tokens,
        "titlecase_token_frac": sum(1 for t in tokens if t[:1].isupper()) / n_tokens,
        "numeric_token_frac": sum(1 for t in tokens if any(c.isdigit() for c in t)) / n_tokens,
        "digit_ratio": sum(c.isdigit() for c in text) / n_chars,
        "allcaps_token_frac": sum(1 for t in tokens if _ALLCAPS_RE.match(t.strip(".,"))) / n_tokens,
        "avg_word_len": float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
        "avg_word_len_alpha": float(np.mean([len(t) for t in alpha_tok])) if alpha_tok else 0.0,
        "type_token_ratio": len({t.lower() for t in tokens}) / n_tokens,
        "ttr_alpha_w200": (len({t.lower() for t in alpha_tok[:200]}) / max(1, len(alpha_tok[:200]))),
        "avg_sentence_len": len(tokens) / n_sentences,
        "punct_ratio": len(_PUNCT_RE.findall(text)) / n_chars,
        "uppercase_ratio": sum(c.isupper() for c in alpha_chars) / max(1, len(alpha_chars)),
        "math_token_ratio": sum(bool(_MATH_TOKEN_RE.search(t)) for t in tokens) / n_tokens,
        "char_count": len(text),
    }


def pair_to_long(df, keep_cols=()):
    """Cặp (human_text, ai_text) -> bảng long. paper_id giữ nguyên để CV theo group."""
    keep = [c for c in ("paper_id", "sample_id", *keep_cols) if c in df.columns]
    human = df[keep].copy()
    human["text"] = df["human_text"].values
    human["label"] = 0
    ai = df[keep].copy()
    ai["text"] = df["ai_text"].values
    ai["label"] = 1
    long = pd.concat([human, ai], ignore_index=True)
    long = long[long["text"].notna() & long["text"].astype(str).str.strip().ne("")]
    return long.reset_index(drop=True)


def featurize(long):
    F = pd.DataFrame([surface_features(t) for t in long["text"]])[FEATURE_COLS]
    y = long["label"].astype(int).to_numpy()
    g = long["paper_id"].astype(str).to_numpy()
    return F, y, g


def _models():
    return {
        "GBM": HistGradientBoostingClassifier(max_depth=4, random_state=RANDOM_STATE),
        "LogReg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
    }


def grouped_auc(estimator, X, y, g, n_splits=N_SPLITS):
    """AUROC cross-val theo group=paper_id.

    BẮT BUỘC theo group: chia fold ngẫu nhiên thì bản human và bản AI của cùng một
    bài rơi vào hai fold khác nhau -> rò rỉ chủ đề, AUROC bị thổi phồng.
    """
    n_splits = min(n_splits, *[len(np.unique(g[y == c])) for c in np.unique(y)])
    if n_splits < 2:
        return float("nan"), float("nan")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = [
        roc_auc_score(y[te], clone(estimator).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
        for tr, te in cv.split(X, y, g)
    ]
    return float(np.mean(scores)), float(np.std(scores))


def _matrix(F, cols):
    return F[list(cols)].replace([np.inf, -np.inf], np.nan).to_numpy()


def audit_frame(df, tag, ngram_baseline=False):
    """Audit MỘT dataframe dạng cặp. Trả dict kết quả, không in gì."""
    long = pair_to_long(df)
    F, y, g = featurize(long)

    combos = {
        "all": FEATURE_COLS,
        "process_only": PROCESS_COLS,
        "fidelity_only": FIDELITY_COLS,
        "style_only": STYLE_COLS,
        "legacy_g6_6feat": LEGACY_G6_COLS,
    }
    totals = {}
    for combo, cols in combos.items():
        for mname, model in _models().items():
            if combo != "all" and mname == "LogReg":
                continue          # LogReg chỉ cần cho bộ đầy đủ, tránh chạy 8 lần
            mean, std = grouped_auc(model, _matrix(F, cols), y, g)
            totals[f"{combo}|{mname}"] = (mean, std)

    per_feature = []
    for col in FEATURE_COLS:
        X1 = _matrix(F, [col])
        finite = X1[np.isfinite(X1)]
        if len(np.unique(finite)) < 2:
            mean, std = 0.5, 0.0
        else:
            mean, std = grouped_auc(
                HistGradientBoostingClassifier(max_depth=3, random_state=RANDOM_STATE),
                X1, y, g)
        per_feature.append({
            "dataset": tag, "feature": col, "channel": CHANNEL[col],
            "auc_mean": mean, "auc_std": std,
            "human_mean": float(F.loc[y == 0, col].mean()),
            "ai_mean": float(F.loc[y == 1, col].mean()),
            "tier_max_auc": TIER_MAX_AUC.get(CHANNEL[col]),
            "over_threshold": bool(mean > TIER_MAX_AUC.get(CHANNEL[col], np.inf)),
            "style_too_low": bool(CHANNEL[col] == "style" and mean < STYLE_MIN_AUC),
        })

    ngram = None
    if ngram_baseline:
        # Trần của "mô hình nông": TF-IDF char 3-5gram + LogReg. Người phản biện sẽ
        # hỏi "bag-of-ngram đã được bao nhiêu?" -> trả lời trước khi bị hỏi.
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                              max_features=200_000, sublinear_tf=True)
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        texts = long["text"].astype(str).to_numpy()
        scores = []
        for tr, te in cv.split(texts, y, g):
            Xtr = vec.fit_transform(texts[tr])
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, y[tr])
            scores.append(roc_auc_score(y[te], clf.predict_proba(vec.transform(texts[te]))[:, 1]))
        ngram = (float(np.mean(scores)), float(np.std(scores)))

    def worst(tier):
        return max((r for r in per_feature if r["channel"] == tier),
                   key=lambda r: r["auc_mean"])

    worst_process, worst_fidelity = worst("process"), worst("fidelity")
    flattened_style = [r["feature"] for r in per_feature if r["style_too_low"]]
    return {
        "dataset": tag,
        "n_rows": len(long),
        "n_groups": int(pd.Series(g).nunique()),
        "totals": totals,
        "per_feature": per_feature,
        "ngram": ngram,
        "worst_process": worst_process,
        "worst_fidelity": worst_fidelity,
        "flattened_style": flattened_style,
        # Cổng cứng = process. fidelity vượt ngưỡng thì WARN chứ không FAIL: nó không
        # sửa được bằng biến đổi văn bản, chỉ ép xuống được bằng lọc cặp, và mỗi mức
        # lọc đều có cái giá phải trả bằng dữ liệu.
        "verdict": ("FAIL" if worst_process["auc_mean"] > PROCESS_MAX_AUC
                    else "WARN" if worst_fidelity["auc_mean"] > FIDELITY_MAX_AUC
                    else "PASS"),
    }


def print_audit(res):
    print("=" * 86)
    print(f"AUDIT  {res['dataset']}   rows={res['n_rows']:,}  groups={res['n_groups']:,}")
    print("=" * 86)
    print("  TỔNG (mô hình bề mặt, CV theo paper_id):")
    notes = {
        "legacy_g6_6feat": "  <- cách đo CŨ (6 đặc trưng), in để so, KHÔNG phán quyết",
        "process_only": f"  <- cổng CỨNG (mỗi kênh <= {PROCESS_MAX_AUC})",
        "fidelity_only": f"  <- cổng MỀM (mỗi kênh <= {FIDELITY_MAX_AUC})",
        "style_only": "  <- văn phong thật, cao là ĐÚNG",
        "all": "  <- con số trung thực để báo cáo",
    }
    for key, (mean, std) in res["totals"].items():
        combo, model = key.split("|")
        print(f"    {combo:<18}{model:<8}{mean:.4f} ± {std:.4f}{notes.get(combo, '')}")
    if res["ngram"]:
        print(f"    {'char 3-5gram':<18}{'LogReg':<8}{res['ngram'][0]:.4f} ± {res['ngram'][1]:.4f}"
              "  <- trần mô hình NÔNG")

    print(f"\n  {'đặc trưng':<24}{'tầng':<10}{'AUC lẻ':>9}{'human':>12}{'AI':>12}   ngưỡng")
    order = {"process": 0, "fidelity": 1, "style": 2, "derived": 3}
    for r in sorted(res["per_feature"], key=lambda r: (order[r["channel"]], -r["auc_mean"])):
        flag = " !!" if r["over_threshold"] else (" ??" if r["style_too_low"] else "   ")
        thr = f"<= {r['tier_max_auc']}" if r["tier_max_auc"] else (
            f">= {STYLE_MIN_AUC}" if r["channel"] == "style" else "-")
        print(f" {flag}{r['feature']:<24}{r['channel']:<10}{r['auc_mean']:>9.4f}"
              f"{r['human_mean']:>12.4f}{r['ai_mean']:>12.4f}   {thr}")

    wp, wf = res["worst_process"], res["worst_fidelity"]
    print(f"\n  process tệ nhất : {wp['feature']:<22}{wp['auc_mean']:.4f}  (cổng cứng {PROCESS_MAX_AUC})")
    print(f"  fidelity tệ nhất: {wf['feature']:<22}{wf['auc_mean']:.4f}  (cổng mềm {FIDELITY_MAX_AUC})")
    if res["flattened_style"]:
        print(f"  ?? văn phong bị làm phẳng: {', '.join(res['flattened_style'])}"
              " -> chuẩn hoá quá tay, đang xoá chính tín hiệu cần học")
    reason = {"FAIL": "  -> còn dấu vết QUY TRÌNH, phải sửa cách dựng dữ liệu",
              "WARN": "  -> process sạch, nhưng fidelity còn lệch: siết lọc cặp hoặc ghi rõ trong KLTN",
              "PASS": ""}
    print(f"  PHÁN QUYẾT: {res['verdict']}{reason[res['verdict']]}")
    print()


def leakage_check(frames):
    """Rò rỉ cross-split: trùng paper_id hoặc trùng văn bản sau chuẩn hoá."""
    def text_hash(t):
        return hashlib.sha256(re.sub(r"\s+", " ", str(t)).strip().lower().encode()).hexdigest()

    rows = []
    for split, df in frames.items():
        for col in ("human_text", "ai_text"):
            if col not in df.columns:
                continue
            sub = df[df[col].notna()]
            rows.append(pd.DataFrame({
                "split": split,
                "paper_id": sub["paper_id"].astype(str).values,
                "text_hash": [text_hash(t) for t in sub[col]],
            }))
    L = pd.concat(rows, ignore_index=True)
    splits = sorted(L["split"].unique())
    paper = pd.DataFrame(0, index=splits, columns=splits, dtype=int)
    text = paper.copy()
    for a in splits:
        for b in splits:
            paper.loc[a, b] = len(set(L[L.split == a].paper_id) & set(L[L.split == b].paper_id))
            text.loc[a, b] = len(set(L[L.split == a].text_hash) & set(L[L.split == b].text_hash))
    off_paper = sum(paper.loc[a, b] for i, a in enumerate(splits) for b in splits[i + 1:])
    off_text = sum(text.loc[a, b] for i, a in enumerate(splits) for b in splits[i + 1:])
    return paper, text, int(off_paper), int(off_text)


def variant_paths(tag):
    return {sp: OUTPUT_DIR / f"dataset_{sp}_{tag}.parquet"
            for sp in ("train", "val", "test_ood_model", "test_fp")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*", default=None,
                    help="biến thể: normalized | deconf | deconf2 (mặc định: tất cả tìm thấy)")
    ap.add_argument("--split", nargs="+", default=["train"],
                    help="split cần audit (mặc định chỉ train)")
    ap.add_argument("--ngram", action="store_true", help="thêm trần char n-gram (chậm)")
    ap.add_argument("--no-leakage", action="store_true")
    args = ap.parse_args()

    tags = args.tags or [t for t in ("normalized", "deconf", "deconf2")
                         if variant_paths(t)["train"].exists()]
    if not tags:
        print("Không tìm thấy biến thể nào trong", OUTPUT_DIR)
        return 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_per_feature, all_totals = [], []
    for tag in tags:
        paths = variant_paths(tag)
        for split in args.split:
            p = paths[split]
            if not p.exists():
                print(f"[skip] {p.name} không tồn tại")
                continue
            df = pd.read_parquet(p)
            if not {"human_text", "ai_text"} <= set(df.columns):
                print(f"[skip] {p.name} không phải dạng cặp")
                continue
            if df["ai_text"].isna().all() or df["human_text"].isna().all():
                print(f"[skip] {p.name} chỉ có một lớp")
                continue
            res = audit_frame(df.dropna(subset=["human_text", "ai_text"]),
                              f"{tag}/{split}", ngram_baseline=args.ngram)
            print_audit(res)
            all_per_feature += res["per_feature"]
            for key, (mean, std) in res["totals"].items():
                combo, model = key.split("|")
                all_totals.append({"dataset": res["dataset"], "combo": combo, "model": model,
                                   "auc_mean": mean, "auc_std": std,
                                   "verdict": res["verdict"]})

        if not args.no_leakage:
            frames = {sp: pd.read_parquet(p) for sp, p in paths.items() if p.exists()}
            if len(frames) > 1:
                paper, text, off_p, off_t = leakage_check(frames)
                print(f"  Rò rỉ cross-split ({tag}): paper_id trùng = {off_p}"
                      f" -> {'PASS' if off_p == 0 else 'FAIL'}"
                      f" | văn bản trùng = {off_t} -> {'PASS' if off_t == 0 else 'FAIL'}\n")
                paper.to_csv(REPORT_DIR / f"leakage_paper_{tag}.csv")
                text.to_csv(REPORT_DIR / f"leakage_text_{tag}.csv")

    if all_per_feature:
        pd.DataFrame(all_per_feature).to_csv(REPORT_DIR / "per_feature_auc.csv", index=False)
        pd.DataFrame(all_totals).to_csv(REPORT_DIR / "totals_auc.csv", index=False)
        (REPORT_DIR / "audit_config.json").write_text(json.dumps({
            "feature_cols": FEATURE_COLS, "channel": CHANNEL,
            "process_max_auc": PROCESS_MAX_AUC,
            "fidelity_max_auc": FIDELITY_MAX_AUC,
            "style_min_auc": STYLE_MIN_AUC,
            "n_splits": N_SPLITS, "random_state": RANDOM_STATE,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Báo cáo ->", REPORT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
