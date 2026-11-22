import os, json, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# ── Bản dataset dùng để train/eval ────────────────────────────────────
#   "deconf2"    = 02c_deconfound_v2.ipynb  <- ĐANG DÙNG
#                  canonicalize + truncate_pair_sentences + lọc notation_gap 85%.
#                  shortcut_audit.py đo đủ 22 đặc trưng: tầng process 0.5055 (cổng
#                  cứng, PASS) · tổng 0.8909. Train 7.025 cặp (85% của 8.265).
#   "deconf"     = 02b_deconfound.ipynb. ĐỪNG dùng cho nhánh này. Hai lý do:
#                  (a) con số "surface-GBM 0.6867, G6 PASS" mà 02b tự báo chỉ đo 6
#                      đặc trưng, thiếu avg_word_len (0.7795) lẫn type_token_ratio
#                      (0.7727). Đo đủ 22 thì là 0.9114 -> WARN, không PASS.
#                  (b) truncate_pair cắt thô đúng từ thứ min(len) nên chỉ 52,4% văn
#                      bản còn kết thúc đúng dấu câu (trước khi cắt: 99,0%). Detector
#                      ở file này chấm điểm bằng log-probability của từng token, nên
#                      token bị chặt giữa câu là perplexity GIẢ — và nó BẤT ĐỐI XỨNG,
#                      vì câu AI dài hơn (22,4 vs 20,4 từ) nên bên AI mất nhiều từ
#                      hơn một cách có hệ thống.
#   "normalized" = bản GỐC, CÒN shortcut: chỉ ĐẾM SỐ TỪ đã đoán đúng human/AI ở
#                  AUROC 0.7193 (surface-GBM 0.8080). G6 FAIL.
#
# PROVENANCE của số cũ (REPORT_2026-08-01.md §5) — đã truy được, ĐỪNG suy diễn lại:
# log còn lưu trong output notebook ghi sha256 val=c9d3957ae20b ood=2e9a47ca22e0
# fp=584cb43da181, khớp ĐÚNG dataset_{val,test_ood_model,test_fp}_normalized.parquet.
# Tức mọi số static/detectgpt trong báo cáo đó chạy trên "normalized" — chưa
# canonicalize, chưa cắt cặp — KHÔNG phải "deconf" như dòng tag cũ ghi (dòng đó nằm
# trên %%writefile nên chưa bao giờ tới được worker).
# Hệ quả: AUROC(val) < 0.5 của Fast-DetectGPT (0.4760) và Binoculars (0.4650) KHÔNG
# do tiền xử lý. Nguyên nhân phải tìm ở chỗ khác — mismatch thời đại backbone (gpt2-xl
# 2019 / gpt-j 2021 / falcon 2023 vs generator 2026), hoặc bug implementation. PHẢI có
# positive control trên một tập đã biết kết quả công bố trước khi tin bảng so sánh.
#
# PHẢI trùng tag với supervised_models/ và adversarial_models/ — cả 8 notebook đó đã
# chạy trên "deconf2". Lệch tag thì hai bảng số KHÔNG so được với nhau: deconf 8.265
# cặp vs deconf2 7.025 cặp, và nội dung text cũng khác (cắt theo từ vs cắt biên câu).
#
# PHẢI đặt TRONG body của %%writefile, KHÔNG phải ở đầu cell notebook. Đặt ở đầu cell
# thì %%writefile bị đẩy khỏi dòng 0 -> Jupyter hiểu thành line magic và báo "Line
# magic function `%%writefile` not found"; mà kể cả chạy được thì worker vẫn NameError
# vì nó là tiến trình riêng, không thấy biến của notebook.
#
# LƯU Ý: Kaggle dataset phải chứa dataset_{train,val,test_ood_model,test_fp}_deconf2.parquet
DATASET_TAG = "deconf2"
ENCODER_MODEL = "roberta-base"
MAX_LEN = 512
MIN_SUBSAMPLE = 40
STEP = 10
NUM_REPEATS = 3
RUN_NAME = "phd"
MODEL_DESC = ENCODER_MODEL

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
DEVICE = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"
SMOKE_TEST = os.environ.get("SMOKE", "0") == "1"
SMOKE_PER_CLASS = 16
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def log(*a):
    print(f"[rank{RANK}]", *a, flush=True)

from transformers import AutoTokenizer, AutoModel
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

tok = AutoTokenizer.from_pretrained(ENCODER_MODEL)
encoder = AutoModel.from_pretrained(ENCODER_MODEL).to(DEVICE).eval()
log("loaded:", ENCODER_MODEL, "on", DEVICE)

def _mst_length(points):
    d = cdist(points, points)
    return minimum_spanning_tree(d).sum()

def _phd_estimate(points, min_points=MIN_SUBSAMPLE, num_repeats=NUM_REPEATS, step=STEP, rng=None):
    n = points.shape[0]
    if n <= min_points + step:
        return np.nan
    rng = rng or np.random.default_rng(0)
    sizes = list(range(min_points, n, step))
    if sizes[-1] != n:
        sizes.append(n)
    log_n, log_e = [], []
    for m in sizes:
        lengths = [_mst_length(points[rng.choice(n, size=m, replace=False)]) for _ in range(num_repeats)]
        log_n.append(np.log(m))
        log_e.append(np.log(np.mean(lengths) + 1e-9))
    slope, _ = np.polyfit(log_n, log_e, 1)
    return 1.0 / (1.0 - slope) if slope < 1 else np.nan

@torch.no_grad()
def _embed_tokens(text):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    out = encoder(**enc, output_hidden_states=True)
    return out.hidden_states[-1][0].float().cpu().numpy()

def score(texts):
    """score cao = AI. Hướng ĐĂNG KÝ TRƯỚC theo paper (Tulchinskii et al., NeurIPS 2023:
    văn bản người có intrinsic dimension CAO hơn AI) -> score = -PHD. Không suy hướng từ
    nhãn val — xem statistical_branch_plan.md §4.

    _phd_estimate trả NaN khi không ước lượng được: text <= MIN_SUBSAMPLE+STEP token, hoặc
    slope >= 1. Trả NaN, KHÔNG trả 0.0. Lý do: -PHD nằm khoảng -8..-12, nên 0.0 CAO HƠN mọi
    mẫu đo được -> mẫu không đo được bị xếp thành "giống AI nhất" trong ROC. Nếu văn bản
    ngắn lệch về một lớp thì đó là bias có hệ thống. Cell metric loại NaN và báo số lượng.
    """
    out, n_nan = [], 0
    for text in texts:
        pts = _embed_tokens(text)
        d = _phd_estimate(pts)
        if np.isnan(d):
            n_nan += 1
            out.append(np.nan)
        else:
            out.append(-d)
    if n_nan:
        log(f"[!] {n_nan}/{len(texts)} mau khong uoc luong duoc PHD -> score = NaN (bi loai o metric)")
    return np.array(out)

# ── Data + sharding + chấm điểm + lưu partial (mỗi rank ghi 1 file riêng) ────
KAGGLE_INPUT = "/kaggle/input/datasets/<kaggle-user>/halfdatadetector"

def find_data_dir():
    target = f"dataset_val_{DATASET_TAG}.parquet"
    if (Path(KAGGLE_INPUT) / target).exists():
        return Path(KAGGLE_INPUT)
    for base in [Path(KAGGLE_INPUT), Path("output"), Path("."), Path("/kaggle/input")]:
        if base.exists():
            hits = list(base.rglob(target))
            if hits:
                return hits[0].parent
    raise FileNotFoundError(f"Khong tim thay {target}. Kiem tra KAGGLE_INPUT = {KAGGLE_INPUT}")

DATA_DIR = find_data_dir()
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUT_DIR = OUT_DIR / f"artifacts_{RUN_NAME}"
OUT_DIR.mkdir(parents=True, exist_ok=True)   # exist_ok=True -> an toàn dù 2 rank cùng gọi
log("DATA_DIR:", DATA_DIR, "| OUT_DIR:", OUT_DIR)

def melt_pairs(df):
    h = pd.DataFrame({"text": df["human_text"], "label": 0, "paper_id": df["paper_id"],
                       "ai_degree": 0, "generator_family": "human"})
    a = pd.DataFrame({"text": df["ai_text"], "label": 1, "paper_id": df["paper_id"],
                       "ai_degree": df.get("ai_degree", 3),
                       "generator_family": df.get("generator_family", "unknown")})
    out = pd.concat([h, a], ignore_index=True)
    return out[out["text"].notna() & (out["text"].astype(str).str.len() > 0)].reset_index(drop=True)

def melt_fp(df):
    has_ai = df["ai_text"].notna() if "ai_text" in df.columns else pd.Series(False, index=df.index)
    text = df["ai_text"].where(has_ai, df["human_text"]) if "ai_text" in df.columns else df["human_text"]
    out = pd.DataFrame({"text": text, "label": 0, "paper_id": df["paper_id"],
                         "ai_degree": df.get("ai_degree", 0).fillna(0).astype(int) if "ai_degree" in df.columns else 0,
                         "generator_family": "human"})
    return out[out["text"].notna() & (out["text"].astype(str).str.len() > 0)].reset_index(drop=True)

def load(name, fp_mode=False):
    p = DATA_DIR / f"dataset_{name}_{DATASET_TAG}.parquet"
    if not p.exists():
        log(f"thieu {p.name} - bo qua"); return None
    df = pd.read_parquet(p)
    return melt_fp(df) if fp_mode else melt_pairs(df)

def maybe_smoke(df):
    if df is None or not SMOKE_TEST:
        return df
    n = min(SMOKE_PER_CLASS, int(df["label"].value_counts().min()))
    return (df.groupby("label", group_keys=False)
              .sample(n=n, random_state=SEED)
              .sample(frac=1, random_state=SEED)
              .reset_index(drop=True))

for split_name, fp_mode in [("val", False), ("test_ood_model", False), ("test_fp", True)]:
    df = load(split_name, fp_mode=fp_mode)
    df = maybe_smoke(df)
    if df is None or len(df) == 0:
        continue
    shard = df.iloc[RANK::WORLD_SIZE].reset_index(drop=True)   # chia dòng, không chia theo model
    log(f"{split_name}: {len(df)} dong tong -> shard {len(shard)} dong (rank {RANK}/{WORLD_SIZE})")
    if len(shard) == 0:
        continue
    shard["score"] = score(shard["text"].tolist())
    if SMOKE_TEST:
        log(f"[SMOKE] KHONG luu partial cho {split_name} ({len(shard)} dong) - chi de kiem tra chay duoc.")
        continue
    out_path = OUT_DIR / f"partial_{split_name}_rank{RANK}.parquet"
    shard[["paper_id", "label", "ai_degree", "generator_family", "score"]].to_parquet(out_path, index=False)
    log(f"da luu {out_path.name} ({len(shard)} dong)")

log("XONG phan cua rank nay.")

# ═══ CHẾ ĐỘ MIX — chấm tập trộn của 06_mixing để dựng đường dose-response ═════════
# Bật bằng: RUN_MIX=1 <lệnh chạy worker như thường>
#
# Vì sao nằm ở worker này chứ không ở 07_score_mix: 07 load checkpoint bằng
# AutoModelForSequenceClassification. Bốn detector zero-shot KHÔNG có checkpoint
# (falcon/gpt-j đóng băng), nên chúng phải được chấm bằng chính score() ở trên. Và PHẢI là
# score() đó, không viết lại — hai đường trong cùng một figure mà dùng hai hàm chấm khác
# nhau thì figure không nói được gì.
#
# Tập mix KHÔNG đi qua vòng split thường: text đã được 06 assemble sẵn nên không melt, và
# không có label nhị phân — chỉ có mix_ratio_actual liên tục. Vòng split thường tính
# AUROC/τ, vô nghĩa ở đây. Nên mix ghi riêng pred_mix*.parquet và KHÔNG vào metrics.json.
#
# Schema đầu ra = ĐÚNG 15 cột KEEP của 07_score_mix + score -> gộp chung được với
# scores_all.parquet mà không lệch cột.
MIX_NAME = "dataset_test_ood_model_mix.parquet"
MIX_KEEP = ["sample_id", "paper_id", "source", "generator_family", "mix_structure",
            "mix_seed", "mix_k_slots", "n_slots", "mix_ratio_actual", "n_seams",
            "word_count", "align_score", "was_mixed", "ai_degree", "in_spec_band"]
RUN_MIX = os.environ.get("RUN_MIX", "0") == "1"
# 0 = FULL (20.358 dòng). Đặt số n để lấy n dòng MỖI Ô (bin x mix_structure) = 20 ô.
MIX_PER_CELL = int(os.environ.get("MIX_PER_CELL", "0"))


def find_mix_file():
    """mix.parquet ở old_dataset/ khi chạy local, ở /kaggle/input khi chạy Kaggle."""
    for base in [Path(KAGGLE_INPUT), Path("/kaggle/input"), Path("."), Path(".."),
                 Path("old_dataset"), Path("output")]:
        try:
            if base.exists():
                hits = sorted(base.rglob(MIX_NAME))
                if hits:
                    return hits[0]
        except OSError:
            continue
    return None


def load_mix():
    p = find_mix_file()
    if p is None:
        log(f"khong thay {MIX_NAME} -> bo qua che do mix")
        return None, None
    df = pd.read_parquet(p)
    have = [c for c in MIX_KEEP if c in df.columns]
    missing = [c for c in MIX_KEEP if c not in df.columns]
    if missing:
        log(f"[!] mix thieu cot {missing} -> de NaN de concat voi 07 khong lech cot")
    out = df[["text"] + have].copy()
    for c in missing:
        out[c] = np.nan
    out = out[out["text"].notna() & (out["text"].astype(str).str.len() > 0)]
    if MIX_PER_CELL > 0:
        _b = (out["mix_ratio_actual"].fillna(0.0) * 10).astype(int).clip(0, 9)
        out = (out.assign(_bin=_b)
                  .groupby(["_bin", "mix_structure"], group_keys=False)
                  .apply(lambda g: g.sample(n=min(len(g), MIX_PER_CELL), random_state=SEED))
                  .drop(columns="_bin"))
        log(f"mix: subsample {MIX_PER_CELL} dong/o (bin x structure)")
    return out.reset_index(drop=True), p


def run_mix():
    import math as _math
    import time as _time

    _rank = int(globals().get("RANK", 0))
    _ws = int(globals().get("WORLD_SIZE", 1))
    df, src = load_mix()
    if df is None or len(df) == 0:
        return
    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    shard = df.iloc[_rank::_ws].reset_index(drop=True) if _ws > 1 else df
    log(f"mix: {len(df)} dong tong -> {len(shard)} dong cua rank {_rank}/{_ws}  <- {src}")
    if SMOKE_TEST:
        _s = score(shard["text"].tolist()[:8])
        log(f"[SMOKE] mix: 8 dong, finite={np.isfinite(_s).all()}, "
            f"min/max={float(np.min(_s)):.4f}/{float(np.max(_s)):.4f}; KHONG luu.")
        return

    ck = OUT_DIR / f"chunks_mix_rank{_rank}"
    ck.mkdir(parents=True, exist_ok=True)
    step = int(globals().get("CHECKPOINT_ROWS", 0) or 200)
    n_ch = _math.ceil(len(shard) / step)
    log(f"mix: {n_ch} chunk x {step} dong, checkpoint tai {ck}")
    t_all = _time.time()
    for k in range(n_ch):
        a, b = k * step, min((k + 1) * step, len(shard))
        p = ck / f"mix_chunk{k:05d}.parquet"
        if p.exists():
            log(f"resume mix chunk {k + 1}/{n_ch}: da co")
            continue
        part = shard.iloc[a:b].copy()
        t0 = _time.time()
        part["score"] = score(part["text"].tolist())
        tmp = p.with_suffix(".tmp.parquet")
        part.drop(columns=["text"]).to_parquet(tmp, index=False)
        os.replace(tmp, p)          # atomic -> het gio session khong de lai chunk viet do dang
        rate = (b - a) / max(_time.time() - t0, 1e-9)
        log(f"mix chunk {k + 1}/{n_ch}: {rate:.2f} mau/s, "
            f"ETA con lai ~{(len(shard) - b) / max(rate, 1e-9) / 3600:.2f}h")

    parts = sorted(ck.glob("mix_chunk*.parquet"))
    if parts:
        allp = pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)
        dst = OUT_DIR / (f"pred_mix_rank{_rank}.parquet" if _ws > 1 else "pred_mix.parquet")
        allp.to_parquet(dst, index=False)
        log(f"da luu {dst.name}: {len(allp)} dong, wall={_time.time() - t_all:.0f}s")
        log(f"     cot: {list(allp.columns)}")


if RUN_MIX:
    run_mix()
