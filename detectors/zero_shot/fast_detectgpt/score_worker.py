import os, json, random, warnings, time, math, hashlib, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
# Paper black-box setting: GPT-J sampling / GPT-Neo-2.7B scoring (Table 2).
IMPL_VERSION = "paper-blackbox-analytic-v4-2026-07-28"
SAMPLING_MODEL = "EleutherAI/gpt-j-6b"
SCORING_MODEL = "EleutherAI/gpt-neo-2.7B"
MAX_LEN = 512
BATCH_SIZE = 1       # an toàn VRAM cho GPT-J FP16 trên T4; thử 2 sau khi SMOKE ổn định
CHECKPOINT_ROWS = 100
RUN_NAME = "fast_detectgpt_gptj_neo27"
MODEL_DESC = SAMPLING_MODEL + "/" + SCORING_MODEL + " (sampling/scoring)"
EVAL_SETTING = "paper_black_box_surrogate"
RUN_SIGNATURE = "|".join([IMPL_VERSION, SAMPLING_MODEL, SCORING_MODEL,
                          str(MAX_LEN), "fp16-weights-fp32-statistics", EVAL_SETTING])

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("Can 2 GPU: GPT-J tren cuda:0 va GPT-Neo-2.7B tren cuda:1.")
SAMPLING_DEVICE = torch.device("cuda:0")
SCORING_DEVICE = torch.device("cuda:1")
SMOKE_TEST = os.environ.get("SMOKE", "0") == "1"
SMOKE_PER_CLASS = 16
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def log(*a):
    print("[fastdetect]", *a, flush=True)

from transformers import AutoTokenizer, AutoModelForCausalLM

sampling_tok = AutoTokenizer.from_pretrained(SAMPLING_MODEL)
scoring_tok = AutoTokenizer.from_pretrained(SCORING_MODEL)
for _tok in {id(sampling_tok): sampling_tok, id(scoring_tok): scoring_tok}.values():
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token

# Load tuần tự với low_cpu_mem_usage để giảm đỉnh RAM host. Mỗi model nằm trọn trên một T4.
sampling_model = AutoModelForCausalLM.from_pretrained(
    SAMPLING_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(SAMPLING_DEVICE).eval()
sampling_model.config.use_cache = False
gc.collect(); torch.cuda.empty_cache()
scoring_model = AutoModelForCausalLM.from_pretrained(
    SCORING_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(SCORING_DEVICE).eval()
scoring_model.config.use_cache = False
gc.collect(); torch.cuda.empty_cache()
log("loaded:", SAMPLING_MODEL, "on", SAMPLING_DEVICE, "|", SCORING_MODEL, "on", SCORING_DEVICE)
log("impl:", IMPL_VERSION, "| setting:", EVAL_SETTING)
_vocab_crop_logged = False

@torch.inference_mode()
def score(texts, batch_size=BATCH_SIZE):
    """score cao = AI. Conditional probability curvature dạng đóng (Eq. Fast-DetectGPT)."""
    global _vocab_crop_logged
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        score_enc_cpu = scoring_tok(batch, return_tensors="pt", truncation=True, max_length=MAX_LEN,
                                    padding=True, return_token_type_ids=False)
        sample_enc_cpu = sampling_tok(batch, return_tensors="pt", truncation=True,
                                      max_length=MAX_LEN, padding=True,
                                      return_token_type_ids=False)
        if (sample_enc_cpu["input_ids"].shape != score_enc_cpu["input_ids"].shape or
                not torch.equal(sample_enc_cpu["input_ids"], score_enc_cpu["input_ids"])):
            raise ValueError("GPT-J/GPT-Neo tokenizers khong tuong thich token IDs.")
        sample_enc = sample_enc_cpu.to(SAMPLING_DEVICE)
        score_enc = score_enc_cpu.to(SCORING_DEVICE)
        labels = score_enc["input_ids"][:, 1:]
        attn = score_enc["attention_mask"][:, 1:].float()

        # Hai forward pass được schedule trên hai GPU độc lập. use_cache=False giảm VRAM.
        sample_logits_gpu0 = sampling_model(**sample_enc, use_cache=False).logits[:, :-1, :]
        score_logits = scoring_model(**score_enc, use_cache=False).logits[:, :-1, :]
        # Code goc Fast-DetectGPT cat ve min vocab khi hai LM co output embedding padding khac nhau.
        # GPT-J thuong xuat 50,400 logits, GPT-Neo/GPT-2 tokenizer dung 50,257 token IDs.
        sample_vocab, score_vocab = sample_logits_gpu0.size(-1), score_logits.size(-1)
        common_vocab = min(sample_vocab, score_vocab)
        if int(labels.max().item()) >= common_vocab:
            raise ValueError(f"Label token ID vuot common_vocab={common_vocab}.")
        if sample_vocab != score_vocab and not _vocab_crop_logged:
            log(f"vocab logits mismatch {sample_vocab} vs {score_vocab}; "
                f"crop ca hai ve {common_vocab} nhu code goc Fast-DetectGPT.")
            _vocab_crop_logged = True
        sample_logits_gpu0 = sample_logits_gpu0[..., :common_vocab]
        score_logits = score_logits[..., :common_vocab]
        sample_logits = sample_logits_gpu0.to(SCORING_DEVICE, non_blocking=True)
        del sample_logits_gpu0

        logp_score = torch.log_softmax(score_logits.float(), dim=-1)
        p_sample = torch.softmax(sample_logits.float(), dim=-1)

        obs_logp = logp_score.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        mean_t = (p_sample * logp_score).sum(dim=-1)
        mean_t2 = (p_sample * logp_score.pow(2)).sum(dim=-1)
        var_t = (mean_t2 - mean_t.pow(2)).clamp(min=0.0)  # chỉ chặn sai số âm do floating point

        # Paper Eq. (3),(7): Var(sum_t Z_t) = sum_t Var(Z_t),
        # nen sigma cua TOAN CHUOI = sqrt(sum_t var_t), KHONG PHAI sum_t sqrt(var_t).
        num = ((obs_logp - mean_t) * attn).sum(dim=1)
        den = (var_t * attn).sum(dim=1).clamp(min=1e-12).sqrt()
        batch_scores = num / den
        if not torch.isfinite(batch_scores).all():
            raise FloatingPointError("Fast-DetectGPT sinh score NaN/Inf.")
        out.extend(batch_scores.float().cpu().tolist())
        del sample_logits, score_logits, logp_score, p_sample, obs_logp, mean_t, mean_t2, var_t
    return np.array(out)

# ── Data + checkpoint theo chunk; một process điều khiển cả hai GPU ──────────
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
CHUNK_DIR = OUT_DIR / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
log("DATA_DIR:", DATA_DIR, "| OUT_DIR:", OUT_DIR)

def sha256_file(path, block_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

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

def validate_chunk(part, path, split_name, data_sha, expected_n, chunk_idx, start, end):
    required = {"_row_id", "_run_signature", "_data_sha256", "_expected_n", "_chunk_idx",
                "paper_id", "label", "ai_degree", "generator_family", "score"}
    missing = required.difference(part.columns)
    ok = (not missing and set(part["_run_signature"].unique()) == {RUN_SIGNATURE}
          and set(part["_data_sha256"].unique()) == {data_sha}
          and set(part["_expected_n"].unique()) == {expected_n}
          and set(part["_chunk_idx"].unique()) == {chunk_idx}
          and part["_row_id"].tolist() == list(range(start, end))
          and np.isfinite(part["score"].to_numpy()).all())
    if not ok:
        raise RuntimeError(f"Checkpoint cu/hong: {path}. Doi ten OUT_DIR hoac xoa dung folder run nay.")

def score_split(split_name, fp_mode=False):
    data_path = DATA_DIR / f"dataset_{split_name}_{DATASET_TAG}.parquet"
    df = maybe_smoke(load(split_name, fp_mode=fp_mode))
    if df is None or len(df) == 0:
        return
    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    if SMOKE_TEST:
        t0 = time.time()
        smoke_scores = score(df["text"].tolist())
        log(f"[SMOKE] {split_name}: {len(df)} dong, {time.time()-t0:.1f}s, "
            f"finite={np.isfinite(smoke_scores).all()}; KHONG luu checkpoint.")
        return

    data_sha = sha256_file(data_path)
    expected_n = len(df)
    n_chunks = math.ceil(expected_n / CHECKPOINT_ROWS)
    log(f"{split_name}: {expected_n} dong, {n_chunks} chunks, sha256={data_sha[:12]}...")
    split_t0 = time.time(); newly_scored = 0
    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHECKPOINT_ROWS
        end = min(start + CHECKPOINT_ROWS, expected_n)
        out_path = CHUNK_DIR / f"{split_name}_chunk{chunk_idx:05d}.parquet"
        if out_path.exists():
            part = pd.read_parquet(out_path)
            validate_chunk(part, out_path, split_name, data_sha, expected_n, chunk_idx, start, end)
            log(f"resume {split_name} chunk {chunk_idx+1}/{n_chunks}: da co {len(part)} dong")
            continue

        part = df.iloc[start:end].copy()
        t0 = time.time()
        part["score"] = score(part["text"].tolist())
        part["_run_signature"] = RUN_SIGNATURE
        part["_data_sha256"] = data_sha
        part["_expected_n"] = expected_n
        part["_chunk_idx"] = chunk_idx
        chunk_cols = ["_row_id", "_run_signature", "_data_sha256", "_expected_n", "_chunk_idx",
                      "paper_id", "label", "ai_degree", "generator_family", "score"]
        tmp_path = out_path.with_suffix(".tmp.parquet")
        part[chunk_cols].to_parquet(tmp_path, index=False)
        os.replace(tmp_path, out_path)  # atomic: session ngat khong de lai checkpoint nua voi
        newly_scored += len(part)
        dt = time.time() - t0
        rate = len(part) / max(dt, 1e-9)
        remaining = expected_n - end
        log(f"saved {split_name} chunk {chunk_idx+1}/{n_chunks}: {len(part)} dong, "
            f"{rate:.2f} samples/s, ETA moi ~{remaining/max(rate,1e-9)/3600:.2f}h")
    log(f"XONG {split_name}: new={newly_scored}, wall={time.time()-split_t0:.1f}s")

for _split_name, _fp_mode in [("val", False), ("test_ood_model", False), ("test_fp", True)]:
    score_split(_split_name, fp_mode=_fp_mode)

for _dev in [0, 1]:
    log(f"peak VRAM cuda:{_dev} = {torch.cuda.max_memory_allocated(_dev)/1024**3:.2f} GiB")
log("XONG FULL worker." if not SMOKE_TEST else "XONG SMOKE worker.")

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
