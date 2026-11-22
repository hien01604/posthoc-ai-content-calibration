import os, json, random, warnings, time, math, hashlib, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path

# 2 checkpoint Falcon ~29 GB: day cache sang /kaggle/temp (scratch) de khong an quota output 20 GB.
if Path("/kaggle/temp").exists():
    os.environ.setdefault("HF_HOME", "/kaggle/temp/hf_cache")

import numpy as np
import pandas as pd
import torch

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
# ── CONFIG — khop mac dinh cua Binoculars.__init__ ───────────────────────────
IMPL_VERSION = "paper-exact-falcon-pair-modelparallel-v1-2026-07-28"
OBSERVER_MODEL = "tiiuae/falcon-7b"            # M_obs, code goc: DEVICE_1
PERFORMER_MODEL = "tiiuae/falcon-7b-instruct"  # M_perf, code goc: DEVICE_2
MAX_LEN = 512          # = max_token_observed mac dinh
BATCH_SIZE = 1         # an toan VRAM cho 7B fp16 tren T4; thu 2 sau khi SMOKE on dinh
POS_CHUNK = 64         # so vi tri token / luot -> khong giu [B,512,65024] fp32 mot cuc
CHECKPOINT_ROWS = 100
RUN_NAME = "binoculars_falcon7b"
MODEL_DESC = f"{OBSERVER_MODEL} + {PERFORMER_MODEL} (observer/performer)"
FORMULA = "ppl=logPPL_performer(shift1) / xppl=H(p=observer,q=performer,no-shift,all-L); score=-B"
RUN_SIGNATURE = "|".join([IMPL_VERSION, OBSERVER_MODEL, PERFORMER_MODEL, str(MAX_LEN),
                          "fp16-weights-fp32-statistics", FORMULA])

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("Can 2 GPU: falcon-7b tren cuda:0, falcon-7b-instruct tren cuda:1. "
                       "1 GPU khong du 2x13.44 GiB.")
OBSERVER_DEVICE = torch.device("cuda:0")
PERFORMER_DEVICE = torch.device("cuda:1")
SMOKE_TEST = os.environ.get("SMOKE", "0") == "1"
SMOKE_PER_CLASS = 16
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def log(*a):
    print("[binoculars]", *a, flush=True)

from transformers import AutoTokenizer, AutoModelForCausalLM

# Paper footnote 2 + assert_tokenizer_consistency() cua code goc: 2 model PHAI chung tokenizer.
obs_tok = AutoTokenizer.from_pretrained(OBSERVER_MODEL)
prf_tok = AutoTokenizer.from_pretrained(PERFORMER_MODEL)
for _t in (obs_tok, prf_tok):
    if _t.pad_token is None:
        _t.pad_token = _t.eos_token
    _t.padding_side = "right"   # causal LM + forward thuong: right-padding moi giu dung position id
if obs_tok.get_vocab() != prf_tok.get_vocab():
    raise RuntimeError("Tokenizer cua observer va performer khac nhau - Binoculars khong dinh nghia duoc.")

# Load tuan tu, low_cpu_mem_usage -> giam dinh RAM host (moi model ~14.4 GB tren dia).
observer = AutoModelForCausalLM.from_pretrained(
    OBSERVER_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(OBSERVER_DEVICE).eval()
observer.config.use_cache = False
gc.collect(); torch.cuda.empty_cache()
performer = AutoModelForCausalLM.from_pretrained(
    PERFORMER_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(PERFORMER_DEVICE).eval()
performer.config.use_cache = False
gc.collect(); torch.cuda.empty_cache()
log("loaded:", OBSERVER_MODEL, "on", OBSERVER_DEVICE, "|", PERFORMER_MODEL, "on", PERFORMER_DEVICE)
log("impl:", IMPL_VERSION)
for _d in (0, 1):
    log(f"VRAM sau khi load cuda:{_d} = {torch.cuda.memory_allocated(_d)/1024**3:.2f} GiB")

@torch.inference_mode()
def score(texts, batch_size=BATCH_SIZE):
    """score cao = AI.

    Khop reference implementation (ahans30/Binoculars, binoculars/{detector,metrics}.py):
      ppl   = perplexity(encodings, PERFORMER_logits)   -> shift 1 token, mean CE co attention-mask
      x_ppl = entropy(p=OBSERVER, q=PERFORMER)          -> KHONG shift, mean tren toan bo L vi tri
      B     = ppl / x_ppl        (paper: B thap = AI)   -> tra ve -B de "cao = AI"
    Tat ca o log-space (mean NLL), khong exp() o dau ca.
    """
    out, n_short = [], 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # tokenize 2 lan -> 2 BatchEncoding doc lap (BatchEncoding.to() sua in-place, khong clone!)
        enc_obs = obs_tok(batch, return_tensors="pt", truncation=True, max_length=MAX_LEN,
                          padding=True, return_token_type_ids=False)
        enc_prf = prf_tok(batch, return_tensors="pt", truncation=True, max_length=MAX_LEN,
                          padding=True, return_token_type_ids=False)
        if not torch.equal(enc_obs["input_ids"], enc_prf["input_ids"]):
            raise ValueError("2 tokenizer cho ra token IDs khac nhau.")
        enc_obs = enc_obs.to(OBSERVER_DEVICE)
        enc_prf = enc_prf.to(PERFORMER_DEVICE)
        ids = enc_prf["input_ids"]
        am = enc_prf["attention_mask"].float()
        B, L = ids.shape
        n_short += int((am.sum(1) < 64).sum().item())

        # Launch CA HAI forward truoc khi cham vao ket qua -> 2 T4 chay chong nhau (kernel async).
        logits_obs_g0 = observer(**enc_obs).logits
        logits_prf = performer(**enc_prf).logits

        # Code goc crop ve min vocab khi 2 LM co output embedding padding khac nhau (falcon: 65024 ca hai).
        cv = min(logits_obs_g0.size(-1), logits_prf.size(-1))
        if int(ids.max().item()) >= cv:
            raise ValueError(f"Token ID vuot common vocab = {cv}.")
        logits_prf = logits_prf[..., :cv]

        ppl_sum = torch.zeros(B, device=PERFORMER_DEVICE, dtype=torch.float32)
        ce_sum = torch.zeros(B, device=PERFORMER_DEVICE, dtype=torch.float32)
        for a in range(0, L, POS_CHUNK):
            e = min(a + POS_CHUNK, L)
            # chuyen tung chunk logits observer sang GPU cua performer (~8 MB voi chunk 64)
            lp_obs = torch.log_softmax(
                logits_obs_g0[:, a:e, :cv].to(PERFORMER_DEVICE, non_blocking=True).float(), dim=-1)
            lp_prf = torch.log_softmax(logits_prf[:, a:e].float(), dim=-1)

            # mau so - Eq. (3): -sum_v M_obs(v) * log M_perf(v), tren MOI vi tri (khong shift)
            ce_sum += (-(lp_obs.exp() * lp_prf).sum(-1) * am[:, a:e]).sum(1)

            # tu so - Eq. (2): log-PPL cua PERFORMER, logit tai vi tri p du doan token p+1
            pe = min(e, L - 1)
            if pe > a:
                lab = ids[:, a + 1:pe + 1]
                msk = am[:, a + 1:pe + 1]
                tok_lp = lp_prf[:, :pe - a].gather(-1, lab.unsqueeze(-1)).squeeze(-1)
                ppl_sum += -(tok_lp * msk).sum(1)
            del lp_obs, lp_prf
        del logits_obs_g0, logits_prf

        ppl = ppl_sum / am[:, 1:].sum(1).clamp(min=1)     # shift -> mau la L-1 token thuc
        xppl = ce_sum / am.sum(1).clamp(min=1)            # khong shift -> mau la L token thuc
        b = ppl / xppl.clamp(min=1e-9)
        if not torch.isfinite(b).all():
            raise FloatingPointError("Binoculars sinh score NaN/Inf (fp16 overflow?).")
        out.extend((-b).float().cpu().tolist())
    if n_short:
        log(f"[!] {n_short} doc < 64 token - paper (Fig. 2) chi cho thay on dinh tu ~50-100 token tro len")
    return np.array(out)

# ── Data + checkpoint theo chunk; mot process dieu khien ca hai GPU ──────────
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

# melt_pairs/melt_fp: GIU Y HET supervised_models/ va adversarial_models/ (bo cot sample_weight
# vi khong train) -> cung tap dong, cung nhan, moi so sanh duoc.
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

CHUNK_COLS = ["_row_id", "_run_signature", "_data_sha256", "_expected_n", "_chunk_idx",
              "paper_id", "label", "ai_degree", "generator_family", "score"]

def validate_chunk(part, path, data_sha, expected_n, chunk_idx, start, end):
    missing = set(CHUNK_COLS).difference(part.columns)
    ok = (not missing and set(part["_run_signature"].unique()) == {RUN_SIGNATURE}
          and set(part["_data_sha256"].unique()) == {data_sha}
          and set(part["_expected_n"].unique()) == {expected_n}
          and set(part["_chunk_idx"].unique()) == {chunk_idx}
          and part["_row_id"].tolist() == list(range(start, end))
          and np.isfinite(part["score"].to_numpy()).all())
    if not ok:
        raise RuntimeError(f"Checkpoint cu/hong: {path}. Doi CONFIG thi phai xoa folder "
                           f"{CHUNK_DIR} truoc khi chay lai.")

def score_split(split_name, fp_mode=False):
    data_path = DATA_DIR / f"dataset_{split_name}_{DATASET_TAG}.parquet"
    df = maybe_smoke(load(split_name, fp_mode=fp_mode))
    if df is None or len(df) == 0:
        return
    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    if SMOKE_TEST:
        t0 = time.time()
        s = score(df["text"].tolist())
        log(f"[SMOKE] {split_name}: {len(df)} dong, {time.time()-t0:.1f}s, "
            f"{len(df)/max(time.time()-t0,1e-9):.2f} samples/s, finite={np.isfinite(s).all()}, "
            f"score[min/max]={s.min():.4f}/{s.max():.4f}; KHONG luu checkpoint.")
        return

    data_sha = sha256_file(data_path)
    expected_n = len(df)
    n_chunks = math.ceil(expected_n / CHECKPOINT_ROWS)
    log(f"{split_name}: {expected_n} dong, {n_chunks} chunks, sha256={data_sha[:12]}...")
    split_t0 = time.time(); newly = 0
    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHECKPOINT_ROWS
        end = min(start + CHECKPOINT_ROWS, expected_n)
        out_path = CHUNK_DIR / f"{split_name}_chunk{chunk_idx:05d}.parquet"
        if out_path.exists():
            validate_chunk(pd.read_parquet(out_path), out_path, data_sha, expected_n,
                           chunk_idx, start, end)
            log(f"resume {split_name} chunk {chunk_idx+1}/{n_chunks}: da co")
            continue
        part = df.iloc[start:end].copy()
        t0 = time.time()
        part["score"] = score(part["text"].tolist())
        part["_run_signature"] = RUN_SIGNATURE
        part["_data_sha256"] = data_sha
        part["_expected_n"] = expected_n
        part["_chunk_idx"] = chunk_idx
        tmp = out_path.with_suffix(".tmp.parquet")
        part[CHUNK_COLS].to_parquet(tmp, index=False)
        os.replace(tmp, out_path)   # atomic: session ngat khong de lai checkpoint nua voi
        newly += len(part)
        rate = len(part) / max(time.time() - t0, 1e-9)
        log(f"saved {split_name} chunk {chunk_idx+1}/{n_chunks}: {rate:.2f} samples/s, "
            f"ETA con lai ~{(expected_n-end)/max(rate,1e-9)/3600:.2f}h")
    log(f"XONG {split_name}: moi cham {newly} dong, wall={time.time()-split_t0:.1f}s")

for _split, _fp in [("val", False), ("test_ood_model", False), ("test_fp", True)]:
    score_split(_split, fp_mode=_fp)

for _d in (0, 1):
    log(f"peak VRAM cuda:{_d} = {torch.cuda.max_memory_allocated(_d)/1024**3:.2f} GiB")
log("XONG SMOKE worker." if SMOKE_TEST else "XONG FULL worker.")

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
