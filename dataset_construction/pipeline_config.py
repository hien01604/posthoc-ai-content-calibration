"""Cấu hình dùng chung cho 02b / 02c / 03 / 04 — MỘT chỗ duy nhất để bật/tắt.

Trước đây mỗi notebook có biến USE_DECONFOUNDED riêng, và chúng đã LỆCH nhau thật
(03=True mà 04=False). Lệch kiểu đó không báo lỗi: bạn đo shortcut trên dữ liệu này
rồi melt ra bảng train từ dữ liệu kia. Gom vào đây để không xảy ra nữa.

    import pipeline_config as cfg
    cfg.DATASET_VARIANT       # "normalized" | "deconf" | "deconf2"
    cfg.DATASET_TAG           # hậu tố file, trùng DATASET_VARIANT
    cfg.pair_path(OUTPUT_DIR, "train")   -> .../dataset_train_deconf2.parquet
    cfg.long_path(OUTPUT_DIR, "train")   -> .../dataset_train_final_long_deconf2.parquet

Đổi MỘT dòng dưới đây là cả 03 và 04 đổi theo.
"""

# ── Biến thể dataset ────────────────────────────────────────────────────
#   "normalized"  bản gốc, CÒN shortcut. process 0,7568 · tổng 0,9343 · G6 FAIL.
#                 word_count một mình đã đoán đúng human/AI ở AUROC 0,7201.
#   "deconf"      output 02b_deconfound (canonicalize + truncate_pair).
#                 process 0,7267 · tổng 0,9114. 02b tự chấm 0,6867 "PASS" nhưng đó
#                 là do nó chỉ đo 6 đặc trưng, không có avg_word_len lẫn
#                 type_token_ratio. Đo đủ 22 đặc trưng thì KHÔNG pass:
#                   - word_count 0,5000 và ends_terminal 0,5236 khi đứng riêng,
#                     nhưng GHÉP HAI CÁI = 0,7264 (cắt thô làm ~nửa văn bản kết thúc
#                     giữa câu, và bên nào bị cắt lại đoán được từ độ dài chung)
#   "deconf2"     output 02c_deconfound_v2 (canonicalize + truncate_pair_sentences
#                 + lọc notation_gap 85%). process 0,5055 · tổng 0,8909.
#                 <- MẶC ĐỊNH. Đây là bản dùng để train.
#
# Cả ba bản cùng tồn tại trên đĩa, không bản nào bị ghi đè, nên chạy được cả ba để
# so sánh trong KLTN (bảng ablation ở docs/SHORTCUT_DEFENSE.md).
DATASET_VARIANT = "deconf2"

_VALID_VARIANTS = ("normalized", "deconf", "deconf2")
assert DATASET_VARIANT in _VALID_VARIANTS, (
    f"DATASET_VARIANT={DATASET_VARIANT!r} không hợp lệ, phải là một trong {_VALID_VARIANTS}"
)

DATASET_TAG = DATASET_VARIANT
LONG_SUFFIX = "_final_long" if DATASET_VARIANT == "normalized" else f"_final_long_{DATASET_TAG}"

# Giữ tên cũ cho code đã viết trước 2026-08-12. "deconf2" cũng là đã khử confound nên
# cờ này True với cả hai bản deconf — đừng dùng nó để phân biệt deconf vs deconf2.
USE_DECONFOUNDED = DATASET_VARIANT != "normalized"

SPLITS = ["train", "val", "test_ood_model", "test_fp"]


def pair_path(output_dir, split):
    """Đường dẫn dataset dạng CẶP (human_text + ai_text trên cùng dòng).

    Với "normalized", val/test_ood_model ưu tiên bản *_perturbed (đã qua pipeline
    attack) nếu file tồn tại — giữ đúng hành vi cũ của 04. Với các bản deconf thì
    luôn dùng file của biến thể: 02b/02c đã áp biến đổi cho cả cột attack.
    """
    if DATASET_VARIANT == "normalized":
        if split in {"val", "test_ood_model"}:
            perturbed = output_dir / f"dataset_{split}_perturbed.parquet"
            if perturbed.exists():
                return perturbed
        return output_dir / f"dataset_{split}_normalized.parquet"
    return output_dir / f"dataset_{split}_{DATASET_TAG}.parquet"


def long_path(output_dir, split):
    """Đường dẫn bảng LONG (1 dòng = 1 mẫu) mà notebook train đọc."""
    return output_dir / f"dataset_{split}{LONG_SUFFIX}.parquet"


def matched_path(output_dir, split):
    """Tập eval matched (make_matched_eval.py). None nếu chưa dựng."""
    p = output_dir / f"matched_eval_{split}_{DATASET_TAG}.parquet"
    return p if p.exists() else None


def banner():
    """In ra biến thể đang dùng — gọi ở đầu mọi notebook để không chạy nhầm."""
    desc = {
        "normalized": "bản gốc, CÒN shortcut (process 0,7568)",
        "deconf": "02b: canonicalize + truncate_pair (process 0,7267)",
        "deconf2": "02c: canonicalize + truncate_pair_sentences + lọc notation (process 0,5055)",
    }[DATASET_VARIANT]
    return f"DATASET VARIANT : {DATASET_VARIANT} — {desc}"
