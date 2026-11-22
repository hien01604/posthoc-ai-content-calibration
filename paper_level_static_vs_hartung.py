"""So một tau TĨNH (chốt ở k=KMAX) với tau(k) Hartung, tách theo TỪNG k — cho nhánh mean.

    python3 paper_level_static_vs_hartung.py

Trả lời câu hỏi: Table V (hiệu chuẩn trực tiếp mức tài liệu, tau tĩnh, population k
trải rộng 19..262) kiểm soát đúng FPR TRUNG BÌNH — nhưng có kiểm soát đúng ở TỪNG k
riêng lẻ không? Dùng lại đúng population có kiểm soát của paper_level_tau_k.py (576
bài >=20 đoạn, chỉ đổi k bằng subsampling, không đổi population) để tách câu hỏi này
khỏi nhiễu "bài khác nhau ở k khác nhau" — cùng nguyên tắc đã dùng cho Fig. 1 (§V.1).

ĐẦU VÀO
-------
  output/tau_k_mean_16models.csv  (từ paper_level_tau_k.py — chạy trước nếu chưa có)
      cột fpr_const       = FPR khi dùng tau TĨNH (chốt ở k=KMAX=20), đo tại từng k.
      cột fpr_law_nocorr  = FPR khi dùng tau(k) = z_(1-a) * sqrt(1+(k-1)*rho) — công thức
                            Hartung THÔ, KHÔNG vá hệ số c (quyết định: bỏ c cho gọn câu
                            chuyện, chỉ cần 1 tham số ước lượng là rho — xem outline §VI.2).

ĐẦU RA (output/)
----------------
  static_vs_hartung_by_k.csv   FPR (%) theo k, trung vị + [thấp-cao] qua 16 checkpoint,
                                cho cả tau tĩnh và tau(k) Hartung (không hệ số c) — nguồn
                                cho Table IV trong outline_rivf2026_v2.md §VI.2
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
KS_SHOW = [1, 2, 3, 5, 8, 10, 15, 20]


def main():
    T = pd.read_csv(OUT / "tau_k_mean_16models.csv")

    # Trung bình qua 5 lần chia cal/eval trước (khử nhiễu chia tập), còn 1 số/(model,k).
    per_model_k = T.groupby(["model", "k"]).agg(
        fpr_const=("fpr_const", "mean"),
        fpr_hartung=("fpr_law_nocorr", "mean"),
        fpr_emp=("fpr_emp", "mean"),
        rho=("rho", "mean"),
    ).reset_index()

    rows = []
    for k in KS_SHOW:
        sub = per_model_k[per_model_k.k == k]
        row = {"k": k, "n_checkpoints": len(sub)}
        for col in ("fpr_const", "fpr_hartung", "fpr_emp"):
            v = sub[col].to_numpy() * 100
            row[f"{col}_median_pct"] = np.median(v)
            row[f"{col}_min_pct"] = v.min()
            row[f"{col}_max_pct"] = v.max()
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "static_vs_hartung_by_k.csv", index=False)

    u = per_model_k.drop_duplicates("model")
    print(f"rho: trung vi {u.rho.median():.3f}  dai [{u.rho.min():.3f}-{u.rho.max():.3f}]\n")
    print("k    tau tinh (chot k=20)      tau(k) Hartung (khong c)   muc tieu")
    for _, r in R.iterrows():
        print(f"{int(r.k):>3}  {r.fpr_const_median_pct:5.2f}% ({r.fpr_const_min_pct:.2f}-{r.fpr_const_max_pct:.2f}%)   "
              f"{r.fpr_hartung_median_pct:5.2f}% ({r.fpr_hartung_min_pct:.2f}-{r.fpr_hartung_max_pct:.2f}%)     1%")
    print("\n-> output/static_vs_hartung_by_k.csv")


if __name__ == "__main__":
    main()
