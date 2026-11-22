"""Sinh bang LaTeX "ba nguon nguong cho thong ke MEAN, 16/16 checkpoint" tu CSV goc.

    python3 make_table_ktransfer.py

Sinh truc tiep tu output/mean_ktransfer_16models.csv (paper_level_mean_ktransfer.py) nen
khong the lech voi so trong §IV-C. Ghi ra paper/table_ktransfer.tex de \\input vao bai.
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
SRC = BASE / "output" / "mean_ktransfer_16models.csv"
DST = BASE / "paper" / "table_ktransfer.tex"

NICE = {
    "roberta_large": "RoBERTa-large",
    "xlm_roberta_large": "XLM-R-large",
    "deberta_v3_large": "DeBERTa-v3-large",
    "modernbert_large": "ModernBERT-large",
    "greater_roberta_large": "GREATER + RoBERTa-large",
    "greater_xlm_roberta_large": "GREATER + XLM-R-large",
    "radar": "RADAR",
    "radar_xlm": "RADAR + XLM-R-large",
}


def label(key):
    stem, seed = key.rsplit("_seed", 1)
    return "%s (s%s)" % (NICE[stem], seed)


PAPER = BASE / "paper" / "rivf2026.tex"
BEGIN = "% >>> TABLE_KTRANSFER_BEGIN -- sinh boi make_table_ktransfer.py, DUNG sua tay"
END = "% <<< TABLE_KTRANSFER_END"


def main():
    d = pd.read_csv(SRC).sort_values(["fpr_tau_chunk", "miss_tau_chunk"]).reset_index(drop=True)
    assert len(d) == 16, len(d)
    build_full(d)                 # ban table* 2 cot, chi luu file du phong
    inject(build_compact(d))      # BAN GON 1 cot la ban dang dung trong bai


def inject(block):
    """Nhung bang thang vao rivf2026.tex giua hai moc, de bai chi con MOT file .tex
    ma van chay lai script duoc. Lan dau thi thay cho dong \\input{table_ktransfer}."""
    s = PAPER.read_text(encoding="utf-8")
    new = BEGIN + "\n" + block + "\n" + END
    if BEGIN in s and END in s:
        i, j = s.index(BEGIN), s.index(END) + len(END)
        s = s[:i] + new + s[j:]
        how = "cap nhat giua hai moc"
    elif "\\input{table_ktransfer}" in s:
        s = s.replace("\\input{table_ktransfer}", new, 1)
        how = "thay cho \\input, dat moc lan dau"
    else:
        print("!! khong tim thay moc lan dong \\input -- BO QUA rivf2026.tex")
        return
    PAPER.write_text(s, encoding="utf-8")
    print("-> %s  (%s)" % (PAPER, how))


def build_full(d):
    L = []
    A = L.append
    A(r"\begin{table*}[t]")
    A(r"\caption{Three threshold sources for the mean statistic, all 16 checkpoints ---")
    A(r"the per-checkpoint breakdown behind \sect{IV-C}. Evaluated on")
    A(r"\texttt{test\_ood\_model} (637 papers per class, $k\approx2.38$). Subscripts give the")
    A(r"length at which $\tau$ was pinned: $1$ is the segment-level threshold from")
    A(r"\texttt{val}, $22.6$ is \texttt{test\_fp\_full}, and $2.38$ is the control, $\tau$")
    A(r"fitted on one half of \texttt{test\_ood\_model} and measured on the other over 200")
    A(r"repetitions. The first two are scored on all 637 pairs.}")
    A(r"\label{tab:ktransfer}")
    A(r"% SOURCE: make_table_ktransfer.py <- output/mean_ktransfer_16models.csv")
    A(r"%          (paper_level_mean_ktransfer.py) -- KHONG go tay, chay lai script de cap nhat")
    A(r"\centering")
    A(r"\footnotesize")
    A(r"\setlength{\tabcolsep}{4.5pt}")
    A(r"\begin{tabular}{@{}lrrrrrrrrr@{}}")
    A(r"\toprule")
    A(r"& \multicolumn{3}{c}{Threshold $\tau$} & \multicolumn{2}{c}{Pinned at $k{\approx}1$}"
      r" & \multicolumn{2}{c}{Pinned at $k{\approx}22.6$} & \multicolumn{2}{c}{Correct $k{\approx}2.38$} \\")
    A(r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}")
    A(r"Model & $\tau_{1}$ & $\tau_{22.6}$ & $\tau_{2.38}$ & FPR & Miss & FPR & Miss & FPR & Miss \\")
    A(r"\midrule")

    for r in d.itertuples():
        A("%s & %.4f & %.4f & %.4f & %.2f\\%% & %.2f\\%% & %.2f\\%% & %.2f\\%% & %.2f\\%% & %.2f\\%% \\\\"
          % (label(r.model), r.tau_chunk, r.tau_mean_k, r.tau_ood,
             100 * r.fpr_tau_chunk, 100 * r.miss_tau_chunk,
             100 * r.fpr_tau_mean_k, 100 * r.miss_tau_mean_k,
             100 * r.fpr_tau_ood, 100 * r.miss_tau_ood))

    A(r"\midrule")
    med = d[["fpr_tau_chunk", "miss_tau_chunk", "fpr_tau_mean_k",
             "miss_tau_mean_k", "fpr_tau_ood", "miss_tau_ood"]].median() * 100
    A(r"\textbf{Median (16/16)} & --- & --- & --- & "
      + " & ".join(r"\textbf{%.2f\%%}" % v for v in med) + r" \\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table*}")

    body = "\n".join(L)
    DST.write_text(body + "\n", encoding="utf-8")
    print("-> %s  (table*, 2 cot, ~0.45 trang)" % DST)
    print("   Median: " + "  ".join("%.2f" % v for v in med)
          + "   (doi chieu §IV-C: 0.24 / 17.82 / 5.10 / 1.10 / 1.31 / 6.16)")
    return body


def build_compact(d):
    """Ban gon: bo 3 cot nguong tau (gia tri hieu chuan, khong mang lap luan), giu
    nguyen 16 checkpoint va ca 6 cot FPR/Miss. Vua MOT cot IEEE."""
    L = []
    A = L.append
    A(r"\begin{table}[t]")
    A(r"\caption{Cost of pinning the threshold at the wrong length. Only $k{\approx}2.38$")
    A(r"matches the split it is scored on. Percentages.}")
    A(r"\label{tab:ktransfer}")
    A(r"% SOURCE: make_table_ktransfer.py <- output/mean_ktransfer_16models.csv")
    A(r"\centering")
    A(r"\scriptsize")
    A(r"\setlength{\tabcolsep}{2.5pt}")   # khop ban camera-ready (chinh tren Overleaf cho vua cot)
    A(r"\begin{tabular}{@{}lrrrrrr@{}}")
    A(r"\toprule")
    A(r"& \multicolumn{2}{c}{$k{\approx}1$} & \multicolumn{2}{c}{$k{\approx}22.6$}"
      r" & \multicolumn{2}{c}{$k{\approx}2.38$} \\")
    A(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    A(r"Model & FPR & Miss & FPR & Miss & FPR & Miss \\")
    A(r"\midrule")
    for r in d.itertuples():
        A("%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\"
          % (label(r.model), 100 * r.fpr_tau_chunk, 100 * r.miss_tau_chunk,
             100 * r.fpr_tau_mean_k, 100 * r.miss_tau_mean_k,
             100 * r.fpr_tau_ood, 100 * r.miss_tau_ood))
    A(r"\midrule")
    med = d[["fpr_tau_chunk", "miss_tau_chunk", "fpr_tau_mean_k",
             "miss_tau_mean_k", "fpr_tau_ood", "miss_tau_ood"]].median() * 100
    A(r"\textbf{Median} & " + " & ".join(r"\textbf{%.2f}" % v for v in med) + r" \\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    body = "\n".join(L)
    p = DST.with_name("table_ktransfer_compact.tex")
    p.write_text(body + "\n", encoding="utf-8")
    print("-> %s  (table, 1 cot, ~0.25 trang)" % p)
    return body


if __name__ == "__main__":
    main()
