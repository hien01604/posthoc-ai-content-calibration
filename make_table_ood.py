"""Sinh bang §V-C: chuyen rho_z ra ngoai mau + so sanh power ghep cap.

    python3 make_table_ood.py

Thay cho ~200 chu dang ta bang loi mot bang 4 luat x (FPR, miss, so lan thang).
Moi so sinh tu CSV goc, khong go tay. Chen thang vao rivf2026.tex giua hai moc.
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
REV = BASE / "revision_experiments_rivf2026" / "results"
PAPER = BASE / "paper" / "rivf2026.tex"
DST = BASE / "paper" / "table_ood.tex"

BEGIN = "% >>> TABLE_OOD_BEGIN -- sinh boi make_table_ood.py, DUNG sua tay"
END = "% <<< TABLE_OOD_END"


def main():
    h = pd.read_csv(OUT / "hartung_ood_16models.csv")
    m = pd.read_csv(OUT / "mean_ktransfer_16models.csv")
    r3 = pd.read_csv(REV / "rev3_paired_miss.csv")
    r4 = pd.read_csv(REV / "rev4_fpr_matched.csv")

    def rng(x):
        return 100 * x.median(), 100 * x.min(), 100 * x.max()

    z_f, z_fl, z_fh = rng(h.fpr_hartung)
    z_m, z_ml, z_mh = rng(h.miss_hartung)
    c_f, c_fl, c_fh = rng(m.fpr_tau_ood)
    c_m, c_ml, c_mh = rng(m.miss_tau_ood)

    L = []
    A = L.append
    A(r"\begin{table}[t]")
    A(r"\caption{$\rho_z$ from the full-length corpus, reused unchanged on the")
    A(r"held-out-generator split ($k{\approx}2.38$); median [range] over 16 checkpoints.")
    A(r"Rows 2--4 threshold the raw mean: fitted at this length on calibration halves")
    A(r"(Table~\ref{tab:ktransfer}), then scored on $Z_k$'s 637 pairs, then re-fitted to")
    A(r"$Z_k$'s FPR. Wins: checkpoints where $Z_k$ misses fewer.}")
    A(r"\label{tab:ood}")
    A(r"% SOURCE: make_table_ood.py <- hartung_ood_16models.csv, mean_ktransfer_16models.csv,")
    A(r"%          rev3_paired_miss.csv, rev4_fpr_matched.csv")
    A(r"\centering")
    A(r"\scriptsize")
    A(r"\setlength{\tabcolsep}{3pt}")
    A(r"\begin{tabular}{@{}lccc@{}}")
    A(r"\toprule")
    A(r"Rule & FPR (\%) & Miss (\%) & Wins \\")
    A(r"\midrule")
    A(r"1. $Z_k$ (proposed) & %.2f\,[%.2f,%.2f] & %.2f\,[%.2f,%.2f] & --- \\"
      % (z_f, z_fl, z_fh, z_m, z_ml, z_mh))
    A(r"\midrule")
    A(r"2. Raw mean, fitted here & %.2f\,[%.2f,%.2f] & %.2f\,[%.2f,%.2f] & --- \\"
      % (c_f, c_fl, c_fh, c_m, c_ml, c_mh))
    A(r"3. Same, on $Z_k$'s pairs & %.2f & %.2f & %d/16 \\"
      % (100 * r3.fpr_ctrl.median(), 100 * r3.miss_ctrl.median(),
         (r3.miss_z < r3.miss_ctrl).sum()))
    A(r"4. Raw mean, FPR of $Z_k$ & \emph{matched} & %.2f & %d/16 \\"
      % (100 * r4.miss_ctrl.median(), (r4.delta_miss > 0).sum()))
    A(r"\bottomrule")
    A(r"\end{tabular}")
    nsig = ((r4.p_mcnemar < 0.05) & (r4.delta_miss > 0)).sum()
    nagainst = ((r4.p_mcnemar < 0.05) & (r4.delta_miss < 0)).sum()
    A(r"\\[2pt]\footnotesize In row 4, %d checkpoint--AI-document pairs are flagged only"
      % r4.b_only_z.sum())
    A(r"by $Z_k$ and %d only by the raw mean; unadjusted McNemar tests ($5\%%$) favour"
      % r4.c_only_ctrl.sum())
    A(r"$Z_k$ for %d checkpoints and the raw mean for %d." % (nsig, nagainst))
    A(r"\end{table}")
    body = "\n".join(L)

    DST.write_text(body + "\n", encoding="utf-8")
    print("-> %s" % DST)

    s = PAPER.read_text(encoding="utf-8")
    new = BEGIN + "\n" + body + "\n" + END
    if BEGIN in s and END in s:
        i, j = s.index(BEGIN), s.index(END) + len(END)
        PAPER.write_text(s[:i] + new + s[j:], encoding="utf-8")
        print("-> %s (cap nhat giua hai moc)" % PAPER)
    else:
        print("!! chua co moc trong rivf2026.tex; chen tay mot lan:")
        print("   %s\n   %s" % (BEGIN, END))


if __name__ == "__main__":
    main()
