"""Sinh bang "khoang tin cay chuyen sang tai lieu co AI" tu CSV goc.

    python3 make_table_mixed.py

Thay cho doan van 153 chu o §IV-B von dang ta bang loi mot luoi 5 pi x 5 loai khoang.
Sinh truc tiep tu f6_coverage_mixed_emp.csv nen khong the lech voi bai.
Nhung ai chen thang vao rivf2026.tex giua hai moc, giong make_table_ktransfer.py.
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
SRC = BASE / "final_experiments_rivf2026" / "results" / "f6_coverage_mixed_emp.csv"
PAPER = BASE / "paper" / "rivf2026.tex"
DST = BASE / "paper" / "table_mixed.tex"

BEGIN = "% >>> TABLE_MIXED_BEGIN -- sinh boi make_table_mixed.py, DUNG sua tay"
END = "% <<< TABLE_MIXED_END"


def build(k=20):
    d = pd.read_csv(SRC)
    pm = d.groupby(["model", "k", "true_pi"])[
        ["cov_emp_null", "cov_emp_pi", "cov_emp_plugin", "cov_deff", "cov_naive",
         "hw_emp_plugin"]].mean().reset_index()
    m = pm[pm.k == k].groupby("true_pi").median(numeric_only=True)

    L = []
    A = L.append
    A(r"\begin{table}[t]")
    A(r"\caption{Interval coverage (\%) on documents containing AI: $k{=}20$ human")
    A(r"segments from one real paper, $\pi k$ of them replaced by AI scores.}")
    A(r"\label{tab:mixed}")
    A(r"% SOURCE: make_table_mixed.py <- f6_coverage_mixed_emp.csv (f6_mixed_pi_empirical_ci.py)")
    A(r"\centering")
    A(r"\scriptsize")
    A(r"\setlength{\tabcolsep}{3.2pt}")
    A(r"\begin{tabular}{@{}rccccc@{}}")
    A(r"\toprule")
    A(r"& \multicolumn{3}{c}{Empirical, calibrated at} &"
      r" \multicolumn{2}{c}{Gaussian} \\")
    A(r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}")
    A(r"$\pi$ & $\pi{=}0$ & $\hat{\pi}$ (plug-in) & true $\pi$ &"
      r" $+$deff & plain \\")
    A(r"\midrule")
    for pi in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = m.loc[pi]
        A("%.2f & %.1f & %.1f & %.1f & %.1f & %.1f \\\\"
          % (pi, 100 * r.cov_emp_null, 100 * r.cov_emp_plugin, 100 * r.cov_emp_pi,
             100 * r.cov_deff, 100 * r.cov_naive))
    A(r"\midrule")
    hw = m.hw_emp_plugin
    A(r"\multicolumn{6}{@{}l@{}}{\footnotesize Plug-in half-width narrows "
      + r"$%.3f\rightarrow%.3f$ as $\hat{\pi}$ goes $0\rightarrow1$.}\\" % (hw.iloc[0], hw.iloc[-1]))
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    return "\n".join(L), m


def main():
    body, m = build()
    DST.write_text(body + "\n", encoding="utf-8")
    print("-> %s" % DST)

    s = PAPER.read_text(encoding="utf-8")
    new = BEGIN + "\n" + body + "\n" + END
    if BEGIN in s and END in s:
        i, j = s.index(BEGIN), s.index(END) + len(END)
        s = s[:i] + new + s[j:]
        PAPER.write_text(s, encoding="utf-8")
        print("-> %s (cap nhat giua hai moc)" % PAPER)
    else:
        print("!! chua co moc trong rivf2026.tex -- chen tay khoi duoi mot lan:")
        print("   %s ... %s" % (BEGIN, END))

    print("\nSo trong bang (doi chieu doan van cu):")
    print((m[["cov_emp_null", "cov_emp_plugin", "cov_emp_pi",
              "cov_deff", "cov_naive"]] * 100).round(1).to_string())


if __name__ == "__main__":
    main()
