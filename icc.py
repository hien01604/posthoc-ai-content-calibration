"""Uoc luong tuong quan noi tai lieu tren thang THO -- mot ban duy nhat dung chung.

LICH SU (quan trong, dung xoa):
  Truoc 2026-08-29 moi script tu cai lai cong thuc
        rho = 1 - MSW / Var_pool
  trong do MSW tinh tren K0 doan DAU moi bai, con Var_pool tinh tren TOAN BO doan.
  Tu so va mau so khac population => rho bi dim xuong manh (trung vi 0.031 thay vi
  0.136; co checkpoint cham san 0.0). Con so 0.044 tung in trong bai den tu day.

  Ban vao dung ICC(1) mot chieu chuan, cung cong thuc `icc_kish` da co san o
  paper_level_rho_compare.py:62 -- nho vay so tren thang tho va thang z moi so sanh
  duoc voi nhau, dung nhu §V-B tuyen bo.
"""
import numpy as np

K0_DEFAULT = 20


def icc1(papers, k0=K0_DEFAULT):
    """ICC(1) mot chieu, nhom bang nhau k0 doan/bai.

        ICC = (MSB - MSW) / (MSB + (k0-1) * MSW)

    `papers`: list mang 1-D diem so, moi mang la mot tai lieu. Bai ngan hon k0 bi bo.
    """
    G = np.asarray([np.asarray(p, dtype=float)[:k0]
                    for p in papers if len(p) >= k0], dtype=float)
    if G.ndim != 2 or G.shape[0] < 2:
        return 0.0
    n, k = G.shape
    msw = float(np.mean(G.var(axis=1, ddof=1)))
    msb = float(k * np.sum((G.mean(axis=1) - G.mean()) ** 2) / (n - 1))
    den = msb + (k - 1) * msw
    if not np.isfinite(den) or den <= 0:
        return 0.0
    return float(max(0.0, (msb - msw) / den))


def icc1_and_var(papers, k0=K0_DEFAULT):
    """Nhu icc1() nhung tra kem phuong sai tong (mot so script cu can ca hai)."""
    keep = [np.asarray(p, dtype=float) for p in papers if len(p) >= k0]
    var_total = float(np.concatenate(keep).var(ddof=1)) if keep else 0.0
    return icc1(papers, k0), var_total


def icc1_legacy_buggy(papers, k0=K0_DEFAULT):
    """CHI de tai hien con so cu 0.044 khi can doi chieu. KHONG dung de bao cao."""
    keep = [np.asarray(p, dtype=float) for p in papers if len(p) >= k0]
    if not keep:
        return 0.0
    within = float(np.mean([p[:k0].var(ddof=1) for p in keep]))
    total = float(np.concatenate(keep).var(ddof=1))
    return float(max(0.0, 1.0 - within / total))
