"""Phần toán của Lastde/Lastde++ — port từ py_scripts/baselines/scoring_methods/fastMDE.py
   (https://github.com/TrustMedia-zju/Lastde_Detector).

Khác biệt duy nhất so với code gốc: thay `torch.vmap(histcounts)` bằng histogram vectorised
(`scatter_add_`), vì `torch.histc` không có batching rule nên `vmap` rơi về vòng lặp tuần tự.
Ngữ nghĩa được giữ y nguyên và cell VERIFY của notebook chứng minh max|Δ| = 0.
"""
import math

import torch


def histogram_probs(x, eps_bins):
    """Tương đương `torch.histc(x[..., s], bins=eps_bins, min=-1, max=1)` cho từng lát của
    chiều cuối, rồi chuẩn hoá về tổng 1.

    x: [..., S] (float)  ->  [eps_bins, S]

    Bám đúng 2 quy ước của torch.histc:
      * phần tử ngoài [min, max] bị **bỏ** (không dồn vào bin biên),
      * bin = min(floor((x - min) / (max - min) * bins), bins - 1)  -> x == max vào bin cuối.
    """
    S = x.shape[-1]
    flat = x.reshape(-1, S).float()
    keep = (flat >= -1.0) & (flat <= 1.0)
    idx = torch.floor((flat + 1.0) * (eps_bins / 2.0)).long().clamp_(0, eps_bins - 1)
    col = torch.arange(S, device=x.device).expand_as(idx)
    lin = (idx * S + col)[keep]
    hist = torch.zeros(eps_bins * S, device=x.device, dtype=torch.float32)
    hist.scatter_add_(0, lin, torch.ones_like(lin, dtype=torch.float32))
    hist = hist.view(eps_bins, S)
    return hist / hist.sum(dim=0, keepdim=True)


def de_from_probs(probs, eps_bins):
    """DE(s, ε, τ) = -(1/ln ε) Σ_i P_i ln P_i,  chỉ tính trên P_i > 0.   (Eq. 8)

    Giữ NGUYÊN VĂN biểu thức của code gốc, kể cả các chi tiết ảnh hưởng bit cuối:
      * `torch.nansum` để bỏ bin rỗng (0 * log 0 = nan), không dùng `where`;
      * `-1 / torch.log(torch.tensor(eps_bins))` rồi NHÂN — không dùng `math.log` (float64)
        rồi chia, vì lệch ~3e-08 và cell VERIFY sẽ báo không bit-identical.
    """
    return -1 / torch.log(torch.tensor(eps_bins)) * torch.nansum(
        probs * torch.log(probs), dim=0)


def calculate_DE(seq, embed_size, eps_bins):
    """seq: [1, T, S] -> DE cho từng sample: [S].   (Eq. 3-6, 8)"""
    orbits = seq.unfold(1, embed_size, 1)                       # [1, T-s+1, S, s]
    cos = torch.nn.functional.cosine_similarity(
        orbits[:, :-1], orbits[:, 1:], dim=-1)                  # [1, T-s, S]
    return de_from_probs(histogram_probs(cos, eps_bins), eps_bins)


def agg_mde(loglik, embed_size, eps_bins, tau_prime, agg="std"):
    """Agg(MDE) — mẫu số của Eq. 9.  loglik: [1, n, S] -> [S].

    `torch.std` mặc định unbiased=True (ddof=1), khớp code gốc. KHÔNG thay bằng np.std (ddof=0).
    """
    des = []
    for tau in range(1, tau_prime + 1):
        scaled = loglik.unfold(1, tau, 1).mean(dim=3)            # Eq. 2: [1, n-tau+1, S]
        des.append(calculate_DE(scaled, embed_size, eps_bins))
    mde = torch.stack(des, dim=0)                                # [tau_prime, S]
    out = torch.std(mde, dim=0)                                  # ddof=1
    return torch.exp(out) if agg == "expstd" else out


def lastde(loglik, embed_size, eps_mult, tau_prime, agg="std"):
    """Lastde(t, θ) = mean(log p) / Agg(MDE).   (Eq. 9)   loglik: [1, n, S] -> [1, S].

    ε = eps_mult * n với n = số token, đúng `int(args.epsilon * log_likelihood.shape[1])`.
    """
    n = loglik.shape[1]
    eps_bins = int(eps_mult * n)
    return loglik.mean(dim=1) / agg_mde(loglik, embed_size, eps_bins, tau_prime, agg)


def min_tokens_required(embed_size, tau_prime):
    """Cần ít nhất 2 segment liền kề ở scale lớn nhất: n - τ' - s + 2 >= 2."""
    return tau_prime + embed_size
