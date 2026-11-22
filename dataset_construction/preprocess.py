"""Tiền xử lý CHÍNH THỨC của detector — nguồn chân lý duy nhất.

Import module này ở MỌI nơi: 02b_deconfound, các notebook train, và code inference.
Không copy-paste hàm sang chỗ khác — lệch một ký tự là mismatch train/inference.

    from preprocess import canonicalize, truncate_pair

Hai tầng, KHÁC NHAU về phạm vi áp dụng:

  canonicalize(text)          -> áp cho MỘT văn bản. Dùng ở train VÀ inference.
                                 = phiên chuyển non-ASCII + chuẩn hoá dấu câu.
  truncate_pair(human, ai)    -> áp cho MỘT CẶP. CHỈ dùng khi dựng dataset.
                                 Không dùng được ở inference (chỉ có 1 văn bản).

Vì sao tách: `truncate_pair` cần cả hai văn bản của một cặp để lấy min độ dài, nên
nó là bước DỰNG DỮ LIỆU chống confound, không phải bước tiền xử lý. Ở inference chỉ
cần `canonicalize`. Điều đó nhất quán: model được huấn luyện để KHÔNG dùng độ dài,
nên gặp độ dài tự nhiên lúc inference cũng không sao.

Thứ tự bắt buộc khi dựng dataset: canonicalize -> truncate_pair -> (không gì nữa).
Phiên chuyển TRƯỚC khi cắt, để độ dài được cân bằng SAU khi ký hiệu đã thành chữ.

── BỔ SUNG 2026-08-12 (dựng dataset v2, xem 02c_deconfound_v2.ipynb) ─────────
  truncate_pair_sentences(h, a)  -> thay cho truncate_pair. Cân bằng số từ NHƯNG
                                    dừng ở BIÊN CÂU.
  notation_profile(text)         -> đo mật độ ký hiệu/số/tên riêng của MỘT văn bản.
  notation_gap(h_prof, a_prof)   -> mức "AI rewrite bỏ mất ký hiệu của bài gốc".

VÌ SAO CẦN (đã đo, xem docs/SHORTCUT_DEFENSE.md):

1. `truncate_pair` cắt THÔ theo từ nên làm tỉ lệ văn bản kết thúc đúng dấu câu tụt
   99,0% -> 52,4%. Nó cân bằng ở CẢ HAI lớp (AUC lẻ 0,524) nên không thành shortcut,
   nhưng ~nửa tập train kết thúc giữa câu, trong khi lúc inference model gặp văn bản
   nguyên vẹn -> lệch phân phối do CHÍNH ta tạo ra. `truncate_pair_sentences` giữ
   word_count cân bằng tương đương mà không cắt giữa câu.

2. Ba kênh artifact chưa từng được xét ở cổng G6 cũ (chỉ 6 feature):
       single_char_frac    human 0,1017 vs AI 0,0653   AUC lẻ 0,624
       titlecase_token_frac human 0,1104 vs AI 0,0913   AUC lẻ 0,620
       numeric_token_frac  human 0,0245 vs AI 0,0180   AUC lẻ 0,575
   Đây KHÔNG phải văn phong AI mà là artifact TRUNG THỰC-NỘI DUNG: human_text là
   đoạn bài báo thật nên còn biến toán (x, k, n), số liệu và tên riêng; bản AI viết
   lại thì lược đi. Đo được: lệch có TRƯỚC canonicalize (0,0912 vs 0,0488) nên không
   phải do bỏ ngoặc, và allcaps_token_frac chỉ 0,551 nên không phải do acronym.
   Không sửa được bằng phẫu thuật văn bản (xoá số = xoá nội dung) -> lọc ở mức CẶP:
   bỏ những cặp mà bản AI lược ký hiệu nhiều nhất. Đây đồng thời là tiêu chí CHẤT
   LƯỢNG sinh dữ liệu, nên bỏ chúng là đúng theo cả hai nghĩa.
"""

import re
import unicodedata

__all__ = [
    "canonicalize", "truncate_pair", "NON_ASCII_TABLE", "non_ascii_coverage",
    "split_sentences", "truncate_pair_sentences", "notation_profile", "notation_gap",
]

# ── Bảng phiên chuyển non-ASCII -> ASCII, GIỮ NGHĨA ──────────────────────
# Nguyên tắc: xoá ký hiệu làm câu vô nghĩa ("δ(q,x,y)" -> "(q,x,y)"), nên phiên
# chuyển thành chữ đọc được ("delta(q,x,y)"). Vừa diệt được non_ascii_ratio (về 0
# cho cả hai lớp) vừa giữ nội dung -> XAI/SHAP đọc được, người phản biện đọc được.

# Bọc KHOẢNG TRẮNG hai bên. Không bọc thì "μcomponents" -> "mucomponents" và
# "μα" -> "mualpha" — đo được 555 chỗ dính Hy Lạp–Latin + 216 chỗ Hy Lạp–Hy Lạp
# trên mẫu 4.000 dòng train. Bọc rồi thì "alpha -th", dài hơn một token nhưng đọc
# được; truncate_pair chạy SAU nên độ dài vẫn cân bằng tuyệt đối.
_GREEK = {
    "α": " alpha ", "β": " beta ", "γ": " gamma ", "δ": " delta ", "ε": " epsilon ",
    "ζ": " zeta ", "η": " eta ", "θ": " theta ", "ϑ": " theta ", "ι": " iota ",
    "κ": " kappa ", "λ": " lambda ", "μ": " mu ", "µ": " mu ", "ν": " nu ", "ξ": " xi ",
    "ο": " o ", "π": " pi ", "ϖ": " pi ", "ρ": " rho ", "ς": " sigma ", "σ": " sigma ",
    "τ": " tau ", "υ": " upsilon ", "φ": " phi ", "ϕ": " phi ", "χ": " chi ",
    "ψ": " psi ", "ω": " omega ",
    "Α": " Alpha ", "Β": " Beta ", "Γ": " Gamma ", "Δ": " Delta ", "∆": " Delta ",
    "Ε": " Epsilon ", "Ζ": " Zeta ", "Η": " Eta ", "Θ": " Theta ", "Ι": " Iota ",
    "Κ": " Kappa ", "Λ": " Lambda ", "Μ": " Mu ", "Ν": " Nu ", "Ξ": " Xi ",
    "Ο": " O ", "Π": " Pi ", "Ρ": " Rho ", "Σ": " Sigma ", "Τ": " Tau ",
    "Υ": " Upsilon ", "Φ": " Phi ", "Χ": " Chi ", "Ψ": " Psi ", "Ω": " Omega ",
}

# Toán tử -> dạng ASCII/LaTeX-ish. Bọc khoảng trắng để không dính vào từ bên cạnh.
_MATH = {
    "∈": " in ", "∉": " notin ", "∋": " ni ", "⊂": " subset ", "⊃": " supset ",
    "⊆": " subseteq ", "⊇": " supseteq ", "∪": " union ", "∩": " intersect ",
    "∀": " forall ", "∃": " exists ", "¬": " not ", "∧": " and ", "∨": " or ",
    "→": " -> ", "←": " <- ", "↔": " <-> ", "↦": " mapsto ", "⇒": " => ",
    "⇐": " <= ", "⇔": " <=> ", "≤": " <= ", "≥": " >= ", "≠": " != ",
    "≈": " approx ", "≃": " approx ", "≅": " cong ", "≡": " equiv ",
    "∞": " infinity ", "∑": " sum ", "∏": " prod ", "∫": " integral ",
    "√": " sqrt ", "∂": " partial ", "∇": " nabla ", "∝": " propto ",
    "×": " x ", "÷": " / ", "·": " . ", "∘": " compose ", "⊕": " oplus ",
    "⊗": " otimes ", "⊙": " odot ", "±": " +- ", "∓": " -+ ",
    "∅": " emptyset ", "⊥": " bot ", "⊤": " top ", "⊢": " vdash ",
    "⊨": " models ", "≺": " prec ", "≻": " succ ", "⌈": " ceil ",
    "⌉": " ceil ", "⌊": " floor ", "⌋": " floor ", "ℓ": "l", "ℝ": "R",
    "ℕ": "N", "ℤ": "Z", "ℚ": "Q", "ℂ": "C", "†": " dagger ", "∥": " || ",
    # Bổ sung sau khi chạy non_ascii_coverage() trên train — 15 ký tự sót nhiều nhất.
    # Không map thì chúng thành khoảng trắng, tức mất nghĩa như bản xoá cứng.
    "∼": " ~ ", "⁄": "/", "⋆": " * ", "§": " section ", "‖": " || ",
    "⊲": " lhd ", "⊳": " rhd ", "↓": " down ", "↑": " up ",
    "⊑": " sqsubseteq ", "⊒": " sqsupseteq ", "⊓": " sqcap ", "⊔": " sqcup ",
    "♯": " sharp ", "♭": " flat ", "≪": " << ", "≫": " >> ",
    "⟨": " < ", "⟩": " > ", "〈": " < ", "〉": " > ",
    "∗": " * ", "∙": " . ", "≫": " >> ", "⇑": " up ", "⇓": " down ",
    # "∆" KHÔNG khai lại ở đây: _GREEK đã có (bản đã bọc space) và _MATH merge
    # SAU nên bản ở đây sẽ ghi đè -> quay lại lỗi dính từ "∆X" -> "DeltaX".
    "∴": " therefore ", "∵": " because ", "∦": " nparallel ",
}

# Typography: dấu nháy/gạch cong -> ASCII. Đây là phần "văn phong AI" mà L2 cũ xoá.
_TYPO = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": "''", "‴": "'''",
    "–": "-", "—": "-", "―": "-", "−": "-", "‐": "-", "‑": "-", "‒": "-",
    # U+2022 BULLET: marker khi bóc PDF. KHÔNG map thành "." — sẽ tạo ranh giới
    # câu GIẢ, làm sai avg_sentence_len (feature TÍN HIỆU THẬT, phải đo đúng).
    "•": " * ", "◦": " * ", "‣": " * ", "▪": " * ", "●": " * ",
    "…": "...", " ": " ", "­": "", "​": "",
    " ": " ", " ": " ", "·": ".",
}

# Chỉ số trên/dưới và phân số -> chữ số thường
_DIGITS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
    "⁷": "7", "⁸": "8", "⁹": "9", "₀": "0", "₁": "1", "₂": "2", "₃": "3",
    "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "½": " 1/2 ", "⅓": " 1/3 ", "¼": " 1/4 ", "¾": " 3/4 ", "⅔": " 2/3 ",
}

NON_ASCII_TABLE = {**_TYPO, **_DIGITS, **_GREEK, **_MATH}

# ":=" / "::=" là toán tử gán trong văn học thuật CS. Phải bảo toàn TRƯỚC khi
# map [;:!?] -> "." — nếu không thì ":=" thành ".=" (đo được 152 chỗ trên train).
_ASSIGN_RE = re.compile(r"::=|:=")
_ASSIGN_HOLD = "\x00ASSIGN\x00"

_SENT_PUNCT_RE = re.compile(r"[;:!?]")
# Dấu nháy thì XOÁ để "don't" -> "dont" (giữ nguyên 1 token).
_DROP_QUOTE_RE = re.compile(r"['\u2019]")
# Ngoặc/nháy kép thì thay bằng KHOẢNG TRẮNG, không xoá: xoá sẽ dính token
# ("Th(E)" -> "ThE"). Thay bằng space thì "Th E", đọc được.
_SPACE_PUNCT_RE = re.compile(r"[\"()\[\]{}]")
_WS_RE = re.compile(r"\s+")


def _transliterate(text):
    """non-ASCII -> ASCII, giữ nghĩa. Ba tầng, tầng sau bắt phần tầng trước bỏ sót."""
    text = "".join(NON_ASCII_TABLE.get(c, c) for c in text)
    # é -> e, ü -> u : tách dấu tổ hợp rồi bỏ (an toàn cho tiếng Âu, KHÔNG cho CJK)
    text = "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))
    # sót lại (CJK, emoji, ký hiệu lạ) -> khoảng trắng, không xoá trắng để
    # không dính hai từ vào nhau
    return "".join(c if ord(c) < 128 else " " for c in text)


def canonicalize(text):
    """Tiền xử lý CHÍNH THỨC cho một văn bản. Dùng ở train VÀ inference.

    1. phiên chuyển non-ASCII -> ASCII (giữ nghĩa)
    2. [;:!?] -> "."           (chuẩn hoá ranh giới câu)
    3. xoá dấu nháy ' (don't -> dont); [ " ( ) [ ] { } ] -> khoảng trắng
    4. gộp whitespace
    """
    if text is None:
        return None

    text = _transliterate(str(text))
    text = _ASSIGN_RE.sub(_ASSIGN_HOLD, text)
    text = _SENT_PUNCT_RE.sub(".", text)
    text = _DROP_QUOTE_RE.sub("", text)
    text = _SPACE_PUNCT_RE.sub(" ", text)
    text = text.replace(_ASSIGN_HOLD, " := ")
    return _WS_RE.sub(" ", text).strip()


def truncate_pair(human_text, ai_text):
    """Cắt cả hai về cùng số từ L = min(len). CHỈ dùng khi dựng dataset.

    Gọi SAU canonicalize() — phiên chuyển làm đổi số từ ("∈" -> "in"), nên cắt
    trước thì độ dài không còn cân bằng.

    Trả (human, ai). Nếu một bên None (ví dụ test_fp) thì trả nguyên, không cắt.
    """
    if human_text is None or ai_text is None:
        return human_text, ai_text

    h_tok, a_tok = str(human_text).split(), str(ai_text).split()
    k = min(len(h_tok), len(a_tok))
    return " ".join(h_tok[:k]), " ".join(a_tok[:k])


# ── DỰNG DATASET v2 ────────────────────────────────────────────────────────
# Cùng bộ tách câu với 03_diagnostics/shortcut_audit. PHẢI trùng: nếu cắt theo một
# định nghĩa câu rồi lại đo avg_sentence_len theo định nghĩa khác thì con số báo cáo
# không nói về thứ vừa làm. 06_mixing dùng thêm điều kiện (?=[A-Z]) — đó là chỗ khác,
# nó cần align câu-với-câu nên thà ít biên còn hơn biên sai.
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    """Tách câu theo đúng định nghĩa mà cổng G6 dùng để đếm câu."""
    if text is None:
        return []
    return [s for s in _SENT_BOUNDARY_RE.split(str(text).strip()) if s]


def truncate_pair_sentences(human_text, ai_text, tol_words=8, tol_frac=0.03):
    """Cân bằng số từ một cặp NHƯNG chỉ cắt ở BIÊN CÂU. Thay cho truncate_pair.

    Duyệt mọi cặp tiền tố CÂU (i, j), giữ lại những cặp có |chênh số từ| <= tol, rồi
    chọn cặp GIỮ ĐƯỢC NHIỀU TỪ NHẤT; đồng hạng thì lấy cặp cân bằng hơn.
    tol = max(tol_words, tol_frac x min(tổng hai bên)) — mặc định max(8 từ, 3%).

    HAI PHIÊN BẢN ĐÃ THỬ VÀ BỎ, để không ai làm lại:

      a) cắt thô đúng từ thứ L = min(số từ hai bên)  -> `truncate_pair`, bản 02b.
         word_count về 0,5000 tuyệt đối, nhưng tỉ lệ văn bản kết thúc đúng dấu câu
         tụt 99,0% -> 52,4%: nửa tập train kết thúc giữa câu.

      b) mỗi bên lấy tiền tố câu DÀI NHẤT <= L (không dung sai, không tìm kiếm).
         ends_terminal về 0,501 nhưng word_count RÒ LẠI 0,500 -> 0,577. Lý do: câu AI
         dài hơn (22,4 vs 20,4 từ) nên khi cắt về biên câu, bên AI mất nhiều từ hơn
         một cách CÓ HỆ THỐNG. Lập luận "mỗi bên mất ~1/2 câu nên đối xứng" là sai vì
         hai bên có độ dài câu khác nhau.

    Nên phải có dung sai + tìm kiếm: dung sai cho phép chọn tổ hợp tiền tố mà hai bên
    số từ gần bằng, tìm kiếm đảm bảo không đánh đổi bằng việc cắt cụt nội dung.
    Ràng buộc "<= L" bị BỎ (khác bản b): giữ nó thì với văn bản ít câu, tổ hợp duy
    nhất còn lại là tiền tố rất ngắn (đo trên cặp 16/17 từ 3 câu: chỉ giữ 3 từ).

    Duyệt vét cạn (n_h+1) x (n_a+1) ~ 200 tổ hợp/cặp — không đáng tối ưu.
    Fallback khi không tổ hợp nào trong dung sai: lấy tổ hợp chênh ít nhất.
    """
    if human_text is None or ai_text is None:
        return human_text, ai_text

    h_sents, a_sents = split_sentences(human_text), split_sentences(ai_text)
    if not h_sents or not a_sents:
        return truncate_pair(human_text, ai_text)

    def prefix_words(sents):
        out, total = [0], 0
        for s in sents:
            total += len(s.split())
            out.append(total)
        return out

    h_cum, a_cum = prefix_words(h_sents), prefix_words(a_sents)
    tol = max(tol_words, int(round(tol_frac * min(h_cum[-1], a_cum[-1]))))

    combos = [(h_cum[i], a_cum[j], i, j)
              for i in range(1, len(h_cum)) for j in range(1, len(a_cum))]
    feasible = [c for c in combos if abs(c[0] - c[1]) <= tol]
    # giữ nhiều nhất; đồng hạng -> chênh ít nhất
    pool = feasible or combos
    _, _, i, j = max(pool, key=lambda c: (min(c[0], c[1]), -abs(c[0] - c[1])))
    return " ".join(h_sents[:i]), " ".join(a_sents[:j])


_ALLCAPS_RE = re.compile(r"^[A-Z]{2,}$")


def notation_profile(text):
    """Mật độ ký hiệu/số/tên riêng của MỘT văn bản — đầu vào của notation_gap.

    Bốn kênh, đo trên văn bản ĐÃ canonicalize (dạng cuối model nhìn thấy):
      single_char_frac     token 1 ký tự  -> biến toán (x, k, n) của bài báo gốc
      titlecase_token_frac token viết hoa chữ đầu -> tên riêng, tên phương pháp
      numeric_token_frac   token có chữ số -> số liệu
      allcaps_token_frac   ALL-CAPS -> acronym. ĐO ĐỂ BÁO CÁO, không vào gap: đã đo
                           AUC lẻ chỉ 0,551 nên nó không phải kênh rò rỉ.
    """
    tokens = str(text).split() if text is not None else []
    n = max(1, len(tokens))
    return {
        "single_char_frac": sum(1 for t in tokens if len(t) == 1) / n,
        "titlecase_token_frac": sum(1 for t in tokens if t[:1].isupper()) / n,
        "numeric_token_frac": sum(1 for t in tokens if any(c.isdigit() for c in t)) / n,
        "allcaps_token_frac": sum(1 for t in tokens if _ALLCAPS_RE.match(t.strip(".,"))) / n,
    }


# Ba kênh vào gap. allcaps bị loại có lý do đo được (xem notation_profile).
NOTATION_GAP_KEYS = ("single_char_frac", "titlecase_token_frac", "numeric_token_frac")


def notation_gap(human_profile, ai_profile):
    """Tổng |chênh| trên ba kênh artifact = mức bản AI lược ký hiệu so với bài gốc.

    Dùng TRỊ TUYỆT ĐỐI, không dùng chênh có dấu: cặp mà AI *thêm* số/tên riêng cũng
    là cặp không trung thực nội dung, và cũng rò rỉ theo hướng ngược. Lọc theo |.|
    thì diệt cả hai chiều.
    """
    return sum(abs(human_profile[k] - ai_profile[k]) for k in NOTATION_GAP_KEYS)


def non_ascii_coverage(texts):
    """QA: ký tự non-ASCII nào bị rơi xuống tầng "-> khoảng trắng"?

    Trả (n_total, n_mapped, [(ký tự, số lần) sót nhiều nhất]). Nếu danh sách sót
    có ký hiệu toán phổ biến thì phải bổ sung vào NON_ASCII_TABLE — nếu không,
    ký hiệu đó bị xoá và câu mất nghĩa như bản L2 cũ.
    """
    from collections import Counter

    total = Counter()
    for t in texts:
        if t is None:
            continue
        total.update(c for c in str(t) if ord(c) > 127)

    mapped = sum(v for c, v in total.items() if c in NON_ASCII_TABLE)
    leftover = Counter()
    for c, v in total.items():
        if c in NON_ASCII_TABLE:
            continue
        # Dấu tổ hợp đứng lẻ: _transliterate xoá nó là ĐÚNG (nó là dấu của chữ
        # bên cạnh, không phải ký hiệu độc lập) -> không tính là sót.
        if unicodedata.combining(c):
            mapped += v
            continue

        # NFKD có thể vẫn cứu được (chữ có dấu) -> không tính là sót
        stripped = "".join(x for x in unicodedata.normalize("NFKD", c)
                           if not unicodedata.combining(x))
        if not stripped or any(ord(x) > 127 for x in stripped):
            leftover[c] = v
        else:
            mapped += v

    return sum(total.values()), mapped, leftover.most_common(15)
