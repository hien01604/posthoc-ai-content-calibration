# Supplementary Material

**Estimating the AI-Generated Content Fraction of a Document: Length-Robust Document-Level Calibration via Distributional Transformation**

> Moi bang duoi day duoc **sinh tu dong** tu `supplement_rivf2026/results/*.csv` bang `render_supplement.py`. Khong con so nao duoc go tay, nen chung khong the lech voi bang trong bai bao.

Tai lieu nay chua nhung bang ma ban chinh (gioi han 6 trang IEEE) khong du cho. Moi muc ghi ro no tra loi phan nao cua bai.

---

## S1. Muoi sau checkpoint

### S1a. Hieu nang muc doan va tham so hieu chuan

Bo sung cho **§III-B**, cho hien tung checkpoint dang sau cac dai gop nhu "AUROC 0.9746--0.9875".

`rho` la uoc luong Hartung tren thang z. Bai bao cao **trung binh qua 5 lan chia** (cot `rho (mean)`; trung vi qua 16 checkpoint = 0.303). Code trien khai `paper_level_hartung_ood.py` lai dung **trung vi qua 5 lan chia** (cot `rho (median)`); chenh lech khong dang ke nhung duoc ghi ra day cho minh bach.

| ID | Architecture | HF base model | Seed | Type | AUROC (OOD) | TPR@FPR1% (OOD) | TPR@tau (OOD) | FPR@tau (OOD) | mu0 (val) | mu1 (val) | rho (mean) | rho (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | DeBERTa-v3-large | microsoft/deberta-v3-large | 42 | Supervised | 0.9858 | 0.9095 | 0.9254 | 0.0192 | 0.0450 | 0.9597 | 0.3608 | 0.3537 |
| C02 | DeBERTa-v3-large | microsoft/deberta-v3-large | 43 | Supervised | 0.9875 | 0.8930 | 0.9155 | 0.0185 | 0.0413 | 0.9611 | 0.3374 | 0.3448 |
| C03 | DeBERTa-v3-large | microsoft/deberta-v3-large | 44 | Supervised | 0.9858 | 0.8976 | 0.9174 | 0.0211 | 0.0695 | 0.9692 | 0.3917 | 0.3976 |
| C04 | ModernBERT-large | answerdotai/ModernBERT-large | 42 | Supervised | 0.9808 | 0.8923 | 0.8970 | 0.0119 | 0.0253 | 0.9446 | 0.2663 | 0.2576 |
| C05 | ModernBERT-large | answerdotai/ModernBERT-large | 43 | Supervised | 0.9808 | 0.8989 | 0.9082 | 0.0145 | 0.0334 | 0.9522 | 0.2288 | 0.2241 |
| C06 | ModernBERT-large | answerdotai/ModernBERT-large | 44 | Supervised | 0.9816 | 0.8923 | 0.8943 | 0.0132 | 0.0467 | 0.9590 | 0.2666 | 0.2641 |
| C07 | RoBERTa-large | roberta-large | 42 | Supervised | 0.9832 | 0.8897 | 0.9036 | 0.0165 | 0.0658 | 0.9643 | 0.2942 | 0.2900 |
| C08 | RoBERTa-large | roberta-large | 43 | Supervised | 0.9834 | 0.9022 | 0.9069 | 0.0132 | 0.0467 | 0.9546 | 0.2695 | 0.2764 |
| C09 | RoBERTa-large | roberta-large | 44 | Supervised | 0.9835 | 0.8970 | 0.9082 | 0.0159 | 0.0436 | 0.9495 | 0.3511 | 0.3593 |
| C10 | XLM-RoBERTa-large | xlm-roberta-large | 42 | Supervised | 0.9808 | 0.8897 | 0.9009 | 0.0112 | 0.0440 | 0.9590 | 0.2039 | 0.2106 |
| C11 | XLM-RoBERTa-large | xlm-roberta-large | 43 | Supervised | 0.9832 | 0.8910 | 0.8910 | 0.0092 | 0.0585 | 0.9592 | 0.2688 | 0.2716 |
| C12 | XLM-RoBERTa-large | xlm-roberta-large | 44 | Supervised | 0.9824 | 0.8884 | 0.8937 | 0.0112 | 0.0711 | 0.9593 | 0.3121 | 0.3231 |
| C13 | GREATER + RoBERTa-large | roberta-large | 42 | Adversarial | 0.9839 | 0.8983 | 0.9022 | 0.0112 | 0.0849 | 0.9100 | 0.2781 | 0.2796 |
| C14 | GREATER + XLM-RoBERTa-large | xlm-roberta-large | 42 | Adversarial | 0.9832 | 0.8930 | 0.8818 | 0.0079 | 0.0840 | 0.9073 | 0.3593 | 0.3645 |
| C15 | RADAR + RoBERTa-large | roberta-large | 42 | Adversarial | 0.9808 | 0.8778 | 0.8871 | 0.0145 | 0.0314 | 0.9363 | 0.4012 | 0.4069 |
| C16 | RADAR + XLM-RoBERTa-large | xlm-roberta-large | 42 | Adversarial | 0.9746 | 0.8124 | 0.8435 | 0.0152 | 0.0584 | 0.9383 | 0.3340 | 0.3376 |

**Doc luu y.** `TPR@tau` la TPR tai nguong `tau` duoc hieu chuan o FPR = 1%% tren `val`; FPR thuc te cua nguong do tren `test_ood_model` cao hon 1%% o mot so checkpoint. `TPR@FPR1%` la TPR tai dung FPR 1%% do **tren chinh** `test_ood_model`. Bai bao cao cot `TPR@tau` (0.8435--0.9254).

Nguon: `results/s1_checkpoints.csv`

### S1b. Ket qua thong ke muc tai lieu theo tung checkpoint

Bang nay noi truc tiep model voi hai ket qua chinh cua bai: do dung tam cua `pi-hat` va quy tac `Z_k` tren `test_ood_model` (637 bai human + 637 bai AI). Cac cot FPR/miss deu la phan tram.

| ID | Architecture | Seed | pi-hat H | pi-hat AI | rho | FPR z % | Miss z % | FPR raw % | Miss raw % | McNemar p | BH q<.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | DeBERTa-v3-large | 42 | 0.016 | 0.984 | 0.354 | 1.88 | 3.30 | 1.10 | 8.16 | 9.31e-10 | True |
| C02 | DeBERTa-v3-large | 43 | 0.004 | 0.981 | 0.345 | 1.26 | 2.67 | 0.94 | 5.02 | 1.49e-03 | True |
| C03 | DeBERTa-v3-large | 44 | 0.021 | 0.982 | 0.398 | 1.73 | 3.14 | 1.41 | 11.93 | 4.09e-16 | True |
| C04 | ModernBERT-large | 42 | 0.005 | 0.973 | 0.258 | 0.94 | 3.92 | 1.41 | 3.61 | 8.32e-01 | False |
| C05 | ModernBERT-large | 43 | 0.010 | 0.978 | 0.224 | 1.26 | 3.45 | 1.10 | 5.81 | 6.10e-05 | True |
| C06 | ModernBERT-large | 44 | 0.012 | 0.975 | 0.264 | 0.78 | 4.08 | 0.94 | 4.08 | 1.00e+00 | False |
| C07 | RoBERTa-large | 42 | 0.006 | 0.987 | 0.290 | 1.41 | 3.61 | 1.41 | 11.46 | 1.65e-13 | True |
| C08 | RoBERTa-large | 43 | 0.001 | 0.994 | 0.276 | 1.41 | 3.45 | 0.94 | 4.40 | 2.38e-01 | False |
| C09 | RoBERTa-large | 44 | 0.006 | 0.991 | 0.359 | 1.26 | 4.08 | 1.26 | 6.12 | 7.20e-03 | True |
| C10 | XLM-RoBERTa-large | 42 | 0.006 | 0.989 | 0.211 | 1.41 | 3.30 | 1.10 | 4.71 | 6.36e-02 | False |
| C11 | XLM-RoBERTa-large | 43 | -0.009 | 0.980 | 0.272 | 0.94 | 4.08 | 0.94 | 6.12 | 4.10e-02 | False |
| C12 | XLM-RoBERTa-large | 44 | 0.004 | 0.995 | 0.323 | 0.78 | 4.87 | 1.26 | 10.68 | 9.33e-09 | True |
| C13 | GREATER + RoBERTa-large | 42 | 0.008 | 0.988 | 0.280 | 1.26 | 2.83 | 1.10 | 5.97 | 1.80e-04 | True |
| C14 | GREATER + XLM-RoBERTa-large | 42 | -0.002 | 0.975 | 0.365 | 0.47 | 4.87 | 0.94 | 6.75 | 5.76e-02 | False |
| C15 | RADAR + RoBERTa-large | 42 | 0.001 | 0.966 | 0.407 | 0.78 | 5.18 | 1.41 | 4.24 | 2.63e-01 | False |
| C16 | RADAR + XLM-RoBERTa-large | 42 | 0.017 | 0.969 | 0.338 | 1.10 | 8.16 | 1.41 | 19.15 | 1.05e-14 | True |

**Pham vi so sanh raw.** De ghep cap tren cung 637 bai, cot raw dung trung binh cua 200 nguong duoc chot tren cac nua calibration, roi ap mot lan cho toan bo tap. Vi vay miss raw trung vi la **6,04%**, khac nhe uoc luong split-evaluation **6,16%** o Table IV. FPR cung khong khop hoan toan theo checkpoint; McNemar o day do bat dong tai cac diem van hanh quan sat, khong chung minh uu the power o FPR bang nhau tuyet doi.

Nguon: `results/s1b_document_results.csv`, `revision_experiments_rivf2026/results/rev3_paired_miss.csv`

---

## S2. Dataset

### S2a. Thong ke split

Bo sung cho **§III-A** va **§III-C**. So o day la so **thuc te da cham diem** (sau khi loc khu nhieu be mat), khong phai so truoc khi loc. `k` tinh theo (paper_id, label) -- dung dinh nghia cua bai.

| Split | # papers | # segments | # human | # AI | Median k | Mean k | P75 k | Max k | Source |
|---|---|---|---|---|---|---|---|---|---|
| val | 183.0 | 876 | 438 | 438 | 3.00 | 2.39 | 3.00 | 3.0 | scored parquet (measured here) |
| test_ood_model | 637.0 | 3028 | 1514 | 1514 | 3.00 | 2.38 | 3.00 | 3.0 | scored parquet (measured here) |
| test_fp | 1529.0 | 4290 | 4290 | 0 | 3.00 | 2.81 | 3.00 | 3.0 | scored parquet (measured here) |
| test_fp_full | 1200.0 | 27119 | 27119 | 0 | 19.00 | 22.60 | 27.00 | 262.0 | scored parquet (measured here) |
| train |  | 14050 | 7025 | 7025 |  |  |  |  | REPORT_2026-08-13.md Sec 3.4 (no scored artifact) |

Nguon: `results/s2b_splits.csv`

### S2b. Model sinh van ban AI: 21 model tren 7 ho

Bo sung cho **§III-A**. Diem thiet ke quan trong: **cac ho generator roi nhau hoan toan** giua tap huan luyen/thuc va tap OOD -- day chinh la nghia cua ten `test_ood_model`.

| Split | Generator family | # AI segments | % of AI in split |
|---|---|---|---|
| val | mistral | 104 | 23.74 |
| val | openai | 99 | 22.60 |
| val | anthropic | 96 | 21.92 |
| val | qwen | 80 | 18.26 |
| val | meta | 59 | 13.47 |
| test_ood_model | google | 856 | 56.54 |
| test_ood_model | deepseek | 658 | 43.46 |

Nguon: `results/s2c_generators.csv`

#### Danh sach day du -- bai bao tro toi day tu §III-A

Parse truc tiep tu `FAMILY_BY_KEY` trong `genai.ipynb`. Cot cuoi doi chieu cheo voi `generator_family` co that trong parquet da cham diem.

| OpenRouter model ID | Family | Split pool | Seen in scored data |
|---|---|---|---|
| anthropic/claude-haiku-4.5 | anthropic | train/val | True |
| anthropic/claude-sonnet-5 | anthropic | train/val | True |
| deepseek/deepseek-v4-flash | deepseek | test_ood_model | True |
| deepseek/deepseek-v4-pro | deepseek | test_ood_model | True |
| google/gemini-2.5-flash | google | test_ood_model | True |
| google/gemini-2.5-flash-lite | google | test_ood_model | True |
| google/gemma-4-26b-a4b-it | google | test_ood_model | True |
| google/gemma-4-31b-it | google | test_ood_model | True |
| meta-llama/llama-3.1-8b-instruct | meta | train/val | True |
| meta-llama/llama-3.3-70b-instruct | meta | train/val | True |
| mistralai/ministral-14b-2512 | mistral | train/val | True |
| mistralai/ministral-8b-2512 | mistral | train/val | True |
| mistralai/mistral-large-2512 | mistral | train/val | True |
| mistralai/mistral-small-2603 | mistral | train/val | True |
| openai/gpt-5-mini | openai | train/val | True |
| openai/gpt-5-nano | openai | train/val | True |
| openai/gpt-oss-120b | openai | train/val | True |
| openai/gpt-oss-20b | openai | train/val | True |
| qwen/qwen3-coder | qwen | train/val | True |
| qwen/qwen3-next-80b-a3b-instruct | qwen | train/val | True |
| qwen/qwen3.5-122b-a10b | qwen | train/val | True |

**15 model / 5 ho** cho train+val; **6 model / 2 ho** (`google`, `deepseek`) giu rieng cho `test_ood_model` va **khong xuat hien trong train duoi bat ky hinh thuc nao** -- day la nghia chinh xac cua "OOD-model".

#### Cau hinh sinh

| Setting | Key | Meaning | Detail | Share % |
|---|---|---|---|---|
| generation mode | B | rewrite the human passage | templates: B1, B2, B3 | 55.00 |
| generation mode | A | write from scratch | templates: A1, A2 | 25.00 |
| generation mode | C | continue the passage | templates: C1 | 10.00 |
| generation mode | D | expand from bullet notes | templates: D1 | 10.00 |
| temperature | 0.3 |  | assigned by hash of (paper_id, source) | 20.00 |
| temperature | 0.5 |  | assigned by hash of (paper_id, source) | 20.00 |
| temperature | 0.7 |  | assigned by hash of (paper_id, source) | 20.00 |
| temperature | 0.9 |  | assigned by hash of (paper_id, source) | 20.00 |
| temperature | 1.1 |  | assigned by hash of (paper_id, source) | 20.00 |
| family quota (train) | TRAIN_FAMILY_QUOTA | 0.25 | soft cap per family | 25.00 |

Mode, template va temperature deu duoc gan bang **hash tat dinh cua (paper_id, source)** -- khong dung random, nen chay lai cho ket qua y het.

Nguon: `results/s2d_generator_models.csv`, `results/s2e_generation_config.csv`

### S2c. Cong thuc dung `test_fp_full`

Bo sung cho **§III-C**. Doc tu `output/testfp_full_manifest.json` -- day la recipe day du de dung lai tap nay tu dau.

| Property | Value |
|---|---|
| source corpus | results/with_doi_le_2018.jsonl |
| chunker | get-dataset.ipynb split_into_sentence_chunks (copy nguyên văn) |
| target words / chunk | 275 |
| min words / chunk | 200 |
| max words / chunk | 350 |
| median words / chunk | 290 |
| drop first chunk | True |
| canonicalized | True |
| paper_ids excluded from | train, val, test_ood_model, test_fp |
| published year range | 2010-2015 |
| n papers | 1200 |
| n chunks | 27119 |
| chunks/paper min | 10 |
| chunks/paper p25 | 14 |
| chunks/paper median | 19 |
| chunks/paper p75 | 27 |
| chunks/paper mean | 22.6 |
| chunks/paper max | 262 |
| label | 0 (human) cho MỌI dòng — bài xuất bản trước ChatGPT |

Diem quan trong: dung **dung ham** `split_into_sentence_chunks` cua `get-dataset.ipynb`, chi bo buoc `random.sample(k=3)` -- nen `test_fp_full` va cac split khac cat doan **giong het nhau**, chi khac so doan giu lai moi bai. **5.274** `paper_id` xuat hien o split khac da bi loai truoc khi dung.

Nguon: `results/s2f_construction_recipe.csv`

### S2d. Schema record-level

Bo sung cho **§III-A**. Doc truc tiep tu parquet.

**`pred_val.parquet`**

| Field | Type | # unique | Example | Description |
|---|---|---|---|---|
| paper_id | object | 183 | 1001.2897 | arXiv ID cua bai |
| label | int64 | 2 | 0 | 0 = human, 1 = AI |
| ai_degree | int64 | 3 | 0 | 0 thuan human | 1 AI-polish | 3 AI viet lai | 4 AI viet moi |
| generator_family | object | 6 | human | ho model sinh AI, hoac 'human' |
| score | float64 | 539 | 0.9996155500411987 | diem detector (mot checkpoint / file) |

**`pred_test_ood_model.parquet`**

| Field | Type | # unique | Example | Description |
|---|---|---|---|---|
| paper_id | object | 637 | 1001.1401 | arXiv ID cua bai |
| label | int64 | 2 | 0 | 0 = human, 1 = AI |
| ai_degree | int64 | 3 | 0 | 0 thuan human | 1 AI-polish | 3 AI viet lai | 4 AI viet moi |
| generator_family | object | 3 | human | ho model sinh AI, hoac 'human' |
| score | float64 | 1560 | 0.0017274586716666818 | diem detector (mot checkpoint / file) |

**`pred_test_fp.parquet`**

| Field | Type | # unique | Example | Description |
|---|---|---|---|---|
| paper_id | object | 1529 | 1001.0529 | arXiv ID cua bai |
| label | int64 | 1 | 0 | 0 = human, 1 = AI |
| ai_degree | int64 | 2 | 1 | 0 thuan human | 1 AI-polish | 3 AI viet lai | 4 AI viet moi |
| generator_family | object | 1 | human | ho model sinh AI, hoac 'human' |
| score | float64 | 2613 | 0.021064694970846176 | diem detector (mot checkpoint / file) |

**`pred_test_fp_full.parquet`**

| Field | Type | # unique | Example | Description |
|---|---|---|---|---|
| sample_id | object | 27119 | 1503.08289__full_000 | dinh danh doan |
| paper_id | object | 1200 | 1503.08289 | arXiv ID cua bai |
| source | object | 262 | full_chunk_000 | vi tri doan trong bai (full_chunk_NNN) |
| label | int64 | 1 | 0 | 0 = human, 1 = AI |
| artifact | object | 1 | artifacts_roberta_large_seed42 | ten artifact checkpoint |
| branch | object | 1 | supervised | nhanh huan luyen |
| model | object | 1 | roberta_large | ten run |
| seed | int64 | 1 | 42 | seed |
| score | float64 | 27067 | 0.002407731954008341 | diem detector (mot checkpoint / file) |

**`dataset_test_fp_full_deconf2.parquet`**

| Field | Type | # unique | Example | Description |
|---|---|---|---|---|
| sample_id | object | 27119 | 1503.08289__full_000 | dinh danh doan |
| paper_id | object | 1200 | 1503.08289 | arXiv ID cua bai |
| source | object | 262 | full_chunk_000 | vi tri doan trong bai (full_chunk_NNN) |
| human_text | object | 27113 | For example, if theta i is considered twice as good as theta | van ban nguoi viet |
| ai_text | object | 0 |  | van ban AI ghep cap |
| ai_degree | int64 | 1 | 0 | 0 thuan human | 1 AI-polish | 3 AI viet lai | 4 AI viet moi |
| sample_weight | float64 | 1 | 1.0 | trong so can bang |
| split | object | 1 | test_fp_full | train / val / test_ood_model / test_fp / test_fp_full |
| published_year | int64 | 6 | 2015 | nam cong bo |
| categories | object | 410 | cs.AI | danh muc arXiv |
| notation_gap | float64 | 0 |  | chenh lech ky hieu |
| notation_ok | bool | 1 | True | co qua bo loc ky hieu khong |

Nguon: `results/s2a_schema.csv`

---

## S3. Do tan mac qua 16 checkpoint cho moi bang chinh

Tra loi **Revision Brief muc 4**. Khoang tin cay bootstrap trong Table III cua bai do **bat dinh doi voi quan the bai bao**; no khong noi gi ve viec 16 checkpoint tan mac ra sao. Bang nay bo sung dung phan do.

### Table I

| Quantity | Median | Min | Max | # ckpt |
|---|---|---|---|---|
| Table I: mean pi-hat, human | 0.006 | -0.009 | 0.021 | 16 |
| Table I: mean pi-hat, AI | 0.981 | 0.966 | 0.995 | 16 |
| Table I: median pi-hat, human | -0.040 | -0.072 | -0.021 | 16 |
| Table I: median pi-hat, AI | 1.048 | 1.034 | 1.070 | 16 |

### Table II

| Quantity | Median | Min | Max | # ckpt |
|---|---|---|---|---|
| Segment level (val, k~1): FPR | 0.24 | 0.00 | 0.63 | 16 |
| Segment level (val, k~1): miss | 17.82 | 13.50 | 24.80 | 16 |
| k~22.6 (test_fp_full): FPR | 5.10 | 3.77 | 7.85 | 16 |
| k~22.6 (test_fp_full): miss | 1.10 | 0.63 | 4.40 | 16 |
| Correct k~2.38 (control): FPR | 1.31 | 1.23 | 1.40 | 16 |
| Correct k~2.38 (control): miss | 6.16 | 3.85 | 20.50 | 16 |

### Table III

| Quantity | Median | Min | Max | # ckpt |
|---|---|---|---|---|
| k=1, Static: FPR | 4.19 | 1.43 | 8.66 | 16 |
| k=1, Raw scale, empirical quantile: FPR | 1.10 | 0.87 | 1.36 | 16 |
| k=1, z-scale (proposed): FPR | 0.97 | 0.85 | 1.12 | 16 |
| k=2, Static: FPR | 5.67 | 2.15 | 8.24 | 16 |
| k=2, Raw scale, empirical quantile: FPR | 0.48 | 0.22 | 1.62 | 16 |
| k=2, z-scale (proposed): FPR | 0.96 | 0.76 | 1.11 | 16 |
| k=3, Static: FPR | 3.15 | 2.00 | 4.91 | 16 |
| k=3, Raw scale, empirical quantile: FPR | 0.70 | 0.34 | 1.53 | 16 |
| k=3, z-scale (proposed): FPR | 0.91 | 0.63 | 1.07 | 16 |
| k=5, Static: FPR | 2.04 | 1.65 | 3.30 | 16 |
| k=5, Raw scale, empirical quantile: FPR | 0.56 | 0.34 | 1.31 | 16 |
| k=5, z-scale (proposed): FPR | 0.88 | 0.49 | 1.11 | 16 |
| k=8, Static: FPR | 1.31 | 1.06 | 1.89 | 16 |
| k=8, Raw scale, empirical quantile: FPR | 0.66 | 0.43 | 1.36 | 16 |
| k=8, z-scale (proposed): FPR | 0.84 | 0.43 | 1.18 | 16 |
| k=10, Static: FPR | 1.13 | 0.87 | 1.58 | 16 |
| k=10, Raw scale, empirical quantile: FPR | 0.70 | 0.47 | 1.44 | 16 |
| k=10, z-scale (proposed): FPR | 0.84 | 0.39 | 1.20 | 16 |
| k=15, Static: FPR | 0.85 | 0.63 | 1.25 | 16 |
| k=15, Raw scale, empirical quantile: FPR | 0.94 | 0.53 | 1.57 | 16 |
| k=15, z-scale (proposed): FPR | 0.81 | 0.33 | 1.25 | 16 |
| k=20, Static: FPR | 0.71 | 0.56 | 1.21 | 16 |
| k=20, Raw scale, empirical quantile: FPR | 1.12 | 0.57 | 2.11 | 16 |
| k=20, z-scale (proposed): FPR | 0.76 | 0.31 | 1.24 | 16 |

### Table IV

| Quantity | Median | Min | Max | # ckpt |
|---|---|---|---|---|
| z-score (out of sample): FPR | 1.26 | 0.47 | 1.88 | 16 |
| z-score (out of sample): miss | 3.77 | 2.67 | 8.16 | 16 |
| Correct k~2.38 (control): FPR | 1.31 | 1.23 | 1.40 | 16 |
| Correct k~2.38 (control): miss | 6.16 | 3.85 | 20.50 | 16 |

Xem them `figures/figS1_fpr_boxplot.pdf`: boxplot phan bo FPR cua 16 checkpoint tai moi `k`, ba phuong phap canh nhau.

Nguon: `results/s4_ranges_main_tables.csv`

---

## S4. Pool hieu chuan `m` va do phan giai p-value

Tra loi **Revision Brief muc 6.2**, va la co so cho cau canh bao them vao §V-B.

| Dung o | Pool | m (median) | m min | m max | p nho nhat | # muc p duoi 1% | z lon nhat |
|---|---|---|---|---|---|---|---|
| Sec V-A / V-C (controlled experiment) | human segments in the calibration half of test_fp_full | 9027 | 8785 | 9432 | 5.54e-05 | 90.28 | 3.866 |
| Sec V-D (out-of-sample transfer) | human segments of val | 438 | 438 | 438 | 1.14e-03 | 4.39 | 3.051 |

Trong phep chuyen ngoai mau (§V-D), pool chi co **438** doan human, nen `z_i` bi chan tren o **3.05** va chi co **4** muc p roi rac nam duoi `alpha = 1%`. Do la mot gioi han that ve do phan giai duoi phai, khong chi la van de co mau.

Nguon: `results/s6_calibration_pool.csv`

---

## S5. Da kiem dinh cho McNemar

Tra loi **Revision Brief muc 6.1**. McNemar chay 16 lan (mot lan / checkpoint), nen p tho phai duoc hieu chinh.

| Tieu chi | So checkpoint co y nghia |
|---|---|
| Tho (p < 0.05) | **10** / 16 |
| Benjamini--Hochberg (q = 0.05) | **9** / 16 |
| Bonferroni (p < 0.05/16) | **8** / 16 |

| Checkpoint | miss (z) | miss (control) | b: only z | c: only control | p | raw | BH | Bonf. |
|---|---|---|---|---|---|---|---|---|
| deberta_v3_large_seed44 | 0.0314 | 0.1193 | 57 | 1 | 4.09e-16 | True | True | True |
| radar_xlm_seed42 | 0.0816 | 0.1915 | 80 | 10 | 1.05e-14 | True | True | True |
| roberta_large_seed42 | 0.0361 | 0.1146 | 52 | 2 | 1.65e-13 | True | True | True |
| deberta_v3_large_seed42 | 0.0330 | 0.0816 | 31 | 0 | 9.31e-10 | True | True | True |
| xlm_roberta_large_seed44 | 0.0487 | 0.1068 | 41 | 4 | 9.33e-09 | True | True | True |
| modernbert_large_seed43 | 0.0345 | 0.0581 | 15 | 0 | 6.10e-05 | True | True | True |
| greater_roberta_large_seed42 | 0.0283 | 0.0597 | 24 | 4 | 1.80e-04 | True | True | True |
| deberta_v3_large_seed43 | 0.0267 | 0.0502 | 18 | 3 | 1.49e-03 | True | True | True |
| roberta_large_seed44 | 0.0408 | 0.0612 | 17 | 4 | 7.20e-03 | True | True | False |
| xlm_roberta_large_seed43 | 0.0408 | 0.0612 | 24 | 11 | 4.10e-02 | True | False | False |
| greater_xlm_roberta_large_seed42 | 0.0487 | 0.0675 | 23 | 11 | 5.76e-02 | False | False | False |
| xlm_roberta_large_seed42 | 0.0330 | 0.0471 | 14 | 5 | 6.36e-02 | False | False | False |
| roberta_large_seed43 | 0.0345 | 0.0440 | 12 | 6 | 2.38e-01 | False | False | False |
| radar_seed42 | 0.0518 | 0.0424 | 7 | 13 | 2.63e-01 | False | False | False |
| modernbert_large_seed42 | 0.0392 | 0.0361 | 10 | 12 | 8.32e-01 | False | False | False |
| modernbert_large_seed44 | 0.0408 | 0.0408 | 3 | 3 | 1.00e+00 | False | False | False |

> 16 checkpoint **khong doc lap** -- chung cham cung 637 bai AI -- nen ca Bonferroni lan BH deu bao thu qua muc o day. Con so dang tin la so checkpoint rieng le vuot nguong, khong phai mot p gop.

Nguon: `results/s6_multiple_testing.csv`

---

## S6. FPR muc tai lieu theo tieu linh vuc arXiv

Tra loi **Revision Brief muc 4** (dong cuoi). Ban truoc cua bai chi neu hai diem cuc tri (`cs.CV` 3.90%, `cs.DS` 23.53%) va tu ghi chu rang chung duoc do **truoc** khi chuyen sang khung pi-hat -- tuc chung khong phai FPR cua phuong phap dang de xuat. Bang duoi do lai **tat ca** tieu linh vuc duoi dung quy tac trien khai (z-scale, `k` that cua tung bai), va kem quy tac nguong tinh de so sanh.

### z-scale (proposed)

| Subfield | # papers | Median k | FPR (pooled) % | Wilson lo | Wilson hi | min over ckpt | max over ckpt | n>=20? |
|---|---|---|---|---|---|---|---|---|
| cs.IT | 213 | 19 | 1.14 | 0.84 | 1.56 | 0.00 | 4.23 | True |
| cs.NI | 91 | 19 | 1.10 | 0.68 | 1.78 | 0.00 | 2.20 | True |
| cs.LO | 90 | 24 | 1.81 | 1.24 | 2.63 | 0.00 | 12.22 | True |
| cs.CV | 77 | 17 | 0.24 | 0.08 | 0.71 | 0.00 | 2.60 | True |
| cs.AI | 65 | 28 | 1.15 | 0.66 | 2.01 | 0.00 | 4.62 | True |
| cs.SI | 63 | 18 | 3.87 | 2.84 | 5.25 | 0.00 | 7.94 | True |
| eess.SY | 46 | 18 | 0.82 | 0.37 | 1.77 | 0.00 | 4.35 | True |
| cs.SE | 43 | 18 | 0.15 | 0.03 | 0.82 | 0.00 | 2.33 | True |
| cs.CR | 39 | 19 | 0.96 | 0.44 | 2.08 | 0.00 | 2.56 | True |
| cs.DS | 34 | 21 | 2.21 | 1.27 | 3.82 | 0.00 | 8.82 | True |
| cs.LG | 33 | 20 | 0.00 | 0.00 | 0.72 | 0.00 | 0.00 | True |
| cs.CY | 29 | 17 | 3.45 | 2.13 | 5.53 | 0.00 | 10.34 | True |
| cs.DC | 27 | 16 | 2.08 | 1.10 | 3.91 | 0.00 | 3.70 | True |
| cs.PL | 26 | 21 | 7.21 | 5.10 | 10.11 | 0.00 | 15.38 | True |
| cs.GT | 25 | 23 | 2.00 | 1.02 | 3.90 | 0.00 | 8.00 | True |
| cs.FL | 24 | 19 | 3.12 | 1.80 | 5.38 | 0.00 | 12.50 | True |
| cs.CL | 23 | 19 | 0.00 | -0.00 | 1.03 | 0.00 | 0.00 | True |
| cs.DM | 23 | 23 | 0.54 | 0.15 | 1.96 | 0.00 | 4.35 | True |
| cs.IR | 21 | 20 | 5.95 | 3.89 | 9.01 | 4.76 | 9.52 | True |
| cs.DL | 18 | 18 | 0.00 | 0.00 | 1.32 | 0.00 | 0.00 | False |
| cs.DB | 16 | 18 | 1.56 | 0.61 | 3.95 | 0.00 | 6.25 | False |
| cs.NE | 16 | 16 | 0.00 | 0.00 | 1.48 | 0.00 | 0.00 | False |
| cs.CC | 15 | 20 | 0.00 | -0.00 | 1.58 | 0.00 | 0.00 | False |
| cs.CE | 15 | 18 | 7.50 | 4.80 | 11.54 | 6.67 | 13.33 | False |
| cs.OH | 15 | 17 | 0.00 | -0.00 | 1.58 | 0.00 | 0.00 | False |
| math.NA | 14 | 20 | 11.61 | 8.05 | 16.46 | 7.14 | 21.43 | False |
| cs.CG | 13 | 20 | 3.85 | 1.96 | 7.40 | 0.00 | 7.69 | False |
| cs.HC | 13 | 14 | 0.00 | 0.00 | 1.81 | 0.00 | 0.00 | False |
| cs.RO | 12 | 18 | 0.52 | 0.09 | 2.89 | 0.00 | 8.33 | False |
| cs.SC | 11 | 17 | 1.14 | 0.31 | 4.05 | 0.00 | 9.09 | False |
| cs.ET | 10 | 24 | 0.00 | -0.00 | 2.34 | 0.00 | 0.00 | False |
| cs.MA | 10 | 22 | 1.25 | 0.34 | 4.44 | 0.00 | 10.00 | False |
| cs.MM | 7 | 14 | 0.00 | 0.00 | 3.32 | 0.00 | 0.00 | False |
| cs.SD | 7 | 25 | 0.00 | 0.00 | 3.32 | 0.00 | 0.00 | False |
| cs.AR | 5 | 15 | 11.25 | 6.03 | 20.02 | 0.00 | 20.00 | False |
| cs.MS | 5 | 11 | 15.00 | 8.79 | 24.41 | 0.00 | 20.00 | False |
| cs.PF | 5 | 19 | 0.00 | -0.00 | 4.58 | 0.00 | 0.00 | False |
| cs.OS | 1 | 21 | 0.00 | 0.00 | 19.36 | 0.00 | 0.00 | False |

### static (raw scale)

| Subfield | # papers | Median k | FPR (pooled) % | Wilson lo | Wilson hi | min over ckpt | max over ckpt | n>=20? |
|---|---|---|---|---|---|---|---|---|
| cs.IT | 213 | 19 | 0.50 | 0.31 | 0.80 | 0.00 | 0.94 | True |
| cs.NI | 91 | 19 | 0.62 | 0.33 | 1.17 | 0.00 | 1.10 | True |
| cs.LO | 90 | 24 | 0.14 | 0.04 | 0.51 | 0.00 | 1.11 | True |
| cs.CV | 77 | 17 | 0.08 | 0.01 | 0.46 | 0.00 | 1.30 | True |
| cs.AI | 65 | 28 | 0.48 | 0.21 | 1.12 | 0.00 | 3.08 | True |
| cs.SI | 63 | 18 | 2.78 | 1.93 | 3.99 | 0.00 | 4.76 | True |
| eess.SY | 46 | 18 | 0.14 | 0.02 | 0.77 | 0.00 | 2.17 | True |
| cs.SE | 43 | 18 | 0.00 | 0.00 | 0.56 | 0.00 | 0.00 | True |
| cs.CR | 39 | 19 | 0.16 | 0.03 | 0.90 | 0.00 | 2.56 | True |
| cs.DS | 34 | 21 | 0.18 | 0.03 | 1.03 | 0.00 | 2.94 | True |
| cs.LG | 33 | 20 | 0.00 | 0.00 | 0.72 | 0.00 | 0.00 | True |
| cs.CY | 29 | 17 | 1.94 | 1.02 | 3.64 | 0.00 | 6.90 | True |
| cs.DC | 27 | 16 | 2.08 | 1.10 | 3.91 | 0.00 | 3.70 | True |
| cs.PL | 26 | 21 | 6.73 | 4.70 | 9.56 | 0.00 | 11.54 | True |
| cs.GT | 25 | 23 | 0.25 | 0.04 | 1.40 | 0.00 | 4.00 | True |
| cs.FL | 24 | 19 | 1.04 | 0.41 | 2.65 | 0.00 | 4.17 | True |
| cs.CL | 23 | 19 | 0.00 | -0.00 | 1.03 | 0.00 | 0.00 | True |
| cs.DM | 23 | 23 | 0.00 | -0.00 | 1.03 | 0.00 | 0.00 | True |
| cs.IR | 21 | 20 | 4.76 | 2.95 | 7.59 | 0.00 | 9.52 | True |
| cs.DL | 18 | 18 | 0.00 | 0.00 | 1.32 | 0.00 | 0.00 | False |
| cs.DB | 16 | 18 | 0.78 | 0.21 | 2.80 | 0.00 | 6.25 | False |
| cs.NE | 16 | 16 | 0.39 | 0.07 | 2.18 | 0.00 | 6.25 | False |
| cs.CC | 15 | 20 | 0.00 | -0.00 | 1.58 | 0.00 | 0.00 | False |
| cs.CE | 15 | 18 | 9.58 | 6.47 | 13.97 | 6.67 | 13.33 | False |
| cs.OH | 15 | 17 | 0.00 | -0.00 | 1.58 | 0.00 | 0.00 | False |
| math.NA | 14 | 20 | 9.38 | 6.21 | 13.91 | 0.00 | 14.29 | False |
| cs.CG | 13 | 20 | 2.40 | 1.03 | 5.50 | 0.00 | 7.69 | False |
| cs.HC | 13 | 14 | 0.00 | 0.00 | 1.81 | 0.00 | 0.00 | False |
| cs.RO | 12 | 18 | 0.00 | 0.00 | 1.96 | 0.00 | 0.00 | False |
| cs.SC | 11 | 17 | 0.00 | 0.00 | 2.14 | 0.00 | 0.00 | False |
| cs.ET | 10 | 24 | 0.00 | -0.00 | 2.34 | 0.00 | 0.00 | False |
| cs.MA | 10 | 22 | 0.00 | -0.00 | 2.34 | 0.00 | 0.00 | False |
| cs.MM | 7 | 14 | 0.00 | 0.00 | 3.32 | 0.00 | 0.00 | False |
| cs.SD | 7 | 25 | 0.00 | 0.00 | 3.32 | 0.00 | 0.00 | False |
| cs.AR | 5 | 15 | 5.00 | 1.96 | 12.16 | 0.00 | 20.00 | False |
| cs.MS | 5 | 11 | 5.00 | 1.96 | 12.16 | 0.00 | 20.00 | False |
| cs.PF | 5 | 19 | 0.00 | -0.00 | 4.58 | 0.00 | 0.00 | False |
| cs.OS | 1 | 21 | 0.00 | 0.00 | 19.36 | 0.00 | 0.00 | False |

Voi 19 tieu linh vuc co it nhat 20 bai, FPR di tu **0.00%** den **7.21%**, trung vi **1.15%**. Khoang Wilson cua `cs.PL` la [5.10%, 10.11%], loai tru muc tieu 1% -- nen thien lech theo tieu linh vuc la **that**, khong bi phep bien doi xoa di. Nhung 30/38 tieu linh vuc co duoi 40 bai, nen tung uoc luong rieng le rat yeu.

Nguon: `results/s7_subfield_summary.csv`, `results/s7_subfield_fpr.csv`

---

## S7. pi-hat tren van ban tron o muc CAU

Dat so cho han che dinh tinh o **§VI**: mo hinh tron 2 thanh phan gia dinh moi doan thuoc **tron ven** mot trong hai cuc, nen pi-hat phai doc la "muc do giong van phong AI", khong phai ti le ky tu do AI go.

`artifacts/pred_mix/scores_all.parquet`: **325.728** dong da cham diem, **609** bai, **16** checkpoint. Moi dong la MOT doan (~270 tu) bi thay mot so **cau** bang ban AI tuong ung, kem ti le tu that.

> **Pham vi -- doc truoc khi trich dan.** Day **khong** phai phan bac §IV-A. §IV-A tron o muc **doan** (thay tron ven tung doan trong tai lieu k=20), noi pi-hat affine theo dung dinh nghia. Bang duoi tron o muc **cau ben trong mot doan** (k=1), nam **ngoai** pham vi bai tuyen bo. Do tan mac giam theo 1/sqrt(k) khi gop nhieu doan, nhung **thien lech thi khong** -- do la diem dang luu y.

| True ratio bin | n | True mean | pi-hat mean | pi-hat median | Bias | SD |
|---|---|---|---|---|---|---|
| (-0.001, 0.1] | 27408.0 | 0.021 | -0.012 | -0.045 | -0.032 | 0.175 |
| (0.1, 0.2] | 40416.0 | 0.153 | 0.047 | -0.043 | -0.106 | 0.278 |
| (0.2, 0.3] | 30016.0 | 0.248 | 0.124 | -0.038 | -0.123 | 0.364 |
| (0.3, 0.4] | 35664.0 | 0.354 | 0.259 | -0.026 | -0.096 | 0.453 |
| (0.4, 0.5] | 35824.0 | 0.448 | 0.377 | 0.010 | -0.071 | 0.494 |
| (0.5, 0.6] | 32928.0 | 0.553 | 0.512 | 0.551 | -0.041 | 0.510 |
| (0.6, 0.7] | 34336.0 | 0.647 | 0.644 | 1.026 | -0.003 | 0.494 |
| (0.7, 0.8] | 26976.0 | 0.753 | 0.738 | 1.040 | -0.015 | 0.463 |
| (0.8, 0.9] | 36864.0 | 0.848 | 0.864 | 1.044 | 0.016 | 0.384 |
| (0.9, 1.0] | 25296.0 | 0.973 | 0.948 | 1.045 | -0.025 | 0.294 |

pi-hat **doc hut** cac ti le trung gian, manh nhat la **-0.123** tai ti le that 0.248. Trung vi gan nhu **nhi thuc** -- am o ti le thap roi nhay len ~1,04 tu bin 0,6 tro di -- nen mot doan tron nua-nua thuong bi doc thanh "toan AI" hoac "toan nguoi", hiem khi thanh "mot nua". Dat cau AI **lien khoi** bi doc hut it hon (-0,067) so voi **rai rac** (-0,144).

Nguon: `results/s8_submix_pihat.csv`, `results/s8_submix_by_checkpoint.csv`

---

## S8. Ket qua tu khung bai TRUOC (`REPORT_2026-08-15.md`)

Bao cao ngay 15/08 ghi lai mot khung bai **cu**: chan doan lech **don vi** (nguong hieu chuan cho *doan*, quyet dinh dua ra o *bai*) roi chua bang Sidak/Bonferroni. Bai hien tai da chuyen sang khung khac (pi-hat + `Z_k`). So lieu cu **van dung** -- moi con so duoi day da duoc do lai tu CSV -- va van tra loi duoc nhung cau reviewer se hoi, nen giu o day thay vi vut di.

> **CANH BAO -- doc truoc khi trich.** Bang trong muc nay do luat gan co **`any`** (gan co bai neu **co doan nao** vuot nguong). Table III cua bai do luat **`mean`** (so trung binh diem cua bai voi mot nguong). **Hai luat khac nhau, khong dat canh nhau duoc.** Cung `test_fp_full`, cung k=20: luat `any` cho **10,11%** con luat `mean` cho **0,71%** -- chenh 14 lan, va ca hai deu dung, vi chung tra loi hai cau hoi khac nhau.

### S8a. Lech don vi duoi luat `any`

Nguong chot tren `val` cho 1%% **muc doan**, ap nguyen si, gan co bai neu co doan nao vuot:

| Checkpoint | FPR segment % | FPR paper % | If independent % | Formula error % | Ratio |
|---|---|---|---|---|---|
| greater_xlm_roberta_large_seed42 | 0.41 | 6.75 | 8.79 | 30.15 | 16.34 |
| radar_seed42 | 0.48 | 7.58 | 10.10 | 33.22 | 15.82 |
| modernbert_large_seed44 | 0.44 | 7.75 | 9.37 | 20.95 | 17.51 |
| roberta_large_seed43 | 0.53 | 8.92 | 11.11 | 24.60 | 16.79 |
| greater_roberta_large_seed42 | 0.69 | 9.67 | 14.19 | 46.76 | 13.94 |
| xlm_roberta_large_seed44 | 0.62 | 9.83 | 12.88 | 30.93 | 15.78 |
| roberta_large_seed42 | 0.65 | 10.08 | 13.29 | 31.82 | 15.63 |
| xlm_roberta_large_seed43 | 0.65 | 10.25 | 13.43 | 31.03 | 15.70 |
| modernbert_large_seed42 | 0.73 | 11.17 | 14.93 | 33.74 | 15.22 |
| deberta_v3_large_seed42 | 0.83 | 11.92 | 16.74 | 40.45 | 14.30 |
| modernbert_large_seed43 | 0.80 | 12.67 | 16.21 | 27.96 | 15.76 |
| xlm_roberta_large_seed42 | 1.03 | 14.83 | 20.14 | 35.77 | 14.42 |
| roberta_large_seed44 | 1.18 | 15.25 | 22.71 | 48.94 | 12.88 |
| deberta_v3_large_seed44 | 1.17 | 15.25 | 22.53 | 47.76 | 13.01 |
| deberta_v3_large_seed43 | 1.25 | 16.00 | 23.73 | 48.29 | 12.84 |
| radar_xlm_seed42 | 2.66 | 28.33 | 42.84 | 51.18 | 10.66 |

FPR muc doan **0.41--2.66%** giu dung cam ket; cung nguong do o muc bai cho **6.75--28.33%**, tuc **10.7--17.5 lan**, tren **16/16 checkpoint khong ngoai le**. Huan luyen doi khang (GREATER, RADAR) **khong** cuu duoc -- chung nam rai khap dai.

Nguon: `results/s9_unit_mismatch.csv`

### S8b. Duong cong co kiem soat theo `k` (luat `any`)

**Cung 576 bai, cung checkpoint, cung nguong -- chi doi so doan nhin vao.** Nen quan the khong doi theo `k`.

| k | # papers | FPR (any) % | SD % |
|---|---|---|---|
| 1.0 | 576.0 | 0.76 | 0.32 |
| 2.0 | 576.0 | 1.24 | 0.45 |
| 3.0 | 576.0 | 1.92 | 0.53 |
| 5.0 | 576.0 | 3.21 | 0.66 |
| 10.0 | 576.0 | 5.81 | 0.58 |
| 15.0 | 576.0 | 7.87 | 0.52 |
| 20.0 | 576.0 | 10.11 | 0.70 |

Tang **don dieu, khong bac nhay**. O `k=3` (kich thuoc cua cac benchmark ngan) FPR chi **1,92%**; o `k=20` (bai khoa hoc that) la **10,11%** -- benchmark ngan lam van de trong **nho hon ~5 lan** thuc te.

Nguon: `results/s9_k_curve_any.csv`

### S8c. Co tap hieu chuan quyet dinh nguong co dung duoc khong

Lien quan truc tiep **§V-B** cua bai (pool `m=438`). Sidak doi phan vi $1-\alpha/k$; voi `k=19` va muc tieu 1% thi can phan vi **0,999471**, ma 438 doan chi phan giai toi **0,99772**.

| Pool | # chunks | Pipeline / note | FPR median % | min % | max % | Spread |
|---|---|---|---|---|---|---|
| val | 438 | different from deployment | 3.98 | 0.60 | 9.89 | 9.30 |
| cal subsample | 438 | same as deployment | 3.90 | 3.60 | 4.26 | 0.66 |
| cal subsample | 2000 | same as deployment | 1.40 | 1.23 | 1.63 | 0.40 |
| cal full | 13960 | same as deployment | 1.05 | 0.81 | 1.30 | 0.49 |
| theoretical floor (k=19) | 1891 | quantile 0.999471 |  |  |  |  |
| theoretical floor (k=20) | 1991 | quantile 0.999498 |  |  |  |  |
| theoretical floor (k=22) | 2190 | quantile 0.999543 |  |  |  |  |
| theoretical floor (k=23) | 2289 | quantile 0.999563 |  |  |  |  |

**Hai chan doan tach bach:**

1. **La CO MAU, khong phai lech phan phoi.** Cung pipeline ma chi 438 doan van hong y het (3,90% vs 3,98%).

2. **Lech phan phoi gay benh KHAC.** Cung 438 doan: cung pipeline lech **deu** (bien do 0,66 diem), khac pipeline lech **vo huong** (bien do 9,30 diem). Co mau gay sai lech co he thong; lech phan phoi gay sai lech **khong doan duoc huong**.

San ly thuyet la **>=1.891 doan** cho `k=19`; do thuc nghiem thi 2.000 doan cho 1,40% va ~13.960 doan (≈7x san) cho 1,05%. `val` co **438** -- thieu khoang **4,3 lan**.

Nguon: `results/s9_pool_size.csv`

### S8d. Gia phai tra khi doi diem van hanh

Sidak ha `alpha` muc doan tu 1% xuong **0,0529%** (voi `k=19`), nen moi con so TPR do o 1% phai do lai:

| Checkpoint | k | TPR @1%  | TPR @0.1% | TPR @Sidak | Loss (pts) |
|---|---|---|---|---|---|
| radar_xlm_seed42 | 19 | 76.42 | 59.05 | 52.31 | 24.11 |
| xlm_roberta_large_seed42 | 19 | 89.76 | 80.78 | 78.07 | 11.69 |
| radar_seed42 | 19 | 90.36 | 84.68 | 81.11 | 9.25 |
| modernbert_large_seed43 | 19 | 91.35 | 84.94 | 82.96 | 8.39 |
| xlm_roberta_large_seed43 | 19 | 90.03 | 83.55 | 81.97 | 8.06 |
| roberta_large_seed44 | 19 | 90.09 | 84.87 | 82.96 | 7.13 |
| greater_roberta_large_seed42 | 19 | 90.89 | 85.60 | 83.75 | 7.13 |
| greater_xlm_roberta_large_seed42 | 19 | 90.16 | 84.61 | 83.09 | 7.07 |
| modernbert_large_seed44 | 19 | 91.35 | 86.39 | 84.28 | 7.07 |
| modernbert_large_seed42 | 19 | 90.69 | 85.47 | 83.88 | 6.80 |
| deberta_v3_large_seed43 | 19 | 91.15 | 85.20 | 84.48 | 6.67 |
| xlm_roberta_large_seed44 | 19 | 90.16 | 85.01 | 84.15 | 6.01 |
| roberta_large_seed43 | 19 | 92.01 | 86.79 | 86.00 | 6.01 |
| roberta_large_seed42 | 19 | 91.02 | 86.13 | 85.14 | 5.88 |
| deberta_v3_large_seed42 | 19 | 92.87 | 87.71 | 87.05 | 5.81 |
| deberta_v3_large_seed44 | 19 | 91.48 | 87.52 | 86.33 | 5.15 |

Trung vi mat **6,97 diem** (90,79% -> 83,82%), nhung **ca xau nhat mat 24,1 diem** (`radar_xlm` s42: 76,42% -> 52,31%). Phai bao ca dai, khong duoc dung trung vi che ca xau nhat. Day la TPR muc **doan**; muc bai chua do duoc vi `test_fp_full` chi co lop human.

Nguon: `results/s9_tpr_cost.csv`

---

## Cach tai tao

```bash
python supplement_rivf2026/build_supplement.py   # S1, S2, S3, S4, S5
python supplement_rivf2026/generator_matrix.py   # S2d, S2e
python supplement_rivf2026/subfield_fpr.py       # S6
python supplement_rivf2026/submix_pihat.py       # S7
python supplement_rivf2026/report_tables.py      # S8, S2c
python supplement_rivf2026/make_boxplot.py       # figS1
python supplement_rivf2026/render_supplement.py  # tai sinh chinh file nay
```

Khong buoc nao chay lai detector; tat ca deu la hau xu ly tren `artifacts/` va `output/` co san.
