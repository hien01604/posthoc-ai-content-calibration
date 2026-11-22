# Data

The analysis in the paper reads only **segment scores**: for every 275-word segment,
the probability that it is AI-generated according to each of 16 detector checkpoints.
Those scores, the labels and the arXiv identifiers are released here. The text itself
is not.

## Released

### `artifacts/kaggle/artifacts_<checkpoint>/` — 16 checkpoints

| File | Rows | Content |
|---|---|---|
| `metrics.json` | — | threshold `tau` fixed at 1 % segment FPR on `val`; AUROC, TPR, FPR per split |
| `pred_val.parquet` | 876 | validation split (anchors `μ₀, μ₁`) |
| `pred_test_ood_model.parquet` | 3,028 | held-out-generator split: 1,514 human/AI segment pairs, 637 papers |
| `pred_test_fp.parquet` | 4,290 | false-positive split: 3,335 human and 955 lightly AI-edited segments |
| `pred_matched_*.parquet` | 296 / 1,206 | length/surface-matched subsets (§III-A) |
| `pred_test_fp_full.parquet` | 27,119 | full-length corpus (same as below) |

Columns: `paper_id` (arXiv identifier), `label` (0 human, 1 AI), `ai_degree`
(0 human, 1 polished, 3 rewrite, 4 written from scratch), `generator_family`, `score`
(detector probability that the segment is AI-written; higher = more AI-like).

Checkpoints: RoBERTa-large, DeBERTa-v3-large, XLM-R-large, ModernBERT-large (seeds 42,
43, 44 each) and GREATER-RoBERTa, GREATER-XLM-R, RADAR, RADAR-XLM-R (seed 42).

### `artifacts/kaggle_testfp_full/pred_testfp_full/` — full-length corpus

One file per checkpoint, 27,119 segments of 1,200 human-written arXiv Computer Science
papers from 2010–2015 (median 19 segments per paper, maximum 262). Columns:
`sample_id`, `paper_id`, `source` (segment position), `label` (always 0), `artifact`,
`branch`, `model`, `seed`, `score`. Every document-level FPR in the paper comes from
these files.

### `artifacts/pred_mix/scores_all.parquet` — mixed documents (Supplement S8)

Scores of all 16 checkpoints on documents that splice human and AI segments at known
ratios (`mix_ratio_actual`, `mix_structure`, `n_seams`, ...), used only by
`supplement_rivf2026/submix_pihat.py`. No text.

### `output/`

* Result CSV files of the pipeline (the `% SOURCE:` comments in the paper name them).
* `dataset_test_fp_full_deconf2.parquet` — metadata of the full-length corpus with the
  text columns removed: `paper_id`, `categories` (arXiv subfield), `published_year`,
  segment position, and `n_words` (words per segment).
* `ai_clean.parquet` — metadata of every AI generation with the text removed:
  generating model, family, generation mode, prompt template, temperature, split pool.
* `testfp_full_manifest.json`, `deconf2_manifest.json` — how the full-length corpus and
  the de-confounded corpus were built (chunking parameters, filters, surface audits).
* `shortcut_audit/`, `shortcut_audit_dagpap24/` — surface-feature audits (§I, §III-A).

## Withheld

| What | Why | How to obtain or rebuild |
|---|---|---|
| Text of the arXiv papers | arXiv papers carry their authors' licenses, which do not all permit redistribution | download by `paper_id` from arXiv; segment with `build_testfp_full.py` (275-word chunks at sentence boundaries, first chunk dropped) |
| Text of the AI generations | each generation rewrites or continues a specific human passage, so it would partly redistribute that passage | regenerate with `dataset_construction/genai.ipynb` (needs an OpenRouter key) |
| Detector weights | 16 large transformer checkpoints, too large to host here | retrain with `detectors/supervised/`, `detectors/adversarial/` |
| DAGPap24 | third-party dataset | obtain from the DAGPap24 shared task; audit with `dataset_construction/audit_dagpap24.py` |

The shipped `supplement_rivf2026/results/s2a_schema.csv` describes the full corpus,
including its text columns. Re-running `build_supplement.py` here regenerates it from the
text-free metadata, so its `human_text`/`ai_text` rows are replaced by `n_words`; every
other supplementary table is reproduced exactly.

Without the text, two numbers of the paper cannot be recomputed here and are reported as
`SKIP` by the verifier: the share of AI samples containing typographic punctuation
(37 %, §III-A) and the median DAGPap24 length (5,400 words, §I). Everything else,
including all tables and figures, is recomputed from the released files.

## Splits

Papers are assigned to splits by a hash of their arXiv identifier, with weights
70 % (training + validation), 15 % (held-out generator) and 15 % (false positive).
Google and DeepSeek models generate only the held-out-generator split. The full-length
corpus is a separate set of 1,200 papers, disjoint from all other splits.
