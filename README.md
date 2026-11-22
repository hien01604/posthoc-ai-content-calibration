# Post-Hoc Calibration of AI-Content Estimates Across Document Length

Code and data for the RIVF 2026 paper

> Pham Quang Dai Phuc, Pham Thi Minh Hien, Nguyen Dinh Thuan.
> *Post-Hoc Calibration of AI-Content Estimates Across Document Length.*
> RIVF 2026.

The paper trains no detector. It takes the **segment scores** of 16 existing detector
checkpoints and studies how to aggregate them into two document-level outputs whose
calibration does not change with document length *k*:

| Output | Question it answers | How |
|---|---|---|
| **Estimate** `π̂` with an interval | How AI-like is this document? | `π̂ = (s̄ − μ₀)/(μ₁ − μ₀)`; distribution-free interval calibrated near the reported `π̂` |
| **Decision** at a fixed FPR | Should this document be flagged? | each segment score → `z`-score against human text, then `Σ zᵢ > z₁₋α · √(k(1+(k−1)ρ_z))` |

The central finding is that the order matters: correcting for within-document
correlation on the **raw** score scale fails as *k* grows, while the same correction
applied **after** the `z`-transform holds the target FPR at every length tested.

This repository contains everything needed to recompute every number, table and figure
in the paper from those segment scores, plus a script that checks the numbers printed in
`paper/rivf2026.tex` against the recomputed ones.

---

## Quick start

Requires Python ≥ 3.10. CPU only; no GPU or network access is needed.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python reproduce.py --verify         # ~10 s: check the shipped results against the paper
python reproduce.py                  # ~20 min on a laptop CPU: recompute everything, then verify
```

`--verify` recomputes 101 numbers reported in the paper from the shipped CSV files and
segment scores, compares each with the value printed in the paper, and prints one
`[OK]`/`[!!]` line per number; it exits with a non-zero code if any number differs.
Two checks need data that we do not redistribute (raw text and DAGPap24; see
[DATA.md](DATA.md)) and show `[--] SKIP`; the other 99 run on the released files.

Every step is seeded, so `python reproduce.py` rewrites the shipped CSV files with
identical values. Use `--list` to see the steps, `--only NAME` or `--from NAME` to run
part of the pipeline. Logs go to `logs/`.

---

## Where each result comes from

| Paper | Script | Output |
|---|---|---|
| Table I–II (schema, splits) | `supplement_rivf2026/build_supplement.py` | `supplement_rivf2026/results/s2a_schema.csv`, `s2b_splits.csv` |
| §III-B checkpoint AUROC/TPR | — (from `metrics.json`) | `artifacts/kaggle/artifacts_*/metrics.json` |
| §III-D, §IV-A `π̂` anchors and centring | `paper_level_pihat.py` | `output/pihat_sanity_16models.csv` |
| §IV-A mixtures at known `π` | `extra_experiments_rivf2026/exp01_calibration.py` | `extra_experiments_rivf2026/results/exp01_calibration_table.csv` |
| Table III (interval on human documents) | `final_experiments_rivf2026/f1_pihat_ci.py` | `final_experiments_rivf2026/results/f1_coverage_null.csv` |
| Table IV (interval at `π > 0`, plug-in) | `final_experiments_rivf2026/f6_mixed_pi_empirical_ci.py` → `make_table_mixed.py` | `f6_coverage_mixed_emp.csv` |
| Table V (threshold pinned at the wrong *k*) | `paper_level_mean_ktransfer.py` → `make_table_ktransfer.py` | `output/mean_ktransfer_16models.csv` |
| Fig. 1, §V-A (FPR vs *k*, bootstrap, *p* = 0.044) | `final_experiments_rivf2026/f5_deff_necessity.py`, `f7_cluster_bootstrap.py` → `make_fig1.py` | `f5_deff_necessity.csv`, `f7_cluster_*.csv` |
| §V-B `ρ_z` (Hartung), raw vs `z` ICC | `paper_level_tau_k.py`, `paper_level_rho_compare.py` | `output/tau_k_mean_16models.csv`, `rho_compare_16models.csv` |
| §V-B `T_k` shape, ±20 % sensitivity, budget, oracle | `f4_null_shape.py`, `exp04_rho_sensitivity.py`, `f8_rho_sample_size.py`, `exp06_oracle.py` | corresponding CSV in `results/` |
| Table VI (transfer of `ρ_z`, power at matched FPR) | `paper_level_hartung_ood.py`, `revision_experiments_rivf2026/rev3_*.py`, `rev4_*.py` → `make_table_ood.py` | `output/hartung_ood_16models.csv`, `rev3_paired_miss.csv`, `rev4_fpr_matched.csv` |
| §V-D per-bin conformal baseline | `final_experiments_rivf2026/f3_conformal_bins.py` | `f3_conformal_overall.csv` |
| §VI subfield FPR | `supplement_rivf2026/subfield_fpr.py` | `s7_subfield_summary.csv` |
| Fig. 2 (pipeline diagram) | `make_fig2.py` | `paper/figures/fig2_pipeline.pdf` |

Every table and number in `paper/rivf2026.tex` carries a `% SOURCE:` comment naming the
script and file it comes from. Tables IV–VI are written into the `.tex` by the
`make_table_*.py` scripts between `% >>> ... BEGIN` / `% <<< ... END` markers.

---

## Applying the method to another detector

All you need is per-segment scores from your detector and some unlabeled human
documents of the kind you will screen. Calibration is done once per detector:

1. **Anchors** `μ₀, μ₁`: mean segment score of labeled human and AI validation
   segments (`paper_level_pihat.py`). They define `π̂`.
2. **Human pool and `ρ_z`**: from about **200 human documents** (no AI labels needed),
   keep their segment scores as the null pool and estimate the within-document
   correlation of the transformed scores with Hartung's estimator
   (`rho_hartung()` in `paper_level_tau_k.py`). Fewer documents leave `ρ_z` too
   uncertain (`f8_rho_sample_size.py`).
3. **Interval for `π̂`**: on mixtures of those human documents with AI segments at a
   few known `π`, record the 2.5/97.5 % percentiles of `π̂ − π`
   (`f6_mixed_pi_empirical_ci.py`).

For a new document with *k* segments:

* **Report** `π̂` with the interval calibrated at the grid value of `π` nearest to it.
* **Decide**: `pᵢ = (#{pool ≥ sᵢ} + ½)/(m + 1)`, `zᵢ = Φ⁻¹(1 − pᵢ)`; flag the document if
  `Σ zᵢ > z₁₋α · √(k(1 + (k − 1)ρ_z))` (`z₁₋α = 2.33` for 1 % FPR).

The decision uses human text only, so its FPR does not depend on how well the detector
separates the classes; its power does. Re-estimate `μ₀`, the pool and `ρ_z` for a new
domain, a new segmentation (we use 275-word chunks at sentence boundaries) or a
different period of human writing.

---

## Repository layout

```
reproduce.py                    run everything in order, then verify
verify_paper_numbers.py         recompute + compare the numbers of the paper (core)
verify_paper_numbers_ext.py     ... and the remaining ones (dataset, Sec. IV-A, VI ...)
icc.py                          one-way ICC used for rho_raw
paper_level_*.py                core pipeline (pi-hat, rho_z, thresholds, transfer)
make_table_*.py, make_fig*.py   write Tables IV-VI into the .tex, draw Fig. 1-2
final_experiments_rivf2026/     f1-f8: intervals, bootstrap, conformal, null shape, budget
extra_experiments_rivf2026/     mixtures, rho sensitivity, oracle, per-checkpoint FPR
revision_experiments_rivf2026/  paired and FPR-matched power comparisons
supplement_rivf2026/            supplementary tables (SUPPLEMENT.md) and subfield FPR
artifacts/                      INPUT: segment scores of the 16 checkpoints (no text)
output/                         intermediate results and corpus metadata (no text)
paper/                          LaTeX source, figures, generated tables
dataset_construction/           how the corpus was built (not needed to reproduce)
detectors/                      how the 16 checkpoints were trained; zero-shot baselines
```

The scripts locate their inputs relative to their own folder, so keep this layout.
Code comments and console output are in Vietnamese without diacritics; variable names,
file names and this documentation are in English.

`dataset_construction/` and `detectors/` document how the corpus and the checkpoints
were produced (Kaggle notebooks, GPU training, generation through the OpenRouter API).
They are provided for transparency and are **not** needed to reproduce the paper; they
need the raw corpus, API keys and GPUs, and their outputs are already in `artifacts/`.

---

## Data

See [DATA.md](DATA.md) for the files, their columns, what is withheld and why. In
short: we release the segment scores, labels and arXiv identifiers that the analysis
uses, but not the text of the arXiv papers or of the AI generations, and not the model
weights.

## License

Code: MIT ([LICENSE](LICENSE)). Released scores and result tables: CC BY 4.0.

## Citation

See [CITATION.cff](CITATION.cff).
