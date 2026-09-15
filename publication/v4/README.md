# DUET manuscript V4: evidence and reproducibility

This is the corrected working manuscript, not a submitted or accepted article.
Author/affiliation, funding, contribution and archive identifier fields remain
explicitly marked for completion. The software license is in the repository root.

## Build documents without fitting models

From the repository root, with `python-docx` and `pandas` installed:

```sh
python scripts/build_v4_documents.py --output output_v4
```

This uses only the files included here. `manuscript.md` is the resolved text;
`tables.json` identifies each CSV and its caption; `figures/` contains the five
composite figures. `additional_file_1_samples.csv` is the sample/library mapping.
The supplementary document splits wide tables into readable column panels.
Archived benchmark columns ending in `_mb` use bytes / 2^20, i.e. **MiB**.

## Recalculate from saved results

The `saved_results_*.zip` archives contain simulation per-gene outputs and truth,
plus the small saved summaries needed for independent verification. They do not
contain human raw reads or expression matrices. Extract all four archives at the
repository root. Their paths populate `results/` and `publication/v4/inputs/`.
Then run:

```sh
python scripts/audit_v4_saved_results.py --inputs publication/v4/inputs --output audit_recomputed
```

This performs no fitting. It reproduces Tables S14–S16 from saved output, using
the original five-method common tested-gene universe. Thresholds are unchanged:
calibration replicates 1–50, evaluation 51–100. Category confidence intervals
resample whole evaluation replicates (2,000 draws, NumPy seed 0).
The original simulation scoring code is `scripts/sim_score.py`; a full rerun of
model fits additionally requires the original count matrices and the R/Python
environment described in `environment.json`. Large inputs are not in this repository.

`figure_code/` includes results-only rendering for the performance summary,
benchmark grid and effect-size figure using `figure_data/`. Figure 1 uses the included IFIT3 expression-only fixture (no donor or cell identifiers); its results-only generator is `figure_code/make_fig1.py`. The accuracy figure's original generator is `scripts/make_fig2_revised.py`.

## Interpretation corrections

- At the separately calibrated thresholds, DESeq2 is more sensitive in every
  alternative simulation category, including db. Nominal-BH category sensitivity
  does not demonstrate a DUET advantage at comparable error rates.
- Historical CDR timings used unequal EB/gating settings. They are retained in
  S5, excluded from the speed-ratio plots and not claimed as matched benchmarks.
  `scripts/bench_duet_run_matched.py` supplies explicit settings for a future run;
  it has **not** generated the reported historical measurements.
- Pancreas data were processed in an earlier SCRAP analysis. The HPAP collection
  was unchanged, but the historical run configuration and joint UMI deduplication
  across repeated library runs remain unverified. Integer count reconstruction
  and summing donor–barcode rows do not prove molecular deduplication. Pancreas
  results characterize a fixed technical input and are not biological validation.
- Donor partitions reuse donors. Under a true global null, FDR is P(any rejection),
  not the mean proportion of genes rejected. Partition frequencies are conditional
  diagnostics. No universal FDR guarantee is claimed for the two-stage rule.
- GLIMES and MAST mixed-model strata have different scored-gene universes. Compare
  methods within each stratum; saved results do not establish general superiority.
- `linear_logfc` is a ratio of back-transformed means of log1p expression, not
  arithmetic mean counts. Rank explicitly by descending `neglog10p`.

## What was not rerun

No simulation generation, full DE model fitting, mixed models, alignment,
integration or multi-day timing benchmark was run for V4. New analyses summarize
existing scores and donor partitions. The historical environment record describes
the original analysis; it is not a claim that every input was produced at the
recorded dirty working-tree commit. SHA256SUMS records the publication files.

## Remaining submission items

1. Final author details, funding and CRediT statement.
2. A persistent archive identifier and externally accessible release/data location.
3. Historical SCRAP provenance and molecular-deduplication evidence if pancreas is
   to support biological inference. Otherwise retain its restricted technical role.
4. A matched CDR rerun only if a quantitative covariate speed claim is desired.

Formatting follows the [BMC Bioinformatics instructions](https://link.springer.com/journal/12859/submission-guidelines):
structured abstract under 350 words, double-spaced main text, line/page numbers,
figures in first-citation order and editable tables. Journal acceptance is not implied.
