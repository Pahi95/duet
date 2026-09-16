# DUET

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22801244.svg)](https://doi.org/10.5281/zenodo.22801244)
[![PyPI](https://img.shields.io/pypi/v/duet-de.svg)](https://pypi.org/project/duet-de/)
[![tests](https://github.com/Pahi95/duet/actions/workflows/tests.yml/badge.svg)](https://github.com/Pahi95/duet/actions/workflows/tests.yml)

**DUET (Detection–Expression Unified Test)** is a Python implementation of two-part hurdle testing for
single-cell differential expression, with a donor-level pseudobulk arm. It works directly on sparse
AnnData objects and needs no R installation.

- **Hurdle test.** A logistic model for detection and a Gaussian model for expression among detected
  cells, as in MAST. With empirical Bayes enabled, the moderated-t tail of the continuous component is
  converted to a χ² statistic and added to the detection likelihood-ratio statistic.
- **Log-scale significance.** `neglog10p` stays finite and keeps genes in order when the p-value
  underflows to zero, which happens routinely at tens of thousands of cells.
- **Sample-level calls.** `run_duet_pseudobulk` tests summed counts per donor with PyDESeq2 and reports
  its call next to the hurdle statistic.

## Main results

From the accompanying manuscript (DUET 1.0.0):

- **Agreement with R MAST** on identical cells, genes and design: detection statistics agree, and
  coefficients agree with MAST's unregularized glm fit. The moderated continuous statistic differs,
  because the variance moderation differs.
- **Speed**, CPU time on the same input:
  - two-group design without covariates: 39–257-fold less end to end, about 800-fold less for the model
    fit and test;
  - with the cellular detection rate as a covariate: 13–14-fold less.
- **Calibration**: cell-level significance across donors was poorly calibrated in simulations and donor
  partitions. Donor-level questions should be tested at the sample level. The hurdle statistic then
  serves as a complementary ranking and as a description of detection and expression changes.

## Install

```sh
pip install duet-de                 # core: NumPy, SciPy, pandas, AnnData
pip install "duet-de[pseudobulk]"   # adds PyDESeq2 for the sample-level arm
pip install "duet-de[extra]"        # optional: statsmodels, tqdm, psutil
```

Python 3.11 or later. From a clone of this repository, use `pip install -e ".[pseudobulk]"`.

## Quick start

### Cell-level hurdle test

```python
import pandas as pd
from duet import run_duet

# adata.X: log1p-normalized expression. sample_col is the obs column holding the CONDITION.
path = run_duet(adata, output_dir="results", celltype_col="celltype",
                sample_col="condition", ref_label="ctrl", test_label="stim",
                mast_compat=True)
scores = pd.read_csv(path)
ranked = scores.sort_values("neglog10p", ascending=False)
```

Output rows keep the input gene order, so sort by `neglog10p` explicitly; `pvalue` may be exactly zero.
Across donors, use these scores for ranking and description, not as donor-level significance.

### Donor-level calls

```python
from duet import run_duet_pseudobulk

# adata.layers["counts"]: raw integer counts
path = run_duet_pseudobulk(adata, output_dir="results", celltype_col="celltype",
                           condition_col="condition", donor_col="donor",
                           ref_label="ctrl", test_label="stim", counts_layer="counts",
                           paired=False)
```

Set `paired=True` only when every donor contributes both conditions; pairing is not detected
automatically. The `call` column is:

| call | meaning |
|---|---|
| `significant` | pseudobulk BH-adjusted p < 0.05 and the same sign as the DUET coefficient |
| `direction_conflict` | pseudobulk significant, opposite sign |
| `ranked_only` | not significant at the sample level, cell-level FDR < 0.05; exploratory ranking only |
| `ns`, `not_tested` | not significant, or not tested |

The direction filter does not add a formal FDR guarantee beyond that of the pseudobulk test.

### Command line

```sh
duet data.h5ad --celltype-col celltype --sample-col condition --ref-label ctrl --test-label stim \
     --mast-compat --covariates "" --output-dir results
```

`duet --help` lists every option.

## Output

The main columns of `run_duet` are listed below. All columns are described in Additional file 3 of the
manuscript.

| column | meaning |
|---|---|
| `coef` | difference of group means of log1p expression over all cells (MAST-style logFC, natural log) |
| `pvalue`, `fdr` | hurdle p-value (may underflow to 0) and BH-adjusted p-value within the cell type |
| `neglog10p` | underflow-safe −log10 p; use for ranking |
| `p_detect`, `p_continuous`, `stat_*`, `df_*` | component p-values, statistics and degrees of freedom |
| `detect_rate_ref`, `detect_rate_test`, `mean_expr_ref`, `mean_expr_test` | per-group summaries |
| `detection_separated` | detection rate exactly 0 or 1 in a group: the test is valid, the coefficient is not finite |
| `tested`, `skip_reason` | whether a gene was tested, and why not |

`linear_logfc=True` reports a log2 ratio of back-transformed group log1p means instead. This is not the
ratio of arithmetic mean counts, and effect sizes differ in scale between methods.

## Reproducing the manuscript analyses

`scripts/` contains the analyses behind the manuscript. `python scripts/make_all_results.py --dry-run`
lists every step and its command.

**Input data.** Put these in the input directory (`DUET_INPUTS`, default `../inputs`):
- **Kang et al. 2018** (PBMCs, IFN-β): built by `build_kang_h5ad.py`.
- **Crowell et al. 2020** (mouse cortex, LPS): built by `build_crowell_h5ad.py` (+ `get_crowell.R`,
  muscData).
- **Pancreas analysis input** (17,093 ductal, endothelial and stellate cells from PDAC and HPAP islet
  preparations): https://doi.org/10.5281/zenodo.22801671, saved as `PancreasCorrected.h5ad`.

The comparisons also need R with MAST, DESeq2, muscat, edgeR, limma, lme4 and GLIMES. Results are
written to `results/`, which is not versioned.

| Analysis | Scripts |
|---|---|
| Agreement with R MAST | `run_de_comparison.py`, `run_mast_dataset.py`, `mast_settings_check.py` (+ `mast_components.R`), `validate_covariate.py` (+ `.R`), `validate_engine.py` |
| Benchmark | `bench_prepare.py`, `bench_harness.py`, `bench_duet_run.py`, `bench_duet_run_matched.py`, `bench_mast_steps.R`, `collect_evidence.py` |
| p-value underflow | `underflow_check.py`, `eb_logspace_report.py` |
| muscat simulations (100 replicates) | `simulate_replicates.R`, `sim_calls.py` (+ `bench_mast_run.R`), `sim_score.py`, `sim_pb_edger_voom.R`, `sim_score_pb_extra.py` |
| Calibration diagnostics | `fdr_scale_diag.py`, `arm_offset.py`, `null_calibration.py`, `null_permutations.py`, `simulate_batch_nb.py` |
| Donor-swap null | `donor_swap_partitions.py`, `donor_swap_sexgenes.py` |
| Donor-aware models | `donor_aware_methods.py`, `donor_aware_run.R`, `install_glimes.R`, `evaluate_sim_replicates.py`, `evaluate_sim.py` |
| Pseudobulk implementation | `validate_pseudobulk_engines.py` (+ `pseudobulk_deseq2.R`) |

## Citation

If you use DUET, please cite the software. The manuscript citation will be added on publication.

- Software, all versions: https://doi.org/10.5281/zenodo.22801244
- DUET 1.0.0, the version used in the manuscript: https://doi.org/10.5281/zenodo.22801245
- Pancreas analysis input: https://doi.org/10.5281/zenodo.22801671

Citation metadata: [CITATION.cff](CITATION.cff). The name is unrelated to the DUET protein-stability
server (Pires et al., Nucleic Acids Res 2014).

## Tests and license

```sh
pip install -e ".[test]"
pytest
```

MIT; see [LICENSE](LICENSE).
